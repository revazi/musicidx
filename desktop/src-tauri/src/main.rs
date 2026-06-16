#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, State, Theme};

#[derive(Serialize)]
struct DesktopState {
    current_dir: String,
    cli_path: String,
    prefix_args: String,
}

#[derive(Serialize)]
struct MusicidxOutput {
    status: i32,
    success: bool,
    stdout: String,
    stderr: String,
}

type RunningChildren = Arc<Mutex<HashMap<String, Child>>>;

#[derive(Clone, Default)]
struct ProcessState {
    children: RunningChildren,
}

#[derive(Clone, Serialize)]
struct MusicidxStreamEvent {
    request_id: String,
    stream: String,
    line: String,
    status: Option<i32>,
    success: Option<bool>,
    done: bool,
}

#[tauri::command]
fn desktop_state() -> DesktopState {
    DesktopState {
        current_dir: default_working_dir(),
        cli_path: musicidx_cli_path(),
        prefix_args: musicidx_prefix_args().join(" "),
    }
}

#[tauri::command]
fn set_window_theme(app: AppHandle, theme: String) -> Result<(), String> {
    let Some(window) = app.get_webview_window("main") else {
        return Ok(());
    };
    let requested_theme = match theme.trim().to_lowercase().as_str() {
        "dark" => Some(Theme::Dark),
        "light" => Some(Theme::Light),
        "system" | "" => None,
        other => return Err(format!("unsupported theme: {other}")),
    };
    window
        .set_theme(requested_theme)
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn run_musicidx(
    args: Vec<String>,
    cwd: Option<String>,
    cli_path: Option<String>,
    prefix_args: Option<String>,
    env_overrides: Option<HashMap<String, String>>,
) -> Result<MusicidxOutput, String> {
    let (resolved_cli_path, mut command) = build_musicidx_command(
        args,
        cwd,
        cli_path,
        prefix_args,
        env_overrides.unwrap_or_default(),
    )?;

    let output = command
        .output()
        .map_err(|error| format!("failed to run `{resolved_cli_path}`: {error}"))?;

    Ok(MusicidxOutput {
        status: output.status.code().unwrap_or(-1),
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

#[tauri::command]
fn run_musicidx_stream(
    app: AppHandle,
    processes: State<'_, ProcessState>,
    request_id: String,
    args: Vec<String>,
    cwd: Option<String>,
    cli_path: Option<String>,
    prefix_args: Option<String>,
    env_overrides: Option<HashMap<String, String>>,
) -> Result<(), String> {
    let request_id_for_error = request_id.clone();
    let (resolved_cli_path, mut command) = build_musicidx_command(
        args,
        cwd,
        cli_path,
        prefix_args,
        env_overrides.unwrap_or_default(),
    )?;

    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    let children = processes.children.clone();

    thread::spawn(move || {
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                emit_stream_event(
                    &app,
                    MusicidxStreamEvent {
                        request_id: request_id_for_error,
                        stream: String::from("stderr"),
                        line: format!("failed to run `{resolved_cli_path}`: {error}"),
                        status: Some(-1),
                        success: Some(false),
                        done: true,
                    },
                );
                return;
            }
        };

        let stdout_thread = child.stdout.take().map(|stdout| {
            spawn_reader_thread(
                app.clone(),
                request_id.clone(),
                String::from("stdout"),
                stdout,
            )
        });
        let stderr_thread = child.stderr.take().map(|stderr| {
            spawn_reader_thread(
                app.clone(),
                request_id.clone(),
                String::from("stderr"),
                stderr,
            )
        });

        if let Ok(mut guard) = children.lock() {
            guard.insert(request_id.clone(), child);
        } else {
            emit_stream_event(
                &app,
                MusicidxStreamEvent {
                    request_id,
                    stream: String::from("stderr"),
                    line: String::from("failed to track musicidx child process"),
                    status: Some(-1),
                    success: Some(false),
                    done: true,
                },
            );
            return;
        }

        let wait_result = wait_for_tracked_child(&children, &request_id);

        if let Some(handle) = stdout_thread {
            let _ = handle.join();
        }
        if let Some(handle) = stderr_thread {
            let _ = handle.join();
        }

        let (status, success, line) = match wait_result {
            Ok(status) => (
                status.code().unwrap_or(-1),
                status.success(),
                format!("musicidx exited with {}", status.code().unwrap_or(-1)),
            ),
            Err(error) => (-1, false, format!("failed waiting for musicidx: {error}")),
        };

        emit_stream_event(
            &app,
            MusicidxStreamEvent {
                request_id,
                stream: String::from("status"),
                line,
                status: Some(status),
                success: Some(success),
                done: true,
            },
        );
    });

    Ok(())
}

#[tauri::command]
fn cancel_musicidx(processes: State<'_, ProcessState>, request_id: String) -> Result<bool, String> {
    let mut guard = processes
        .children
        .lock()
        .map_err(|_| String::from("failed to lock process registry"))?;
    let Some(child) = guard.get_mut(&request_id) else {
        return Ok(false);
    };
    kill_process_tree(child).map_err(|error| error.to_string())?;
    Ok(true)
}

#[cfg(windows)]
fn kill_process_tree(child: &mut Child) -> std::io::Result<()> {
    let pid = child.id().to_string();
    let status = Command::new("taskkill")
        .args(["/PID", &pid, "/T", "/F"])
        .status();
    if status.is_ok_and(|status| status.success()) {
        return Ok(());
    }
    child.kill()
}

#[cfg(unix)]
fn kill_process_tree(child: &mut Child) -> std::io::Result<()> {
    kill_unix_descendants(child.id());
    child.kill()
}

#[cfg(unix)]
fn kill_unix_descendants(pid: u32) {
    let Ok(output) = Command::new("pgrep").args(["-P", &pid.to_string()]).output() else {
        return;
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let Ok(child_pid) = line.trim().parse::<u32>() else {
            continue;
        };
        kill_unix_descendants(child_pid);
        let _ = Command::new("kill")
            .args(["-KILL", &child_pid.to_string()])
            .status();
    }
}

#[cfg(not(any(unix, windows)))]
fn kill_process_tree(child: &mut Child) -> std::io::Result<()> {
    child.kill()
}

fn wait_for_tracked_child(
    children: &RunningChildren,
    request_id: &str,
) -> std::io::Result<std::process::ExitStatus> {
    loop {
        {
            let mut guard = children
                .lock()
                .map_err(|_| std::io::Error::other("failed to lock process registry"))?;
            let Some(child) = guard.get_mut(request_id) else {
                return Err(std::io::Error::other("tracked process disappeared"));
            };
            if let Some(status) = child.try_wait()? {
                guard.remove(request_id);
                return Ok(status);
            }
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn build_musicidx_command(
    args: Vec<String>,
    cwd: Option<String>,
    cli_path: Option<String>,
    prefix_args: Option<String>,
    env_overrides: HashMap<String, String>,
) -> Result<(String, Command), String> {
    if args.is_empty() {
        return Err(String::from("no musicidx arguments provided"));
    }

    let resolved_cli_path = cli_path
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(musicidx_cli_path);
    let mut command = Command::new(&resolved_cli_path);

    let resolved_prefix_args = prefix_args
        .map(|value| split_prefix_args(&value))
        .filter(|values| !values.is_empty())
        .unwrap_or_else(musicidx_prefix_args);
    command.args(resolved_prefix_args);
    command.args(args);

    if let Some(cwd) = cwd
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        apply_dotenv(&mut command, &cwd);
        command.current_dir(cwd);
    }
    apply_env_overrides(&mut command, env_overrides);

    Ok((resolved_cli_path, command))
}

fn spawn_reader_thread<R: Read + Send + 'static>(
    app: AppHandle,
    request_id: String,
    stream: String,
    reader: R,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let reader = BufReader::new(reader);
        for line in reader.lines().map_while(Result::ok) {
            emit_stream_event(
                &app,
                MusicidxStreamEvent {
                    request_id: request_id.clone(),
                    stream: stream.clone(),
                    line,
                    status: None,
                    success: None,
                    done: false,
                },
            );
        }
    })
}

fn emit_stream_event(app: &AppHandle, event: MusicidxStreamEvent) {
    let _ = app.emit("musicidx-output", event);
}

fn default_working_dir() -> String {
    if let Ok(configured) = env::var("MUSICIDX_WORKING_DIR") {
        if !configured.trim().is_empty() {
            return configured;
        }
    }

    let Ok(cwd) = env::current_dir() else {
        return String::from(".");
    };

    if cwd.file_name().is_some_and(|name| name == "src-tauri") {
        if let Some(repo_root) = cwd.parent().and_then(Path::parent) {
            return repo_root.display().to_string();
        }
    }
    if cwd.file_name().is_some_and(|name| name == "desktop") {
        if let Some(repo_root) = cwd.parent() {
            return repo_root.display().to_string();
        }
    }
    cwd.display().to_string()
}

fn apply_dotenv(command: &mut Command, cwd: &str) {
    let dotenv_path = Path::new(cwd).join(".env");
    let Ok(contents) = fs::read_to_string(dotenv_path) else {
        return;
    };

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line).trim();
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() || env::var_os(key).is_some() {
            continue;
        }
        command.env(key, unquote_env_value(value.trim()));
    }
}

fn apply_env_overrides(command: &mut Command, env_overrides: HashMap<String, String>) {
    for (key, value) in env_overrides {
        let key = key.trim();
        let value = value.trim();
        if !key.is_empty() && !value.is_empty() {
            command.env(key, value);
        }
    }
}

fn unquote_env_value(value: &str) -> String {
    let quoted = (value.starts_with('"') && value.ends_with('"'))
        || (value.starts_with('\'') && value.ends_with('\''));
    if quoted && value.len() >= 2 {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

fn musicidx_cli_path() -> String {
    env::var("MUSICIDX_CLI_PATH").unwrap_or_else(|_| String::from("musicidx"))
}

fn musicidx_prefix_args() -> Vec<String> {
    env::var("MUSICIDX_CLI_PREFIX_ARGS")
        .map(|value| split_prefix_args(&value))
        .unwrap_or_default()
}

fn split_prefix_args(value: &str) -> Vec<String> {
    value.split_whitespace().map(str::to_string).collect()
}

fn main() {
    tauri::Builder::default()
        .manage(ProcessState::default())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            desktop_state,
            set_window_theme,
            run_musicidx,
            run_musicidx_stream,
            cancel_musicidx
        ])
        .run(tauri::generate_context!())
        .expect("error while running MusicIdx desktop app");
}
