#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager, State, Theme};

#[derive(Serialize)]
struct DesktopState {
    current_dir: String,
    cli_path: String,
    prefix_args: String,
    models_path: String,
    semantic_model: String,
}

#[derive(Serialize)]
struct MusicidxOutput {
    status: i32,
    success: bool,
    stdout: String,
    stderr: String,
}

#[derive(Serialize)]
struct PlaybackSource {
    path: String,
    transcoded: bool,
    detail: String,
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
fn desktop_state(app: AppHandle) -> DesktopState {
    let packaged_cli = packaged_cli_path(&app);
    let env_cli = env::var("MUSICIDX_CLI_PATH")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let use_packaged_cli = env_cli.is_none() && packaged_cli.is_some();

    let packaged_models = packaged_models_path(&app);
    let packaged_semantic_model = packaged_models
        .as_ref()
        .map(|path| path.join("all-MiniLM-L6-v2"))
        .filter(|path| path.exists())
        .map(|path| path.display().to_string())
        .unwrap_or_default();

    DesktopState {
        current_dir: default_working_dir(&app),
        cli_path: env_cli
            .or(packaged_cli)
            .unwrap_or_else(|| String::from("musicidx")),
        prefix_args: if use_packaged_cli {
            String::new()
        } else {
            musicidx_prefix_args().join(" ")
        },
        models_path: packaged_models
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
        semantic_model: packaged_semantic_model,
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
    app: AppHandle,
    args: Vec<String>,
    cwd: Option<String>,
    cli_path: Option<String>,
    prefix_args: Option<String>,
    env_overrides: Option<HashMap<String, String>>,
) -> Result<MusicidxOutput, String> {
    let (resolved_cli_path, mut command) = build_musicidx_command(
        &app,
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
        &app,
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

#[tauri::command]
fn open_track(path: String) -> Result<(), String> {
    let path = existing_path(path)?;
    open_path_with_system(&path)
}

#[tauri::command]
fn reveal_track(path: String) -> Result<(), String> {
    let path = existing_path(path)?;
    reveal_path_with_system(&path)
}

#[tauri::command]
fn prepare_playback_fallback(app: AppHandle, path: String) -> Result<PlaybackSource, String> {
    let input = existing_path(path)?;
    let ffmpeg = packaged_bin_path(&app, "ffmpeg")
        .map(|path| path.display().to_string())
        .or_else(|| env::var("MUSICIDX_FFMPEG_PATH").ok())
        .unwrap_or_else(|| String::from("ffmpeg"));
    let cache_dir = app
        .path()
        .app_cache_dir()
        .map_err(|error| format!("could not resolve app cache dir: {error}"))?
        .join("playback-cache");
    fs::create_dir_all(&cache_dir)
        .map_err(|error| format!("could not create playback cache: {error}"))?;
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let output = cache_dir.join(format!("musicidx-playback-{timestamp_ms}.m4a"));

    let transcode = Command::new(&ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&input)
        .args(["-vn", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"])
        .arg(&output)
        .output()
        .map_err(|error| format!("failed to run ffmpeg playback fallback `{ffmpeg}`: {error}"))?;
    if !transcode.status.success() {
        let stderr = String::from_utf8_lossy(&transcode.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&transcode.stdout).trim().to_string();
        let detail = if stderr.is_empty() { stdout } else { stderr };
        return Err(format!("ffmpeg playback fallback failed: {detail}"));
    }

    Ok(PlaybackSource {
        path: output.display().to_string(),
        transcoded: true,
        detail: format!("transcoded with {ffmpeg}"),
    })
}

fn existing_path(path: String) -> Result<PathBuf, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(String::from("track path is empty"));
    }
    let path = PathBuf::from(trimmed);
    if !path.exists() {
        return Err(format!("track path does not exist: {}", path.display()));
    }
    Ok(path)
}

#[cfg(target_os = "macos")]
fn open_path_with_system(path: &Path) -> Result<(), String> {
    let mut command = Command::new("open");
    command.arg(path);
    run_external_command(&mut command, "open track")
}

#[cfg(target_os = "macos")]
fn reveal_path_with_system(path: &Path) -> Result<(), String> {
    let mut command = Command::new("open");
    command.args(["-R"]).arg(path);
    run_external_command(&mut command, "reveal track")
}

#[cfg(target_os = "windows")]
fn open_path_with_system(path: &Path) -> Result<(), String> {
    let mut command = Command::new("cmd");
    command.args(["/C", "start", ""]).arg(path);
    run_external_command(&mut command, "open track")
}

#[cfg(target_os = "windows")]
fn reveal_path_with_system(path: &Path) -> Result<(), String> {
    let mut command = Command::new("explorer");
    command.arg(format!("/select,{}", path.display()));
    run_external_command(&mut command, "reveal track")
}

#[cfg(all(unix, not(target_os = "macos")))]
fn open_path_with_system(path: &Path) -> Result<(), String> {
    let mut command = Command::new("xdg-open");
    command.arg(path);
    run_external_command(&mut command, "open track")
}

#[cfg(all(unix, not(target_os = "macos")))]
fn reveal_path_with_system(path: &Path) -> Result<(), String> {
    let target = path.parent().unwrap_or(path);
    let mut command = Command::new("xdg-open");
    command.arg(target);
    run_external_command(&mut command, "reveal track")
}

fn run_external_command(command: &mut Command, action: &str) -> Result<(), String> {
    let status = command
        .status()
        .map_err(|error| format!("failed to {action}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("failed to {action}: exited with {status}"))
    }
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
    app: &AppHandle,
    args: Vec<String>,
    cwd: Option<String>,
    cli_path: Option<String>,
    prefix_args: Option<String>,
    env_overrides: HashMap<String, String>,
) -> Result<(String, Command), String> {
    if args.is_empty() {
        return Err(String::from("no musicidx arguments provided"));
    }

    let explicit_cli_path = cli_path
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let env_cli_path = env::var("MUSICIDX_CLI_PATH")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let packaged_cli = packaged_cli_path(app);
    let using_packaged_cli = explicit_cli_path.is_none() && env_cli_path.is_none() && packaged_cli.is_some();
    let resolved_cli_path = explicit_cli_path
        .or(env_cli_path)
        .or(packaged_cli)
        .unwrap_or_else(|| String::from("musicidx"));
    let mut command = Command::new(&resolved_cli_path);

    let resolved_prefix_args = prefix_args
        .map(|value| split_prefix_args(&value))
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| {
            if using_packaged_cli {
                Vec::new()
            } else {
                musicidx_prefix_args()
            }
        });
    command.args(resolved_prefix_args);
    command.args(args);

    if let Some(cwd) = cwd
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        apply_dotenv(&mut command, &cwd);
        command.current_dir(cwd);
    }
    apply_packaged_env_defaults(app, &mut command, &env_overrides);
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

fn default_working_dir(app: &AppHandle) -> String {
    if let Ok(configured) = env::var("MUSICIDX_WORKING_DIR") {
        if !configured.trim().is_empty() {
            return configured;
        }
    }

    if packaged_cli_path(app).is_some() {
        if let Some(app_data_dir) = app_data_dir(app) {
            return app_data_dir.display().to_string();
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

fn app_data_dir(app: &AppHandle) -> Option<PathBuf> {
    let Ok(path) = app.path().app_data_dir() else {
        return None;
    };
    let _ = fs::create_dir_all(&path);
    Some(path)
}

fn packaged_cli_path(app: &AppHandle) -> Option<String> {
    packaged_resource_path(
        app,
        &[
            &[
                "resources",
                "musicidx-bin",
                "musicidx",
                executable_name("musicidx"),
            ],
            &["resources", "musicidx-bin", executable_name("musicidx")],
            &["musicidx-bin", "musicidx", executable_name("musicidx")],
            &["musicidx-bin", executable_name("musicidx")],
        ],
    )
    .map(|path| path.display().to_string())
}

fn apply_packaged_env_defaults(
    app: &AppHandle,
    command: &mut Command,
    env_overrides: &HashMap<String, String>,
) {
    if !has_env_override(env_overrides, "MUSICIDX_DB_PATH") && env::var_os("MUSICIDX_DB_PATH").is_none()
    {
        if let Some(app_data_dir) = app_data_dir(app) {
            command.env("MUSICIDX_DB_PATH", app_data_dir.join("musicidx.sqlite"));
        }
    }
    if !has_env_override(env_overrides, "MUSICIDX_MODELS_PATH")
        && env::var_os("MUSICIDX_MODELS_PATH").is_none()
    {
        if let Some(models_path) = packaged_models_path(app) {
            command.env("MUSICIDX_MODELS_PATH", models_path);
        }
    }
    if !has_env_override(env_overrides, "MUSICIDX_FFPROBE_PATH")
        && env::var_os("MUSICIDX_FFPROBE_PATH").is_none()
    {
        if let Some(ffprobe_path) = packaged_bin_path(app, "ffprobe") {
            command.env("MUSICIDX_FFPROBE_PATH", ffprobe_path);
        }
    }
    if !has_env_override(env_overrides, "MUSICIDX_FPCALC_PATH")
        && env::var_os("MUSICIDX_FPCALC_PATH").is_none()
    {
        if let Some(fpcalc_path) = packaged_bin_path(app, "fpcalc") {
            command.env("MUSICIDX_FPCALC_PATH", fpcalc_path);
        }
    }
    if env::var_os("DYLD_LIBRARY_PATH").is_none() {
        if let Some(lib_path) = packaged_lib_path(app) {
            command.env("DYLD_LIBRARY_PATH", lib_path);
        }
    }
}

fn packaged_models_path(app: &AppHandle) -> Option<PathBuf> {
    packaged_resource_path(app, &[&["resources", "models"], &["models"]])
}

fn packaged_bin_path(app: &AppHandle, name: &'static str) -> Option<PathBuf> {
    let executable = executable_name(name);
    packaged_resource_path(
        app,
        &[
            &["resources", "bin", executable],
            &["bin", executable],
            &["resources", executable],
        ],
    )
}

fn packaged_lib_path(app: &AppHandle) -> Option<PathBuf> {
    packaged_resource_path(app, &[&["resources", "lib"], &["lib"]])
}

fn packaged_resource_path(app: &AppHandle, relative_candidates: &[&[&str]]) -> Option<PathBuf> {
    let Ok(resource_dir) = app.path().resource_dir() else {
        return None;
    };
    for relative in relative_candidates {
        let mut path = resource_dir.clone();
        for part in *relative {
            path.push(part);
        }
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn executable_name(name: &'static str) -> &'static str {
    #[cfg(windows)]
    {
        match name {
            "musicidx" => "musicidx.exe",
            "ffprobe" => "ffprobe.exe",
            "fpcalc" => "fpcalc.exe",
            "ffmpeg" => "ffmpeg.exe",
            other => other,
        }
    }
    #[cfg(not(windows))]
    {
        name
    }
}

fn has_env_override(env_overrides: &HashMap<String, String>, key: &str) -> bool {
    env_overrides
        .get(key)
        .is_some_and(|value| !value.trim().is_empty())
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
            cancel_musicidx,
            open_track,
            reveal_track,
            prepare_playback_fallback
        ])
        .run(tauri::generate_context!())
        .expect("error while running MusicIdx desktop app");
}
