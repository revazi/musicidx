import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  Activity,
  ArrowLeft,
  CheckCircle2,
  ChevronDown,
  Database,
  Download,
  FolderOpen,
  ListChecks,
  Loader2,
  Play,
  Search,
  Settings,
  Sparkles,
  Tags,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { Badge } from "./components/ui/badge";
import { Button } from "./components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./components/ui/card";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import { Progress } from "./components/ui/progress";
import { Separator } from "./components/ui/separator";
import { cn } from "./lib/utils";

type MusicidxOutput = {
  status: number;
  success: boolean;
  stdout: string;
  stderr: string;
};

type MusicidxStreamEvent = {
  request_id: string;
  stream: "stdout" | "stderr" | "status";
  line: string;
  status: number | null;
  success: boolean | null;
  done: boolean;
};

type DesktopState = {
  current_dir: string;
  cli_path: string;
  prefix_args: string;
};

type SearchResult = {
  track_id: string;
  path: string;
  title?: string | null;
  artist?: string | null;
  album?: string | null;
  genre?: string | null;
  score?: number;
  why?: string[];
  scores?: Record<string, number>;
  matched_tags?: Array<{ tag: string; score: number; source: string }>;
};

type SearchPayload = {
  query?: string;
  parser?: string;
  llm_error?: string | null;
  diagnostics?: Record<string, unknown>;
  results?: SearchResult[];
};

type SettingsState = {
  cwd: string;
  cliPath: string;
  prefixArgs: string;
  dbPath: string;
  modelsPath: string;
  ffprobePath: string;
  fpcalcPath: string;
  llmProvider: string;
  geminiKey: string;
  musicFolder: string;
  semanticModel: string;
  exportPath: string;
};

type IndexStep = {
  id: string;
  label: string;
  args: () => string[];
};

const SETTINGS_KEY = "musicidx.desktop.settings.v2";
const DEFAULT_SEMANTIC_MODEL = ".musicidx-models/all-MiniLM-L6-v2";

const defaultSettings: SettingsState = {
  cwd: "",
  cliPath: "",
  prefixArgs: "",
  dbPath: "",
  modelsPath: "",
  ffprobePath: "",
  fpcalcPath: "",
  llmProvider: "gemini",
  geminiKey: "",
  musicFolder: "",
  semanticModel: DEFAULT_SEMANTIC_MODEL,
  exportPath: "",
};

export default function App() {
  const [view, setView] = useState<"main" | "settings">("main");
  const [settings, setSettings] = useState<SettingsState>(defaultSettings);
  const [query, setQuery] = useState("chill bar");
  const [limit, setLimit] = useState(10);
  const [useLlm, setUseLlm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [statusError, setStatusError] = useState(false);
  const [rawOutput, setRawOutput] = useState("Ready.");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [advancedIndexing, setAdvancedIndexing] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [pipelineStep, setPipelineStep] = useState("Idle");
  const [pipelineCompleted, setPipelineCompleted] = useState(0);

  const indexSteps = useMemo<IndexStep[]>(
    () => [
      {
        id: "scan",
        label: "Scan files",
        args: () => ["scan", required(settings.musicFolder, "Music folder"), "--json"],
      },
      { id: "metadata", label: "Read metadata", args: () => ["metadata", "--json"] },
      { id: "fingerprint", label: "Fingerprint", args: () => ["fingerprint", "--json"] },
      { id: "basic", label: "Audio features", args: () => ["analyze-basic", "--json"] },
      { id: "tags", label: "ML tags", args: () => ["analyze-tags", "--json"] },
      {
        id: "embed",
        label: "Profile embeddings",
        args: () => {
          const args = ["embed", "--json"];
          if (settings.semanticModel.trim()) {
            args.splice(1, 0, "--model", settings.semanticModel.trim());
          }
          return args;
        },
      },
    ],
    [settings.musicFolder, settings.semanticModel],
  );

  const pipelinePercent = Math.round((pipelineCompleted / indexSteps.length) * 100);

  useEffect(() => {
    void initializeDesktopState();
  }, []);

  async function initializeDesktopState() {
    try {
      const state = await invoke<DesktopState>("desktop_state");
      const saved = loadSettings();
      setSettings({
        ...saved,
        cwd: saved.cwd || state.current_dir,
        cliPath: saved.cliPath || state.cli_path,
        prefixArgs: saved.prefixArgs || state.prefix_args,
      });
      updateStatus("Ready");
    } catch (error) {
      updateStatus("Could not load desktop state", true);
      writeRaw(error);
    }
  }

  function updateSettings(patch: Partial<SettingsState>) {
    setSettings((current) => {
      const next = { ...current, ...patch };
      persistSettings(next);
      return next;
    });
  }

  function saveSettings() {
    persistSettings(settings);
    updateStatus("Settings saved");
  }

  async function runFullIndexing() {
    persistSettings(settings);
    setPipelineRunning(true);
    setPipelineCompleted(0);
    setResults([]);
    try {
      for (const [index, step] of indexSteps.entries()) {
        setPipelineStep(step.label);
        await runJsonCommand(step.args());
        setPipelineCompleted(index + 1);
      }
      setPipelineStep("Complete");
      updateStatus("Indexing complete");
    } catch (error) {
      setPipelineStep("Stopped");
      writeRaw(error);
      updateStatus("Indexing failed", true);
    } finally {
      setPipelineRunning(false);
    }
  }

  async function runAdvancedStep(step: IndexStep) {
    persistSettings(settings);
    setPipelineRunning(true);
    setPipelineStep(step.label);
    setPipelineCompleted(0);
    try {
      await runJsonCommand(step.args());
      setPipelineCompleted(indexSteps.length);
      setPipelineStep("Complete");
    } finally {
      setPipelineRunning(false);
    }
  }

  async function searchMusic() {
    persistSettings(settings);
    const args = [
      "search",
      required(query, "Query"),
      "--format",
      "json",
      "--concise",
      "--limit",
      String(normalizedLimit()),
      "--explain",
    ];
    appendCommonSearchArgs(args);
    const payload = await runJsonCommand<SearchPayload>(args);
    setResults(payload.results ?? []);
    writeRaw({
      summary: {
        query: payload.query,
        parser: payload.parser,
        llm_error: payload.llm_error,
        diagnostics: payload.diagnostics,
        result_count: payload.results?.length ?? 0,
      },
      results: payload.results ?? [],
    });
  }

  async function parseIntent() {
    const args = ["parse", required(query, "Query"), "--json"];
    appendCommonSearchArgs(args);
    await runJsonCommand(args);
  }

  async function runEval() {
    const args = ["eval", "eval/search_queries.json", "--limit", String(normalizedLimit()), "--json"];
    appendCommonSearchArgs(args);
    await runJsonCommand(args);
  }

  async function exportPlaylist() {
    let out = settings.exportPath.trim();
    if (!out) {
      const selected = await save({ defaultPath: "musicidx-playlist.m3u" });
      if (typeof selected !== "string") {
        return;
      }
      out = selected;
      updateSettings({ exportPath: selected });
    }

    const args = [
      "export",
      required(query, "Query"),
      "--limit",
      String(normalizedLimit()),
      "--out",
      out,
      "--absolute-paths",
    ];
    appendCommonSearchArgs(args);
    await runTextCommand(args);
  }

  async function saveFeedback(result: SearchResult, rating: "good" | "bad" | "neutral") {
    const args = [
      "feedback",
      "--track-id",
      result.track_id,
      "--query",
      required(query, "Query"),
      "--rating",
      rating,
      "--json",
    ];
    await runJsonCommand(args);
    updateStatus(`Saved ${rating} feedback`);
  }

  async function chooseDirectory(field: keyof SettingsState) {
    const selected = await open({ directory: true, multiple: false });
    if (typeof selected === "string") {
      updateSettings({ [field]: selected } as Partial<SettingsState>);
    }
  }

  async function chooseSavePath(field: keyof SettingsState, defaultPath: string) {
    const selected = await save({ defaultPath });
    if (typeof selected === "string") {
      updateSettings({ [field]: selected } as Partial<SettingsState>);
    }
  }

  async function runJsonCommand<T = unknown>(args: string[]): Promise<T> {
    const output = await runTextCommand(args);
    const parsed = parseJsonish<T>(output.stdout);
    writeRaw(parsed);
    return parsed;
  }

  async function runTextCommand(args: string[]): Promise<MusicidxOutput> {
    setBusy(true);
    updateStatus(`Running: ${displayCommand(args)}`);
    setRawOutput(`$ ${displayCommand(args)}\n\n`);
    try {
      const output = await runStreamedProcess(args);
      if (!output.success) {
        throw new Error(output.stderr || output.stdout || `musicidx exited with ${output.status}`);
      }
      updateStatus(`Done: exit ${output.status}`);
      return output;
    } catch (error) {
      updateStatus("Command failed", true);
      writeRaw(error);
      throw error;
    } finally {
      setBusy(false);
    }
  }

  async function runStreamedProcess(args: string[]): Promise<MusicidxOutput> {
    const requestId = randomId();
    let unlisten: UnlistenFn | null = null;
    let stdout = "";
    let stderr = "";

    return new Promise<MusicidxOutput>((resolve, reject) => {
      let settled = false;
      const cleanup = () => {
        if (unlisten) {
          unlisten();
          unlisten = null;
        }
      };

      listen<MusicidxStreamEvent>("musicidx-output", (event) => {
        const payload = event.payload;
        if (payload.request_id !== requestId || settled) {
          return;
        }

        if (!payload.done) {
          const line = `${payload.line}\n`;
          if (payload.stream === "stdout") {
            stdout += line;
            setRawOutput((current) => `${current}${line}`);
          } else if (payload.stream === "stderr") {
            stderr += line;
            setRawOutput((current) => `${current}[stderr] ${line}`);
          }
          return;
        }

        settled = true;
        cleanup();
        resolve({
          status: payload.status ?? -1,
          success: payload.success ?? false,
          stdout,
          stderr,
        });
      })
        .then((listener) => {
          unlisten = listener;
          return invoke<void>("run_musicidx_stream", {
            requestId,
            args,
            cwd: settings.cwd.trim() || null,
            cliPath: settings.cliPath.trim() || null,
            prefixArgs: settings.prefixArgs.trim() || null,
            envOverrides: envOverrides(settings),
          });
        })
        .catch((error) => {
          settled = true;
          cleanup();
          reject(error);
        });
    });
  }

  function appendCommonSearchArgs(args: string[]) {
    if (settings.semanticModel.trim() && args[0] !== "embed") {
      args.push("--semantic-model", settings.semanticModel.trim());
    }
    if (useLlm) {
      args.push("--llm", "--llm-provider", settings.llmProvider.trim() || "gemini");
    }
  }

  function normalizedLimit() {
    return Math.max(1, Math.min(100, Number.isFinite(limit) ? limit : 10));
  }

  function updateStatus(message: string, isError = false) {
    setStatus(message);
    setStatusError(isError);
  }

  function writeRaw(value: unknown) {
    setRawOutput(formatValue(value));
  }

  function displayCommand(args: string[]) {
    return [settings.cliPath.trim() || "musicidx", settings.prefixArgs.trim(), ...args]
      .filter(Boolean)
      .join(" ");
  }

  if (view === "settings") {
    return (
      <Shell
        status={status}
        statusError={statusError}
        onDoctor={() => runJsonCommand(["doctor", "--json"])}
        onSettings={() => setView("settings")}
      >
        <Card className="border-none shadow-sm">
          <CardHeader className="flex flex-row items-start gap-4 space-y-0">
            <Button variant="ghost" size="icon" onClick={() => setView("main")}>
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div className="space-y-1">
              <CardTitle>Settings</CardTitle>
              <CardDescription>
                Local paths and optional LLM configuration. These stay on this device.
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <section className="space-y-4 rounded-lg border bg-background/40 p-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <h3 className="text-sm font-semibold">Indexing</h3>
                  <p className="text-sm text-muted-foreground">
                    Configure the music folder and optional advanced indexing steps.
                  </p>
                </div>
                <Badge variant={pipelineRunning ? "default" : "secondary"}>
                  {pipelineRunning ? "Running" : pipelineStep}
                </Badge>
              </div>
              <Field label="Music folder">
                <PathInput
                  value={settings.musicFolder}
                  onChange={(musicFolder) => updateSettings({ musicFolder })}
                  onChoose={() => chooseDirectory("musicFolder")}
                />
              </Field>
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-muted-foreground">{pipelineStep}</span>
                  <span className="tabular-nums text-muted-foreground">{pipelinePercent}%</span>
                </div>
                <Progress value={pipelinePercent} />
              </div>
              <div className="flex flex-col gap-2 sm:flex-row">
                <Button disabled={busy || pipelineRunning || !settings.musicFolder} onClick={runFullIndexing}>
                  {pipelineRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                  Run Indexing
                </Button>
                <Button
                  variant="outline"
                  type="button"
                  onClick={() => setAdvancedIndexing((value) => !value)}
                >
                  Advanced steps
                  <ChevronDown
                    className={cn("h-4 w-4 transition-transform", advancedIndexing && "rotate-180")}
                  />
                </Button>
              </div>
              {advancedIndexing ? (
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {indexSteps.map((step) => (
                    <Button
                      key={step.id}
                      variant="secondary"
                      disabled={busy || pipelineRunning}
                      onClick={() => runAdvancedStep(step)}
                    >
                      {step.label}
                    </Button>
                  ))}
                  <Button variant="outline" disabled={busy} onClick={runEval}>
                    <ListChecks className="h-4 w-4" />
                    Run eval
                  </Button>
                  <Button variant="outline" disabled={busy} onClick={() => runJsonCommand(["db-info", "--json"])}>
                    <Database className="h-4 w-4" />
                    DB info
                  </Button>
                </div>
              ) : null}
            </section>

            <div className="grid gap-4 md:grid-cols-2">
              <Field label="Working directory" className="md:col-span-2">
                <PathInput
                  value={settings.cwd}
                  onChange={(cwd) => updateSettings({ cwd })}
                  onChoose={() => chooseDirectory("cwd")}
                />
              </Field>
              <Field label="CLI path">
                <Input
                  value={settings.cliPath}
                  placeholder="musicidx or uv"
                  onChange={(event) => updateSettings({ cliPath: event.target.value })}
                />
              </Field>
              <Field label="CLI prefix args">
                <Input
                  value={settings.prefixArgs}
                  placeholder="run musicidx"
                  onChange={(event) => updateSettings({ prefixArgs: event.target.value })}
                />
              </Field>
              <Field label="SQLite DB path">
                <PathInput
                  value={settings.dbPath}
                  placeholder="optional MUSICIDX_DB_PATH"
                  onChange={(dbPath) => updateSettings({ dbPath })}
                  onChoose={() => chooseSavePath("dbPath", "musicidx.sqlite")}
                />
              </Field>
              <Field label="Models path">
                <PathInput
                  value={settings.modelsPath}
                  placeholder="optional MUSICIDX_MODELS_PATH"
                  onChange={(modelsPath) => updateSettings({ modelsPath })}
                  onChoose={() => chooseDirectory("modelsPath")}
                />
              </Field>
              <Field label="Semantic model" className="md:col-span-2">
                <Input
                  value={settings.semanticModel}
                  onChange={(event) => updateSettings({ semanticModel: event.target.value })}
                />
              </Field>
              <Field label="Playlist export path" className="md:col-span-2">
                <PathInput
                  value={settings.exportPath}
                  placeholder="optional default .m3u path"
                  onChange={(exportPath) => updateSettings({ exportPath })}
                  onChoose={() => chooseSavePath("exportPath", "musicidx-playlist.m3u")}
                />
              </Field>
              <Field label="ffprobe path">
                <Input
                  value={settings.ffprobePath}
                  placeholder="optional MUSICIDX_FFPROBE_PATH"
                  onChange={(event) => updateSettings({ ffprobePath: event.target.value })}
                />
              </Field>
              <Field label="fpcalc path">
                <Input
                  value={settings.fpcalcPath}
                  placeholder="optional MUSICIDX_FPCALC_PATH"
                  onChange={(event) => updateSettings({ fpcalcPath: event.target.value })}
                />
              </Field>
              <Field label="LLM provider">
                <Input
                  value={settings.llmProvider}
                  onChange={(event) => updateSettings({ llmProvider: event.target.value })}
                />
              </Field>
              <Field label="Gemini API key">
                <Input
                  type="password"
                  value={settings.geminiKey}
                  placeholder="optional GEMINI_API_KEY"
                  onChange={(event) => updateSettings({ geminiKey: event.target.value })}
                />
              </Field>
            </div>

            <Separator />

            <div className="grid gap-3 rounded-lg bg-muted p-4 text-sm md:grid-cols-2">
              <div className="min-w-0">
                <p className="text-muted-foreground">CLI command</p>
                <code className="wrap-anywhere">
                  {[settings.cliPath, settings.prefixArgs].filter(Boolean).join(" ") || "musicidx"}
                </code>
              </div>
              <div>
                <p className="text-muted-foreground">Config source</p>
                <p>Desktop settings + repo .env</p>
              </div>
            </div>

            <div className="flex flex-col gap-2 sm:flex-row">
              <Button onClick={saveSettings}>Save settings</Button>
              <Button variant="outline" onClick={() => runJsonCommand(["doctor", "--json"])}>
                Run doctor
              </Button>
            </div>
          </CardContent>
        </Card>
      </Shell>
    );
  }

  return (
    <Shell
      status={status}
      statusError={statusError}
      onDoctor={() => runJsonCommand(["doctor", "--json"])}
      onSettings={() => setView("settings")}
    >
      <Card className="border-none bg-card/80 shadow-sm backdrop-blur">
        <CardContent className="p-4 sm:p-5">
          <div className="grid gap-3">
            <div className="flex min-w-0 items-center gap-2 rounded-full border bg-background/70 px-3 py-2 shadow-sm focus-within:ring-1 focus-within:ring-ring">
              <Search className="h-4 w-4 shrink-0 text-muted-foreground" />
              <input
                className="h-10 min-w-0 flex-1 bg-transparent text-base outline-none placeholder:text-muted-foreground"
                value={query}
                placeholder="Search your library…"
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    void searchMusic();
                  }
                }}
              />
              <Button className="rounded-full px-5" disabled={busy} onClick={searchMusic}>
                Search
              </Button>
            </div>
            <div className="flex flex-wrap items-center justify-between gap-3 px-1 text-sm text-muted-foreground">
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-2">
                  <span>Limit</span>
                  <Input
                    className="h-8 w-20"
                    type="number"
                    min={1}
                    max={100}
                    value={limit}
                    onChange={(event) => setLimit(Number.parseInt(event.target.value, 10))}
                  />
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={useLlm}
                    onChange={(event) => setUseLlm(event.target.checked)}
                  />
                  LLM hints
                </label>
              </div>
              <Button variant="ghost" size="sm" disabled={busy} onClick={parseIntent}>
                <Sparkles className="h-4 w-4" />
                Parse intent
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="border-none shadow-sm">
        <CardHeader className="flex flex-row items-start justify-between gap-4 space-y-0">
          <div>
            <CardTitle>Results</CardTitle>
            <CardDescription>Feedback buttons improve future local ranking.</CardDescription>
          </div>
          <Badge variant="secondary">{results.length} results</Badge>
        </CardHeader>
        <CardContent>
          {results.length === 0 ? (
            <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
              Run a search to see tracks.
            </div>
          ) : (
            <div className="grid gap-3">
              {results.map((result, index) => (
                <ResultCard
                  key={result.track_id}
                  result={result}
                  index={index}
                  busy={busy}
                  onFeedback={saveFeedback}
                />
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <details className="rounded-xl border bg-card p-4 text-card-foreground shadow-sm" open>
        <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm font-medium">
          <span>Live/raw output</span>
          <span className="text-xs text-muted-foreground">expand/collapse command logs</span>
        </summary>
        <div className="mt-3 flex items-center justify-between gap-3">
          <p className="text-sm text-muted-foreground">Streaming stdout/stderr from the local CLI.</p>
          <Button variant="outline" size="sm" onClick={() => setRawOutput("Ready.")}>Clear</Button>
        </div>
        <pre className="mt-3 max-h-96 max-w-full overflow-y-auto overflow-x-hidden whitespace-pre-wrap rounded-lg bg-zinc-950 p-4 text-xs text-zinc-100 wrap-anywhere">
          {rawOutput}
        </pre>
      </details>

      {(pipelineRunning || pipelineStep !== "Idle") && (
        <div className="fixed bottom-24 left-4 z-40 w-[min(calc(100vw-2rem),22rem)] rounded-xl border bg-card/95 p-4 shadow-xl backdrop-blur">
          <div className="mb-2 flex items-center justify-between gap-3 text-sm">
            <span className="font-medium">Indexing</span>
            <span className="tabular-nums text-muted-foreground">{pipelinePercent}%</span>
          </div>
          <Progress value={pipelinePercent} />
          <p className="mt-2 wrap-anywhere text-xs text-muted-foreground">{pipelineStep}</p>
        </div>
      )}

      <FloatingActionButton
        side="left"
        label={settings.musicFolder ? "Run indexing" : "Select folder in Settings"}
        disabled={busy || pipelineRunning}
        onClick={() => {
          if (!settings.musicFolder) {
            setView("settings");
            return;
          }
          void runFullIndexing();
        }}
      >
        {pipelineRunning ? <Loader2 className="h-5 w-5 animate-spin" /> : <Tags className="h-5 w-5" />}
      </FloatingActionButton>

      <FloatingActionButton
        side="right"
        label="Export playlist"
        disabled={busy}
        onClick={() => void exportPlaylist()}
      >
        <Download className="h-5 w-5" />
      </FloatingActionButton>
    </Shell>
  );
}

function Shell({
  children,
  status,
  statusError,
  onDoctor,
  onSettings,
}: {
  children: React.ReactNode;
  status: string;
  statusError: boolean;
  onDoctor: () => void;
  onSettings: () => void;
}) {
  return (
    <div className="dark min-h-screen bg-background text-foreground [background:radial-gradient(circle_at_20%_0%,rgba(168,85,247,0.18),transparent_28rem),radial-gradient(circle_at_90%_10%,rgba(217,70,239,0.10),transparent_24rem),hsl(var(--background))]">
      <div className="mx-auto grid w-full max-w-6xl gap-4 px-4 py-4 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-4 rounded-xl border bg-card p-5 shadow-sm md:flex-row md:items-center md:justify-between">
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
              Local-first music search
            </p>
            <h1 className="mt-1 text-3xl font-semibold tracking-tight">MusicIdx</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Index, search, judge, and export your local music library.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={statusError ? "destructive" : "secondary"} className="max-w-full wrap-anywhere">
              {status}
            </Badge>
            <Button variant="outline" onClick={onDoctor}>
              <Activity className="h-4 w-4" />
              Doctor
            </Button>
            <Button variant="outline" size="icon" onClick={onSettings} aria-label="Settings">
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </header>
        {children}
      </div>
    </div>
  );
}

function Field({
  label,
  children,
  className,
}: {
  label: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("grid gap-2", className)}>
      <Label>{label}</Label>
      {children}
    </div>
  );
}

function PathInput({
  value,
  onChange,
  onChoose,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  onChoose: () => void;
  placeholder?: string;
}) {
  return (
    <div className="grid min-w-0 gap-2 sm:grid-cols-[minmax(0,1fr)_auto]">
      <Input value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} />
      <Button variant="secondary" onClick={onChoose}>
        <FolderOpen className="h-4 w-4" />
        Choose
      </Button>
    </div>
  );
}

function FloatingActionButton({
  side,
  label,
  disabled,
  onClick,
  children,
}: {
  side: "left" | "right";
  label: string;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <div
      className={cn(
        "group fixed bottom-5 z-50",
        side === "left" ? "left-4" : "right-4",
      )}
    >
      <div
        className={cn(
          "pointer-events-none absolute bottom-16 whitespace-nowrap rounded-md border bg-card px-3 py-1.5 text-xs text-card-foreground opacity-0 shadow-lg transition-opacity group-hover:opacity-100",
          side === "left" ? "left-0" : "right-0",
        )}
      >
        {label}
      </div>
      <Button
        size="icon"
        disabled={disabled}
        onClick={onClick}
        aria-label={label}
        className="h-14 w-14 rounded-full bg-primary/90 shadow-2xl shadow-primary/20 backdrop-blur hover:bg-primary"
      >
        {children}
      </Button>
    </div>
  );
}

function ResultCard({
  result,
  index,
  busy,
  onFeedback,
}: {
  result: SearchResult;
  index: number;
  busy: boolean;
  onFeedback: (result: SearchResult, rating: "good" | "bad" | "neutral") => Promise<void>;
}) {
  const [savedRating, setSavedRating] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const title = result.title || result.path.split(/[\\/]/).pop() || result.track_id;
  const meta = [result.artist, result.album, result.genre].filter(Boolean).join(" · ");
  const why = result.why?.length ? result.why.join("; ") : "No explanation available.";
  const tagText = result.matched_tags?.length
    ? result.matched_tags.map((tag) => `${tag.tag} ${tag.score.toFixed(2)}`).join(", ")
    : "";
  const hasLongContent =
    why.length > 180 || tagText.length > 140 || result.path.length > 120 || title.length > 80;

  async function rate(rating: "good" | "bad" | "neutral") {
    await onFeedback(result, rating);
    setSavedRating(rating);
  }

  return (
    <div className="grid min-w-0 gap-3 rounded-lg border bg-background/40 p-4">
      <div className="flex min-w-0 items-start justify-between gap-3">
        <div className="min-w-0">
          <h3
            className={cn(
              "wrap-anywhere text-sm font-semibold",
              !expanded && hasLongContent && "line-clamp-2",
            )}
          >
            {index + 1}. {title}
          </h3>
          <p className="wrap-anywhere text-sm text-muted-foreground">{meta || "Unknown artist/album"}</p>
        </div>
        <Badge variant="secondary">{(result.score ?? 0).toFixed(3)}</Badge>
      </div>
      <div
        className={cn(
          "grid min-w-0 gap-2 transition-all",
          !expanded && hasLongContent && "max-h-24 overflow-hidden",
        )}
      >
        <p className="wrap-anywhere text-sm text-muted-foreground">{why}</p>
        {tagText ? (
          <p className="wrap-anywhere text-xs text-muted-foreground">Tags: {tagText}</p>
        ) : null}
        <p className="wrap-anywhere text-xs text-muted-foreground">{result.path}</p>
      </div>
      {hasLongContent ? (
        <Button
          size="sm"
          variant="ghost"
          className="w-fit px-0 text-muted-foreground hover:bg-transparent hover:text-foreground"
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? "Show less" : "Show full result"}
        </Button>
      ) : null}
      <div className="flex flex-wrap gap-2">
        <Button size="sm" variant="outline" disabled={busy || savedRating === "good"} onClick={() => rate("good")}>
          {savedRating === "good" ? <CheckCircle2 className="h-3.5 w-3.5" /> : <ThumbsUp className="h-3.5 w-3.5" />}
          Good
        </Button>
        <Button size="sm" variant="outline" disabled={busy || savedRating === "bad"} onClick={() => rate("bad")}>
          <ThumbsDown className="h-3.5 w-3.5" />
          Bad
        </Button>
        <Button size="sm" variant="ghost" disabled={busy || savedRating === "neutral"} onClick={() => rate("neutral")}>
          Neutral
        </Button>
      </div>
    </div>
  );
}

function envOverrides(settings: SettingsState): Record<string, string> {
  const env: Record<string, string> = {};
  setEnv(env, "MUSICIDX_DB_PATH", settings.dbPath);
  setEnv(env, "MUSICIDX_MODELS_PATH", settings.modelsPath);
  setEnv(env, "MUSICIDX_FFPROBE_PATH", settings.ffprobePath);
  setEnv(env, "MUSICIDX_FPCALC_PATH", settings.fpcalcPath);
  setEnv(env, "GEMINI_API_KEY", settings.geminiKey);
  return env;
}

function setEnv(env: Record<string, string>, key: string, value: string) {
  const trimmed = value.trim();
  if (trimmed) {
    env[key] = trimmed;
  }
}

function persistSettings(settings: SettingsState) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function loadSettings(): SettingsState {
  try {
    return { ...defaultSettings, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") };
  } catch {
    return defaultSettings;
  }
}

function parseJsonish<T>(text: string): T {
  const trimmed = text.trim();
  try {
    return JSON.parse(trimmed) as T;
  } catch {
    const firstObject = trimmed.indexOf("{");
    const lastObject = trimmed.lastIndexOf("}");
    if (firstObject >= 0 && lastObject > firstObject) {
      return JSON.parse(trimmed.slice(firstObject, lastObject + 1)) as T;
    }
    const firstArray = trimmed.indexOf("[");
    const lastArray = trimmed.lastIndexOf("]");
    if (firstArray >= 0 && lastArray > firstArray) {
      return JSON.parse(trimmed.slice(firstArray, lastArray + 1)) as T;
    }
    throw new Error(`Could not parse JSON output: ${trimmed.slice(0, 240)}`);
  }
}

function formatValue(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (value instanceof Error) {
    return value.message;
  }
  return JSON.stringify(value, null, 2);
}

function randomId(): string {
  return crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function required(value: string, label: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    throw new Error(`${label} is required`);
  }
  return trimmed;
}
