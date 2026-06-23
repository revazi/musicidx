import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  Activity,
  ArrowLeft,
  CheckCircle2,
  ChevronDown,
  Database,
  Download,
  SkipBack,
  SkipForward,
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
  Trash2,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

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

type PlaybackSource = {
  path: string;
  transcoded: boolean;
  detail: string;
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
  models_path?: string;
  semantic_model?: string;
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
  saved_feedback_rating?: "good" | "bad" | "neutral" | null;
};

type SearchPayload = {
  query?: string;
  parser?: string;
  llm_error?: string | null;
  diagnostics?: Record<string, unknown>;
  results?: SearchResult[];
};

type ThemeMode = "system" | "light" | "dark";
type IndexAnalysisMode = "quality" | "full";

type SettingsState = {
  cwd: string;
  cliPath: string;
  prefixArgs: string;
  dbPath: string;
  modelsPath: string;
  ffprobePath: string;
  fpcalcPath: string;
  llmProvider: string;
  llmModel: string;
  geminiKey: string;
  musicFolder: string;
  semanticModel: string;
  exportPath: string;
  themeMode: ThemeMode;
  indexResourceProfile: string;
  indexAnalysisMode: IndexAnalysisMode;
  backgroundIndexingEnabled: boolean;
  backgroundIndexIntervalMinutes: number;
  backgroundIndexResourceProfile: string;
};

type IndexStep = {
  id: string;
  label: string;
  args: () => string[];
};

type IndexCommandPayload = {
  duration_sec?: number;
  peak_memory_mb?: number | null;
  child_peak_memory_mb?: number | null;
  processed?: number;
  updated?: number;
  skipped?: number;
  errors?: number;
  added?: number;
  unchanged?: number;
  modified?: number;
  missing?: number;
  total_seen?: number;
  root_missing?: boolean;
  batches?: number;
  batch_size?: number;
  quick?: boolean;
  chunked?: boolean;
  chunk_sec?: number;
  workers?: number;
};

type PipelineSummary = {
  id: string;
  label: string;
  text: string;
  hasErrors: boolean;
};

const SETTINGS_KEY = "musicidx.desktop.settings.v2";
const DEFAULT_SEMANTIC_MODEL = ".musicidx-models/all-MiniLM-L6-v2";
const MAX_RAW_OUTPUT_CHARS = 512 * 1024;
const DEFAULT_BACKGROUND_INDEX_INTERVAL_MINUTES = 1;
const BACKGROUND_INDEX_INTERVAL_OPTIONS_MINUTES = [1, 5, 10, 30, 60] as const;

const defaultSettings: SettingsState = {
  cwd: "",
  cliPath: "",
  prefixArgs: "",
  dbPath: "",
  modelsPath: "",
  ffprobePath: "",
  fpcalcPath: "",
  llmProvider: "gemini",
  llmModel: "gemini-2.0-flash",
  geminiKey: "",
  musicFolder: "",
  semanticModel: DEFAULT_SEMANTIC_MODEL,
  exportPath: "",
  themeMode: "system",
  indexResourceProfile: "auto",
  indexAnalysisMode: "quality",
  backgroundIndexingEnabled: true,
  backgroundIndexIntervalMinutes: DEFAULT_BACKGROUND_INDEX_INTERVAL_MINUTES,
  backgroundIndexResourceProfile: "balanced",
};

export default function App() {
  const [view, setView] = useState<"main" | "settings">("main");
  const [settings, setSettings] = useState<SettingsState>(defaultSettings);
  const [systemTheme, setSystemTheme] = useState<"light" | "dark">(() =>
    window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light",
  );
  const [query, setQuery] = useState("chill bar");
  const [limit, setLimit] = useState(10);
  const [useLlm, setUseLlm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [statusError, setStatusError] = useState(false);
  const [rawOutput, setRawOutput] = useState("Ready.");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [playerTrack, setPlayerTrack] = useState<SearchResult | null>(null);
  const [playerSrc, setPlayerSrc] = useState("");
  const [playerError, setPlayerError] = useState("");
  const [playerUsingFallback, setPlayerUsingFallback] = useState(false);
  const [advancedIndexing, setAdvancedIndexing] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [pipelineMode, setPipelineMode] = useState<"manual" | "background" | null>(null);
  const [pipelineStep, setPipelineStep] = useState("Idle");
  const [pipelineCompleted, setPipelineCompleted] = useState(0);
  const [pipelineSummaries, setPipelineSummaries] = useState<PipelineSummary[]>([]);
  const [backgroundStatus, setBackgroundStatus] = useState("Background watcher idle");
  const [settingsSaveStatus, setSettingsSaveStatus] = useState("");
  const [canceling, setCanceling] = useState(false);
  const currentRequestIdRef = useRef<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const cancelRequestedRef = useRef(false);
  const backgroundJobRunningRef = useRef(false);
  const busyRef = useRef(false);
  const pipelineRunningRef = useRef(false);

  const indexSteps = useMemo<IndexStep[]>(
    () => [
      {
        id: "scan",
        label: "Scan files",
        args: () => ["scan", required(settings.musicFolder, "Music folder"), "--json"],
      },
      {
        id: "metadata",
        label: "Read metadata",
        args: () => ["metadata", "--missing-only", "--json"],
      },
      {
        id: "fingerprint",
        label: "Fingerprint",
        args: () => ["fingerprint", "--missing-only", "--json"],
      },
      {
        id: "basic",
        label: "Audio features",
        args: () => basicAnalysisArgs(settings.indexAnalysisMode, settings.indexResourceProfile),
      },
      {
        id: "tags",
        label: "ML tags",
        args: () => [
          "analyze-tags",
          "--missing-only",
          "--workers",
          "auto",
          "--resource-profile",
          settings.indexResourceProfile,
          "--subprocess-batches",
          "--batch-size",
          "auto",
          "--json",
        ],
      },
      {
        id: "embed",
        label: "Profile embeddings",
        args: () => {
          const args = [
            "embed",
            "--batch-size",
            "auto",
            "--resource-profile",
            settings.indexResourceProfile,
            "--json",
          ];
          if (settings.semanticModel.trim()) {
            args.splice(1, 0, "--model", settings.semanticModel.trim());
          }
          return args;
        },
      },
    ],
    [
      settings.indexAnalysisMode,
      settings.indexResourceProfile,
      settings.musicFolder,
      settings.semanticModel,
    ],
  );

  const backgroundIndexSteps = useMemo<IndexStep[]>(
    () => [
      {
        id: "scan",
        label: "Scan files",
        args: () => ["scan", required(settings.musicFolder, "Music folder"), "--json"],
      },
      {
        id: "metadata",
        label: "Read metadata",
        args: () => ["metadata", "--missing-only", "--json"],
      },
      {
        id: "fingerprint",
        label: "Fingerprint",
        args: () => ["fingerprint", "--missing-only", "--json"],
      },
      {
        id: "basic",
        label: "Audio features",
        args: () => [
          "analyze-basic",
          "--chunked",
          "--chunk-sec",
          "auto",
          "--workers",
          "auto",
          "--resource-profile",
          settings.backgroundIndexResourceProfile,
          "--json",
        ],
      },
      {
        id: "tags",
        label: "ML tags",
        args: () => [
          "analyze-tags",
          "--missing-only",
          "--workers",
          "auto",
          "--resource-profile",
          settings.backgroundIndexResourceProfile,
          "--subprocess-batches",
          "--batch-size",
          "auto",
          "--json",
        ],
      },
      {
        id: "embed",
        label: "Profile embeddings",
        args: () => {
          const args = [
            "embed",
            "--batch-size",
            "auto",
            "--resource-profile",
            settings.backgroundIndexResourceProfile,
            "--json",
          ];
          if (settings.semanticModel.trim()) {
            args.splice(1, 0, "--model", settings.semanticModel.trim());
          }
          return args;
        },
      },
    ],
    [settings.backgroundIndexResourceProfile, settings.musicFolder, settings.semanticModel],
  );

  const pipelinePercent = Math.round((pipelineCompleted / indexSteps.length) * 100);
  const effectiveTheme = settings.themeMode === "system" ? systemTheme : settings.themeMode;
  const backgroundIndexIntervalMinutes = normalizeBackgroundIndexIntervalMinutes(
    settings.backgroundIndexIntervalMinutes,
  );
  const backgroundIndexIntervalMs = backgroundIndexIntervalMinutes * 60 * 1000;
  const backgroundIndexIntervalLabel = formatMinutes(backgroundIndexIntervalMinutes);

  useEffect(() => {
    void initializeDesktopState();
  }, []);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setSystemTheme(media.matches ? "dark" : "light");
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    busyRef.current = busy;
  }, [busy]);

  useEffect(() => {
    pipelineRunningRef.current = pipelineRunning;
  }, [pipelineRunning]);

  useEffect(() => {
    if (!playerSrc || !audioRef.current) {
      return;
    }
    audioRef.current.load();
    void audioRef.current.play().catch((error) => {
      writeRaw(error);
      updateStatus("Could not play track in MusicIdx", true);
    });
  }, [playerSrc]);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", effectiveTheme === "dark");
    document.documentElement.classList.toggle("light", effectiveTheme === "light");
    document.documentElement.style.colorScheme = effectiveTheme;
    void invoke("set_window_theme", {
      theme: settings.themeMode === "system" ? "system" : effectiveTheme,
    }).catch(() => undefined);
  }, [effectiveTheme, settings.themeMode]);

  useEffect(() => {
    if (!settings.backgroundIndexingEnabled) {
      setBackgroundStatus("Background indexing disabled");
      return undefined;
    }
    if (!settings.musicFolder.trim()) {
      setBackgroundStatus("Background watcher waiting for a music folder");
      return undefined;
    }

    setBackgroundStatus(
      `Background watcher idle · checks every ${backgroundIndexIntervalLabel} · ${settings.backgroundIndexResourceProfile} profile`,
    );
    const interval = window.setInterval(() => {
      if (!busyRef.current && !pipelineRunningRef.current) {
        void runBackgroundCheck();
      }
    }, backgroundIndexIntervalMs);
    return () => window.clearInterval(interval);
  }, [
    settings.backgroundIndexingEnabled,
    settings.musicFolder,
    settings.cwd,
    settings.cliPath,
    settings.prefixArgs,
    settings.dbPath,
    settings.modelsPath,
    settings.ffprobePath,
    settings.fpcalcPath,
    settings.semanticModel,
    settings.backgroundIndexResourceProfile,
    backgroundIndexIntervalMs,
    backgroundIndexIntervalLabel,
  ]);

  async function initializeDesktopState() {
    try {
      const state = await invoke<DesktopState>("desktop_state");
      const saved = loadSettings();
      setSettings({
        ...saved,
        cwd: saved.cwd || state.current_dir,
        cliPath: saved.cliPath || state.cli_path,
        prefixArgs: saved.prefixArgs || state.prefix_args,
        modelsPath: saved.modelsPath || state.models_path || "",
        semanticModel: packagedSemanticModelDefault(saved.semanticModel, state.semantic_model),
      });
      updateStatus("Ready");
    } catch (error) {
      updateStatus("Could not load desktop state", true);
      writeRaw(error);
    }
  }

  function updateSettings(patch: Partial<SettingsState>) {
    setSettingsSaveStatus("");
    setSettings((current) => {
      const next = { ...current, ...patch };
      persistSettings(next);
      return next;
    });
  }

  function saveSettings() {
    persistSettings(settings);
    const message = `Settings saved · ${formatClock(new Date())}`;
    setSettingsSaveStatus(message);
    updateStatus(message);
  }

  async function runBackgroundCheck() {
    if (
      backgroundJobRunningRef.current ||
      busyRef.current ||
      pipelineRunningRef.current ||
      !settings.backgroundIndexingEnabled ||
      !settings.musicFolder.trim()
    ) {
      return;
    }

    backgroundJobRunningRef.current = true;
    cancelRequestedRef.current = false;
    setCanceling(false);
    try {
      setBackgroundStatus("Background watcher checking for changes…");
      const scanStep = backgroundIndexSteps[0];
      const scanPayload = await runJsonCommand<IndexCommandPayload>(scanStep.args());
      const checkedAt = formatClock(new Date());
      const changeCount = scanChangeCount(scanPayload);
      const needsDerivedIndexing = scanNeedsDerivedIndexing(scanPayload);
      if (changeCount <= 0) {
        setBackgroundStatus(
          scanPayload.root_missing
            ? `Music folder not found · no active tracks to mark missing · checked ${checkedAt}`
            : `No changes · checked ${checkedAt}`,
        );
        if (scanPayload.root_missing) {
          updateStatus("Music folder not found", true);
        }
        return;
      }
      if (!needsDerivedIndexing) {
        const message = scanPayload.root_missing
          ? `Music folder not found · ${describeScanChanges(scanPayload)} marked missing · checked ${checkedAt}`
          : `${describeScanChanges(scanPayload)} · library state updated · checked ${checkedAt}`;
        setBackgroundStatus(message);
        setPipelineSummaries([summarizeIndexStep(scanStep, scanPayload)]);
        updateStatus(
          scanPayload.root_missing ? "Music folder not found; tracks marked missing" : "Library removals updated",
          Boolean(scanPayload.root_missing),
        );
        return;
      }

      setBackgroundStatus(`${describeScanChanges(scanPayload)} · indexing…`);
      setPipelineMode("background");
      setPipelineRunning(true);
      setPipelineCompleted(1);
      setPipelineStep("Background changes detected");
      setPipelineSummaries([summarizeIndexStep(scanStep, scanPayload)]);

      const remainingSteps = backgroundIndexSteps.slice(1);
      for (const [index, step] of remainingSteps.entries()) {
        setPipelineStep(`Background: ${step.label}`);
        const payload = await runJsonCommand<IndexCommandPayload>(step.args());
        setPipelineSummaries((current) => [...current, summarizeIndexStep(step, payload)]);
        setPipelineCompleted(index + 2);
      }
      setPipelineStep("Complete");
      setBackgroundStatus(`Indexed ${describeScanChanges(scanPayload)} · ${formatClock(new Date())}`);
      updateStatus("Background indexing complete");
    } catch (error) {
      setPipelineStep(cancelRequestedRef.current ? "Cancelled" : "Stopped");
      setBackgroundStatus(
        cancelRequestedRef.current ? "Background indexing cancelled" : "Background indexing failed",
      );
      writeRaw(error);
      updateStatus(
        cancelRequestedRef.current ? "Background indexing cancelled" : "Background indexing failed",
        !cancelRequestedRef.current,
      );
    } finally {
      backgroundJobRunningRef.current = false;
      setPipelineRunning(false);
      setPipelineMode(null);
      setCanceling(false);
    }
  }

  async function runFullIndexing() {
    persistSettings(settings);
    cancelRequestedRef.current = false;
    setCanceling(false);
    setPipelineMode("manual");
    setPipelineRunning(true);
    setPipelineCompleted(0);
    setPipelineSummaries([]);
    setResults([]);
    try {
      for (const [index, step] of indexSteps.entries()) {
        setPipelineStep(step.label);
        const payload = await runJsonCommand<IndexCommandPayload>(step.args());
        setPipelineSummaries((current) => [...current, summarizeIndexStep(step, payload)]);
        setPipelineCompleted(index + 1);
        if (step.id === "scan" && payload.root_missing) {
          const message =
            (payload.missing ?? 0) > 0
              ? `Music folder not found · ${describeScanChanges(payload)} marked missing`
              : "Music folder not found · no active tracks to mark missing";
          setPipelineStep("Music folder not found");
          setBackgroundStatus(`${message} · ${formatClock(new Date())}`);
          updateStatus("Music folder not found; update the folder path in Settings", true);
          return;
        }
      }
      setPipelineStep("Complete");
      updateStatus("Indexing complete");
    } catch (error) {
      setPipelineStep(cancelRequestedRef.current ? "Cancelled" : "Stopped");
      writeRaw(error);
      updateStatus(cancelRequestedRef.current ? "Indexing cancelled" : "Indexing failed", !cancelRequestedRef.current);
    } finally {
      setPipelineRunning(false);
      setPipelineMode(null);
      setCanceling(false);
    }
  }

  async function runAdvancedStep(step: IndexStep) {
    persistSettings(settings);
    cancelRequestedRef.current = false;
    setCanceling(false);
    setPipelineMode("manual");
    setPipelineRunning(true);
    setPipelineStep(step.label);
    setPipelineCompleted(0);
    setPipelineSummaries([]);
    try {
      const payload = await runJsonCommand<IndexCommandPayload>(step.args());
      setPipelineSummaries([summarizeIndexStep(step, payload)]);
      setPipelineCompleted(indexSteps.length);
      setPipelineStep("Complete");
    } catch (error) {
      setPipelineStep(cancelRequestedRef.current ? "Cancelled" : "Stopped");
      writeRaw(error);
      updateStatus(cancelRequestedRef.current ? "Indexing cancelled" : "Indexing failed", !cancelRequestedRef.current);
    } finally {
      setPipelineRunning(false);
      setPipelineMode(null);
      setCanceling(false);
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

  async function pruneAllMissing() {
    const confirmed = window.confirm(
      "Prune all tracks marked missing from the database? This does not delete music files.",
    );
    if (!confirmed) {
      return;
    }
    await runJsonCommand(["prune-missing", "--all", "--json"]);
    updateStatus("Pruned missing track rows");
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

  async function playTrack(result: SearchResult) {
    const title = result.title || result.path.split(/[\\/]/).pop() || "track";
    try {
      setPlayerError("");
      setPlayerUsingFallback(false);
      setPlayerTrack(result);
      setPlayerSrc(convertFileSrc(result.path));
      updateStatus(`Playing ${title}`);
    } catch (error) {
      writeRaw(error);
      updateStatus("Could not prepare track for playback", true);
    }
  }

  function playAdjacentTrack(direction: -1 | 1) {
    if (!playerTrack || results.length === 0) {
      return;
    }
    const currentIndex = results.findIndex((result) => result.track_id === playerTrack.track_id);
    if (currentIndex < 0) {
      return;
    }
    const nextIndex = currentIndex + direction;
    const nextTrack = results[nextIndex];
    if (nextTrack) {
      void playTrack(nextTrack);
    }
  }

  async function revealTrack(result: SearchResult) {
    try {
      await invoke<void>("reveal_track", { path: result.path });
      updateStatus("Revealed track in file manager");
    } catch (error) {
      writeRaw(error);
      updateStatus("Could not reveal track", true);
    }
  }

  async function preparePlaybackFallback(result: SearchResult, reason: string) {
    if (playerUsingFallback) {
      setPlayerError(reason);
      return;
    }
    try {
      setPlayerError("Preparing playable preview with ffmpeg…");
      const fallback = await invoke<PlaybackSource>("prepare_playback_fallback", { path: result.path });
      setPlayerUsingFallback(true);
      setPlayerSrc(convertFileSrc(fallback.path));
      setPlayerError("");
      updateStatus("Playing transcoded preview");
    } catch (error) {
      const message = `${reason}; fallback failed: ${formatValue(error)}`;
      setPlayerError(message);
      writeRaw(error);
      updateStatus("Could not prepare playable preview", true);
    }
  }

  async function openTrackExternally(result: SearchResult) {
    try {
      await invoke<void>("open_track", { path: result.path });
      updateStatus("Opening track in system player");
    } catch (error) {
      writeRaw(error);
      updateStatus("Could not open track externally", true);
    }
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

  async function cancelCurrentCommand() {
    const requestId = currentRequestIdRef.current;
    if (!requestId) {
      return;
    }
    cancelRequestedRef.current = true;
    setCanceling(true);
    updateStatus("Cancelling…");
    try {
      await invoke<boolean>("cancel_musicidx", { requestId });
    } catch (error) {
      updateStatus("Could not cancel command", true);
      writeRaw(error);
      setCanceling(false);
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
    currentRequestIdRef.current = requestId;

    return new Promise<MusicidxOutput>((resolve, reject) => {
      let settled = false;
      const cleanup = () => {
        if (unlisten) {
          unlisten();
          unlisten = null;
        }
        if (currentRequestIdRef.current === requestId) {
          currentRequestIdRef.current = null;
        }
        setCanceling(false);
      };

      listen<MusicidxStreamEvent>("musicidx-output", (event) => {
        const payload = event.payload;
        if (payload.request_id !== requestId || settled) {
          return;
        }

        if (!payload.done) {
          const line = `${payload.line}\n`;
          if (payload.stream === "stdout") {
            stdout = clampRawOutput(`${stdout}${line}`);
            setRawOutput((current) => clampRawOutput(`${current}${line}`));
          } else if (payload.stream === "stderr") {
            stderr = clampRawOutput(`${stderr}${line}`);
            setRawOutput((current) => clampRawOutput(`${current}[stderr] ${line}`));
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
      if (settings.llmModel.trim()) {
        args.push("--llm-model", settings.llmModel.trim());
      }
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
    setRawOutput(clampRawOutput(formatValue(value)));
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
              <label className="flex items-start gap-2 rounded-lg border bg-muted/50 p-3 text-sm">
                <input
                  className="mt-1"
                  type="checkbox"
                  checked={settings.backgroundIndexingEnabled}
                  onChange={(event) =>
                    updateSettings({ backgroundIndexingEnabled: event.target.checked })
                  }
                />
                <span className="grid gap-1">
                  <span className="font-medium">Background auto-indexing</span>
                  <span className="text-muted-foreground">
                    Checks for folder changes every {backgroundIndexIntervalLabel} while the app is open.
                  </span>
                </span>
              </label>
              <Field label="Background check interval">
                <select
                  className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  value={backgroundIndexIntervalMinutes}
                  disabled={!settings.backgroundIndexingEnabled}
                  onChange={(event) =>
                    updateSettings({
                      backgroundIndexIntervalMinutes: Number(event.target.value),
                    })
                  }
                >
                  {BACKGROUND_INDEX_INTERVAL_OPTIONS_MINUTES.map((minutes) => (
                    <option key={minutes} value={minutes}>
                      Every {formatMinutes(minutes)}
                    </option>
                  ))}
                </select>
              </Field>
              <div className="rounded-lg border bg-muted/50 p-3 text-sm text-muted-foreground">
                {backgroundStatus}
              </div>
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
                  <Button variant="outline" disabled={busy} onClick={() => runJsonCommand(["failed", "--json"])}>
                    <Activity className="h-4 w-4" />
                    Failed tracks
                  </Button>
                  <Button variant="outline" disabled={busy} onClick={() => runJsonCommand(["missing", "--json"])}>
                    <FolderOpen className="h-4 w-4" />
                    Missing tracks
                  </Button>
                  <Button
                    variant="outline"
                    disabled={busy}
                    onClick={() => runJsonCommand(["retry-failed", "--all", "--json"])}
                  >
                    <CheckCircle2 className="h-4 w-4" />
                    Retry all failed
                  </Button>
                  <Button variant="outline" disabled={busy} onClick={pruneAllMissing}>
                    <Trash2 className="h-4 w-4" />
                    Prune all missing
                  </Button>
                </div>
              ) : null}
            </section>

            <div className="grid gap-4 md:grid-cols-2">
              <Field label="Theme">
                <select
                  className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  value={settings.themeMode}
                  onChange={(event) =>
                    updateSettings({ themeMode: event.target.value as ThemeMode })
                  }
                >
                  <option value="system">System ({systemTheme})</option>
                  <option value="dark">Dark</option>
                  <option value="light">Light</option>
                </select>
              </Field>
              <Field label="Manual indexing type">
                <select
                  className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  value={settings.indexAnalysisMode}
                  onChange={(event) =>
                    updateSettings({ indexAnalysisMode: event.target.value as IndexAnalysisMode })
                  }
                >
                  <option value="quality">Full-track: chunked safe</option>
                  <option value="full">Full-track: larger chunks</option>
                </select>
              </Field>
              <Field label="Manual indexing resource profile">
                <select
                  className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  value={settings.indexResourceProfile}
                  onChange={(event) => updateSettings({ indexResourceProfile: event.target.value })}
                >
                  <option value="auto">Auto</option>
                  <option value="low">Low impact</option>
                  <option value="balanced">Balanced</option>
                  <option value="full">Full</option>
                </select>
              </Field>
              <Field label="Background indexing resource profile">
                <select
                  className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  value={settings.backgroundIndexResourceProfile}
                  disabled={!settings.backgroundIndexingEnabled}
                  onChange={(event) =>
                    updateSettings({ backgroundIndexResourceProfile: event.target.value })
                  }
                >
                  <option value="auto">Auto</option>
                  <option value="low">Low impact</option>
                  <option value="balanced">Balanced</option>
                  <option value="full">Full</option>
                </select>
              </Field>
              <div className="rounded-lg border bg-muted p-3 text-sm text-muted-foreground md:col-span-2">
                MusicIdx now avoids quick sampling by default. Manual and background indexing analyze the whole file in chunks; larger chunks trade memory for fewer decoder passes.
              </div>
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
              <Field label="LLM model">
                <Input
                  value={settings.llmModel}
                  placeholder="gemini-2.0-flash"
                  onChange={(event) => updateSettings({ llmModel: event.target.value })}
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

            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <Button onClick={saveSettings}>Save settings</Button>
              <Button variant="outline" onClick={() => runJsonCommand(["doctor", "--json"])}>
                Run doctor
              </Button>
              {settingsSaveStatus ? (
                <span className="text-sm text-muted-foreground">{settingsSaveStatus}</span>
              ) : null}
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
            <div className="rounded-lg border bg-muted/50 px-3 py-2 text-xs text-muted-foreground wrap-anywhere">
              {backgroundStatus}
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
                  onPlay={playTrack}
                  onReveal={revealTrack}
                  playingTrackId={playerTrack?.track_id ?? null}
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

      {playerTrack ? (
        <div className="fixed bottom-20 right-4 z-40 w-[min(calc(100vw-2rem),28rem)] rounded-xl border bg-card/90 p-4 shadow-xl backdrop-blur-md">
          <div className="mb-3 flex items-start justify-between gap-3">
            <div className="min-w-0">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Now playing</p>
              <p className="wrap-anywhere text-sm font-semibold">
                {playerTrack.title || playerTrack.path.split(/[\\/]/).pop() || playerTrack.track_id}
              </p>
              <p className="wrap-anywhere text-xs text-muted-foreground">
                {[playerTrack.artist, playerTrack.album].filter(Boolean).join(" · ") || playerTrack.path}
              </p>
            </div>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                audioRef.current?.pause();
                setPlayerTrack(null);
                setPlayerSrc("");
                setPlayerError("");
                setPlayerUsingFallback(false);
              }}
            >
              Close
            </Button>
          </div>
          <div className="grid gap-2">
            <audio
              key={playerSrc}
              ref={audioRef}
              controls
              preload="metadata"
              className="w-full"
              onEnded={() => playAdjacentTrack(1)}
              onError={() => {
                const detail = describeAudioError(audioRef.current);
                if (playerTrack) {
                  void preparePlaybackFallback(playerTrack, detail);
                } else {
                  setPlayerError(detail);
                  updateStatus(detail, true);
                }
              }}
            >
              <source src={playerSrc} type={audioMimeType(playerTrack.path)} />
            </audio>
            {playerError ? (
              <div className="grid gap-2 rounded-lg border border-destructive/30 bg-destructive/10 p-2 text-xs text-destructive">
                <p className="wrap-anywhere">{playerError}</p>
                <Button size="sm" variant="outline" onClick={() => void openTrackExternally(playerTrack)}>
                  Open externally
                </Button>
              </div>
            ) : null}
            <div className="flex items-center justify-between gap-2">
              <Button
                size="sm"
                variant="outline"
                disabled={!hasAdjacentTrack(results, playerTrack, -1)}
                onClick={() => playAdjacentTrack(-1)}
              >
                <SkipBack className="h-3.5 w-3.5" />
                Previous
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={!hasAdjacentTrack(results, playerTrack, 1)}
                onClick={() => playAdjacentTrack(1)}
              >
                Next
                <SkipForward className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>
        </div>
      ) : null}

      {pipelineRunning && (
        <div className="fixed bottom-20 left-4 z-40 w-[min(calc(100vw-2rem),20rem)] rounded-xl border bg-card/85 p-4 shadow-xl backdrop-blur-md">
          <div className="mb-2 flex items-center justify-between gap-3 text-sm">
            <span className="font-medium">
              {pipelineMode === "background" ? "Background indexing" : "Indexing"}
            </span>
            <span className="tabular-nums text-muted-foreground">{pipelinePercent}%</span>
          </div>
          <Progress value={pipelinePercent} />
          <div className="mt-2 flex items-center justify-between gap-3">
            <p className="min-w-0 wrap-anywhere text-xs text-muted-foreground">{pipelineStep}</p>
            <Button size="sm" variant="outline" disabled={canceling} onClick={cancelCurrentCommand}>
              {canceling ? "Cancelling…" : "Cancel"}
            </Button>
          </div>
          {pipelineSummaries.length > 0 ? (
            <div className="mt-3 max-h-32 space-y-1 overflow-y-auto rounded-lg bg-muted/60 p-2 text-xs">
              {pipelineSummaries.map((summary) => (
                <div
                  key={summary.id}
                  className={cn(
                    "grid gap-0.5 wrap-anywhere",
                    summary.hasErrors ? "text-destructive" : "text-muted-foreground",
                  )}
                >
                  <span className="font-medium text-foreground">{summary.label}</span>
                  <span>{summary.text}</span>
                </div>
              ))}
            </div>
          ) : null}
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
    <div className="min-h-screen bg-background text-foreground [background:radial-gradient(circle_at_20%_0%,rgba(168,85,247,0.16),transparent_28rem),radial-gradient(circle_at_90%_10%,rgba(217,70,239,0.08),transparent_24rem),hsl(var(--background))]">
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
        "group fixed bottom-5 z-50 flex h-11 w-11 items-center justify-center",
        side === "left" ? "left-4" : "right-4",
      )}
    >
      <div
        role="tooltip"
        className={cn(
          "pointer-events-none absolute bottom-[calc(100%+0.6rem)] max-w-[calc(100vw-2rem)] whitespace-nowrap rounded-md border bg-card/90 px-3 py-1.5 text-xs text-card-foreground opacity-0 shadow-lg backdrop-blur-md transition-all duration-150 group-hover:-translate-y-0.5 group-hover:opacity-100 group-focus-within:-translate-y-0.5 group-focus-within:opacity-100",
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
        title={label}
        className="h-11 w-11 rounded-full border border-primary/25 bg-primary/70 text-primary-foreground shadow-xl shadow-primary/15 backdrop-blur-md hover:bg-primary/85"
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
  onPlay,
  onReveal,
  playingTrackId,
}: {
  result: SearchResult;
  index: number;
  busy: boolean;
  onFeedback: (result: SearchResult, rating: "good" | "bad" | "neutral") => Promise<void>;
  onPlay: (result: SearchResult) => Promise<void>;
  onReveal: (result: SearchResult) => Promise<void>;
  playingTrackId: string | null;
}) {
  const [savedRating, setSavedRating] = useState<"good" | "bad" | "neutral" | null>(
    result.saved_feedback_rating ?? null,
  );
  const [feedbackPending, setFeedbackPending] = useState<"good" | "bad" | "neutral" | null>(null);
  const [feedbackError, setFeedbackError] = useState("");
  const [expanded, setExpanded] = useState(false);
  const title = result.title || result.path.split(/[\\/]/).pop() || result.track_id;
  const meta = [result.artist, result.album, result.genre].filter(Boolean).join(" · ");
  const why = result.why?.length ? result.why.join("; ") : "No explanation available.";
  const tagText = result.matched_tags?.length
    ? result.matched_tags.map((tag) => `${tag.tag} ${tag.score.toFixed(2)}`).join(", ")
    : "";
  const hasLongContent =
    why.length > 180 || tagText.length > 140 || result.path.length > 120 || title.length > 80;
  const isPlaying = playingTrackId === result.track_id;

  useEffect(() => {
    setSavedRating(result.saved_feedback_rating ?? null);
    setFeedbackError("");
    setFeedbackPending(null);
  }, [result.track_id, result.saved_feedback_rating]);

  async function rate(rating: "good" | "bad" | "neutral") {
    setFeedbackPending(rating);
    setFeedbackError("");
    try {
      await onFeedback(result, rating);
      setSavedRating(rating);
    } catch (error) {
      setFeedbackError(formatValue(error));
    } finally {
      setFeedbackPending(null);
    }
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
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap gap-2">
          <Button size="sm" variant={isPlaying ? "default" : "secondary"} onClick={() => void onPlay(result)}>
            <Play className="h-3.5 w-3.5" />
            {isPlaying ? "Playing" : "Play"}
          </Button>
          <Button size="sm" variant="outline" onClick={() => void onReveal(result)}>
            <FolderOpen className="h-3.5 w-3.5" />
            Show
          </Button>
        </div>
        <div className="grid gap-1 sm:justify-items-end">
          <div className="flex flex-wrap gap-2 sm:justify-end">
            <Button
              size="sm"
              variant={savedRating === "good" ? "default" : "outline"}
              disabled={busy || feedbackPending !== null || savedRating === "good"}
              onClick={() => rate("good")}
            >
              {feedbackPending === "good" ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : savedRating === "good" ? (
                <CheckCircle2 className="h-3.5 w-3.5" />
              ) : (
                <ThumbsUp className="h-3.5 w-3.5" />
              )}
              Good
            </Button>
            <Button
              size="sm"
              variant={savedRating === "bad" ? "destructive" : "outline"}
              disabled={busy || feedbackPending !== null || savedRating === "bad"}
              onClick={() => rate("bad")}
            >
              {feedbackPending === "bad" ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <ThumbsDown className="h-3.5 w-3.5" />
              )}
              Bad
            </Button>
            <Button
              size="sm"
              variant={savedRating === "neutral" ? "secondary" : "ghost"}
              disabled={busy || feedbackPending !== null || savedRating === "neutral"}
              onClick={() => rate("neutral")}
            >
              {feedbackPending === "neutral" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
              Neutral
            </Button>
          </div>
          {savedRating ? (
            <p className="text-xs text-muted-foreground">Judged: {savedRating}</p>
          ) : null}
          {feedbackError ? (
            <p className="max-w-xs wrap-anywhere text-xs text-destructive">{feedbackError}</p>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function summarizeIndexStep(step: IndexStep, payload: IndexCommandPayload): PipelineSummary {
  const counts = [
    formatCount("added", payload.added),
    formatCount("modified", payload.modified),
    formatCount("missing", payload.missing),
    formatCount("processed", payload.processed),
    formatCount("updated", payload.updated),
    formatCount("skipped", payload.skipped),
    formatCount("errors", payload.errors),
    formatCount("batches", payload.batches),
  ].filter(Boolean);
  const runtime = [
    payload.duration_sec !== undefined ? `${payload.duration_sec}s` : null,
    payload.peak_memory_mb ? `${payload.peak_memory_mb}MB peak` : null,
    payload.child_peak_memory_mb ? `${payload.child_peak_memory_mb}MB child peak` : null,
    payload.workers ? `${payload.workers} worker${payload.workers === 1 ? "" : "s"}` : null,
    payload.quick ? "first 120s sample" : null,
    payload.chunked
      ? `${payload.quick ? "sample" : "full track"} chunks ${payload.chunk_sec ?? "auto"}s`
      : null,
    payload.batch_size ? `batch ${payload.batch_size}` : null,
  ].filter(Boolean);
  return {
    id: `${step.id}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    label: step.label,
    text: [...counts, ...runtime].join(" · ") || "complete",
    hasErrors: (payload.errors ?? 0) > 0,
  };
}

function formatCount(label: string, value: number | undefined): string | null {
  return value === undefined ? null : `${value} ${label}`;
}

function basicAnalysisArgs(mode: IndexAnalysisMode, resourceProfile: string): string[] {
  const args = ["analyze-basic"];
  args.push("--chunked", "--chunk-sec", mode === "full" ? "300" : "auto");
  args.push("--workers", "auto", "--resource-profile", resourceProfile, "--json");
  return args;
}

function scanChangeCount(payload: IndexCommandPayload): number {
  return (payload.added ?? 0) + (payload.modified ?? 0) + (payload.missing ?? 0);
}

function scanNeedsDerivedIndexing(payload: IndexCommandPayload): boolean {
  return (payload.added ?? 0) > 0 || (payload.modified ?? 0) > 0;
}

function describeScanChanges(payload: IndexCommandPayload): string {
  const parts = [
    formatCount("added", payload.added),
    formatCount("modified", payload.modified),
    formatCount("missing", payload.missing),
  ].filter((part): part is string => Boolean(part && !part.startsWith("0 ")));
  return parts.length ? parts.join(", ") : "changes";
}

function formatClock(date: Date): string {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function envOverrides(settings: SettingsState): Record<string, string> {
  const env: Record<string, string> = {};
  setEnv(env, "MUSICIDX_DB_PATH", settings.dbPath);
  setEnv(env, "MUSICIDX_MODELS_PATH", settings.modelsPath);
  setEnv(env, "MUSICIDX_FFPROBE_PATH", settings.ffprobePath);
  setEnv(env, "MUSICIDX_FPCALC_PATH", settings.fpcalcPath);
  setEnv(env, "MUSICIDX_GEMINI_MODEL", settings.llmModel);
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
    const loaded = { ...defaultSettings, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") };
    return {
      ...loaded,
      indexAnalysisMode: normalizeIndexAnalysisMode(loaded.indexAnalysisMode),
      backgroundIndexIntervalMinutes: normalizeBackgroundIndexIntervalMinutes(
        loaded.backgroundIndexIntervalMinutes,
      ),
    };
  } catch {
    return defaultSettings;
  }
}

function normalizeBackgroundIndexIntervalMinutes(value: unknown): number {
  const minutes = Number(value);
  if ((BACKGROUND_INDEX_INTERVAL_OPTIONS_MINUTES as readonly number[]).includes(minutes)) {
    return minutes;
  }
  return DEFAULT_BACKGROUND_INDEX_INTERVAL_MINUTES;
}

function normalizeIndexAnalysisMode(value: unknown): IndexAnalysisMode {
  if (value === "full") {
    return "full";
  }
  return "quality";
}

function formatMinutes(minutes: number): string {
  return `${minutes} minute${minutes === 1 ? "" : "s"}`;
}

function hasAdjacentTrack(results: SearchResult[], current: SearchResult, direction: -1 | 1): boolean {
  const index = results.findIndex((result) => result.track_id === current.track_id);
  return index >= 0 && index + direction >= 0 && index + direction < results.length;
}

function audioMimeType(path: string): string {
  const extension = path.split(".").pop()?.toLowerCase();
  if (extension === "mp3") return "audio/mpeg";
  if (extension === "m4a" || extension === "aac") return "audio/mp4";
  if (extension === "wav") return "audio/wav";
  if (extension === "flac") return "audio/flac";
  if (extension === "ogg") return "audio/ogg";
  if (extension === "opus") return "audio/opus";
  if (extension === "aif" || extension === "aiff") return "audio/aiff";
  return "audio/mpeg";
}

function describeAudioError(audio: HTMLAudioElement | null): string {
  const code = audio?.error?.code;
  const suffix = audio?.currentSrc ? ` (${audio.currentSrc})` : "";
  if (code === 1) return `Playback aborted${suffix}`;
  if (code === 2) return `Could not load track via local asset protocol${suffix}`;
  if (code === 3) return `Could not decode this audio file in MusicIdx${suffix}`;
  if (code === 4) {
    return `This audio source or codec is not supported by the embedded player${suffix}`;
  }
  return `Could not play this file in MusicIdx${suffix}`;
}

function packagedSemanticModelDefault(current: string, packaged: string | undefined): string {
  const packagedModel = (packaged || "").trim();
  if (!packagedModel) {
    return current;
  }
  const currentModel = current.trim();
  if (!currentModel || currentModel === DEFAULT_SEMANTIC_MODEL) {
    return packagedModel;
  }
  return current;
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

function clampRawOutput(value: string): string {
  if (value.length <= MAX_RAW_OUTPUT_CHARS) {
    return value;
  }
  return `[output truncated to last ${Math.round(MAX_RAW_OUTPUT_CHARS / 1024)}KB]\n${value.slice(
    -MAX_RAW_OUTPUT_CHARS,
  )}`;
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
