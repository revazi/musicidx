import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  Activity,
  AlertTriangle,
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

import { LibraryBrowsePanel } from "./components/LibraryBrowsePanel";
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
import type {
  BrowsePayload,
  MatchEvidenceItem,
  MatchReport,
  MatchTrackPayload,
  SearchFeatureRange,
  SearchIntentSummary,
  SearchLlmHints,
  SearchNotice,
  SearchPayload,
  SearchResult,
  SearchSortSpec,
  SearchSuggestion,
  SearchTypeExample,
} from "./types";

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

type HealthWarning = {
  code: string;
  severity: "error" | "warning" | string;
  message: string;
};

type HealthCoverage = {
  count?: number;
  tracks?: number;
  coverage?: number;
  missing?: number;
};

type IndexHealthPayload = {
  ready: boolean;
  db_path: string;
  models_path: string;
  semantic_model: string;
  tracks: {
    total: number;
    active: number;
    missing: number;
    failed: number;
    quarantined: number;
  };
  audio_features: HealthCoverage;
  derived_tags: HealthCoverage;
  context_fit: HealthCoverage;
  profiles: HealthCoverage & {
    schema_v2?: number;
    with_embedding_text?: number;
  };
  embeddings: HealthCoverage & {
    current?: number;
    stale?: number;
    selected_model?: string | null;
  };
  warnings: HealthWarning[];
  recommended_actions: string[];
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
  debugMode: boolean;
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
  tags_written?: number;
  contexts_written?: number;
  schema_version?: number;
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
  debugMode: false,
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
  const [lastSearchPayload, setLastSearchPayload] = useState<SearchPayload | null>(null);
  const [browsePayload, setBrowsePayload] = useState<BrowsePayload | null>(null);
  const [browseExpanded, setBrowseExpanded] = useState(false);
  const [browseQuery, setBrowseQuery] = useState("");
  const [browseSort, setBrowseSort] = useState("artist");
  const [browseSortDirection, setBrowseSortDirection] = useState<"asc" | "desc">("asc");
  const [matchPayloads, setMatchPayloads] = useState<Record<string, MatchTrackPayload>>({});
  const [matchingTrackId, setMatchingTrackId] = useState<string | null>(null);
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
  const [indexHealth, setIndexHealth] = useState<IndexHealthPayload | null>(null);
  const [indexHealthLoading, setIndexHealthLoading] = useState(false);
  const [canceling, setCanceling] = useState(false);
  const currentRequestIdRef = useRef<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const cancelRequestedRef = useRef(false);
  const backgroundJobRunningRef = useRef(false);
  const busyRef = useRef(false);
  const pipelineRunningRef = useRef(false);

  const indexSteps = useMemo<IndexStep[]>(
    () =>
      buildIndexSteps({
        musicFolder: settings.musicFolder,
        analysisMode: settings.indexAnalysisMode,
        resourceProfile: settings.indexResourceProfile,
        semanticModel: settings.semanticModel,
      }),
    [
      settings.indexAnalysisMode,
      settings.indexResourceProfile,
      settings.musicFolder,
      settings.semanticModel,
    ],
  );

  const backgroundIndexSteps = useMemo<IndexStep[]>(
    () =>
      buildIndexSteps({
        musicFolder: settings.musicFolder,
        analysisMode: "quality",
        resourceProfile: settings.backgroundIndexResourceProfile,
        semanticModel: settings.semanticModel,
      }),
    [settings.backgroundIndexResourceProfile, settings.musicFolder, settings.semanticModel],
  );

  const pipelinePercent = Math.round((pipelineCompleted / indexSteps.length) * 100);
  const effectiveTheme = settings.themeMode === "system" ? systemTheme : settings.themeMode;
  const backgroundIndexIntervalMinutes = normalizeBackgroundIndexIntervalMinutes(
    settings.backgroundIndexIntervalMinutes,
  );
  const backgroundIndexIntervalMs = backgroundIndexIntervalMinutes * 60 * 1000;
  const backgroundIndexIntervalLabel = formatMinutes(backgroundIndexIntervalMinutes);
  const settingsPathWarnings = settingsHealthWarnings(settings);
  const working = busy || pipelineRunning || indexHealthLoading;
  const workingLabel = pipelineRunning
    ? `${pipelineMode === "background" ? "Background indexing" : "Indexing"}: ${pipelineStep}`
    : indexHealthLoading
      ? "Checking index health"
      : busy
        ? status
        : "";

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
    setMatchPayloads({});
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

  async function searchMusic(queryOverride?: string) {
    persistSettings(settings);
    const searchQuery = queryOverride ?? query;
    if (queryOverride !== undefined) {
      setQuery(queryOverride);
    }
    const args = [
      "search",
      required(searchQuery, "Query"),
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
    setLastSearchPayload(payload);
    setMatchPayloads({});
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

  async function loadBrowse(
    path?: string | null,
    options: BrowseLoadOptions = {},
  ) {
    persistSettings(settings);
    const args = buildBrowseArgs(
      resolveBrowseLoadOptions(path, options, {
        browseQuery,
        browseSort,
        browseSortDirection,
        musicFolder: settings.musicFolder,
      }),
    );
    const payload = await runJsonCommand<BrowsePayload>(args);
    setBrowsePayload(payload);
    setBrowseExpanded(true);
    updateStatus(browseStatusMessage(payload), false);
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

  async function checkIndexHealth() {
    const args = ["index-health", "--json"];
    if (settings.semanticModel.trim()) {
      args.push("--semantic-model", settings.semanticModel.trim());
    }
    setIndexHealthLoading(true);
    try {
      const payload = await runJsonCommand<IndexHealthPayload>(args);
      setIndexHealth(payload);
      const hasError = payload.warnings.some((warning) => warning.severity === "error");
      updateStatus(payload.ready ? "Index health ready" : "Index health needs attention", hasError);
    } finally {
      setIndexHealthLoading(false);
    }
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

  async function findMatches(result: SearchResult) {
    if (matchPayloads[result.track_id]) {
      setMatchPayloads((current) => {
        const next = { ...current };
        delete next[result.track_id];
        return next;
      });
      return;
    }
    const args = [
      "match-track",
      "--track-id",
      result.track_id,
      "--against-library",
      "--limit",
      "5",
      "--json",
    ];
    setMatchingTrackId(result.track_id);
    try {
      const payload = await runJsonCommand<MatchTrackPayload>(args);
      setMatchPayloads((current) => ({ ...current, [result.track_id]: payload }));
      updateStatus(
        payload.count
          ? `Found ${payload.count} closest candidate${payload.count === 1 ? "" : "s"}`
          : "No nearby candidates found",
      );
    } finally {
      setMatchingTrackId(null);
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
    cancelRequestedRef.current = true;
    setCanceling(true);
    updateStatus("Cancelling…");
    const requestId = currentRequestIdRef.current;
    if (!requestId) {
      return;
    }
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
        working={working}
        workingLabel={workingLabel}
        canceling={canceling}
        onCancel={cancelCurrentCommand}
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
              <div className="space-y-3 rounded-lg border bg-background/60 p-3 text-sm">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                  <div>
                    <h4 className="font-medium">Index health</h4>
                    <p className="text-muted-foreground">
                      Check DB readiness, profile schema, context-fit coverage, and embedding freshness.
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge
                      variant={
                        indexHealth
                          ? indexHealth.ready
                            ? "default"
                            : indexHealth.warnings.some((warning) => warning.severity === "error")
                              ? "destructive"
                              : "secondary"
                          : "secondary"
                      }
                    >
                      {indexHealth ? (indexHealth.ready ? "Ready" : "Needs attention") : "Not checked"}
                    </Badge>
                    <Button variant="outline" size="sm" disabled={busy || indexHealthLoading} onClick={checkIndexHealth}>
                      {indexHealthLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Activity className="h-3.5 w-3.5" />}
                      Check index health
                    </Button>
                  </div>
                </div>
                {settingsPathWarnings.length > 0 ? (
                  <div className="grid gap-1 rounded-md border border-destructive/30 bg-destructive/10 p-2 text-destructive">
                    {settingsPathWarnings.map((warning) => (
                      <p key={warning} className="flex gap-2">
                        <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                        <span>{warning}</span>
                      </p>
                    ))}
                  </div>
                ) : null}
                {indexHealth ? <IndexHealthCard health={indexHealth} /> : null}
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
                  <Button variant="outline" disabled={busy || indexHealthLoading} onClick={checkIndexHealth}>
                    <Activity className="h-4 w-4" />
                    Index health
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
              <label className="flex items-start gap-2 rounded-lg border bg-muted/50 p-3 text-sm">
                <input
                  className="mt-1"
                  type="checkbox"
                  checked={settings.debugMode}
                  onChange={(event) => updateSettings({ debugMode: event.target.checked })}
                />
                <span className="grid gap-1">
                  <span className="font-medium">Debug mode</span>
                  <span className="text-muted-foreground">
                    Show raw output, search parameters, rank/evidence details, and full match evidence.
                  </span>
                </span>
              </label>
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
      working={working}
      workingLabel={workingLabel}
      canceling={canceling}
      onCancel={cancelCurrentCommand}
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
                  if (event.key === "Enter" && !busy) {
                    void searchMusic();
                  }
                }}
              />
              <Button className="rounded-full px-5" disabled={busy} onClick={() => void searchMusic()}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
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
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                Parse intent
              </Button>
            </div>
            {lastSearchPayload ? (
              <SearchParametersPanel
                payload={lastSearchPayload}
                debugMode={settings.debugMode}
                onSearchSuggestion={searchMusic}
              />
            ) : null}
            <LibraryBrowsePanel
              payload={browsePayload}
              expanded={browseExpanded}
              busy={busy}
              matchPayloads={matchPayloads}
              matchingTrackId={matchingTrackId}
              debugMode={settings.debugMode}
              query={browseQuery}
              sort={browseSort}
              sortDirection={browseSortDirection}
              onQueryChange={setBrowseQuery}
              onSortChange={setBrowseSort}
              onSortDirectionChange={setBrowseSortDirection}
              onToggle={() => setBrowseExpanded((value) => !value)}
              onLoad={() => void loadBrowse(browsePayload?.cwd ?? (settings.musicFolder || null), { offset: 0 })}
              onOpenPath={(path) => void loadBrowse(path, { offset: 0 })}
              onSubmitSearch={() => void loadBrowse(browsePayload?.cwd ?? (settings.musicFolder || null), { offset: 0 })}
              onClearSearch={() => {
                setBrowseQuery("");
                void loadBrowse(browsePayload?.cwd ?? (settings.musicFolder || null), { query: "", offset: 0 });
              }}
              onPage={(offset) => void loadBrowse(browsePayload?.cwd ?? (settings.musicFolder || null), { offset })}
              onPlay={(track) => void playTrack(track)}
              onReveal={(track) => void revealTrack(track)}
              onMatch={(track) => void findMatches(track)}
              renderMatches={(payload) => (
                <MatchSummaryPanel
                  payload={payload}
                  debugMode={settings.debugMode}
                  onPlayCandidate={(report) => {
                    const candidate = searchResultFromMatchReport(report);
                    if (candidate) {
                      void playTrack(candidate);
                    }
                  }}
                  onRevealCandidate={(report) => {
                    const candidate = searchResultFromMatchReport(report);
                    if (candidate) {
                      void revealTrack(candidate);
                    }
                  }}
                />
              )}
            />
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
                  onMatch={findMatches}
                  matchPayload={matchPayloads[result.track_id] ?? null}
                  matchLoading={matchingTrackId === result.track_id}
                  debugMode={settings.debugMode}
                  playingTrackId={playerTrack?.track_id ?? null}
                />
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {settings.debugMode ? (
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
      ) : null}

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

type ShellProps = {
  children: React.ReactNode;
  status: string;
  statusError: boolean;
  working: boolean;
  workingLabel: string;
  canceling: boolean;
  onCancel: () => void;
  onDoctor: () => void;
  onSettings: () => void;
};

function Shell(props: ShellProps) {
  return (
    <div className="min-h-screen bg-background text-foreground [background:radial-gradient(circle_at_20%_0%,rgba(168,85,247,0.16),transparent_28rem),radial-gradient(circle_at_90%_10%,rgba(217,70,239,0.08),transparent_24rem),hsl(var(--background))]">
      <div className="mx-auto grid w-full max-w-6xl gap-4 px-4 py-4 sm:px-6 lg:px-8">
        <ShellHeader {...props} />
        <WorkingBanner {...props} />
        {props.children}
      </div>
    </div>
  );
}

function ShellHeader({ status, statusError, working, onDoctor, onSettings }: ShellProps) {
  return (
    <header className="flex flex-col gap-4 rounded-xl border bg-card p-5 shadow-sm md:flex-row md:items-center md:justify-between">
      <ShellTitle />
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={statusBadgeVariant(statusError)} className="max-w-full wrap-anywhere">
          {status}
        </Badge>
        <Button variant="outline" disabled={working} onClick={onDoctor}>
          <DoctorIcon working={working} />
          Doctor
        </Button>
        <Button variant="outline" size="icon" onClick={onSettings} aria-label="Settings">
          <Settings className="h-4 w-4" />
        </Button>
      </div>
    </header>
  );
}

function ShellTitle() {
  return (
    <div className="min-w-0">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
        Local-first music search
      </p>
      <h1 className="mt-1 text-3xl font-semibold tracking-tight">MusicIdx</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Index, search, judge, and export your local music library.
      </p>
    </div>
  );
}

function statusBadgeVariant(statusError: boolean): "destructive" | "secondary" {
  return statusError ? "destructive" : "secondary";
}

function DoctorIcon({ working }: { working: boolean }) {
  return working ? <Loader2 className="h-4 w-4 animate-spin" /> : <Activity className="h-4 w-4" />;
}

function WorkingBanner({ working, workingLabel, status, canceling, onCancel }: ShellProps) {
  if (!working) {
    return null;
  }
  return (
    <div className="sticky top-3 z-50 rounded-xl border bg-card/95 p-3 shadow-lg backdrop-blur-md">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex min-w-0 items-center gap-2 text-sm">
          <Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary" />
          <span className="font-medium text-foreground">Working</span>
          <span className="wrap-anywhere text-muted-foreground">{workingLabel || status}</span>
        </div>
        <Button size="sm" variant="outline" disabled={canceling} onClick={onCancel}>
          {cancelButtonText(canceling)}
        </Button>
      </div>
    </div>
  );
}

function cancelButtonText(canceling: boolean): string {
  return canceling ? "Cancelling…" : "Cancel";
}

type EvidenceCountStages = ReadonlyArray<readonly [string, Record<string, unknown>]>;

type SearchPanelModel = {
  intent: SearchIntentSummary;
  diagnostics: Record<string, unknown>;
  evidenceCountStages: EvidenceCountStages;
  suggestions: SearchSuggestion[];
  examples: SearchTypeExample[];
  notice: SearchNotice | null;
  noticeText: string;
  visible: boolean;
};

function SearchParametersPanel({
  payload,
  debugMode,
  onSearchSuggestion,
}: {
  payload: SearchPayload;
  debugMode: boolean;
  onSearchSuggestion: (query: string) => Promise<void>;
}) {
  const model = searchPanelModel(payload, debugMode);
  if (!model.visible) {
    return null;
  }
  return (
    <div className="grid gap-3 rounded-lg border bg-muted/40 p-3 text-xs text-muted-foreground">
      <SearchDebugBadges payload={payload} intent={model.intent} debugMode={debugMode} />
      <SearchLlmFallback payload={payload} debugMode={debugMode} />
      <SearchNoticeMessage notice={model.notice} noticeText={model.noticeText} />
      <SearchSuggestionButtons suggestions={model.suggestions} onSearchSuggestion={onSearchSuggestion} />
      <SearchTypeExamplesPanel examples={model.examples} onSearchSuggestion={onSearchSuggestion} />
      <SearchLlmHintsPanel hints={payload.llm_hints} debugMode={debugMode} />
      <SearchDebugIntentGrid
        payload={payload}
        intent={model.intent}
        diagnostics={model.diagnostics}
        evidenceCountStages={model.evidenceCountStages}
        debugMode={debugMode}
      />
      <SearchEvidenceCountsPanel evidenceCountStages={model.evidenceCountStages} debugMode={debugMode} />
      <SearchRankingWeightsPanel weights={recordValue(model.diagnostics.weights)} debugMode={debugMode} />
    </div>
  );
}

function searchPanelModel(payload: SearchPayload, debugMode: boolean): SearchPanelModel {
  const diagnostics = payload.diagnostics ?? {};
  const suggestions = visibleSearchSuggestions(diagnostics, debugMode);
  const examples = visibleSearchExamples(diagnostics, debugMode);
  const notice = searchNoticeFromDiagnostics(diagnostics);
  const noticeText = noticeTextForPanel(notice, debugMode);
  return {
    intent: payload.intent ?? {},
    diagnostics,
    evidenceCountStages: searchEvidenceCountStages(diagnostics),
    suggestions,
    examples,
    notice,
    noticeText,
    visible: panelHasVisibleContent(debugMode, noticeText, suggestions, examples),
  };
}

function visibleSearchSuggestions(
  diagnostics: Record<string, unknown>,
  debugMode: boolean,
): SearchSuggestion[] {
  const suggestions = searchSuggestionsFromDiagnostics(diagnostics);
  return debugMode ? suggestions : compactSearchSuggestions(suggestions);
}

function visibleSearchExamples(
  diagnostics: Record<string, unknown>,
  debugMode: boolean,
): SearchTypeExample[] {
  const examples = searchTypeExamplesFromDiagnostics(diagnostics);
  return debugMode ? examples : compactSearchTypeExamples(examples);
}

function noticeTextForPanel(notice: SearchNotice | null, debugMode: boolean): string {
  return notice ? searchNoticeDisplayText(notice, debugMode) : "";
}

function panelHasVisibleContent(
  debugMode: boolean,
  noticeText: string,
  suggestions: SearchSuggestion[],
  examples: SearchTypeExample[],
): boolean {
  return debugMode || Boolean(noticeText || suggestions.length || examples.length);
}

function SearchDebugBadges({
  payload,
  intent,
  debugMode,
}: {
  payload: SearchPayload;
  intent: SearchIntentSummary;
  debugMode: boolean;
}) {
  if (!debugMode) {
    return null;
  }
  const badges = searchDebugBadgeModel(payload, intent);
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="font-medium text-foreground">Search parameters used</span>
      <Badge variant="secondary">parser: {badges.parser}</Badge>
      <Badge variant={badges.parserVariant}>{badges.parserMode}</Badge>
      <Badge variant="secondary">limit: {badges.limit}</Badge>
      <Badge variant="secondary">{badges.semantic}</Badge>
      <Badge variant="outline">raw calibrated scores</Badge>
    </div>
  );
}

type SearchDebugBadgeModel = {
  parser: string;
  parserMode: string;
  parserVariant: "secondary" | "outline";
  limit: string | number;
  semantic: string;
};

function searchDebugBadgeModel(
  payload: SearchPayload,
  intent: SearchIntentSummary,
): SearchDebugBadgeModel {
  return {
    parser: parserLabel(payload),
    parserMode: searchParserMode(payload),
    parserVariant: parserBadgeVariant(payload),
    limit: intentLimitLabel(intent),
    semantic: semanticSearchLabel(intent),
  };
}

function parserLabel(payload: SearchPayload): string {
  return payload.parser || "dynamic";
}

function parserBadgeVariant(payload: SearchPayload): "secondary" | "outline" {
  return payload.llm_error ? "outline" : "secondary";
}

function intentLimitLabel(intent: SearchIntentSummary): string | number {
  return intent.limit ?? "auto";
}

function semanticSearchLabel(intent: SearchIntentSummary): string {
  return intent.use_semantic ? "semantic search on" : "semantic search off";
}

function searchParserMode(payload: SearchPayload): string {
  if (payload.llm_error) {
    return "local parser fallback";
  }
  return payload.parser && payload.parser !== "dynamic" ? "LLM hints" : "local parser";
}

function SearchLlmFallback({ payload, debugMode }: { payload: SearchPayload; debugMode: boolean }) {
  if (!debugMode || !payload.llm_error) {
    return null;
  }
  return (
    <div className="rounded-md border bg-background/50 p-2 text-muted-foreground">
      LLM hints unavailable or rejected; local parser used. {payload.llm_error.split("\n")[0]}
    </div>
  );
}

function SearchNoticeMessage({
  notice,
  noticeText,
}: {
  notice: SearchNotice | null;
  noticeText: string;
}) {
  if (!noticeText) {
    return null;
  }
  const model = searchNoticeModel(notice);
  return (
    <div className={model.className}>
      <span className="font-medium text-foreground">{model.label}: </span>
      {noticeText}
    </div>
  );
}

type SearchNoticeModel = { className: string; label: string };

function searchNoticeModel(notice: SearchNotice | null): SearchNoticeModel {
  const warning = notice?.level === "warning";
  return {
    className: cn("rounded-md border p-2", searchNoticeClass(warning)),
    label: warning ? "Search note" : "Search tip",
  };
}

function searchNoticeClass(warning: boolean): string {
  return warning
    ? "border-yellow-500/30 bg-yellow-500/10 text-yellow-700 dark:text-yellow-300"
    : "bg-background/50";
}

function SearchSuggestionButtons({
  suggestions,
  onSearchSuggestion,
}: {
  suggestions: SearchSuggestion[];
  onSearchSuggestion: (query: string) => Promise<void>;
}) {
  if (!suggestions.length) {
    return null;
  }
  const hasCorrectionSuggestion = suggestions.some((suggestion) => suggestion.kind === "correction");
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-md border bg-background/50 p-2">
      <span className="font-medium text-foreground">
        {hasCorrectionSuggestion ? "Did you mean / try:" : "Try:"}
      </span>
      {suggestions.map((suggestion) => (
        <Button
          key={`${suggestion.kind || "suggestion"}:${suggestion.query}`}
          size="sm"
          variant={suggestion.kind === "correction" ? "default" : "outline"}
          title={suggestion.reason}
          onClick={() => void onSearchSuggestion(suggestion.query)}
        >
          {suggestion.label ? `${suggestion.label}: ` : ""}{suggestion.query}
          {typeof suggestion.confidence === "number" ? ` ${(suggestion.confidence * 100).toFixed(0)}%` : ""}
        </Button>
      ))}
    </div>
  );
}

function SearchTypeExamplesPanel({
  examples,
  onSearchSuggestion,
}: {
  examples: SearchTypeExample[];
  onSearchSuggestion: (query: string) => Promise<void>;
}) {
  if (!examples.length) {
    return null;
  }
  return (
    <details className="rounded-md border bg-background/50 p-2">
      <summary className="cursor-pointer font-medium text-foreground">Search types / examples</summary>
      <div className="mt-2 flex flex-wrap gap-2">
        {examples.map((example, index) => (
          <SearchTypeExampleItem
            key={`${example.type || "type"}:${example.query || index}`}
            example={example}
            index={index}
            onSearchSuggestion={onSearchSuggestion}
          />
        ))}
      </div>
    </details>
  );
}

function SearchTypeExampleItem({
  example,
  index,
  onSearchSuggestion,
}: {
  example: SearchTypeExample;
  index: number;
  onSearchSuggestion: (query: string) => Promise<void>;
}) {
  const query = example.query;
  if (!query) {
    return <SearchTypeTip example={example} index={index} />;
  }
  return <SearchTypeButton example={example} query={query} onSearchSuggestion={onSearchSuggestion} />;
}

function SearchTypeTip({ example, index }: { example: SearchTypeExample; index: number }) {
  return (
    <span className="rounded-md border bg-muted/50 px-2 py-1" title={example.reason}>
      {searchTypeLabel(example, "Tip")}: {example.reason || index}
    </span>
  );
}

function SearchTypeButton({
  example,
  query,
  onSearchSuggestion,
}: {
  example: SearchTypeExample;
  query: string;
  onSearchSuggestion: (query: string) => Promise<void>;
}) {
  return (
    <Button size="sm" variant="outline" title={example.reason} onClick={() => void onSearchSuggestion(query)}>
      {searchTypeLabel(example, "Example")}: {query}
    </Button>
  );
}

function searchTypeLabel(example: SearchTypeExample, fallback: string): string {
  return example.label || example.type || fallback;
}

function SearchLlmHintsPanel({
  hints,
  debugMode,
}: {
  hints: SearchLlmHints | null | undefined;
  debugMode: boolean;
}) {
  if (!debugMode) {
    return null;
  }
  if (!hints) {
    return null;
  }
  return (
    <details className="rounded-md border bg-background/50 p-2">
      <summary className="cursor-pointer font-medium text-foreground">
        LLM-provided hints before local validation/merge
      </summary>
      <div className="mt-2 grid gap-2 md:grid-cols-2">
        {llmHintBlocks(hints).map((block) => (
          <SearchParamBlock key={block.label} label={block.label} value={block.value} />
        ))}
      </div>
    </details>
  );
}

function llmHintBlocks(hints: SearchLlmHints): SearchDebugIntentValue[] {
  return [
    { label: "LLM contexts", value: chipList(hints.contexts) },
    { label: "LLM prefer concepts", value: chipList(hints.prefer_tag_concepts, 12) },
    { label: "LLM avoid concepts", value: chipList(hints.avoid_tag_concepts, 12) },
    { label: "LLM features", value: formatFeaturePreferences(hints.feature_preferences) },
    { label: "LLM sort", value: formatSortSpecList(hints.sort_by) },
    { label: "LLM notes", value: hints.notes || "none" },
  ];
}

function SearchDebugIntentGrid({
  payload,
  intent,
  diagnostics,
  evidenceCountStages,
  debugMode,
}: {
  payload: SearchPayload;
  intent: SearchIntentSummary;
  diagnostics: Record<string, unknown>;
  evidenceCountStages: EvidenceCountStages;
  debugMode: boolean;
}) {
  if (!debugMode) {
    return null;
  }
  const values = searchDebugIntentValues(payload, intent, diagnostics, evidenceCountStages);
  return (
    <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
      {values.map((item) => (
        <SearchParamBlock key={item.label} label={item.label} value={item.value} />
      ))}
    </div>
  );
}

type SearchDebugIntentValue = { label: string; value: string };

function searchDebugIntentValues(
  payload: SearchPayload,
  intent: SearchIntentSummary,
  diagnostics: Record<string, unknown>,
  evidenceCountStages: EvidenceCountStages,
): SearchDebugIntentValue[] {
  const scoreWarnings = stringList(diagnostics.score_warnings);
  return [
    { label: "Query", value: payload.query || "—" },
    { label: "Semantic model", value: intent.semantic_model || "not indexed/disabled" },
    { label: "Contexts", value: chipList(intent.contexts) },
    { label: "Prefer tags", value: chipList(intent.prefer_tags, 12) },
    { label: "Avoid tags", value: chipList(intent.avoid_tags, 10) },
    { label: "Feature ranges", value: formatFeatureRanges(intent) },
    { label: "Sort", value: formatSortSpecs(intent) },
    { label: "Score calibration", value: formatScoreCalibration(diagnostics) },
    { label: "Diagnostics", value: formatSearchDiagnostics(diagnostics, scoreWarnings) },
    { label: "Evidence counts", value: formatEvidenceCountStages(evidenceCountStages) },
  ];
}

function SearchEvidenceCountsPanel({
  evidenceCountStages,
  debugMode,
}: {
  evidenceCountStages: EvidenceCountStages;
  debugMode: boolean;
}) {
  const hasCounts = evidenceCountStages.some(([, counts]) => Object.keys(counts).length > 0);
  if (!debugMode || !hasCounts) {
    return null;
  }
  return (
    <details className="rounded-md border bg-background/50 p-2">
      <summary className="cursor-pointer font-medium text-foreground">Evidence source counts</summary>
      <div className="mt-2 grid gap-2 md:grid-cols-3">
        {evidenceCountStages.map(([stage, counts]) => (
          <SearchParamBlock key={stage} label={stage} value={formatRecordCounts(counts)} />
        ))}
      </div>
    </details>
  );
}

function SearchRankingWeightsPanel({
  weights,
  debugMode,
}: {
  weights: Record<string, unknown>;
  debugMode: boolean;
}) {
  if (!debugMode || Object.keys(weights).length === 0) {
    return null;
  }
  return (
    <details className="rounded-md border bg-background/50 p-2">
      <summary className="cursor-pointer font-medium text-foreground">Ranking weights</summary>
      <p className="mt-2 wrap-anywhere">{formatWeights(weights)}</p>
    </details>
  );
}

function SearchParamBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-md bg-background/50 p-2">
      <p className="font-medium text-foreground">{label}</p>
      <p className="wrap-anywhere">{value || "—"}</p>
    </div>
  );
}

function MatchSummaryPanel({
  payload,
  debugMode,
  onPlayCandidate,
  onRevealCandidate,
}: {
  payload: MatchTrackPayload;
  debugMode: boolean;
  onPlayCandidate: (report: MatchReport) => void;
  onRevealCandidate: (report: MatchReport) => void;
}) {
  const reports = payload.reports ?? [];
  return (
    <div className="grid gap-2 rounded-lg border bg-muted/40 p-3 text-xs text-muted-foreground">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-foreground">Closest tracks</span>
        <Badge variant="secondary">{payload.count ?? reports.length} candidates</Badge>
        <span>ranked by decoded fingerprint/soundwave → name → duration → metadata → features → audio embedding</span>
      </div>
      {reports.length ? (
        <div className="grid gap-2">
          {reports.slice(0, 5).map((report, index) => (
            <MatchReportRow
              key={`${report.track_b?.track_id || index}`}
              report={report}
              debugMode={debugMode}
              onPlayCandidate={onPlayCandidate}
              onRevealCandidate={onRevealCandidate}
            />
          ))}
        </div>
      ) : (
        <p>No nearby candidates found.</p>
      )}
    </div>
  );
}

function MatchReportRow({
  report,
  debugMode,
  onPlayCandidate,
  onRevealCandidate,
}: {
  report: MatchReport;
  debugMode: boolean;
  onPlayCandidate: (report: MatchReport) => void;
  onRevealCandidate: (report: MatchReport) => void;
}) {
  const candidate = searchResultFromMatchReport(report);
  return (
    <div className="rounded-md border bg-background/50 p-2">
      <MatchReportHeader report={report} />
      <OptionalText className="mt-1 wrap-anywhere text-foreground" text={report.candidate_summary} />
      <OptionalText
        className="mt-1 wrap-anywhere text-muted-foreground"
        text={debugMode ? report.track_b?.path : null}
      />
      <MatchReportActions
        report={report}
        hasCandidate={Boolean(candidate)}
        onPlayCandidate={onPlayCandidate}
        onRevealCandidate={onRevealCandidate}
      />
      <MatchReportDebugDetails report={report} debugMode={debugMode} />
    </div>
  );
}

function MatchReportHeader({ report }: { report: MatchReport }) {
  const model = matchReportHeaderModel(report);
  return (
    <div className="flex min-w-0 items-start justify-between gap-3">
      <div className="min-w-0">
        <p className="wrap-anywhere font-medium text-foreground">{model.title}</p>
        <div className="mt-1 flex flex-wrap gap-2">
          <span>{model.closest}</span>
          <span>{model.identity}</span>
        </div>
      </div>
      <div className="flex shrink-0 flex-wrap justify-end gap-1">
        <Badge variant={model.strengthVariant}>{model.strength}</Badge>
        <Badge variant={model.decisionVariant}>{model.decision}</Badge>
      </div>
    </div>
  );
}

type MatchReportHeaderModel = {
  title: string;
  closest: string;
  identity: string;
  strength: string;
  strengthVariant: "default" | "secondary" | "outline";
  decision: string;
  decisionVariant: "secondary" | "outline";
};

function matchReportHeaderModel(report: MatchReport): MatchReportHeaderModel {
  return {
    title: matchReportTitle(report),
    closest: `closest ${numericText(report.candidate_score)}`,
    identity: `identity ${matchConfidenceLabel(report)} ${numericText(report.confidence_score)}`,
    strength: report.candidate_strength || "weak",
    strengthVariant: matchStrengthVariant(report.candidate_strength),
    decision: report.decision || "unknown",
    decisionVariant: matchDecisionVariant(report),
  };
}

function matchConfidenceLabel(report: MatchReport): string {
  return report.confidence || "low";
}

function matchDecisionVariant(report: MatchReport): "secondary" | "outline" {
  return report.identity_decision === "same" ? "secondary" : "outline";
}

function matchReportTitle(report: MatchReport): string {
  return joinOptionalPrefix(trackArtistPrefix(report.track_b), matchTrackTitle(report.track_b));
}

function trackArtistPrefix(track: MatchReport["track_b"]): string {
  return track?.artist ? `${track.artist} · ` : "";
}

function matchTrackTitle(track: MatchReport["track_b"]): string {
  return trackTitleValue(track) || trackIdValue(track) || "candidate";
}

function trackTitleValue(track: MatchReport["track_b"]): string {
  return track?.title || "";
}

function trackIdValue(track: MatchReport["track_b"]): string {
  return track?.track_id || "";
}

function joinOptionalPrefix(prefix: string, text: string): string {
  return `${prefix}${text}`;
}

function OptionalText({ className, text }: { className: string; text?: string | null }) {
  if (!text) {
    return null;
  }
  return <p className={className}>{text}</p>;
}

function MatchReportActions({
  report,
  hasCandidate,
  onPlayCandidate,
  onRevealCandidate,
}: {
  report: MatchReport;
  hasCandidate: boolean;
  onPlayCandidate: (report: MatchReport) => void;
  onRevealCandidate: (report: MatchReport) => void;
}) {
  return (
    <div className="mt-2 flex flex-wrap gap-2">
      <Button size="sm" variant="secondary" disabled={!hasCandidate} onClick={() => onPlayCandidate(report)}>
        <Play className="h-3.5 w-3.5" />
        Play
      </Button>
      <Button size="sm" variant="outline" disabled={!hasCandidate} onClick={() => onRevealCandidate(report)}>
        <FolderOpen className="h-3.5 w-3.5" />
        Show
      </Button>
    </div>
  );
}

function MatchReportDebugDetails({ report, debugMode }: { report: MatchReport; debugMode: boolean }) {
  if (!debugMode) {
    return null;
  }
  return (
    <>
      {matchDebugTextItems(report).map((item) => (
        <OptionalText key={item.key} className={item.className} text={item.text} />
      ))}
      <MatchEvidenceDetails evidence={report.evidence} />
    </>
  );
}

type MatchDebugTextItem = { key: string; className: string; text: string | null };

function matchDebugTextItems(report: MatchReport): MatchDebugTextItem[] {
  return [
    matchDebugTextItem("closest", "mt-1 wrap-anywhere", listText("Closest signals", report.candidate_reasons)),
    matchDebugTextItem(
      "score",
      "mt-1 wrap-anywhere text-muted-foreground",
      scoreBreakdownText(report.candidate_scores),
    ),
    matchDebugTextItem("identity", "mt-1 wrap-anywhere", listText("Identity", report.reasons)),
    matchDebugTextItem(
      "warnings",
      "mt-1 wrap-anywhere text-yellow-700 dark:text-yellow-300",
      plainListText(report.warnings),
    ),
  ];
}

function matchDebugTextItem(key: string, className: string, text: string | null): MatchDebugTextItem {
  return { key, className, text };
}

function listText(label: string, values: string[] | undefined): string | null {
  return values?.length ? `${label}: ${values.join("; ")}` : null;
}

function plainListText(values: string[] | undefined): string | null {
  return values?.length ? values.join("; ") : null;
}

function scoreBreakdownText(scores: Record<string, number> | undefined): string | null {
  return scores ? `Score breakdown: ${formatMatchCandidateScores(scores)}` : null;
}

function MatchEvidenceDetails({ evidence }: { evidence?: MatchEvidenceItem[] }) {
  return (
    <details className="mt-2 rounded-md border bg-muted/30 p-2">
      <summary className="cursor-pointer font-medium text-foreground">Evidence details</summary>
      <div className="mt-2 flex flex-wrap gap-1">
        {matchEvidenceChips(evidence).map((chip) => (
          <Badge key={chip} variant="outline">{chip}</Badge>
        ))}
      </div>
      <p className="mt-2 wrap-anywhere text-muted-foreground">
        {formatMatchEvidenceDetails(evidence)}
      </p>
    </details>
  );
}

function matchEvidenceChips(evidence: MatchEvidenceItem[] | undefined): string[] {
  if (!evidence?.length) {
    return ["no evidence details"];
  }
  return evidence
    .filter((item) => item.status && item.status !== "missing")
    .map((item) => `${(item.source || "source").replaceAll("_", " ")}: ${item.status} ${numericText(item.score)}`)
    .slice(0, 8);
}

function formatMatchEvidenceDetails(evidence: MatchEvidenceItem[] | undefined): string {
  if (!evidence?.length) {
    return "No evidence details returned.";
  }
  return evidence
    .map((item) => {
      const details = item.details && Object.keys(item.details).length
        ? ` · ${formatEvidenceDetailsObject(item.details)}`
        : "";
      const decisive = item.decisive ? " decisive" : "";
      return `${(item.source || "source").replaceAll("_", " ")} ${item.status || "unknown"} ${numericText(item.score)}${decisive}${details}`;
    })
    .join("; ");
}

function formatEvidenceDetailsObject(details: Record<string, unknown>): string {
  return Object.entries(details)
    .slice(0, 5)
    .map(([key, value]) => `${key.replaceAll("_", " ")}=${formatCompactUnknown(value)}`)
    .join(", ");
}

function formatCompactUnknown(value: unknown): string {
  const formatter = compactUnknownFormatter(value);
  return formatter ? formatter(value) : String(value ?? "—");
}

function compactUnknownFormatter(value: unknown): ((value: unknown) => string) | null {
  if (Array.isArray(value)) {
    return formatCompactArray;
  }
  return value && typeof value === "object" ? () => "{…}" : null;
}

function formatCompactArray(value: unknown): string {
  const values = Array.isArray(value) ? value : [];
  const suffix = values.length > 4 ? ", …" : "";
  return `[${values.slice(0, 4).map(formatCompactUnknown).join(", ")}${suffix}]`;
}

function matchStrengthVariant(strength: MatchReport["candidate_strength"]): "default" | "secondary" | "outline" {
  if (strength === "strong") {
    return "default";
  }
  if (strength === "medium") {
    return "secondary";
  }
  return "outline";
}

function searchResultFromMatchReport(report: MatchReport): SearchResult | null {
  const track = playableMatchTrack(report);
  return track ? searchResultFromMatchTrack(report, track) : null;
}

type PlayableMatchTrack = NonNullable<MatchReport["track_b"]> & { path: string };

function playableMatchTrack(report: MatchReport): PlayableMatchTrack | null {
  const track = report.track_b;
  return track?.path ? { ...track, path: track.path } : null;
}

function searchResultFromMatchTrack(
  report: MatchReport,
  track: PlayableMatchTrack,
): SearchResult {
  return {
    track_id: track.track_id || track.path,
    path: track.path,
    title: track.title,
    artist: track.artist,
    album: track.album,
    score: matchReportScore(report),
    confidence: report.confidence ?? null,
    why: matchReportWhy(report),
  };
}

function matchReportScore(report: MatchReport): number {
  return report.candidate_score ?? report.confidence_score ?? 0;
}

function matchReportWhy(report: MatchReport): string[] | undefined {
  return report.candidate_reasons?.length ? report.candidate_reasons : report.reasons;
}

function IndexHealthCard({ health }: { health: IndexHealthPayload }) {
  const active = health.tracks.active ?? 0;
  const hasWarnings = health.warnings.length > 0;
  return (
    <div className="grid gap-3">
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        <HealthMetric label="Active tracks" value={`${active}`} detail={`${health.tracks.total ?? 0} total`} />
        <HealthMetric label="Audio features" value={coverageText(health.audio_features, active)} />
        <HealthMetric label="Profiles v2" value={`${health.profiles.schema_v2 ?? 0}/${active}`} />
        <HealthMetric label="Embeddings" value={`${health.embeddings.current ?? 0}/${active} current`} />
        <HealthMetric label="Derived tags" value={coverageText(health.derived_tags, active)} />
        <HealthMetric label="Context scores" value={coverageText(health.context_fit, active)} />
        <HealthMetric label="Failed" value={`${health.tracks.failed ?? 0}`} />
        <HealthMetric label="Quarantined" value={`${health.tracks.quarantined ?? 0}`} />
      </div>
      <div className="min-w-0 rounded-md bg-muted/50 p-2 text-xs text-muted-foreground">
        <p className="wrap-anywhere">DB: {health.db_path}</p>
        <p className="wrap-anywhere">Models: {health.models_path}</p>
        <p className="wrap-anywhere">
          Semantic model: {health.embeddings.selected_model || health.semantic_model}
        </p>
      </div>
      {hasWarnings ? (
        <div className="grid gap-2">
          {health.warnings.map((warning) => (
            <div
              key={warning.code}
              className={cn(
                "flex gap-2 rounded-md border p-2 text-xs",
                warning.severity === "error"
                  ? "border-destructive/40 bg-destructive/10 text-destructive"
                  : "bg-muted/50 text-muted-foreground",
              )}
            >
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>
                <span className="font-medium">{warning.code}</span>: {warning.message}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <p className="flex gap-2 rounded-md border bg-muted/50 p-2 text-xs text-muted-foreground">
          <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
          Index is ready for search.
        </p>
      )}
      {health.recommended_actions.length > 0 ? (
        <div className="grid gap-1 rounded-md border bg-muted/50 p-2 text-xs text-muted-foreground">
          <p className="font-medium text-foreground">Recommended actions</p>
          {health.recommended_actions.map((action) => (
            <p key={action}>• {action}</p>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function HealthMetric({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="rounded-md border bg-muted/40 p-2">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-medium">{value}</p>
      {detail ? <p className="text-xs text-muted-foreground">{detail}</p> : null}
    </div>
  );
}

function coverageText(coverage: HealthCoverage, activeTracks: number): string {
  const tracks = coverage.tracks ?? 0;
  return `${tracks}/${activeTracks}`;
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
  onMatch,
  matchPayload,
  matchLoading,
  debugMode,
  playingTrackId,
}: {
  result: SearchResult;
  index: number;
  busy: boolean;
  onFeedback: (result: SearchResult, rating: "good" | "bad" | "neutral") => Promise<void>;
  onPlay: (result: SearchResult) => Promise<void>;
  onReveal: (result: SearchResult) => Promise<void>;
  onMatch: (result: SearchResult) => Promise<void>;
  matchPayload: MatchTrackPayload | null;
  matchLoading: boolean;
  debugMode: boolean;
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
  const rankReasonText = formatRankReason(result.rank_reason);
  const evidenceText = formatCandidateEvidence(result.candidate_evidence);
  const hasLongContent = debugMode
    ? why.length > 180 ||
      tagText.length > 140 ||
      rankReasonText.length > 160 ||
      evidenceText.length > 160 ||
      result.path.length > 120 ||
      title.length > 80
    : why.length > 180 || title.length > 80;
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
        <div className="flex shrink-0 flex-col items-end gap-1">
          <Badge variant="secondary">{(result.score ?? 0).toFixed(3)}</Badge>
          {result.confidence ? (
            <Badge variant={result.confidence === "low" ? "outline" : "secondary"}>
              {result.confidence} confidence
            </Badge>
          ) : null}
        </div>
      </div>
      <div
        className={cn(
          "grid min-w-0 gap-2 transition-all",
          !expanded && hasLongContent && "max-h-24 overflow-hidden",
        )}
      >
        <p className="wrap-anywhere text-sm text-muted-foreground">{why}</p>
        {debugMode && rankReasonText ? (
          <p className="wrap-anywhere text-xs text-muted-foreground">Rank: {rankReasonText}</p>
        ) : null}
        {debugMode && evidenceText ? (
          <p className="wrap-anywhere text-xs text-muted-foreground">Evidence: {evidenceText}</p>
        ) : null}
        {debugMode && tagText ? (
          <p className="wrap-anywhere text-xs text-muted-foreground">Tags: {tagText}</p>
        ) : null}
        {debugMode ? <p className="wrap-anywhere text-xs text-muted-foreground">{result.path}</p> : null}
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
          <Button size="sm" variant={matchPayload ? "secondary" : "outline"} disabled={busy || matchLoading} onClick={() => void onMatch(result)}>
            {matchLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ListChecks className="h-3.5 w-3.5" />}
            {matchPayload ? "Hide matches" : "Matches"}
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
      {matchPayload ? (
        <MatchSummaryPanel
          payload={matchPayload}
          debugMode={debugMode}
          onPlayCandidate={(report) => {
            const candidate = searchResultFromMatchReport(report);
            if (candidate) {
              void onPlay(candidate);
            }
          }}
          onRevealCandidate={(report) => {
            const candidate = searchResultFromMatchReport(report);
            if (candidate) {
              void onReveal(candidate);
            }
          }}
        />
      ) : null}
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
    formatCount("derived tags", payload.tags_written),
    formatCount("context scores", payload.contexts_written),
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
    payload.schema_version ? `profile schema v${payload.schema_version}` : null,
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

type BrowseLoadOptions = {
  query?: string;
  sort?: string;
  direction?: "asc" | "desc";
  offset?: number;
};

type BuildBrowseArgsOptions = Required<BrowseLoadOptions> & {
  path?: string | null;
  musicFolder: string;
};

type BrowseLoadContext = {
  browseQuery: string;
  browseSort: string;
  browseSortDirection: "asc" | "desc";
  musicFolder: string;
};

function resolveBrowseLoadOptions(
  path: string | null | undefined,
  options: BrowseLoadOptions,
  context: BrowseLoadContext,
): BuildBrowseArgsOptions {
  return {
    path,
    musicFolder: context.musicFolder,
    query: optionValue(options.query, context.browseQuery),
    sort: optionValue(options.sort, context.browseSort),
    direction: optionValue(options.direction, context.browseSortDirection),
    offset: optionValue(options.offset, 0),
  };
}

function optionValue<T>(value: T | undefined, fallback: T): T {
  return value === undefined ? fallback : value;
}

function buildBrowseArgs(options: BuildBrowseArgsOptions): string[] {
  const args = [
    "browse",
    "--json",
    "--limit",
    "50",
    "--offset",
    String(Math.max(0, options.offset)),
    "--sort",
    options.sort,
  ];
  addFlag(args, options.direction === "desc", "--desc");
  addOptionValue(args, "--query", options.query.trim());
  addOptionValue(args, "--path", options.path || options.musicFolder.trim());
  return args;
}

function addFlag(args: string[], enabled: boolean, flag: string) {
  if (enabled) {
    args.push(flag);
  }
}

function addOptionValue(args: string[], flag: string, value: string) {
  if (value) {
    args.push(flag, value);
  }
}

function browseStatusMessage(payload: BrowsePayload): string {
  if (!payload.cwd) {
    return "No indexed library roots found";
  }
  return payload.mode === "search" ? "Library search loaded" : "Library browser loaded";
}

type BuildIndexStepsOptions = {
  musicFolder: string;
  analysisMode: IndexAnalysisMode;
  resourceProfile: string;
  semanticModel: string;
};

function buildIndexSteps(options: BuildIndexStepsOptions): IndexStep[] {
  return [
    {
      id: "scan",
      label: "Scan files",
      args: () => ["scan", required(options.musicFolder, "Music folder"), "--json"],
    },
    {
      id: "metadata",
      label: "Read metadata",
      args: () => ["metadata", "--missing-only", "--json"],
    },
    {
      id: "repair-metadata",
      label: "Repair metadata",
      args: () => [
        "repair-metadata",
        "--from-filename",
        "--from-duplicates",
        "--missing-only",
        "--json",
      ],
    },
    {
      id: "fingerprint",
      label: "Fingerprint",
      args: () => ["fingerprint", "--missing-only", "--json"],
    },
    {
      id: "basic",
      label: "Audio features",
      args: () => basicAnalysisArgs(options.analysisMode, options.resourceProfile),
    },
    {
      id: "tags",
      label: "ML tags",
      args: () => tagAnalysisArgs(options.resourceProfile),
    },
    {
      id: "derived",
      label: "Derived tags + context fit",
      args: () => ["rebuild-derived", "--json"],
    },
    {
      id: "profiles",
      label: "Rebuild profiles",
      args: () => ["rebuild-profiles", "--json"],
    },
    {
      id: "embed",
      label: "Profile embeddings",
      args: () => embeddingArgs(options.resourceProfile, options.semanticModel),
    },
  ];
}

function basicAnalysisArgs(mode: IndexAnalysisMode, resourceProfile: string): string[] {
  const args = ["analyze-basic"];
  args.push("--chunked", "--chunk-sec", mode === "full" ? "300" : "auto");
  args.push("--workers", "auto", "--resource-profile", resourceProfile, "--json");
  return args;
}

function tagAnalysisArgs(resourceProfile: string): string[] {
  return [
    "analyze-tags",
    "--missing-only",
    "--workers",
    "auto",
    "--resource-profile",
    resourceProfile,
    "--subprocess-batches",
    "--batch-size",
    "auto",
    "--json",
  ];
}

function embeddingArgs(resourceProfile: string, semanticModel: string): string[] {
  const args = ["embed", "--batch-size", "auto", "--resource-profile", resourceProfile, "--json"];
  if (semanticModel.trim()) {
    args.splice(1, 0, "--model", semanticModel.trim());
  }
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

function searchSuggestionsFromDiagnostics(diagnostics: Record<string, unknown>): SearchSuggestion[] {
  return arrayValue(diagnostics.suggested_queries)
    .map(searchSuggestionFromUnknown)
    .filter(isPresent);
}

function searchSuggestionFromUnknown(value: unknown): SearchSuggestion | null {
  const record = recordValue(value);
  const query = stringValue(record.query).trim();
  if (!query) {
    return null;
  }
  return {
    query,
    confidence: numberValue(record.confidence),
    reason: stringOrUndefined(record.reason),
    kind: stringOrUndefined(record.kind),
    type: stringOrUndefined(record.type),
    label: stringOrUndefined(record.label),
  };
}

function searchTypeExamplesFromDiagnostics(diagnostics: Record<string, unknown>): SearchTypeExample[] {
  return arrayValue(diagnostics.search_type_examples).map(searchTypeExampleFromUnknown);
}

function searchTypeExampleFromUnknown(value: unknown): SearchTypeExample {
  const record = recordValue(value);
  return {
    type: stringOrUndefined(record.type),
    label: stringOrUndefined(record.label),
    query: stringOrNull(record.query),
    reason: stringOrUndefined(record.reason),
  };
}

function compactSearchSuggestions(suggestions: SearchSuggestion[]): SearchSuggestion[] {
  const seen = new Set<string>();
  const deduped = suggestions.filter((suggestion) => {
    const key = suggestion.query.trim().toLowerCase();
    if (!key || seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
  const corrections = deduped.filter((suggestion) => suggestion.kind === "correction").slice(0, 3);
  const examples = deduped.filter((suggestion) => suggestion.kind !== "correction").slice(0, 5);
  return corrections.length ? [...corrections, ...examples] : examples.slice(0, 6);
}

function compactSearchTypeExamples(examples: SearchTypeExample[]): SearchTypeExample[] {
  const preferredTypes = [
    "mood",
    "context",
    "occasion",
    "feature_sort",
    "feature_filter",
    "vocals",
    "negation",
    "closest_tracks",
  ];
  const byType = new Map(examples.map((example) => [example.type, example]));
  const preferred = preferredTypes
    .map((type) => byType.get(type))
    .filter((example): example is SearchTypeExample => Boolean(example));
  const remainder = examples.filter((example) => !example.type || !preferredTypes.includes(example.type));
  return [...preferred, ...remainder].slice(0, 8);
}

function searchNoticeFromDiagnostics(diagnostics: Record<string, unknown>): SearchNotice | null {
  const value = diagnostics.result_notice;
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const record = value as Record<string, unknown>;
  return {
    level: typeof record.level === "string" ? record.level : undefined,
    message: typeof record.message === "string" ? record.message : undefined,
  };
}

const SEARCH_NOTICE_COPY_RULES = [
  {
    tokens: ["no strong local evidence"],
    text: "No strong local evidence found. Try one of the suggestions below.",
  },
  {
    tokens: ["low confidence", "weak"],
    text: "Best matches look weak. Try a more specific query, or enable Debug mode for score details.",
  },
  {
    tokens: ["semantic"],
    text: "Results rely mostly on similarity signals. Try adding an artist, genre, mood, or feature.",
  },
] as const;

function searchNoticeDisplayText(notice: SearchNotice, debugMode: boolean): string {
  const message = notice.message?.trim() || "";
  return debugMode ? message : mappedNoticeMessage(message);
}

function mappedNoticeMessage(message: string): string {
  const lower = message.toLowerCase();
  const match = SEARCH_NOTICE_COPY_RULES.find((rule) =>
    rule.tokens.some((token) => lower.includes(token)),
  );
  return match?.text || message;
}

function chipList(values: string[] | undefined, max = 8): string {
  if (!values || values.length === 0) {
    return "none";
  }
  const shown = values.slice(0, max);
  const suffix = values.length > max ? ` +${values.length - max} more` : "";
  return `${shown.join(", ")}${suffix}`;
}

function formatFeaturePreferences(preferences: Record<string, string> | undefined): string {
  if (!preferences || Object.keys(preferences).length === 0) {
    return "none";
  }
  return Object.entries(preferences)
    .map(([field, level]) => `${field.replaceAll("_", " ")} ${level.replaceAll("_", " ")}`)
    .join(" · ");
}

function formatFeatureRanges(intent: SearchIntentSummary): string {
  const entries = Object.entries(intent.feature_ranges ?? {});
  return entries.length ? entries.map(formatFeatureRange).join(" · ") : "none";
}

function formatFeatureRange([field, range]: [string, SearchFeatureRange]): string {
  const low = numericText(range.low);
  const high = numericText(range.high);
  const source = range.source ? ` ${range.source.replaceAll("_", " ")}` : "";
  return `${field.replaceAll("_", " ")} ${low}–${high}${source}`;
}

function formatSortSpecs(intent: SearchIntentSummary): string {
  return formatSortSpecList(intent.sort_by);
}

function formatSortSpecList(specs: SearchSortSpec[] | undefined): string {
  const values = specs ?? [];
  return values.length ? values.map(formatSortSpec).join(" · ") : "none";
}

function formatSortSpec(spec: SearchSortSpec): string {
  const field = (spec.field || "unknown").replaceAll("_", " ");
  const direction = spec.direction === "asc" ? "lowest first" : "highest first";
  return `${field} ${direction}`;
}

function formatScoreCalibration(diagnostics: Record<string, unknown>): string {
  return String(diagnostics.score_calibration || diagnostics.score_normalization || "raw");
}

function searchEvidenceCountStages(diagnostics: Record<string, unknown>): EvidenceCountStages {
  return [
    ["scored", recordValue(diagnostics.scored_evidence_source_counts)],
    ["filtered", recordValue(diagnostics.filtered_evidence_source_counts)],
    ["limited", recordValue(diagnostics.limited_evidence_source_counts)],
  ] as const;
}

function formatSearchDiagnostics(diagnostics: Record<string, unknown>, scoreWarnings: string[]): string {
  return searchDiagnosticParts(diagnostics, scoreWarnings).filter(isNonEmptyString).join(" · ");
}

function searchDiagnosticParts(diagnostics: Record<string, unknown>, scoreWarnings: string[]): Array<string | null> {
  return [
    metricText("top", diagnostics.top_raw_score),
    metricText("filtered", diagnostics.filtered_candidate_count),
    duplicateDiagnosticText(diagnostics),
    diversityDiagnosticText(diagnostics),
    warningDiagnosticText(scoreWarnings),
  ];
}

function duplicateDiagnosticText(diagnostics: Record<string, unknown>): string | null {
  return diagnostics.duplicate_suppression_enabled === false
    ? "duplicates shown"
    : metricText("deduped", diagnostics.duplicate_suppressed_count);
}

function diversityDiagnosticText(diagnostics: Record<string, unknown>): string {
  return diagnostics.diversity_suppression_enabled === false ? "same artists shown" : "";
}

function warningDiagnosticText(scoreWarnings: string[]): string {
  return scoreWarnings.length ? `warnings ${scoreWarnings.join(", ")}` : "";
}

function formatEvidenceCountStages(
  stages: ReadonlyArray<readonly [string, Record<string, unknown>]>,
): string {
  const parts = stages.map(evidenceStageText).filter(isNonEmptyString);
  return parts.length ? parts.join(" · ") : "none";
}

function evidenceStageText([stage, counts]: readonly [string, Record<string, unknown>]): string {
  return positiveStageText(stage, recordNumberTotal(counts));
}

function positiveStageText(stage: string, total: number): string {
  return total > 0 ? `${stage} ${total}` : "";
}

function recordNumberTotal(counts: Record<string, unknown>): number {
  return Object.values(counts).reduce(addFiniteNumber, 0);
}

function addFiniteNumber(total: number, value: unknown): number {
  return isFiniteNumber(value) ? total + value : total;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function formatRecordCounts(counts: Record<string, unknown>): string {
  const entries = Object.entries(counts).filter(
    ([, value]) => typeof value === "number" && Number.isFinite(value) && value > 0,
  );
  if (!entries.length) {
    return "none";
  }
  return entries
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => `${key.replaceAll("_", " ")} ${numericText(value)}`)
    .join(" · ");
}

function formatRankReason(reason: SearchResult["rank_reason"]): string {
  return reason ? rankReasonParts(reason).join(" · ") : "";
}

function rankReasonParts(reason: NonNullable<SearchResult["rank_reason"]>): string[] {
  return [rankSummaryText(reason), rankModeText(reason), rankSortText(reason)].filter(isNonEmptyString);
}

function rankSummaryText(reason: NonNullable<SearchResult["rank_reason"]>): string {
  return reason.summary || reason.primary?.replaceAll("_", " ") || "";
}

function rankModeText(reason: NonNullable<SearchResult["rank_reason"]>): string {
  return reason.mode ? `mode ${reason.mode.replaceAll("_", " ")}` : "";
}

function rankSortText(reason: NonNullable<SearchResult["rank_reason"]>): string {
  return reason.sort ? rankSortLabel(reason.sort).trim() : "";
}

function rankSortLabel(sort: NonNullable<NonNullable<SearchResult["rank_reason"]>["sort"]>): string {
  return `${sortLabel(sort)} ${sort.direction || ""} ${numericText(sort.value)}`;
}

function sortLabel(sort: NonNullable<NonNullable<SearchResult["rank_reason"]>["sort"]>): string {
  return sort.label || sort.field || "sort";
}

function formatMatchCandidateScores(scores: Record<string, number>): string {
  const ordered = [
    "name",
    "content_hash",
    "chromaprint",
    "fingerprint_similarity",
    "duration",
    "artist_similarity",
    "artist_title_norm",
    "album_similarity",
    "filename_stem",
    "feature_similarity",
    "audio_embedding",
  ];
  return ordered
    .map((key) => [key, scores[key]] as const)
    .filter(([, value]) => typeof value === "number" && Number.isFinite(value) && value > 0)
    .map(([key, value]) => `${key.replaceAll("_", " ")} ${numericText(value)}`)
    .join(" · ") || "none";
}

function formatCandidateEvidence(evidence: SearchResult["candidate_evidence"]): string {
  return evidence ? candidateEvidenceParts(evidence).join(" · ") : "";
}

function candidateEvidenceParts(evidence: NonNullable<SearchResult["candidate_evidence"]>): string[] {
  return [retrievalSourcesText(evidence), semanticOnlyText(evidence), identitySignalsText(evidence)].filter(
    isNonEmptyString,
  );
}

function retrievalSourcesText(evidence: NonNullable<SearchResult["candidate_evidence"]>): string {
  return evidence.retrieved_by?.length ? evidence.retrieved_by.map(prettyKey).join(", ") : "none";
}

function semanticOnlyText(evidence: NonNullable<SearchResult["candidate_evidence"]>): string {
  return evidence.semantic_only ? "semantic only" : "";
}

function identitySignalsText(evidence: NonNullable<SearchResult["candidate_evidence"]>): string {
  const signals = Object.entries(evidence.identity ?? {})
    .filter(([, available]) => available)
    .map(([key]) => prettyKey(key))
    .join(", ");
  return signals ? `identity ${signals}` : "";
}

function prettyKey(value: string): string {
  return value.replaceAll("_", " ");
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function formatWeights(weights: Record<string, unknown>): string {
  return Object.entries(weights)
    .map(([key, value]) => `${key} ${numericText(value)}`)
    .join(" · ");
}

function metricText(label: string, value: unknown): string | null {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  return `${label} ${numericText(value)}`;
}

function numericText(value: unknown): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value.toFixed(Math.abs(value) >= 10 ? 0 : 3).replace(/\.0+$/, "");
  }
  return String(value ?? "—");
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function recordValue(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function stringOrUndefined(value: unknown): string | undefined {
  const text = stringValue(value);
  return text || undefined;
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined;
}

function isPresent<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

function settingsHealthWarnings(settings: SettingsState): string[] {
  const warnings: string[] = [];
  const dbPath = normalizePathForWarning(settings.dbPath);
  const modelsPath = normalizePathForWarning(settings.modelsPath);
  if (dbPath && modelsPath && dbPath.startsWith(`${modelsPath}/`)) {
    warnings.push("DB path is inside the models folder. Use a separate musicidx.sqlite path.");
  }
  if (modelsPath.includes("desktop/src-tauri/target/debug/resources/models")) {
    warnings.push("Models path points to Tauri debug resources. Use the repo .musicidx-models folder in dev.");
  }
  return warnings;
}

function normalizePathForWarning(value: string): string {
  return value.trim().replace(/\\/g, "/").replace(/\/+$/, "");
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
