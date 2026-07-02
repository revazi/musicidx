import { ArrowLeft, Database, FolderOpen, ListChecks, Loader2, Play, Search } from "lucide-react";
import type { ReactNode } from "react";

import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import type { BrowseFolder, BrowsePayload, BrowseRoot, BrowseTrack, MatchTrackPayload } from "../types";

type BrowseActions = {
  onOpenPath: (path: string) => void;
  onPlay: (track: BrowseTrack) => void;
  onReveal: (track: BrowseTrack) => void;
  onMatch: (track: BrowseTrack) => void;
};

type BrowseSearchControlsProps = {
  query: string;
  sort: string;
  sortDirection: "asc" | "desc";
  onQueryChange: (query: string) => void;
  onSortChange: (sort: string) => void;
  onSortDirectionChange: (direction: "asc" | "desc") => void;
};

type BrowsePagingActions = {
  onSubmitSearch: () => void;
  onClearSearch: () => void;
  onPage: (offset: number) => void;
};

type LibraryBrowsePanelProps = {
  payload: BrowsePayload | null;
  expanded: boolean;
  busy: boolean;
  matchPayloads: Record<string, MatchTrackPayload>;
  matchingTrackId: string | null;
  debugMode: boolean;
  onToggle: () => void;
  onLoad: () => void;
  renderMatches?: (payload: MatchTrackPayload) => ReactNode;
} & BrowseActions & BrowseSearchControlsProps & BrowsePagingActions;

const LIBRARY_SORT_OPTIONS = [
  { value: "artist:asc", label: "Artist A→Z" },
  { value: "artist:desc", label: "Artist Z→A" },
  { value: "title:asc", label: "Title A→Z" },
  { value: "title:desc", label: "Title Z→A" },
  { value: "album:asc", label: "Album A→Z" },
  { value: "genre:asc", label: "Genre A→Z" },
  { value: "bpm:desc", label: "BPM high→low" },
  { value: "bpm:asc", label: "BPM low→high" },
  { value: "duration:desc", label: "Longest first" },
  { value: "duration:asc", label: "Shortest first" },
  { value: "indexed_at:desc", label: "Recently indexed" },
  { value: "path:asc", label: "File path A→Z" },
] as const;

export function LibraryBrowsePanel(props: LibraryBrowsePanelProps) {
  return (
    <div className="grid gap-3 rounded-lg border bg-background/60 p-3 text-sm">
      <LibraryBrowseHeader {...libraryBrowseHeaderProps(props)} />
      <LibraryBrowseExpandedBody props={props} />
    </div>
  );
}

type LibraryBrowseExpandedBodyProps = { props: LibraryBrowsePanelProps };

function LibraryBrowseExpandedBody({ props }: LibraryBrowseExpandedBodyProps) {
  if (!props.expanded) {
    return null;
  }
  return <LibraryBrowseBody {...props} />;
}

function libraryBrowseHeaderProps(props: LibraryBrowsePanelProps): LibraryBrowseHeaderProps {
  return {
    payload: props.payload,
    expanded: props.expanded,
    busy: props.busy,
    visibleTrackCount: props.payload?.tracks?.length ?? 0,
    onToggle: props.onToggle,
    onLoad: props.onLoad,
  };
}

type LibraryBrowseHeaderProps = {
  payload: BrowsePayload | null;
  expanded: boolean;
  busy: boolean;
  visibleTrackCount: number;
  onToggle: () => void;
  onLoad: () => void;
};

function LibraryBrowseHeader(props: LibraryBrowseHeaderProps) {
  const model = libraryBrowseHeaderModel(props);
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <LibraryBrowseHeaderTitle cwd={model.cwd} onToggle={props.onToggle} />
      <div className="flex flex-wrap items-center gap-2">
        <OptionalBadge text={model.countText} />
        <Button size="sm" variant="outline" disabled={props.busy} onClick={props.onLoad}>
          <LibraryBrowseLoadIcon busy={props.busy} />
          {model.loadLabel}
        </Button>
        <Button size="sm" variant="ghost" onClick={props.onToggle}>{model.toggleLabel}</Button>
      </div>
    </div>
  );
}

type LibraryBrowseHeaderModel = {
  cwd: string;
  countText: string;
  loadLabel: string;
  toggleLabel: string;
};

function libraryBrowseHeaderModel(props: LibraryBrowseHeaderProps): LibraryBrowseHeaderModel {
  return {
    cwd: libraryBrowseCwd(props.payload),
    countText: libraryBrowseCountText(props.payload, props.visibleTrackCount),
    loadLabel: libraryBrowseLoadLabel(props.payload),
    toggleLabel: libraryBrowseToggleLabel(props.expanded),
  };
}

function libraryBrowseCwd(payload: BrowsePayload | null): string {
  if (!payload) {
    return "";
  }
  return payload.cwd || "";
}

function libraryBrowseLoadLabel(payload: BrowsePayload | null): string {
  return payload ? "Refresh" : "Browse";
}

function libraryBrowseToggleLabel(expanded: boolean): string {
  return expanded ? "Hide" : "Show";
}

function libraryBrowseCountText(payload: BrowsePayload | null, visibleTrackCount: number): string {
  return payload ? `${visibleTrackCount}/${payload.track_count ?? visibleTrackCount} files` : "";
}

function LibraryBrowseHeaderTitle({ cwd, onToggle }: { cwd: string; onToggle: () => void }) {
  return (
    <button className="flex min-w-0 items-center gap-2 text-left" type="button" onClick={onToggle}>
      <Database className="h-4 w-4 shrink-0 text-muted-foreground" />
      <span className="font-medium">Indexed library</span>
      <OptionalInlineText text={cwd} />
    </button>
  );
}

function OptionalInlineText({ text }: { text: string }) {
  if (!text) {
    return null;
  }
  return <span className="wrap-anywhere text-xs text-muted-foreground">{text}</span>;
}

function OptionalBadge({ text }: { text: string }) {
  if (!text) {
    return null;
  }
  return <Badge variant="secondary">{text}</Badge>;
}

function LibraryBrowseLoadIcon({ busy }: { busy: boolean }) {
  return busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FolderOpen className="h-3.5 w-3.5" />;
}

function LibraryBrowseBody(props: LibraryBrowsePanelProps) {
  const model = libraryBrowseBodyModel(props.payload);
  return (
    <div className="grid gap-3">
      <LibraryBrowseWarning warning={model.warning} />
      <LibraryBrowseControls {...libraryBrowseControlsProps(props)} />
      <LibraryBrowseNavigation model={model} payload={props.payload} onOpenPath={props.onOpenPath} />
      <LibraryBrowsePager payload={props.payload} busy={props.busy} onPage={props.onPage} />
      <LibraryBrowseTrackSection props={props} tracks={model.tracks} />
      <LibraryBrowsePager payload={props.payload} busy={props.busy} onPage={props.onPage} />
    </div>
  );
}

type LibraryBrowseBodyModel = {
  warning?: string | null;
  roots: BrowseRoot[];
  folders: BrowseFolder[];
  tracks: BrowseTrack[];
};

function libraryBrowseBodyModel(payload: BrowsePayload | null): LibraryBrowseBodyModel {
  if (!payload) {
    return emptyBrowseBodyModel();
  }
  return browseBodyModelFromPayload(payload);
}

function emptyBrowseBodyModel(): LibraryBrowseBodyModel {
  return { warning: undefined, roots: [], folders: [], tracks: [] };
}

function browseBodyModelFromPayload(payload: BrowsePayload): LibraryBrowseBodyModel {
  return {
    warning: payload.warning,
    roots: payload.roots ?? [],
    folders: payload.folders ?? [],
    tracks: payload.tracks ?? [],
  };
}

function libraryBrowseControlsProps(
  props: LibraryBrowsePanelProps,
): BrowseSearchControlsProps & { busy: boolean; onSubmitSearch: () => void; onClearSearch: () => void } {
  return {
    query: props.query,
    sort: props.sort,
    sortDirection: props.sortDirection,
    busy: props.busy,
    onQueryChange: props.onQueryChange,
    onSortChange: props.onSortChange,
    onSortDirectionChange: props.onSortDirectionChange,
    onSubmitSearch: props.onSubmitSearch,
    onClearSearch: props.onClearSearch,
  };
}

function LibraryBrowseNavigation({
  model,
  payload,
  onOpenPath,
}: {
  model: LibraryBrowseBodyModel;
  payload: BrowsePayload | null;
  onOpenPath: (path: string) => void;
}) {
  return (
    <>
      <LibraryBrowseRoots payload={payload} roots={model.roots} onOpenPath={onOpenPath} />
      <LibraryBrowseLocation payload={payload} onOpenPath={onOpenPath} />
      <LibraryBrowseFolders folders={model.folders} onOpenPath={onOpenPath} />
    </>
  );
}

function LibraryBrowseTrackSection({ props, tracks }: { props: LibraryBrowsePanelProps; tracks: BrowseTrack[] }) {
  return (
    <LibraryBrowseTracks
      payload={props.payload}
      tracks={tracks}
      busy={props.busy}
      matchPayloads={props.matchPayloads}
      matchingTrackId={props.matchingTrackId}
      debugMode={props.debugMode}
      onOpenPath={props.onOpenPath}
      onPlay={props.onPlay}
      onReveal={props.onReveal}
      onMatch={props.onMatch}
      renderMatches={props.renderMatches}
    />
  );
}

function LibraryBrowseWarning({ warning }: { warning?: string | null }) {
  if (!warning) {
    return null;
  }
  return (
    <div className="rounded-md border border-yellow-500/30 bg-yellow-500/10 p-2 text-xs text-yellow-700 dark:text-yellow-300">
      {warning}
    </div>
  );
}

function LibraryBrowseControls({
  query,
  sort,
  sortDirection,
  busy,
  onQueryChange,
  onSortChange,
  onSortDirectionChange,
  onSubmitSearch,
  onClearSearch,
}: BrowseSearchControlsProps & {
  busy: boolean;
  onSubmitSearch: () => void;
  onClearSearch: () => void;
}) {
  const selectedSort = friendlySortValue(sort, sortDirection);
  return (
    <div className="grid gap-2 rounded-md border bg-muted/40 p-2 text-xs">
      <div className="grid grid-cols-[minmax(0,1fr)_minmax(8rem,25%)] items-center gap-2">
        <Input
          className="h-8"
          value={query}
          placeholder="Find title, artist, album, genre, or filename…"
          onChange={(event) => onQueryChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !busy) {
              onSubmitSearch();
            }
          }}
        />
        <select
          className="h-8 min-w-0 rounded-md border border-input bg-background px-2 text-xs"
          value={selectedSort}
          aria-label="Sort indexed library"
          onChange={(event) => {
            const [nextSort, nextDirection] = event.target.value.split(":") as [string, "asc" | "desc"];
            onSortChange(nextSort);
            onSortDirectionChange(nextDirection);
          }}
        >
          {LIBRARY_SORT_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <div className="col-span-2 flex justify-end gap-2">
          <Button size="sm" variant="outline" disabled={busy || !query.trim()} onClick={onClearSearch}>
            Clear
          </Button>
          <Button size="sm" disabled={busy} onClick={onSubmitSearch}>
            {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Search className="h-3.5 w-3.5" />}
            Search
          </Button>
        </div>
      </div>
      <p className="text-muted-foreground">
        Simple local SQLite search. It does not use semantic ranking or LLM hints.
      </p>
    </div>
  );
}

function friendlySortValue(sort: string, direction: "asc" | "desc"): string {
  const value = `${sort}:${direction}`;
  return LIBRARY_SORT_OPTIONS.some((option) => option.value === value) ? value : "artist:asc";
}

function LibraryBrowsePager({
  payload,
  busy,
  onPage,
}: {
  payload: BrowsePayload | null;
  busy: boolean;
  onPage: (offset: number) => void;
}) {
  const model = libraryBrowsePagerModel(payload, busy);
  if (!model) {
    return null;
  }
  return <LibraryBrowsePagerView model={model} onPage={onPage} />;
}

type LibraryBrowsePagerModel = {
  label: string;
  previousOffset: number;
  nextOffset: number;
  previousDisabled: boolean;
  nextDisabled: boolean;
};

function libraryBrowsePagerModel(payload: BrowsePayload | null, busy: boolean): LibraryBrowsePagerModel | null {
  const metrics = browsePageMetrics(payload);
  return metrics.visible ? browsePagerModelFromMetrics(metrics, busy) : null;
}

type BrowsePageMetrics = {
  visible: boolean;
  mode: string;
  query: string;
  limit: number;
  offset: number;
  total: number;
  shown: number;
  hasMore: boolean;
};

function browsePageMetrics(payload: BrowsePayload | null): BrowsePageMetrics {
  if (!payload) {
    return emptyBrowsePageMetrics();
  }
  return browsePageMetricsFromPayload(payload);
}

function emptyBrowsePageMetrics(): BrowsePageMetrics {
  return {
    visible: false,
    mode: "browse",
    query: "",
    limit: 50,
    offset: 0,
    total: 0,
    shown: 0,
    hasMore: false,
  };
}

function browsePageMetricsFromPayload(payload: BrowsePayload): BrowsePageMetrics {
  const totals = browsePageTotals(payload);
  return {
    visible: browsePageVisibility(totals),
    mode: payload.mode || "browse",
    query: payload.query || "",
    limit: totals.limit,
    offset: totals.offset,
    total: totals.total,
    shown: browseShownCount(payload),
    hasMore: totals.hasMore,
  };
}

function browseShownCount(payload: BrowsePayload): number {
  return payload.tracks?.length ?? 0;
}

function hasBrowsePages(payload: BrowsePayload | null): boolean {
  if (!payload) {
    return false;
  }
  return browsePageVisibility(browsePageTotals(payload));
}

type BrowsePageTotals = { limit: number; offset: number; total: number; hasMore: boolean };

function browsePageTotals(payload: BrowsePayload): BrowsePageTotals {
  return {
    limit: numberOr(payload.limit, 50),
    offset: numberOr(payload.offset, 0),
    total: numberOr(payload.track_count, 0),
    hasMore: Boolean(payload.has_more),
  };
}

function browsePageVisibility(totals: BrowsePageTotals): boolean {
  return totals.total > totals.limit || totals.offset > 0 || totals.hasMore;
}

function browsePagerModelFromMetrics(metrics: BrowsePageMetrics, busy: boolean): LibraryBrowsePagerModel {
  return {
    label: browsePagerLabel(metrics),
    previousOffset: Math.max(0, metrics.offset - metrics.limit),
    nextOffset: metrics.offset + metrics.limit,
    previousDisabled: busy || metrics.offset <= 0,
    nextDisabled: busy || !metrics.hasMore,
  };
}

function browsePagerLabel(metrics: BrowsePageMetrics): string {
  return `${browsePagerKind(metrics.mode)}: ${browsePageRange(metrics)} of ${metrics.total}${browseQuerySuffix(metrics.query)}`;
}

function browsePagerKind(mode: string): string {
  return mode === "search" ? "Search results" : "Files";
}

function browsePageRange(metrics: BrowsePageMetrics): string {
  return `${browsePageStart(metrics)}–${metrics.offset + metrics.shown}`;
}

function browsePageStart(metrics: BrowsePageMetrics): number {
  return metrics.total > 0 && metrics.shown > 0 ? metrics.offset + 1 : 0;
}

function browseQuerySuffix(query: string): string {
  return query ? ` for “${query}”` : "";
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === "number" ? value : fallback;
}

function LibraryBrowsePagerView({
  model,
  onPage,
}: {
  model: LibraryBrowsePagerModel;
  onPage: (offset: number) => void;
}) {
  return (
    <div className="flex flex-col gap-2 rounded-md border bg-background/50 p-2 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
      <span>{model.label}</span>
      <div className="flex gap-2">
        <Button size="sm" variant="outline" disabled={model.previousDisabled} onClick={() => onPage(model.previousOffset)}>
          Previous
        </Button>
        <Button size="sm" variant="outline" disabled={model.nextDisabled} onClick={() => onPage(model.nextOffset)}>
          Next
        </Button>
      </div>
    </div>
  );
}

function LibraryBrowseRoots({
  payload,
  roots,
  onOpenPath,
}: {
  payload: BrowsePayload | null;
  roots: BrowseRoot[];
  onOpenPath: (path: string) => void;
}) {
  if (!roots.length && payload) {
    return (
      <div className="rounded-md border border-dashed p-3 text-xs text-muted-foreground">
        No indexed roots yet. Run indexing from Settings or the tag button first.
      </div>
    );
  }
  if (!roots.length) {
    return null;
  }
  return (
    <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
      <span className="font-medium text-foreground">Roots</span>
      {roots.map((root) => (
        <LibraryBrowseRootButton key={root.path} root={root} activeRoot={payload?.root || ""} onOpenPath={onOpenPath} />
      ))}
    </div>
  );
}

function LibraryBrowseRootButton({
  root,
  activeRoot,
  onOpenPath,
}: {
  root: BrowseRoot;
  activeRoot: string;
  onOpenPath: (path: string) => void;
}) {
  return (
    <Button size="sm" variant={rootButtonVariant(root, activeRoot)} onClick={() => onOpenPath(root.path)}>
      {rootButtonText(root)}
    </Button>
  );
}

function rootButtonVariant(root: BrowseRoot, activeRoot: string): "secondary" | "outline" {
  return activeRoot === root.path ? "secondary" : "outline";
}

function rootButtonText(root: BrowseRoot): string {
  return `${root.name || root.path} ${rootTrackCountText(root)}`.trim();
}

function rootTrackCountText(root: BrowseRoot): string {
  return typeof root.track_count === "number" ? `(${root.track_count})` : "";
}

function LibraryBrowseLocation({
  payload,
  onOpenPath,
}: {
  payload: BrowsePayload | null;
  onOpenPath: (path: string) => void;
}) {
  if (!payload?.cwd) {
    return null;
  }
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
      {payload.parent ? (
        <Button size="sm" variant="outline" onClick={() => onOpenPath(payload.parent || "")}>
          <ArrowLeft className="h-3.5 w-3.5" />
          Parent
        </Button>
      ) : null}
      <span className="wrap-anywhere">{payload.cwd}</span>
    </div>
  );
}

function LibraryBrowseFolders({
  folders,
  onOpenPath,
}: {
  folders: BrowseFolder[];
  onOpenPath: (path: string) => void;
}) {
  if (!folders.length) {
    return null;
  }
  return (
    <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
      {folders.map((folder) => (
        <button
          key={folder.path}
          type="button"
          className="flex min-w-0 items-center justify-between gap-2 rounded-md border bg-muted/40 p-2 text-left hover:bg-muted"
          onClick={() => onOpenPath(folder.path)}
        >
          <span className="min-w-0 wrap-anywhere font-medium">{folder.name}</span>
          <Badge variant="outline">{folder.track_count ?? 0}</Badge>
        </button>
      ))}
    </div>
  );
}

function LibraryBrowseTracks({
  payload,
  tracks,
  busy,
  matchPayloads,
  matchingTrackId,
  debugMode,
  onPlay,
  onReveal,
  onMatch,
  renderMatches,
}: {
  payload: BrowsePayload | null;
  tracks: BrowseTrack[];
  busy: boolean;
  matchPayloads: Record<string, MatchTrackPayload>;
  matchingTrackId: string | null;
  debugMode: boolean;
  renderMatches?: (payload: MatchTrackPayload) => ReactNode;
} & BrowseActions) {
  if (!tracks.length) {
    return <LibraryBrowseEmpty payload={payload} />;
  }
  return (
    <div className="grid gap-2">
      {tracks.map((track) => (
        <LibraryBrowseTrackRow
          key={track.track_id}
          track={track}
          busy={busy}
          matchPayload={matchPayloads[track.track_id] ?? null}
          matchLoading={matchingTrackId === track.track_id}
          debugMode={debugMode}
          onPlay={onPlay}
          onReveal={onReveal}
          onMatch={onMatch}
          renderMatches={renderMatches}
        />
      ))}
    </div>
  );
}

function LibraryBrowseEmpty({ payload }: { payload: BrowsePayload | null }) {
  const message = !payload
    ? "Browse indexed folders/files before searching."
    : payload.mode === "search"
      ? "No indexed tracks matched this simple search. Try fewer keywords or a different sort."
      : "No direct tracks on this page. Open a child folder, search this library, or choose another root.";
  return (
    <div className="rounded-md border border-dashed p-3 text-xs text-muted-foreground">
      {message}
    </div>
  );
}

function LibraryBrowseTrackRow({
  track,
  busy,
  matchPayload,
  matchLoading,
  debugMode,
  onPlay,
  onReveal,
  onMatch,
  renderMatches,
}: {
  track: BrowseTrack;
  busy: boolean;
  matchPayload: MatchTrackPayload | null;
  matchLoading: boolean;
  debugMode: boolean;
  onPlay: (track: BrowseTrack) => void;
  onReveal: (track: BrowseTrack) => void;
  onMatch: (track: BrowseTrack) => void;
  renderMatches?: (payload: MatchTrackPayload) => ReactNode;
}) {
  return (
    <div className="rounded-md border bg-muted/30 p-2">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <p className="wrap-anywhere font-medium">{browseTrackTitle(track)}</p>
          <p className="wrap-anywhere text-xs text-muted-foreground">{browseTrackMeta(track)}</p>
          {debugMode ? <p className="wrap-anywhere text-xs text-muted-foreground">{track.path}</p> : null}
        </div>
        <LibraryBrowseTrackActions
          track={track}
          busy={busy}
          hasMatches={Boolean(matchPayload)}
          matchLoading={matchLoading}
          onPlay={onPlay}
          onReveal={onReveal}
          onMatch={onMatch}
        />
      </div>
      {matchPayload && renderMatches ? <div className="mt-2">{renderMatches(matchPayload)}</div> : null}
    </div>
  );
}

function LibraryBrowseTrackActions(props: {
  track: BrowseTrack;
  busy: boolean;
  hasMatches: boolean;
  matchLoading: boolean;
  onPlay: (track: BrowseTrack) => void;
  onReveal: (track: BrowseTrack) => void;
  onMatch: (track: BrowseTrack) => void;
}) {
  const model = trackActionModel(props);
  return (
    <div className="flex shrink-0 flex-wrap gap-2">
      <Button size="sm" variant="secondary" onClick={() => props.onPlay(props.track)}>
        <Play className="h-3.5 w-3.5" />
        Play
      </Button>
      <Button size="sm" variant="outline" onClick={() => props.onReveal(props.track)}>
        <FolderOpen className="h-3.5 w-3.5" />
        Show
      </Button>
      <Button size="sm" variant={model.matchVariant} disabled={model.matchDisabled} onClick={() => props.onMatch(props.track)}>
        <TrackMatchIcon loading={props.matchLoading} />
        {model.matchLabel}
      </Button>
    </div>
  );
}

type TrackActionModel = {
  matchVariant: "secondary" | "outline";
  matchDisabled: boolean;
  matchLabel: string;
};

function trackActionModel(props: { busy: boolean; hasMatches: boolean; matchLoading: boolean }): TrackActionModel {
  return {
    matchVariant: props.hasMatches ? "secondary" : "outline",
    matchDisabled: props.busy || props.matchLoading,
    matchLabel: props.hasMatches ? "Hide matches" : "Matches",
  };
}

function TrackMatchIcon({ loading }: { loading: boolean }) {
  return loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ListChecks className="h-3.5 w-3.5" />;
}

function browseTrackTitle(track: BrowseTrack): string {
  return track.title || track.path.split(/[\\/]/).pop() || track.track_id;
}

function browseTrackMeta(track: BrowseTrack): string {
  return trackMetaParts(track).filter(Boolean).join(" · ");
}

function trackMetaParts(track: BrowseTrack): string[] {
  return [artistAlbumGenreText(track), bpmText(track.bpm), durationText(track.duration_sec), missingText(track.missing)];
}

function artistAlbumGenreText(track: BrowseTrack): string {
  return [track.artist, track.album, track.genre].filter(Boolean).join(" · ") || "Unknown artist/album";
}

function bpmText(bpm: BrowseTrack["bpm"]): string {
  return typeof bpm === "number" ? `${Math.round(bpm)} BPM` : "";
}

function durationText(seconds: BrowseTrack["duration_sec"]): string {
  return typeof seconds === "number" ? formatDuration(seconds) : "";
}

function missingText(missing: BrowseTrack["missing"]): string {
  return missing ? "missing" : "";
}

function formatDuration(seconds: number): string {
  const safeSeconds = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds % 60;
  return `${minutes}:${remainder.toString().padStart(2, "0")}`;
}
