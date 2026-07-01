export type SearchResult = {
  track_id: string;
  path: string;
  title?: string | null;
  artist?: string | null;
  album?: string | null;
  genre?: string | null;
  score?: number;
  raw_score?: number;
  confidence?: "high" | "medium" | "low" | string | null;
  warnings?: string[];
  rank_reason?: {
    mode?: string;
    primary?: string;
    summary?: string;
    signals?: string[];
    components?: Array<{ name?: string; label?: string; score?: number }>;
    sort?: {
      field?: string;
      label?: string;
      direction?: string;
      source?: string;
      value?: number | null;
    };
  } | null;
  candidate_evidence?: {
    retrieved_by?: string[];
    sources?: Array<{
      source?: string;
      role?: string;
      score?: number;
      matched?: boolean;
      details?: Record<string, unknown>;
    }>;
    identity?: Record<string, boolean>;
    semantic_only?: boolean;
  } | null;
  why?: string[];
  scores?: Record<string, number>;
  matched_tags?: Array<{ tag: string; score: number; source: string }>;
  saved_feedback_rating?: "good" | "bad" | "neutral" | null;
};

export type SearchFeatureRange = {
  low?: number;
  high?: number;
  source?: string;
};

export type SearchSortSpec = {
  field?: string;
  direction?: string;
  source?: string;
};

export type SearchSuggestion = {
  query: string;
  confidence?: number;
  reason?: string;
  kind?: "correction" | "example" | string;
  type?: string;
  label?: string;
};

export type SearchTypeExample = {
  type?: string;
  label?: string;
  query?: string | null;
  reason?: string;
};

export type SearchNotice = {
  level?: "info" | "warning" | "error" | string;
  message?: string;
};

export type SearchLlmHints = {
  contexts?: string[];
  prefer_tag_concepts?: string[];
  avoid_tag_concepts?: string[];
  feature_preferences?: Record<string, string>;
  sort_by?: SearchSortSpec[];
  limit?: number | null;
  notes?: string | null;
};

export type SearchIntentSummary = {
  limit?: number;
  contexts?: string[];
  prefer_tags?: string[];
  avoid_tags?: string[];
  feature_ranges?: Record<string, SearchFeatureRange>;
  sort_by?: SearchSortSpec[];
  semantic_model?: string | null;
  use_semantic?: boolean;
};

export type SearchPayload = {
  query?: string;
  parser?: string;
  llm_error?: string | null;
  llm_hints?: SearchLlmHints | null;
  intent?: SearchIntentSummary;
  diagnostics?: Record<string, unknown>;
  results?: SearchResult[];
};

export type BrowseRoot = {
  path: string;
  name?: string;
  track_count?: number;
};

export type BrowseFolder = {
  name: string;
  path: string;
  track_count?: number;
};

export type BrowseTrack = SearchResult & {
  bpm?: number | null;
  duration_sec?: number | null;
  missing?: boolean;
};

export type BrowsePayload = {
  db_path?: string;
  roots?: BrowseRoot[];
  cwd?: string | null;
  root?: string | null;
  parent?: string | null;
  folders?: BrowseFolder[];
  tracks?: BrowseTrack[];
  track_count?: number;
  limit?: number;
  offset?: number;
  has_more?: boolean;
  mode?: "browse" | "search" | string;
  query?: string;
  sort?: string;
  sort_direction?: "asc" | "desc" | string;
  include_missing?: boolean;
  warning?: string | null;
};

export type MatchTrackInfo = {
  track_id?: string;
  title?: string | null;
  artist?: string | null;
  album?: string | null;
  path?: string;
};

export type MatchEvidenceItem = {
  source?: string;
  role?: string;
  status?: string;
  score?: number;
  decisive?: boolean;
  details?: Record<string, unknown>;
};

export type MatchReport = {
  decision?: string;
  identity_decision?: string;
  confidence?: string;
  confidence_score?: number;
  candidate_score?: number;
  candidate_kind?: string;
  candidate_strength?: "strong" | "medium" | "weak" | string;
  candidate_summary?: string;
  candidate_reasons?: string[];
  candidate_scores?: Record<string, number>;
  reasons?: string[];
  warnings?: string[];
  evidence?: MatchEvidenceItem[];
  track_a?: MatchTrackInfo;
  track_b?: MatchTrackInfo;
};

export type MatchTrackPayload = {
  track_id?: string;
  against_library?: boolean;
  count?: number;
  reports?: MatchReport[];
};
