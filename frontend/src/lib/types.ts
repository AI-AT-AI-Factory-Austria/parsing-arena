export type Scores = {
  overall: number;
  tables: number;
  charts: number;
  content: number;
  formatting: number;
  grounding: number;
};

export type Availability = {
  runnable: boolean;
  backend: "local" | "cpu" | "endpoint" | "api" | null;
  reason: string | null;
  loaded: boolean | null;
};

export type ArenaModel = {
  id: string;
  name: string;
  publisher: string;
  params: string;
  params_b: number | null;
  license: string;
  kind: "open" | "reference";
  implemented: boolean;
  hf_repo: string | null;
  hf_url: string | null;
  revision: string | null;
  verified: boolean;
  lineage: string;
  pipeline: string;
  /** null for models that are not on the ParseBench leaderboard. */
  scores: Scores | null;
  availability: Availability;
};

export type ModelsResponse = {
  models: ArenaModel[];
  reference: ArenaModel;
  llamaparse_credits_per_page: number;
  llamaparse_usd_per_credit: number;
  source: string;
};

export type LocalDevice = {
  available: boolean;
  device?: string;
  device_name?: string;
  runtime?: string;
  memory_gb?: number;
  torch?: string;
  detail?: string;
};

export type Health = {
  status: string;
  version: string;
  llamaparse_configured: boolean;
  local_device: LocalDevice | null;
  max_upload_mb: number;
  max_pages_per_experiment: number;
};

export type DocumentInfo = {
  id: string;
  filename: string;
  media_type: string;
  size_bytes: number;
  page_count: number;
  created_at: string;
  thumbnail_url: string;
  file_url: string;
};

export type RunStatus = "queued" | "loading" | "running" | "completed" | "failed";
export type ExperimentStatus = "queued" | "running" | "completed" | "failed";

export type RunProgress = {
  phase?: "queued" | "loading" | "uploading" | "parsing" | "layout" | "recognition" | "done";
  page?: number;
  page_index?: number;
  regions_done?: number;
  regions_total?: number | null;
  pages_done?: number;
  last_page_seconds?: number;
  job_id?: string;
};

export type RunMeta = {
  backend?: string;
  device_name?: string;
  runtime?: string;
  dtype?: string;
  endpoint?: string;
  pipeline?: string;
  tier?: string;
  job_id?: string;
  credits?: number | null;
  [key: string]: unknown;
};

export type Run = {
  id: string;
  model_id: string;
  model_name: string;
  kind: "open" | "reference";
  status: RunStatus;
  started_at: string | null;
  finished_at: string | null;
  load_seconds: number | null;
  parse_seconds: number | null;
  seconds_per_page: number | null;
  pages_done: number;
  pages_total: number;
  progress: RunProgress;
  meta: RunMeta;
  error: string | null;
};

export type PageElement = {
  category: string;
  bbox: [number, number, number, number];
  content: string;
  layout_order: number;
};

export type PageResult = {
  page_number: number;
  markdown: string;
  seconds: number | null;
  elements: PageElement[] | null;
  error: string | null;
};

export type RunDetail = Run & {
  markdown: string | null;
  pages: PageResult[];
};

export type Experiment = {
  id: string;
  status: ExperimentStatus;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  pages: number[];
  error: string | null;
  queue_position: number | null;
  document: DocumentInfo;
  runs: Run[];
};

export type ExperimentDetail = Omit<Experiment, "runs"> & { runs: RunDetail[] };
