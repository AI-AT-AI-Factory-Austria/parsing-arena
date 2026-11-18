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
  runtime_label?: string | null;
  execution_label?: string | null;
  execution_detail?: string | null;
  max_pages?: number | null;
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
  specialty: string;
  /** null for models that are not on the ParseBench leaderboard. */
  scores: Scores | null;
  availability: Availability;
};

export type ModelsResponse = {
  cpu_only: boolean;
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
  judge_configured: boolean;
  local_device: LocalDevice | null;
  runtime_profile?: string;
  runtime_profile_error?: string | null;
  max_upload_mb: number;
  max_pages_per_experiment: number;
};

export type JudgingKeyStatus = {
  llamaparse_configured: boolean;
  judge_configured: boolean;
  llamaparse_region: "us" | "eu";
};

export type DocumentInfo = {
  id: string;
  filename: string;
  media_type: string;
  size_bytes: number;
  page_count: number;
  sha256: string;
  created_at: string;
  thumbnail_url: string;
  file_url: string;
};

export type RunStatus = "queued" | "loading" | "running" | "completed" | "failed";
export type ExperimentStatus = "queued" | "running" | "completed" | "failed";

export type RunProgress = {
  phase?: "queued" | "loading" | "uploading" | "parsing" | "document" | "layout" | "recognition" | "reconstructing" | "done";
  pair_pages?: number[];
  pairs_done?: number;
  pairs_total?: number;
  page?: number;
  page_index?: number;
  regions_done?: number;
  regions_total?: number | null;
  pages_done?: number;
  last_page_seconds?: number;
  job_id?: string;
};

export type RunMeta = {
  cross_page_tables?: { source_pages: number[]; html: string }[];
  cross_page_reconstruction?: {
    status: "completed" | "partial" | "not_needed" | "disabled";
    pairs_checked?: number;
    pairs_total?: number;
    seconds?: number;
    issues?: { source_pages: number[]; message: string }[];
  };
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
  judgment: Judgment | null;
};

export type Verdict = "excellent" | "pass" | "fail";

export type JudgeIssue = {
  severity: "minor" | "major";
  category: string;
  message: string;
  reference_quote: string;
  candidate_quote: string;
};

export type PageJudgment = {
  page_number: number;
  verdict: Verdict;
  issues: JudgeIssue[];
};

export type Judgment = {
  status: "queued" | "running" | "completed" | "failed";
  verdict: Verdict | null;
  model: string;
  rubric_version: string;
  outdated: boolean;
  reference_run_id: string;
  pages_done: number;
  pages_total: number;
  pages: PageJudgment[];
  finished_at: string | null;
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
  batch_id: string | null;
  cpu_only: boolean;
  status: ExperimentStatus;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  pages: number[];
  error: string | null;
  queue_position: number | null;
  document: DocumentInfo;
  runs: Run[];
  judge_configured: boolean;
  judge_model: string;
};

export type ExperimentDetail = Omit<Experiment, "runs"> & { runs: RunDetail[] };

export type Batch = {
  id: string;
  created_at: string;
  status: ExperimentStatus;
  cpu_only: boolean;
  documents: DocumentInfo[];
  page_count: number;
  model_count: number;
  completed_runs: number;
  total_runs: number;
  judging: boolean;
};

export type ModelAggregate = {
  model_id: string;
  model_name: string;
  kind: "open" | "reference";
  status: ExperimentStatus;
  pages_total: number;
  pages_completed: number;
  pages_failed: number;
  timed_pages: number;
  parse_seconds: number | null;
  seconds_per_page: number | null;
  judged_pages: number;
  excellent: number;
  passed: number;
  failed: number;
  pass_rate: number | null;
};

export type BatchDetail = Batch & {
  experiments: ExperimentDetail[];
  aggregates: ModelAggregate[];
};

export type SampleDocument = {
  id: string;
  title: string;
  document: DocumentInfo;
  experiment_id: string | null;
  status: ExperimentStatus | "unprepared";
  model_count: number;
  completed_models: number;
  judging: boolean;
};
