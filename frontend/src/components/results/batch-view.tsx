"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import clsx from "clsx";
import { ArrowLeft, Layers3, RotateCcw, Trash2 } from "lucide-react";
import { api, ApiError, pageImageUrl } from "@/lib/api";
import { formatDateTime, pluralize } from "@/lib/format";
import { sortModelsBySize } from "@/lib/model-presentation";
import { usePolling } from "@/lib/use-polling";
import { useNow } from "@/lib/use-now";
import type { ArenaModel, BatchDetail } from "@/lib/types";
import { Chip, ProgressBar, Spinner, StatusChip } from "@/components/ui";
import { JudgingConfirmation } from "@/components/judging-confirmation";
import { BatchOverview } from "./batch-overview";
import { Comparison } from "./comparison";
import { ExperimentSkeleton } from "./experiment-view";
import type { PageChoice } from "./output-panel";
import { RunBoard } from "./run-board";

const active = (batch: BatchDetail) => ["queued", "running"].includes(batch.status);

export function BatchView({ id }: { id: string }) {
  const router = useRouter();
  const now = useNow();
  const { data: batch, error, refresh } = usePolling(id, () => api.batch(id), (d) => active(d) || d.judging ? 1500 : null);
  const catalog = usePolling("batch-models", () => api.models(), () => null);
  const [scope, setScope] = useState("overview");
  const [modelId, setModelId] = useState("");
  const [page, setPage] = useState<PageChoice>(1);
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [confirmJudging, setConfirmJudging] = useState<"run" | "judge" | null>(null);
  if (!batch) return error ? <div className="card p-8"><h1 className="text-2xl">{error instanceof ApiError && error.status === 404 ? "Batch not found" : "Could not load this batch"}</h1><p className="mt-2 text-muted">{error.message}</p><Link className="btn btn-primary mt-5" href="/arena">Back to the arena</Link></div> : <ExperimentSkeleton />;

  const ordered = catalog.data ? [catalog.data.reference, ...sortModelsBySize(catalog.data.models)] : [];
  const models = new Map<string, ArenaModel>(ordered.map((m) => [m.id, m]));
  const order = new Map(ordered.map((m, i) => [m.id, i]));
  const source = batch.experiments.find((e) => e.id === scope);
  const experiment = source ? { ...source, runs: [...source.runs].sort((a, b) => (order.get(a.model_id) ?? 99) - (order.get(b.model_id) ?? 99)) } : undefined;
  const selected = experiment?.runs.find((r) => r.model_id === modelId) ?? experiment?.runs[0];
  const hasReference = batch.aggregates.some((r) => r.kind === "reference");
  const canJudge = experiment?.runs.some((r) => r.kind === "reference" && r.status === "completed") && experiment.runs.some((r) => r.kind !== "reference" && r.status === "completed" && (!r.judgment || r.judgment.status === "failed" || r.judgment.outdated));
  const selectDocument = (experimentId: string) => {
    setScope(experimentId);
    setPage(batch.experiments.find((e) => e.id === experimentId)?.pages[0] ?? 1);
  };
  const rerun = async () => {
    setBusy("rerun"); setActionError(null);
    try { const next = await api.rerunBatch(id); router.push(`/batches/${next.id}`); }
    catch (err) { setActionError((err as Error).message); }
    finally { setBusy(null); }
  };
  const remove = async () => {
    setBusy("delete"); setActionError(null);
    try { await api.deleteBatch(id); router.push("/arena"); }
    catch (err) { setActionError((err as Error).message); setBusy(null); }
  };
  const judge = async () => {
    if (!experiment) return;
    setBusy("judge"); setActionError(null);
    try { await api.judge(experiment.id); refresh(); }
    catch (err) { setActionError((err as Error).message); }
    finally { setBusy(null); }
  };

  return <div className="space-y-7">
    {confirmJudging && <JudgingConfirmation
      filename={confirmJudging === "judge" && experiment ? experiment.document.filename : batch.documents.map((d) => d.filename).join(", ")}
      documentCount={confirmJudging === "judge" ? 1 : batch.documents.length}
      pageCount={confirmJudging === "judge" && experiment ? experiment.pages.length : batch.page_count}
      mode={confirmJudging}
      onKeysSaved={() => { catalog.refresh(); refresh(); }}
      onCancel={() => setConfirmJudging(null)}
      onConfirm={() => { setConfirmJudging(null); if (confirmJudging === "judge") void judge(); else void rerun(); }}
    />}
    <header>
      <Link href="/arena" className="inline-flex items-center gap-1.5 text-sm text-muted hover:text-ink"><ArrowLeft className="size-4" /> Arena</Link>
      <div className="mt-4 flex flex-wrap items-start justify-between gap-5">
        <div><div className="flex flex-wrap items-center gap-3"><h1 className="text-3xl tracking-tight sm:text-4xl">{batch.documents.length === 1 ? batch.documents[0].filename : "Batch results"}</h1><StatusChip status={batch.status} />{batch.cpu_only && <Chip tone="outline">CPU only</Chip>}</div>
          <p className="mt-2 text-sm text-muted">{pluralize(batch.documents.length, "document")} · {pluralize(batch.page_count, "page")} · {pluralize(batch.model_count, "model")} · {formatDateTime(batch.created_at)}</p></div>
        <div className="flex flex-wrap gap-2">
          <button type="button" className="btn btn-outline btn-sm" disabled={active(batch) || busy !== null} onClick={() => hasReference ? setConfirmJudging("run") : void rerun()}>{busy === "rerun" ? <Spinner /> : <RotateCcw className="size-3.5" />} Run batch again</button>
          <button type="button" className={clsx("btn btn-sm", confirmDelete ? "bg-danger text-white" : "btn-ghost")} disabled={active(batch) || batch.judging || busy !== null} onClick={() => confirmDelete ? void remove() : setConfirmDelete(true)}><Trash2 className="size-3.5" />{confirmDelete ? "Delete entire batch" : "Delete"}</button>
          {confirmDelete && <button className="btn btn-ghost btn-sm" type="button" onClick={() => setConfirmDelete(false)}>Cancel</button>}
        </div>
      </div>
      {(active(batch) || batch.judging) && <div className="mt-5 max-w-xl"><div className="mb-2 flex items-center justify-between text-xs text-muted"><span>{active(batch) ? "Parsing documents in sequence" : "Finishing quality checks"}</span><span>{batch.completed_runs} / {batch.total_runs} document–model runs finished</span></div><ProgressBar value={batch.total_runs ? batch.completed_runs / batch.total_runs : 0} indeterminate={!active(batch) && batch.judging} label="Batch progress" /></div>}
      {(actionError || error) && <p role="alert" className="mt-4 rounded-xl bg-danger-wash p-3 text-sm text-danger">{actionError ?? error?.message}</p>}
    </header>
    <nav className="flex flex-wrap gap-2 border-b border-line pb-4" aria-label="Result scope">
      <button type="button" onClick={() => setScope("overview")} aria-pressed={scope === "overview"} className={clsx("flex items-center gap-2 rounded-xl border px-4 py-3 text-sm transition", scope === "overview" ? "border-brand bg-brand text-white" : "border-line bg-white hover:border-ash")}><Layers3 className="size-4" /> Batch overview</button>
      {batch.experiments.map((e) => <button key={e.id} type="button" onClick={() => selectDocument(e.id)} aria-pressed={scope === e.id} title={e.document.filename} className={clsx("flex max-w-[300px] items-center gap-2 rounded-xl border px-3 py-2 text-left text-sm transition", scope === e.id ? "border-brand bg-brand-wash text-brand" : "border-line bg-white hover:border-ash")}><img src={e.document.thumbnail_url} alt="" className="h-8 w-6 rounded border border-line object-cover object-top" /><span className="truncate">{e.document.filename}<span className="mt-0.5 block text-[11px] text-muted">{pluralize(e.pages.length, "page")}</span></span></button>)}
    </nav>
    {!experiment ? <BatchOverview batch={batch} models={models} onDocument={selectDocument} onModel={(id) => { setModelId(id); selectDocument(batch.experiments.find((e) => e.runs.some((r) => r.model_id === id && r.status === "completed"))?.id ?? batch.experiments[0].id); }} /> : <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-2" aria-label="Document pages">
        <span className="mr-2 text-xs text-muted">Selected pages</span>
        {experiment.pages.map((n) => <button type="button" key={n} aria-pressed={page === n} onClick={() => setPage(n)} className={clsx("flex items-center gap-2 rounded-lg border p-1.5 pr-3 text-xs", page === n ? "border-brand bg-brand-wash text-brand" : "border-line bg-white")}><img src={`${pageImageUrl(experiment.document.id, n)}?thumbnail=true`} alt="" className="h-10 w-8 rounded object-contain" /> Page {n}</button>)}
      </div>
      <Comparison key={experiment.id} experiment={experiment} models={models} selectedRunId={selected?.id ?? ""} onSelectRun={(id) => setModelId(experiment.runs.find((r) => r.id === id)?.model_id ?? "")} onJudge={canJudge ? () => setConfirmJudging("judge") : undefined} judgeBusy={busy === "judge"} pageSelection={page} onPageChange={setPage} />
      <RunBoard experiment={experiment} models={models} now={now} selectedRunId={selected?.id ?? ""} onSelectRun={(id) => { setModelId(experiment.runs.find((r) => r.id === id)?.model_id ?? ""); document.getElementById("comparison")?.scrollIntoView({ block: "start" }); }} />
    </div>}
  </div>;
}
