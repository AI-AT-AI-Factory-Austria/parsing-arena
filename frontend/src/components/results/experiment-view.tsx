"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowLeft, FileQuestion, RotateCcw, Scale, Trash2, TriangleAlert, WifiOff } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { fileKind, formatDateTime, formatSeconds, pluralize } from "@/lib/format";
import { sortModelsBySize } from "@/lib/model-presentation";
import { useNow } from "@/lib/use-now";
import { usePolling } from "@/lib/use-polling";
import type { ArenaModel, ExperimentDetail } from "@/lib/types";
import { Chip, Spinner, StatusChip } from "@/components/ui";
import { JudgingConfirmation } from "@/components/judging-confirmation";
import { Comparison } from "./comparison";
import { RunBoard } from "./run-board";
import { isJudging } from "./judgment";

const POLL_MS = 1500;

function isActive(experiment: ExperimentDetail) {
  return experiment.status === "queued" || experiment.status === "running";
}

function summary(experiment: ExperimentDetail, now: number | null): string {
  const done = experiment.runs.filter((r) => r.status === "completed" || r.status === "failed");
  switch (experiment.status) {
    case "queued":
      return experiment.queue_position != null
        ? `Waiting in the queue (#${experiment.queue_position + 1})`
        : "Waiting to start";
    case "running": {
      const started = experiment.started_at ? new Date(experiment.started_at).getTime() : null;
      const running = started !== null && now !== null ? ` · ${formatSeconds((now - started) / 1000)}` : "";
      return `Running · ${done.length} of ${pluralize(experiment.runs.length, "model")} finished${running}`;
    }
    case "completed":
    case "failed": {
      if (!experiment.started_at || !experiment.finished_at) return "";
      const seconds =
        (new Date(experiment.finished_at).getTime() - new Date(experiment.started_at).getTime()) /
        1000;
      return `Finished in ${formatSeconds(seconds)}`;
    }
  }
}

export function ExperimentView({ id }: { id: string }) {
  const router = useRouter();
  const now = useNow();
  const { data: experiment, error, replace, refresh } = usePolling(
    id,
    () => api.experiment(id),
    (data) => (isActive(data) || data.runs.some(isJudging) ? POLL_MS : null),
  );
  const catalog = usePolling("models", () => api.models(), () => null);
  const [busy, setBusy] = useState<"rerun" | "delete" | "judge" | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [confirmJudging, setConfirmJudging] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState("");

  if (!experiment) {
    if (error instanceof ApiError && error.status === 404) {
      return (
        <div className="card mx-auto mt-6 flex max-w-xl flex-col items-center px-8 py-14 text-center">
          <span className="grid size-12 place-items-center rounded-2xl bg-fog text-muted">
            <FileQuestion aria-hidden className="size-5" />
          </span>
          <h1 className="mt-5 text-2xl tracking-tight">Experiment not found</h1>
          <p className="mt-2 text-muted">It may have been deleted.</p>
          <Link href="/arena" className="btn btn-primary mt-7">
            Back to the arena
          </Link>
        </div>
      );
    }
    if (error) {
      return (
        <p className="card mt-6 flex items-center gap-3 p-6 text-danger">
          <WifiOff aria-hidden className="size-5" /> {error.message}
        </p>
      );
    }
    return <ExperimentSkeleton />;
  }

  const orderedModels = catalog.data
    ? [catalog.data.reference, ...sortModelsBySize(catalog.data.models)]
    : [];
  const models = new Map<string, ArenaModel>(orderedModels.map((model) => [model.id, model]));
  const modelOrder = new Map(orderedModels.map((model, index) => [model.id, index]));
  // Match the picker in every results view, including saved runs and equal-size models.
  // Keep historical models that are no longer in the catalog at the end.
  const displayedExperiment = {
    ...experiment,
    runs: [...experiment.runs].sort((a, b) =>
      (modelOrder.get(a.model_id) ?? Number.MAX_SAFE_INTEGER)
      - (modelOrder.get(b.model_id) ?? Number.MAX_SAFE_INTEGER),
    ),
  };
  const doc = experiment.document;
  const active = isActive(experiment);
  const judging = experiment.runs.some(isJudging);
  const canJudge = experiment.judge_configured && experiment.runs.some(
    (r) => r.kind === "reference" && r.status === "completed",
  ) && experiment.runs.some(
    (r) => r.kind !== "reference" && r.status === "completed" && (!r.judgment || r.judgment.status === "failed" || r.judgment.outdated),
  );
  const allPages = experiment.pages.length === doc.page_count;
  const selectedRun = displayedExperiment.runs.find((run) => run.id === selectedRunId)
    ?? displayedExperiment.runs[0];

  const rerun = async () => {
    if (active || busy !== null) return;
    setBusy("rerun");
    setActionError(null);
    try {
      const next = await api.rerun(experiment.id);
      router.push(`/experiments/${next.id}`);
    } catch (err) {
      setActionError((err as Error).message);
      setBusy(null);
    }
  };

  const remove = async () => {
    setBusy("delete");
    setActionError(null);
    try {
      await api.deleteExperiment(experiment.id);
      router.push("/arena");
    } catch (err) {
      setActionError((err as Error).message);
      setBusy(null);
      setConfirmDelete(false);
    }
  };

  const judge = async () => {
    setBusy("judge");
    setActionError(null);
    try {
      replace(await api.judge(id));
      refresh();
    } catch (err) {
      setActionError((err as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-8">
      {confirmJudging && (
        <JudgingConfirmation
          filename={doc.filename}
          pageCount={experiment.pages.length}
          onCancel={() => setConfirmJudging(false)}
          onConfirm={() => {
            setConfirmJudging(false);
            void rerun();
          }}
        />
      )}
      <div>
        <Link
          href="/arena"
          className="inline-flex items-center gap-1.5 text-sm text-muted transition-colors hover:text-ink"
        >
          <ArrowLeft aria-hidden className="size-4" /> Arena
        </Link>
        <div className="mt-4 flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex min-w-0 items-center gap-5">
            <a
              href={doc.file_url}
              target="_blank"
              rel="noreferrer"
              title="Open the original file"
              className="hidden shrink-0 overflow-hidden rounded-xl border border-line bg-white sm:block"
            >
              <img src={doc.thumbnail_url} alt="" className="h-20 w-auto max-w-[64px] object-contain" />
            </a>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-3">
                <h1 className="min-w-0 max-w-full break-words text-3xl tracking-[-0.02em] sm:text-4xl">
                  {doc.filename}
                </h1>
                <StatusChip status={experiment.status} />
                {experiment.cpu_only && <Chip tone="outline">CPU only</Chip>}
              </div>
              <p className="mt-2 text-sm text-muted">
                {fileKind(doc.media_type)} ·{" "}
                {allPages
                  ? pluralize(doc.page_count, "page")
                  : `${pluralize(experiment.pages.length, "page")} of ${doc.page_count}`}{" "}
                · Started {formatDateTime(experiment.created_at)}
                {summary(experiment, now) && <> · {summary(experiment, now)}</>}
              </p>
            </div>
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-2">
            {canJudge && (
              <button type="button" onClick={() => void judge()} disabled={busy !== null} className="btn btn-outline btn-sm">
                {busy === "judge" ? <Spinner className="size-3.5" /> : <Scale aria-hidden className="size-3.5" />}
                Judge outputs
              </button>
            )}
            <button
              type="button"
              onClick={() => {
                if (experiment.runs.some((run) => run.kind === "reference")) {
                  setConfirmJudging(true);
                } else {
                  void rerun();
                }
              }}
              disabled={active || busy !== null}
              title={active ? "Available once this run has finished" : undefined}
              className="btn btn-outline btn-sm"
            >
              {busy === "rerun" ? (
                <Spinner className="size-3.5" />
              ) : (
                <RotateCcw aria-hidden className="size-3.5" />
              )}
              Run again
            </button>
            {confirmDelete ? (
              <>
                <button
                  type="button"
                  onClick={() => void remove()}
                  disabled={busy !== null || judging}
                  className="btn btn-sm bg-danger text-white hover:bg-danger/90"
                >
                  {busy === "delete" ? (
                    <Spinner className="size-3.5" />
                  ) : (
                    <Trash2 aria-hidden className="size-3.5" />
                  )}
                  Delete for good
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmDelete(false)}
                  className="btn btn-ghost btn-sm"
                >
                  Cancel
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={() => setConfirmDelete(true)}
                disabled={active || judging || busy !== null}
                title={active || judging ? "Wait until parsing and quality checks finish" : undefined}
                className="btn btn-ghost btn-sm"
              >
                <Trash2 aria-hidden className="size-3.5" /> Delete
              </button>
            )}
          </div>
        </div>
        {(actionError || experiment.error) && (
          <p
            role="alert"
            className="mt-4 flex items-start gap-2 rounded-2xl bg-danger-wash px-4 py-3 text-sm text-danger"
          >
            <TriangleAlert aria-hidden className="mt-0.5 size-4 shrink-0" />
            {actionError ?? experiment.error}
          </p>
        )}
        {error && (
          <p className="mt-4 flex items-center gap-2 text-sm text-amber">
            <WifiOff aria-hidden className="size-4" /> Lost the connection to the arena API.
            Retrying…
          </p>
        )}
      </div>

      <Comparison
        experiment={displayedExperiment}
        models={models}
        selectedRunId={selectedRun?.id ?? ""}
        onSelectRun={setSelectedRunId}
        onJudge={canJudge ? () => void judge() : undefined}
        judgeBusy={busy === "judge"}
      />

      <RunBoard
        experiment={displayedExperiment}
        models={models}
        now={now}
        selectedRunId={selectedRun?.id ?? ""}
        onSelectRun={(runId) => {
          setSelectedRunId(runId);
          document.getElementById("comparison")?.scrollIntoView({ block: "start" });
        }}
      />
    </div>
  );
}

export function ExperimentSkeleton() {
  return (
    <div className="space-y-8" aria-busy="true" aria-label="Loading experiment">
      <div className="animate-pulse space-y-4">
        <div className="h-4 w-16 rounded bg-fog" />
        <div className="h-9 w-80 max-w-full rounded-lg bg-fog" />
        <div className="h-4 w-96 max-w-full rounded bg-fog" />
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {[0, 1].map((i) => (
          <div key={i} className="card h-[420px] animate-pulse" />
        ))}
      </div>
      <div className="card h-56 animate-pulse" />
    </div>
  );
}
