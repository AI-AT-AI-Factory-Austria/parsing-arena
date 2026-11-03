"use client";

import { useState } from "react";
import Link from "next/link";
import clsx from "clsx";
import { ChevronRight, History, Trash2, TriangleAlert } from "lucide-react";
import { api } from "@/lib/api";
import { formatRelative, formatSeconds, pluralize } from "@/lib/format";
import { useNow } from "@/lib/use-now";
import { usePolling } from "@/lib/use-polling";
import type { Experiment, Run } from "@/lib/types";
import { Spinner, StatusChip } from "@/components/ui";

function RunChip({ run }: { run: Run }) {
  const active = run.status === "loading" || run.status === "running";
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs",
        run.status === "failed" ? "border-danger/30 text-danger" : "border-line text-body",
      )}
      title={run.error ?? undefined}
    >
      {active && <Spinner className="size-3 text-brand" />}
      {run.status === "completed" && <span className="size-1.5 rounded-full bg-leaf" />}
      {run.status === "queued" && <span className="size-1.5 rounded-full bg-ash" />}
      {run.status === "failed" && <span className="size-1.5 rounded-full bg-danger" />}
      <span className="max-w-[11rem] truncate">{run.model_name}</span>
      {run.status === "completed" && run.seconds_per_page != null && (
        <span className="font-mono text-faint">{formatSeconds(run.seconds_per_page)}/p</span>
      )}
    </span>
  );
}

function ExperimentRow({ experiment, now, onDeleted }: {
  experiment: Experiment;
  now: number | null;
  onDeleted: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const busy = experiment.status === "queued" || experiment.status === "running";
  const doc = experiment.document;
  const pageLabel =
    experiment.pages.length === doc.page_count
      ? pluralize(doc.page_count, "page")
      : `${experiment.pages.length} of ${doc.page_count} pages`;

  const remove = async () => {
    setDeleting(true);
    setError(null);
    try {
      await api.deleteExperiment(experiment.id);
      onDeleted();
    } catch (err) {
      setError((err as Error).message);
      setDeleting(false);
      setConfirming(false);
    }
  };

  return (
    <li className="group relative flex flex-col gap-4 p-4 transition-colors hover:bg-smoke/70 sm:flex-row sm:items-center sm:gap-5 sm:px-6">
      <img
        src={doc.thumbnail_url}
        alt=""
        className="hidden h-16 w-12 shrink-0 rounded-lg border border-line bg-white object-cover object-top sm:block"
      />
      <div className="min-w-0 flex-1">
        <Link
          href={`/experiments/${experiment.id}`}
          className="truncate font-medium text-ink after:absolute after:inset-0 after:content-['']"
        >
          {doc.filename}
        </Link>
        <p className="mt-0.5 text-sm text-faint">
          {pageLabel} · {formatRelative(experiment.created_at, now)}
        </p>
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {experiment.runs.map((run) => (
            <RunChip key={run.id} run={run} />
          ))}
        </div>
        {error && <p className="mt-2 text-xs text-danger">{error}</p>}
      </div>
      <div className="relative z-10 flex items-center gap-2 self-start sm:self-center">
        <StatusChip
          status={experiment.status}
          label={
            experiment.status === "queued" && experiment.queue_position != null
              ? `Queued · #${experiment.queue_position + 1}`
              : undefined
          }
        />
        {!busy &&
          (confirming ? (
            <span className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => void remove()}
                disabled={deleting}
                className="btn btn-sm bg-danger text-white hover:bg-danger/90"
              >
                {deleting ? <Spinner className="size-3" /> : <Trash2 aria-hidden className="size-3" />}
                Delete
              </button>
              <button
                type="button"
                onClick={() => setConfirming(false)}
                className="btn btn-ghost btn-sm"
              >
                Cancel
              </button>
            </span>
          ) : (
            <button
              type="button"
              onClick={() => setConfirming(true)}
              aria-label={`Delete experiment for ${doc.filename}`}
              className="grid size-8 place-items-center rounded-full text-stone opacity-100 transition hover:bg-danger-wash hover:text-danger sm:opacity-0 sm:group-hover:opacity-100 sm:focus-visible:opacity-100"
            >
              <Trash2 aria-hidden className="size-4" />
            </button>
          ))}
        <ChevronRight aria-hidden className="size-5 text-stone transition group-hover:translate-x-0.5 group-hover:text-ink" />
      </div>
    </li>
  );
}

export function PastExperiments() {
  const now = useNow();
  const { data, error, loading, refresh } = usePolling(
    "experiments",
    () => api.listExperiments(),
    (d) => (d.experiments.some((e) => e.status === "queued" || e.status === "running") ? 2500 : null),
  );
  const experiments = data?.experiments ?? [];

  return (
    <section aria-labelledby="past-heading">
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="eyebrow">History</p>
          <h2 id="past-heading" className="mt-2 text-2xl tracking-tight sm:text-3xl">
            Past experiments
          </h2>
        </div>
        {experiments.length > 0 && (
          <p className="text-sm text-faint">{pluralize(experiments.length, "experiment")}</p>
        )}
      </div>

      <div className="card mt-5 overflow-hidden">
        {loading && !data && (
          <ul aria-busy="true" className="divide-y divide-line">
            {[0, 1, 2].map((i) => (
              <li key={i} className="flex animate-pulse items-center gap-5 px-6 py-5">
                <div className="h-16 w-12 rounded-lg bg-fog" />
                <div className="flex-1 space-y-2">
                  <div className="h-3 w-56 rounded bg-fog" />
                  <div className="h-3 w-32 rounded bg-fog" />
                </div>
              </li>
            ))}
          </ul>
        )}
        {!data && error && (
          <p className="flex items-center gap-2 px-6 py-6 text-sm text-danger">
            <TriangleAlert aria-hidden className="size-4" /> {error.message}
          </p>
        )}
        {data && experiments.length === 0 && (
          <div className="flex flex-col items-center px-6 py-14 text-center">
            <span className="grid size-12 place-items-center rounded-2xl bg-fog text-muted">
              <History aria-hidden className="size-5" />
            </span>
            <p className="mt-4 font-medium">No experiments yet</p>
            <p className="mt-1 text-sm text-muted">
              Runs you start appear here, so you can come back to their results later.
            </p>
          </div>
        )}
        {experiments.length > 0 && (
          <ul className="divide-y divide-line">
            {experiments.map((experiment) => (
              <ExperimentRow
                key={experiment.id}
                experiment={experiment}
                now={now}
                onDeleted={refresh}
              />
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
