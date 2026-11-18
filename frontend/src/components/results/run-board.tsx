"use client";

import clsx from "clsx";
import { formatSeconds } from "@/lib/format";
import type { ArenaModel, ExperimentDetail, Run } from "@/lib/types";
import { ProgressBar, StatusChip } from "@/components/ui";
import { runFraction, runPhaseText } from "./run-progress";
import { JudgmentSummary } from "./judgment";

function elapsed(run: Run, now: number | null): number | null {
  if (!run.started_at || now === null) return null;
  return Math.max(0, (now - new Date(run.started_at).getTime()) / 1000);
}

function backendText(run: Run): string | null {
  const m = run.meta;
  if (run.kind === "reference") return m.tier ? `LlamaParse API · ${m.tier}` : "LlamaParse API";
  if (m.cpu_only) return `CPU only${m.backend ? ` · ${m.backend}` : ""}`;
  if (m.backend === "endpoint") return "vLLM endpoint";
  if (m.backend === "pypdf") return "Text layer only · no model";
  if (m.backend === "tesseract") return `Tesseract ${m.version ?? ""} · ${m.lang ?? "eng"} · CPU`;
  if (m.backend === "docling") return `Docling Standard · CPU, ${m.threads ?? 4} threads`;
  if (m.backend === "transformers") {
    const device = m.device_name ? String(m.device_name).replace(/\(TM\)/g, "") : "local";
    return [device, m.runtime, m.dtype].filter(Boolean).join(" · ");
  }
  return null;
}

export function RunBoard({
  experiment,
  models,
  now,
  selectedRunId,
  onSelectRun,
}: {
  experiment: ExperimentDetail;
  models: Map<string, ArenaModel>;
  now: number | null;
  selectedRunId: string;
  onSelectRun: (id: string) => void;
}) {
  const runs = experiment.runs;
  const slowest = Math.max(
    0,
    ...runs.filter((r) => r.status === "completed").map((r) => r.seconds_per_page ?? 0),
  );

  return (
    <div className="card overflow-hidden">
      <div className="flex flex-col gap-1 border-b border-line px-5 py-4 sm:flex-row sm:items-baseline sm:justify-between sm:px-6">
        <h2 className="shrink-0 text-lg tracking-tight">Speed & quality</h2>
        <p className="max-w-3xl text-sm text-faint">
          Open models run one after another on the same hardware; model loading is not counted as
          parse time. Quality is checked per page against LlamaParse; Excellent and Pass count as passed.
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[820px] table-fixed text-left text-sm">
          <colgroup>
            <col className="w-[25%]" />
            <col className="w-[16%]" />
            <col className="w-[20%]" />
            <col className="w-[13%]" />
            <col className="w-[26%]" />
          </colgroup>
          <thead>
            <tr className="eyebrow text-[11px]">
              <th scope="col" className="py-3 pl-5 pr-4 font-normal sm:pl-6">
                Model
              </th>
              <th scope="col" className="py-3 pr-4 font-normal">
                Status
              </th>
              <th scope="col" className="py-3 pr-4 font-normal">
                Seconds per page
              </th>
              <th scope="col" className="py-3 pr-4 text-right font-normal">
                Parse time
              </th>
              <th scope="col" className="py-3 pl-5 pr-5 font-normal sm:pr-6">
                Quality across pages
              </th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => {
              const model = models.get(run.model_id);
              const fraction = runFraction(run);
              const active = run.status === "loading" || run.status === "running";
              const reference = run.kind === "reference";
              return (
                <tr
                  key={run.id}
                  className={clsx(
                    "border-t border-line align-middle",
                    run.id === selectedRunId && "bg-brand-wash",
                  )}
                >
                  <th scope="row" className="py-4 pl-5 pr-4 font-normal sm:pl-6">
                    <div className="flex flex-wrap items-center gap-2">
                      <button
                        type="button"
                        onClick={() => onSelectRun(run.id)}
                        aria-pressed={run.id === selectedRunId}
                        aria-controls="selected-model-output"
                        title={`View ${run.model_name} output`}
                        className="rounded text-left font-medium text-brand hover:underline"
                      >
                        {run.model_name}
                      </button>
                      {!reference && model && (
                        <span className="rounded-full bg-brand-wash px-2 py-px font-mono text-[11px] text-brand">
                          {model.params}
                        </span>
                      )}
                    </div>
                    {backendText(run) && (
                      <p className="mt-0.5 truncate text-xs text-faint" title={backendText(run) ?? undefined}>{backendText(run)}</p>
                    )}
                  </th>
                  <td className="py-4 pr-4">
                    {active || run.status === "queued" ? (
                      <div className="max-w-[220px]">
                        <div className="flex items-center justify-between gap-3 text-xs">
                          <span className="truncate text-body">
                            {runPhaseText(run, experiment.queue_position)}
                          </span>
                          {active && (
                            <span className="shrink-0 font-mono tabular-nums text-faint">
                              {formatSeconds(elapsed(run, now))}
                            </span>
                          )}
                        </div>
                        <ProgressBar
                          className="mt-2"
                          value={fraction ?? 0}
                          indeterminate={active && fraction === null}
                          tone={reference ? "mint" : "brand"}
                          label={`${run.model_name} progress`}
                        />
                      </div>
                    ) : (
                      <StatusChip
                        status={run.status}
                        title={run.error ?? undefined}
                        label={run.status === "failed" ? "Failed" : undefined}
                      />
                    )}
                    {run.status === "failed" && run.error && (
                      <p className="mt-1.5 max-w-[340px] text-xs leading-relaxed text-danger">
                        {run.error}
                      </p>
                    )}
                  </td>
                  <td className="py-4 pr-4">
                    {run.status === "completed" && run.seconds_per_page != null ? (
                      <div className="flex max-w-[240px] items-center gap-3">
                        <div className="h-2 flex-1 overflow-hidden rounded-full bg-fog">
                          <div
                            className={clsx(
                              "h-full rounded-full",
                              reference ? "bg-leaf" : "bg-brand",
                            )}
                            style={{
                              width: `${slowest ? Math.max(3, (run.seconds_per_page / slowest) * 100) : 0}%`,
                            }}
                          />
                        </div>
                        <span className="w-16 shrink-0 text-right font-mono tabular-nums text-ink">
                          {formatSeconds(run.seconds_per_page)}
                        </span>
                      </div>
                    ) : (
                      <span className="text-stone">–</span>
                    )}
                  </td>
                  <td className="py-4 pr-4 text-right font-mono tabular-nums text-body">
                    {run.status === "completed" ? formatSeconds(run.parse_seconds) : "–"}
                  </td>
                  <td className="py-4 pl-5 pr-5 sm:pr-6">
                    <button
                      type="button"
                      className="rounded-lg text-left"
                      onClick={() => onSelectRun(run.id)}
                      aria-label={`View quality feedback for ${run.model_name}`}
                      aria-controls="selected-model-output"
                    >
                      <JudgmentSummary run={run} />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
