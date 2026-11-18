"use client";

import { ArrowUpRight, Layers3 } from "lucide-react";
import { formatSeconds, pluralize } from "@/lib/format";
import type { ArenaModel, BatchDetail } from "@/lib/types";
import { Chip, StatusChip } from "@/components/ui";

export function BatchOverview({ batch, models, onModel, onDocument }: {
  batch: BatchDetail; models: Map<string, ArenaModel>;
  onModel: (id: string) => void; onDocument: (id: string) => void;
}) {
  const order = new Map([...models.keys()].map((id, i) => [id, i]));
  const rows = [...batch.aggregates].sort((a, b) => (order.get(a.model_id) ?? 99) - (order.get(b.model_id) ?? 99));
  const slowest = Math.max(0, ...rows.map((r) => r.seconds_per_page ?? 0));
  const judgingEnabled = rows.some((r) => r.kind === "reference");
  return <div className="space-y-6">
    <div className="card overflow-hidden">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-line px-5 py-5 sm:px-6">
        <div>
          <h2 className="flex items-center gap-2 text-xl tracking-tight"><Layers3 aria-hidden className="size-5 text-brand" />Across the whole batch</h2>
          <p className="mt-1 max-w-2xl text-sm leading-relaxed text-muted">Each page counts equally, across all {pluralize(batch.documents.length, "document")}.
            Results fill in as parsing and judging finish.</p>
        </div>
        <Chip tone="brand">{pluralize(batch.page_count, "page")} per model</Chip>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[900px] text-left text-sm">
          <thead className="eyebrow bg-smoke/60 text-[11px]"><tr>
            <th className="px-6 py-3 font-normal" scope="col">Model</th>
            <th className="px-4 py-3 font-normal" scope="col">Pages extracted</th>
            <th className="px-4 py-3 font-normal" scope="col">Average / page</th>
            <th className="px-4 py-3 text-right font-normal" scope="col">Total parse time</th>
            <th className="px-6 py-3 font-normal" scope="col">Quality across pages</th>
          </tr></thead>
          <tbody>{rows.map((row) => <tr key={row.model_id} className="border-t border-line">
            <th scope="row" className="max-w-[260px] px-6 py-5 font-normal">
              <button type="button" onClick={() => onModel(row.model_id)} className="text-left font-medium text-brand hover:underline">{row.model_name}<ArrowUpRight aria-hidden className="ml-1 inline size-3.5" /></button>
              <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-faint">
                <span>{models.get(row.model_id)?.params}</span>
                <StatusChip status={row.status} />
              </div>
            </th>
            <td className="px-4 py-5"><span className="font-mono tabular-nums">{row.pages_completed} / {row.pages_total}</span>
              {row.pages_failed > 0 && <p className="mt-1 text-xs text-danger">{pluralize(row.pages_failed, "page")} failed</p>}
            </td>
            <td className="min-w-[180px] px-4 py-5">
              <div className="flex max-w-[230px] items-center gap-3">
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-fog"><div className="h-full rounded-full bg-brand" style={{ width: `${row.seconds_per_page !== null && slowest ? Math.max(3, 100 * row.seconds_per_page / slowest) : 0}%` }} /></div>
                <span className="whitespace-nowrap font-mono tabular-nums">{formatSeconds(row.seconds_per_page)}</span>
              </div>
              {row.timed_pages > 0 && row.timed_pages < row.pages_total && <p className="mt-1 text-xs text-faint">Based on {row.timed_pages} / {row.pages_total} pages</p>}
            </td>
            <td className="px-4 py-5 text-right font-mono tabular-nums">{formatSeconds(row.parse_seconds)}</td>
            <td className="min-w-[280px] px-6 py-5">
              {row.kind === "reference" ? <Chip tone="mint">Reference</Chip> : row.judged_pages > 0 ? <>
                <div className="flex flex-wrap gap-1.5">
                  <Chip tone="mint">{row.excellent} Excellent</Chip>
                  <Chip tone="amber">{row.passed} Pass</Chip>
                  <Chip tone="danger">{row.failed} Fail</Chip>
                </div>
                <p className="mt-2 text-xs text-muted"><strong className="font-medium text-ink">{row.pass_rate}% pass rate</strong> · {row.judged_pages} / {row.pages_total} judged</p>
              </> : <span className="text-xs text-faint">{judgingEnabled ? (batch.judging || batch.status === "running" || batch.status === "queued" ? "Waiting for checks" : "No checks available") : "Judging not enabled"}</span>}
            </td>
          </tr>)}</tbody>
        </table>
      </div>
      <div className="border-t border-line bg-smoke/50 px-6 py-4 text-xs leading-relaxed text-muted">
        Average = total parsing time ÷ processed pages. Model loading is excluded; failed runs have no timing.
        Pass rate includes Excellent and Pass, using only pages with a current judgment. Unchecked pages are not treated as passes or failures.
      </div>
    </div>
    <div>
      <h2 className="text-lg tracking-tight">Inspect individual documents</h2>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {batch.experiments.map((experiment) => <button type="button" key={experiment.id} onClick={() => onDocument(experiment.id)} className="card flex items-center gap-4 p-4 text-left transition hover:border-brand/50">
          <img src={experiment.document.thumbnail_url} alt="" className="h-16 w-12 shrink-0 rounded-md border border-line object-cover object-top" />
          <div className="min-w-0 flex-1"><p className="truncate text-sm font-medium">{experiment.document.filename}</p><p className="mt-1 text-xs text-muted">{pluralize(experiment.pages.length, "selected page")}</p><div className="mt-2"><StatusChip status={experiment.status} /></div></div>
          <ArrowUpRight aria-hidden className="size-4 shrink-0 text-brand" />
        </button>)}
      </div>
    </div>
  </div>;
}
