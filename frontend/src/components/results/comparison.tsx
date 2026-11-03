"use client";

import { useEffect, useEffectEvent, useState } from "react";
import clsx from "clsx";
import { ChevronLeft, ChevronRight, ExternalLink } from "lucide-react";
import { pageImageUrl } from "@/lib/api";
import type { ArenaModel, ExperimentDetail } from "@/lib/types";
import { OutputPanel, type PageChoice, type ViewMode } from "./output-panel";
import { CategoryLegend, PagePreview } from "./page-preview";

const ORIGINAL = "original";

function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-full bg-fog p-1 text-sm">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={value === option.value}
          onClick={() => onChange(option.value)}
          className={clsx(
            "rounded-full px-3.5 py-1.5 transition-colors",
            value === option.value ? "bg-white text-ink shadow-sm" : "text-muted hover:text-ink",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Comparison({
  experiment,
  models,
}: {
  experiment: ExperimentDetail;
  models: Map<string, ArenaModel>;
}) {
  const pages = experiment.pages;
  const [page, setPage] = useState<PageChoice>(pages[0] ?? 1);
  const [view, setView] = useState<ViewMode>("rendered");
  const [hidden, setHidden] = useState<string[]>([]);
  const [layoutRunId, setLayoutRunId] = useState<string>("");

  const index = page === "all" ? -1 : pages.indexOf(page);
  const go = (delta: number) => {
    if (index < 0) return;
    const next = pages[index + delta];
    if (next !== undefined) setPage(next);
  };
  const onArrow = useEffectEvent(go);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable]")) return;
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      if (event.key === "ArrowLeft") onArrow(-1);
      if (event.key === "ArrowRight") onArrow(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const panels = [
    { id: ORIGINAL, label: "Original page" },
    ...experiment.runs.map((run) => ({ id: run.id, label: run.model_name })),
  ];
  const visible = panels.filter((p) => !hidden.includes(p.id));
  const layoutRuns = experiment.runs.filter(
    (r) => r.kind === "open" && r.status === "completed" && r.pages.some((p) => p.elements?.length),
  );
  const layoutRun = layoutRuns.find((r) => r.id === layoutRunId);
  const elements =
    layoutRun && page !== "all"
      ? (layoutRun.pages.find((p) => p.page_number === page)?.elements ?? null)
      : null;

  return (
    <section aria-labelledby="compare-heading" className="space-y-4">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
        <div>
          <h2 id="compare-heading" className="text-2xl tracking-tight">
            Compare the outputs
          </h2>
          <p className="mt-1 text-sm text-muted">
            The original page next to every model&apos;s output. Use ← and → to flip pages.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1 rounded-full bg-white p-1 shadow-card">
            <button
              type="button"
              onClick={() => go(-1)}
              disabled={index <= 0}
              aria-label="Previous page"
              className="grid size-8 place-items-center rounded-full text-ink transition-colors hover:bg-fog disabled:text-ash disabled:hover:bg-transparent"
            >
              <ChevronLeft aria-hidden className="size-4" />
            </button>
            <label htmlFor="page-select" className="sr-only">
              Page
            </label>
            <select
              id="page-select"
              value={String(page)}
              onChange={(e) =>
                setPage(e.target.value === "all" ? "all" : Number(e.target.value))
              }
              className="cursor-pointer rounded-full bg-transparent px-2 py-1 text-sm outline-none"
            >
              {pages.map((p, i) => (
                <option key={p} value={p}>
                  Page {p} ({i + 1} of {pages.length})
                </option>
              ))}
              {pages.length > 1 && <option value="all">All pages</option>}
            </select>
            <button
              type="button"
              onClick={() => go(1)}
              disabled={index < 0 || index >= pages.length - 1}
              aria-label="Next page"
              className="grid size-8 place-items-center rounded-full text-ink transition-colors hover:bg-fog disabled:text-ash disabled:hover:bg-transparent"
            >
              <ChevronRight aria-hidden className="size-4" />
            </button>
          </div>
          <Segmented
            label="Output view"
            value={view}
            onChange={setView}
            options={[
              { value: "rendered", label: "Rendered" },
              { value: "markdown", label: "Markdown" },
            ]}
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Visible panels">
        <span className="eyebrow mr-1 text-[11px]">Show</span>
        {panels.map((panel) => {
          const on = !hidden.includes(panel.id);
          return (
            <button
              key={panel.id}
              type="button"
              aria-pressed={on}
              onClick={() =>
                setHidden(on ? [...hidden, panel.id] : hidden.filter((h) => h !== panel.id))
              }
              disabled={on && visible.length === 1}
              className={clsx(
                "rounded-full border px-3 py-1 text-xs transition-colors disabled:cursor-not-allowed",
                on
                  ? "border-ink bg-ink text-white"
                  : "border-line bg-white text-muted hover:border-ash hover:text-ink",
              )}
            >
              {panel.label}
            </button>
          );
        })}
      </div>

      <div className="overflow-x-auto pb-2">
        <div
          className="grid gap-4"
          style={{ gridTemplateColumns: `repeat(${visible.length}, minmax(340px, 1fr))` }}
        >
          {visible.map((panel) => {
            if (panel.id === ORIGINAL) {
              return (
                <section
                  key={panel.id}
                  aria-label="Original page"
                  className="card flex h-[calc(100dvh-14rem)] min-h-[520px] flex-col overflow-hidden"
                >
                  <header className="flex items-center gap-2 border-b border-line px-4 py-3">
                    <div className="min-w-0 flex-1">
                      <h3 className="font-medium">Original</h3>
                      <p className="mt-0.5 text-xs text-faint">
                        {page === "all" ? `${pages.length} pages` : `Page ${page}`}
                      </p>
                    </div>
                    {layoutRuns.length > 0 && page !== "all" && (
                      <>
                        <label htmlFor="layout-select" className="sr-only">
                          Layout boxes
                        </label>
                        <select
                          id="layout-select"
                          value={layoutRunId}
                          onChange={(e) => setLayoutRunId(e.target.value)}
                          className="max-w-[11rem] cursor-pointer truncate rounded-full border border-line bg-white px-3 py-1.5 text-xs outline-none hover:border-ash"
                        >
                          <option value="">No layout boxes</option>
                          {layoutRuns.map((r) => (
                            <option key={r.id} value={r.id}>
                              Layout: {r.model_name}
                            </option>
                          ))}
                        </select>
                      </>
                    )}
                    {page !== "all" && (
                      <a
                        href={pageImageUrl(experiment.document.id, page)}
                        target="_blank"
                        rel="noreferrer"
                        title="Open the page image"
                        aria-label="Open the page image in a new tab"
                        className="grid size-8 shrink-0 place-items-center rounded-full text-muted hover:bg-fog hover:text-ink"
                      >
                        <ExternalLink aria-hidden className="size-4" />
                      </a>
                    )}
                  </header>
                  <div className="flex-1 space-y-4 overflow-auto bg-smoke p-4">
                    {(page === "all" ? pages : [page]).map((p) => (
                      <figure key={p}>
                        <PagePreview
                          src={pageImageUrl(experiment.document.id, p)}
                          alt={`Page ${p} of ${experiment.document.filename}`}
                          elements={page === "all" ? null : elements}
                        />
                        {page === "all" && (
                          <figcaption className="mt-1.5 text-center text-xs text-faint">
                            Page {p}
                          </figcaption>
                        )}
                      </figure>
                    ))}
                    {elements && elements.length > 0 && <CategoryLegend elements={elements} />}
                  </div>
                </section>
              );
            }
            const run = experiment.runs.find((r) => r.id === panel.id)!;
            return (
              <OutputPanel
                key={panel.id}
                run={run}
                model={models.get(run.model_id)}
                page={page}
                view={view}
                filename={experiment.document.filename}
                queuePosition={experiment.queue_position}
              />
            );
          })}
        </div>
      </div>
    </section>
  );
}
