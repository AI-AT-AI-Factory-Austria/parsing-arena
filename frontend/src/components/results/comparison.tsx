"use client";

import { useEffect, useEffectEvent, useRef, useState } from "react";
import clsx from "clsx";
import { ChevronLeft, ChevronRight, ExternalLink } from "lucide-react";
import { pageImageUrl } from "@/lib/api";
import type { ArenaModel, ExperimentDetail } from "@/lib/types";
import { OutputPanel, type PageChoice, type ViewMode } from "./output-panel";
import { CategoryLegend, PagePreview } from "./page-preview";
import { judgeNotice } from "./judgment";

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
  selectedRunId,
  onSelectRun,
  onJudge,
  judgeBusy,
  pageSelection,
  onPageChange,
}: {
  experiment: ExperimentDetail;
  models: Map<string, ArenaModel>;
  selectedRunId: string;
  onSelectRun: (id: string) => void;
  onJudge?: () => void;
  judgeBusy: boolean;
  pageSelection?: PageChoice;
  onPageChange?: (page: PageChoice) => void;
}) {
  const pages = experiment.pages;
  const sectionRef = useRef<HTMLElement>(null);
  const [localPage, setLocalPage] = useState<PageChoice>(pages[0] ?? 1);
  const page = pageSelection ?? localPage;
  const setPage = onPageChange ?? setLocalPage;
  const [view, setView] = useState<ViewMode>("rendered");
  const [showLayout, setShowLayout] = useState(false);
  const selectedIndex = Math.max(0, experiment.runs.findIndex((r) => r.id === selectedRunId));
  const selectedRun = experiment.runs[selectedIndex];

  const index = page === "all" ? -1 : pages.indexOf(page);
  const go = (delta: number) => {
    if (index < 0) return;
    const next = pages[index + delta];
    if (next !== undefined) setPage(next);
  };
  const goModel = (delta: number) => {
    const next = experiment.runs[selectedIndex + delta];
    if (next) onSelectRun(next.id);
  };
  const onArrow = useEffectEvent((event: KeyboardEvent) => {
    if (!sectionRef.current?.getClientRects().length || experiment.runs.length < 2) return;
    event.preventDefault();
    goModel(event.key === "ArrowLeft" ? -1 : 1);
  });

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable], [role='radio']")) return;
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
        return;
      }
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") onArrow(event);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const elements =
    selectedRun && page !== "all"
      ? (selectedRun.pages.find((p) => p.page_number === page)?.elements ?? null)
      : null;

  return (
    <section ref={sectionRef} id="comparison" aria-labelledby="compare-heading" className="space-y-4">
      <div>
        <h2 id="compare-heading" className="text-2xl tracking-tight">
          Compare the outputs
        </h2>
        <p className="mt-1 text-sm text-muted">
          Pick a model to compare with the original. Use ← and → to switch models.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Model results">
        <span className="eyebrow mr-1 text-[11px]">Model</span>
        {experiment.runs.map((run) => {
          const selected = run.id === selectedRun?.id;
          return (
            <button
              key={run.id}
              type="button"
              aria-pressed={selected}
              aria-controls="selected-model-output"
              onClick={() => onSelectRun(run.id)}
              className={clsx(
                "rounded-full border px-3.5 py-2 text-sm transition-colors",
                selected
                  ? "border-ink bg-ink text-white"
                  : "border-line bg-white text-muted hover:border-ash hover:text-ink",
              )}
            >
              {run.model_name}
            </button>
          );
        })}
      </div>

      <div className="flex xl:justify-end">
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

      <div className="grid min-w-0 gap-4 md:grid-cols-2">
        <section
          aria-label="Original page"
          className="card flex h-[calc(100dvh-14rem)] min-h-[520px] min-w-0 flex-col overflow-hidden"
        >
          <header className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-3">
            <div className="min-w-0 flex-1">
              <h3 className="font-medium">Original</h3>
              <p className="mt-0.5 text-xs text-faint">
                {page === "all" ? `${pages.length} pages` : `Page ${page}`}
              </p>
            </div>
            {page !== "all" && (
              <>
                <label
                  className="flex items-center gap-2 text-xs text-muted"
                  title="Show the selected model's detected layout"
                >
                  <input
                    type="checkbox"
                    checked={showLayout}
                    disabled={!elements?.length}
                    onChange={(e) => setShowLayout(e.target.checked)}
                    className="size-3.5 accent-brand"
                  />
                  Layout boxes
                </label>
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
              </>
            )}
          </header>
          <div className="min-h-0 flex-1 space-y-4 overflow-auto bg-smoke p-4">
            {(page === "all" ? pages : [page]).map((p) => (
              <figure key={p}>
                <PagePreview
                  src={pageImageUrl(experiment.document.id, p)}
                  alt={`Page ${p} of ${experiment.document.filename}`}
                  elements={showLayout ? elements : null}
                />
                {page === "all" && (
                  <figcaption className="mt-1.5 text-center text-xs text-faint">
                    Page {p}
                  </figcaption>
                )}
              </figure>
            ))}
            {showLayout && elements && elements.length > 0 && <CategoryLegend elements={elements} />}
          </div>
        </section>
        {selectedRun ? (
          <OutputPanel
            run={selectedRun}
            model={models.get(selectedRun.model_id)}
            page={page}
            view={view}
            filename={experiment.document.filename}
            queuePosition={experiment.queue_position}
            judgeNotice={judgeNotice(experiment, selectedRun)}
            onJudge={selectedRun.status === "completed" ? onJudge : undefined}
            onSelectPage={setPage}
            judgeBusy={judgeBusy}
            navigation={
              <div
                className="flex shrink-0 items-center gap-1 rounded-full border border-line p-1"
                role="group"
                aria-label="Switch model"
              >
                <button
                  type="button"
                  onClick={() => goModel(-1)}
                  disabled={selectedIndex === 0}
                  aria-label="Previous model"
                  title="Previous model (←)"
                  className="grid size-8 place-items-center rounded-full text-ink hover:bg-fog disabled:text-ash disabled:hover:bg-transparent"
                >
                  <ChevronLeft aria-hidden className="size-4" />
                </button>
                <span
                  className="min-w-12 text-center text-xs tabular-nums text-muted"
                  aria-live="polite"
                  aria-atomic="true"
                >
                  {selectedIndex + 1} / {experiment.runs.length}
                  <span className="sr-only"> · {selectedRun.model_name}</span>
                </span>
                <button
                  type="button"
                  onClick={() => goModel(1)}
                  disabled={selectedIndex === experiment.runs.length - 1}
                  aria-label="Next model"
                  title="Next model (→)"
                  className="grid size-8 place-items-center rounded-full text-ink hover:bg-fog disabled:text-ash disabled:hover:bg-transparent"
                >
                  <ChevronRight aria-hidden className="size-4" />
                </button>
              </div>
            }
          />
        ) : (
          <div
            id="selected-model-output"
            className="card grid min-h-[520px] place-items-center p-6 text-sm text-muted"
          >
            No model runs in this experiment.
          </div>
        )}
      </div>
    </section>
  );
}
