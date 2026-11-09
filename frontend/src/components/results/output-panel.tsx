"use client";

import { useState } from "react";
import { Check, Copy, Download, TriangleAlert } from "lucide-react";
import { formatSeconds } from "@/lib/format";
import type { ArenaModel, RunDetail } from "@/lib/types";
import { Markdown } from "@/components/markdown";
import { Chip, ProgressBar, Spinner } from "@/components/ui";
import { runFraction, runPhaseText } from "./run-progress";
import { JudgmentFeedback } from "./judgment";
import { CrossPageTables } from "./cross-page-tables";

export type PageChoice = number | "all";
export type ViewMode = "rendered" | "markdown";

function contentFor(run: RunDetail, page: PageChoice) {
  if (page === "all") return { markdown: run.markdown ?? "", seconds: null, error: null };
  const result = run.pages.find((p) => p.page_number === page);
  if (!result) return null;
  return { markdown: result.markdown, seconds: result.seconds, error: result.error };
}

function PanelAction({
  label,
  onClick,
  children,
}: {
  label: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className="grid size-8 place-items-center rounded-full text-muted transition-colors hover:bg-fog hover:text-ink"
    >
      {children}
    </button>
  );
}

export function OutputPanel({
  run,
  model,
  page,
  view,
  filename,
  queuePosition,
  navigation,
  judgeNotice,
  onJudge,
  onSelectPage,
  judgeBusy,
}: {
  run: RunDetail;
  model: ArenaModel | undefined;
  page: PageChoice;
  view: ViewMode;
  filename: string;
  queuePosition: number | null;
  navigation: React.ReactNode;
  judgeNotice: string;
  onJudge?: () => void;
  onSelectPage: (page: PageChoice) => void;
  judgeBusy: boolean;
}) {
  const [copiedContent, setCopiedContent] = useState<string | null>(null);
  const contentKey = `${run.id}:${page}`;
  const copied = copiedContent === contentKey;
  const reference = run.kind === "reference";
  const content = run.status === "completed" ? contentFor(run, page) : null;

  const copy = async () => {
    if (!content) return;
    await navigator.clipboard.writeText(content.markdown);
    setCopiedContent(contentKey);
    setTimeout(() => setCopiedContent(null), 1500);
  };

  const download = () => {
    const blob = new Blob([run.markdown ?? ""], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${filename.replace(/\.[^.]+$/, "")}.${run.model_id}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section
      id="selected-model-output"
      aria-label={`${run.model_name} output`}
      className="card flex h-[calc(100dvh-14rem)] min-h-[520px] min-w-0 flex-col overflow-hidden"
    >
      <header className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-3">
        <div className="min-w-0 flex-1 basis-40">
          <div className="flex items-center gap-2">
            <h3 className="truncate font-medium">{run.model_name}</h3>
            {reference ? (
              <Chip tone="mint">Reference</Chip>
            ) : (
              model && (
                <span className="shrink-0 rounded-full bg-brand-wash px-2 py-px font-mono text-[11px] text-brand">
                  {model.params}
                </span>
              )
            )}
          </div>
          <p className="mt-0.5 text-xs text-faint">
            {run.status === "completed"
              ? content?.seconds != null
                ? `${formatSeconds(content.seconds)} for this page`
                : page === "all"
                  ? `${formatSeconds(run.parse_seconds)} in total`
                  : run.meta.processing_mode === "whole_document"
                    ? `${formatSeconds(run.parse_seconds)} total · ${run.pages_total} ${run.pages_total === 1 ? "page" : "pages"} processed together`
                  : reference
                    ? "Per-page time not reported by the API"
                    : "No timing for this page"
              : runPhaseText(run, queuePosition)}
            {run.meta.runtime === "Vulkan" && run.meta.format === "GGUF F16" && (
              <span> · AMD Vulkan · F16 GGUF</span>
            )}
            {run.meta.backend === "SkyVision.cpp" && (
              <span> · CPU · F16 GGUF</span>
            )}
            {run.meta.processing_mode === "whole_document" && (
              <span> · {String(run.meta.runtime)} · BF16 GGUF</span>
            )}
          </p>
        </div>
        {navigation}
        {content && (
          <div className="flex shrink-0 items-center">
            <PanelAction label={copied ? "Copied" : "Copy Markdown"} onClick={() => void copy()}>
              {copied ? (
                <Check aria-hidden className="size-4 text-leaf" />
              ) : (
                <Copy aria-hidden className="size-4" />
              )}
            </PanelAction>
            <PanelAction label="Download the whole document as Markdown" onClick={download}>
              <Download aria-hidden className="size-4" />
            </PanelAction>
          </div>
        )}
      </header>

      <JudgmentFeedback
        key={`judgment:${contentKey}`}
        run={run}
        page={page}
        notice={judgeNotice}
        onJudge={onJudge}
        onSelectPage={onSelectPage}
        busy={judgeBusy}
      />

      <div key={`output:${contentKey}`} className="min-h-0 flex-1 overflow-auto px-5 py-4">
        {run.status === "completed" && <CrossPageTables run={run} page={page} view={view} onSelectPage={onSelectPage} />}
        {run.status === "failed" && (
          <div className="flex items-start gap-2.5 rounded-2xl bg-danger-wash p-4 text-sm text-danger">
            <TriangleAlert aria-hidden className="mt-0.5 size-4 shrink-0" />
            <div>
              <p className="font-medium">This run failed.</p>
              <p className="mt-1 leading-relaxed">{run.error}</p>
            </div>
          </div>
        )}
        {(run.status === "queued" || run.status === "loading" || run.status === "running") && (
          <div className="flex h-full flex-col items-center justify-center text-center">
            {run.status === "queued" ? (
              <span className="size-2 rounded-full bg-ash" />
            ) : (
              <Spinner className="size-5 text-brand" />
            )}
            <p className="mt-3 text-sm text-muted">{runPhaseText(run, queuePosition)}</p>
            {run.status !== "queued" && (
              <ProgressBar
                className="mt-4 w-40"
                value={runFraction(run) ?? 0}
                indeterminate={runFraction(run) === null}
                tone={reference ? "mint" : "brand"}
                label={`${run.model_name} progress`}
              />
            )}
          </div>
        )}
        {run.status === "completed" && !content && (
          <p className="text-sm text-muted">This page was not part of the output.</p>
        )}
        {content?.error && (
          <p className="mb-3 flex items-start gap-2 rounded-xl bg-amber-wash px-3 py-2 text-xs text-amber">
            <TriangleAlert aria-hidden className="mt-px size-3.5 shrink-0" /> {content.error}
          </p>
        )}
        {content &&
          !content.error &&
          (content.markdown.trim() === "" ? (
            <p className="text-sm text-muted">The model returned no text for this page.</p>
          ) : view === "rendered" ? (
            <Markdown markdown={content.markdown} runId={run.id} />
          ) : (
            <pre className="whitespace-pre-wrap break-words font-mono text-[12.5px] leading-relaxed text-body">
              {content.markdown}
            </pre>
          ))}
      </div>
    </section>
  );
}
