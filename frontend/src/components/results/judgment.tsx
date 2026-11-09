"use client";

import { CheckCheck, CircleCheck, CircleX, Scale } from "lucide-react";
import type { ExperimentDetail, Run, Verdict } from "@/lib/types";
import { Chip, Spinner } from "@/components/ui";
import type { PageChoice } from "./output-panel";

const VERDICTS = {
  excellent: { label: "Excellent", tone: "mint", Icon: CheckCheck },
  pass: { label: "Pass", tone: "amber", Icon: CircleCheck },
  fail: { label: "Fail", tone: "danger", Icon: CircleX },
} as const;

export function isJudging(run: Run) {
  return run.judgment?.status === "queued" || run.judgment?.status === "running";
}

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  const { label, tone, Icon } = VERDICTS[verdict];
  return <Chip tone={tone}><Icon aria-hidden className="size-3.5" />{label}</Chip>;
}

export function JudgmentBadge({ run }: { run: Run }) {
  if (run.kind === "reference") return <Chip tone="outline">Reference</Chip>;
  const judgment = run.judgment;
  if (judgment?.status === "completed" && judgment.verdict) {
    return <VerdictBadge verdict={judgment.verdict} />;
  }
  if (isJudging(run)) {
    return (
      <Chip tone="brand">
        <Spinner className="size-3" />
        {judgment?.status === "queued" ? "Queued" : `Checking ${judgment?.pages_done}/${judgment?.pages_total}`}
      </Chip>
    );
  }
  return (
    <Chip tone="neutral" title={judgment?.error ?? undefined}>
      {judgment?.status === "failed" ? "Check unavailable" : run.status === "failed" ? "No output" : "Not judged"}
    </Chip>
  );
}

export function judgeNotice(experiment: ExperimentDetail, run: Run) {
  if (!experiment.judge_configured) return "Quality checks are not configured on this server.";
  const reference = experiment.runs.find((r) => r.kind === "reference");
  if (!reference) return "Include LlamaParse in a new experiment to enable quality checks.";
  if (reference.status === "failed") return "The LlamaParse reference failed, so quality cannot be checked.";
  if (reference.status !== "completed") return "Waiting for the LlamaParse reference to finish.";
  if (run.status === "failed") return "This model produced no usable output to judge.";
  if (run.status !== "completed") return "Quality will be checked when this model finishes.";
  return "Compare this extraction with the LlamaParse reference.";
}

export function JudgmentFeedback({
  run, page, notice, onJudge, onSelectPage, busy,
}: {
  run: Run;
  page: PageChoice;
  notice: string;
  onJudge?: () => void;
  onSelectPage: (page: PageChoice) => void;
  busy: boolean;
}) {
  if (run.kind === "reference") {
    return <p className="border-b border-line bg-mint-wash px-4 py-3 text-xs text-leaf-strong">LlamaParse reference · Used as the baseline for quality checks.</p>;
  }
  const judgment = run.judgment;
  const selectedPage = page === "all" ? null : judgment?.pages.find((p) => p.page_number === page);
  const verdict = page === "all" ? judgment?.verdict : selectedPage?.verdict;
  const pages = page === "all" ? judgment?.pages ?? [] : selectedPage ? [selectedPage] : [];
  const findings = pages.flatMap((p) => p.issues.map((issue) => ({ ...issue, page: p.page_number })));
  const modelName = judgment?.model.startsWith("mistral-large-4") ? "Mistral Large 4" : judgment?.model;

  return (
    <aside aria-label="Extraction quality" className="max-h-56 shrink-0 overflow-y-auto border-b border-line bg-smoke/70 px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Scale aria-hidden className="size-4 text-muted" />
          Quality {page === "all" ? "· All pages" : `· Page ${page}`}
        </div>
        {verdict ? <VerdictBadge verdict={verdict} /> : <JudgmentBadge run={run} />}
      </div>
      {isJudging(run) && (
        <p role="status" className="mt-2 text-xs text-muted">
          {judgment?.status === "queued" ? "Waiting for the judge…" : `Comparing with LlamaParse · ${judgment?.pages_done} of ${judgment?.pages_total} pages checked.`}
        </p>
      )}
      {judgment?.status === "failed" && <p className="mt-2 text-xs text-muted">{judgment.error}</p>}
      {!judgment && <p className="mt-2 text-xs text-muted">{notice}</p>}
      {verdict === "excellent" && <p className="mt-2 text-xs text-muted">No discrepancies found against the reference.</p>}
      {findings.length > 0 && (
        <ul className="mt-2 list-disc space-y-1.5 pl-4 text-sm leading-relaxed text-body">
          {findings.slice(0, 3).map((issue, i) => (
            <li key={i}>
              {page === "all" && <button type="button" className="mr-1 text-brand hover:underline" onClick={() => onSelectPage(issue.page)}>Page {issue.page}:</button>}
              {issue.message}
            </li>
          ))}
        </ul>
      )}
      {findings.length > 0 && (
        <details className="mt-2 text-xs text-muted">
          <summary className="w-fit cursor-pointer rounded hover:text-ink">
            {findings.length > 3 ? `All ${findings.length} findings and evidence` : "View evidence"}
          </summary>
          <ul className="mt-2 space-y-3">
            {findings.map((issue, i) => (
              <li key={i} className="rounded-xl border border-line bg-white p-3">
                <p className="font-medium text-body">
                  <button type="button" className="text-brand hover:underline" onClick={() => onSelectPage(issue.page)}>Page {issue.page}</button>
                  {" · "}{issue.severity === "major" ? "Material issue" : "Minor issue"}
                </p>
                <p className="mt-1">{issue.message}</p>
                {issue.reference_quote && <p className="mt-2 break-words"><span className="font-medium">Reference: </span>“{issue.reference_quote}”</p>}
                {issue.candidate_quote && <p className="mt-1 break-words"><span className="font-medium">Extracted: </span>“{issue.candidate_quote}”</p>}
              </li>
            ))}
          </ul>
        </details>
      )}
      {judgment?.status === "completed" && (
        <p className="mt-2 text-[11px] text-faint">
          {modelName} · Compared with LlamaParse
          {page !== "all" && judgment.pages_total > 1 && judgment.verdict && (
            <> · Overall: {VERDICTS[judgment.verdict].label} (worst page)</>
          )}
        </p>
      )}
      {onJudge && (!judgment || judgment.status === "failed") && (
        <button type="button" disabled={busy} onClick={onJudge} className="mt-2 rounded text-xs font-medium text-brand hover:underline disabled:opacity-50">
          {busy ? "Queuing…" : judgment ? "Retry quality check" : "Judge outputs"}
        </button>
      )}
    </aside>
  );
}
