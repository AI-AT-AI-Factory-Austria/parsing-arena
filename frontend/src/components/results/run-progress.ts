import type { Run } from "@/lib/types";

/** Fraction done for a run, or `null` when progress is not measurable. */
export function runFraction(run: Run): number | null {
  if (run.status === "completed") return 1;
  if (run.kind === "reference") return null;
  if (run.progress.phase === "document") return null;
  const total = run.pages_total || 1;
  const p = run.progress;
  let within = 0;
  if (p.phase === "layout") within = 0.08;
  if (p.phase === "recognition") {
    within = p.regions_total ? 0.1 + 0.9 * ((p.regions_done ?? 0) / p.regions_total) : 0.1;
  }
  return Math.min(1, (run.pages_done + within) / total);
}

export function runPhaseText(run: Run, queuePosition?: number | null): string {
  const p = run.progress;
  switch (run.status) {
    case "queued":
      if (queuePosition != null) return `In the queue · #${queuePosition + 1} in line`;
      return run.kind === "reference" ? "Starting" : "Waiting for its turn";
    case "loading":
      return "Loading the model";
    case "failed":
      return run.error ?? "Failed";
    case "completed":
      return "Done";
    case "running":
      break;
  }
  if (run.kind === "reference") {
    if (p.phase === "uploading") return "Uploading to LlamaParse";
    if (p.phase === "queued") return "Queued at LlamaParse";
    return "LlamaParse is parsing";
  }
  if (p.phase === "document") return `Parsing ${run.pages_total} ${run.pages_total === 1 ? "page" : "pages"} together`;
  const page = (p.page_index ?? run.pages_done) + 1;
  const where = `page ${Math.min(page, run.pages_total)} of ${run.pages_total}`;
  if (p.phase === "layout") return `Detecting layout · ${where}`;
  if (p.phase === "recognition") {
    return p.regions_total
      ? `Reading region ${Math.min((p.regions_done ?? 0) + 1, p.regions_total)} of ${p.regions_total} · ${where}`
      : `Reading regions · ${where}`;
  }
  return `Parsing · ${where}`;
}
