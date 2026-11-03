const LOCALE = "en-GB";

export function formatSeconds(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "–";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes} min ${String(total % 60).padStart(2, "0")} s`;
  return `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, "0")} min`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatScore(score: number): string {
  return score.toFixed(1);
}

export function formatDateTime(iso: string): string {
  return new Intl.DateTimeFormat(LOCALE, { dateStyle: "medium", timeStyle: "short" }).format(
    new Date(iso),
  );
}

export function formatRelative(iso: string, now: number | null): string {
  if (now === null) return formatDateTime(iso);
  const diff = Math.max(0, (now - new Date(iso).getTime()) / 1000);
  if (diff < 45) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)} h ago`;
  if (diff < 2 * 86400) return "yesterday";
  return new Intl.DateTimeFormat(LOCALE, { dateStyle: "medium" }).format(new Date(iso));
}

export function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

export function fileKind(mediaType: string): string {
  if (mediaType === "application/pdf") return "PDF";
  const sub = mediaType.split("/")[1] ?? mediaType;
  return sub.replace("jpeg", "JPG").toUpperCase();
}

/** Expand "1-3, 5" into [1, 2, 3, 5]; throws with a readable message. */
export function parsePageSelection(input: string, pageCount: number): number[] {
  const pages = new Set<number>();
  for (const raw of input.split(",")) {
    const part = raw.trim();
    if (!part) continue;
    const match = /^(\d+)(?:\s*-\s*(\d+))?$/.exec(part);
    if (!match) throw new Error(`"${part}" is not a page or range`);
    const start = Number(match[1]);
    const end = match[2] ? Number(match[2]) : start;
    if (start < 1 || end < start) throw new Error(`"${part}" is not a valid range`);
    if (end > pageCount) throw new Error(`The document has only ${pluralize(pageCount, "page")}`);
    for (let page = start; page <= end; page++) pages.add(page);
  }
  if (pages.size === 0) throw new Error("Enter at least one page");
  return [...pages].sort((a, b) => a - b);
}
