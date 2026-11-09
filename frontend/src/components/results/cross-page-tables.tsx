import type { RunDetail } from "@/lib/types";
import { Markdown } from "@/components/markdown";
import type { PageChoice, ViewMode } from "./output-panel";

export function CrossPageTables({ run, page, view, onSelectPage }: {
  run: RunDetail;
  page: PageChoice;
  view: ViewMode;
  onSelectPage: (page: PageChoice) => void;
}) {
  const tables = (run.meta.cross_page_tables ?? []).filter(
    table => page === "all" || table.source_pages.includes(page),
  );
  const issues = run.meta.cross_page_reconstruction?.issues ?? [];
  if (!tables.length && !issues.length) return null;
  return (
    <div className="mb-4 space-y-2 text-xs">
      {tables.length > 0 && (
        <div className="rounded-xl border border-line bg-fog px-3 py-2.5">
          <p className="font-medium">{tables.length} {tables.length === 1 ? "table" : "tables"} reconstructed across pages</p>
          <p className="mt-1 text-muted">
            {page === "all"
              ? "The document below contains the joined tables."
              : "The page below keeps its original extraction. View a joined table here or "}
            {page !== "all" && <button type="button" className="text-brand underline underline-offset-2" onClick={() => onSelectPage("all")}>open the whole document</button>}
            {page !== "all" && "."}
            {" "}Quality scores assess the original page extractions.
          </p>
          {page !== "all" && tables.map((table, index) => (
            <details key={`${table.source_pages.join("-")}:${index}`} className="mt-2 border-t border-line pt-2">
              <summary className="cursor-pointer font-medium">Joined table · pages {table.source_pages.join(", ")}</summary>
              <div className="mt-3">
                {view === "rendered"
                  ? <Markdown markdown={table.html} runId={run.id} />
                  : <pre className="whitespace-pre-wrap break-words font-mono">{table.html}</pre>}
              </div>
            </details>
          ))}
        </div>
      )}
      {issues.length > 0 && (
        <details className="rounded-xl bg-amber-wash px-3 py-2 text-amber">
          <summary className="cursor-pointer">Some table continuations could not be reconstructed. Page extractions are available.</summary>
          <ul className="mt-2 space-y-1">
            {issues.map((issue, index) => <li key={index}>Pages {issue.source_pages.join(", ")}: {issue.message}</li>)}
          </ul>
        </details>
      )}
    </div>
  );
}
