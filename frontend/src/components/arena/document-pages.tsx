"use client";

import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { ChevronLeft, ChevronRight, Expand, FileText, X } from "lucide-react";
import { pageImageUrl } from "@/lib/api";
import { formatBytes, pluralize } from "@/lib/format";
import type { DocumentInfo } from "@/lib/types";

export type DocumentSelection = { document: DocumentInfo; pages: number[] };
export const MAX_BATCH_PAGES = 12;

function Preview({ document: doc, page, onClose, onPage }: {
  document: DocumentInfo; page: number; onClose: () => void; onPage: (page: number) => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    dialog?.showModal();
    return () => dialog?.close();
  }, []);
  return <dialog ref={ref} onCancel={(e) => { e.preventDefault(); onClose(); }}
    aria-label={`Preview of ${doc.filename}`}
    className="m-auto h-[90dvh] w-[calc(100%-2rem)] max-w-4xl rounded-2xl border border-line bg-smoke p-0 text-ink shadow-float backdrop:bg-ink/50">
    <div className="sticky top-0 z-10 flex items-center gap-3 border-b border-line bg-white p-4">
      <p className="min-w-0 flex-1 truncate text-sm font-medium">{doc.filename}</p>
      <button type="button" aria-label="Previous preview page" disabled={page === 1} onClick={() => onPage(page - 1)} className="btn btn-ghost btn-sm"><ChevronLeft className="size-4" /></button>
      <span className="whitespace-nowrap text-xs tabular-nums">{page} / {doc.page_count}</span>
      <button type="button" aria-label="Next preview page" disabled={page === doc.page_count} onClick={() => onPage(page + 1)} className="btn btn-ghost btn-sm"><ChevronRight className="size-4" /></button>
      <button type="button" aria-label="Close preview" onClick={onClose} className="btn btn-ghost btn-sm"><X className="size-4" /></button>
    </div>
    <img src={pageImageUrl(doc.id, page)} alt={`Page ${page} of ${doc.filename}`} className="mx-auto w-full max-w-3xl p-4" />
  </dialog>;
}

function DocumentCard({ item, remaining, disabled, onPages, onRemove }: {
  item: DocumentSelection; remaining: number; disabled: boolean;
  onPages: (pages: number[]) => void; onRemove: () => void;
}) {
  const doc = item.document;
  const [offset, setOffset] = useState(0);
  const [preview, setPreview] = useState<number | null>(null);
  const visible = Array.from({ length: Math.min(12, doc.page_count - offset) }, (_, i) => i + offset + 1);
  return <section className="card overflow-hidden" aria-label={`Pages of ${doc.filename}`}>
    {preview !== null && <Preview document={doc} page={preview} onPage={setPreview} onClose={() => setPreview(null)} />}
    <div className="flex items-start gap-3 border-b border-line p-4">
      <span className="mt-0.5 grid size-9 shrink-0 place-items-center rounded-xl bg-brand-wash text-brand"><FileText aria-hidden className="size-4" /></span>
      <div className="min-w-0 flex-1">
        <h3 className="break-words text-sm font-medium">{doc.filename}</h3>
        <p className="mt-1 text-xs text-faint">{pluralize(doc.page_count, "page")} · {formatBytes(doc.size_bytes)} · <span className="text-brand">{item.pages.length} selected</span></p>
      </div>
      <button type="button" disabled={disabled} onClick={onRemove} aria-label={`Remove ${doc.filename}`} className="rounded-full p-1.5 text-muted hover:bg-danger-wash hover:text-danger"><X className="size-4" /></button>
    </div>
    <div className="flex items-center justify-between gap-2 px-4 pt-3 text-xs">
      <span className="text-faint">Select pages to include</span>
      <div className="flex gap-3">
        <button type="button" className="text-brand hover:underline disabled:text-stone" disabled={disabled || doc.page_count > item.pages.length + remaining || item.pages.length === doc.page_count}
          onClick={() => onPages(Array.from({ length: doc.page_count }, (_, i) => i + 1))}>Select all</button>
        <button type="button" className="text-muted hover:underline disabled:text-stone" disabled={disabled || item.pages.length === 0} onClick={() => onPages([])}>Clear</button>
      </div>
    </div>
    <div className="grid grid-cols-3 gap-3 p-4 sm:grid-cols-4 lg:grid-cols-3 xl:grid-cols-4">
      {visible.map((page) => {
        const checked = item.pages.includes(page);
        const locked = disabled || (!checked && remaining === 0);
        return <div key={page} className={clsx("group relative overflow-hidden rounded-xl border transition", checked ? "border-brand bg-brand-wash ring-1 ring-brand/10" : "border-line bg-smoke")}>
          <button type="button" onClick={() => setPreview(page)} className="relative block w-full bg-white p-2" aria-label={`Preview ${doc.filename}, page ${page}`}>
            <img loading="lazy" src={`${pageImageUrl(doc.id, page)}?thumbnail=true`} alt={`Page ${page}`} className="aspect-[3/4] w-full object-contain" />
            <span className="absolute bottom-2 right-2 rounded-full bg-white/95 p-1.5 text-muted shadow-sm group-hover:text-brand"><Expand aria-hidden className="size-3" /></span>
          </button>
          <label className={clsx("flex items-center gap-2 border-t px-2 py-2.5 text-xs", checked ? "border-brand/20 text-brand" : "border-line text-muted", locked ? "cursor-not-allowed" : "cursor-pointer")}>
            <input type="checkbox" checked={checked} disabled={locked} aria-label={`Include ${doc.filename}, page ${page}`} className="size-3.5 accent-brand"
              onChange={() => onPages(checked ? item.pages.filter((n) => n !== page) : [...item.pages, page].sort((a, b) => a - b))} />
            Page {page}
          </label>
        </div>;
      })}
    </div>
    {doc.page_count > 12 && <div className="flex items-center justify-between border-t border-line px-4 py-3 text-xs text-muted">
      <button type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 12))} className="btn btn-ghost btn-sm"><ChevronLeft className="size-3.5" /> Previous</button>
      <span>{offset + 1}–{offset + visible.length} of {doc.page_count}</span>
      <button type="button" disabled={offset + 12 >= doc.page_count} onClick={() => setOffset(offset + 12)} className="btn btn-ghost btn-sm">More pages <ChevronRight className="size-3.5" /></button>
    </div>}
  </section>;
}

export function DocumentPages({ items, limit, disabled, onChange }: {
  items: DocumentSelection[]; limit: number; disabled: boolean; onChange: (items: DocumentSelection[]) => void;
}) {
  const total = items.reduce((sum, item) => sum + item.pages.length, 0);
  return <div className="space-y-4">
    <div className="flex items-center justify-between gap-3 pt-2">
      <div><h2 className="text-lg tracking-tight">Your batch</h2><p className="mt-0.5 text-xs text-muted">{pluralize(items.length, "document")} · Click a preview to take a closer look.</p></div>
      <span aria-live="polite" className="shrink-0 rounded-full bg-brand-wash px-3 py-1.5 text-sm font-medium tabular-nums text-brand">{total} / {limit} pages</span>
    </div>
    {total === limit && <p className="text-xs text-muted">All {limit} page slots are filled. Deselect a page to choose another.</p>}
    {items.map((item) => <DocumentCard key={item.document.id} item={item} remaining={Math.max(0, limit - total)} disabled={disabled}
      onRemove={() => onChange(items.filter((x) => x.document.id !== item.document.id))}
      onPages={(pages) => { if (pages.length <= item.pages.length + limit - total) onChange(items.map((x) => x.document.id === item.document.id ? { ...x, pages } : x)); }} />)}
  </div>;
}
