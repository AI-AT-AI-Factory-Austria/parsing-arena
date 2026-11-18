"use client";

import { useEffect, useId, useRef, useState } from "react";
import { ArrowRight, TriangleAlert, X } from "lucide-react";
import { pluralize } from "@/lib/format";

export function JudgingConfirmation({ filename, pageCount, documentCount = 1, onCancel, onConfirm }: {
  filename: string;
  pageCount: number;
  documentCount?: number;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const submitted = useRef(false);
  const [confirmation, setConfirmation] = useState("");
  const id = useId();

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialog.showModal();
    return () => {
      dialog.close();
      document.body.style.overflow = overflow;
    };
  }, []);

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby={`${id}-title`}
      aria-describedby={`${id}-details ${id}-responsibility`}
      onCancel={(event) => {
        event.preventDefault();
        onCancel();
      }}
      className="m-auto max-h-[calc(100dvh-2rem)] w-[calc(100%-2rem)] max-w-lg overflow-y-auto rounded-[var(--radius-panel)] border border-line bg-white p-6 text-ink shadow-float backdrop:bg-ink/40 backdrop:backdrop-blur-sm sm:p-8"
    >
      <div className="flex items-start justify-between gap-4">
        <span className="grid size-11 place-items-center rounded-2xl bg-danger-wash text-danger">
          <TriangleAlert aria-hidden className="size-5" />
        </span>
        <button type="button" onClick={onCancel} aria-label="Cancel automatic judging"
          className="grid size-8 place-items-center rounded-full text-muted hover:bg-fog hover:text-ink">
          <X aria-hidden className="size-4" />
        </button>
      </div>
      <h2 id={`${id}-title`} className="mt-4 text-2xl tracking-tight">
        Confirm automatic judging
      </h2>
      <p className="mt-2 break-words text-sm text-muted">
        {filename} · {pluralize(pageCount, "selected page")}
      </p>
      <div id={`${id}-details`} className="mt-5 space-y-2 text-sm leading-relaxed text-body">
        <p><strong>LlamaParse API</strong> receives {documentCount > 1 ? <><strong>all {documentCount} selected documents in full</strong></> : <><strong>the full document</strong></>} to process the selected pages.</p>
        <p><strong>Mistral API</strong> receives the extracted text from LlamaParse and the selected models to judge the results.</p>
      </div>
      <p id={`${id}-responsibility`} className="mt-4 rounded-2xl bg-danger-wash p-4 text-sm font-medium leading-relaxed text-danger">
        You are responsible for ensuring that sharing {documentCount > 1 ? "these documents and their" : "this document and its"} contents with
        these external services is appropriate given its sensitivity, and that you are
        authorised to do so.
      </p>
      <form className="mt-5" onSubmit={(event) => {
        event.preventDefault();
        if (confirmation !== "OK" || submitted.current) return;
        submitted.current = true;
        onConfirm();
      }}>
        <label htmlFor={`${id}-confirmation`} className="block text-sm font-medium">
          Type <span className="font-mono">OK</span> to confirm and proceed.
        </label>
        <input
          id={`${id}-confirmation`}
          value={confirmation}
          onChange={(event) => setConfirmation(event.target.value)}
          autoFocus
          autoComplete="off"
          autoCapitalize="off"
          spellCheck={false}
          required
          pattern="OK"
          className="mt-2 w-full rounded-xl border border-line px-3.5 py-2.5 font-mono text-base outline-none focus:border-brand"
        />
        <div className="mt-6 flex flex-wrap justify-end gap-3">
          <button type="button" onClick={onCancel} className="btn btn-outline">Cancel</button>
          <button type="submit" disabled={confirmation !== "OK"} className="btn btn-primary">
            Confirm & start <ArrowRight aria-hidden className="size-4" />
          </button>
        </div>
      </form>
    </dialog>
  );
}
