"use client";

import { useEffect, useId, useRef, useState } from "react";
import { ArrowRight, Check, Save, TriangleAlert, X } from "lucide-react";
import { api } from "@/lib/api";
import { pluralize } from "@/lib/format";
import type { JudgingKeyStatus } from "@/lib/types";
import { Spinner } from "@/components/ui";

export function JudgingConfirmation({ filename, pageCount, documentCount = 1, mode = "run", onCancel, onConfirm, onKeysSaved }: {
  filename: string;
  pageCount: number;
  documentCount?: number;
  mode?: "run" | "judge";
  onCancel: () => void;
  onConfirm: () => void;
  onKeysSaved?: () => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const submitted = useRef(false);
  const [confirmation, setConfirmation] = useState("");
  const [keys, setKeys] = useState<JudgingKeyStatus | null>(null);
  const [llamaKey, setLlamaKey] = useState("");
  const [mistralKey, setMistralKey] = useState("");
  const [region, setRegion] = useState<"us" | "eu">("us");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [keyError, setKeyError] = useState<string | null>(null);
  const needsLlama = mode === "run";
  const ready = !!keys && (!needsLlama || keys.llamaparse_configured) && keys.judge_configured;
  const canSave = !!keys && !ready && !saving &&
    (!needsLlama || keys.llamaparse_configured || !!llamaKey.trim()) &&
    (keys.judge_configured || !!mistralKey.trim());
  const id = useId();

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    const overflow = document.body.style.overflow;
    let cancelled = false;
    document.body.style.overflow = "hidden";
    dialog.showModal();
    void api.judgingKeys().then((status) => {
      if (cancelled) return;
      setKeys(status);
      setRegion(status.llamaparse_region);
    }).catch((error: Error) => { if (!cancelled) setKeyError(error.message); });
    return () => {
      cancelled = true;
      dialog.close();
      document.body.style.overflow = overflow;
    };
  }, []);

  const saveKeys = async () => {
    if (!canSave || !keys) return;
    setSaving(true);
    setKeyError(null);
    try {
      const status = await api.saveJudgingKeys({
        ...(needsLlama && !keys.llamaparse_configured ? { llama_cloud_api_key: llamaKey.trim(), llamaparse_region: region } : {}),
        ...(!keys.judge_configured ? { mistral_api_key: mistralKey.trim() } : {}),
      });
      setLlamaKey("");
      setMistralKey("");
      setKeys(status);
      setSaved(true);
      onKeysSaved?.();
    } catch (error) {
      setKeyError((error as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const reloadKeys = async () => {
    setKeyError(null);
    try {
      const status = await api.judgingKeys();
      setKeys(status);
      setRegion(status.llamaparse_region);
      if (status.llamaparse_configured) setLlamaKey("");
      if (status.judge_configured) setMistralKey("");
    } catch (error) { setKeyError((error as Error).message); }
  };

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby={`${id}-title`}
      aria-describedby={`${id}-details ${id}-responsibility`}
      onCancel={(event) => {
        event.preventDefault();
        if (!saving) onCancel();
      }}
      className="m-auto max-h-[calc(100dvh-2rem)] w-[calc(100%-2rem)] max-w-lg overflow-y-auto rounded-[var(--radius-panel)] border border-line bg-white p-6 text-ink shadow-float backdrop:bg-ink/40 backdrop:backdrop-blur-sm sm:p-8"
    >
      <div className="flex items-start justify-between gap-4">
        <span className="grid size-11 place-items-center rounded-2xl bg-danger-wash text-danger">
          <TriangleAlert aria-hidden className="size-5" />
        </span>
        <button type="button" disabled={saving} onClick={onCancel} aria-label="Cancel automatic judging"
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
        {needsLlama && <p><strong>LlamaParse API</strong> receives {documentCount > 1 ? <><strong>all {documentCount} selected documents in full</strong></> : <><strong>the full document</strong></>} to process the selected pages.</p>}
        <p><strong>Mistral API</strong> receives the extracted text from LlamaParse and the selected models to judge the results.</p>
        {!needsLlama && <p>The saved LlamaParse reference is reused.</p>}
      </div>
      <p id={`${id}-responsibility`} className="mt-4 rounded-2xl bg-danger-wash p-4 text-sm font-medium leading-relaxed text-danger">
        You are responsible for ensuring that sharing {documentCount > 1 ? "these documents and their" : "this document and its"} contents with
        these external services is appropriate given its sensitivity, and that you are
        authorised to do so.
      </p>
      <section aria-label="Judging API keys" className="mt-5 rounded-2xl border border-line bg-smoke/50 p-4">
        {!keys && !keyError && <p role="status" className="flex items-center gap-2 text-sm text-muted"><Spinner />Checking API keys…</p>}
        {ready && <p role="status" className="flex items-center gap-2 text-sm text-leaf-strong"><Check aria-hidden className="size-4" />{saved ? "API keys saved for future runs." : "API keys are already configured."}</p>}
        {keys && !ready && <>
          <h3 className="text-sm font-medium">Add missing API keys</h3>
          <p className="mt-1 text-xs leading-relaxed text-muted">Save once for this installation. Keys stay on the server and are shared by users of this installation. Saving does not start a run.</p>
          {needsLlama && (keys.llamaparse_configured ? <p className="mt-3 flex items-center gap-1.5 text-xs text-leaf-strong"><Check aria-hidden className="size-3.5" />LlamaParse key configured</p> : <div className="mt-3">
            <label htmlFor={`${id}-llama-key`} className="block text-sm font-medium">LlamaParse API key</label>
            <input id={`${id}-llama-key`} type="password" autoComplete="new-password" spellCheck={false} autoCapitalize="off" value={llamaKey} onChange={(e) => setLlamaKey(e.target.value)} disabled={saving} maxLength={4096} className="mt-1.5 w-full rounded-xl border border-line bg-white px-3 py-2 text-sm outline-none focus:border-brand" />
            <label htmlFor={`${id}-region`} className="mt-3 block text-xs font-medium">LlamaParse key region</label>
            <select id={`${id}-region`} value={region} onChange={(e) => setRegion(e.target.value as "us" | "eu")} disabled={saving} className="mt-1 w-full rounded-lg border border-line bg-white px-3 py-2 text-sm">
              <option value="us">US</option><option value="eu">EU</option>
            </select>
          </div>)}
          {keys.judge_configured ? <p className="mt-3 flex items-center gap-1.5 text-xs text-leaf-strong"><Check aria-hidden className="size-3.5" />Mistral key configured</p> : <div className="mt-3">
            <label htmlFor={`${id}-mistral-key`} className="block text-sm font-medium">Mistral API key</label>
            <input id={`${id}-mistral-key`} type="password" autoComplete="new-password" spellCheck={false} autoCapitalize="off" value={mistralKey} onChange={(e) => setMistralKey(e.target.value)} disabled={saving} maxLength={4096} className="mt-1.5 w-full rounded-xl border border-line bg-white px-3 py-2 text-sm outline-none focus:border-brand" />
          </div>}
          <button type="button" onClick={() => void saveKeys()} disabled={!canSave} className="btn btn-outline btn-sm mt-4">{saving ? <Spinner /> : <Save aria-hidden className="size-3.5" />}{saving ? "Saving…" : "Save API keys"}</button>
        </>}
        {keyError && <div className="mt-2 text-sm text-danger"><p role="alert">{keyError}</p><button type="button" onClick={() => void reloadKeys()} className="mt-2 rounded text-xs underline">Refresh key status</button></div>}
      </section>
      <form className="mt-5" onSubmit={(event) => {
        event.preventDefault();
        if (confirmation !== "OK" || !ready || saving || submitted.current) return;
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
          autoComplete="off"
          autoCapitalize="off"
          spellCheck={false}
          required
          pattern="OK"
          className="mt-2 w-full rounded-xl border border-line px-3.5 py-2.5 font-mono text-base outline-none focus:border-brand"
        />
        <div className="mt-6 flex flex-wrap justify-end gap-3">
          <button type="button" disabled={saving} onClick={onCancel} className="btn btn-outline">Cancel</button>
          <button type="submit" disabled={confirmation !== "OK" || !ready || saving} className="btn btn-primary">
            {needsLlama ? "Confirm & start" : "Confirm & judge"} <ArrowRight aria-hidden className="size-4" />
          </button>
        </div>
      </form>
    </dialog>
  );
}
