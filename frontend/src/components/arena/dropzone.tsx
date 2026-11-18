"use client";

import { useId, useRef, useState } from "react";
import clsx from "clsx";
import { CloudUpload, FileText } from "lucide-react";
import { ProgressBar, Spinner } from "@/components/ui";

export const ACCEPT = ".pdf,.png,.jpg,.jpeg,.webp,.tif,.tiff";
const EXTENSIONS = ACCEPT.split(",");

export function validateFile(file: File, maxMb: number): string | null {
  const name = file.name.toLowerCase();
  if (!EXTENSIONS.some((ext) => name.endsWith(ext))) {
    return "Use a PDF, PNG, JPG, WebP or TIFF file.";
  }
  if (file.size > maxMb * 1024 * 1024) return `The file is larger than ${maxMb} MB.`;
  if (file.size === 0) return "The file is empty.";
  return null;
}

export function Dropzone({
  onFiles,
  onSample,
  upload,
  maxMb,
  disabled,
  sampleLoading = false,
  compact = false,
}: {
  onFiles: (files: File[]) => void;
  onSample: () => void;
  upload: { name: string; progress: number } | null;
  maxMb: number;
  disabled?: boolean;
  sampleLoading?: boolean;
  compact?: boolean;
}) {
  const inputId = useId();
  const [dragging, setDragging] = useState(false);
  const depth = useRef(0);

  if (upload) {
    const processing = upload.progress >= 1;
    return (
      <div className={clsx("flex flex-col items-center justify-center rounded-[var(--radius-panel)] border border-line bg-white p-6 text-center", compact ? "min-h-[180px]" : "min-h-[300px]")}>
        <span className="grid size-14 place-items-center rounded-2xl bg-brand-wash text-brand">
          <FileText aria-hidden className="size-6" />
        </span>
        <p className="mt-5 max-w-sm truncate font-medium">{upload.name}</p>
        <p className="mt-1 text-sm text-muted" aria-live="polite">
          {processing
            ? "Reading pages…"
            : `Uploading… ${Math.round(upload.progress * 100)}%`}
        </p>
        <ProgressBar
          className="mt-5 w-full max-w-xs"
          value={upload.progress}
          indeterminate={processing}
          label="Upload progress"
        />
      </div>
    );
  }

  const pick = (files: FileList | null) => {
    if (files?.length && !disabled) onFiles(Array.from(files));
  };

  return (
    <div
      onDragEnter={(e) => {
        e.preventDefault();
        depth.current += 1;
        setDragging(true);
      }}
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
      }}
      onDragLeave={() => {
        depth.current = Math.max(0, depth.current - 1);
        if (depth.current === 0) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        depth.current = 0;
        setDragging(false);
        if (!disabled) pick(e.dataTransfer.files);
      }}
      className={clsx(
        "relative flex flex-col items-center justify-center rounded-[var(--radius-panel)] border-2 border-dashed text-center transition-colors",
        compact ? "min-h-[180px] p-5" : "min-h-[300px] p-8",
        dragging ? "border-brand bg-brand-wash" : "border-ash bg-white hover:border-stone",
      )}
    >
      <span
        className={clsx(
          "grid place-items-center rounded-[20px] transition-colors",
          compact ? "size-10" : "size-16",
          dragging ? "bg-brand text-white" : "bg-brand-soft text-brand",
        )}
      >
        <CloudUpload aria-hidden className={compact ? "size-5" : "size-7"} />
      </span>
      <p className={clsx("tracking-tight", compact ? "mt-3 text-base" : "mt-6 text-xl")}>
        {dragging ? "Drop to add documents" : compact ? "Add more documents to your batch" : "Drag and drop your documents here"}
      </p>
      <p className="mt-2 text-sm text-muted">
        PDF, PNG, JPG, WebP or TIFF · up to {maxMb} MB each
      </p>
      <div className={clsx("flex flex-wrap items-center justify-center gap-3", compact ? "mt-4" : "mt-7")}>
        <label
          htmlFor={inputId}
          aria-disabled={disabled}
          className="btn btn-dark cursor-pointer has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-brand"
        >
          {compact ? "Add documents" : "Choose files"}
          <input
            id={inputId}
            type="file"
            multiple
            accept={ACCEPT}
            disabled={disabled}
            className="sr-only"
            onChange={(e) => {
              pick(e.target.files);
              e.target.value = "";
            }}
          />
        </label>
        <button type="button" onClick={onSample} disabled={disabled || sampleLoading} className="btn btn-ghost">
          {sampleLoading ? <><Spinner className="size-4" /> Opening sample…</> : "Use a sample report"}
        </button>
      </div>
    </div>
  );
}
