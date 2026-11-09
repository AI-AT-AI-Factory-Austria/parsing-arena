"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import clsx from "clsx";
import { ArrowRight, Check, Cpu, ExternalLink, RefreshCw, TriangleAlert, X } from "lucide-react";
import { api } from "@/lib/api";
import { fileKind, formatBytes, formatScore, parsePageSelection, pluralize } from "@/lib/format";
import { usePolling } from "@/lib/use-polling";
import type { ArenaModel, DocumentInfo, Health, ModelsResponse } from "@/lib/types";
import { Chip, Spinner } from "@/components/ui";
import { ACCEPT, Dropzone, validateFile } from "./dropzone";
import { SampleGallery } from "./sample-gallery";

type Catalog = { models: ModelsResponse; health: Health };

function backendLabel(model: ArenaModel, health: Health): string {
  const backend = model.availability.backend;
  if (backend === "api") return "Hosted API";
  if (backend === "endpoint") return "vLLM endpoint";
  if (model.availability.runtime_label) return model.availability.runtime_label;
  if (backend === "cpu") return "CPU";
  const device = health.local_device;
  if (device?.available && device.device !== "cpu") return "Local GPU";
  return "Local CPU";
}

export function ArenaWorkspace() {
  const router = useRouter();
  const catalog = usePolling<Catalog>(
    "catalog",
    async () => {
      const [models, health] = await Promise.all([api.models(), api.health()]);
      return { models, health };
    },
    () => null,
  );

  const [doc, setDoc] = useState<DocumentInfo | null>(null);
  const [upload, setUpload] = useState<{ name: string; progress: number } | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [openingSample, setOpeningSample] = useState(false);
  const [selection, setSelection] = useState<string[] | null>(null);
  const [pageMode, setPageMode] = useState<"all" | "custom">("all");
  const [pageInput, setPageInput] = useState("");
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  const data = catalog.data;
  const maxMb = data?.health.max_upload_mb ?? 100;
  const maxPages = data?.health.max_pages_per_experiment ?? 50;

  const choices: ArenaModel[] = data
    ? [...data.models.models.filter((m) => m.implemented), data.models.reference]
    : [];
  const comingSoon = data ? data.models.models.filter((m) => !m.implemented) : [];
  const selected =
    selection ??
    choices.filter((m) => m.availability.runnable && m.kind === "open").map((m) => m.id);
  const reference = data?.models.reference;
  const referenceSelected = reference ? selected.includes(reference.id) : false;

  let pages: number[] = [];
  let pageError: string | null = null;
  if (doc) {
    if (pageMode === "all") {
      pages = Array.from({ length: doc.page_count }, (_, i) => i + 1);
    } else {
      try {
        pages = parsePageSelection(pageInput, doc.page_count);
      } catch (err) {
        pageError = (err as Error).message;
      }
    }
    if (!pageError && pages.length > maxPages) {
      pageError = `Select at most ${maxPages} pages per run (this selection has ${pages.length}).`;
    }
  }

  const credits =
    data && referenceSelected ? pages.length * data.models.llamaparse_credits_per_page : 0;
  const usd = data ? credits * data.models.llamaparse_usd_per_credit : 0;
  const runsLocally = choices.some(
    (m) => selected.includes(m.id) && m.availability.backend === "local",
  );
  const canStart = Boolean(doc) && selected.length > 0 && !pageError && !starting && !upload;

  const adoptDocument = (document: DocumentInfo) => {
    setDoc(document);
    setStarting(false);
    setStartError(null);
    if (document.page_count > maxPages) {
      setPageMode("custom");
      setPageInput(`1-${maxPages}`);
    } else {
      setPageMode("all");
      setPageInput("");
    }
  };

  const handleFile = async (file: File) => {
    const problem = validateFile(file, maxMb);
    if (problem) {
      setUploadError(problem);
      return;
    }
    setUploadError(null);
    setUpload({ name: file.name, progress: 0 });
    try {
      const document = await api.uploadDocument(file, (progress) =>
        setUpload({ name: file.name, progress }),
      );
      adoptDocument(document);
    } catch (err) {
      setUploadError((err as Error).message);
    } finally {
      setUpload(null);
    }
  };

  const handleSample = async () => {
    setUploadError(null);
    setOpeningSample(true);
    try {
      const sample = await api.randomSample();
      router.push(`/experiments/${sample.experiment_id}`);
    } catch (err) {
      setUploadError((err as Error).message);
    } finally {
      setOpeningSample(false);
    }
  };

  const toggle = (id: string) => {
    setSelection(selected.includes(id) ? selected.filter((m) => m !== id) : [...selected, id]);
  };

  const start = async () => {
    if (!doc) return;
    setStarting(true);
    setStartError(null);
    try {
      const experiment = await api.createExperiment({
        document_id: doc.id,
        models: selected,
        pages: pageMode === "custom" ? pageInput : undefined,
      });
      router.push(`/experiments/${experiment.id}`);
    } catch (err) {
      setStartError((err as Error).message);
    } finally {
      // The page stays mounted (hidden) after navigating to the results, so
      // coming back must find the button usable again.
      setStarting(false);
    }
  };

  return (
    <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(360px,1fr)]">
      {/* Document */}
      <section aria-labelledby="doc-heading" className="space-y-3">
        <h2 id="doc-heading" className="sr-only">
          Document
        </h2>
        {doc && !upload ? (
          <div className="card flex flex-col gap-6 p-5 sm:flex-row sm:p-6">
            <a
              href={doc.file_url}
              target="_blank"
              rel="noreferrer"
              className="block shrink-0 self-start overflow-hidden rounded-2xl border border-line bg-smoke"
              title="Open the original file"
            >
              <img
                src={doc.thumbnail_url}
                alt={`First page of ${doc.filename}`}
                className="h-56 w-auto max-w-[180px] object-contain sm:h-64"
              />
            </a>
            <div className="flex min-w-0 flex-1 flex-col">
              <p className="eyebrow">Document</p>
              <p className="mt-2 break-words text-2xl leading-snug tracking-tight">{doc.filename}</p>
              <p className="mt-2 text-sm text-muted">
                {fileKind(doc.media_type)} · {pluralize(doc.page_count, "page")} ·{" "}
                {formatBytes(doc.size_bytes)}
              </p>
              <div className="mt-auto flex flex-wrap gap-2 pt-6">
                <a
                  href={doc.file_url}
                  target="_blank"
                  rel="noreferrer"
                  className="btn btn-outline btn-sm"
                >
                  <ExternalLink aria-hidden className="size-3.5" /> Open original
                </a>
                <label className="btn btn-ghost btn-sm cursor-pointer has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-brand">
                  <RefreshCw aria-hidden className="size-3.5" /> Replace
                  <input
                    type="file"
                    accept={ACCEPT}
                    className="sr-only"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      e.target.value = "";
                      if (file) void handleFile(file);
                    }}
                  />
                </label>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  onClick={() => {
                    setDoc(null);
                    setStartError(null);
                  }}
                >
                  <X aria-hidden className="size-3.5" /> Remove
                </button>
              </div>
            </div>
          </div>
        ) : (
          <Dropzone
            onFile={(file) => void handleFile(file)}
            onSample={() => void handleSample()}
            upload={upload}
            maxMb={maxMb}
            disabled={openingSample}
            sampleLoading={openingSample}
          />
        )}
        {uploadError && (
          <p
            role="alert"
            className="flex items-start gap-2 rounded-2xl bg-danger-wash px-4 py-3 text-sm text-danger"
          >
            <TriangleAlert aria-hidden className="mt-0.5 size-4 shrink-0" /> {uploadError}
          </p>
        )}
        <SampleGallery />
      </section>

      {/* Run settings */}
      <section aria-labelledby="settings-heading" className="card min-w-0 p-5 sm:p-6">
        <h2 id="settings-heading" className="text-xl tracking-tight">
          Run settings
        </h2>

        {data && reference && (
          <fieldset className="mt-6 min-w-0 border-b border-line pb-6">
            <legend className="eyebrow">Automatic judging</legend>
            <label
              className={clsx(
                "mt-3 flex gap-3 rounded-2xl border p-3.5 transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-brand",
                !reference.availability.runnable
                  ? "cursor-not-allowed border-line bg-smoke"
                  : referenceSelected
                    ? "cursor-pointer border-brand bg-brand-wash"
                    : "cursor-pointer border-line hover:border-ash",
              )}
            >
              <input
                type="checkbox"
                className="sr-only"
                checked={reference.availability.runnable && referenceSelected}
                disabled={!reference.availability.runnable}
                onChange={() => toggle(reference.id)}
                aria-labelledby="reference-name"
                aria-describedby="reference-details"
              />
              <span
                aria-hidden
                className={clsx(
                  "mt-0.5 grid size-5 shrink-0 place-items-center rounded-md border transition-colors",
                  referenceSelected ? "border-brand bg-brand text-white" : "border-ash bg-white",
                )}
              >
                {referenceSelected && <Check className="size-3.5" strokeWidth={3} />}
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span id="reference-name" className="font-medium">{reference.name}</span>
                  <Chip tone="outline">External API</Chip>
                </span>
                <span id="reference-details" className="mt-1.5 block text-xs font-bold leading-relaxed text-danger">
                  Uploads selected pages to the LlamaParse API. Mistral receives the extracted
                  text and uses this reference to judge each model.
                </span>
                {!reference.availability.runnable && reference.availability.reason && (
                  <span className="mt-1.5 block text-xs text-amber">
                    {reference.availability.reason}
                  </span>
                )}
                {!data.health.judge_configured && (
                  <span className="mt-1.5 block text-xs text-amber">
                    Automatic judging is unavailable: Mistral is not configured.
                  </span>
                )}
              </span>
            </label>
          </fieldset>
        )}

        <fieldset className="mt-6 min-w-0">
          <legend className="eyebrow">Models</legend>
          {!data && catalog.error && (
            <p className="mt-3 flex items-start gap-2 text-sm text-danger">
              <TriangleAlert aria-hidden className="mt-0.5 size-4 shrink-0" />
              {catalog.error.message}
            </p>
          )}
          {!data && !catalog.error && (
            <div className="mt-3 space-y-2" aria-busy="true">
              {[0, 1, 2].map((i) => (
                <div key={i} className="h-[68px] animate-pulse rounded-2xl bg-smoke" />
              ))}
            </div>
          )}
          {data && (
            <ul className="mt-3 space-y-2">
              {choices.filter((model) => model.kind !== "reference").map((model) => {
                const runnable = model.availability.runnable;
                const checked = runnable && selected.includes(model.id);
                return (
                  <li key={model.id}>
                    <label
                      className={clsx(
                        "flex gap-3 rounded-2xl border p-3.5 transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-brand",
                        !runnable
                          ? "cursor-not-allowed border-line bg-smoke"
                          : checked
                            ? "cursor-pointer border-brand bg-brand-wash"
                            : "cursor-pointer border-line hover:border-ash",
                      )}
                    >
                      <input
                        type="checkbox"
                        className="peer sr-only"
                        checked={checked}
                        disabled={!runnable}
                        onChange={() => toggle(model.id)}
                      />
                      <span
                        aria-hidden
                        className={clsx(
                          "mt-0.5 grid size-5 shrink-0 place-items-center rounded-md border transition-colors",
                          checked ? "border-brand bg-brand text-white" : "border-ash bg-white",
                        )}
                      >
                        {checked && <Check className="size-3.5" strokeWidth={3} />}
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="flex items-center justify-between gap-2">
                          <span className="flex min-w-0 items-center gap-2">
                            <span className="truncate font-medium">{model.name}</span>
                          </span>
                          <span
                            className="shrink-0 font-mono text-sm tabular-nums text-muted"
                            title="ParseBench overall score"
                          >
                            {model.scores ? formatScore(model.scores.overall) : "–"}
                          </span>
                        </span>
                        <span className="mt-0.5 block text-xs text-faint">
                          {`${model.publisher} · ${model.params} · ${backendLabel(model, data.health)}`}
                        </span>
                        {(model.id === "marker_fast" || model.id === "opendataloader_core") && (
                          <span className="mt-1 block text-xs text-muted">
                            Embedded PDF text only · No OCR on scanned pages
                          </span>
                        )}
                        {model.id === "unlimited-ocr-2" && (
                          <span className="mt-1 block text-xs text-muted">
                            Reads up to {model.availability.max_pages ?? 16} pages together. This local
                            runtime does not provide the optimized long-document memory savings.
                          </span>
                        )}
                        {!runnable && model.availability.reason && (
                          <span className="mt-1.5 block text-xs text-amber">
                            {model.availability.reason}
                          </span>
                        )}
                      </span>
                    </label>
                  </li>
                );
              })}
            </ul>
          )}
          {comingSoon.length > 0 && (
            <p className="mt-3 text-xs text-faint">
              {pluralize(comingSoon.length, "more model")} coming soon:{" "}
              {comingSoon.map((m) => m.name).join(", ")}.{" "}
              <Link href="/#models" className="underline underline-offset-2 hover:text-ink">
                See the line-up
              </Link>
            </p>
          )}
        </fieldset>

        <fieldset className="mt-7 min-w-0" disabled={!doc}>
          <legend className="eyebrow">Pages</legend>
          <div
            role="radiogroup"
            aria-label="Pages to parse"
            className="mt-3 inline-flex rounded-full bg-fog p-1 text-sm"
          >
            {(["all", "custom"] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                role="radio"
                aria-checked={pageMode === mode}
                onClick={() => setPageMode(mode)}
                disabled={!doc}
                className={clsx(
                  "rounded-full px-4 py-1.5 transition-colors disabled:cursor-not-allowed",
                  pageMode === mode ? "bg-white text-ink shadow-sm" : "text-muted hover:text-ink",
                )}
              >
                {mode === "all"
                  ? doc
                    ? `All ${pluralize(doc.page_count, "page")}`
                    : "All pages"
                  : "Choose pages"}
              </button>
            ))}
          </div>
          {pageMode === "custom" && doc && (
            <div className="mt-3">
              <label htmlFor="page-input" className="sr-only">
                Pages, for example 1-3, 5
              </label>
              <input
                id="page-input"
                value={pageInput}
                onChange={(e) => setPageInput(e.target.value)}
                placeholder="e.g. 1-3, 5"
                aria-invalid={Boolean(pageError)}
                aria-describedby="page-help"
                className={clsx(
                  "w-full rounded-xl border bg-white px-3.5 py-2.5 font-mono text-sm outline-none transition-colors focus:border-brand",
                  pageError ? "border-danger" : "border-line",
                )}
              />
              <p
                id="page-help"
                className={clsx("mt-1.5 text-xs", pageError ? "text-danger" : "text-faint")}
              >
                {pageError ?? `${pluralize(pages.length, "page")} selected`}
              </p>
            </div>
          )}
          {pageMode === "all" && pageError && (
            <p className="mt-2 text-xs text-danger">{pageError}</p>
          )}
        </fieldset>

        <div className="mt-7 border-t border-line pt-5">
          <dl className="space-y-1.5 text-sm">
            <div className="flex justify-between gap-4">
              <dt className="text-muted">Models</dt>
              <dd>{selected.length || "–"}</dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt className="text-muted">Pages</dt>
              <dd>{doc && !pageError ? pages.length : "–"}</dd>
            </div>
            {referenceSelected && (
              <div className="flex justify-between gap-4">
                <dt className="text-muted">LlamaParse credits</dt>
                <dd>
                  {doc && !pageError ? (
                    <>
                      ≈ {credits.toLocaleString("en-GB")}{" "}
                      <span className="text-faint">(${usd.toFixed(2)})</span>
                    </>
                  ) : (
                    "–"
                  )}
                </dd>
              </div>
            )}
          </dl>
          <button
            type="button"
            onClick={() => void start()}
            disabled={!canStart}
            className="btn btn-primary btn-lg mt-5 w-full"
          >
            {starting ? (
              <>
                <Spinner className="size-4" /> Starting…
              </>
            ) : (
              <>
                Start the run <ArrowRight aria-hidden className="size-[18px]" />
              </>
            )}
          </button>
          {startError && (
            <p role="alert" className="mt-3 text-sm text-danger">
              {startError}
            </p>
          )}
          {!doc && <p className="mt-3 text-center text-xs text-faint">Add a document to start.</p>}
          {data && (
            <div className="mt-5 space-y-2 text-xs leading-relaxed text-faint">
              {runsLocally && data.health.local_device?.available && (
                <p className="flex gap-2">
                  <Cpu aria-hidden className="mt-px size-3.5 shrink-0" />
                  Local models run on{" "}
                  {data.health.local_device.device_name ?? data.health.local_device.device}
                  , one model at a time so timings stay comparable.
                </p>
              )}
              {data.health.runtime_profile && data.health.runtime_profile !== "disabled" && (
                <p>
                  Runtime profile: {data.health.runtime_profile === "windows-amd"
                    ? "Windows / AMD (ROCm + Vulkan)"
                    : data.health.runtime_profile === "linux-nvidia"
                      ? "Linux / NVIDIA (CUDA)" : "Native"}
                </p>
              )}
              {data.health.runtime_profile_error && (
                <p role="alert" className="text-danger">{data.health.runtime_profile_error}</p>
              )}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
