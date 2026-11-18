"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import clsx from "clsx";
import { ArrowRight, Check, Cpu, TriangleAlert } from "lucide-react";
import { api } from "@/lib/api";
import { usePolling } from "@/lib/use-polling";
import type { ArenaModel, Health, ModelsResponse } from "@/lib/types";
import { Chip, Spinner } from "@/components/ui";
import { JudgingConfirmation } from "@/components/judging-confirmation";
import { Dropzone, validateFile } from "./dropzone";
import { DocumentPages, MAX_BATCH_PAGES, type DocumentSelection } from "./document-pages";
import { SampleGallery } from "./sample-gallery";
import { ModelPicker } from "./model-picker";

type Catalog = { models: ModelsResponse; health: Health; cpuOnly: boolean };

export function ArenaWorkspace() {
  const router = useRouter();
  const [cpuOnly, setCpuOnly] = useState(false);
  const catalog = usePolling<Catalog>(
    `catalog:${cpuOnly}`,
    async () => {
      const requestedCpuOnly = cpuOnly;
      const [models, health] = await Promise.all([api.models(requestedCpuOnly), api.health()]);
      if (requestedCpuOnly && models.cpu_only !== true) {
        throw new Error("Restart the backend to enable CPU-only runs.");
      }
      return { models, health, cpuOnly: requestedCpuOnly };
    },
    () => null,
  );

  const [documents, setDocuments] = useState<DocumentSelection[]>([]);
  const uploadLock = useRef(false);
  const [upload, setUpload] = useState<{ name: string; progress: number } | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [openingSample, setOpeningSample] = useState(false);
  const [selection, setSelection] = useState<string[] | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [confirmJudging, setConfirmJudging] = useState(false);

  // Do not allow a start against the previous mode's catalog while it reloads.
  const data = catalog.data?.cpuOnly === cpuOnly ? catalog.data : undefined;
  const maxMb = data?.health.max_upload_mb ?? 100;
  const maxPages = Math.min(MAX_BATCH_PAGES, data?.health.max_pages_per_experiment ?? MAX_BATCH_PAGES);

  const choices: ArenaModel[] = data
    ? [...data.models.models.filter((m) => m.implemented), data.models.reference]
    : [];
  const selected = choices
    .filter((m) => m.availability.runnable && (selection ? selection.includes(m.id) : m.kind === "open"))
    .map((m) => m.id);
  const reference = data?.models.reference;
  const referenceSelected = reference ? selected.includes(reference.id) : false;

  const selectedDocuments = documents.filter((item) => item.pages.length > 0);
  const pageCount = documents.reduce((sum, item) => sum + item.pages.length, 0);

  const credits =
    data && referenceSelected ? pageCount * data.models.llamaparse_credits_per_page : 0;
  const usd = data ? credits * data.models.llamaparse_usd_per_credit : 0;
  const runsLocally = choices.some(
    (m) => selected.includes(m.id) && m.availability.backend === "local",
  );
  const canStart = pageCount > 0 && pageCount <= maxPages && selected.length > 0 && !starting && !upload && !openingSample;

  const handleFiles = async (files: File[]) => {
    if (uploadLock.current || starting) return;
    uploadLock.current = true;
    setUploadError(null);
    setStartError(null);
    const next = [...documents];
    const errors: string[] = [];
    try {
      for (const file of files) {
        if (next.length >= MAX_BATCH_PAGES) {
          errors.push(`A batch can contain up to ${MAX_BATCH_PAGES} documents. Remove a document to add another.`);
          break;
        }
        const problem = validateFile(file, maxMb);
        if (problem) { errors.push(`${file.name}: ${problem}`); continue; }
        setUpload({ name: file.name, progress: 0 });
        try {
          const document = await api.uploadDocument(file, (progress) => setUpload({ name: file.name, progress }));
          if (next.some((item) => item.document.sha256 === document.sha256)) {
            errors.push(`${file.name} is already in this batch.`);
            continue;
          }
          const remaining = Math.max(0, maxPages - next.reduce((sum, item) => sum + item.pages.length, 0));
          next.push({ document, pages: Array.from({ length: Math.min(remaining, document.page_count) }, (_, i) => i + 1) });
          setDocuments([...next]);
        } catch (err) { errors.push(`${file.name}: ${(err as Error).message}`); }
      }
    } finally {
      setUpload(null);
      uploadLock.current = false;
      if (errors.length) setUploadError(errors.join(" "));
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
    if (!canStart) return;
    setStarting(true);
    setStartError(null);
    try {
      const batch = await api.createBatch({
        documents: selectedDocuments.map((item) => ({ document_id: item.document.id, pages: item.pages })),
        models: selected,
        cpu_only: cpuOnly,
      });
      router.push(`/batches/${batch.id}`);
    } catch (err) {
      setStartError((err as Error).message);
    } finally {
      // The page stays mounted (hidden) after navigating to the results, so
      // coming back must find the button usable again.
      setStarting(false);
    }
  };

  const requestStart = () => {
    if (!canStart) return;
    if (referenceSelected) setConfirmJudging(true);
    else void start();
  };

  return (
    <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
      {confirmJudging && selectedDocuments.length > 0 && (
        <JudgingConfirmation
          filename={selectedDocuments.map((item) => item.document.filename).join(", ")}
          documentCount={selectedDocuments.length}
          pageCount={pageCount}
          onCancel={() => setConfirmJudging(false)}
          onConfirm={() => {
            setConfirmJudging(false);
            void start();
          }}
        />
      )}
      {/* Document */}
      <section aria-labelledby="doc-heading" className="space-y-3">
        <h2 id="doc-heading" className="sr-only">
          Document
        </h2>
        <Dropzone
          onFiles={(files) => void handleFiles(files)}
          onSample={() => void handleSample()}
          upload={upload}
          maxMb={maxMb}
          compact={documents.length > 0}
          disabled={openingSample || starting || Boolean(upload) || confirmJudging}
          sampleLoading={openingSample}
        />
        {documents.length > 0 && <DocumentPages items={documents} limit={maxPages}
          disabled={starting || Boolean(upload) || confirmJudging} onChange={setDocuments} />}
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

        <label className={clsx(
          "mt-5 flex cursor-pointer items-start gap-3 rounded-2xl border p-3.5 transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-brand",
          cpuOnly ? "border-brand bg-brand-wash" : "border-line hover:border-ash",
        )}>
          <input
            type="checkbox"
            checked={cpuOnly}
            disabled={starting}
            onChange={(event) => {
              setCpuOnly(event.target.checked);
              setStartError(null);
            }}
            className="mt-1 size-5 shrink-0 accent-brand"
            aria-labelledby="cpu-mode-label"
            aria-describedby="cpu-mode-help"
          />
          <span>
            <span id="cpu-mode-label" className="flex items-center gap-2 font-medium">
              <Cpu aria-hidden className="size-4" />Use CPU only
            </span>
            <span id="cpu-mode-help" className="mt-1 block text-xs leading-relaxed text-muted">
              Runs the selected extraction methods locally without a GPU. Larger models can be much slower.
              Optional API judging is a separate setting below. Saved samples keep their original results.
            </span>
          </span>
        </label>

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
                  Uploads each selected document in full to the LlamaParse API to process its selected pages. Mistral receives the extracted
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
            <ModelPicker
              models={choices.filter((model) => model.kind !== "reference")}
              health={data.health}
              selected={selected}
              onToggle={toggle}
            />
          )}
        </fieldset>

        <div className="mt-7 border-t border-line pt-5">
          <dl className="space-y-1.5 text-sm">
            <div className="flex justify-between gap-4">
              <dt className="text-muted">Documents in this run</dt>
              <dd>{selectedDocuments.length}</dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt className="text-muted">Models</dt>
              <dd>{selected.length || "–"}</dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt className="text-muted">Selected pages</dt>
              <dd>{pageCount || "–"}</dd>
            </div>
            {referenceSelected && (
              <div className="flex justify-between gap-4">
                <dt className="text-muted">LlamaParse credits</dt>
                <dd>
                  {pageCount > 0 ? (
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
            onClick={requestStart}
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
          {pageCount === 0 && <p className="mt-3 text-center text-xs text-faint">Add documents and select at least one page to start.</p>}
          {data && (
            <div className="mt-5 space-y-2 text-xs leading-relaxed text-faint">
              {(cpuOnly || (runsLocally && data.health.local_device?.available)) && (
                <p className="flex gap-2">
                  <Cpu aria-hidden className="mt-px size-3.5 shrink-0" />
                  Local models run on{" "}
                  {cpuOnly ? "CPU" : data.health.local_device?.device_name ?? data.health.local_device?.device}
                  , one model at a time so timings stay comparable.
                </p>
              )}
              {!cpuOnly && data.health.runtime_profile && data.health.runtime_profile !== "disabled" && (
                <p>
                  Runtime profile: {data.health.runtime_profile === "windows-amd"
                    ? "Windows / AMD (ROCm + Vulkan)"
                    : data.health.runtime_profile === "linux-nvidia"
                      ? "Linux / NVIDIA (CUDA)" : "Native"}
                </p>
              )}
              {!cpuOnly && data.health.runtime_profile_error && (
                <p role="alert" className="text-danger">{data.health.runtime_profile_error}</p>
              )}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
