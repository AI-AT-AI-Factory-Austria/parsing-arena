"use client";

import Link from "next/link";
import { ArrowRight, TriangleAlert } from "lucide-react";
import { api } from "@/lib/api";
import { pluralize } from "@/lib/format";
import type { SampleDocument } from "@/lib/types";
import { usePolling } from "@/lib/use-polling";
import { Spinner } from "@/components/ui";

function SampleCardContent({ sample }: { sample: SampleDocument }) {
  const preparing = sample.status === "queued" || sample.status === "running";
  const finished = sample.status === "completed" || sample.status === "failed";
  const failed = finished ? sample.model_count - sample.completed_models : 0;
  return (
    <>
      <img
        src={sample.document.thumbnail_url}
        alt=""
        loading="lazy"
        className="h-24 w-16 shrink-0 rounded-md border border-line bg-white object-contain"
      />
      <span className="flex min-w-0 flex-1 flex-col items-start">
        <span className="line-clamp-3 text-sm font-medium leading-snug" title={sample.document.filename}>
          {sample.title}
        </span>
        <span className="mt-1 text-xs text-faint">
          {pluralize(sample.document.page_count, "page")}
          {finished && ` · ${pluralize(sample.completed_models, "output")} saved`}
          {failed > 0 && <span className="text-amber"> · {failed} failed</span>}
        </span>
        <span className="mt-3 inline-flex items-center gap-1.5 text-xs font-medium text-brand">
          {preparing ? (
            <><Spinner className="size-3" /> View progress</>
          ) : sample.experiment_id ? (
            <>View results <ArrowRight aria-hidden className="size-3.5" /></>
          ) : (
            <span className="text-faint">Results not prepared yet</span>
          )}
        </span>
      </span>
    </>
  );
}

export function SampleGallery() {
  const catalog = usePolling(
    "sample-documents",
    () => api.samples(),
    (data) => data.samples.some((s) => s.status === "queued" || s.status === "running" || s.judging)
      ? 4000 : 30000,
  );

  return (
    <section aria-labelledby="samples-heading" className="pt-5">
      <h2 id="samples-heading" className="text-lg tracking-tight">Sample documents</h2>
      <p className="mt-1 text-sm text-muted">Explore saved results without running the models again.</p>
      {!catalog.data && !catalog.error && (
        <div className="mt-4 grid gap-3 sm:grid-cols-2" aria-busy="true" aria-label="Loading samples">
          {[0, 1].map((i) => <div key={i} className="h-32 animate-pulse rounded-2xl bg-fog" />)}
        </div>
      )}
      {catalog.error && (
        <p role="alert" className="mt-4 flex items-center gap-2 text-sm text-danger">
          <TriangleAlert aria-hidden className="size-4 shrink-0" />
          {catalog.error.message}
        </p>
      )}
      {catalog.data?.samples.length === 0 && (
        <p className="mt-4 text-sm text-faint">No sample documents are available yet.</p>
      )}
      {catalog.data && (
        <ul className="mt-4 grid gap-3 sm:grid-cols-2">
          {catalog.data.samples.map((sample) => (
            <li key={sample.id} className="min-w-0">
              {sample.experiment_id ? (
                <Link
                  href={`/experiments/${sample.experiment_id}`}
                  className="flex h-full gap-3 rounded-2xl border border-line bg-white p-4 transition-colors hover:border-brand hover:bg-brand-wash focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand"
                >
                  <SampleCardContent sample={sample} />
                </Link>
              ) : (
                <div className="flex h-full gap-3 rounded-2xl border border-line bg-smoke p-4">
                  <SampleCardContent sample={sample} />
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
