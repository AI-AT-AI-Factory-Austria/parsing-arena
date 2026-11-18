import type { Metadata } from "next";
import { Suspense } from "react";
import { BatchView } from "@/components/results/batch-view";
import { ExperimentSkeleton } from "@/components/results/experiment-view";

export const metadata: Metadata = { title: "Batch results" };

export default function BatchPage({ params }: { params: Promise<{ id: string }> }) {
  return <div className="flex-1 bg-smoke">
    <div className="mx-auto max-w-[1680px] px-4 pb-24 pt-28 sm:px-6 lg:pt-32">
      <Suspense fallback={<ExperimentSkeleton />}>
        {params.then(({ id }) => <BatchView key={id} id={id} />)}
      </Suspense>
    </div>
  </div>;
}
