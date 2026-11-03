import type { Metadata } from "next";
import { Suspense } from "react";
import { ExperimentSkeleton, ExperimentView } from "@/components/results/experiment-view";

export const metadata: Metadata = { title: "Results" };

export default function ExperimentPage({ params }: PageProps<"/experiments/[id]">) {
  return (
    <div className="flex-1 bg-smoke">
      <div className="mx-auto max-w-[1680px] px-4 pb-24 pt-28 sm:px-6 lg:pt-32">
        <Suspense fallback={<ExperimentSkeleton />}>
          {params.then(({ id }) => (
            <ExperimentView key={id} id={id} />
          ))}
        </Suspense>
      </div>
    </div>
  );
}
