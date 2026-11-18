import type { Metadata } from "next";
import { ArenaWorkspace } from "@/components/arena/arena-workspace";
import { PastExperiments } from "@/components/arena/past-experiments";

export const metadata: Metadata = {
  title: "Arena",
  description: "Compare OCR models on up to 12 pages across your documents in one batch.",
};

export default function ArenaPage() {
  return (
    <div className="flex-1 bg-smoke">
      <div className="mx-auto max-w-[1440px] px-4 pb-24 pt-28 sm:px-6 lg:pt-32">
        <p className="eyebrow">Arena</p>
        <h1 className="mt-3 text-4xl tracking-[-0.02em] sm:text-5xl">
          Put your documents in the arena
        </h1>
        <p className="mt-4 max-w-2xl text-lg leading-relaxed text-muted">
          Add documents, pick up to 12 pages, and choose your models. Compare individual
          outputs and see how each method performs across the whole batch.
        </p>
        <div className="mt-10">
          <ArenaWorkspace />
        </div>
        <div className="mt-16">
          <PastExperiments />
        </div>
      </div>
    </div>
  );
}
