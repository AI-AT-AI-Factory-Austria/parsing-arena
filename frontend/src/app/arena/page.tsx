import type { Metadata } from "next";
import { ArenaWorkspace } from "@/components/arena/arena-workspace";
import { PastExperiments } from "@/components/arena/past-experiments";

export const metadata: Metadata = {
  title: "Arena",
  description: "Upload a document and run open OCR models side by side with LlamaParse.",
};

export default function ArenaPage() {
  return (
    <div className="flex-1 bg-smoke">
      <div className="mx-auto max-w-[1440px] px-4 pb-24 pt-28 sm:px-6 lg:pt-32">
        <p className="eyebrow">Arena</p>
        <h1 className="mt-3 text-4xl tracking-[-0.02em] sm:text-5xl">
          Put a document in the arena
        </h1>
        <p className="mt-4 max-w-2xl text-lg leading-relaxed text-muted">
          Drop a file, choose the models and pages, then start. The results page opens right away
          and fills in live as each model finishes.
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
