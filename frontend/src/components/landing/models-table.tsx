import clsx from "clsx";
import { ArrowUpRight, ServerOff } from "lucide-react";
import { modelSizeLabel, sortModelsBySize } from "@/lib/model-presentation";
import { getModels } from "@/lib/server-api";
import type { ArenaModel } from "@/lib/types";
import { Chip, Dot } from "@/components/ui";

const BACKEND_LABEL: Record<string, string> = {
  local: "Runs locally",
  cpu: "Runs on the CPU",
  endpoint: "vLLM endpoint",
  api: "Hosted API",
};

function StatusCell({ model }: { model: ArenaModel }) {
  const { availability } = model;
  if (!availability.runnable) {
    return (
      <Chip tone="amber" title={availability.reason ?? undefined}>
        <Dot /> {model.kind === "reference" ? "Needs API key" : "Setup needed"}
      </Chip>
    );
  }
  return (
    <div className="flex flex-col items-start gap-1.5">
      <Chip tone="mint"><Dot /> In arena</Chip>
      {availability.backend && (
        <span className="text-[11px] text-faint">
          {availability.runtime_label ?? BACKEND_LABEL[availability.backend]}
        </span>
      )}
    </div>
  );
}

function ModelRow({ model }: { model: ArenaModel }) {
  const reference = model.kind === "reference";
  return (
    <tr className={clsx(
      "group border-t border-line align-top transition-colors",
      reference ? "bg-mint-wash" : "hover:bg-smoke/70",
    )}>
      <th scope="row" className="py-5 pl-5 pr-5 font-normal sm:pl-6">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          {model.hf_url ? (
            <a href={model.hf_url} target="_blank" rel="noreferrer"
              className="inline-flex items-center gap-1 font-medium text-ink hover:text-brand">
              {model.name}
              <ArrowUpRight aria-hidden className="size-3.5 shrink-0 text-stone group-hover:text-brand" />
              <span className="sr-only">(Hugging Face model card)</span>
            </a>
          ) : <span className="font-medium text-ink">{model.name}</span>}
          {reference && <Chip tone="mint">Reference</Chip>}
        </div>
        <p className="mt-1 text-xs text-faint">{model.publisher}</p>
      </th>
      <td className="py-5 pr-5">
        <span title={"Parameters: " + model.params} className="inline-block whitespace-nowrap rounded-md border border-line bg-white px-2 py-1 font-mono text-xs tabular-nums text-muted">
          {modelSizeLabel(model)}
        </span>
      </td>
      <td className="py-5 pr-6 text-sm leading-6 text-body">{model.specialty}</td>
      <td className="hidden py-5 pr-5 text-xs leading-5 text-muted xl:table-cell">{model.license}</td>
      <td className="py-5 pr-5 sm:pr-6"><StatusCell model={model} /></td>
    </tr>
  );
}

export async function ModelsTable() {
  const data = await getModels();
  if (!data) {
    return (
      <div className="card flex items-start gap-4 p-6">
        <ServerOff aria-hidden className="mt-0.5 size-5 shrink-0 text-amber" />
        <div>
          <p className="font-medium">The arena API is offline.</p>
          <p className="mt-1 text-sm text-muted">
            Start the backend (see the README) and reload this page to see the model line-up.
          </p>
        </div>
      </div>
    );
  }
  return (
    <div className="card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[740px] border-collapse text-left">
          <caption className="sr-only">OCR methods, ordered by approximate model size, with their strengths and runtime</caption>
          <thead>
            <tr className="eyebrow text-[11px]">
              <th scope="col" className="w-[28%] py-3.5 pl-5 pr-5 font-normal sm:pl-6">Model</th>
              <th scope="col" className="py-3.5 pr-5 font-normal">Size</th>
              <th scope="col" className="w-[30%] py-3.5 pr-6 font-normal">Strengths</th>
              <th scope="col" className="hidden py-3.5 pr-5 font-normal xl:table-cell">License</th>
              <th scope="col" className="py-3.5 pr-5 font-normal sm:pr-6">Runtime</th>
            </tr>
          </thead>
          <tbody>
            <ModelRow model={data.reference} />
            {sortModelsBySize(data.models).map((model) => <ModelRow key={model.id} model={model} />)}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function ModelsTableSkeleton() {
  return (
    <div className="card overflow-hidden p-6" aria-busy="true" aria-label="Loading models">
      <div className="space-y-5">
        {Array.from({ length: 8 }, (_, i) => (
          <div key={i} className="flex animate-pulse items-center gap-6">
            <div className="h-3 w-48 rounded bg-fog" />
            <div className="h-3 w-16 rounded bg-fog" />
            <div className="h-3 w-64 rounded bg-fog" />
            <div className="ml-auto h-5 w-20 rounded-full bg-fog" />
          </div>
        ))}
      </div>
    </div>
  );
}
