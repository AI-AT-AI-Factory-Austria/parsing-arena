import clsx from "clsx";
import { ArrowUpRight, ServerOff } from "lucide-react";
import { formatScore } from "@/lib/format";
import { getModels } from "@/lib/server-api";
import type { ArenaModel, Scores } from "@/lib/types";
import { Chip, Dot } from "@/components/ui";

const DIMENSIONS: { key: keyof Omit<Scores, "overall">; label: string }[] = [
  { key: "tables", label: "Tables" },
  { key: "charts", label: "Charts" },
  { key: "content", label: "Content" },
  { key: "formatting", label: "Formatting" },
  { key: "grounding", label: "Grounding" },
];

const BACKEND_LABEL: Record<string, string> = {
  local: "Runs locally",
  cpu: "Runs on the CPU",
  endpoint: "vLLM endpoint",
  api: "Hosted API",
};

function StatusCell({ model }: { model: ArenaModel }) {
  const { availability } = model;
  if (model.kind === "open" && !model.implemented) {
    return <Chip tone="outline">Coming soon</Chip>;
  }
  if (!availability.runnable) {
    return (
      <Chip tone="amber" title={availability.reason ?? undefined}>
        <Dot /> {model.kind === "reference" ? "Needs API key" : "Setup needed"}
      </Chip>
    );
  }
  return (
    <div className="flex flex-col items-start gap-1">
      <Chip tone="mint">
        <Dot /> In arena
      </Chip>
      {availability.backend && (
        <span className="pl-1 text-[11px] text-faint">{BACKEND_LABEL[availability.backend]}</span>
      )}
    </div>
  );
}

function ScoreBar({ value, reference }: { value: number; reference?: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-10 text-right font-mono text-[15px] font-medium tabular-nums text-ink">
        {formatScore(value)}
      </span>
      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-fog">
        <div
          className={clsx("h-full rounded-full", reference ? "bg-leaf" : "bg-brand")}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

function ModelRow({ model, rank }: { model: ArenaModel; rank: number | null }) {
  const reference = model.kind === "reference";
  return (
    <tr
      className={clsx(
        "group border-t border-line align-top transition-colors",
        reference ? "bg-mint-wash" : "hover:bg-smoke/70",
      )}
    >
      <td className="py-4 pl-5 pr-2 font-mono text-xs text-faint tabular-nums sm:pl-6">
        {reference ? "Ref" : (rank ?? "–")}
      </td>
      <td className="min-w-[17rem] py-4 pr-6">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          {model.hf_url ? (
            <a
              href={model.hf_url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 whitespace-nowrap font-medium text-ink hover:text-brand"
            >
              {model.name}
              <ArrowUpRight aria-hidden className="size-3.5 text-stone group-hover:text-brand" />
              <span className="sr-only">(Hugging Face model card)</span>
            </a>
          ) : (
            <span className="whitespace-nowrap font-medium text-ink">{model.name}</span>
          )}
          {reference && <Chip tone="mint">Reference</Chip>}
          {!model.verified && (
            <Chip
              tone="amber"
              className="!px-2 !text-[11px]"
              title="Submitted result that ParseBench has not independently verified yet."
            >
              Unverified
            </Chip>
          )}
        </div>
        {model.lineage && (
          <p title={model.lineage} className="mt-1 max-w-[19rem] truncate text-xs text-faint">
            {model.lineage}
          </p>
        )}
      </td>
      <td className="whitespace-nowrap py-4 pr-6 text-sm text-body">{model.publisher}</td>
      <td className="whitespace-nowrap py-4 pr-6 font-mono text-sm tabular-nums text-body">
        {model.params}
      </td>
      <td className="py-4 pr-6">
        {model.scores ? (
          <ScoreBar value={model.scores.overall} reference={reference} />
        ) : (
          <span className="text-xs text-faint" title="Not on the ParseBench leaderboard">
            Not benchmarked
          </span>
        )}
      </td>
      {DIMENSIONS.map((d) => (
        <td
          key={d.key}
          className="hidden py-4 pr-4 text-right font-mono text-[13px] tabular-nums text-muted xl:table-cell"
        >
          {model.scores ? formatScore(model.scores[d.key]) : "–"}
        </td>
      ))}
      <td className="whitespace-nowrap py-4 pr-6 text-sm text-body">{model.license}</td>
      <td className="py-4 pr-5 sm:pr-6">
        <StatusCell model={model} />
      </td>
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
  const open = [...data.models].sort(
    (a, b) => (b.scores?.overall ?? -1) - (a.scores?.overall ?? -1),
  );
  return (
    <div className="card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[920px] border-collapse text-left">
          <caption className="sr-only">
            Open OCR models in the arena with their ParseBench scores
          </caption>
          <thead>
            <tr className="eyebrow text-[11px]">
              <th scope="col" className="py-3.5 pl-5 pr-2 font-normal sm:pl-6">
                #
              </th>
              <th scope="col" className="py-3.5 pr-6 font-normal">
                Model
              </th>
              <th scope="col" className="py-3.5 pr-6 font-normal">
                Publisher
              </th>
              <th scope="col" className="py-3.5 pr-6 font-normal">
                Params
              </th>
              <th scope="col" className="py-3.5 pr-6 font-normal">
                ParseBench
              </th>
              {DIMENSIONS.map((d) => (
                <th
                  key={d.key}
                  scope="col"
                  className="hidden py-3.5 pr-4 text-right font-normal xl:table-cell"
                >
                  {d.label}
                </th>
              ))}
              <th scope="col" className="py-3.5 pr-6 font-normal">
                License
              </th>
              <th scope="col" className="py-3.5 pr-5 font-normal sm:pr-6">
                Status
              </th>
            </tr>
          </thead>
          <tbody>
            <ModelRow model={data.reference} rank={null} />
            {open.map((model, i) => (
              <ModelRow key={model.id} model={model} rank={model.scores ? i + 1 : null} />
            ))}
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
            <div className="h-3 w-6 rounded bg-fog" />
            <div className="h-3 w-48 rounded bg-fog" />
            <div className="h-3 w-24 rounded bg-fog" />
            <div className="h-3 w-12 rounded bg-fog" />
            <div className="h-3 w-40 rounded bg-fog" />
            <div className="ml-auto h-5 w-20 rounded-full bg-fog" />
          </div>
        ))}
      </div>
    </div>
  );
}
