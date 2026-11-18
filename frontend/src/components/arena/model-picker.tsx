import clsx from "clsx";
import { Check } from "lucide-react";
import { groupModelsBySize, modelSizeLabel } from "@/lib/model-presentation";
import type { ArenaModel, Health } from "@/lib/types";
import { executionLabel } from "@/lib/execution-label";

export function ModelPicker({ models, selected, onToggle }: {
  models: ArenaModel[];
  health: Health;
  selected: string[];
  onToggle: (id: string) => void;
}) {
  const selectedCount = models.filter((m) => m.availability.runnable && selected.includes(m.id)).length;
  return (
    <div>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-muted">
        <p>Approximate size · largest first</p>
        <span className="rounded-full bg-fog px-2.5 py-1 tabular-nums">
          {selectedCount} of {models.length} selected
        </span>
      </div>
      <div className="mt-5 space-y-6">
        {groupModelsBySize(models).map((group) => (
          <section key={group.id} aria-labelledby={`models-${group.id}`}>
            <div className="mb-2.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 px-1">
              <h3 id={`models-${group.id}`} className="text-sm font-medium">{group.label}</h3>
              <span className="text-[11px] text-faint">{group.detail}</span>
            </div>
            <ul className="space-y-2">
              {group.models.map((model) => {
                const runnable = model.availability.runnable;
                const checked = runnable && selected.includes(model.id);
                return (
                  <li key={model.id}>
                    <label className={clsx(
                      "flex gap-3 rounded-2xl border p-3.5 transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-brand sm:p-4",
                      !runnable ? "cursor-not-allowed border-line bg-smoke"
                        : checked ? "cursor-pointer border-brand/40 bg-brand/[0.035] hover:border-brand/70"
                          : "cursor-pointer border-line hover:border-ash hover:bg-smoke/50",
                    )}>
                      <input
                        type="checkbox"
                        className="sr-only"
                        checked={checked}
                        disabled={!runnable}
                        onChange={() => onToggle(model.id)}
                        aria-labelledby={`model-${model.id}-name`}
                        aria-describedby={`model-${model.id}-details`}
                      />
                      <span aria-hidden className={clsx(
                        "mt-1 grid size-5 shrink-0 place-items-center rounded-md border transition-colors",
                        checked ? "border-brand bg-brand text-white" : "border-ash bg-white",
                      )}>
                        {checked && <Check className="size-3.5" strokeWidth={3} />}
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="flex flex-wrap items-start justify-between gap-x-3 gap-y-1">
                          <span id={`model-${model.id}-name`} className="min-w-0 flex-1 basis-36 font-medium leading-6 [overflow-wrap:anywhere]">
                            {model.name}
                          </span>
                          <span title={`Parameters: ${model.params}`} className="mt-0.5 shrink-0 rounded-md border border-line bg-white px-2 py-0.5 font-mono text-[11px] tabular-nums text-muted">
                            {modelSizeLabel(model)}
                          </span>
                        </span>
                        <span id={`model-${model.id}-details`} className="block">
                          <span className="mt-1 block text-[13px] leading-5 text-body">
                            <span className="font-medium text-muted">Strengths: </span>
                            <span>{model.specialty}</span>
                          </span>
                          <span className="mt-1.5 block text-[11px] leading-4 text-faint" title={model.availability.execution_detail ?? undefined}>
                            {model.publisher} · {executionLabel(model.availability)}
                          </span>
                          {model.id === "unlimited-ocr-2" && (
                            <span className="mt-1.5 block text-xs leading-5 text-muted">
                              Up to {model.availability.max_pages ?? 16} pages together; optimized long-document memory savings are unavailable in this runtime.
                            </span>
                          )}
                          {!runnable && model.availability.reason && (
                            <span className="mt-1.5 block text-xs leading-5 text-amber">{model.availability.reason}</span>
                          )}
                        </span>
                      </span>
                    </label>
                  </li>
                );
              })}
            </ul>
          </section>
        ))}
      </div>
      <p className="mt-4 px-1 text-[11px] leading-5 text-faint">
        M / B = million / billion parameters. Pipeline sizes are approximate; + indicates auxiliary models.
      </p>
    </div>
  );
}
