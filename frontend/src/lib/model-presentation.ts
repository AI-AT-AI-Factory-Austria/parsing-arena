import type { ArenaModel } from "./types";

// Place this variable-size Docling pipeline beside Docling Standard without
// presenting an invented parameter count to the user.
function sortSize(model: ArenaModel): number {
  if (model.id === "opendataloader_docling") return 0.3;
  return model.params_b ?? 0;
}

export function sortModelsBySize(models: ArenaModel[]): ArenaModel[] {
  return [...models].sort((a, b) => sortSize(b) - sortSize(a));
}

export function modelSizeLabel(model: ArenaModel): string {
  if (model.kind === "reference") return "API";
  if (model.id === "tesseract") return "Classic OCR";
  if (model.params_b === null) return "Pipeline";
  if (model.params_b === 0) return "No model";
  const billions = model.params_b;
  const count = Number((billions >= 1 ? billions : billions * 1000).toFixed(1));
  const approximate = model.params.startsWith("≈") ? "≈" : "";
  const auxiliary = model.params.includes("auxiliaries") ? "+" : "";
  return `${approximate}${count}${billions >= 1 ? "B" : "M"}${auxiliary}`;
}

export function groupModelsBySize(models: ArenaModel[]) {
  const sorted = sortModelsBySize(models);
  return [
    { id: "large", label: "Large models", detail: "1B+ parameters", models: sorted.filter((m) => sortSize(m) >= 1) },
    { id: "compact", label: "Compact models & pipelines", detail: "100M–1B · variable-size pipelines", models: sorted.filter((m) => sortSize(m) >= 0.1 && sortSize(m) < 1) },
    { id: "lightweight", label: "Lightweight & classic", detail: "Under 100M · classic tools", models: sorted.filter((m) => sortSize(m) < 0.1) },
  ].filter((group) => group.models.length > 0);
}
