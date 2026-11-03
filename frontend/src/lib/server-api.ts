import type { ModelsResponse } from "./types";

const BACKEND_URL = (process.env.BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

/** Server-side read of the model catalog; `null` when the backend is offline. */
export async function getModels(): Promise<ModelsResponse | null> {
  try {
    const res = await fetch(`${BACKEND_URL}/api/models`, { cache: "no-store" });
    return res.ok ? ((await res.json()) as ModelsResponse) : null;
  } catch {
    return null;
  }
}
