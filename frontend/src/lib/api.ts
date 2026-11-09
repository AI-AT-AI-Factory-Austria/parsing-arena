import type {
  DocumentInfo,
  Experiment,
  ExperimentDetail,
  Health,
  ModelsResponse,
  SampleDocument,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function errorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const { detail } = body as { detail: unknown };
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d) => (d && typeof d === "object" && "msg" in d ? String(d.msg) : String(d)))
        .join("; ");
    }
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: { Accept: "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError("The arena API is not reachable. Is the backend running?", 0);
  }
  if (!res.ok) {
    const fallback =
      res.status >= 500
        ? "The arena API is not reachable. Is the backend running?"
        : `Request failed (${res.status}).`;
    const body = await res.json().catch(() => null);
    throw new ApiError(errorMessage(body, fallback), res.status);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

function postJson<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

/** Upload with progress reporting, which fetch() cannot do for request bodies. */
function uploadDocument(
  file: File,
  onProgress?: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<DocumentInfo> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/documents");
    xhr.responseType = "json";
    xhr.setRequestHeader("Accept", "application/json");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress?.(event.loaded / event.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response as DocumentInfo);
      } else {
        const fallback =
          xhr.status === 413 ? "The file is too large." : `Upload failed (${xhr.status}).`;
        reject(new ApiError(errorMessage(xhr.response, fallback), xhr.status));
      }
    };
    xhr.onerror = () =>
      reject(new ApiError("The arena API is not reachable. Is the backend running?", 0));
    xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
    signal?.addEventListener("abort", () => xhr.abort(), { once: true });
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

export const api = {
  health: () => request<Health>("/api/health"),
  models: () => request<ModelsResponse>("/api/models"),
  uploadDocument,
  samples: () => request<{ samples: SampleDocument[] }>("/api/samples"),
  randomSample: () => request<SampleDocument>("/api/samples/random", { cache: "no-store" }),
  createExperiment: (body: { document_id: string; models: string[]; pages?: string }) =>
    postJson<ExperimentDetail>("/api/experiments", body),
  listExperiments: (limit = 50) =>
    request<{ experiments: Experiment[] }>(`/api/experiments?limit=${limit}`),
  experiment: (id: string) =>
    request<ExperimentDetail>(`/api/experiments/${encodeURIComponent(id)}`),
  rerun: (id: string) =>
    postJson<ExperimentDetail>(`/api/experiments/${encodeURIComponent(id)}/rerun`),
  judge: (id: string) =>
    postJson<ExperimentDetail>(`/api/experiments/${encodeURIComponent(id)}/judge`),
  deleteExperiment: (id: string) =>
    request<void>(`/api/experiments/${encodeURIComponent(id)}`, { method: "DELETE" }),
};

export function pageImageUrl(documentId: string, page: number): string {
  return `/api/documents/${encodeURIComponent(documentId)}/pages/${page}`;
}

export function runFileUrl(runId: string, path: string): string {
  return `/api/runs/${encodeURIComponent(runId)}/files/${path.replace(/^\.?\//, "")}`;
}
