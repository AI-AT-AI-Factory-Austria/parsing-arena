"use client";

import { useCallback, useEffect, useEffectEvent, useState } from "react";
import { ApiError } from "./api";

const RETRY_AFTER_ERROR_MS = 4000;

export type PollState<T> = {
  data: T | undefined;
  error: Error | undefined;
  loading: boolean;
};

/**
 * Load `key` and keep reloading it while `nextDelay` returns a delay.
 * Transient errors keep the last data and retry; 404s stop the loop.
 */
export function usePolling<T>(
  key: string | null,
  load: () => Promise<T>,
  nextDelay: (data: T) => number | null,
) {
  const [state, setState] = useState<PollState<T>>({
    data: undefined,
    error: undefined,
    loading: true,
  });
  const [version, setVersion] = useState(0);
  const loadLatest = useEffectEvent(load);
  const delayFor = useEffectEvent(nextDelay);

  useEffect(() => {
    if (key === null) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async () => {
      try {
        const data = await loadLatest();
        if (cancelled) return;
        setState({ data, error: undefined, loading: false });
        const delay = delayFor(data);
        if (delay !== null) timer = setTimeout(tick, delay);
      } catch (error) {
        if (cancelled) return;
        setState((prev) => ({ data: prev.data, error: error as Error, loading: false }));
        if (!(error instanceof ApiError && error.status === 404)) {
          timer = setTimeout(tick, RETRY_AFTER_ERROR_MS);
        }
      }
    };
    void tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [key, version]);

  const refresh = useCallback(() => setVersion((v) => v + 1), []);
  const replace = useCallback(
    (data: T) => setState({ data, error: undefined, loading: false }),
    [],
  );
  return { ...state, refresh, replace };
}
