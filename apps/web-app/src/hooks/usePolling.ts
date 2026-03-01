import { useEffect, useState } from "react";

type PollingState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
  lastUpdatedAt: number | null;
};

export function usePolling<T>(fetcher: () => Promise<T>, intervalMs: number, enabled: boolean): PollingState<T> {
  const [state, setState] = useState<PollingState<T>>({
    data: null,
    loading: false,
    error: null,
    lastUpdatedAt: null,
  });

  useEffect(() => {
    if (!enabled) {
      setState((current) => ({ ...current, loading: false }));
      return;
    }

    let active = true;
    let firstRun = true;

    const poll = async () => {
      if (!active) {
        return;
      }

      if (firstRun) {
        setState((current) => ({ ...current, loading: true, error: null }));
      }

      try {
        const data = await fetcher();
        if (!active) {
          return;
        }

        setState({ data, loading: false, error: null, lastUpdatedAt: Date.now() });
      } catch (error) {
        if (!active) {
          return;
        }

        const message = error instanceof Error ? error.message : "Unable to refresh data.";
        setState((current) => ({ ...current, loading: false, error: message, lastUpdatedAt: current.lastUpdatedAt }));
      } finally {
        firstRun = false;
      }
    };

    void poll();
    const timer = window.setInterval(() => {
      void poll();
    }, intervalMs);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [enabled, fetcher, intervalMs]);

  return state;
}
