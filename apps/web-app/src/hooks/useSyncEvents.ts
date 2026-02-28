import { useEffect, useState } from "react";
import { getAccessToken } from "../auth/session";

type SyncSnapshot = {
  status: "ok" | "idle";
  last_ingest_at: string | null;
  last_prediction_at: string | null;
};

type SyncEventsState = {
  connected: boolean;
  snapshot: SyncSnapshot | null;
  error: string | null;
};

const initialState: SyncEventsState = {
  connected: false,
  snapshot: null,
  error: null,
};

function buildApiUrl(path: string): string {
  const baseUrl = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
  const apiPrefix = ((import.meta.env.VITE_API_V1_PREFIX as string | undefined)?.trim() || "/v1").replace(/\/$/, "");
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (!baseUrl) {
    return `${apiPrefix}${normalizedPath}`;
  }

  const alreadyPrefixed = baseUrl.endsWith(apiPrefix);
  return alreadyPrefixed ? `${baseUrl}${normalizedPath}` : `${baseUrl}${apiPrefix}${normalizedPath}`;
}

function parseSseBlock(block: string): { event: string; data: string } | null {
  const lines = block.split("\n").map((line) => line.trimEnd());
  let event = "message";
  const data: string[] = [];

  for (const line of lines) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
      continue;
    }
    if (line.startsWith("data:")) {
      data.push(line.slice(5).trim());
    }
  }

  if (data.length === 0) {
    return null;
  }

  return { event, data: data.join("\n") };
}

export function useSyncEvents(): SyncEventsState {
  const [state, setState] = useState<SyncEventsState>(initialState);

  useEffect(() => {
    const token = getAccessToken();
    const hasApiBaseUrl = typeof import.meta.env.VITE_API_BASE_URL === "string" && import.meta.env.VITE_API_BASE_URL.length > 0;
    const useMockApi =
      import.meta.env.VITE_USE_MOCK_API === "true" ||
      import.meta.env.MODE === "test" ||
      (import.meta.env.DEV && !hasApiBaseUrl);

    if (!token || useMockApi) {
      return;
    }

    const controller = new AbortController();

    async function connect() {
      try {
        const response = await fetch(buildApiUrl("/sync/events"), {
          method: "GET",
          headers: {
            Accept: "text/event-stream",
            Authorization: `Bearer ${token}`,
          },
          signal: controller.signal,
        });

        if (!response.ok || !response.body) {
          setState({ connected: false, snapshot: null, error: "Live sync unavailable." });
          return;
        }

        setState((current) => ({ ...current, connected: true, error: null }));

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (!controller.signal.aborted) {
          const { value, done } = await reader.read();
          if (done) {
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split("\n\n");
          buffer = blocks.pop() ?? "";

          for (const block of blocks) {
            const parsed = parseSseBlock(block);
            if (!parsed || parsed.event !== "sync") {
              continue;
            }

            try {
              const snapshot = JSON.parse(parsed.data) as SyncSnapshot;
              setState({ connected: true, snapshot, error: null });
            } catch {
              setState({ connected: true, snapshot: null, error: "Live sync payload error." });
            }
          }
        }
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }

        const message = error instanceof Error ? error.message : "Live sync unavailable.";
        setState({ connected: false, snapshot: null, error: message });
      }
    }

    connect();

    return () => {
      controller.abort();
    };
  }, []);

  return state;
}
