import { httpApiClient } from "./httpApiClient";
import { apiClient as mockApiClient } from "./mockApiClient";

const hasApiBaseUrl = typeof import.meta.env.VITE_API_BASE_URL === "string" && import.meta.env.VITE_API_BASE_URL.length > 0;
const useMockApi =
  import.meta.env.VITE_USE_MOCK_API === "true" ||
  import.meta.env.MODE === "test";

if (!useMockApi && !hasApiBaseUrl && import.meta.env.DEV) {
  // eslint-disable-next-line no-console
  console.warn("VITE_API_BASE_URL is not set. Using same-origin '/v1' endpoints (requires Vite proxy in dev).");
}

export const apiClient = useMockApi ? mockApiClient : httpApiClient;
