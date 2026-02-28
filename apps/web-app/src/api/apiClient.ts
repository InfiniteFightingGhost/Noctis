import { httpApiClient } from "./httpApiClient";
import { apiClient as mockApiClient } from "./mockApiClient";

const hasApiBaseUrl = typeof import.meta.env.VITE_API_BASE_URL === "string" && import.meta.env.VITE_API_BASE_URL.length > 0;
const useMockApi =
  import.meta.env.VITE_USE_MOCK_API === "true" ||
  import.meta.env.MODE === "test" ||
  (import.meta.env.DEV && !hasApiBaseUrl);

export const apiClient = useMockApi ? mockApiClient : httpApiClient;
