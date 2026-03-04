import { httpApiClient } from "./httpApiClient";
import { apiClient as mockApiClient } from "./mockApiClient";

const MOCK_EMAIL = "andrean1710taja1234@gmail.com";

const getStorage = () => {
  if (typeof window !== "undefined" && window.localStorage && typeof window.localStorage.getItem === "function") {
    return window.localStorage;
  }
  return null;
};

const isMockSession = () => {
  return getStorage()?.getItem("noctis_use_mock_api") === "true";
};

const baseUseMockApi =
  import.meta.env.VITE_USE_MOCK_API === "true" ||
  import.meta.env.MODE === "test";

// Create a proxy to dynamically switch between HTTP and Mock clients
export const apiClient = new Proxy(mockApiClient, {
  get(_target, prop: keyof typeof mockApiClient) {
    const useMock = baseUseMockApi || isMockSession();
    const activeClient = useMock ? (mockApiClient as any) : (httpApiClient as any);
    const value = activeClient[prop];

    if (prop === "login") {
      return async (payload: any) => {
        if (payload && payload.email === MOCK_EMAIL) {
          getStorage()?.setItem("noctis_use_mock_api", "true");
          return mockApiClient.login(payload);
        }
        getStorage()?.removeItem("noctis_use_mock_api");
        return httpApiClient.login(payload);
      };
    }

    if (prop === "logout") {
      return async () => {
        getStorage()?.removeItem("noctis_use_mock_api");
        return activeClient.logout();
      };
    }

    if (typeof value === "function") {
      return value.bind(activeClient);
    }

    return value;
  },
});
