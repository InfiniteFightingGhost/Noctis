import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

const loadAnalytics = () => {
  void import("./analytics/bootstrap")
    .then(({ bootstrapAnalytics }) => bootstrapAnalytics())
    .catch(() => undefined);
};

const requestIdleCallbackFn = (
  window as Window & { requestIdleCallback?: (callback: () => void) => number }
).requestIdleCallback;

if (typeof requestIdleCallbackFn === "function") {
  requestIdleCallbackFn(() => loadAnalytics());
} else {
  window.setTimeout(loadAnalytics, 150);
}
