import { Suspense } from "react";
import { BrowserRouter, useRoutes } from "react-router-dom";
import { routeObjects } from "./app/routes";

function RouteLoader() {
  const element = useRoutes(routeObjects);
  return element;
}

export function AppShell() {
  return (
    <Suspense fallback={<div className="page-state">Loading page...</div>}>
      <RouteLoader />
    </Suspense>
  );
}

export function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  );
}
