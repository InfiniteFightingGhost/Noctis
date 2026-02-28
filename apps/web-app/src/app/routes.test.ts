import { describe, expect, it } from "vitest";
import { ROUTE_TABLE, routeObjects } from "./routes";

describe("route table", () => {
  it("matches the exact active route contract", () => {
    expect(ROUTE_TABLE).toEqual([
      { path: "/", label: "Home" },
      { path: "/trends", label: "Trends" },
      { path: "/night", label: "Night" },
      { path: "/settings", label: "Settings" },
    ]);
  });

  it("includes a wildcard not-found route", () => {
    const wildcardRoute = routeObjects.find((route) => route.path === "*");
    expect(wildcardRoute).toBeDefined();
  });
});
