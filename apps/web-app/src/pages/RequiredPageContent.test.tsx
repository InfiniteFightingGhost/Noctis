import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it } from "vitest";
import { AppShell } from "../App";
import { setDeviceConnected, setLoggedIn } from "../auth/session";

const requiredPages = [
  { route: "/", marker: "Summary Hypnogram" },
  { route: "/trends", marker: "Sleep Score Longitudinal" },
  { route: "/night", marker: "Epoch Hypnogram" },
  { route: "/settings", marker: "Live user identity from backend" },
] as const;

describe("required page content", () => {
  beforeEach(() => {
    setLoggedIn(true);
    setDeviceConnected(true);
  });

  it.each(requiredPages)("renders required content for $route", async ({ route, marker }) => {
    render(
      <MemoryRouter initialEntries={[route]}>
        <AppShell />
      </MemoryRouter>,
    );

    expect(await screen.findByText(marker)).toBeInTheDocument();
  });
});
