import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it } from "vitest";
import { AppShell } from "../App";
import { setDeviceConnected, setLoggedIn } from "../auth/session";

describe("trends page", () => {
  beforeEach(() => {
    setLoggedIn(true);
    setDeviceConnected(true);
  });

  it("includes required longitudinal metrics", async () => {
    render(
      <MemoryRouter initialEntries={["/trends"]}>
        <AppShell />
      </MemoryRouter>,
    );

    await screen.findByText("Sleep Score Longitudinal");
    expect(screen.getByText("TST Trend")).toBeInTheDocument();
    expect(screen.getByText("REM Percent Longitudinal")).toBeInTheDocument();
    expect(screen.getByText("Deep Percent Longitudinal")).toBeInTheDocument();
    expect(screen.getByText("Fragmentation Longitudinal")).toBeInTheDocument();
    expect(screen.getByText("HR Mean Longitudinal")).toBeInTheDocument();
    expect(screen.getByText("HRV Longitudinal")).toBeInTheDocument();
  });

  it("updates rendered trend window when filter changes", async () => {
    render(
      <MemoryRouter initialEntries={["/trends"]}>
        <AppShell />
      </MemoryRouter>,
    );

    expect(await screen.findByText("2026-02-01")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "7D" }));

    expect(await screen.findByText("2026-02-06")).toBeInTheDocument();
  });
});
