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

    // Initial 30D view
    expect(await screen.findByText(/Moving average 30 nights/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "7D" }));

    // Switched to 7D view
    expect(await screen.findByText(/Moving average 7 nights/i)).toBeInTheDocument();
  });
});
