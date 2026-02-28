import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppShell } from "../App";
import { apiClient } from "../api/apiClient";
import { setDeviceConnected, setLoggedIn } from "../auth/session";

describe("settings page", () => {
  beforeEach(() => {
    setLoggedIn(true);
    setDeviceConnected(true);
  });

  it("contains tabbed settings sections and profile data", async () => {
    render(
      <MemoryRouter initialEntries={["/settings"]}>
        <AppShell />
      </MemoryRouter>,
    );

    expect(await screen.findByRole("tab", { name: "Profile" })).toBeInTheDocument();
    expect(screen.getByText("Live user identity from backend")).toBeInTheDocument();
    expect(screen.getByText("Username")).toBeInTheDocument();
    expect(screen.getByText("Email")).toBeInTheDocument();
    expect(screen.getByText("Account Actions")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Device" }));
    expect(await screen.findByText("Connected Device")).toBeInTheDocument();
    expect(screen.getByLabelText("single-device-section")).toBeInTheDocument();
    expect(screen.getByText("Linked account:")).toBeInTheDocument();
  });

  it("shows action error message when export fails", async () => {
    const spy = vi.spyOn(apiClient, "requestDataExport").mockRejectedValue(new Error("Export unavailable"));

    render(
      <MemoryRouter initialEntries={["/settings"]}>
        <AppShell />
      </MemoryRouter>,
    );

    await screen.findByRole("tab", { name: "Profile" });
    await screen.findByText("Account Actions");
    fireEvent.click(screen.getByRole("button", { name: "Data export" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Export unavailable");
    spy.mockRestore();
  });
});
