import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppShell } from "../App";
import { setDeviceConnected, setLoggedIn } from "../auth/session";

describe("home page buttons", () => {
  beforeEach(() => {
    setLoggedIn(true);
    setDeviceConnected(true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("toggles playback state from the mini play button", async () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <AppShell />
      </MemoryRouter>,
    );

    const playButton = await screen.findByRole("button", { name: "Play sleep tracking preview" });
    expect(playButton).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(playButton);
    expect(playButton).toHaveAttribute("aria-pressed", "true");
  });

  it("downloads the mount report when clicked", async () => {
    const createObjectUrlMock = vi.fn(() => "blob:noctis-report");
    const revokeObjectUrlMock = vi.fn();
    const anchorClickMock = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);

    Object.defineProperty(window.URL, "createObjectURL", {
      writable: true,
      value: createObjectUrlMock,
    });
    Object.defineProperty(window.URL, "revokeObjectURL", {
      writable: true,
      value: revokeObjectUrlMock,
    });

    render(
      <MemoryRouter initialEntries={["/"]}>
        <AppShell />
      </MemoryRouter>,
    );

    const downloadButton = await screen.findByRole("button", { name: "Download mount report" });
    fireEvent.click(downloadButton);

    expect(createObjectUrlMock).toHaveBeenCalledTimes(1);
    expect(revokeObjectUrlMock).toHaveBeenCalledWith("blob:noctis-report");
    expect(anchorClickMock).toHaveBeenCalledTimes(1);
    expect(await screen.findByText("Mount report downloaded.")).toBeInTheDocument();
  });
});
