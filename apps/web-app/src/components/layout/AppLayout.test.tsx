import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppLayout } from "./AppLayout";

describe("app layout controls", () => {
  beforeEach(() => {
    const storage = window.localStorage as { removeItem?: (key: string) => void };
    if (typeof storage.removeItem === "function") {
      storage.removeItem("noctis-theme-preference");
    }

    delete document.documentElement.dataset.theme;

    vi.stubGlobal(
      "matchMedia",
      vi.fn().mockImplementation(() => ({
        matches: false,
        media: "(prefers-color-scheme: dark)",
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function renderLayout() {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<div>Page content</div>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
  }

  it("switches to dark theme from the header button", () => {
    renderLayout();

    fireEvent.click(screen.getByRole("button", { name: "Switch to dark theme" }));

    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("opens and closes the mobile navigation button state", () => {
    renderLayout();

    const menuButton = screen.getByRole("button", { name: "Open menu" });
    fireEvent.click(menuButton);

    const tabsShell = document.getElementById("primary-navigation");
    if (!tabsShell) {
      throw new Error("Expected primary navigation wrapper to be rendered");
    }

    expect(menuButton).toHaveAttribute("aria-expanded", "true");
    expect(tabsShell).toHaveClass("is-open");

    fireEvent.click(screen.getByRole("button", { name: "Close menu" }));
    expect(tabsShell).not.toHaveClass("is-open");
  });
});
