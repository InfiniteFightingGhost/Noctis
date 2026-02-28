import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AppShell } from "../App";
import { clearAuthSession } from "../auth/session";

describe("not found routing", () => {
  it("renders a not found page for unknown routes", async () => {
    clearAuthSession();

    render(
      <MemoryRouter initialEntries={["/this-route-does-not-exist"]}>
        <AppShell />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Page not found")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Go to Login" })).toHaveAttribute("href", "/login");
  });
});
