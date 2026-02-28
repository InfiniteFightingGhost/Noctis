import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { PrimaryNavigation } from "./PrimaryNavigation";

describe("primary navigation", () => {
  it("does not show a devices tab", () => {
    render(
      <MemoryRouter>
        <PrimaryNavigation />
      </MemoryRouter>,
    );

    expect(screen.queryByRole("link", { name: /devices/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /insights/i })).not.toBeInTheDocument();
    expect(screen.getAllByRole("link")).toHaveLength(4);
  });
});
