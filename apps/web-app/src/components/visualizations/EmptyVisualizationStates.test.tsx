import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Hypnogram } from "./Hypnogram";
import { LineChart } from "./LineChart";
import { TransitionMatrix } from "./TransitionMatrix";

describe("visualization empty states", () => {
  it("shows empty message for line chart", () => {
    render(<LineChart points={[]} />);
    expect(screen.getByText("No data for now.")).toBeInTheDocument();
  });

  it("shows empty message for hypnogram", () => {
    render(<Hypnogram mode="summary" context="home" epochs={[]} showConfidenceOverlay />);
    expect(screen.getByText("No data for now.")).toBeInTheDocument();
  });

  it("shows empty message for transition matrix", () => {
    render(<TransitionMatrix matrix={[]} />);
    expect(screen.getByText("No data for now.")).toBeInTheDocument();
  });
});
