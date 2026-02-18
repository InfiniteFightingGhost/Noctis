import { resolvePrimaryActionLabel } from "./sleep-summary.utils";

describe("resolvePrimaryActionLabel", () => {
  it("returns Fix Tracking when data quality is not ok", () => {
    expect(resolvePrimaryActionLabel("error", { label: "Improve", action: "open" }))
      .toBe("Fix Tracking");
  });

  it("returns the backend label when ok", () => {
    expect(resolvePrimaryActionLabel("ok", { label: "Improve Tonight", action: "open" }))
      .toBe("Improve Tonight");
  });
});
