import { formatMinutesAsClock } from "./format";

describe("formatMinutesAsClock", () => {
  it("formats minutes into hours and minutes", () => {
    expect(formatMinutesAsClock(125)).toBe("2:05");
  });

  it("returns placeholder for invalid input", () => {
    expect(formatMinutesAsClock(null)).toBe("--");
  });
});
