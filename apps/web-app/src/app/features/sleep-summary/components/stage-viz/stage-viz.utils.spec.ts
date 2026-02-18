import { findStageAtMinute, mapXToMinute } from "./stage-viz.utils";

describe("mapXToMinute", () => {
  it("maps x position to minute offset", () => {
    expect(mapXToMinute(50, 100, 200)).toBe(100);
  });

  it("clamps to the last minute on the right edge", () => {
    expect(mapXToMinute(100, 100, 200)).toBe(199);
  });
});

describe("findStageAtMinute", () => {
  it("finds the matching stage bin", () => {
    const bins = [
      { startMinFromBedtime: 0, durationMin: 30, stage: "awake" as const },
      { startMinFromBedtime: 30, durationMin: 60, stage: "light" as const },
    ];

    expect(findStageAtMinute(bins, 35)).toBe("light");
  });
});
