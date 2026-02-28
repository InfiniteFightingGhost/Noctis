import { describe, expect, it } from "vitest";
import { nightSchema } from "./contracts";

describe("night contract", () => {
  it("supports explicit CAP unavailable state", () => {
    const parsed = nightSchema.parse({
      date: "2026-02-26",
      epochs: [
        {
          epochIndex: 0,
          stage: "light",
          confidence: 0.87,
          probabilities: { wake: 0.04, light: 0.8, deep: 0.08, rem: 0.08 },
        },
      ],
      transitions: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      arousalIndex: 6.1,
      capRateConditional: {
        available: false,
        reason: "Not enough CAP-labeled segments",
      },
      cardiopulmonary: {
        avgRespiratoryRate: 14,
        minSpO2: 93,
        avgHeartRate: 57,
      },
    });

    expect(parsed.capRateConditional.available).toBe(false);
    if (!parsed.capRateConditional.available) {
      expect(parsed.capRateConditional.reason.length).toBeGreaterThan(0);
    }
  });
});
