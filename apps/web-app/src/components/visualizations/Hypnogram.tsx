import { useMemo, useState } from "react";
import type { Stage } from "../../api/contracts";
import { useContainerWidth } from "../../hooks/useContainerWidth";

type HypnogramEpoch = {
  epochIndex: number;
  stage: Stage;
  confidence: number;
};

type HypnogramProps = {
  mode: "summary" | "detail";
  context: "home" | "night";
  epochs: HypnogramEpoch[];
  showConfidenceOverlay: boolean;
};

const stageToY: Record<Stage, number> = {
  wake: 0,
  light: 1,
  deep: 2,
  rem: 3,
};

const MAX_RENDERED_EPOCHS = 480;

export function Hypnogram({ mode, context, epochs, showConfidenceOverlay }: HypnogramProps) {
  const { containerRef, width: containerWidth } = useContainerWidth(mode === "detail" ? 360 : 320);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState(0);
  const [selectedEpoch, setSelectedEpoch] = useState<HypnogramEpoch | null>(null);
  const width = Math.max(mode === "detail" ? 320 : 280, containerWidth);
  const height = width < 420 ? 180 : 200;
  const zoomMax = mode === "detail" ? 8 : 4;
  const minimumVisible = mode === "detail" ? 48 : 12;
  const requestedVisible = Math.max(minimumVisible, Math.floor(epochs.length / zoom));
  const visibleCount = Math.min(MAX_RENDERED_EPOCHS, requestedVisible);
  const showWindowControls = mode === "summary" || epochs.length > MAX_RENDERED_EPOCHS;
  const maxOffset = Math.max(0, epochs.length - visibleCount);
  const clampedOffset = Math.min(offset, maxOffset);
  const visibleEpochs = useMemo(
    () => epochs.slice(clampedOffset, clampedOffset + visibleCount),
    [epochs, clampedOffset, visibleCount],
  );

  const plottedEpochs = useMemo(
    () =>
      visibleEpochs.map((epoch, index) => {
        const x = (index / Math.max(1, visibleEpochs.length - 1)) * width;
        const y = 24 + stageToY[epoch.stage] * 42;
        return { epoch, x, y };
      }),
    [visibleEpochs, width],
  );

  const polylinePoints = useMemo(
    () => plottedEpochs.map((point) => `${point.x},${point.y}`).join(" "),
    [plottedEpochs],
  );

  if (epochs.length === 0) {
    return (
      <div ref={containerRef} className="chart-empty" role="status" aria-live="polite">
        No data for now.
      </div>
    );
  }

  const stageGuides = [
    { label: "Wake", y: 24 },
    { label: "Light", y: 66 },
    { label: "Deep", y: 108 },
    { label: "REM", y: 150 },
  ];

  return (
    <div ref={containerRef} className="hypnogram" data-context={context} data-mode={mode}>
      {showWindowControls ? (
        <div className="hypnogram-controls">
          <label>
            Zoom
            <input
              type="range"
              min={1}
              max={zoomMax}
              step={1}
              value={zoom}
              onChange={(event) => {
                setZoom(Number(event.target.value));
                setOffset(0);
              }}
            />
          </label>
          <label>
            Scroll
            <input
              type="range"
              min={0}
              max={maxOffset}
              step={1}
              value={clampedOffset}
              onChange={(event) => setOffset(Number(event.target.value))}
            />
          </label>
        </div>
      ) : null}
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${context} hypnogram`}>
        <rect x="0" y="0" width={width} height={height} fill="var(--surface-alt)" rx="12" />
        {stageGuides.map((guide) => (
          <g key={guide.label}>
            <line
              x1={0}
              y1={guide.y}
              x2={width}
              y2={guide.y}
              stroke="var(--border)"
              strokeOpacity="0.45"
              strokeWidth="1"
            />
            <text x={8} y={guide.y - 4} fontSize="10" fill="var(--muted)">
              {guide.label}
            </text>
          </g>
        ))}
        <polyline fill="none" stroke="var(--accent)" strokeWidth="3" points={polylinePoints} />
        {showConfidenceOverlay
          ? plottedEpochs.map(({ epoch, x, y }) => {
              const opacity = Math.max(0.2, epoch.confidence);
              return (
                <circle
                  key={epoch.epochIndex}
                  cx={x}
                  cy={y}
                  r={mode === "detail" ? 2.2 : 1.8}
                  fill="var(--accent-strong)"
                  fillOpacity={opacity}
                  tabIndex={0}
                  role="button"
                  aria-label={`Epoch ${epoch.epochIndex} ${epoch.stage} confidence ${epoch.confidence.toFixed(2)}`}
                  onClick={() => setSelectedEpoch(epoch)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      setSelectedEpoch(epoch);
                    }
                  }}
                />
              );
            })
          : null}
      </svg>
      {selectedEpoch ? (
        <p className="chart-footnote">
          Selected epoch {selectedEpoch.epochIndex}: {selectedEpoch.stage} at confidence {selectedEpoch.confidence.toFixed(2)}.
        </p>
      ) : null}
      <p className="chart-footnote">
        {mode === "summary"
          ? "Summary view supports scroll, zoom, and tap-confidence interactions."
          : "Detail view keeps epoch rendering windowed for large-night performance."}
      </p>
    </div>
  );
}
