import { useContainerWidth } from "../../hooks/useContainerWidth";

type DataPoint = {
  x: string;
  y: number;
};

type LineChartProps = {
  points: DataPoint[];
  yDomain?: { min: number; max: number };
  showBand?: { lower: number; upper: number };
};

export function LineChart({ points, yDomain, showBand }: LineChartProps) {
  const { containerRef, width: containerWidth } = useContainerWidth(320);

  if (points.length === 0) {
    return (
      <div ref={containerRef} className="chart-empty" role="status" aria-live="polite">
        No data for now.
      </div>
    );
  }

  const width = Math.max(280, containerWidth);
  const height = width < 480 ? 200 : 220;
  const domainMin = yDomain?.min ?? Math.min(...points.map((point) => point.y));
  const domainMax = yDomain?.max ?? Math.max(...points.map((point) => point.y));
  const span = Math.max(1, domainMax - domainMin);
  const chartTop = 16;
  const chartBottom = 20;
  const chartHeight = height - chartTop - chartBottom;
  const gridRows = [0.2, 0.4, 0.6, 0.8];
  const mapped = points.map((point, index) => {
    const x = (index / Math.max(1, points.length - 1)) * width;
    const y = height - ((point.y - domainMin) / span) * chartHeight - chartBottom;
    return { ...point, x, y };
  });

  const areaPath =
    mapped.length > 1
      ? `M ${mapped[0].x} ${height - chartBottom} L ${mapped.map((point) => `${point.x} ${point.y}`).join(" L ")} L ${mapped[mapped.length - 1].x} ${height - chartBottom} Z`
      : "";

  return (
    <div ref={containerRef} className="chart-svg-shell">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="line chart">
        <defs>
          <linearGradient id="lineAreaGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.28" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <rect x="0" y="0" width={width} height={height} fill="var(--surface-alt)" rx="12" />
        {gridRows.map((ratio) => {
          const y = chartTop + chartHeight * ratio;
          return (
            <line
              key={ratio}
              x1={0}
              y1={y}
              x2={width}
              y2={y}
              stroke="var(--border)"
              strokeOpacity="0.5"
              strokeWidth="1"
            />
          );
        })}
        {showBand ? (
          <rect
            x="0"
            y={height - ((showBand.upper - domainMin) / span) * chartHeight - chartBottom}
            width={width}
            height={((showBand.upper - showBand.lower) / span) * chartHeight}
            fill="var(--accent-muted)"
            fillOpacity="0.28"
          />
        ) : null}
        {areaPath ? <path d={areaPath} fill="url(#lineAreaGradient)" /> : null}
        <polyline
          fill="none"
          stroke="var(--accent)"
          strokeWidth="3"
          points={mapped.map((point) => `${point.x},${point.y}`).join(" ")}
        />
        {mapped.length > 0 ? (
          <circle
            cx={mapped[mapped.length - 1].x}
            cy={mapped[mapped.length - 1].y}
            r="4"
            fill="var(--accent-strong)"
          />
        ) : null}
      </svg>
    </div>
  );
}
