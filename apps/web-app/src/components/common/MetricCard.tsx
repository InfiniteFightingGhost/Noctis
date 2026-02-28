type MetricCardProps = {
  title: string;
  value: string;
  delta?: string;
  tone?: "positive" | "negative" | "neutral";
};

export function MetricCard({ title, value, delta, tone = "neutral" }: MetricCardProps) {
  return (
    <article className="metric-card">
      <p className="metric-title">{title}</p>
      <p className="metric-value">{value}</p>
      {delta ? <p className={`metric-delta ${tone}`}>{delta}</p> : null}
    </article>
  );
}
