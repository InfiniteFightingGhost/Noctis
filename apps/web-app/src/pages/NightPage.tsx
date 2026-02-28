import { useCallback } from "react";
import { apiClient } from "../api/apiClient";
import { ChartContainer } from "../components/common/ChartContainer";
import { MetricCard } from "../components/common/MetricCard";
import { PageState } from "../components/common/PageState";
import { Hypnogram } from "../components/visualizations/Hypnogram";
import { TransitionMatrix } from "../components/visualizations/TransitionMatrix";
import { useAsyncResource } from "../hooks/useAsyncResource";

export default function NightPage() {
  const loadNight = useCallback(() => apiClient.getNight(), []);
  const { loading, error, data } = useAsyncResource(loadNight);

  if (loading) {
    return <PageState mode="loading" />;
  }

  if (error || !data) {
    return <PageState mode="error" message={error ?? "Unable to load night details."} />;
  }

  const capRateValue = data.capRateConditional.available
    ? data.capRateConditional.value.toFixed(2)
    : "Unavailable";
  const capRateDelta = data.capRateConditional.available
    ? "if available"
    : `if available: ${data.capRateConditional.reason}`;

  return (
    <section className="page-grid night-page">
      <section className="chart-card night-intro" aria-label="Night analysis introduction">
        <div className="night-intro-copy">
          <p className="eyebrow">Night Deep Dive</p>
          <h1>Single-Night Analysis</h1>
          <p>Inspect epoch-level transitions, respiratory behavior, and arousal dynamics for {data.date}.</p>
        </div>

        <div className="metric-grid four-up night-kpis">
          <MetricCard title="Arousal Index" value={data.arousalIndex.toFixed(1)} />
          <MetricCard title="CAP Rate (Cond.)" value={capRateValue} delta={capRateDelta} />
          <MetricCard title="Avg Respiratory Rate" value={`${data.cardiopulmonary.avgRespiratoryRate} br/min`} />
          <MetricCard title="Min SpO2 / Avg HR" value={`${data.cardiopulmonary.minSpO2}% / ${data.cardiopulmonary.avgHeartRate} bpm`} />
        </div>
      </section>

      <ChartContainer title="Epoch Hypnogram" subtitle={`Night of ${data.date}`}>
        <Hypnogram mode="detail" context="night" epochs={data.epochs} showConfidenceOverlay />
      </ChartContainer>

      <ChartContainer title="Transition Heatmap + Matrix">
        <TransitionMatrix matrix={data.transitions} />
      </ChartContainer>
    </section>
  );
}
