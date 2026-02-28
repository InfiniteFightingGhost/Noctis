import { useCallback, useState, type CSSProperties } from "react";
import { apiClient } from "../api/apiClient";
import { ChartContainer } from "../components/common/ChartContainer";
import { MetricCard } from "../components/common/MetricCard";
import { PageState } from "../components/common/PageState";
import { Hypnogram } from "../components/visualizations/Hypnogram";
import { useAsyncResource } from "../hooks/useAsyncResource";
import { useSyncEvents } from "../hooks/useSyncEvents";

export default function HomePage() {
  const loadHome = useCallback(() => apiClient.getHome(), []);
  const { loading, error, data } = useAsyncResource(loadHome);
  const syncEvents = useSyncEvents();
  const [isPreviewPlaying, setIsPreviewPlaying] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState("");

  if (loading) {
    return <PageState mode="loading" />;
  }

  if (error || !data) {
    return <PageState mode="error" message={error ?? "Unable to load home view."} />;
  }

  const deltas = data.metrics.deltaVs7DayBaseline;
  const sleepScore = Math.max(0, Math.min(100, data.metrics.sleepScore));
  const ringRadius = 52;
  const ringCircumference = 2 * Math.PI * ringRadius;
  const ringTarget = ringCircumference * (1 - sleepScore / 100);
  const ringStyle = {
    "--ring-circumference": ringCircumference,
    "--ring-target": ringTarget,
  } as CSSProperties;

  const formatStageLabel = (stage: string) => {
    if (stage.toLowerCase() === "rem") {
      return "REM";
    }
    return stage.charAt(0).toUpperCase() + stage.slice(1);
  };

  const handleDownloadMountReport = () => {
    if (typeof window === "undefined" || typeof window.URL.createObjectURL !== "function") {
      setDownloadStatus("Download is not available in this environment.");
      return;
    }

    const reportBody = [
      "Noctis mount report",
      `Date: ${data.date}`,
      `Sleep score: ${data.metrics.sleepScore}`,
      `Total sleep minutes: ${data.metrics.totalSleepMinutes}`,
      `Sleep efficiency: ${data.metrics.sleepEfficiency}`,
      `Fragmentation index: ${data.continuityMetrics.fragmentationIndex.toFixed(2)}`,
      `Entropy: ${data.continuityMetrics.entropy.toFixed(2)}`,
      `WASO minutes: ${data.continuityMetrics.wasoMinutes}`,
    ].join("\n");

    const reportBlob = new Blob([reportBody], { type: "text/plain;charset=utf-8" });
    const objectUrl = window.URL.createObjectURL(reportBlob);
    const anchor = document.createElement("a");

    anchor.href = objectUrl;
    anchor.download = `noctis-mount-report-${data.date}.txt`;
    anchor.click();
    window.URL.revokeObjectURL(objectUrl);
    setDownloadStatus("Mount report downloaded.");
  };

  const syncStatusLabel = (() => {
    if (syncEvents.snapshot?.last_ingest_at) {
      return `Live sync: ${new Date(syncEvents.snapshot.last_ingest_at).toLocaleTimeString()}`;
    }

    if (syncEvents.connected && syncEvents.snapshot?.status === "idle") {
      return "Live sync connected, no data yet.";
    }

    return `Last sync snapshot: ${data.date}`;
  })();

  return (
    <section className="page-grid home-page">
      <section className="home-hero-split">
        <article className="home-hero-copy glass-card">
          <p className="eyebrow">Sleep OS dashboard</p>
          <h1>Personal sleep tracking.</h1>
          <p>
            Build healthier nights with restorative rhythms, confidence-aware insights, and a calm dashboard
            tuned to your nightly baseline.
          </p>

          <div className="mini-product-card">
            <p className="mini-label">SLEEP TRACKING</p>
            <div className="mini-device" aria-hidden="true">
              <span className="mini-device-screen" />
              <span className="mini-device-band" />
            </div>
            <div className="mini-content">
              <p>Program focus: continuity and recovery quality.</p>
              <button
                type="button"
                className={isPreviewPlaying ? "mini-play is-playing" : "mini-play"}
                aria-label="Play sleep tracking preview"
                aria-pressed={isPreviewPlaying}
                onClick={() => setIsPreviewPlaying((current) => !current)}
              >
                {isPreviewPlaying ? "||" : ">"}
              </button>
            </div>
            <div className="mini-progress" aria-hidden="true">
              <span className={isPreviewPlaying ? "mini-progress-fill is-playing" : "mini-progress-fill"} />
            </div>
          </div>
        </article>

        <section className="home-dashboard-grid">
          <article className="big-analytics-card glass-card">
            <div className="analytics-header">
              <h2>Night analytics</h2>
              <p>Sleep score confidence-backed summary.</p>
            </div>

            <div className="score-ring-layout">
              <div className="score-ring" style={ringStyle}>
                <svg viewBox="0 0 140 140" role="img" aria-label={`Sleep score ${sleepScore}%`}>
                  <defs>
                    <linearGradient id="sleepRingGradient" x1="10%" y1="0%" x2="90%" y2="100%">
                      <stop offset="0%" stopColor="#7ea6a1" />
                      <stop offset="55%" stopColor="#4e8a84" />
                      <stop offset="100%" stopColor="#d8bd97" />
                    </linearGradient>
                  </defs>
                  <circle cx="70" cy="70" r={ringRadius} className="progress-ring-track" />
                  <circle
                    cx="70"
                    cy="70"
                    r={ringRadius}
                    className="progress-ring-fill"
                    style={{ stroke: "url(#sleepRingGradient)" }}
                  />
                </svg>
                <div className="score-copy">
                  <strong>{sleepScore}%</strong>
                  <span>sleep score</span>
                </div>
              </div>

              <ul className="score-stats">
                <li>
                  <span>TST</span>
                  <strong>{data.metrics.totalSleepMinutes} min</strong>
                </li>
                <li>
                  <span>Efficiency</span>
                  <strong>{data.metrics.sleepEfficiency}%</strong>
                </li>
                <li>
                  <span>Delta vs 7D</span>
                  <strong>{deltas.sleepScore}</strong>
                </li>
              </ul>
            </div>

            <p className="hypnogram-marker">Summary Hypnogram</p>
            <Hypnogram
          mode="summary"
          context="home"
          epochs={data.summaryHypnogram.epochs}
          showConfidenceOverlay
        />
          </article>

          <article className="wearable-card glass-card">
            <div className="wearable-floating">
              <span className="wearable-avatar" aria-hidden="true" />
              <p>{syncStatusLabel}</p>
            </div>
            <h2>Mounted device sync</h2>
            <p>Mountable sensor consistency this week.</p>
            <ul className="value-list">
              <li>
                <span>Fragmentation</span>
                <strong>{data.continuityMetrics.fragmentationIndex.toFixed(2)}</strong>
              </li>
              <li>
                <span>Entropy</span>
                <strong>{data.continuityMetrics.entropy.toFixed(2)}</strong>
              </li>
              <li>
                <span>WASO</span>
                <strong>{data.continuityMetrics.wasoMinutes} min</strong>
              </li>
            </ul>
            <button
              type="button"
              className="wearable-download"
              aria-label="Download mount report"
              onClick={handleDownloadMountReport}
            >
              <span aria-hidden="true">v</span>
              Download mount report
            </button>
            {downloadStatus ? <p className="status-message wearable-download-note">{downloadStatus}</p> : null}
          </article>

          <article className="dreams-card glass-card">
            <h2>EMDR Dreams</h2>
            <p className="dreams-copy">{data.aiSummary}</p>
          </article>
        </section>
      </section>

      <div className="metric-grid five-up">
        <MetricCard title="Sleep Score" value={String(data.metrics.sleepScore)} delta={`${deltas.sleepScore} vs 7D`} tone="positive" />
        <MetricCard title="TST" value={`${data.metrics.totalSleepMinutes} min`} delta={`${deltas.totalSleepMinutes} min`} tone="positive" />
        <MetricCard title="Sleep Efficiency" value={`${data.metrics.sleepEfficiency}%`} delta={`${deltas.sleepEfficiency.toFixed(1)}%`} />
        <MetricCard title="REM %" value={`${data.metrics.remPercent}%`} delta={`${deltas.remPercent.toFixed(1)}%`} tone="negative" />
        <MetricCard title="Deep %" value={`${data.metrics.deepPercent}%`} delta={`${deltas.deepPercent.toFixed(1)}%`} tone="positive" />
      </div>

      <div className="split-grid">
        <ChartContainer title="Stage Breakdown">
          {data.stageBreakdown.length > 0 ? (
            <ul className="value-list">
              {data.stageBreakdown.map((entry) => (
                <li key={entry.stage}>
                  <span>{formatStageLabel(entry.stage)}</span>
                  <strong>
                    {entry.minutes} min ({entry.percent}%)
                  </strong>
                </li>
              ))}
            </ul>
          ) : (
            <p className="chart-empty">No data for now.</p>
          )}
        </ChartContainer>

        <ChartContainer title="Latency Triplet">
          <ul className="value-list">
            <li>
              <span>Sleep onset</span>
              <strong>{data.latencyTriplet.sleepOnsetMinutes} min</strong>
            </li>
            <li>
              <span>REM latency</span>
              <strong>{data.latencyTriplet.remLatencyMinutes} min</strong>
            </li>
            <li>
              <span>Deep latency</span>
              <strong>{data.latencyTriplet.deepLatencyMinutes} min</strong>
            </li>
          </ul>
        </ChartContainer>
      </div>

    </section>
  );
}
