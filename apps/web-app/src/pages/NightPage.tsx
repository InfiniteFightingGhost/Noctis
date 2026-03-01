import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient } from "../api/apiClient";
import type { BackendEpochResponse, BackendPredictionResponse } from "../api/contracts";
import { ChartContainer } from "../components/common/ChartContainer";
import { MetricCard } from "../components/common/MetricCard";
import { PageState } from "../components/common/PageState";
import { Hypnogram } from "../components/visualizations/Hypnogram";
import { TransitionMatrix } from "../components/visualizations/TransitionMatrix";
import { useAsyncResource } from "../hooks/useAsyncResource";
import { usePolling } from "../hooks/usePolling";
import { useSyncEvents } from "../hooks/useSyncEvents";

const POLL_INTERVAL_MS = 5_000;
const LOOKBACK_HOURS = 12;
const LIVE_ROWS_LIMIT = 14;

type LiveTab = "epochs" | "predictions";

function buildIsoTimeRange(): { fromIso: string; toIso: string } {
  const to = new Date();
  const from = new Date(to.getTime() - LOOKBACK_HOURS * 60 * 60 * 1000);
  return {
    fromIso: from.toISOString(),
    toIso: to.toISOString(),
  };
}

function formatTimestamp(value: string): string {
  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) {
    return value;
  }

  return new Date(parsed).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatFeatureCount(payload: Record<string, unknown>): string {
  const count = Object.keys(payload).length;
  return `${count} feature${count === 1 ? "" : "s"}`;
}

function mapPredictionStage(stage: string): string {
  const normalized = stage.trim().toUpperCase();

  if (normalized === "W" || normalized === "WAKE" || normalized === "AWAKE" || normalized === "0") {
    return "wake";
  }
  if (normalized === "N3" || normalized === "DEEP" || normalized === "2") {
    return "deep";
  }
  if (normalized === "R" || normalized === "REM" || normalized === "3") {
    return "rem";
  }
  if (normalized === "N1" || normalized === "N2" || normalized === "LIGHT" || normalized === "1") {
    return "light";
  }

  return stage.toLowerCase();
}

export default function NightPage() {
  const syncEvents = useSyncEvents();
  const [activeTab, setActiveTab] = useState<LiveTab>("epochs");
  const [recordingId, setRecordingId] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);
  const [showNewDataBadge, setShowNewDataBadge] = useState(false);
  const lastPredictionRef = useRef<string | null>(null);
  const prevDataCountRef = useRef<number>(0);
  const loadNight = useCallback(() => apiClient.getNight(), [refreshToken]);
  const { loading, error, data } = useAsyncResource(loadNight);

  useEffect(() => {
    let active = true;

    const loadRecording = async () => {
      try {
        const summary = await apiClient.getLatestSleepSummary();
        if (!active) {
          return;
        }
        setRecordingId(summary.recordingId);
      } catch {
        if (!active) {
          return;
        }
        setRecordingId(null);
      }
    };

    void loadRecording();
    return () => {
      active = false;
    };
  }, [refreshToken]);

  const loadEpochs = useCallback(async (): Promise<BackendEpochResponse[]> => {
    if (!recordingId) {
      return [];
    }

    const { fromIso, toIso } = buildIsoTimeRange();
    return apiClient.getRecordingEpochs(recordingId, fromIso, toIso);
  }, [recordingId, refreshToken]);

  const loadPredictions = useCallback(async (): Promise<BackendPredictionResponse[]> => {
    if (!recordingId) {
      return [];
    }

    const { fromIso, toIso } = buildIsoTimeRange();
    return apiClient.getRecordingPredictions(recordingId, fromIso, toIso);
  }, [recordingId, refreshToken]);

  const epochsPolling = usePolling(loadEpochs, POLL_INTERVAL_MS, Boolean(recordingId) && activeTab === "epochs");
  const predictionsPolling = usePolling(loadPredictions, POLL_INTERVAL_MS, Boolean(recordingId) && activeTab === "predictions");

  useEffect(() => {
    const currentData = activeTab === "epochs" ? epochsPolling.data : predictionsPolling.data;
    const currentCount = currentData?.length ?? 0;

    if (currentCount > prevDataCountRef.current && prevDataCountRef.current > 0) {
      setShowNewDataBadge(true);
      const timer = window.setTimeout(() => setShowNewDataBadge(false), 3000);
      return () => window.clearTimeout(timer);
    }
    prevDataCountRef.current = currentCount;
  }, [epochsPolling.data, predictionsPolling.data, activeTab]);

  useEffect(() => {
    const lastPrediction = syncEvents.snapshot?.last_prediction_at ?? null;
    if (lastPrediction && lastPrediction !== lastPredictionRef.current) {
      lastPredictionRef.current = lastPrediction;
      setRefreshToken((value) => value + 1);
    }
  }, [syncEvents.snapshot?.last_prediction_at]);

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

  const epochRows = (epochsPolling.data ?? []).slice(-LIVE_ROWS_LIMIT).reverse();
  const predictionRows = (predictionsPolling.data ?? []).slice(-LIVE_ROWS_LIMIT).reverse();
  const liveLoading = activeTab === "epochs" ? epochsPolling.loading : predictionsPolling.loading;
  const liveError = activeTab === "epochs" ? epochsPolling.error : predictionsPolling.error;
  const liveCount = activeTab === "epochs" ? (epochsPolling.data?.length ?? 0) : (predictionsPolling.data?.length ?? 0);
  const lastUpdated = activeTab === "epochs" ? epochsPolling.lastUpdatedAt : predictionsPolling.lastUpdatedAt;

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

      <ChartContainer title="Live Recording Feed" subtitle="Polling backend every 5 seconds">
        <div className="night-live-header">
          <div className="filter-row" role="tablist" aria-label="Live data tab selector">
            <button
              type="button"
              role="tab"
              className={`pill ${activeTab === "epochs" ? "is-selected" : ""}`}
              aria-selected={activeTab === "epochs"}
              onClick={() => {
                setActiveTab("epochs");
                prevDataCountRef.current = epochsPolling.data?.length ?? 0;
              }}
            >
              Epochs
            </button>
            <button
              type="button"
              role="tab"
              className={`pill ${activeTab === "predictions" ? "is-selected" : ""}`}
              aria-selected={activeTab === "predictions"}
              onClick={() => {
                setActiveTab("predictions");
                prevDataCountRef.current = predictionsPolling.data?.length ?? 0;
              }}
            >
              Predictions
            </button>
            {showNewDataBadge && <span className="new-data-badge">New Data</span>}
          </div>

          <div className="night-live-meta">
            <span>
              {recordingId ? `Recording ${recordingId} · ${liveCount} row${liveCount === 1 ? "" : "s"}` : "Waiting for recording id"}
            </span>
            {lastUpdated && (
              <span className="last-updated">
                Last updated: {new Date(lastUpdated).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
              </span>
            )}
          </div>
        </div>

        {!recordingId ? (
          <div className="chart-empty">No active recording was found yet. Start a session and this feed will auto-refresh.</div>
        ) : liveError ? (
          <div className="chart-empty">{liveError}</div>
        ) : activeTab === "epochs" ? (
          epochRows.length === 0 && liveLoading ? (
            <div className="chart-empty">Loading epochs...</div>
          ) : epochRows.length === 0 ? (
            <div className="chart-empty">No epochs in the current polling window.</div>
          ) : (
            <div className="night-live-table-wrap">
              <table className="night-live-table" aria-label="Live epoch rows">
                <thead>
                  <tr>
                    <th scope="col">Epoch</th>
                    <th scope="col">Start</th>
                    <th scope="col">Schema</th>
                    <th scope="col">Payload</th>
                  </tr>
                </thead>
                <tbody>
                  {epochRows.map((row) => (
                    <tr key={`${row.recording_id}-${row.epoch_index}`}>
                      <td>{row.epoch_index}</td>
                      <td>{formatTimestamp(row.epoch_start_ts)}</td>
                      <td>{row.feature_schema_version}</td>
                      <td>{formatFeatureCount(row.features_payload)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        ) : predictionRows.length === 0 && liveLoading ? (
          <div className="chart-empty">Loading predictions...</div>
        ) : predictionRows.length === 0 ? (
          <div className="chart-empty">No predictions in the current polling window.</div>
        ) : (
          <div className="night-live-table-wrap">
            <table className="night-live-table" aria-label="Live prediction rows">
              <thead>
                <tr>
                  <th scope="col">Window End</th>
                  <th scope="col">Stage</th>
                  <th scope="col">Confidence</th>
                  <th scope="col">Model</th>
                </tr>
              </thead>
              <tbody>
                {predictionRows.map((row) => (
                  <tr key={row.id}>
                    <td>{formatTimestamp(row.window_end_ts)}</td>
                    <td>{mapPredictionStage(row.predicted_stage)}</td>
                    <td>{row.confidence.toFixed(2)}</td>
                    <td>{row.model_version}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </ChartContainer>
    </section>
  );
}
