import { Suspense, lazy, useCallback, useState } from "react";
import { apiClient } from "../api/apiClient";
import { ChartContainer } from "../components/common/ChartContainer";
import { MetricCard } from "../components/common/MetricCard";
import { PageState } from "../components/common/PageState";
import { useAsyncResource } from "../hooks/useAsyncResource";

const LineChart = lazy(() => import("../components/visualizations/LineChart").then((module) => ({ default: module.LineChart })));

const filters = ["7D", "30D", "90D", "Custom"] as const;

export default function TrendsPage() {
  const [selectedFilter, setSelectedFilter] = useState<(typeof filters)[number]>("30D");
  const loadTrends = useCallback(() => apiClient.getTrends(selectedFilter), [selectedFilter]);
  const { loading, error, data } = useAsyncResource(loadTrends);

  if (loading) {
    return <PageState mode="loading" />;
  }

  if (error || !data) {
    return <PageState mode="error" message={error ?? "Unable to load trends."} />;
  }

  const sleepScoreSeries = data.nights.map((night) => ({ x: night.date, y: night.sleepScore }));
  const tstSeries = data.nights.map((night) => ({ x: night.date, y: night.totalSleepMinutes }));
  const efficiencySeries = data.nights.map((night) => ({ x: night.date, y: night.sleepEfficiency }));
  const remSeries = data.nights.map((night) => ({ x: night.date, y: night.remPercent }));
  const deepSeries = data.nights.map((night) => ({ x: night.date, y: night.deepPercent }));
  const fragmentationSeries = data.nights.map((night) => ({ x: night.date, y: night.fragmentationIndex }));
  const hrMeanSeries = data.nights.map((night) => ({ x: night.date, y: night.hrMean }));
  const hrvSeries = data.nights.map((night) => ({ x: night.date, y: night.hrv }));
  const latestConsistency = data.nights[data.nights.length - 1]?.consistencyIndex;
  const hasTrendsData = data.nights.length > 0;

  return (
    <section className="page-grid trends-page">
      <section className="chart-card trends-intro" aria-label="Trends dashboard introduction">
        <div className="trends-intro-copy">
          <p className="eyebrow">Performance Analytics</p>
          <h1>Longitudinal Sleep Trends</h1>
          <p>Track score stability, stage composition, and recovery signals across your selected time window.</p>
        </div>

        <div className="filter-row" role="group" aria-label="Time filters">
          {filters.map((filter) => (
            <button
              key={filter}
              className={filter === selectedFilter ? "pill is-selected" : "pill"}
              type="button"
              onClick={() => setSelectedFilter(filter)}
              aria-pressed={filter === selectedFilter}
            >
              {filter}
            </button>
          ))}
        </div>

        {hasTrendsData ? (
          <div className="metric-grid three-up trends-kpis">
            <MetricCard title="Consistency Index" value={latestConsistency?.toFixed(1) ?? "-"} />
            <MetricCard
              title="Worst-Night Decile"
              value={`${data.worstNightDecile.sleepScore}`}
              delta={data.worstNightDecile.date}
              tone="negative"
            />
            <MetricCard title="Nights In Window" value={`${data.nights.length}`} delta={selectedFilter} />
          </div>
        ) : (
          <p className="chart-empty">No data for now.</p>
        )}
      </section>

      {hasTrendsData ? (
        <>
      <ChartContainer
        title="Sleep Score Longitudinal"
        subtitle={`Moving average ${data.movingAverageWindow} nights with variance band`}
      >
        <Suspense fallback={<div className="page-state">Loading chart...</div>}>
          <LineChart points={sleepScoreSeries} showBand={data.varianceBand} yDomain={{ min: 60, max: 100 }} />
        </Suspense>
      </ChartContainer>

      <div className="split-grid analytics-grid">
        <ChartContainer title="TST Trend">
          <Suspense fallback={<div className="page-state">Loading chart...</div>}>
            <LineChart points={tstSeries} yDomain={{ min: 300, max: 500 }} />
          </Suspense>
        </ChartContainer>
        <ChartContainer title="Efficiency Trend">
          <Suspense fallback={<div className="page-state">Loading chart...</div>}>
            <LineChart points={efficiencySeries} yDomain={{ min: 75, max: 100 }} />
          </Suspense>
        </ChartContainer>
      </div>

      <div className="split-grid">
        <ChartContainer title="REM Percent Longitudinal">
          <Suspense fallback={<div className="page-state">Loading chart...</div>}>
            <LineChart points={remSeries} yDomain={{ min: 10, max: 35 }} />
          </Suspense>
        </ChartContainer>
        <ChartContainer title="Deep Percent Longitudinal">
          <Suspense fallback={<div className="page-state">Loading chart...</div>}>
            <LineChart points={deepSeries} yDomain={{ min: 10, max: 30 }} />
          </Suspense>
        </ChartContainer>
      </div>

      <div className="split-grid">
        <ChartContainer title="Fragmentation Longitudinal">
          <Suspense fallback={<div className="page-state">Loading chart...</div>}>
            <LineChart points={fragmentationSeries} yDomain={{ min: 0, max: 0.5 }} />
          </Suspense>
        </ChartContainer>
        <ChartContainer title="HR Mean Longitudinal">
          <Suspense fallback={<div className="page-state">Loading chart...</div>}>
            <LineChart points={hrMeanSeries} yDomain={{ min: 45, max: 85 }} />
          </Suspense>
        </ChartContainer>
        <ChartContainer title="HRV Longitudinal">
          <Suspense fallback={<div className="page-state">Loading chart...</div>}>
            <LineChart points={hrvSeries} yDomain={{ min: 20, max: 70 }} />
          </Suspense>
        </ChartContainer>
      </div>
        </>
      ) : null}
    </section>
  );
}
