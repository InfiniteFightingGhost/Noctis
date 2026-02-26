import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";
import {
  createApiClient,
  sessionTokenFromPayload,
  toObjectArray,
  type ApiError,
  type ApiClient,
  type JsonObject,
  type JsonValue,
} from "./platform/apiClient";
import { getAppEnv } from "./platform/env";
import { useMultiQuery, useMutation, useQuery, useRoute, type QueryState } from "./platform/hooks";
import { deriveAccess, persistToken, purgeAuthState, readStoredToken, type SessionAccess } from "./platform/session";
import {
  clearOnboardingState,
  currentOnboardingStep,
  isOnboardingIncomplete,
  nextOnboardingRoute,
  reconcileOnboardingState,
  readOnboardingState,
  resolveOnboardingRedirect,
  persistOnboardingState,
  type OnboardingStepId,
  type OnboardingState,
} from "./platform/onboarding";
import { EndpointGrid, EntityLauncherPanel, JsonFormPanel, QueryCardsPage, QueryVisualizationCard } from "./components/sleepOsPrimitives";
import { FailureNotice, ResumeSetupCard, SetupBanner, SetupProgress, type FailureKind, type SetupProgressState } from "./components/onboarding/SetupUI";

type RouteScope = "public" | "user" | "admin" | "internal";

type RouteDef = {
  path: string;
  title: string;
  scope: RouteScope;
};

type Match = {
  route: RouteDef;
  params: Record<string, string>;
};

type SessionStatus = "checking" | "anonymous" | "authenticated";

type SessionState = {
  status: SessionStatus;
  token?: string;
  authMe?: JsonObject;
  accountMe?: JsonObject;
  tenantMe?: JsonObject;
  access: SessionAccess;
  error?: string;
};

const ROUTES: RouteDef[] = [
  { path: "/login", title: "Login", scope: "public" },
  { path: "/register", title: "Register", scope: "public" },
  { path: "/onboarding/connect-device", title: "Connect Device", scope: "user" },
  { path: "/onboarding/start-tracking", title: "Start Tracking", scope: "user" },
  { path: "/app/home", title: "Home", scope: "user" },
  { path: "/app/devices/connect", title: "Connect Device", scope: "user" },
  { path: "/app/tracking/start", title: "Start Tracking", scope: "user" },
  { path: "/app/sleep/latest", title: "Latest Sleep", scope: "user" },
  { path: "/app/sleep/sync", title: "Sleep Sync", scope: "user" },
  { path: "/app/sleep/insights", title: "Sleep Insights", scope: "user" },
  { path: "/app/devices", title: "Devices", scope: "user" },
  { path: "/app/devices/claim", title: "Claim Device", scope: "user" },
  { path: "/app/devices/pair", title: "Pair Device", scope: "user" },
  { path: "/app/recordings", title: "Recordings", scope: "user" },
  { path: "/app/recordings/:recordingId", title: "Recording Detail", scope: "user" },
  { path: "/app/alarm", title: "Alarm", scope: "user" },
  { path: "/app/routines", title: "Routines", scope: "user" },
  { path: "/app/coach", title: "Coach", scope: "user" },
  { path: "/app/challenges", title: "Challenges", scope: "user" },
  { path: "/app/search", title: "Search", scope: "user" },
  { path: "/admin/models", title: "Guidance Library", scope: "admin" },
  { path: "/admin/models/:version", title: "Guidance Version", scope: "admin" },
  { path: "/admin/experiments", title: "Pilot Programs", scope: "admin" },
  { path: "/admin/feature-schemas", title: "Signal Blueprints", scope: "admin" },
  { path: "/admin/feature-schemas/:version", title: "Blueprint Version", scope: "admin" },
  { path: "/admin/evaluation", title: "Quality Review", scope: "admin" },
  { path: "/admin/drift", title: "Trend Stability", scope: "admin" },
  { path: "/admin/replay", title: "Night Playback", scope: "admin" },
  { path: "/internal/monitoring", title: "Experience Health", scope: "internal" },
  { path: "/internal/performance", title: "Speed Check", scope: "internal" },
  { path: "/internal/stress", title: "Capacity Check", scope: "internal" },
  { path: "/internal/faults", title: "Recovery Drills", scope: "internal" },
  { path: "/internal/timescale", title: "History Policies", scope: "internal" },
  { path: "/internal/audit", title: "Activity History", scope: "internal" },
];

const USER_NAV = [
  "/app/home",
  "/app/devices/connect",
  "/app/tracking/start",
  "/app/sleep/latest",
  "/app/sleep/sync",
  "/app/sleep/insights",
  "/app/devices",
  "/app/devices/claim",
  "/app/devices/pair",
  "/app/recordings",
  "/app/alarm",
  "/app/routines",
  "/app/coach",
  "/app/challenges",
  "/app/search",
];

const ADMIN_NAV = [
  "/admin/models",
  "/admin/experiments",
  "/admin/feature-schemas",
  "/admin/evaluation",
  "/admin/drift",
  "/admin/replay",
];

const INTERNAL_NAV = [
  "/internal/monitoring",
  "/internal/performance",
  "/internal/stress",
  "/internal/faults",
  "/internal/timescale",
  "/internal/audit",
];

function normalizePath(path: string): string {
  if (path.length <= 1) {
    return path;
  }
  return path.endsWith("/") ? path.slice(0, -1) : path;
}

function matchRoute(pathname: string): Match | undefined {
  const clean = normalizePath(pathname);
  const pathParts = clean.split("/").filter(Boolean);

  for (const route of ROUTES) {
    const routeParts = normalizePath(route.path).split("/").filter(Boolean);
    if (routeParts.length !== pathParts.length) {
      continue;
    }

    const params: Record<string, string> = {};
    let matched = true;

    for (let i = 0; i < routeParts.length; i += 1) {
      const routePart = routeParts[i];
      const pathPart = pathParts[i];

      if (routePart.startsWith(":")) {
        params[routePart.slice(1)] = decodeURIComponent(pathPart);
        continue;
      }

      if (routePart !== pathPart) {
        matched = false;
        break;
      }
    }

    if (matched) {
      return { route, params };
    }
  }

  return undefined;
}

function routeTitle(path: string): string {
  const exact = ROUTES.find((route) => route.path === path);
  return exact ? exact.title : path;
}

function bindSingleQuery(map: Record<string, QueryState<unknown>>, key: string): QueryState<unknown> {
  return map[key] ?? { status: "idle" };
}

function classifyError(error: ApiError, fallback: FailureKind): FailureKind {
  const detailText = JSON.stringify(error.details ?? "").toLowerCase();
  const messageText = error.message.toLowerCase();
  const combined = `${messageText} ${detailText}`;

  if (error.kind === "network") {
    return "network_issue";
  }
  if (error.kind === "http" && error.status === 403) {
    return "permission_denied";
  }

  const looksLikeCredentialIssue =
    error.kind === "http" &&
    (error.status === 400 ||
      error.status === 401 ||
      error.status === 409 ||
      error.status === 422 ||
      combined.includes("invalid") ||
      combined.includes("credential") ||
      combined.includes("password") ||
      combined.includes("email"));

  if (looksLikeCredentialIssue) {
    return "invalid_credentials";
  }

  const looksLikePairingTimeout =
    (error.kind === "http" && (error.status === 408 || error.status === 410 || error.status === 504)) ||
    combined.includes("timeout") ||
    combined.includes("timed out") ||
    combined.includes("expired");

  if (fallback === "device_not_found" && looksLikePairingTimeout) {
    return "pairing_timeout";
  }

  const looksLikeDeviceNotFound =
    (error.kind === "http" && error.status === 404) ||
    combined.includes("device not found") ||
    combined.includes("unknown device");

  if (fallback === "device_not_found" && looksLikeDeviceNotFound) {
    return "device_not_found";
  }

  if (fallback === "pairing_timeout" && looksLikePairingTimeout) {
    return "pairing_timeout";
  }

  return fallback;
}

const SETUP_STEP_ORDER: OnboardingStepId[] = ["signup", "login", "connect-device", "start-tracking"];

function hasCompletedStep(lastCompleted: OnboardingStepId | null, step: OnboardingStepId): boolean {
  if (!lastCompleted) {
    return false;
  }
  return SETUP_STEP_ORDER.indexOf(lastCompleted) >= SETUP_STEP_ORDER.indexOf(step);
}

function buildSetupProgressState(onboarding: OnboardingState, isAuthenticated: boolean): SetupProgressState {
  const signupComplete = hasCompletedStep(onboarding.lastOnboardingStepCompleted, "signup");
  const loginComplete = isAuthenticated || hasCompletedStep(onboarding.lastOnboardingStepCompleted, "login");

  return {
    signupComplete,
    loginComplete,
    hasConnectedDevice: onboarding.hasConnectedDevice,
    trackingActive: onboarding.trackingActive,
  };
}

function readSignalBoolean(source: JsonObject | undefined, keys: string[]): boolean | undefined {
  if (!source) {
    return undefined;
  }

  for (const key of keys) {
    if (source[key] === true) {
      return true;
    }
    if (source[key] === false) {
      return false;
    }
  }

  return undefined;
}

function deriveConnectedDeviceSignal(payload: JsonValue | undefined): boolean | undefined {
  if (!payload || typeof payload !== "object") {
    return undefined;
  }

  if (Array.isArray(payload)) {
    for (const candidate of payload) {
      if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
        continue;
      }

      const value = readSignalBoolean(candidate as JsonObject, [
        "claimed",
        "is_claimed",
        "isClaimed",
        "connected",
        "is_connected",
        "isConnected",
        "linked",
      ]);

      if (value !== undefined) {
        return value;
      }
    }
    return undefined;
  }

  return readSignalBoolean(payload as JsonObject, [
    "has_connected_device",
    "hasConnectedDevice",
    "device_connected",
    "deviceConnected",
  ]);
}

function deriveTrackingSignal(payload: JsonObject | undefined): boolean | undefined {
  const explicit = readSignalBoolean(payload, [
    "tracking_active",
    "trackingActive",
    "is_tracking",
    "isTracking",
    "tracking_enabled",
    "trackingEnabled",
  ]);
  if (explicit !== undefined) {
    return explicit;
  }

  if (!payload) {
    return undefined;
  }

  const status = typeof payload.status === "string" ? payload.status.trim().toLowerCase() : "";
  if (status === "active" || status === "started" || status === "enabled" || status === "tracking") {
    return true;
  }

  return undefined;
}

function hasProvisioningClaimData(source: JsonObject | undefined): boolean {
  if (!source) {
    return false;
  }

  const claimFields = [source.scopes, source.roles, source.claims, source.permissions];
  for (const field of claimFields) {
    if (Array.isArray(field) && field.some((entry) => typeof entry === "string" && entry.trim().length > 0)) {
      return true;
    }
    if (typeof field === "string" && field.trim().length > 0) {
      return true;
    }
  }

  if (typeof source.role === "string" && source.role.trim().length > 0) {
    return true;
  }

  return source.is_admin === true || source.is_internal === true;
}

function UserHomePage({ api }: { api: ApiClient }) {
  const queries = useMultiQuery(
    {
      overview: (signal) => api.homeOverview(signal),
      report: (signal) => api.reportLatest(signal),
    },
    [api],
  );

  return (
    <QueryCardsPage
      cards={[
        { label: "Home Overview", path: "/v1/home/overview", query: bindSingleQuery(queries, "overview") },
        { label: "Nightly Story", path: "/v1/report/latest", query: bindSingleQuery(queries, "report") },
      ]}
    />
  );
}

function SleepLatestPage({ api }: { api: ApiClient }) {
  const query = useQuery((signal) => api.sleepLatestSummary(signal), [api]);
  return <QueryCardsPage cards={[{ label: "Latest Sleep Summary", path: "/v1/sleep/latest/summary", query }]} />;
}

function SleepSyncPage({ api }: { api: ApiClient }) {
  const status = useQuery((signal) => api.syncStatus(signal), [api]);
  const feedbackMutation = useMutation((payload: JsonObject, signal) => api.submitInsightsFeedback(payload, signal));

  return (
    <EndpointGrid>
      <QueryVisualizationCard label="Night Sync Progress" path="/v1/sync/status" query={status} />
      <JsonFormPanel
        title="Share Insight Feedback"
        path="/v1/insights/feedback"
        submitLabel="Send Feedback"
        placeholder={"recording_id: rec-001\nrating: helpful\ncomment: Insight was accurate."}
        mutation={feedbackMutation}
      />
    </EndpointGrid>
  );
}

function SleepInsightsPage({ api }: { api: ApiClient }) {
  const query = useQuery((signal) => api.reportLatest(signal), [api]);
  const feedbackMutation = useMutation((payload: JsonObject, signal) => api.submitInsightsFeedback(payload, signal));

  return (
    <EndpointGrid>
      <QueryVisualizationCard label="Tonight's Insight Story" path="/v1/report/latest" query={query} />
      <JsonFormPanel
        title="How Helpful Was This Insight"
        path="/v1/insights/feedback"
        submitLabel="Send Feedback"
        placeholder={"insight_id: nightly-1\nsentiment: positive"}
        mutation={feedbackMutation}
      />
    </EndpointGrid>
  );
}

function DevicesPage({ api }: { api: ApiClient }) {
  const list = useQuery((signal) => api.devices(signal), [api]);
  const firstId = useMemo(() => {
    const entries = toObjectArray((list.data as JsonValue) ?? []);
    const first = entries[0];
    return typeof first?.id === "string" ? first.id : undefined;
  }, [list.data]);
  const detail = useQuery((signal) => api.deviceById(firstId ?? "", signal), [api, firstId], Boolean(firstId));

  return (
    <EndpointGrid>
      <QueryVisualizationCard label="Your Devices" path="/v1/devices" query={list} />
      <QueryVisualizationCard
        label="Main Device Snapshot"
        path={`/v1/devices/${firstId ?? "{id}"}`}
        query={detail}
        emptyMessage="No devices connected yet."
      />
    </EndpointGrid>
  );
}

function DevicesClaimPage({ api }: { api: ApiClient }) {
  const mutation = useMutation((payload: JsonObject, signal) => api.claimDeviceById(payload, signal));

  return (
    <EndpointGrid>
      <JsonFormPanel
        title="Add a Device by ID"
        path="/v1/devices/claim/by-id"
        submitLabel="Add Device"
        placeholder={"external_id: R01"}
        mutation={mutation}
      />
    </EndpointGrid>
  );
}

function DevicesPairPage({ api }: { api: ApiClient }) {
  const start = useMutation((payload: JsonObject, signal) => api.startPairing(payload, signal));
  const claim = useMutation((payload: JsonObject, signal) => api.claimPairing(payload, signal));

  return (
    <EndpointGrid>
      <JsonFormPanel
        title="Start Device Pairing"
        path="/v1/devices/pairing/start"
        submitLabel="Start Pairing"
        placeholder={"device_type: ring"}
        mutation={start}
      />
      <JsonFormPanel
        title="Finish Device Pairing"
        path="/v1/devices/pairing/claim"
        submitLabel="Complete Pairing"
        placeholder={"pairing_id: pair-123\ncode: 934122"}
        mutation={claim}
      />
    </EndpointGrid>
  );
}

function RecordingsPage({ api, navigate }: { api: ApiClient; navigate: (path: string) => void }) {
  const recordings = useQuery((signal) => api.recordings(signal), [api]);

  return (
    <EndpointGrid>
      <QueryVisualizationCard label="Sleep Sessions" path="/v1/recordings" query={recordings} />
      <EntityLauncherPanel
        title="Open Session"
        routePattern="/app/recordings/:recordingId"
        query={recordings}
        valueField="id"
        onOpen={(id) => navigate(`/app/recordings/${encodeURIComponent(id)}`)}
      />
    </EndpointGrid>
  );
}

function RecordingDetailPage({ api, recordingId }: { api: ApiClient; recordingId: string }) {
  const queries = useMultiQuery(
    {
      detail: (signal) => api.recordingById(recordingId, signal),
      epochs: (signal) => api.recordingEpochs(recordingId, signal),
      predictions: (signal) => api.recordingPredictions(recordingId, signal),
      summary: (signal) => api.recordingSummary(recordingId, signal),
      evaluation: (signal) => api.recordingEvaluation(recordingId, signal),
    },
    [api, recordingId],
  );

  return (
    <EndpointGrid>
      <QueryVisualizationCard label="Recording" path={`/v1/recordings/${recordingId}`} query={bindSingleQuery(queries, "detail")} />
      <QueryVisualizationCard label="Epochs" path={`/v1/recordings/${recordingId}/epochs`} query={bindSingleQuery(queries, "epochs")} />
      <QueryVisualizationCard
        label="Predictions"
        path={`/v1/recordings/${recordingId}/predictions`}
        query={bindSingleQuery(queries, "predictions")}
      />
      <QueryVisualizationCard label="Summary" path={`/v1/recordings/${recordingId}/summary`} query={bindSingleQuery(queries, "summary")} />
      <QueryVisualizationCard
        label="Evaluation"
        path={`/v1/recordings/${recordingId}/evaluation`}
        query={bindSingleQuery(queries, "evaluation")}
      />
    </EndpointGrid>
  );
}

function AlarmPage({ api }: { api: ApiClient }) {
  const alarm = useQuery((signal) => api.alarm(signal), [api]);
  const update = useMutation((payload: JsonObject, signal) => api.updateAlarm(payload, signal));

  return (
    <EndpointGrid>
      <QueryVisualizationCard label="Wake Plan" path="/v1/alarm" query={alarm} />
      <JsonFormPanel
        title="Adjust Wake Plan"
        path="/v1/alarm"
        submitLabel="Save Wake Plan"
        placeholder={"wake_time: 06:30\nwake_window_minutes: 15\nsunrise_enabled: true"}
        mutation={update}
      />
    </EndpointGrid>
  );
}

function RoutinesPage({ api }: { api: ApiClient }) {
  const queries = useMultiQuery(
    {
      routines: (signal) => api.routines(signal),
      current: (signal) => api.routineCurrent(signal),
    },
    [api],
  );
  const update = useMutation((payload: JsonObject, signal) => api.updateRoutines(payload, signal));

  return (
    <EndpointGrid>
      <QueryVisualizationCard label="Evening Routines" path="/v1/routines" query={bindSingleQuery(queries, "routines")} />
      <QueryVisualizationCard label="Tonight's Routine" path="/v1/routines/current" query={bindSingleQuery(queries, "current")} />
      <JsonFormPanel
        title="Adjust Routine"
        path="/v1/routines"
        submitLabel="Save Routine"
        placeholder={"title: Deep recovery\nfirst_step: Breathing\nstep_duration_minutes: 8"}
        mutation={update}
      />
    </EndpointGrid>
  );
}

function CoachPage({ api }: { api: ApiClient }) {
  const queries = useMultiQuery(
    {
      coach: (signal) => api.coach(signal),
      summary: (signal) => api.coachSummary(signal),
    },
    [api],
  );

  return (
    <EndpointGrid>
      <QueryVisualizationCard
        label="Coach"
        path="/v1/coach"
        query={bindSingleQuery(queries, "coach")}
        emptyMessage="No personalized plan is available yet."
      />
      <QueryVisualizationCard
        label="Coach Summary"
        path="/v1/coach/summary"
        query={bindSingleQuery(queries, "summary")}
        emptyMessage="Your coach summary will appear here once tonight's guidance is ready."
      />
    </EndpointGrid>
  );
}

function ChallengesPage({ api }: { api: ApiClient }) {
  const query = useQuery((signal) => api.challenges(signal), [api]);
  return (
    <QueryCardsPage
      cards={[
        {
          label: "Challenges",
          path: "/v1/challenges",
          query,
          emptyMessage: "No active challenges right now. You are caught up.",
        },
      ]}
    />
  );
}

function SearchPage({ api }: { api: ApiClient }) {
  const [text, setText] = useState("deep sleep");
  const [submitted, setSubmitted] = useState("deep sleep");
  const query = useQuery((signal) => api.search(submitted, signal), [api, submitted], submitted.trim().length > 0);

  return (
    <EndpointGrid>
      <QueryVisualizationCard
        label="Search Results"
        path={`/v1/search?q=${encodeURIComponent(submitted)}`}
        query={query}
        emptyMessage="No results yet. Try a broader phrase."
      />
      <article className="glass-card panel-card">
        <header className="card-head">
          <div>
            <h3>Sleep Topic Search</h3>
            <p className="micro-copy">Look up patterns and guidance from your recent nights.</p>
          </div>
        </header>
        <form
          className="inline-form"
          onSubmit={(event) => {
            event.preventDefault();
            setSubmitted(text.trim());
          }}
        >
          <input value={text} onChange={(event) => setText(event.target.value)} placeholder="Search term" />
          <button className="solid-btn" type="submit">
            Search
          </button>
        </form>
        <p className="state">Current topic: {submitted || "none yet"}</p>
      </article>
    </EndpointGrid>
  );
}

function ModelsPage({ api, navigate }: { api: ApiClient; navigate: (path: string) => void }) {
  const models = useQuery((signal) => api.models(signal), [api]);
  const reload = useMutation((payload: JsonObject, signal) => api.modelReload(payload, signal));

  return (
    <EndpointGrid>
      <QueryVisualizationCard label="Guidance Library" path="/v1/models" query={models} />
      <EntityLauncherPanel
        title="Open Guidance Version"
        routePattern="/admin/models/:version"
        query={models}
        valueField="version"
        onOpen={(version) => navigate(`/admin/models/${encodeURIComponent(version)}`)}
      />
      <JsonFormPanel
        title="Refresh Guidance Library"
        path="/v1/models/reload"
        submitLabel="Refresh"
        placeholder={"reason: manual refresh"}
        mutation={reload}
        confirmMessage="Refresh all guidance profiles now? This may briefly pause live recommendations."
      />
    </EndpointGrid>
  );
}

function ModelVersionPage({ api, version }: { api: ApiClient; version: string }) {
  const model = useQuery((signal) => api.modelByVersion(version, signal), [api, version]);
  const promote = useMutation((payload: JsonObject, signal) => api.modelPromote(version, payload, signal));
  const archive = useMutation((payload: JsonObject, signal) => api.modelArchive(version, payload, signal));
  const rollback = useMutation((payload: JsonObject, signal) => api.modelRollback(version, payload, signal));

  return (
    <EndpointGrid>
      <QueryVisualizationCard label="Guidance Version" path={`/v1/models/${version}`} query={model} />
      <JsonFormPanel
        title="Set as Active Version"
        path={`/v1/models/${version}/promote`}
        submitLabel="Set Active"
        placeholder={"target: daily_guidance"}
        mutation={promote}
        confirmMessage={`Set guidance version ${version} as active now?`}
      />
      <JsonFormPanel
        title="Archive Version"
        path={`/v1/models/${version}/archive`}
        submitLabel="Archive"
        placeholder={"reason: superseded"}
        mutation={archive}
        confirmMessage={`Archive guidance version ${version}? It will no longer appear in active choices.`}
      />
      <JsonFormPanel
        title="Restore Previous Version"
        path={`/v1/models/${version}/rollback`}
        submitLabel="Restore"
        placeholder={"to_version: v1.0.0"}
        mutation={rollback}
        confirmMessage={`Restore from guidance version ${version}? Double-check the version you want to return to.`}
      />
    </EndpointGrid>
  );
}

function ExperimentsPage({ api }: { api: ApiClient }) {
  const query = useQuery((signal) => api.experiments(signal), [api]);
  return <QueryCardsPage cards={[{ label: "Pilot Programs", path: "/v1/experiments", query }]} />;
}

function FeatureSchemasPage({ api, navigate }: { api: ApiClient; navigate: (path: string) => void }) {
  const query = useQuery((signal) => api.featureSchemas(signal), [api]);
  return (
    <EndpointGrid>
      <QueryVisualizationCard label="Signal Blueprints" path="/v1/feature-schemas" query={query} />
      <EntityLauncherPanel
        title="Open Blueprint Version"
        routePattern="/admin/feature-schemas/:version"
        query={query}
        valueField="version"
        onOpen={(version) => navigate(`/admin/feature-schemas/${encodeURIComponent(version)}`)}
      />
    </EndpointGrid>
  );
}

function FeatureSchemaVersionPage({ api, version }: { api: ApiClient; version: string }) {
  const query = useQuery((signal) => api.featureSchemaByVersion(version, signal), [api, version]);
  return (
    <EndpointGrid>
      <QueryVisualizationCard label="Blueprint Version" path={`/v1/feature-schemas/${version}`} query={query} />
    </EndpointGrid>
  );
}

function EvaluationPage({ api }: { api: ApiClient }) {
  const query = useQuery((signal) => api.modelEvaluationGlobal(signal), [api]);
  return <QueryCardsPage cards={[{ label: "Overall Quality Review", path: "/v1/model/evaluation/global", query }]} />;
}

function DriftPage({ api }: { api: ApiClient }) {
  const query = useQuery((signal) => api.modelDrift(signal), [api]);
  return <QueryCardsPage cards={[{ label: "Guidance Stability", path: "/v1/model/drift", query }]} />;
}

function ReplayPage({ api }: { api: ApiClient }) {
  const mutation = useMutation((payload: JsonObject, signal) => api.modelReplay(payload, signal));
  return (
    <EndpointGrid>
      <JsonFormPanel
        title="Replay Night Guidance"
        path="/v1/models/replay"
        submitLabel="Run Playback"
        placeholder={"recording_id: rec-001\nmodel_version: v1.1.0"}
        mutation={mutation}
      />
    </EndpointGrid>
  );
}

function MonitoringPage({ api }: { api: ApiClient }) {
  const query = useQuery((signal) => api.internalMonitoringSummary(signal), [api]);
  return <QueryCardsPage cards={[{ label: "Experience Health Snapshot", path: "/internal/monitoring/summary", query }]} />;
}

function PerformancePage({ api }: { api: ApiClient }) {
  const queries = useMultiQuery(
    {
      performance: (signal) => api.internalPerformance(signal),
      stats: (signal) => api.internalPerformanceStats(signal),
    },
    [api],
  );

  return (
    <EndpointGrid>
      <QueryVisualizationCard label="Response Speed" path="/internal/performance" query={bindSingleQuery(queries, "performance")} />
      <QueryVisualizationCard label="Speed Details" path="/internal/performance/stats" query={bindSingleQuery(queries, "stats")} />
    </EndpointGrid>
  );
}

function StressPage({ api }: { api: ApiClient }) {
  const mutation = useMutation((payload: JsonObject, signal) => api.internalStressRun(payload, signal));
  return (
    <EndpointGrid>
      <JsonFormPanel
        title="Run Capacity Check"
        path="/internal/stress/run"
        submitLabel="Start Check"
        placeholder={"duration_sec: 30\nconcurrency: 20"}
        mutation={mutation}
        confirmMessage="Run a high-load capacity check now? This may briefly slow the experience."
      />
    </EndpointGrid>
  );
}

function FaultsPage({ api }: { api: ApiClient }) {
  const faults = useQuery((signal) => api.internalFaults(signal), [api]);
  const enable = useMutation((payload: JsonObject, signal) => api.internalFaultsEnable(payload, signal));
  const disable = useMutation((payload: JsonObject, signal) => api.internalFaultsDisable(payload, signal));

  return (
    <EndpointGrid>
      <QueryVisualizationCard label="Recovery Scenario Library" path="/internal/faults" query={faults} />
      <JsonFormPanel
        title="Start Scenario"
        path="/internal/faults/enable"
        submitLabel="Start"
        placeholder={"fault: delay_predictions"}
        mutation={enable}
        confirmMessage="Start this recovery scenario? It is designed to test resilience under pressure."
      />
      <JsonFormPanel
        title="Stop Scenario"
        path="/internal/faults/disable"
        submitLabel="Stop"
        placeholder={"fault: delay_predictions"}
        mutation={disable}
        confirmMessage="Stop this recovery scenario now?"
      />
    </EndpointGrid>
  );
}

function TimescalePage({ api }: { api: ApiClient }) {
  const policies = useQuery((signal) => api.internalTimescalePolicies(signal), [api]);
  const dryRun = useMutation((payload: JsonObject, signal) => api.internalTimescaleDryRun(payload, signal));
  const apply = useMutation((payload: JsonObject, signal) => api.internalTimescaleApply(payload, signal));

  return (
    <EndpointGrid>
      <QueryVisualizationCard label="History Policy Settings" path="/internal/timescale/policies" query={policies} />
      <JsonFormPanel
        title="Preview Policy Change"
        path="/internal/timescale/dry-run"
        submitLabel="Preview"
        placeholder={"policy: compress_chunks"}
        mutation={dryRun}
      />
      <JsonFormPanel
        title="Apply History Policy"
        path="/internal/timescale/apply"
        submitLabel="Apply"
        placeholder={"policy: compress_chunks"}
        mutation={apply}
        confirmMessage="Apply these history policy updates now? This takes effect immediately."
      />
    </EndpointGrid>
  );
}

function AuditPage({ api }: { api: ApiClient }) {
  const query = useQuery((signal) => api.internalAuditReport(signal), [api]);
  return <QueryCardsPage cards={[{ label: "Activity History", path: "/internal/audit/report", query }]} />;
}

function LoginPage({
  api,
  onLoggedIn,
  onboarding,
  navigate,
}: {
  api: ApiClient;
  onLoggedIn: (token: string, email: string) => Promise<string>;
  onboarding: OnboardingState;
  navigate: (path: string, replace?: boolean) => void;
}) {
  const [email, setEmail] = useState(onboarding.pendingLoginEmail ?? "demo@noctis.local");
  const [password, setPassword] = useState("demo-password");
  const [failure, setFailure] = useState<FailureKind | undefined>(undefined);
  const mutation = useMutation((payload: JsonObject, signal) => api.login(payload, signal));

  const submit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      setFailure(undefined);
      try {
        const response = await mutation.run({ email, password });
        const token = sessionTokenFromPayload(response);
        if (token) {
          const nextRoute = await onLoggedIn(token, email);
          navigate(nextRoute, true);
        }
      } catch (error) {
        const kind = classifyError(error as ApiError, "invalid_credentials");
        setFailure(kind);
        return;
      }
    },
    [email, mutation, navigate, onLoggedIn, password],
  );

  const progressState = buildSetupProgressState(onboarding, false);

  return (
    <section className="auth-shell">
      <article className="glass-card auth-card">
        <p className="eyebrow">Sleep OS</p>
        <SetupProgress onboarding={progressState} currentStep="login" />
        <h2>Sign in</h2>
        <p className="state">Welcome back. Pick up where you left off.</p>
        <form onSubmit={submit} className="form-stack">
          <input value={email} onChange={(event) => setEmail(event.target.value)} placeholder="Email" />
          <input value={password} onChange={(event) => setPassword(event.target.value)} type="password" placeholder="Password" />
          <button className="solid-btn" type="submit" disabled={mutation.state.status === "loading"}>
            {mutation.state.status === "loading" ? "Logging in..." : "Login"}
          </button>
        </form>
        {failure ? <FailureNotice kind={failure} /> : null}
        <div className="auth-alt-action">
          <p className="state">First time here? Create your account to begin setup.</p>
          <button className="ghost-btn auth-secondary-cta" type="button" onClick={() => navigate("/register")}>
            Create an account
          </button>
        </div>
      </article>
    </section>
  );
}

function RegisterPage({
  api,
  onboarding,
  onRegistrationComplete,
  navigate,
}: {
  api: ApiClient;
  onboarding: OnboardingState;
  onRegistrationComplete: (email: string) => void;
  navigate: (path: string) => void;
}) {
  const [email, setEmail] = useState("new-user@noctis.local");
  const [password, setPassword] = useState("new-password");
  const [failure, setFailure] = useState<FailureKind | undefined>(undefined);
  const mutation = useMutation((payload: JsonObject, signal) => api.register(payload, signal));

  const submit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      setFailure(undefined);
      try {
        await mutation.run({ email, password });
        onRegistrationComplete(email);
      } catch (error) {
        setFailure(classifyError(error as ApiError, "invalid_credentials"));
        return;
      }
    },
    [email, mutation, onRegistrationComplete, password],
  );

  const progressState = buildSetupProgressState(onboarding, false);

  return (
    <section className="auth-shell">
      <article className="glass-card auth-card">
        <p className="eyebrow">Sleep OS</p>
        <SetupProgress onboarding={progressState} currentStep="signup" />
        <h2>Create account</h2>
        <p className="state">Set up your account to start tracking nights.</p>
        <form onSubmit={submit} className="form-stack">
          <input value={email} onChange={(event) => setEmail(event.target.value)} placeholder="Email" />
          <input value={password} onChange={(event) => setPassword(event.target.value)} type="password" placeholder="Password" />
          <button className="solid-btn" type="submit" disabled={mutation.state.status === "loading"}>
            {mutation.state.status === "loading" ? "Registering..." : "Create Account"}
          </button>
        </form>
        {failure ? <FailureNotice kind={failure} /> : null}
        {mutation.state.status === "success" ? <p className="state">Account created. Taking you to login…</p> : null}
        <div className="auth-alt-action">
          <p className="state">Already have an account?</p>
          <button className="ghost-btn" type="button" onClick={() => navigate("/login")}>
            Back to login
          </button>
        </div>
      </article>
    </section>
  );
}

function ConnectDevicePage({
  api,
  onboarding,
  onConnected,
}: {
  api: ApiClient;
  onboarding: OnboardingState;
  onConnected: () => void;
}) {
  const [deviceId, setDeviceId] = useState("R01");
  const [pairingCode, setPairingCode] = useState("934122");
  const [requiresPairingCode, setRequiresPairingCode] = useState(true);
  const [failure, setFailure] = useState<FailureKind | undefined>(undefined);
  const [connected, setConnected] = useState(false);

  const mutation = useMutation(async (payload: { deviceId: string; pairingCode: string }, signal) => {
    if (payload.deviceId.trim().length === 0) {
      throw { kind: "http", status: 404, message: "Device not found." } satisfies ApiError;
    }

    if (requiresPairingCode) {
      const pairing = await api.startPairing({ device_id: payload.deviceId }, signal);
      const pairingId = typeof pairing.pairing_id === "string" ? pairing.pairing_id : "";
      if (pairingId.length === 0 || payload.pairingCode.trim().length === 0) {
        throw { kind: "http", status: 408, message: "Pairing timeout." } satisfies ApiError;
      }
      await api.claimPairing({ pairing_id: pairingId, code: payload.pairingCode }, signal);
    }

    return api.claimDeviceById({ external_id: payload.deviceId }, signal);
  });

  const submit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      setFailure(undefined);
      setConnected(false);

      try {
        await mutation.run({ deviceId, pairingCode });
        setConnected(true);
        onConnected();
      } catch (error) {
        setFailure(classifyError(error as ApiError, "device_not_found"));
      }
    },
    [deviceId, mutation, onConnected, pairingCode],
  );

  const progressState = buildSetupProgressState(onboarding, true);

  return (
    <section className="auth-shell">
      <article className="glass-card auth-card">
        <p className="eyebrow">Sleep OS</p>
        <SetupProgress onboarding={progressState} currentStep="connect-device" />
        <h2>Connect your device</h2>
        <p className="state">Pair your tracker to this account before opening sleep insights.</p>
        <form onSubmit={submit} className="form-stack">
          <label>
            Device ID
            <input
              value={deviceId}
              onChange={(event) => setDeviceId(event.target.value)}
              placeholder="R01"
              autoComplete="off"
            />
          </label>
          <div className="chip-wrap" role="group" aria-label="Pairing flow">
            <button
              type="button"
              className={`chip-btn ${requiresPairingCode ? "active" : ""}`}
              onClick={() => setRequiresPairingCode(true)}
            >
              Pair with code
            </button>
            <button
              type="button"
              className={`chip-btn ${!requiresPairingCode ? "active" : ""}`}
              onClick={() => setRequiresPairingCode(false)}
            >
              Quick connect
            </button>
          </div>
          {requiresPairingCode ? (
            <label>
              Pairing code
              <input
                value={pairingCode}
                onChange={(event) => setPairingCode(event.target.value)}
                placeholder="934122"
                autoComplete="off"
              />
            </label>
          ) : null}
          <button className="solid-btn" type="submit" disabled={mutation.state.status === "loading"}>
            {mutation.state.status === "loading" ? "Connecting..." : "Connect device"}
          </button>
        </form>
        {failure ? <FailureNotice kind={failure} /> : null}
        {connected ? <p className="state">Device connected to this account.</p> : null}
      </article>
    </section>
  );
}

function StartTrackingPage({
  api,
  onboarding,
  onTrackingStarted,
}: {
  api: ApiClient;
  onboarding: OnboardingState;
  onTrackingStarted: () => void;
}) {
  const [failure, setFailure] = useState<FailureKind | undefined>(undefined);
  const [confirmed, setConfirmed] = useState(false);

  const mutation = useMutation(async (_payload: { activate: true }, signal) => {
    return api.startTracking(signal);
  });

  const startTracking = useCallback(async () => {
    setFailure(undefined);
    try {
      await mutation.run({ activate: true });
      setConfirmed(true);
      onTrackingStarted();
    } catch (error) {
      setFailure(classifyError(error as ApiError, "network_issue"));
    }
  }, [mutation, onTrackingStarted]);

  const progressState = buildSetupProgressState(onboarding, true);

  return (
    <section className="auth-shell">
      <article className="glass-card auth-card">
        <p className="eyebrow">Sleep OS</p>
        <SetupProgress onboarding={progressState} currentStep="start-tracking" />
        <h2>Start tracking</h2>
        <p className="state">Enable overnight tracking to begin collecting tonight's sleep data.</p>
        <button className="solid-btn" type="button" onClick={startTracking} disabled={mutation.state.status === "loading"}>
          {mutation.state.status === "loading" ? "Starting..." : "Start tracking now"}
        </button>
        {failure ? <FailureNotice kind={failure} /> : null}
        {confirmed ? <p className="state">Tracking active.</p> : null}
      </article>
    </section>
  );
}

function AccessBlocked({ title, message, onNavigate }: { title: string; message: string; onNavigate: (path: string) => void }) {
  return (
    <section className="auth-shell">
      <article className="glass-card auth-card">
        <h2>{title}</h2>
        <p className="state error">{message}</p>
        <button className="solid-btn" onClick={() => onNavigate("/login")}>Open login</button>
      </article>
    </section>
  );
}

function NavSection({
  title,
  paths,
  currentPath,
  navigate,
}: {
  title: string;
  paths: string[];
  currentPath: string;
  navigate: (path: string) => void;
}) {
  return (
    <div className="nav-section">
      <h4>{title}</h4>
      <div className="nav-stack">
        {paths.map((path) => (
          <button
            key={path}
            className={`nav-btn ${normalizePath(path) === normalizePath(currentPath) ? "active" : ""}`}
            onClick={() => navigate(path)}
          >
            {routeTitle(path)}
          </button>
        ))}
      </div>
    </div>
  );
}

function renderContent(routePath: string, api: ApiClient, navigate: (path: string) => void, params: Record<string, string>): ReactNode {
  switch (routePath) {
    case "/login":
    case "/register":
      return null;
    case "/app/home":
      return <UserHomePage api={api} />;
    case "/app/sleep/latest":
      return <SleepLatestPage api={api} />;
    case "/app/sleep/sync":
      return <SleepSyncPage api={api} />;
    case "/app/sleep/insights":
      return <SleepInsightsPage api={api} />;
    case "/app/devices":
      return <DevicesPage api={api} />;
    case "/app/devices/claim":
      return <DevicesClaimPage api={api} />;
    case "/app/devices/pair":
      return <DevicesPairPage api={api} />;
    case "/app/recordings":
      return <RecordingsPage api={api} navigate={navigate} />;
    case "/app/recordings/:recordingId":
      return <RecordingDetailPage api={api} recordingId={params.recordingId} />;
    case "/app/alarm":
      return <AlarmPage api={api} />;
    case "/app/routines":
      return <RoutinesPage api={api} />;
    case "/app/coach":
      return <CoachPage api={api} />;
    case "/app/challenges":
      return <ChallengesPage api={api} />;
    case "/app/search":
      return <SearchPage api={api} />;
    case "/admin/models":
      return <ModelsPage api={api} navigate={navigate} />;
    case "/admin/models/:version":
      return <ModelVersionPage api={api} version={params.version} />;
    case "/admin/experiments":
      return <ExperimentsPage api={api} />;
    case "/admin/feature-schemas":
      return <FeatureSchemasPage api={api} navigate={navigate} />;
    case "/admin/feature-schemas/:version":
      return <FeatureSchemaVersionPage api={api} version={params.version} />;
    case "/admin/evaluation":
      return <EvaluationPage api={api} />;
    case "/admin/drift":
      return <DriftPage api={api} />;
    case "/admin/replay":
      return <ReplayPage api={api} />;
    case "/internal/monitoring":
      return <MonitoringPage api={api} />;
    case "/internal/performance":
      return <PerformancePage api={api} />;
    case "/internal/stress":
      return <StressPage api={api} />;
    case "/internal/faults":
      return <FaultsPage api={api} />;
    case "/internal/timescale":
      return <TimescalePage api={api} />;
    case "/internal/audit":
      return <AuditPage api={api} />;
    default:
      return <p className="state error">This page is not available yet.</p>;
  }
}

function App() {
  const env = getAppEnv();
  const { pathname, navigate } = useRoute();
  const initialToken = useMemo(() => readStoredToken(), []);
  const [onboarding, setOnboarding] = useState<OnboardingState>(() => readOnboardingState());
  const onboardingRef = useRef(onboarding);
  const [session, setSession] = useState<SessionState>({
    status: initialToken ? "checking" : "anonymous",
    token: initialToken,
    access: { canUseApp: false, canUseAdmin: false, canUseInternal: false },
  });
  const tokenRef = useRef<string | undefined>(initialToken);

  useEffect(() => {
    onboardingRef.current = onboarding;
  }, [onboarding]);

  const api = useMemo(() => createApiClient(() => tokenRef.current), []);

  const hydrateSession = useCallback(
    async (token: string | undefined, options?: { pendingLoginEmail?: string }) => {
      tokenRef.current = token;

      if (!token) {
        setSession({
          status: "anonymous",
          token: undefined,
          access: { canUseApp: false, canUseAdmin: false, canUseInternal: false },
        });
        return undefined;
      }

      setSession((current) => ({ ...current, status: "checking", token }));
      try {
        const [authMe, accountMe, tenantMe, devicesPayload, syncStatus] = await Promise.all([
          api.authMe(),
          api.accountMe(),
          api.tenantMe(),
          api.devices().catch(() => undefined),
          api.syncStatus().catch(() => undefined),
        ]);

        const reconciledOnboarding = reconcileOnboardingState(onboardingRef.current, {
          isAuthenticated: true,
          hasConnectedDevice: deriveConnectedDeviceSignal(devicesPayload),
          trackingActive: deriveTrackingSignal(syncStatus),
          pendingLoginEmail: options?.pendingLoginEmail,
        });
        onboardingRef.current = reconciledOnboarding;
        persistOnboardingState(reconciledOnboarding);
        setOnboarding(reconciledOnboarding);

        setSession({
          status: "authenticated",
          token,
          authMe,
          accountMe,
          tenantMe,
          access: deriveAccess(authMe, accountMe),
        });
        return reconciledOnboarding;
      } catch (error) {
        purgeAuthState(() => {
          tokenRef.current = undefined;
        });
        setSession({
          status: "anonymous",
          token: undefined,
          access: { canUseApp: false, canUseAdmin: false, canUseInternal: false },
          error: error instanceof Error ? error.message : "Session check failed.",
        });
        return undefined;
      }
    },
    [api],
  );

  useEffect(() => {
    void hydrateSession(initialToken);
  }, [hydrateSession, initialToken]);

  const setAndPersistOnboarding = useCallback((updater: (current: OnboardingState) => OnboardingState) => {
    setOnboarding((current) => {
      const next = updater(current);
      persistOnboardingState(next);
      return next;
    });
  }, []);

  const updateOnboarding = useCallback(
    (patch: Partial<OnboardingState>) => {
      setAndPersistOnboarding((current) => ({ ...current, ...patch }));
    },
    [setAndPersistOnboarding],
  );

  const onLoggedIn = useCallback(
    async (token: string, email: string) => {
      persistToken(token);
      const optimisticOnboarding = reconcileOnboardingState(onboardingRef.current, {
        isAuthenticated: true,
        pendingLoginEmail: email,
      });
      onboardingRef.current = optimisticOnboarding;
      persistOnboardingState(optimisticOnboarding);
      setOnboarding(optimisticOnboarding);

      const hydratedOnboarding = await hydrateSession(token, { pendingLoginEmail: email });
      return nextOnboardingRoute(hydratedOnboarding ?? optimisticOnboarding);
    },
    [hydrateSession],
  );

  const onRegistrationComplete = useCallback(
    (email: string) => {
      updateOnboarding({
        lastOnboardingStepCompleted: "signup",
        pendingLoginEmail: email,
        banner: {
          tone: "success",
          message: "Account created. Sign in to continue setup.",
        },
      });
      navigate("/login");
    },
    [navigate, updateOnboarding],
  );

  const onDeviceConnected = useCallback(() => {
    updateOnboarding({
      hasConnectedDevice: true,
      lastOnboardingStepCompleted: "connect-device",
      banner: {
        tone: "success",
        message: "Device connected. Start tracking to finish setup.",
      },
    });
    navigate("/onboarding/start-tracking");
  }, [navigate, updateOnboarding]);

  const onTrackingStarted = useCallback(() => {
    updateOnboarding({
      trackingActive: true,
      lastOnboardingStepCompleted: "start-tracking",
      banner: {
        tone: "success",
        message: "Tracking active. Your sleep dashboard is ready.",
      },
    });
    window.setTimeout(() => {
      navigate("/app/sleep/latest");
    }, 500);
  }, [navigate, updateOnboarding]);

  const dismissBanner = useCallback(() => {
    setAndPersistOnboarding((current) => ({ ...current, banner: undefined }));
  }, [setAndPersistOnboarding]);

  const resetOnboarding = useCallback(() => {
    clearOnboardingState();
    setOnboarding(readOnboardingState());
  }, []);

  const setupProgress = useMemo(
    () => buildSetupProgressState(onboarding, session.status === "authenticated"),
    [onboarding, session.status],
  );

  const logout = useCallback(() => {
    persistToken(undefined);
    resetOnboarding();
    void hydrateSession(undefined);
    navigate("/login");
  }, [hydrateSession, navigate, resetOnboarding]);

  const match = matchRoute(pathname);
  const effectivePath = match ? pathname : session.status === "authenticated" ? "/app/home" : "/login";
  const resolvedMatch = matchRoute(effectivePath) ?? { route: ROUTES[0], params: {} };
  const shouldRedirect = !match && effectivePath !== pathname;

  useEffect(() => {
    if (shouldRedirect) {
      navigate(effectivePath, true);
    }
  }, [effectivePath, navigate, shouldRedirect]);

  const onboardingRedirect =
    session.status === "authenticated" && resolvedMatch.route.scope === "user"
      ? resolveOnboardingRedirect(resolvedMatch.route.path, onboarding)
      : undefined;

  useEffect(() => {
    if (onboardingRedirect && onboardingRedirect.path !== pathname) {
      navigate(onboardingRedirect.path, true);
    }
  }, [navigate, onboardingRedirect, pathname]);

  if (shouldRedirect) {
    return null;
  }

  const isPublic = resolvedMatch.route.scope === "public";
  const needsAuth = resolvedMatch.route.scope !== "public";
  const isCoreAppRoute = resolvedMatch.route.path.startsWith("/app/");

  if (needsAuth && session.status === "checking") {
    return (
      <main className="shell">
        <section className="auth-shell">
          <article className="glass-card auth-card">
            <h2>Preparing your dashboard...</h2>
            <p className="state">Just a moment while we load your sleep view.</p>
          </article>
        </section>
      </main>
    );
  }

  if (needsAuth && session.status !== "authenticated") {
    return <AccessBlocked title="Sign in required" message="Please sign in to open this page." onNavigate={navigate} />;
  }

  if (resolvedMatch.route.scope === "admin" && !session.access.canUseAdmin) {
    return <AccessBlocked title="Additional access required" message="This area is available to care program leads." onNavigate={navigate} />;
  }

  if (resolvedMatch.route.scope === "internal" && !session.access.canUseInternal) {
    return <AccessBlocked title="Additional access required" message="This area is available to reliability specialists." onNavigate={navigate} />;
  }

  if (resolvedMatch.route.scope === "user" && isCoreAppRoute && !session.access.canUseApp) {
    const claimsMissing = !hasProvisioningClaimData(session.authMe) && !hasProvisioningClaimData(session.accountMe);
    if (claimsMissing) {
      return (
        <AccessBlocked
          title="Finalizing account setup"
          message="Sign-in worked, but your app access claims are still provisioning. This is safe. Finish onboarding steps, then refresh in a minute. If this persists, contact support and mention your account email."
          onNavigate={navigate}
        />
      );
    }

    return <AccessBlocked title="Account setup required" message="This account is not ready for the sleep dashboard yet." onNavigate={navigate} />;
  }

  if (onboardingRedirect && onboardingRedirect.path !== pathname) {
    return null;
  }

  if (resolvedMatch.route.path === "/login") {
    return <LoginPage api={api} onLoggedIn={onLoggedIn} onboarding={onboarding} navigate={navigate} />;
  }

  if (resolvedMatch.route.path === "/register") {
    return <RegisterPage api={api} onboarding={onboarding} onRegistrationComplete={onRegistrationComplete} navigate={navigate} />;
  }

  if (resolvedMatch.route.path === "/onboarding/connect-device") {
    return <ConnectDevicePage api={api} onboarding={onboarding} onConnected={onDeviceConnected} />;
  }

  if (resolvedMatch.route.path === "/onboarding/start-tracking") {
    return <StartTrackingPage api={api} onboarding={onboarding} onTrackingStarted={onTrackingStarted} />;
  }

  const content = renderContent(resolvedMatch.route.path, api, navigate, resolvedMatch.params);

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Sleep OS</p>
          <h1>{routeTitle(resolvedMatch.route.path)}</h1>
        </div>
        <div className="topbar-meta">
          <span className="badge">experience: {env.mode}</span>
          {session.status === "authenticated" ? <span className="badge">home space: {String(session.tenantMe?.name ?? "unknown")}</span> : null}
          {session.status === "authenticated" && onboarding.trackingActive ? <span className="badge">tracking active</span> : null}
          {session.error ? <span className="badge error">sign-in needs attention</span> : null}
          {!isPublic ? (
            <button className="ghost-btn" onClick={logout}>
              Sign out
            </button>
          ) : null}
        </div>
      </header>

      <div className="layout-grid">
        {!isPublic ? (
          <aside className="side-nav glass-card">
            <NavSection title="Sleep" paths={USER_NAV} currentPath={pathname} navigate={navigate} />
            {session.access.canUseAdmin ? <NavSection title="Programs" paths={ADMIN_NAV} currentPath={pathname} navigate={navigate} /> : null}
            {session.access.canUseInternal ? <NavSection title="Reliability" paths={INTERNAL_NAV} currentPath={pathname} navigate={navigate} /> : null}
          </aside>
        ) : null}
        <section>
          {onboarding.banner ? <SetupBanner banner={onboarding.banner} onDismiss={dismissBanner} /> : null}
          {resolvedMatch.route.path === "/app/home" && isOnboardingIncomplete(onboarding) ? (
            <ResumeSetupCard route={nextOnboardingRoute(onboarding)} onResume={navigate} />
          ) : null}
          {resolvedMatch.route.path.startsWith("/app/") && isOnboardingIncomplete(onboarding) ? (
            <SetupProgress onboarding={setupProgress} currentStep={currentOnboardingStep(onboarding)} />
          ) : null}
          {content}
        </section>
      </div>
    </main>
  );
}

export default App;
