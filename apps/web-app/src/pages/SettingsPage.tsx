import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiClient } from "../api/apiClient";
import { clearAuthSession } from "../auth/session";
import { ChartContainer } from "../components/common/ChartContainer";
import { PageState } from "../components/common/PageState";
import { useAsyncResource } from "../hooks/useAsyncResource";

export default function SettingsPage() {
  const navigate = useNavigate();
  const loadSettings = useCallback(() => apiClient.getSettings(), []);
  const { loading, error, data } = useAsyncResource(loadSettings);
  const [activeTab, setActiveTab] = useState<"profile" | "device">("profile");
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [activeAction, setActiveAction] = useState<"export" | "replace" | "logout" | null>(null);

  const triggerDownload = (fileName: string, payload: unknown) => {
    if (typeof window === "undefined") {
      return;
    }

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const objectUrl = window.URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.download = fileName;
    anchor.style.display = "none";
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    window.URL.revokeObjectURL(objectUrl);
  };

  if (loading) {
    return <PageState mode="loading" />;
  }

  if (error || !data) {
    return <PageState mode="error" message={error ?? "Unable to load settings."} />;
  }

  const handleExport = async () => {
    try {
      setActiveAction("export");
      setErrorMessage("");
      const result = await apiClient.requestDataExport();
      triggerDownload(result.fileName, result.report);
      setStatusMessage(result.message);
    } catch (actionError) {
      const message = actionError instanceof Error ? actionError.message : "Unable to start export.";
      setStatusMessage("");
      setErrorMessage(message);
    } finally {
      setActiveAction(null);
    }
  };

  const handleLogout = async () => {
    try {
      setActiveAction("logout");
      setErrorMessage("");
      await apiClient.logout();
      clearAuthSession();
      navigate("/login", { replace: true });
    } catch (actionError) {
      const message = actionError instanceof Error ? actionError.message : "Unable to log out.";
      setStatusMessage("");
      setErrorMessage(message);
    } finally {
      setActiveAction(null);
    }
  };

  const handleReplaceDevice = async () => {
    try {
      setActiveAction("replace");
      setErrorMessage("");
      const result = await apiClient.replaceDevice({
        deviceId: data.device.id,
        deviceExternalId: data.device.externalId ?? undefined,
      });
      setStatusMessage(result.message);
    } catch (actionError) {
      const message = actionError instanceof Error ? actionError.message : "Unable to replace device.";
      setStatusMessage("");
      setErrorMessage(message);
    } finally {
      setActiveAction(null);
    }
  };

  return (
    <section className="page-grid">
      <div className="filter-row" role="tablist" aria-label="Settings sections">
        <button
          className={activeTab === "profile" ? "pill is-selected" : "pill"}
          type="button"
          role="tab"
          aria-selected={activeTab === "profile"}
          onClick={() => setActiveTab("profile")}
        >
          Profile
        </button>
        <button
          className={activeTab === "device" ? "pill is-selected" : "pill"}
          type="button"
          role="tab"
          aria-selected={activeTab === "device"}
          onClick={() => setActiveTab("device")}
        >
          Device
        </button>
      </div>

      {activeTab === "profile" ? (
        <>
          <ChartContainer title="Profile" subtitle="Live user identity from backend">
            <ul className="value-list">
              <li>
                <span>Username</span>
                <strong>{data.profile.username}</strong>
              </li>
              <li>
                <span>Email</span>
                <strong>{data.profile.email}</strong>
              </li>
              <li>
                <span>Created</span>
                <strong>{new Date(data.profile.createdAt).toLocaleString()}</strong>
              </li>
              <li>
                <span>Updated</span>
                <strong>{new Date(data.profile.updatedAt).toLocaleString()}</strong>
              </li>
            </ul>
          </ChartContainer>

          <ChartContainer title="Account Actions">
            <div className="button-row">
              <button
                className={activeAction === "export" ? "pill is-selected" : "pill"}
                onClick={handleExport}
                type="button"
                disabled={activeAction !== null}
              >
                Data export
              </button>
              <button className="pill" onClick={handleLogout} type="button" aria-label="Log out" disabled={activeAction !== null}>
                Logout
              </button>
            </div>
            {statusMessage ? <p className="status-message">{statusMessage}</p> : null}
            {errorMessage ? <p className="status-message" role="alert">{errorMessage}</p> : null}
          </ChartContainer>
        </>
      ) : null}

      {activeTab === "device" ? (
        <ChartContainer title="Connected Device" subtitle="Single-device management only">
          <div className="device-card" aria-label="single-device-section">
            <p>
              <strong>Name:</strong> {data.device.name}
            </p>
            <p>
              <strong>Linked account:</strong> {data.profile.username} ({data.profile.email})
            </p>
            <p>
              <strong>Created:</strong> {new Date(data.device.createdAt).toLocaleString()}
            </p>
            <button
              className={activeAction === "replace" ? "pill is-selected" : "pill"}
              onClick={handleReplaceDevice}
              type="button"
              disabled={activeAction !== null}
            >
              Replace device
            </button>
            {statusMessage ? <p className="status-message">{statusMessage}</p> : null}
            {errorMessage ? <p className="status-message" role="alert">{errorMessage}</p> : null}
          </div>
        </ChartContainer>
      ) : null}
    </section>
  );
}
