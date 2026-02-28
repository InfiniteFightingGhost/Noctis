import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiClient } from "../api/apiClient";
import { setDeviceConnected } from "../auth/session";

export default function ConnectDevicePage() {
  const navigate = useNavigate();
  const [deviceExternalId, setDeviceExternalId] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleConnect = async () => {
    try {
      setIsSubmitting(true);
      setErrorMessage("");

      if (!deviceExternalId.trim()) {
        setStatusMessage("");
        setErrorMessage("Device external ID is required.");
        return;
      }

      const result = await apiClient.connectDevice({ deviceExternalId: deviceExternalId.trim() });
      setDeviceConnected(true);
      setStatusMessage(result.message);
      navigate("/", { replace: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to connect device.";
      setStatusMessage("");
      setErrorMessage(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="auth-page">
      <section className="auth-card">
        <p className="eyebrow">Device setup</p>
        <h1>Connect mountable device</h1>
        <p>Pair your mounted sensor node to start nightly ingestion.</p>

        <div className="mountable-preview" aria-label="mountable-device-preview">
          <p>
            <strong>Model:</strong> Noctis Halo S1 Mount
          </p>
          <p>
            <strong>Power:</strong> Wired
          </p>
          <p>
            <strong>Mount status:</strong> Anchored
          </p>
        </div>

        <label>
          Device external ID
          <input
            type="text"
            value={deviceExternalId}
            onChange={(event) => setDeviceExternalId(event.target.value)}
            placeholder="e.g. noctis-halo-s1-001"
            autoCapitalize="none"
            autoCorrect="off"
            required
          />
        </label>

        <button type="button" className="pill auth-submit" onClick={handleConnect} disabled={isSubmitting}>
          {isSubmitting ? "Connecting..." : "Connect device"}
        </button>
        {statusMessage ? <p className="status-message">{statusMessage}</p> : null}
        {errorMessage ? <p className="status-message" role="alert">{errorMessage}</p> : null}
      </section>
    </main>
  );
}
