import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { apiClient } from "../api/apiClient";
import { isDeviceConnected, setAuthSession } from "../auth/session";

export default function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    try {
      setIsSubmitting(true);
      setErrorMessage("");
      const authResult = await apiClient.login({ email, password });
      setStatusMessage("Authentication succeeded.");
      setAuthSession({
        accessToken: authResult.access_token,
        userId: authResult.user.id,
      });

      if (isDeviceConnected()) {
        navigate("/", { replace: true });
        return;
      }

      navigate("/connect-device", { replace: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to log in.";
      setStatusMessage("");
      setErrorMessage(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="auth-page">
      <section className="auth-card">
        <p className="eyebrow">Access</p>
        <h1>Log in to Noctis</h1>
        <p>Continue to your clinical sleep intelligence dashboard.</p>

        <form onSubmit={handleSubmit} className="auth-form">
          <label>
            Email
            <input
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              autoComplete="email"
              inputMode="email"
              required
            />
          </label>
          <label>
            Password
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              minLength={8}
              required
            />
          </label>
          <button type="submit" className="pill auth-submit" disabled={isSubmitting}>
            {isSubmitting ? "Logging in..." : "Log in"}
          </button>
        </form>

        {statusMessage ? <p className="status-message">{statusMessage}</p> : null}
        {errorMessage ? <p className="status-message" role="alert">{errorMessage}</p> : null}

        <p className="auth-link-copy">
          Prefer link navigation? <Link to="/signup">Go to sign up</Link>
        </p>
      </section>
    </main>
  );
}
