import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { apiClient } from "../api/apiClient";
import { setAuthSession, setDeviceConnected } from "../auth/session";

export default function SignupPage() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    try {
      setIsSubmitting(true);
      setErrorMessage("");

      if (password !== confirmPassword) {
        setStatusMessage("");
        setErrorMessage("Passwords do not match.");
        return;
      }

      const authResult = await apiClient.signup({ username, email, password });
      setStatusMessage("Account created.");
      setAuthSession({
        accessToken: authResult.access_token,
        userId: authResult.user.id,
      });
      setDeviceConnected(false);
      navigate("/connect-device", { replace: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to create account.";
      setStatusMessage("");
      setErrorMessage(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="auth-page">
      <section className="auth-card">
        <p className="eyebrow">Onboarding</p>
        <h1>Create your Noctis account</h1>
        <p>Set up secure access before connecting your mountable device.</p>

        <form onSubmit={handleSubmit} className="auth-form">
          <label>
            Username
            <input
              type="text"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              autoComplete="username"
              minLength={3}
              required
            />
          </label>
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
              autoComplete="new-password"
              minLength={8}
              required
            />
          </label>
          <label>
            Confirm password
            <input
              type="password"
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              autoComplete="new-password"
              minLength={8}
              required
            />
          </label>
          <button type="submit" className="pill auth-submit" disabled={isSubmitting}>
            {isSubmitting ? "Creating account..." : "Sign up"}
          </button>
        </form>

        {statusMessage ? <p className="status-message">{statusMessage}</p> : null}
        {errorMessage ? <p className="status-message" role="alert">{errorMessage}</p> : null}

        <p className="auth-link-copy">
          Already have access? <Link to="/login">Log in</Link>
        </p>
      </section>
    </main>
  );
}
