import { Link } from "react-router-dom";
import { isDeviceConnected, isLoggedIn } from "../auth/session";

export default function NotFoundPage() {
  const destination = isLoggedIn() ? (isDeviceConnected() ? "/" : "/connect-device") : "/login";
  const ctaLabel = isLoggedIn() ? (isDeviceConnected() ? "Go to Home" : "Connect Device") : "Go to Login";

  return (
    <main className="auth-page">
      <section className="auth-card">
        <p className="eyebrow">Not Found</p>
        <h1>Page not found</h1>
        <p>The route you entered does not exist.</p>
        <Link to={destination} className="pill auth-submit">
          {ctaLabel}
        </Link>
      </section>
    </main>
  );
}
