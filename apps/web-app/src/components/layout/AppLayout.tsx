import { Suspense, useEffect, useMemo, useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { PrimaryNavigation } from "./PrimaryNavigation";

type ThemePreference = "light" | "dark";

const MOBILE_BREAKPOINT_PX = 768;
const THEME_STORAGE_KEY = "noctis-theme-preference";

function canUseMatchMedia(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function";
}

function getSystemPrefersDark(): boolean {
  if (!canUseMatchMedia()) {
    return false;
  }

  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function getStoredThemePreference(): ThemePreference {
  if (typeof window === "undefined") {
    return "light";
  }

  const storage = window.localStorage;
  if (!storage || typeof storage.getItem !== "function") {
    return getSystemPrefersDark() ? "dark" : "light";
  }

  const stored = storage.getItem(THEME_STORAGE_KEY);
  if (stored === "light" || stored === "dark") {
    return stored;
  }

  return getSystemPrefersDark() ? "dark" : "light";
}

export function AppLayout() {
  const location = useLocation();
  const [isScrolled, setIsScrolled] = useState(false);
  const [now, setNow] = useState(() => new Date());
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [themePreference, setThemePreference] = useState<ThemePreference>(() => getStoredThemePreference());

  useEffect(() => {
    const onScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 60_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const storage = window.localStorage;
    if (!storage || typeof storage.setItem !== "function") {
      return;
    }

    storage.setItem(THEME_STORAGE_KEY, themePreference);
  }, [themePreference]);

  const resolvedTheme = themePreference;

  useEffect(() => {
    document.documentElement.dataset.theme = resolvedTheme;
  }, [resolvedTheme]);

  useEffect(() => {
    setIsMobileMenuOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= MOBILE_BREAKPOINT_PX) {
        setIsMobileMenuOpen(false);
      }
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    const body = document.body;
    body.classList.toggle("mobile-nav-open", isMobileMenuOpen);
    return () => body.classList.remove("mobile-nav-open");
  }, [isMobileMenuOpen]);

  useEffect(() => {
    if (!isMobileMenuOpen) {
      return;
    }

    const onEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsMobileMenuOpen(false);
      }
    };

    window.addEventListener("keydown", onEscape);
    return () => window.removeEventListener("keydown", onEscape);
  }, [isMobileMenuOpen]);

  const timeLabel = useMemo(
    () =>
      new Intl.DateTimeFormat(undefined, {
        hour: "numeric",
        minute: "2-digit",
      }).format(now),
    [now],
  );

  const isDarkTheme = resolvedTheme === "dark";

  return (
    <div className="app-shell">
      <header className={`app-header ${isScrolled ? "scrolled" : ""}`}>
        <div className="header-top-row">
          <div className="header-left-cluster">
            <p className="time-chip" aria-label="Current time">
              {timeLabel}
            </p>
            <button
              type="button"
              className="icon-button"
              aria-label={isDarkTheme ? "Switch to light theme" : "Switch to dark theme"}
              onClick={() => setThemePreference(isDarkTheme ? "light" : "dark")}
              aria-pressed={isDarkTheme}
            >
              <span className="icon-glyph" aria-hidden="true">
                {isDarkTheme ? "☾" : "☼"}
              </span>
            </button>
            <button
              type="button"
              className="icon-button mobile-menu-button"
              aria-label={isMobileMenuOpen ? "Close menu" : "Open menu"}
              aria-expanded={isMobileMenuOpen}
              aria-controls="primary-navigation"
              onClick={() => setIsMobileMenuOpen((current) => !current)}
            >
              <span className="icon-glyph" aria-hidden="true">
                {isMobileMenuOpen ? "✕" : "☰"}
              </span>
            </button>
          </div>

          <div className="brand-cluster" aria-label="Brand">
            <span className="brand-icon" aria-hidden="true" />
            <span className="brand-text">Noctis</span>
          </div>

          <div className="header-right-cluster">
            <span className="avatar-bubble" aria-hidden="true" />
            <strong className="app-availability">App available</strong>
          </div>
        </div>

        <button
          type="button"
          className={`mobile-nav-backdrop ${isMobileMenuOpen ? "is-open" : ""}`}
          onClick={() => setIsMobileMenuOpen(false)}
          aria-label="Close navigation menu"
          tabIndex={isMobileMenuOpen ? 0 : -1}
        />
        <div id="primary-navigation" className={`tabs-shell ${isMobileMenuOpen ? "is-open" : ""}`}>
          <PrimaryNavigation onNavigate={() => setIsMobileMenuOpen(false)} />
        </div>
      </header>
      <main>
        <Suspense fallback={<div className="page-state">Loading page...</div>}>
          <Outlet />
        </Suspense>
      </main>
    </div>
  );
}
