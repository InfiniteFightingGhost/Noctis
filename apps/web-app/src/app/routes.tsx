import { lazy, type ReactNode } from "react";
import { Navigate, type RouteObject } from "react-router-dom";
import { isDeviceConnected, isLoggedIn } from "../auth/session";
import { AppLayout } from "../components/layout/AppLayout";

const HomePage = lazy(() => import("../pages/HomePage"));
const TrendsPage = lazy(() => import("../pages/TrendsPage"));
const NightPage = lazy(() => import("../pages/NightPage"));
const SettingsPage = lazy(() => import("../pages/SettingsPage"));
const LoginPage = lazy(() => import("../pages/LoginPage"));
const SignupPage = lazy(() => import("../pages/SignupPage"));
const ConnectDevicePage = lazy(() => import("../pages/ConnectDevicePage"));
const NotFoundPage = lazy(() => import("../pages/NotFoundPage"));

type GuardProps = {
  children: ReactNode;
};

function RequireAuth({ children }: GuardProps) {
  if (!isLoggedIn()) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

function RedirectIfAuthenticated({ children }: GuardProps) {
  if (isLoggedIn()) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
}

function RequireDeviceConnected({ children }: GuardProps) {
  if (!isDeviceConnected()) {
    return <Navigate to="/connect-device" replace />;
  }

  return <>{children}</>;
}

function RedirectIfDeviceConnected({ children }: GuardProps) {
  if (isDeviceConnected()) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
}

export const ROUTE_TABLE = [
  { path: "/", label: "Home" },
  { path: "/trends", label: "Trends" },
  { path: "/night", label: "Night" },
  { path: "/settings", label: "Settings" },
] as const;

export const routeObjects: RouteObject[] = [
  {
    path: "/",
    element: (
      <RequireAuth>
        <RequireDeviceConnected>
          <AppLayout />
        </RequireDeviceConnected>
      </RequireAuth>
    ),
    children: [
      { index: true, element: <HomePage /> },
      { path: "trends", element: <TrendsPage /> },
      { path: "night", element: <NightPage /> },
      { path: "settings", element: <SettingsPage /> },
    ],
  },
  {
    path: "/login",
    element: (
      <RedirectIfAuthenticated>
        <LoginPage />
      </RedirectIfAuthenticated>
    ),
  },
  {
    path: "/signup",
    element: (
      <RedirectIfAuthenticated>
        <SignupPage />
      </RedirectIfAuthenticated>
    ),
  },
  {
    path: "/connect-device",
    element: (
      <RequireAuth>
        <RedirectIfDeviceConnected>
          <ConnectDevicePage />
        </RedirectIfDeviceConnected>
      </RequireAuth>
    ),
  },
  {
    path: "*",
    element: <NotFoundPage />,
  },
];
