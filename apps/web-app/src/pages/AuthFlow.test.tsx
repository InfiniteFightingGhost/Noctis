import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AppShell } from "../App";
import { setDeviceConnected, setLoggedIn } from "../auth/session";

function resetAuthForTest() {
  setLoggedIn(false);
  setDeviceConnected(false);
}

describe("auth flow", () => {
  beforeEach(() => {
    resetAuthForTest();
  });

  afterEach(() => {
    resetAuthForTest();
  });

  it("renders login page", async () => {
    setLoggedIn(false);

    render(
      <MemoryRouter initialEntries={["/login"]}>
        <AppShell />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Log in to Noctis")).toBeInTheDocument();
  });

  it("does not render the create one button", async () => {
    setLoggedIn(false);

    render(
      <MemoryRouter initialEntries={["/login"]}>
        <AppShell />
      </MemoryRouter>,
    );

    expect(screen.queryByRole("link", { name: "Create one" })).not.toBeInTheDocument();
  });

  it("navigates to signup without blank screen", async () => {
    setLoggedIn(false);

    render(
      <MemoryRouter initialEntries={["/login"]}>
        <AppShell />
      </MemoryRouter>,
    );

    fireEvent.click(await screen.findByRole("link", { name: "Go to sign up" }));
    expect(await screen.findByText("Create your Noctis account")).toBeInTheDocument();
  });

  it("redirects to login after logout", async () => {
    setLoggedIn(true);
    setDeviceConnected(true);

    render(
      <MemoryRouter initialEntries={["/settings"]}>
        <AppShell />
      </MemoryRouter>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Log out" }));
    expect(await screen.findByText("Log in to Noctis")).toBeInTheDocument();
  });

  it("forces connect-device step before app routes", async () => {
    setLoggedIn(true);
    setDeviceConnected(false);

    render(
      <MemoryRouter initialEntries={["/"]}>
        <AppShell />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Connect mountable device")).toBeInTheDocument();
  });

  it("defaults to logged out for first visit", async () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <AppShell />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Log in to Noctis")).toBeInTheDocument();
  });

  it("completes signup to connect-device to home to logout", async () => {
    render(
      <MemoryRouter initialEntries={["/signup"]}>
        <AppShell />
      </MemoryRouter>,
    );

    fireEvent.change(await screen.findByLabelText("Username"), { target: { value: "taylor_noctis" } });
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "taylor@noctis.example" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "Str0ngPass!" } });
    fireEvent.change(screen.getByLabelText("Confirm password"), { target: { value: "Str0ngPass!" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign up" }));

    expect(await screen.findByText("Connect mountable device")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Device external ID"), { target: { value: "noctis-halo-s1-001" } });
    fireEvent.click(screen.getByRole("button", { name: "Connect device" }));

    expect(await screen.findByText("Personal sleep tracking.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("link", { name: "Settings" }));
    fireEvent.click(await screen.findByRole("button", { name: "Log out" }));
    expect(await screen.findByText("Log in to Noctis")).toBeInTheDocument();
  });
});
