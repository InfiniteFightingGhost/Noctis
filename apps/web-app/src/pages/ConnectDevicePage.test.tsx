import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AppShell } from "../App";
import { setDeviceConnected, setLoggedIn } from "../auth/session";

describe("connect device page", () => {
  it("shows an error when the device external ID does not exist", async () => {
    setLoggedIn(true);
    setDeviceConnected(false);

    render(
      <MemoryRouter initialEntries={["/connect-device"]}>
        <AppShell />
      </MemoryRouter>,
    );

    fireEvent.change(await screen.findByLabelText("Device external ID"), {
      target: { value: "noctis-halo-s1-404" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Connect device" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Device not found. Check the external ID and try again.");
  });
});
