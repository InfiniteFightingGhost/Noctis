import { NavLink } from "react-router-dom";
import { ROUTE_TABLE } from "../../app/routes";

type PrimaryNavigationProps = {
  onNavigate?: () => void;
};

export function PrimaryNavigation({ onNavigate }: PrimaryNavigationProps) {
  return (
    <nav aria-label="Primary navigation">
      <ul className="tabs">
        {ROUTE_TABLE.map((route) => (
          <li key={route.path}>
            <NavLink
              to={route.path}
              end={route.path === "/"}
              className={({ isActive }) => (isActive ? "tab active" : "tab")}
              onClick={onNavigate}
            >
              {route.label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}
