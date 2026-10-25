import { NavLink } from "react-router-dom";

const NAV_LINKS = [
  { to: "/", label: "Home" },
  { to: "/detection", label: "Detection" },
  { to: "/water-quality", label: "Water Quality" },
  { to: "/alerts", label: "Alerts" },
];

// decorative swimming fish around the header title -- purely visual, hidden from screen readers
const SWIMMING_FISH = [
  { emoji: "🐟", top: "18%", size: "1rem", duration: "16s", delay: "-2s" },
  { emoji: "🐠", top: "62%", size: "1.3rem", duration: "21s", delay: "-9s" },
  { emoji: "🐡", top: "75%", size: "0.9rem", duration: "18s", delay: "-13s" },
  { emoji: "🐟", top: "35%", size: "1.1rem", duration: "24s", delay: "-4s" },
  { emoji: "🐠", top: "50%", size: "0.85rem", duration: "19s", delay: "-16s" },
];

export default function Nav() {
  return (
    <header className="app-header">
      <div className="swim-layer" aria-hidden="true">
        {SWIMMING_FISH.map((f, i) => (
          <span
            key={i}
            className="swim-fish"
            style={{ top: f.top, fontSize: f.size, animationDuration: f.duration, animationDelay: f.delay }}
          >
            {f.emoji}
          </span>
        ))}
      </div>
      <div className="header-row">
        <div className="header-spacer" aria-hidden="true" />

        <div className="header-center">
          <NavLink to="/" className="header-title">
            Automated Fish Disease Diagnosis Using Hybrid AI
          </NavLink>
          <nav className="main-nav">
            {NAV_LINKS.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                end={l.to === "/"}
                className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}
              >
                {l.label}
              </NavLink>
            ))}
          </nav>
        </div>

        <div className="header-spacer header-spacer-right">
          <NavLink
            to="/dashboard"
            className={({ isActive }) => "nav-dashboard-btn" + (isActive ? " active" : "")}
          >
            Dashboard
          </NavLink>
        </div>
      </div>
    </header>
  );
}
