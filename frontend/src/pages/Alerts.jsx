import { useCallback, useEffect, useState } from "react";
import { fetchHistory, fetchWaterQuality, prettyClassName, timeAgo } from "../api.js";
import PageHeader from "../components/PageHeader.jsx";

export default function Alerts() {
  const [alerts, setAlerts] = useState(null);

  const load = useCallback(async () => {
    const found = [];

    try {
      const wq = await fetchWaterQuality();
      if (wq.status !== "Good") {
        found.push({ level: "critical", icon: "💧", title: "Water quality issue", detail: wq.status, time: "just now" });
      }
    } catch {
      found.push({ level: "warning", icon: "⚠️", title: "Backend unreachable", detail: "Couldn't reach the API to check water quality or history.", time: "just now" });
    }

    try {
      const rows = await fetchHistory(50);
      const severe = rows.filter((r) => r.severity === "High" || r.severity === "Critical");
      for (const r of severe.slice(0, 10)) {
        found.push({
          level: r.severity === "Critical" ? "critical" : "warning",
          icon: "🐟",
          title: `${r.severity} severity: ${prettyClassName(r.predicted_class)}`,
          detail: `${r.filename} · ${Math.round(r.confidence * 100)}% confidence`,
          time: timeAgo(r.timestamp),
        });
      }
    } catch {
      /* already covered by the water-quality catch above if backend is down */
    }

    setAlerts(found);
  }, []);

  useEffect(() => { load(); }, [load]);

  return (
    <>
      <PageHeader
        eyebrow="Alerts"
        title="What needs your attention"
        subtitle="Derived live from your water quality reading and recent high/critical-severity diagnoses - not a separate fake feed."
      />
      <main className="page">
      <section className="card">
        <div className="card-header-row">
          <h2>Active Alerts</h2>
          <button className="btn tiny" onClick={load}>Refresh</button>
        </div>
        <div className="alert-list">
          {alerts === null && <div className="empty-state">Loading…</div>}
          {alerts && alerts.length === 0 && (
            <div className="alert-card ok">
              <div className="alert-icon">✅</div>
              <div>
                <div className="alert-title">All clear</div>
                <div className="alert-detail">Water quality is good and no high/critical-severity diagnoses recently.</div>
              </div>
            </div>
          )}
          {alerts && alerts.map((a, i) => (
            <div className={`alert-card ${a.level}`} key={i}>
              <div className="alert-icon">{a.icon}</div>
              <div>
                <div className="alert-title">{a.title}</div>
                <div className="alert-detail">{a.detail}</div>
              </div>
              <div className="alert-time">{a.time}</div>
            </div>
          ))}
        </div>
      </section>
      </main>
    </>
  );
}
