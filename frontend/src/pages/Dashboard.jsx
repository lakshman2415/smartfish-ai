import { useCallback, useEffect, useState } from "react";
import { fetchHistory, fetchWaterQuality, prettyClassName, severityClass, timeAgo } from "../api.js";
import PageHeader from "../components/PageHeader.jsx";

const COVERAGE = [
  { name: "Bacterial Infection", live: true },
  { name: "Fungal Infection", live: true },
  { name: "Parasitic Infection", live: true },
  { name: "Viral Infection", live: true },
  { name: "Fin & Tail Rot", live: true, note: "(joint)" },
  { name: "Gill Disease", live: true },
  { name: "Ulcer Disease", live: true },
  { name: "Healthy", live: true },
  { name: "Ich (White Spot)", live: false },
  { name: "Dropsy", live: false },
];

export default function Dashboard() {
  const [rows, setRows] = useState([]);
  const [waterStatus, setWaterStatus] = useState("-");

  const load = useCallback(async () => {
    try {
      setRows(await fetchHistory(500));
    } catch {
      setRows([]);
    }
    try {
      const wq = await fetchWaterQuality();
      setWaterStatus(wq.status);
    } catch {
      setWaterStatus("unreachable");
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const total = rows.length;
  const healthy = rows.filter((r) => r.predicted_class === "Healthy").length;
  const diseased = total - healthy;
  const avgConfidence = total ? rows.reduce((s, r) => s + r.confidence, 0) / total : null;
  const highSev = rows.filter((r) => r.severity === "High" || r.severity === "Critical").length;

  const counts = {};
  for (const r of rows) counts[r.predicted_class] = (counts[r.predicted_class] || 0) + 1;
  const maxCount = Math.max(1, ...Object.values(counts));
  const distribution = Object.entries(counts).sort((a, b) => b[1] - a[1]);

  return (
    <>
      <PageHeader
        eyebrow="AI Dashboard"
        title="Aquaculture intelligence overview"
        subtitle="Live insights computed from your real detection history - analyze a few fish images to populate this."
      />
      <main className="page">
      <div className="stat-grid">
        <div className="stat-tile"><div className="stat-tile-value">{total}</div><div className="stat-tile-label">Total Fish Analyzed</div></div>
        <div className="stat-tile"><div className="stat-tile-value">{healthy}</div><div className="stat-tile-label">Healthy Fish</div></div>
        <div className="stat-tile"><div className="stat-tile-value">{diseased}</div><div className="stat-tile-label">Diseased Fish</div></div>
        <div className="stat-tile"><div className="stat-tile-value">{avgConfidence === null ? "—" : Math.round(avgConfidence * 100) + "%"}</div><div className="stat-tile-label">Avg AI Confidence</div></div>
      </div>

      <div className="dashboard-grid">
        <section className="card">
          <h2>Disease Distribution</h2>
          {distribution.length === 0 ? (
            <div className="empty-state">No detections recorded yet.</div>
          ) : (
            <div className="bar-chart">
              {distribution.map(([name, count]) => (
                <div className="bar-row" key={name}>
                  <div className="bar-name">{prettyClassName(name)}</div>
                  <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.round((count / maxCount) * 100)}%` }} /></div>
                  <div className="bar-count">{count}</div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="card">
          <h2>Disease Coverage</h2>
          <p className="coverage-intro">9 diseases diagnosing live today, with more added as new labeled data comes in.</p>
          <div className="coverage-grid">
            {COVERAGE.map((c) => (
              <div className={"coverage-item" + (c.live ? "" : " roadmap")} key={c.name}>
                <span className={"dot " + (c.live ? "dot-live" : "dot-roadmap")} />
                {c.name}
                {c.note && <span className="note">{c.note}</span>}
                {!c.live && <span className="tag-roadmap">In Development</span>}
              </div>
            ))}
          </div>
          <table className="spec-table" style={{ marginTop: 14 }}>
            <tbody>
              <tr><td>Architecture</td><td>EfficientNet-B4 + ViT-B/16, attention fusion</td></tr>
              <tr><td>Fish localization</td><td>Classical CV crop (not a trained detector)</td></tr>
            </tbody>
          </table>
        </section>
      </div>

      <div className="stat-grid" style={{ marginTop: 20 }}>
        <div className="stat-tile"><div className="stat-tile-value">{highSev}</div><div className="stat-tile-label">High-Severity Cases</div></div>
        <div className="stat-tile"><div className="stat-tile-value">{total ? Math.round((healthy / total) * 100) + "%" : "—"}</div><div className="stat-tile-label">Healthy Rate</div></div>
        <div className="stat-tile"><div className="stat-tile-value">{waterStatus}</div><div className="stat-tile-label">Water Quality Status</div></div>
        <div className="stat-tile"><div className="stat-tile-value">9/11</div><div className="stat-tile-label">Diseases Live in Production</div></div>
      </div>

      <section className="card" style={{ marginTop: 20 }}>
        <div className="card-header-row">
          <h2>Diagnosis History</h2>
          <button className="btn tiny" onClick={load}>Refresh</button>
        </div>
        <div className="table-scroll">
          <table className="history-table">
            <thead>
              <tr><th>Time</th><th>Image</th><th>Predicted</th><th>Confidence</th><th>Severity</th></tr>
            </thead>
            <tbody>
              {rows.length === 0 && (
                <tr><td colSpan={5} className="empty-row">No diagnoses yet - try the <a href="/detection">Detection</a> page.</td></tr>
              )}
              {rows.slice(0, 100).map((r) => (
                <tr key={r.id}>
                  <td>{timeAgo(r.timestamp)}</td>
                  <td>{r.filename}</td>
                  <td>{prettyClassName(r.predicted_class)}</td>
                  <td>{Math.round(r.confidence * 100)}%</td>
                  <td><span className={"severity-badge " + severityClass(r.severity)}>{r.severity}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      </main>
    </>
  );
}
