import { useCallback, useEffect, useState } from "react";
import { evaluateWaterQuality, fetchWaterQuality, saveWaterQuality } from "../api.js";
import PageHeader from "../components/PageHeader.jsx";

const FIELDS = [
  { key: "ph", okKey: "ph_ok", icon: "🧪", label: "pH", suffix: "", step: 0.01, min: 0, max: 14 },
  { key: "temperature_c", okKey: null, icon: "🌡️", label: "Temperature", suffix: "°C", step: 0.1, min: -5, max: 45 },
  { key: "dissolved_oxygen_mg_l", okKey: "dissolved_oxygen_ok", icon: "💨", label: "Dissolved O₂", suffix: "mg/L", step: 0.01, min: 0, max: 20 },
  { key: "ammonia_mg_l", okKey: "ammonia_ok", icon: "⚗️", label: "Ammonia", suffix: "mg/L", step: 0.001, min: 0, max: 10 },
];

function parseForm(form) {
  const ph = parseFloat(form.ph);
  const temperature_c = parseFloat(form.temperature_c);
  const dissolved_oxygen_mg_l = parseFloat(form.dissolved_oxygen_mg_l);
  const ammonia_mg_l = parseFloat(form.ammonia_mg_l);
  if ([ph, temperature_c, dissolved_oxygen_mg_l, ammonia_mg_l].some(Number.isNaN)) return null;
  return { ph, temperature_c, dissolved_oxygen_mg_l, ammonia_mg_l };
}

export default function WaterQuality() {
  const [form, setForm] = useState(null);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [source, setSource] = useState(null);

  const load = useCallback(async () => {
    try {
      const data = await fetchWaterQuality();
      setForm({
        ph: data.ph,
        temperature_c: data.temperature_c,
        dissolved_oxygen_mg_l: data.dissolved_oxygen_mg_l,
        ammonia_mg_l: data.ammonia_mg_l,
      });
      setSource(data.source);
      setError(null);
      setDirty(false);
    } catch {
      setError("Backend unreachable");
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  function updateField(key, value) {
    setForm((f) => ({ ...f, [key]: value }));
    setDirty(true);
  }

  async function handleSave() {
    const parsed = parseForm(form);
    if (!parsed) {
      alert("Fill in all four fields with numbers before saving.");
      return;
    }
    setSaving(true);
    try {
      const data = await saveWaterQuality(parsed);
      setSource(data.source);
      setError(null);
      setDirty(false);
    } catch (err) {
      alert("Couldn't save reading: " + err.message);
    } finally {
      setSaving(false);
    }
  }

  // recomputed on every keystroke, live - not just after Save
  const parsed = form && parseForm(form);
  const live = parsed && evaluateWaterQuality(parsed.ph, parsed.temperature_c, parsed.dissolved_oxygen_mg_l, parsed.ammonia_mg_l);
  const isGood = live && live.status === "Good";
  const statusIsBad = error || (live && !isGood);

  return (
    <>
      <PageHeader
        eyebrow="Water Quality"
        title={<>Live pond &amp; tank readings <span className="badge-simulated">editable</span></>}
        subtitle="No physical sensors are wired up yet - enter your own readings below, or leave them as the simulated defaults. The status updates live as you type; Save Reading persists it."
      />
      <main className="page">
        <section className="card">
          <div className="card-header-row">
            <h2>Current Reading{source && <span className="badge-simulated">{source}</span>}</h2>
            <button className="btn tiny" onClick={load}>Refresh</button>
          </div>
          <div className="water-grid">
            {FIELDS.map((f) => {
              const outOfRange = f.okKey && live && !live[f.okKey];
              return (
                <div className={"water-tile" + (outOfRange ? " out-of-range" : "")} key={f.key}>
                  <div className="water-icon">{f.icon}</div>
                  <div className="water-label">{f.label}{f.suffix && ` (${f.suffix})`}</div>
                  <input
                    type="number"
                    className="water-input"
                    value={form ? form[f.key] : ""}
                    step={f.step}
                    min={f.min}
                    max={f.max}
                    onChange={(e) => updateField(f.key, e.target.value)}
                  />
                </div>
              );
            })}
          </div>
          <div className="water-actions">
            <div className={"water-status" + (statusIsBad ? " bad" : "")}>
              <span className="water-status-icon">{error ? "🔌" : isGood ? "✅" : "⚠️"}</span>
              {error || (live ? live.message : "Fill in all four fields to see a status.")}
            </div>
            <button className="btn primary" style={{ width: "auto", marginTop: 0 }} disabled={!dirty || saving || !parsed} onClick={handleSave}>
              {saving ? "Saving…" : "Save Reading"}
            </button>
          </div>
        </section>

        <section className="card" style={{ marginTop: 20 }}>
          <h2>Safe Ranges</h2>
          <table className="spec-table">
            <tbody>
              <tr><td>pH</td><td>6.5 - 8.5</td></tr>
              <tr><td>Dissolved Oxygen</td><td>&gt; 5.0 mg/L</td></tr>
              <tr><td>Ammonia</td><td>&le; 0.05 mg/L</td></tr>
              <tr><td>Temperature</td><td>species-dependent, tracked for trend only</td></tr>
            </tbody>
          </table>
        </section>
      </main>
    </>
  );
}
