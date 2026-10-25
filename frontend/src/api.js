// Use whatever host the page itself was loaded from (localhost, or this
// machine's LAN IP when opened from another device) instead of a hardcoded
// "localhost", which would otherwise point at the OTHER device's own 8000.
export const API_BASE = `http://${window.location.hostname}:8000`;

export async function fetchHistory(limit = 200) {
  const res = await fetch(`${API_BASE}/history?limit=${limit}`);
  if (!res.ok) throw new Error(`history fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchWaterQuality() {
  const res = await fetch(`${API_BASE}/water-quality`);
  if (!res.ok) throw new Error(`water-quality fetch failed: ${res.status}`);
  return res.json();
}

export async function saveWaterQuality(reading) {
  const res = await fetch(`${API_BASE}/water-quality`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(reading),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail?.[0]?.msg || err.detail || `Server returned ${res.status}`);
  }
  return res.json();
}

export async function detectFish(blob, closeup = false) {
  const form = new FormData();
  form.append("file", blob, "capture.jpg");
  form.append("closeup", closeup);
  const res = await fetch(`${API_BASE}/detect-fish`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Server returned ${res.status}`);
  return res.json();
}

export async function predictDisease(blob, closeup = false) {
  const form = new FormData();
  form.append("file", blob, "capture.jpg");
  form.append("closeup", closeup);
  const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server returned ${res.status}`);
  }
  return res.json();
}

export function prettyClassName(name) {
  return name.replaceAll("_", " ");
}

export function timeAgo(isoString) {
  const then = new Date(isoString + "Z").getTime();
  const diffMin = Math.round((Date.now() - then) / 60000);
  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.round(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  return `${Math.round(diffHr / 24)}d ago`;
}

export function severityClass(text) {
  if (text === "None") return "sev-none";
  if (text === "Moderate") return "sev-moderate";
  if (text === "High" || text === "Critical") return "sev-" + text.toLowerCase();
  if (text.startsWith("Uncertain")) return "sev-uncertain";
  return "sev-moderate";
}

export function severityIcon(predictedClass, severity) {
  if (predictedClass === "Healthy") return "✅";
  if (severity === "Critical") return "🚨";
  if (severity === "High") return "⚠️";
  if (severity.startsWith("Uncertain")) return "❔";
  return "🔶";
}

// Mirrors backend/water_quality.py's compute_status()/compute_message() so the
// UI can show live feedback as the user types, before they hit Save. Keep the
// thresholds in sync with that file if they ever change.
const PH_RANGE = [6.5, 8.5];
const MIN_DISSOLVED_OXYGEN = 5.0;
const MAX_AMMONIA = 0.05;

export function evaluateWaterQuality(ph, temperature_c, dissolved_oxygen_mg_l, ammonia_mg_l) {
  const ph_ok = ph >= PH_RANGE[0] && ph <= PH_RANGE[1];
  const dissolved_oxygen_ok = dissolved_oxygen_mg_l >= MIN_DISSOLVED_OXYGEN;
  const ammonia_ok = ammonia_mg_l <= MAX_AMMONIA;
  const allOk = ph_ok && dissolved_oxygen_ok && ammonia_ok;

  let message;
  if (allOk) {
    message = `All readings are within safe ranges - pH (${ph}), dissolved oxygen (${dissolved_oxygen_mg_l} mg/L) and ammonia (${ammonia_mg_l} mg/L) all look healthy.`;
  } else {
    const parts = [];
    if (!ph_ok) parts.push(`pH is ${ph}, outside the safe ${PH_RANGE[0]}-${PH_RANGE[1]} range`);
    if (!dissolved_oxygen_ok) parts.push(`dissolved oxygen is ${dissolved_oxygen_mg_l} mg/L, below the ${MIN_DISSOLVED_OXYGEN} mg/L minimum`);
    if (!ammonia_ok) parts.push(`ammonia is ${ammonia_mg_l} mg/L, above the ${MAX_AMMONIA} mg/L limit`);
    message = "Needs attention: " + parts.join("; ") + ".";
  }

  return { status: allOk ? "Good" : "Needs attention", message, ph_ok, dissolved_oxygen_ok, ammonia_ok };
}
