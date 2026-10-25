import { Link } from "react-router-dom";
import fishImage from "../assets/fish.jpg";
import useTypewriter from "../hooks/useTypewriter.js";

const HERO_TITLES = [
  "Automated Fish Disease Diagnosis Using Hybrid AI",
  "AI-Powered Freshwater Fish Health Monitoring",
  "Smart Fish Disease Detection & Diagnosis",
  "Instant Fish Health Analysis Using Deep Learning",
];

const FEATURES = [
  { icon: "🐟", title: "Fish Disease Detection", text: "Upload or capture a photo and get a prediction across 9 trained disease categories, backed by real Kaggle datasets." },
  { icon: "📊", title: "Confidence & Severity", text: "Every prediction ships with a confidence score and a severity level - not just a bare label." },
  { icon: "🟥", title: "Fish Detection Box", text: "A bounding box highlights the located fish in your photo before classification runs." },
  { icon: "💧", title: "Water Quality Monitoring", text: "Track pH, temperature, dissolved oxygen and ammonia on a live panel (simulated until real probes are wired in)." },
  { icon: "💊", title: "Treatment Recommendations", text: "Actionable treatment and prevention guidance for every diagnosed condition." },
  { icon: "🕘", title: "Diagnosis History", text: "Every diagnosis is logged so you can track patterns and revisit past results." },
];

const BENEFITS = [
  "Cuts diagnostic time from minutes of guesswork to seconds",
  "Trained on real, published freshwater fish disease datasets",
  "Open, inspectable pipeline - no black box",
  "Rejects images that don't look like a fish, instead of guessing",
  "Upfront about current limits - see Coverage on the Dashboard",
  "Runs fully locally - your images stay on your machine",
];

export default function Home() {
  const typedTitle = useTypewriter(HERO_TITLES);

  return (
    <div
      className="home-page"
      style={{ backgroundImage: `url(${fishImage})` }}
    >
      <div className="home-overlay">
        <section className="hero">
          <div className="hero-inner hero-inner-centered">
            <div className="hero-copy">
              <div className="hero-badge">Powered by EfficientNet-B4 + Vision Transformer</div>
              <h1 className="hero-title" aria-label={HERO_TITLES[0]}>
                <span aria-hidden="true">{typedTitle}<span className="typewriter-cursor">|</span></span>
              </h1>
              <p className="hero-sub">
                Upload or capture a fish photo and get an instant disease prediction, confidence
                score, and treatment recommendation - built for freshwater aquaculture, with an
                honest account of what's trained on real data and what isn't yet.
              </p>
              <div className="hero-ctas">
                <Link to="/detection" className="btn primary">Start Detection</Link>
                <Link to="/dashboard" className="btn ghost">View Dashboard</Link>
              </div>
            </div>
          </div>
        </section>

        <section className="section">
          <div className="section-eyebrow">Features</div>
          <div className="section-title">Everything you need for a fish health check</div>
          <p className="section-sub">From image-based disease detection to water quality tracking - built on a real, inspectable pipeline.</p>
          <div className="feature-grid">
            {FEATURES.map((f) => (
              <div className="feature-card" key={f.title}>
                <div className="feature-icon">{f.icon}</div>
                <h3>{f.title}</h3>
                <p>{f.text}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="section">
          <div className="section-eyebrow">Benefits</div>
          <div className="section-title">Built to be useful, not just impressive</div>
          <p className="section-sub">A practical system designed around what real public data can actually support.</p>
          <div className="benefit-list">
            {BENEFITS.map((b) => (
              <div className="benefit-item" key={b}>
                <span className="benefit-check">✓</span>{b}
              </div>
            ))}
          </div>
        </section>

        <div className="info-band">
          <h2>How it actually works</h2>
          <p>Three steps, no black box in between.</p>
          <div className="info-steps">
            <div className="info-step">
              <div className="info-step-icon">📸</div>
              <h3>1. Upload or capture</h3>
              <p>Choose a photo from your device or use your camera to take one on the spot.</p>
            </div>
            <div className="info-step">
              <div className="info-step-icon">🔍</div>
              <h3>2. Fish located &amp; verified</h3>
              <p>A quick check locates the fish and confirms it's actually a fish before anything else runs.</p>
            </div>
            <div className="info-step">
              <div className="info-step-icon">🩺</div>
              <h3>3. Diagnosis &amp; guidance</h3>
              <p>Disease, confidence score, severity, and treatment/prevention advice - all in a few seconds.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
