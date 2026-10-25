import { useEffect, useRef, useState } from "react";
import { detectFish, predictDisease, prettyClassName, severityClass, severityIcon } from "../api.js";
import PageHeader from "../components/PageHeader.jsx";

export default function Detection() {
  const [activeTab, setActiveTab] = useState("upload");
  const [dragOver, setDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [blob, setBlob] = useState(null);

  const [fishRegion, setFishRegion] = useState(null);
  const [closeup, setCloseup] = useState(false);
  const [status, setStatus] = useState({ text: "", level: null });
  const [canDiagnose, setCanDiagnose] = useState(false);
  const [diagnosing, setDiagnosing] = useState(false);
  const [result, setResult] = useState(null);
  const [showAllProbs, setShowAllProbs] = useState(false);
  const [emptyText, setEmptyText] = useState('Upload or capture a fish image to begin.');
  const [cameraActive, setCameraActive] = useState(false);

  const fileInputRef = useRef(null);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => () => stopCamera(), []);

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  }

  function switchTab(tab) {
    setActiveTab(tab);
    if (tab !== "camera") stopCamera();
  }

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraActive(true);
    } catch (err) {
      alert("Couldn't access the camera: " + err.message);
    }
  }

  function captureFrame() {
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((b) => selectImage(b), "image/jpeg", 0.92);
  }

  function selectImage(newBlob) {
    setBlob(newBlob);
    setPreviewUrl(URL.createObjectURL(newBlob));
    setFishRegion(null);
    setResult(null);
    setCanDiagnose(false);
    setEmptyText("Locating the fish…");
    setStatus({ text: "Locating the fish in your photo…", level: null });
  }

  async function runDetection(closeupFlag) {
    if (!blob) return;
    try {
      const data = await detectFish(blob, closeupFlag);
      setFishRegion(data.region);
      if (data.is_fish) {
        const text = closeupFlag
          ? "✓ Treating this as a close-up of the affected area. Ready to check for disease."
          : `✓ Fish detected (${Math.round(data.fish_confidence * 100)}% fish-likeness). Ready to check for disease.`;
        setStatus({ text, level: "ok" });
        setCanDiagnose(true);
        setEmptyText('Fish detected. Click "Yes, Detect Disease" to diagnose.');
      } else {
        setStatus({
          text: `✗ This doesn't look like a fish (${Math.round(data.fish_confidence * 100)}% fish-likeness). If this photo is a close-up of just the affected area rather than the whole fish, tick "This is a close-up" below and we'll try again.`,
          level: "bad",
        });
        setCanDiagnose(false);
        setEmptyText("No fish detected in this image - nothing to diagnose.");
      }
    } catch (err) {
      setStatus({ text: "Couldn't reach the backend to detect a fish: " + err.message, level: "bad" });
    }
  }

  function onImageLoaded() {
    runDetection(closeup);
  }

  function onCloseupToggle(e) {
    const checked = e.target.checked;
    setCloseup(checked);
    if (blob) runDetection(checked);
  }

  async function onDetectDisease() {
    if (!blob) return;
    setDiagnosing(true);
    setEmptyText("Diagnosing…");
    try {
      const data = await predictDisease(blob, closeup);
      setResult(data);
      setShowAllProbs(false);
    } catch (err) {
      alert("Diagnosis failed: " + err.message);
    } finally {
      setDiagnosing(false);
    }
  }

  // redraw the overlay whenever the boxes change or the window resizes
  useEffect(() => {
    drawOverlay();
    window.addEventListener("resize", drawOverlay);
    return () => window.removeEventListener("resize", drawOverlay);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fishRegion, previewUrl]);

  function drawOverlay() {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas || !img.clientWidth || !img.naturalWidth) return;

    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = img.clientWidth / img.naturalWidth;
    const scaleY = img.clientHeight / img.naturalHeight;

    const drawBox = (region, color, label) => {
      if (!region) return;
      const x = region.x * scaleX, y = region.y * scaleY, w = region.w * scaleX, h = region.h * scaleY;
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);
      ctx.font = "600 11px system-ui, sans-serif";
      const textWidth = ctx.measureText(label).width + 8;
      ctx.fillStyle = color;
      ctx.fillRect(x, Math.max(0, y - 16), textWidth, 16);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + 4, Math.max(12, y - 4));
    };

    drawBox(fishRegion, "#0f9b9b", "Fish");
  }

  return (
    <>
      <PageHeader
        eyebrow="AI Disease Detection"
        title="Upload a fish image, get an instant diagnosis"
        subtitle="Step 1: we locate the fish in your photo. Step 2, on your go-ahead: we diagnose it and give you a treatment plan."
      />
      <main className="page">
      <div className="detection-stack">
        <section className="card capture-card">
          <h2>1. Get a fish image</h2>

          <div className="capture-tabs">
            <button className={"tab-btn" + (activeTab === "upload" ? " active" : "")} onClick={() => switchTab("upload")}>Upload Image</button>
            <button className={"tab-btn" + (activeTab === "camera" ? " active" : "")} onClick={() => switchTab("camera")}>Use Camera</button>
          </div>

          {activeTab === "upload" && (
            <label
              className={"dropzone" + (dragOver ? " drag-over" : "")}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                if (e.dataTransfer.files.length) selectImage(e.dataTransfer.files[0]);
              }}
            >
              <span>Drag &amp; drop a fish photo here, or click to browse</span>
              <span className="dropzone-hint">Fish photos only - JPG, PNG. Objects, humans &amp; non-fish images are rejected.</span>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => { if (e.target.files.length) selectImage(e.target.files[0]); }}
              />
            </label>
          )}

          {activeTab === "camera" && (
            <div>
              <video ref={videoRef} autoPlay playsInline />
              <div className="camera-controls">
                <button className="btn secondary" onClick={startCamera}>Start Camera</button>
                <button className="btn secondary" disabled={!cameraActive} onClick={captureFrame}>Capture Frame</button>
              </div>
            </div>
          )}

          <label className="closeup-toggle">
            <input type="checkbox" checked={closeup} onChange={onCloseupToggle} />
            This is a close-up of the affected area (not the whole fish)
          </label>

          {previewUrl && (
            <div className="preview-wrap">
              <div className="preview-box">
                <img ref={imgRef} src={previewUrl} alt="Selected fish" onLoad={onImageLoaded} />
                <canvas ref={canvasRef} id="overlayCanvas" />
              </div>
              {status.text && <div className={"detect-status" + (status.level ? " " + status.level : "")}>{status.text}</div>}
              {canDiagnose && (
                <button className="btn primary" disabled={diagnosing} onClick={onDetectDisease}>
                  {diagnosing ? "Diagnosing…" : "Yes, Detect Disease"}
                </button>
              )}
            </div>
          )}

          <canvas ref={captureCanvasRef} hidden />
        </section>

        <section className={"card result-card" + (result ? " has-result " + severityClass(result.severity) : "")}>
          <h2>2. Diagnosis</h2>
          {!result && (
            <div className="empty-state">
              <div className="empty-state-icon">🔬</div>
              <p>{emptyText}</p>
            </div>
          )}

          {result && (() => {
            const sortedProbs = Object.entries(result.class_probabilities).sort((a, b) => b[1] - a[1]);
            const TOP_N = 4;
            const visibleProbs = showAllProbs ? sortedProbs : sortedProbs.slice(0, TOP_N);
            const hiddenCount = sortedProbs.length - TOP_N;

            return (
              <div className="result-content">
                <div className="result-banner">
                  <div className="result-icon">{severityIcon(result.predicted_class, result.severity)}</div>
                  <div className="result-banner-text">
                    <div className="result-label">Predicted Condition</div>
                    <div className="result-disease">{prettyClassName(result.predicted_class)}</div>
                  </div>
                  <div className={"severity-badge " + severityClass(result.severity)}>{result.severity}</div>
                </div>

                <div className="confidence-block">
                  <div className="confidence-top">
                    <span>Model Confidence</span>
                    <span className="confidence-big">{Math.round(result.confidence * 100)}%</span>
                  </div>
                  <div className="confidence-bar-track">
                    <div className="confidence-bar-fill" style={{ width: `${Math.round(result.confidence * 100)}%` }} />
                  </div>
                </div>

                <div className="probs-section">
                  <div className="probs-heading">Full breakdown</div>
                  <div className="probs-list">
                    {visibleProbs.map(([name, prob], i) => (
                      <div className={"prob-row" + (i === 0 ? " top-pick" : "")} key={name}>
                        <div className="name">{prettyClassName(name)}</div>
                        <div className="track"><div className="fill" style={{ width: `${Math.round(prob * 100)}%` }} /></div>
                        <div className="pct">{Math.round(prob * 100)}%</div>
                      </div>
                    ))}
                  </div>
                  {hiddenCount > 0 && (
                    <button className="probs-toggle" onClick={() => setShowAllProbs((v) => !v)}>
                      {showAllProbs ? "Show fewer ▲" : `Show all ${sortedProbs.length} conditions ▾`}
                    </button>
                  )}
                </div>

                <div className="recommendation-grid">
                  <div className="recommendation-box treatment-box">
                    <div className="recommendation-icon">💊</div>
                    <h3>Treatment</h3>
                    <p>{result.treatment}</p>
                  </div>
                  <div className="recommendation-box prevention-box">
                    <div className="recommendation-icon">🛡️</div>
                    <h3>Prevention</h3>
                    <p>{result.prevention}</p>
                  </div>
                </div>
              </div>
            );
          })()}
        </section>
      </div>
      </main>
    </>
  );
}
