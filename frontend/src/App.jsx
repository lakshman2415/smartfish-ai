import { Routes, Route } from "react-router-dom";
import Nav from "./components/Nav.jsx";
import Home from "./pages/Home.jsx";
import Detection from "./pages/Detection.jsx";
import WaterQuality from "./pages/WaterQuality.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Alerts from "./pages/Alerts.jsx";

export default function App() {
  return (
    <>
      <Nav />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/detection" element={<Detection />} />
        <Route path="/water-quality" element={<WaterQuality />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/alerts" element={<Alerts />} />
      </Routes>
      <footer className="app-footer">Smarter Detection. Early Action. Healthier Fish.</footer>
    </>
  );
}
