import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import "./Navbar.css";
import { startHealthPolling } from "../api/health";

function Navbar() {
  const [backendOk, setBackendOk] = useState(true);

  const startedRef = useRef(false);
  useEffect(() => {
    // Prevent double-start in React 18 Strict Mode (dev) which mounts effects twice
    if (startedRef.current) return;
    startedRef.current = true;
    const stop = startHealthPolling(setBackendOk, 20000);
    return () => stop && stop();
  }, []);

  return (
    <nav className="navbar">
      <div className="navbar-logo">AML Helpdesk</div>
      <div className="navbar-links">
        <Link to="/">Home</Link>
        <Link to="/realtime">Real-Time</Link>
        <Link to="/manual">Manual Analysis</Link>
        <Link to="/statistics">Statistics</Link>
      </div>
      <div className={`backend-status ${backendOk ? "ok" : "down"}`}>
        {backendOk ? "Backend: Online" : "Backend: Offline"}
      </div>
    </nav>
  );
}

export default Navbar;
