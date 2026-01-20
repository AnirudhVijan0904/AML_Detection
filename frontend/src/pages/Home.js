import React from "react";
import { Link } from "react-router-dom";
import "./Home.css";

const Home = () => {
  return (
    <div className="home-container">
      {/* ===== HERO SECTION ===== */}
      <header className="hero-section">
        <h1>Welcome to the AML Helpdesk System</h1>
        <p>
          Monitor and analyze banking transactions in real-time or manually to detect
          potential suspicious activities using AI-powered insights.
        </p>
        <Link to="/realtime" className="get-started-btn">
          Get Started
        </Link>
      </header>

      {/* ===== FEATURE CARDS ===== */}
      <section className="features-section">
        <div className="feature-card">
          <h3>⚡ Real-Time Monitoring</h3>
          <p>
            Automatically analyze transactions as they occur, ensuring immediate
            detection of anomalies.
          </p>
          <Link to="/realtime" className="feature-btn">Go to Real-Time</Link>
        </div>

        <div className="feature-card">
          <h3>✍️ Manual Review</h3>
          <p>
            Manually input or upload transaction data for on-demand analysis and
            validation.
          </p>
          <Link to="/manual" className="feature-btn">Go to Manual Analysis</Link>
        </div>

        <div className="feature-card">
          <h3>📊 Statistics Dashboard</h3>
          <p>
            View analytical insights, alerts, and detection accuracy to track system
            performance.
          </p>
          <Link to="/statistics" className="feature-btn">View Statistics</Link>
        </div>
      </section>

      {/* ===== FOOTER ===== */}
      <footer className="footer">
        <p>© 2025 AML Helpdesk | Powered by AI & Data Intelligence</p>
      </footer>
    </div>
  );
};

export default Home;
