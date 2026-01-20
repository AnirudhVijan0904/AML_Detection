import React, { useState, useEffect } from "react";
import "./RealTime.css";
import { api } from "../api/api";  // Axios API helper

const RealTime = () => {
  const [isStreaming, setIsStreaming] = useState(true);

  const [transactions, setTransactions] = useState([]);
  const [stats, setStats] = useState({
    total: 0,
    suspicious: 0,
    highRisk: 0,
  });

  // Fetch real-time transactions
  const loadTransactions = async () => {
    try {
      // Use relative path so axios baseURL '/api' applies => '/api/realtime/latest'
      const res = await api.get("realtime/latest");
      setTransactions(res.data);
    } catch (err) {
      console.error("Error loading realtime transactions:", err);
    }
  };

  // Fetch statistics
  const loadStats = async () => {
    try {
      // Use relative path so axios baseURL '/api' applies => '/api/stats/summary'
      const res = await api.get("stats/summary");
      setStats(res.data);
    } catch (err) {
      console.error("Error loading stats:", err);
    }
  };

  // Auto refresh: transactions every 5s when streaming; stats every 30s
  useEffect(() => {
    // initial fetch
    loadTransactions();
    loadStats();

    let txnInterval = null;
    let statsInterval = null;

    if (isStreaming) {
      txnInterval = setInterval(() => {
        loadTransactions();
      }, 120000); // 2 minutes
    }

    // refresh stats less frequently (independent of streaming)
    statsInterval = setInterval(() => {
      loadStats();
    }, 30000);

    return () => {
      if (txnInterval) clearInterval(txnInterval);
      if (statsInterval) clearInterval(statsInterval);
    };
  }, [isStreaming]);

  const toggleStream = () => setIsStreaming(!isStreaming);

  return (
    <div className="realtime-container">
      <h1 className="page-title">Real-Time Transaction Monitoring</h1>

      {/* === CONTROLS === */}
      <div className="controls">
        <button className="stream-btn" onClick={toggleStream}>
          {isStreaming ? "⏸ Pause Stream" : "▶️ Start Stream"}
        </button>
        <input
          type="text"
          placeholder="🔍 Filter by Bank or Account"
          className="filter-input"
        />
      </div>

      {/* === SUMMARY CARDS === */}
      <div className="summary-cards">
        <div className="card">
          <h3>Transactions Today</h3>
          <p>{stats.total}</p>
        </div>
        <div className="card">
          <h3>Suspicious</h3>
          <p>{stats.suspicious}</p>
        </div>
        <div className="card">
          <h3>High-Risk Customers</h3>
          <p>{stats.highRisk}</p>
        </div>
      </div>

      {/* === TRANSACTION TABLE === */}
      <table className="transaction-table">
        <thead>
          <tr>
            {transactions.length > 0 &&
              Object.keys(transactions[0]).map((col) => (
                <th key={col}>{col}</th>
              ))}
          </tr>
        </thead>

        <tbody>
          {transactions.map((txn, index) => (
            <tr
              key={index}
              className={
                txn.status === "Suspicious"
                  ? "suspicious"
                  : txn.status === "Under Review"
                  ? "under-review"
                  : "clean"
              }
            >
              {Object.values(txn).map((value, idx) => (
                <td key={idx}>{value}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      {/* === ALERT PANEL === */}
      <div className="alert-panel">
        <h3>🚨 Live Alerts</h3>
        <ul>
          <li>System will generate alerts based on backend data.</li>
        </ul>
      </div>
    </div>
  );
};

export default RealTime;
