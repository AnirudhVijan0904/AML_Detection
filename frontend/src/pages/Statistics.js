import React, { useEffect, useState } from "react";
import "./Statistics.css";
import { api } from "../api/api";
import {
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const Statistics = () => {
  // === State ===
  const [summary, setSummary] = useState({
    total: 0,
    suspicious: 0,
    highRisk: 0,
  });

  // Dummy chart data (backend will provide later if needed)
  const txnTrendData = [
    { date: "Mon", total: 400, suspicious: 12 },
    { date: "Tue", total: 520, suspicious: 18 },
    { date: "Wed", total: 470, suspicious: 14 },
    { date: "Thu", total: 610, suspicious: 21 },
    { date: "Fri", total: 580, suspicious: 16 },
    { date: "Sat", total: 300, suspicious: 7 },
    { date: "Sun", total: 420, suspicious: 9 },
  ];

  const COLORS = ["#4caf50", "#f44336"];

  // === Fetch stats from backend ===
  const loadStats = async () => {
    try {
      // Use relative path so axios baseURL '/api' applies => '/api/stats/summary'
      const res = await api.get("stats/summary");
      setSummary(res.data);
    } catch (err) {
      console.error("Error loading statistics:", err);
    }
  };

  useEffect(() => {
    loadStats();
  }, []);

  // Convert backend values to pie data
  const pieData = [
    { name: "Clean Transactions", value: summary.total - summary.suspicious },
    { name: "Suspicious Transactions", value: summary.suspicious },
  ];

  // Risk bar chart (dummy for now)
  const riskData = [
    { range: "0–20", count: 800 },
    { range: "21–40", count: 650 },
    { range: "41–60", count: 350 },
    { range: "61–80", count: 120 },
    { range: "81–100", count: 40 },
  ];

  return (
    <div className="statistics-container">
      <h1 className="page-title">📈 AML System Statistics</h1>

      {/* === KPI SUMMARY CARDS === */}
      <div className="stats-cards">
        <div className="card">
          <h3>Total Transactions</h3>
          <p>{summary.total}</p>
        </div>
        <div className="card">
          <h3>Suspicious Transactions</h3>
          <p>{summary.suspicious}</p>
        </div>
        <div className="card">
          <h3>High-Risk Customers</h3>
          <p>{summary.highRisk}</p>
        </div>
      </div>

      {/* === LINE CHART === */}
      <div className="chart-container">
        <h2>📊 Daily Transaction Trend</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={txnTrendData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="total"
              stroke="#2196f3"
              name="Total Transactions"
            />
            <Line
              type="monotone"
              dataKey="suspicious"
              stroke="#f44336"
              name="Suspicious"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* === PIE CHART === */}
      <div className="chart-container">
        <h2>🧩 Suspicious vs Clean Distribution</h2>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={pieData}
              dataKey="value"
              cx="50%"
              cy="50%"
              outerRadius={100}
              label
            >
              {pieData.map((entry, index) => (
                <Cell key={index} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* === BAR CHART === */}
      <div className="chart-container">
        <h2>⚠️ Risk Score Distribution</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={riskData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="range" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#ff9800" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default Statistics;
