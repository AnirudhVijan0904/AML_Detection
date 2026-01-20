import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import ManualAnalysis from "./pages/ManualAnalysis";
import RealTime from "./pages/RealTime";
import Statistics from "./pages/Statistics";
import "./App.css";

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/manual" element={<ManualAnalysis />} />
        <Route path="/realtime" element={<RealTime />} />
        <Route path="/statistics" element={<Statistics />} />
      </Routes>
    </Router>
  );
}

export default App;
