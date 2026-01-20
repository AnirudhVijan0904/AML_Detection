// import express from "express";
// import { getStats } from "../services/statsService.js";

// const router = express.Router();

// router.get("/summary", async (req, res) => {
//     try {
//         const stats = await getStats();
//         res.json(stats);
//     } catch (err) {
//         res.status(500).json({ error: err.message });
//     }
// });

// export default router;

const express = require('express');
const { calculateTransactionStats, getCachedTransactionStats } = require('../services/statsService');

const statsRouter = express.Router();

// Default summary returns cached stats if available (fast); falls back to live if missing
statsRouter.get('/summary', getCachedTransactionStats);
// Raw live stats (no cache)
statsRouter.get('/summary/raw', calculateTransactionStats);

module.exports = statsRouter;