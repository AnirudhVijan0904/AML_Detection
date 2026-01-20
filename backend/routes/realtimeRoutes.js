// import express from "express";
// import { getLatestTransactions } from "../services/realtimeService.js";

// const router = express.Router();

// router.get("/latest", async (req, res) => {
//   try {
//     const txns = await getLatestTransactions();
//     res.json(txns);
//   } catch (error) {
//     res.status(500).json({ error: error.message });
//   }
// });

// export default router;
const express = require('express');
const { fetchRealtimeTransactions } = require('../services/realtimeService');

const realtimeRouter = express.Router();

// Keep existing route
realtimeRouter.get('/transactions', fetchRealtimeTransactions);

// Alias used by frontend: /realtime/latest
realtimeRouter.get('/latest', fetchRealtimeTransactions);

module.exports = realtimeRouter;