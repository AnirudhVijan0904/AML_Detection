// import express from "express";
// import { analyzeManualTransaction } from "../services/manualService.js";

// const router = express.Router();

// router.post("/analyze", async (req, res) => {
//     try {
//         const result = await analyzeManualTransaction(req.body);
//         res.json(result);
//     } catch (err) {
//         res.status(500).json({ error: err.message });
//     }
// });

// export default router;

const express = require('express');
const { analyzeManualTransaction } = require('../services/manualService');

const manualRouter = express.Router();

// Quick echo endpoint to verify requests reach the backend
manualRouter.post('/echo', (req, res) => {
  const ts = new Date().toISOString();
  console.log(`[manualRoutes] ECHO at ${ts}. Body:`, JSON.stringify(req.body, null, 2));
  res.json({ ok: true, received: req.body, ts });
});

// Force-debug variant: /api/manual/predict/debug
manualRouter.post('/predict/debug', async (req, res) => {
  try {
    req.body = { ...(req.body || {}), debug: true };
    const result = await analyzeManualTransaction(req.body);
    return res.json(result);
  } catch (err) {
    return res.status(500).json({ error: err.message || 'Prediction failed' });
  }
});

manualRouter.post('/predict', async (req, res) => {
	console.log('[manualRoutes] Received POST /predict request');
	console.log('[manualRoutes] Request body:', JSON.stringify(req.body, null, 2));
	try {
		// Allow debug logs to be returned when ?debug=true
		if (req.query && typeof req.query.debug !== 'undefined') {
			req.body = { ...req.body, debug: req.query.debug === 'true' };
		}

		// Also allow debug via header X-Debug: 1
		if (req.headers && req.headers['x-debug']) {
			const hd = String(req.headers['x-debug']).toLowerCase();
			if (['1', 'true', 'yes'].includes(hd)) {
				req.body = { ...req.body, debug: true };
			}
		}
		const result = await analyzeManualTransaction(req.body);
		console.log('[manualRoutes] Sending response:', JSON.stringify(result, null, 2));
		res.json(result);
	} catch (err) {
		console.error('[manualRoutes] Error:', err);
		res.status(500).json({ error: err.message || 'Prediction failed' });
	}
});

module.exports = manualRouter;