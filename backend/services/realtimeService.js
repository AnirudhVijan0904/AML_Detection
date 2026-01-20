const fs = require("fs");
const csv = require("csv-parser");
const path = require("path");
require("dotenv").config();

const CSV_PATH = process.env.CSV_PATH || path.join(process.cwd(), "data", "transactions.csv");
const { DB_USE, pool } = require('../config/dbConfig');

function parseRow(raw) {
    const row = { ...raw };
    // normalize prediction
    try {
        if (raw.prediction !== undefined) row.prediction = String(raw.prediction).trim();
        else if (raw.pred !== undefined) row.prediction = String(raw.pred).trim();
        else row.prediction = null;
    } catch (e) {
        row.prediction = null;
    }

    // normalize confidence
    try {
        if (raw.confidence !== undefined) row.confidence = parseFloat(raw.confidence);
        else if (raw.conf !== undefined) row.confidence = parseFloat(raw.conf);
        else row.confidence = null;
    } catch (e) {
        row.confidence = null;
    }

    // parse key_factors if stored as JSON string
    try {
        row.key_factors = raw.key_factors ? JSON.parse(raw.key_factors) : [];
    } catch (e) {
        row.key_factors = [];
    }

    // parse timestamp
    try {
        if (raw.saved_at) row.saved_at = new Date(raw.saved_at);
        else if (raw.transaction_date) row.saved_at = new Date(raw.transaction_date);
        else row.saved_at = null;
    } catch (e) {
        row.saved_at = null;
    }

    return row;
}

function computeStatus(row) {
    // simple rule set; adjust thresholds as needed
    if (row.prediction === "1" || row.prediction === 1) return "Suspicious";
    if (row.confidence !== null && !isNaN(row.confidence) && row.confidence >= 0.85) return "Under Review";
    return "Clean";
}

// Stream the CSV and keep only the most recent `limit` rows in memory.
async function getLatestTransactions(limit = 20) {
    if (DB_USE && pool) {
        // Try DB first; on any connection/query error, fall back to CSV
        let conn;
        try {
            conn = await pool.getConnection();
            // Map to common output names where possible
            // First try ordering by lowercase 'timestamp' then fallback to 'Timestamp'
            let rows;
            try {
                const q = `SELECT * FROM transaction ORDER BY timestamp DESC LIMIT ?`;
                [rows] = await conn.query(q, [limit]);
            } catch (e) {
                // fallback to capitalized
                const q = `SELECT * FROM transaction ORDER BY \`Timestamp\` DESC LIMIT ?`;
                [rows] = await conn.query(q, [limit]);
            }

            const parsed = rows.map((r) => {
                // normalize fields across possible column names
                const saved_at = r.timestamp || r.Timestamp || r.saved_at || r.transaction_date || null;
                const prediction = r.is_laundering !== undefined ? r.is_laundering : (r.prediction !== undefined ? r.prediction : null);
                const confidence = r.confidence !== undefined ? r.confidence : (r.conf !== undefined ? r.conf : null);
                let key_factors = [];
                try {
                    key_factors = r.key_factors ? JSON.parse(r.key_factors) : (r.keyFactors ? JSON.parse(r.keyFactors) : []);
                } catch (e) {
                    key_factors = [];
                }
                const normal = {
                    ...r,
                    saved_at: saved_at ? new Date(saved_at) : null,
                    prediction: String(prediction).trim(),
                    confidence: confidence !== null ? parseFloat(confidence) : null,
                    key_factors,
                };
                return { ...normal, status: computeStatus(normal) };
            });
            return parsed;
        } catch (e) {
            console.error('DB getLatestTransactions error, falling back to CSV:', e);
            if (conn) {
                try { conn.release(); } catch (_) {}
            }
            // fall through to CSV path below
        }
    }

    return new Promise((resolve, reject) => {
        if (!fs.existsSync(CSV_PATH)) return resolve([]);

        const buffer = [];

        fs.createReadStream(CSV_PATH)
            .pipe(csv())
            .on('data', (raw) => {
                try {
                    const parsed = parseRow(raw);
                    buffer.push(parsed);
                    // keep the buffer capped at `limit` to avoid OOM
                    if (buffer.length > limit) buffer.shift();
                } catch (e) {
                    // ignore malformed row
                }
            })
            .on('end', () => {
                // buffer currently contains oldest->newest for the last `limit` rows
                // reverse to return newest first
                const latest = buffer
                    .slice()
                    .reverse()
                    .map((r) => ({ ...r, status: computeStatus(r) }));
                resolve(latest);
            })
            .on('error', (err) => {
                console.error('CSV stream error in getLatestTransactions:', err);
                resolve([]);
            });
    });
}

// Express handler used by routes/realtimeRoutes.js
async function fetchRealtimeTransactions(req, res) {
    try {
        const limit = parseInt(req.query.limit, 10) || 20;
        const txns = await getLatestTransactions(limit);
        res.json(txns);
    } catch (err) {
        console.error("fetchRealtimeTransactions error:", err);
        res.status(500).json({ error: err.message || "Failed to load transactions" });
    }
}

module.exports = { fetchRealtimeTransactions, getLatestTransactions };
