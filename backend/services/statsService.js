require('dotenv').config();
const { DB_USE, pool } = require('../config/dbConfig');

// Simple in-memory cache with TTL to avoid repeated DB hits
let memoryCache = {
  value: null,
  lastUpdated: 0,
};
const CACHE_TTL_MS = 60 * 1000; // 60 seconds

async function calculateTransactionStats(req, res) {
  try {
    if (!DB_USE || !pool) {
      console.error('DB mode not enabled but stats endpoint is DB-only');
      return res.status(400).json({ error: 'DB mode disabled; stats require a database' });
    }

    let conn;
    try {
      conn = await pool.getConnection();
      // Ensure cache table exists (best-effort, ignore failures)
      try {
        await conn.query(
          'CREATE TABLE IF NOT EXISTS stats_summary (\n' +
          '  id INT PRIMARY KEY,\n' +
          '  last_updated DATETIME,\n' +
          '  total BIGINT,\n' +
          '  suspicious BIGINT,\n' +
          '  highRisk BIGINT\n' +
          ')'
        );
      } catch (e) {
        // ignore
      }

      // Total transactions
      const [[{ total }]] = await conn.query('SELECT COUNT(*) as total FROM transaction');

      // Suspicious: flexible columns
      let suspicious = null;
      const suspiciousQueries = [
        "SELECT COUNT(*) as suspicious FROM transaction WHERE is_laundering IN (1,'1')",
        "SELECT COUNT(*) as suspicious FROM transaction WHERE `Is Laundering` IN (1,'1')",
        "SELECT COUNT(*) as suspicious FROM transaction WHERE prediction IN (1,'1')",
      ];
      for (const q of suspiciousQueries) {
        try {
          const [[{ suspicious: s }]] = await conn.query(q);
          suspicious = Number(s);
          break;
        } catch (e) {
          // try next
        }
      }

      // High-risk: flexible account column
      let highRisk = null;
      try {
        const [[{ highRisk: h }]] = await conn.query('SELECT COUNT(DISTINCT account) as highRisk FROM transaction WHERE kyc_score IS NOT NULL AND kyc_score < 40');
        highRisk = Number(h);
      } catch (e) {
        try {
          const [[{ highRisk: h }]] = await conn.query('SELECT COUNT(DISTINCT account_number) as highRisk FROM transaction WHERE kyc_score IS NOT NULL AND kyc_score < 40');
          highRisk = Number(h);
        } catch (e2) {
          highRisk = 0;
        }
      }

      // Update in-memory cache
      memoryCache.value = { total: Number(total) || 0, suspicious: suspicious || 0, highRisk: highRisk || 0, cached: true };
      memoryCache.lastUpdated = Date.now();

      // Best-effort DB cache update (non-blocking for response) with correct UPSERT syntax
      try {
        const t = Number(total) || 0;
        const s = suspicious || 0;
        const h = highRisk || 0;
        await conn.query(
          'INSERT INTO stats_summary (id, last_updated, total, suspicious, highRisk) VALUES (1, NOW(), ?, ?, ?) ' +
          'ON DUPLICATE KEY UPDATE last_updated = NOW(), total = ?, suspicious = ?, highRisk = ?',
          [t, s, h, t, s, h]
        );
      } catch (e) {
        // ignore cache write errors
      }

      console.log('DB stats result:', { total: Number(total), suspicious, highRisk });
      res.json({ total: Number(total) || 0, suspicious: suspicious || 0, highRisk: highRisk || 0 });
    } catch (e) {
      console.error('DB calculateTransactionStats error:', e);
      // If memory cache exists, serve it; otherwise return quick fallback
      if (memoryCache.value) {
        return res.json(memoryCache.value);
      }
      return res.json({ total: 0, suspicious: 0, highRisk: 0, cached: false, source: 'fallback' });
    } finally {
      if (conn) {
        try { conn.release(); } catch (_) {}
      }
    }
  } catch (err) {
    console.error('calculateTransactionStats error:', err);
    res.status(500).json({ error: err.message || 'Failed to calculate stats' });
  }
}

async function getCachedTransactionStats(req, res) {
  try {
    if (!DB_USE || !pool) {
      console.error('DB mode not enabled but stats endpoint is DB-only');
      return res.status(400).json({ error: 'DB mode disabled; stats require a database' });
    }

    // Serve memory cache if fresh
    if (memoryCache.value && (Date.now() - memoryCache.lastUpdated) < CACHE_TTL_MS) {
      return res.json(memoryCache.value);
    }

    let conn;
    try {
      conn = await pool.getConnection();
      const [rows] = await conn.query('SELECT last_updated, total, suspicious, highRisk FROM stats_summary WHERE id = 1 LIMIT 1');
      if (rows && rows.length > 0) {
        const r = rows[0];
        const payload = { total: Number(r.total) || 0, suspicious: Number(r.suspicious) || 0, highRisk: Number(r.highRisk) || 0, last_updated: r.last_updated, cached: true };
        // update memory cache for next request
        memoryCache.value = payload;
        memoryCache.lastUpdated = Date.now();
        return res.json(payload);
      }
    } catch (e) {
      console.warn('cached stats read failed, falling back to live:', e.message || e);
    } finally {
      if (conn) {
        try { conn.release(); } catch (_) {}
      }
    }

    // fallback to live calculation
    return calculateTransactionStats(req, res);
  } catch (err) {
    console.error('getCachedTransactionStats overall error:', err);
    // Serve memory cache or quick fallback instead of 500
    if (memoryCache.value) {
      return res.json(memoryCache.value);
    }
    return res.json({ total: 0, suspicious: 0, highRisk: 0, cached: false, source: 'fallback' });
  }
}

module.exports = { calculateTransactionStats, getCachedTransactionStats };

