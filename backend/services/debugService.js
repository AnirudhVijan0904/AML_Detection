const { DB_USE, pool } = require('../config/dbConfig');

async function getDbHealth(req, res) {
  if (!DB_USE || !pool) return res.status(400).json({ ok: false, message: 'DB not enabled' });
  const conn = await pool.getConnection();
  try {
    const [[row]] = await conn.query('SELECT 1 as ok');
    res.json({ ok: !!row && Number(row.ok) === 1 });
  } catch (e) {
    console.error('db health check error:', e);
    res.status(500).json({ ok: false, error: e.message || String(e) });
  } finally {
    conn.release();
  }
}

async function getDbInfo(req, res) {
  if (!DB_USE || !pool) return res.status(400).json({ ok: false, message: 'DB not enabled' });
  const conn = await pool.getConnection();
  try {
    // database name
    const [[{ db }]] = await conn.query('SELECT DATABASE() as db');
    // counts
    const [[{ total }]] = await conn.query('SELECT COUNT(*) as total FROM transaction');
    // suspicious count try flexible column names
    let suspicious = 0;
    try {
      const [[{ suspicious: s }]] = await conn.query("SELECT COUNT(*) as suspicious FROM transaction WHERE is_laundering = 1 OR is_laundering = '1'");
      suspicious = s;
    } catch (e1) {
      try {
        const [[{ suspicious: s }]] = await conn.query("SELECT COUNT(*) as suspicious FROM transaction WHERE prediction = 1 OR prediction = '1'");
        suspicious = s;
      } catch (e2) {
        suspicious = null;
      }
    }

    // high-risk unique accounts (try account, account_number)
    let highRisk = null;
    try {
      const [[{ highRisk: h }]] = await conn.query('SELECT COUNT(DISTINCT account) as highRisk FROM transaction WHERE kyc_score IS NOT NULL AND kyc_score < 40');
      highRisk = Number(h);
    } catch (e) {
      try {
        const [[{ highRisk: h }]] = await conn.query('SELECT COUNT(DISTINCT account_number) as highRisk FROM transaction WHERE kyc_score IS NOT NULL AND kyc_score < 40');
        highRisk = Number(h);
      } catch (e2) {
        highRisk = null;
      }
    }

    res.json({ db, total: Number(total), suspicious, highRisk });
  } catch (e) {
    console.error('db info error:', e);
    res.status(500).json({ ok: false, error: e.message || String(e) });
  } finally {
    conn.release();
  }
}

module.exports = { getDbHealth, getDbInfo };

// Create stats_summary table and a MySQL event to refresh it periodically
async function setupStatsSummary(req, res) {
  if (!DB_USE || !pool) return res.status(400).json({ ok: false, message: 'DB not enabled' });
  const conn = await pool.getConnection();
  try {
    // create table if not exists
    await conn.query(`
      CREATE TABLE IF NOT EXISTS stats_summary (
        id INT PRIMARY KEY,
        last_updated DATETIME,
        total BIGINT,
        suspicious BIGINT,
        highRisk BIGINT
      ) ENGINE=InnoDB;
    `);

    // try to enable event scheduler (may require SUPER privilege)
    try {
      await conn.query("SET GLOBAL event_scheduler = ON");
    } catch (e) {
      console.warn('Unable to enable event scheduler (may lack privilege):', e.message || e);
    }

    // create or replace the event that refreshes the single-row summary every N seconds
    const seconds = Number(process.env.STATS_REFRESH_SECONDS || 60);
    const insertSql = `INSERT INTO stats_summary (id,last_updated,total,suspicious,highRisk) VALUES (1,NOW(),(SELECT COUNT(*) FROM transaction),(SELECT COUNT(*) FROM transaction WHERE is_laundering = 1 OR is_laundering = '1' OR prediction = 1 OR prediction = '1'),(SELECT COUNT(DISTINCT account) FROM transaction WHERE kyc_score IS NOT NULL AND kyc_score < 40)) ON DUPLICATE KEY UPDATE last_updated=VALUES(last_updated), total=VALUES(total), suspicious=VALUES(suspicious), highRisk=VALUES(highRisk)`;

    try {
      await conn.query(`CREATE EVENT IF NOT EXISTS refresh_stats_summary ON SCHEDULE EVERY ${seconds} SECOND DO ${insertSql}`);
    } catch (e) {
      // some MySQL variants or permission sets may fail
      console.warn('Failed to create event refresh_stats_summary:', e.message || e);
    }

    res.json({ ok: true, message: 'stats_summary table created; event creation attempted' });
  } catch (e) {
    console.error('setupStatsSummary error:', e);
    res.status(500).json({ ok: false, error: e.message || String(e) });
  } finally {
    conn.release();
  }
}

module.exports = { getDbHealth, getDbInfo, setupStatsSummary };
