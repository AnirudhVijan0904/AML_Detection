const express = require('express');
const { getDbHealth, getDbInfo, setupStatsSummary } = require('../services/debugService');

const router = express.Router();

router.get('/db-health', getDbHealth);
router.get('/db-info', getDbInfo);
router.get('/db-sample', async (req, res) => {
	const { DB_USE, pool } = require('../config/dbConfig');
	if (!DB_USE || !pool) return res.status(400).json({ ok: false, message: 'DB not enabled' });
	const conn = await pool.getConnection();
	try {
		const [rows] = await conn.query('SELECT * FROM transaction LIMIT 5');
		res.json({ sample: rows });
	} catch (e) {
		console.error('db sample error:', e);
		res.status(500).json({ ok: false, error: e.message || String(e) });
	} finally {
		conn.release();
	}
});

// setup endpoint to initialize stats summary and event
router.post('/setup-stats-summary', setupStatsSummary);

module.exports = router;
