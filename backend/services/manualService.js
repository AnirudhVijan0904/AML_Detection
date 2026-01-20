const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const FEATURE_MAP = {
	fromBank: 'from_bank',
	fromAccount: 'account',
	toBank: 'to_bank_txn',
	toAccount: 'account_1',
	amountReceived: 'amount_received',
	receivingCurrency: 'Receiving Currency',
	amount: 'amount',
	paymentCurrency: 'Payment Currency',
	paymentFormat: 'Payment Format',

	fullName: 'name',
	nationality: 'nationality',
	occupation: 'occupation',
	kycStatus: 'kyc_status',
	kycScore: 'kyc_score',
	isPep: 'is_pep',
	monthlyIncome: 'monthly_income',
	dob: 'date_of_birth',
	customerSince: 'customer_since',

	// Additional raw features provided directly by user (no DB lookup)
	txnCountLast7Days: 'txn_count_last_7_days',
	totalAmountLast30Days: 'total_amount_last_30_days',
	daysSinceLastTxn: 'days_since_last_txn',

	beneficiaryReceiveCount: 'beneficiary_receive_count',
	beneficiaryTotalReceived: 'beneficiary_total_received',
	beneficiaryAvgReceivedAmount: 'beneficiary_avg_received_amount',
	beneficiaryUniqueSenders: 'beneficiary_unique_senders',
	beneficiaryUniqueSenderNationalitiesSoFar: 'beneficiary_unique_sender_nationalities_so_far',
	beneficiaryPepSenderCountAtTimeOfTxn: 'beneficiary_pep_sender_count_at_time_of_txn',
	beneficiaryUniqueSendersAtTimeOfTxn: 'beneficiary_unique_senders_at_time_of_txn',
	beneficiaryReceiveCountSoFar: 'beneficiary_receive_count_so_far',
	beneficiaryTotalReceivedSoFar: 'beneficiary_total_received_so_far',
};

function mapFeatures(input) {
	const output = {};
	for (const uiKey in input) {
		const modelKey = FEATURE_MAP[uiKey];
		if (!modelKey) continue;
		let value = input[uiKey];
		if (typeof value === 'string') value = value.trim();
		// Extra cleanup for identifiers
		if (modelKey === 'account' || modelKey === 'account_1') {
			value = String(value || '').trim();
		}
		output[modelKey] = value;
	}

	// Coerce common boolean-like inputs to numeric flags (e.g., is_pep)
	if (output.is_pep !== undefined) {
		const v = output.is_pep;
		if (typeof v === 'string') {
			const s = v.trim().toLowerCase();
			if (['yes', 'true', '1', 'y'].includes(s)) output.is_pep = 1;
			else if (['no', 'false', '0', 'n'].includes(s)) output.is_pep = 0;
			else output.is_pep = parseInt(v, 10) || 0;
		} else {
			output.is_pep = Number(v) ? 1 : 0;
		}
	}

	// Coerce numeric fields to numbers where appropriate
	const numericKeys = [
		'amount_received', 'amount', 'monthly_income', 'kyc_score',
		'txn_count_last_7_days', 'total_amount_last_30_days', 'days_since_last_txn',
		'beneficiary_receive_count', 'beneficiary_total_received', 'beneficiary_avg_received_amount',
		'beneficiary_unique_senders', 'beneficiary_unique_sender_nationalities_so_far',
		'beneficiary_pep_sender_count_at_time_of_txn', 'beneficiary_unique_senders_at_time_of_txn',
		'beneficiary_receive_count_so_far', 'beneficiary_total_received_so_far'
	];
	for (const k of numericKeys) {
		if (output[k] !== undefined) {
			const num = Number(output[k]);
			output[k] = isNaN(num) ? 0 : num;
		}
	}
	return output;
}

function computeEngineeredFeatures(row) {
	const now = new Date();
	if (row.date_of_birth) {
		try {
			const dob = new Date(row.date_of_birth);
			let age = now.getFullYear() - dob.getFullYear();
			if (now.getMonth() < dob.getMonth() || (now.getMonth() === dob.getMonth() && now.getDate() < dob.getDate())) age -= 1;
			row.age = age;
		} catch (e) {
			row.age = null;
		}
	}

	if (row.customer_since) {
		try {
			const cs = new Date(row.customer_since);
			const months = (now.getFullYear() - cs.getFullYear()) * 12 + (now.getMonth() - cs.getMonth());
			row.customer_tenure_month = months;
		} catch (e) {
			row.customer_tenure_month = null;
		}
	}

	if (row.amount && row.monthly_income) {
		try {
			const amt = Number(row.amount);
			const income = Number(row.monthly_income);
			row.amount_to_income_ratio = income > 0 ? amt / income : null;
		} catch (e) {
			row.amount_to_income_ratio = null;
		}
	}

	return row;
}

function callPythonModel(features, includeDebug = false) {
	return new Promise((resolve, reject) => {
		console.log(`[ManualService] callPythonModel includeDebug=${includeDebug} LOG_PY_STDERR=${process.env.LOG_PY_STDERR}`);
		// Use path relative to the backend directory to avoid cwd issues
		const pythonScript = path.join(__dirname, '..', 'ml', 'predict.py');
		const pythonCmd = process.env.PYTHON_EXECUTABLE || (process.platform === 'win32' ? 'python' : 'python3');

		const py = spawn(pythonCmd, [pythonScript], {
			stdio: ['pipe', 'pipe', 'pipe'],
			cwd: path.join(__dirname, '..'),
		});

		py.on('error', (err) => reject(new Error(`Failed to spawn Python process (${pythonCmd}): ${err.message}`)));

		let stdout = '';
		let stderr = '';
		const MAX_STDERR = 64 * 1024; // cap stderr buffer to 64KB to avoid memory bloat

		py.stdout.on('data', (chunk) => (stdout += chunk.toString()));
		py.stderr.on('data', (chunk) => {
			const data = chunk.toString();
			if (stderr.length < MAX_STDERR) {
				stderr += data.slice(0, Math.max(0, MAX_STDERR - stderr.length));
			}
			// Always log stderr to see Python debug output (binning, encoding, etc.)
			console.error('[Python stderr]', data);
		});

		py.on('close', (code) => {
			try {
				const lines = stdout.trim().split('\n').map((l) => l.trim()).filter(Boolean);
				let result = null;
				for (let i = lines.length - 1; i >= 0; i--) {
					try {
						result = JSON.parse(lines[i]);
						break;
					} catch (e) {
						continue;
					}
				}
				if (!result) return reject(new Error(`No valid JSON output from Python. stderr: ${stderr}`));
				if (result.error) return reject(new Error(`Python model error: ${result.error}`));
				if (includeDebug) {
					result.debug = stderr;
				}
				return resolve(result);
			} catch (e) {
				return reject(new Error(`Failed to parse Python output: ${e.message}. stderr: ${stderr}`));
			}
		});

		py.stdin.write(JSON.stringify(features));
		py.stdin.end();
	});
}

async function analyzeManualTransaction(input) {
	console.log('[ManualService] Received input:', JSON.stringify(input, null, 2));
	try {
		let row = mapFeatures(input || {});
		console.log('[ManualService] Mapped features:', JSON.stringify(row, null, 2));
		
		row = computeEngineeredFeatures(row);
		console.log('[ManualService] After engineering:', JSON.stringify(row, null, 2));
		
		const includeDebug = !!input.debug || process.env.MANUAL_DEBUG === 'true';
		console.log(`[ManualService] Calling Python model... includeDebug=${includeDebug} (query/header/env)`);
		const mlResult = await callPythonModel(row, includeDebug);
		console.log('[ManualService] Python result:', JSON.stringify(mlResult, null, 2));

		// Persist into DB disabled unless SAVE_TO_DB=true
		if (process.env.SAVE_TO_DB === 'true') {
			try {
				await insertPredictionIntoDb(row, mlResult);
				console.log('[ManualService] Saved to database');
			} catch (e) {
				console.error('[ManualService] Failed to insert prediction into DB:', e);
			}
		} else {
			console.log('[ManualService] SAVE_TO_DB=false: Skipping DB insert');
		}

		const response = {
      prediction: mlResult.prediction,
      confidence: mlResult.confidence,
      key_factors: mlResult.key_factors || [],
		};
		if (mlResult.debug) {
			response.debug = mlResult.debug;
		}
		return response;
	} catch (err) {
		console.error('[ManualService] Error:', err);
		throw new Error(`Transaction analysis failed: ${err.message}`);
	}
}

const { DB_USE, pool } = require('../config/dbConfig');

const COLUMN_RENAME_MAP = {
  Timestamp: 'timestamp',
  'From Bank': 'from_bank_txn',
  Account: 'account',
  'To Bank': 'to_bank_txn',
  'Account.1': 'account_1',
  'Amount Received': 'amount_received',
  'Receiving Currency': 'receiving_currency',
  Amount: 'amount',
  'Payment Currency': 'payment_currency',
  'Payment Format': 'payment_format',
  'Is Laundering': 'is_laundering',

  from_bank: 'from_bank',
  account_number: 'account_number',
  profile_type: 'profile_type',
  name: 'name',
  date_of_birth: 'date_of_birth',
  nationality: 'nationality',
  occupation: 'occupation',
  risk_profile: 'risk_profile',
  kyc_score: 'kyc_score',
  monthly_income: 'monthly_income',
  is_pep: 'is_pep',
  sanctions_check: 'sanctions_check',
  customer_since: 'customer_since',
  customer_tenure_month: 'customer_tenure_month',
  kyc_status: 'kyc_status',
  age: 'age',

  amount_to_income_ratio: 'amount_to_income_ratio',
  days_since_last_txn: 'days_since_last_txn',
  txn_count_last_7_days: 'txn_count_last_7_days',
  total_amount_last_30_days: 'total_amount_last_30_days',

  beneficiary_receive_count: 'beneficiary_receive_count',
  beneficiary_total_received: 'beneficiary_total_received',
  beneficiary_avg_received_amount: 'beneficiary_avg_received_amount',
  beneficiary_unique_senders: 'beneficiary_unique_senders',
  beneficiary_unique_sender_nationalities_so_far: 'beneficiary_unique_sender_nationalities_so_far',
  beneficiary_pep_sender_count_at_time_of_txn: 'beneficiary_pep_sender_count_at_time_of_txn',
  beneficiary_unique_senders_at_time_of_txn: 'beneficiary_unique_senders_at_time_of_txn',
  beneficiary_receive_count_so_far: 'beneficiary_receive_count_so_far',
  beneficiary_total_received_so_far: 'beneficiary_total_received_so_far',

  Amount_binned: 'amount_binned',
  'Amount Received_binned': 'amount_received_binned',
  monthly_income_binned: 'monthly_income_binned',
  total_amount_last_30_days_binned: 'total_amount_last_30_days_binned',
  beneficiary_total_received_binned: 'beneficiary_total_received_binned',
  beneficiary_total_received_so_far_binned: 'beneficiary_total_received_so_far_binned',
  beneficiary_unique_senders_binned: 'beneficiary_unique_senders_binned',
  days_since_last_txn_binned: 'days_since_last_txn_binned',
};

async function insertPredictionIntoDb(originalRow, mlResult) {
  if (!DB_USE || !pool) return;
  const conn = await pool.getConnection();
  try {
		const [colsRows] = await conn.query(
			"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'transaction'"
		);
    const colsSet = new Set(colsRows.map((r) => r.COLUMN_NAME));

    const toInsert = {};

    // map originalRow keys to DB columns
    for (const k of Object.keys(originalRow)) {
      const dbCol = COLUMN_RENAME_MAP[k] || COLUMN_RENAME_MAP[k] || null;
      if (!dbCol) continue;
      if (colsSet.has(dbCol)) toInsert[dbCol] = originalRow[k];
    }

    // add prediction/confidence/key_factors/saved_at
    if (mlResult.prediction !== undefined) {
      if (colsSet.has('is_laundering')) toInsert['is_laundering'] = mlResult.prediction;
      else if (colsSet.has('prediction')) toInsert['prediction'] = mlResult.prediction;
    }
    if (mlResult.confidence !== undefined && colsSet.has('confidence')) toInsert['confidence'] = mlResult.confidence;
    if (mlResult.key_factors !== undefined && colsSet.has('key_factors')) toInsert['key_factors'] = JSON.stringify(mlResult.key_factors);
    if (colsSet.has('timestamp')) toInsert['timestamp'] = new Date();

    if (Object.keys(toInsert).length === 0) return;

    const cols = Object.keys(toInsert);
    const placeholders = cols.map(() => '?').join(', ');
    const vals = cols.map((c) => toInsert[c]);

	const sql = `INSERT INTO transaction (${cols.join(',')}) VALUES (${placeholders})`;
    await conn.query(sql, vals);
  } catch (e) {
    console.error('insertPredictionIntoDb error:', e);
  } finally {
    conn.release();
  }
}

module.exports = { analyzeManualTransaction, insertPredictionIntoDb };
