#!/usr/bin/env python3
import sys
import json
import os
from datetime import datetime
import traceback

import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier

# ----- CONFIG -----
# Ensemble model paths (4 base models + 1 stacking model)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CAT_MODEL_PATH = os.path.join(MODELS_DIR, "cat_model.cbm")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
STACKED_MODEL_PATH = os.path.join(MODELS_DIR, "stacked_meta_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "categorical_encoders.pkl")
STACKING_CONFIG_PATH = os.path.join(MODELS_DIR, "stacking_config.pkl")
SAVE_TO_DB = os.environ.get("SAVE_TO_DB", "false").strip().lower() == "true"
USE_DB_FEATURES = os.environ.get("USE_DB_FEATURES", "false").strip().lower() == "true"

# # (bins_config, categorical_features, numeric_features copied from your snippet)
# bins_config = {
#     # Log-binned features (edges given in log scale; we'll bin log1p(values) vs log1p(edges))
#     "amount": [0.0, 1.8441, 4.0596, 5.7781, 7.2526, 8.6675, 10.1235, 11.6798, 13.4355, 15.9486, 26.3736],

#     "amount_received": [
#         0.0, 1.4604, 3.2467, 4.4580, 5.4437, 6.3596, 7.2233, 8.0237, 8.8150, 9.5770,
#         10.3954, 11.2979, 12.2083, 13.1704, 14.1645, 15.3002, 16.5448, 17.8393, 19.7044,
#         22.8825, 26.3736
#     ],

#     "monthly_income": [
#         0.0, 0.5206, 1.0392, 1.5876, 2.1236, 2.6388, 3.1430, 3.6144, 4.1102, 4.6525,
#         5.2394, 5.8274, 6.3532, 6.8123, 7.4512, 8.2301, 8.7556, 9.1587, 9.5840, 9.9852,
#         10.3816, 10.8254, 11.3219, 11.8213, 12.3389, 12.8531, 13.3561, 13.8759, 14.4061,
#         14.9233, 15.4231, 15.9093, 16.4066, 16.9488, 17.5086, 18.0795, 18.6742, 19.2715,
#         19.7918, 20.2718, 20.8142, 21.3665, 21.8847, 22.4040, 22.8359, 23.2913, 23.9472,
#         24.4255, 24.7969, 25.5227, 26.0359
#     ],

#     "total_amount_last_30_days": [
#         0.0, 0.3677, 0.9154, 1.4480, 1.9318, 2.3788, 2.7969, 3.2054, 3.6865, 4.2075,
#         4.6912, 5.1676, 5.6359, 6.0951, 6.5587, 7.0323, 7.5034, 7.9699, 8.4359, 8.9005,
#         9.3652, 9.8270, 10.2892, 10.7596, 11.2332, 11.7020, 12.1570, 12.6021, 13.0558,
#         13.5232, 13.9896, 14.4539, 14.9211, 15.3839, 15.8521, 16.3294, 16.7951, 17.2466,
#         17.7178, 18.1896, 18.6490, 19.1252, 19.5933, 20.0600, 20.5737, 21.0788, 21.5257,
#         21.9958, 22.4558, 22.9315, 23.4346, 23.8773, 24.2818, 24.6916, 25.0841, 25.6109,
#         26.1543, 26.5349, 27.0682, 27.5567, 28.0302
#     ],

#     "beneficiary_total_received": [
#         0.0005, 0.4065, 0.8979, 1.4089, 1.9410, 2.5087, 3.0895, 3.7428, 4.5006, 5.2760,
#         6.0172, 6.6889, 7.2813, 7.8238, 8.3326, 8.8193, 9.2840, 9.7326, 10.1769, 10.6178,
#         11.0489, 11.4854, 11.9211, 12.3440, 12.7768, 13.2322, 13.7035, 14.1846, 14.6725,
#         15.1658, 15.6532, 16.1435, 16.6438, 17.1608, 17.6996, 18.2203, 18.7192, 19.2199,
#         19.7841, 20.4078, 21.0329, 21.7792, 22.6165, 23.3564, 24.0261, 24.6463, 25.2167,
#         25.7556, 26.3363, 27.2238, 28.1588
#     ],

#     "beneficiary_total_received_so_far": [
#         0.0, 1.7499, 5.1668, 7.9832, 9.8973, 11.4924, 12.9978, 14.6227, 16.5272, 18.9334, 27.9706
#     ],

#     # Raw-scale binned features
#     "beneficiary_unique_senders": [
#         1.0, 2.6715, 6.0304, 12.3777, 23.1015, 40.1880, 58.2231, 72.2574, 84.0147, 103.25,
#         222.5, 347.5, 456.0, 761.0, 1241.5, 1506.0
#     ],
# }


bins_config = {
    "amount": [
        0.0, 1.0187, 2.3624, 3.3438, 4.1856, 4.9100, 5.5735, 6.2210, 6.8546,
        7.4728, 8.0670, 8.6665, 9.2847, 9.9030, 10.5701, 11.2767, 11.9944,
        12.7265, 13.4589, 14.1707, 14.9193, 15.7548, 16.6328, 17.6152,
        18.7237, 20.0696, 21.2766, 22.1830, 23.2231, 24.4157, 25.1642
    ],
    "amount_received": [
        0.0, 1.1976, 2.7734, 3.9584, 4.9684, 5.8408, 6.6750, 7.4656, 8.2049,
        8.9250, 9.6161, 10.3593, 11.1846, 12.0305, 12.9291, 13.8640, 14.9651,
        16.2615, 17.8037, 20.1256, 25.1642
    ],
    "monthly_income": [
        0.0, 0.4155, 0.8602, 1.2938, 1.7003, 2.1178, 2.5428, 2.9610, 3.3935,
        3.8202, 4.2252, 4.6455, 5.1082, 5.5890, 6.0291, 6.4342, 6.7940,
        7.2777, 7.8635, 8.3697, 8.7521, 9.0774, 9.4198, 9.7285, 10.0741,
        10.4657, 10.8913, 11.3428, 11.7758, 12.2068, 12.6348, 13.0557,
        13.4723, 13.8900, 14.3305, 14.7710, 15.1963, 15.6201, 16.0383,
        16.4572, 16.8760, 17.2871, 17.6911, 18.0847, 18.4792, 18.9019,
        19.3309, 19.7913, 20.2747, 20.7987, 21.2824, 21.6849, 22.1267,
        22.5303, 22.8895, 23.3044, 23.7636, 24.2230, 24.6518, 25.1098,
        25.5159
    ],
    "total_amount_last_30_days": [
        0.0001, 0.8709, 2.0194, 2.8857, 3.8259, 4.9004, 5.8762, 6.8166,
        7.7586, 8.6771, 9.5709, 10.4402, 11.2862, 12.1086, 12.9235, 13.7399,
        14.5605, 15.4079, 16.2897, 17.2038, 18.1356, 19.1194, 20.2500,
        21.3446, 22.2374, 23.1528, 23.9928, 24.6168, 25.5440, 26.7507, 27.8354
    ],
    "beneficiary_total_received": [
        0.0004, 0.4831, 1.0241, 1.5704, 2.1230, 2.6781, 3.2260, 3.8433,
        4.4934, 5.1636, 5.8543, 6.5052, 7.1201, 7.6882, 8.2279, 8.7406,
        9.2335, 9.7172, 10.2017, 10.6882, 11.1763, 11.6656, 12.1470, 12.6426,
        13.1636, 13.6941, 14.2296, 14.7779, 15.3298, 15.8640, 16.3824,
        16.8980, 17.4553, 18.0152, 18.5307, 19.0411, 19.5713, 20.1010,
        20.6923, 21.3210, 21.9326, 22.5914, 23.1922, 23.7174, 24.3936,
        24.9377, 25.3317, 25.9451, 26.4801, 27.1434, 27.6765
    ],
    "beneficiary_total_received_so_far": [
        0.0, 0.6403, 1.7589, 2.6701, 3.6795, 4.8191, 5.8499, 6.7767, 7.6300,
        8.4077, 9.1237, 9.8099, 10.4832, 11.1480, 11.7979, 12.4446, 13.1100,
        13.7968, 14.5098, 15.2402, 15.9767, 16.7186, 17.4800, 18.2753,
        19.1468, 20.1364, 21.3383, 22.7626, 24.9182, 26.9621, 27.5966
    ],
    "beneficiary_unique_senders": [
        1.0, 5.1289, 12.0361, 26.3138, 45.7078, 64.3732, 200.6076, 436.5,
        761.0, 1241.5, 1506.0
    ],
    "days_since_last_txn": [0.0, 3.1572, 9.1347, 14.0696, 19.0167, 24.0]
}




# Features that should be log-transformed BEFORE binning
LOG_BIN_FEATURES = [
    'amount', 'amount_received', 'monthly_income', 'total_amount_last_30_days',
    'beneficiary_total_received', 'beneficiary_total_received_so_far'
]

categorical_features = [
    'Receiving Currency', 'Payment Currency', 'Payment Format',
    'nationality', 'occupation', 'kyc_status', 'Amount_binned',
    'Amount Received_binned', 'monthly_income_binned',
    'total_amount_last_30_days_binned', 'beneficiary_total_received_binned',
    'beneficiary_total_received_so_far_binned', 'beneficiary_unique_senders_binned',
    'days_since_last_txn_binned'
]

numeric_features = [
    'kyc_score', 'is_pep', 'age', 'amount_to_income_ratio',
    'txn_count_last_7_days', 'beneficiary_receive_count',
    'beneficiary_avg_received_amount',
    'beneficiary_unique_sender_nationalities_so_far',
    'beneficiary_pep_sender_count_at_time_of_txn',
    'beneficiary_unique_senders_at_time_of_txn',
    'beneficiary_receive_count_so_far'
]

# Column rename map: map model/CSV keys -> DB column names
COLUMN_RENAME_MAP = {
    # Customer-level fields
    "from_bank": "from_bank",
    "account_number": "account_number",
    "profile_type": "profile_type",
    "name": "name",
    "date_of_birth": "date_of_birth",
    "nationality": "nationality",
    "occupation": "occupation",
    "risk_profile": "risk_profile",
    "kyc_score": "kyc_score",
    "monthly_income": "monthly_income",
    "is_pep": "is_pep",
    "sanctions_check": "sanctions_check",
    "customer_since": "customer_since",
    "customer_tenure_month": "customer_tenure_month",
    "kyc_status": "kyc_status",
    "age": "age",

    # Derived transaction behavior features
    "amount_to_income_ratio": "amount_to_income_ratio",
    "days_since_last_txn": "days_since_last_txn",
    "txn_count_last_7_days": "txn_count_last_7_days",
    "total_amount_last_30_days": "total_amount_last_30_days",

    # Beneficiary aggregation features
    "beneficiary_receive_count": "beneficiary_receive_count",
    "beneficiary_total_received": "beneficiary_total_received",
    "beneficiary_avg_received_amount": "beneficiary_avg_received_amount",
    "beneficiary_unique_senders": "beneficiary_unique_senders",
    "beneficiary_unique_sender_nationalities_so_far": "beneficiary_unique_sender_nationalities_so_far",
    "beneficiary_pep_sender_count_at_time_of_txn": "beneficiary_pep_sender_count_at_time_of_txn",
    "beneficiary_unique_senders_at_time_of_txn": "beneficiary_unique_senders_at_time_of_txn",
    "beneficiary_receive_count_so_far": "beneficiary_receive_count_so_far",
    "beneficiary_total_received_so_far": "beneficiary_total_received_so_far",

    # Binned / categorical features
    "amount_binned": "amount_binned",
    "amount_received_binned": "amount_received_binned",
    "monthly_income_binned": "monthly_income_binned",
    "total_amount_last_30_days_binned": "total_amount_last_30_days_binned",
    "beneficiary_total_received_binned": "beneficiary_total_received_binned",
    "beneficiary_total_received_so_far_binned": "beneficiary_total_received_so_far_binned",
    "beneficiary_unique_senders_binned": "beneficiary_unique_senders_binned",
    "days_since_last_txn_binned": "days_since_last_txn_binned",
}

# -----------------------
# helpers: bins logic
# -----------------------
def prepare_bins(edges):
    edges = sorted(set(edges))
    if edges[-1] != float("inf"):
        edges = edges + [float("inf")]
    return edges

def apply_bins(df, bins_dict, log_bin_features=None):
    """
    Apply binning to df based on bins_dict.
    For features listed in log_bin_features, bin log1p(values) AGAINST EDGES ALREADY PROVIDED IN LOG SCALE.
    Important: Do NOT log-transform edges; they are assumed to be in log space already.
    """
    if log_bin_features is None:
        log_bin_features = []
    binned_df = df.copy()
    for feature, edges in bins_dict.items():
        if feature not in binned_df.columns:
            # skip silently if feature not present
            continue

        # Prepare values
        vals = pd.to_numeric(binned_df[feature], errors="coerce").fillna(0)

        # Prepare edges
        raw_edges = prepare_bins(edges)

        if feature in log_bin_features:
            # Transform values to log scale; use edges as provided (already in log space)
            vals = np.log1p(vals)
            edges_to_use = raw_edges
            # Ensure -inf coverage if needed
            if edges_to_use[0] > -np.inf and vals.min(skipna=True) < edges_to_use[0]:
                edges_to_use = [-float("inf")] + edges_to_use
            labels = range(1, len(edges_to_use))
            binned_col = pd.cut(
                vals,
                bins=edges_to_use,
                labels=labels,
                include_lowest=True,
                right=True
            )
        else:
            # Binning on raw scale
            edges_to_use = raw_edges
            if edges_to_use[0] > -np.inf and vals.min(skipna=True) < edges_to_use[0]:
                edges_to_use = [-float("inf")] + edges_to_use
            labels = range(1, len(edges_to_use))
            binned_col = pd.cut(
                vals,
                bins=edges_to_use,
                labels=labels,
                include_lowest=True,
                right=True
            )

        binned_df[feature + "_binned"] = binned_col.astype("Int64")
    return binned_df

# -----------------------
# helpers: debug printing
# -----------------------
def _print_feature_block(title: str, df: pd.DataFrame, columns: list | None = None):
    try:
        print(f"\n=== {title} ===", file=sys.stderr)
        if columns is None:
            # Print all columns sorted for determinism
            row_dict = df.iloc[0].to_dict()
            for key, value in sorted(row_dict.items()):
                print(f"{key}: {value}", file=sys.stderr)
        else:
            cols = [c for c in columns if c in df.columns]
            print(f"Columns: {cols}", file=sys.stderr)
            if cols:
                print(f"Values: {df[cols].values[0].tolist()}", file=sys.stderr)
    except Exception as e:
        # Log the error so we can diagnose silently failing prints
        print(f"[ERROR] Failed to print feature block '{title}': {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

# -----------------------
# mapping UI->model
# -----------------------
# FEATURE_MAP: Maps frontend field names to database column names
# -----------------------
FEATURE_MAP = {
    "fromBank": "from_bank",
    "fromAccount": "account",
    "toBank": "to_bank_txn",  # or use a string like "to_bank" if that column exists
    "toAccount": "account_1",
    "amountReceived": "amount_received",
    "receivingCurrency": "receiving_currency",
    "amount": "amount",
    "paymentCurrency": "payment_currency",
    "paymentFormat": "payment_format",

    "fullName": "name",
    "nationality": "nationality",
    "occupation": "occupation",
    "kycStatus": "kyc_status",
    "kycScore": "kyc_score",
    "isPep": "is_pep",
    "monthlyIncome": "monthly_income",
    "dob": "date_of_birth",
    "customerSince": "customer_since",

    # Additional raw features provided by user (no DB lookup)
    "txnCountLast7Days": "txn_count_last_7_days",
    "totalAmountLast30Days": "total_amount_last_30_days",
    "daysSinceLastTxn": "days_since_last_txn",

    "beneficiaryReceiveCount": "beneficiary_receive_count",
    "beneficiaryTotalReceived": "beneficiary_total_received",
    "beneficiaryAvgReceivedAmount": "beneficiary_avg_received_amount",
    "beneficiaryUniqueSenders": "beneficiary_unique_senders",
    "beneficiaryUniqueSenderNationalitiesSoFar": "beneficiary_unique_sender_nationalities_so_far",
    "beneficiaryPepSenderCountAtTimeOfTxn": "beneficiary_pep_sender_count_at_time_of_txn",
    "beneficiaryUniqueSendersAtTimeOfTxn": "beneficiary_unique_senders_at_time_of_txn",
    "beneficiaryReceiveCountSoFar": "beneficiary_receive_count_so_far",
    "beneficiaryTotalReceivedSoFar": "beneficiary_total_received_so_far",
}

# -----------------------
# utility: compute sender-based features from historical data
# -----------------------
def compute_sender_features(row: dict, historical_df: pd.DataFrame) -> dict:
    """
    Compute sender-based features from database:
    - days_since_last_txn: days since last transaction from same sender account
    - txn_count_last_7_days: number of transactions by sender in last 7 days
    - total_amount_last_30_days: total amount sent by sender in last 30 days
    """
    print("\n[DEBUG] === COMPUTING SENDER FEATURES ===", file=sys.stderr)
    now = datetime.now()
    sender_account = row.get("account") or row.get("fromAccount")
    print(f"[DEBUG] Sender account: {sender_account}", file=sys.stderr)
    
    # Fetch matching sender transactions from database
    try:
        import db_config
        if sender_account:
            sql = (
                "SELECT account, account_1, amount, amount_received, timestamp, "
                "is_pep, nationality, from_bank, account_number "
                "FROM transaction WHERE account=%s OR account_number=%s "
                "ORDER BY timestamp DESC LIMIT 500"
            )
            conn = db_config.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (sender_account, sender_account))
                    rows = cur.fetchall()
                if rows:
                    historical_df = pd.DataFrame(rows)
                else:
                    historical_df = pd.DataFrame()
            finally:
                conn.close()
        else:
            historical_df = pd.DataFrame()
    except Exception as e:
        print(f"[WARN] Failed to fetch sender history from DB: {e}", file=sys.stderr)
        historical_df = pd.DataFrame()

    if not sender_account or historical_df.empty:
        # For new accounts with no history, set meaningful defaults based on current transaction
        row["days_since_last_txn"] = 0  # This is the first transaction
        row["txn_count_last_7_days"] = 1  # Count this transaction
        row["total_amount_last_30_days"] = float(row.get("amount") or 0)
        return row
    
    try:
        # Filter transactions from this sender
        account_col = "account" if "account" in historical_df.columns else "account_number"
        sender_txns = historical_df[historical_df[account_col] == sender_account].copy()
        
        if not sender_txns.empty:
            # Use timestamp column for date comparisons
            if "timestamp" in sender_txns.columns:
                sender_txns["transaction_date"] = pd.to_datetime(sender_txns["timestamp"], errors='coerce')
            else:
                # Use current date as placeholder if no date column
                sender_txns["transaction_date"] = now
            
            # Sort by date descending
            sender_txns = sender_txns.sort_values("transaction_date", ascending=False)
            
            # 1) Days since last transaction
            last_txn_date = sender_txns["transaction_date"].iloc[0]
            if pd.notna(last_txn_date):
                days_diff = (now - last_txn_date).days
                row["days_since_last_txn"] = max(0, days_diff)
            else:
                row["days_since_last_txn"] = 0
            
            # 2) Transactions in last 7 days
            seven_days_ago = now - pd.Timedelta(days=7)
            txns_last_7 = sender_txns[sender_txns["transaction_date"] >= seven_days_ago]
            row["txn_count_last_7_days"] = len(txns_last_7)
            
            # 3) Total amount in last 30 days
            thirty_days_ago = now - pd.Timedelta(days=30)
            txns_last_30 = sender_txns[sender_txns["transaction_date"] >= thirty_days_ago]
            if "amount" in txns_last_30.columns:
                total_30 = pd.to_numeric(txns_last_30["amount"], errors='coerce').sum()
                row["total_amount_last_30_days"] = float(total_30) if pd.notna(total_30) else 0
            else:
                row["total_amount_last_30_days"] = float(row.get("amount") or 0)
        else:
            # No transactions found in history, use current transaction as baseline
            row["days_since_last_txn"] = 0
            row["txn_count_last_7_days"] = 1
            row["total_amount_last_30_days"] = float(row.get("amount") or 0)
    except Exception as e:
        print(f"[WARN] Error computing sender features: {e}", file=sys.stderr)
        # On error, use current transaction as baseline
        row["days_since_last_txn"] = 0
        row["txn_count_last_7_days"] = 1
        row["total_amount_last_30_days"] = float(row.get("amount") or 0)
    
    print(f"[DEBUG] Sender features computed:", file=sys.stderr)
    print(f"[DEBUG]   days_since_last_txn = {row.get('days_since_last_txn', 'NOT SET')}", file=sys.stderr)
    print(f"[DEBUG]   txn_count_last_7_days = {row.get('txn_count_last_7_days', 'NOT SET')}", file=sys.stderr)
    print(f"[DEBUG]   total_amount_last_30_days = {row.get('total_amount_last_30_days', 'NOT SET')}", file=sys.stderr)
    
    return row

# -----------------------
# utility: compute beneficiary-based features from historical data
# -----------------------
def compute_beneficiary_features(row: dict, historical_df: pd.DataFrame) -> dict:
    """
    Compute beneficiary-based features:
    - beneficiary_receive_count: number of times this beneficiary received
    - beneficiary_total_received: total amount received by beneficiary
    - beneficiary_avg_received_amount: average amount received
    - beneficiary_unique_senders: number of unique senders
    - beneficiary_unique_sender_nationalities_so_far: unique sender nationalities
    - beneficiary_pep_sender_count_at_time_of_txn: count of PEP senders
    - beneficiary_receive_count_so_far: same as receive_count
    - beneficiary_total_received_so_far: same as total_received
    - beneficiary_unique_senders_at_time_of_txn: same as unique_senders
    """
    print("\n[DEBUG] === COMPUTING BENEFICIARY FEATURES ===", file=sys.stderr)
    beneficiary_account = row.get("account_1") or row.get("toAccount")
    print(f"[DEBUG] Beneficiary account: {beneficiary_account}", file=sys.stderr)
    
    # Fetch matching beneficiary transactions from database
    try:
        import db_config
        if beneficiary_account:
            sql = (
                "SELECT account, to_bank_txn, account_1, from_bank, amount_received, amount, "
                    "timestamp, is_pep, nationality "
                "FROM transaction WHERE account_1=%s "
                    "ORDER BY timestamp DESC LIMIT 500"
            )
            conn = db_config.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (beneficiary_account,))
                    rows = cur.fetchall()
                if rows:
                    historical_df = pd.DataFrame(rows)
                else:
                    historical_df = pd.DataFrame()
            finally:
                conn.close()
        else:
            historical_df = pd.DataFrame()
    except Exception as e:
        print(f"[WARN] Failed to fetch beneficiary history from DB: {e}", file=sys.stderr)
        historical_df = pd.DataFrame()

    if not beneficiary_account or historical_df.empty:
        # For new beneficiary with no history, calculate from current transaction
        current_amount = float(row.get("amount_received") or row.get("amount") or 0)
        sender_account = row.get("account") or row.get("fromAccount") or ""
        sender_nat = row.get("nationality") or "Unknown"
        is_pep = int(row.get("is_pep") or 0)
        
        row["beneficiary_receive_count"] = 1
        row["beneficiary_total_received"] = current_amount
        row["beneficiary_avg_received_amount"] = current_amount
        row["beneficiary_unique_senders"] = 1 if sender_account else 0
        row["beneficiary_unique_sender_nationalities_so_far"] = 1 if sender_nat != "Unknown" else 0
        row["beneficiary_pep_sender_count_at_time_of_txn"] = is_pep
        row["beneficiary_receive_count_so_far"] = 1
        row["beneficiary_total_received_so_far"] = current_amount
        row["beneficiary_unique_senders_at_time_of_txn"] = 1 if sender_account else 0
        
        print(f"[DEBUG] No history found. Using current transaction values:", file=sys.stderr)
        print(f"[DEBUG]   beneficiary_receive_count = {row['beneficiary_receive_count']}", file=sys.stderr)
        print(f"[DEBUG]   beneficiary_total_received = {row['beneficiary_total_received']}", file=sys.stderr)
        print(f"[DEBUG]   beneficiary_unique_senders = {row['beneficiary_unique_senders']}", file=sys.stderr)
        print(f"[DEBUG]   beneficiary_pep_sender_count_at_time_of_txn = {row['beneficiary_pep_sender_count_at_time_of_txn']}", file=sys.stderr)
        return row
    
    try:
        # Filter transactions where this account is the beneficiary/receiver
        benef_col = "account_1"  # This is the correct column name in the database
        benef_txns = historical_df[historical_df[benef_col] == beneficiary_account].copy()
        
        if not benef_txns.empty:
            # 1) Beneficiary receive count
            receive_count = len(benef_txns)
            row["beneficiary_receive_count"] = receive_count
            row["beneficiary_receive_count_so_far"] = receive_count
            
            # 2) Total amount received by beneficiary
            amount_col = "amount_received" if "amount_received" in benef_txns.columns else "amount"
            
            if amount_col in benef_txns.columns:
                total_received = pd.to_numeric(benef_txns[amount_col], errors='coerce').sum()
                row["beneficiary_total_received"] = float(total_received) if pd.notna(total_received) else 0
                row["beneficiary_total_received_so_far"] = float(total_received) if pd.notna(total_received) else 0
                
                # 3) Average amount received: total_amount_to_beneficiary / number of transactions
                if receive_count > 0:
                    row["beneficiary_avg_received_amount"] = float(total_received / receive_count) if pd.notna(total_received) else 0
                else:
                    row["beneficiary_avg_received_amount"] = 0
            else:
                row["beneficiary_total_received"] = 0
                row["beneficiary_total_received_so_far"] = 0
                row["beneficiary_avg_received_amount"] = 0
            
            # 4) Unique senders: number of unique bank accounts who send to beneficiary
            sender_col = "account" if "account" in benef_txns.columns else "from_account"
            
            if sender_col in benef_txns.columns:
                unique_senders = benef_txns[sender_col].nunique()
                row["beneficiary_unique_senders"] = unique_senders
                row["beneficiary_unique_senders_at_time_of_txn"] = unique_senders
            else:
                row["beneficiary_unique_senders"] = 0
                row["beneficiary_unique_senders_at_time_of_txn"] = 0
            
            # 5) Unique sender nationalities: number of different nationalities of people who send to beneficiary
            nat_col = "nationality"
            if nat_col in benef_txns.columns:
                # Filter out null/empty nationalities and count unique
                unique_nationalities = benef_txns[nat_col].dropna().nunique()
                row["beneficiary_unique_sender_nationalities_so_far"] = unique_nationalities
            else:
                row["beneficiary_unique_sender_nationalities_so_far"] = 0
            
            # 6) PEP senders count: the number of PEP people who send money to beneficiary
            pep_col = "is_pep"
            if pep_col in benef_txns.columns:
                # Count rows where is_pep equals 1 or True
                pep_senders = (pd.to_numeric(benef_txns[pep_col], errors='coerce') == 1).sum()
                row["beneficiary_pep_sender_count_at_time_of_txn"] = int(pep_senders)
            else:
                row["beneficiary_pep_sender_count_at_time_of_txn"] = 0
        else:
            # No historical records found, use current transaction as baseline
            current_amount = float(row.get("amount_received") or row.get("amount") or 0)
            sender_account = row.get("account") or row.get("fromAccount") or ""
            sender_nat = row.get("nationality") or "Unknown"
            is_pep = int(row.get("is_pep") or 0)
            
            row["beneficiary_receive_count"] = 1
            row["beneficiary_total_received"] = current_amount
            row["beneficiary_avg_received_amount"] = current_amount
            row["beneficiary_unique_senders"] = 1 if sender_account else 0
            row["beneficiary_unique_sender_nationalities_so_far"] = 1 if sender_nat != "Unknown" else 0
            row["beneficiary_pep_sender_count_at_time_of_txn"] = is_pep
            row["beneficiary_receive_count_so_far"] = 1
            row["beneficiary_total_received_so_far"] = current_amount
            row["beneficiary_unique_senders_at_time_of_txn"] = 1 if sender_account else 0
    except Exception as e:
        print(f"[WARN] Error computing beneficiary features: {e}", file=sys.stderr)
        # On error, use current transaction as baseline
        current_amount = float(row.get("amount_received") or row.get("amount") or 0)
        sender_account = row.get("account") or row.get("fromAccount") or ""
        sender_nat = row.get("nationality") or "Unknown"
        is_pep = int(row.get("is_pep") or 0)
        
        row["beneficiary_receive_count"] = 1
        row["beneficiary_total_received"] = current_amount
        row["beneficiary_avg_received_amount"] = current_amount
        row["beneficiary_unique_senders"] = 1 if sender_account else 0
        row["beneficiary_unique_sender_nationalities_so_far"] = 1 if sender_nat != "Unknown" else 0
        row["beneficiary_pep_sender_count_at_time_of_txn"] = is_pep
        row["beneficiary_receive_count_so_far"] = 1
        row["beneficiary_total_received_so_far"] = current_amount
        row["beneficiary_unique_senders_at_time_of_txn"] = 1 if sender_account else 0
    
    print(f"[DEBUG] Beneficiary features computed successfully:", file=sys.stderr)
    print(f"[DEBUG]   beneficiary_receive_count = {row.get('beneficiary_receive_count', 'NOT SET')}", file=sys.stderr)
    print(f"[DEBUG]   beneficiary_total_received = {row.get('beneficiary_total_received', 'NOT SET')}", file=sys.stderr)
    print(f"[DEBUG]   beneficiary_unique_senders = {row.get('beneficiary_unique_senders', 'NOT SET')}", file=sys.stderr)
    print(f"[DEBUG]   beneficiary_pep_sender_count_at_time_of_txn = {row.get('beneficiary_pep_sender_count_at_time_of_txn', 'NOT SET')}", file=sys.stderr)
    
# utility: compute engineered features
# -----------------------
def compute_engineered_features(row: dict) -> dict:
    now = datetime.now()
    
    # age
    if row.get("date_of_birth"):
        try:
            dob = pd.to_datetime(row["date_of_birth"])
            row["age"] = int(now.year - dob.year - ((now.month, now.day) < (dob.month, dob.day)))
        except Exception:
            row["age"] = np.nan
    else:
        row["age"] = np.nan

    # customer_tenure_month
    if row.get("customer_since"):
        try:
            cs = pd.to_datetime(row["customer_since"])
            months = (now.year - cs.year) * 12 + (now.month - cs.month)
            row["customer_tenure_month"] = int(months)
        except Exception:
            row["customer_tenure_month"] = np.nan
    else:
        row["customer_tenure_month"] = np.nan

    # amount_to_income_ratio
    try:
        amt = float(row.get("amount") or 0)
        income = float(row.get("monthly_income") or 0)
        row["amount_to_income_ratio"] = amt / income if income not in (0, None, np.nan) else np.nan
    except Exception:
        row["amount_to_income_ratio"] = np.nan
    
    # Compute sender/beneficiary features
    # If USE_DB_FEATURES=true, enrich from DB; otherwise, expect raw values from user and fallback to safe defaults
    if USE_DB_FEATURES:
        row = compute_sender_features(row, pd.DataFrame())
        row = compute_beneficiary_features(row, pd.DataFrame())
    else:
        # Ensure presence of sender features
        row.setdefault("days_since_last_txn", 0)
        row.setdefault("txn_count_last_7_days", 0)
        # total_amount_last_30_days: default to current amount if missing
        if "total_amount_last_30_days" not in row:
            try:
                row["total_amount_last_30_days"] = float(row.get("amount") or 0)
            except Exception:
                row["total_amount_last_30_days"] = 0

        # Ensure presence of beneficiary features
        for key, default in [
            ("beneficiary_receive_count", 0),
            ("beneficiary_total_received", float(row.get("amount_received") or row.get("amount") or 0)),
            ("beneficiary_avg_received_amount", float(row.get("amount_received") or row.get("amount") or 0)),
            ("beneficiary_unique_senders", 0),
            ("beneficiary_unique_sender_nationalities_so_far", 0),
            ("beneficiary_pep_sender_count_at_time_of_txn", int(row.get("is_pep") or 0)),
            ("beneficiary_receive_count_so_far", 0),
            ("beneficiary_total_received_so_far", float(row.get("amount_received") or row.get("amount") or 0)),
            ("beneficiary_unique_senders_at_time_of_txn", 0),
        ]:
            row.setdefault(key, default)

    return row

# -----------------------
# helper: save prediction to database
# -----------------------
def insert_prediction_into_db(row: dict):
    """Insert the given row into the transaction table with all features, mapping only existing columns."""
    try:
        import db_config
        conn = db_config.get_connection()
        try:
            with conn.cursor() as cur:
                # fetch available columns
                cur.execute("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'transaction'", (os.environ.get('DB_NAME', ''),))
                cols = [r['COLUMN_NAME'] for r in cur.fetchall()]

                to_insert = {}
                colsSet = set(cols)

                # 1) Use explicit COLUMN_RENAME_MAP when available
                for model_key, db_col in COLUMN_RENAME_MAP.items():
                    if model_key in row and db_col in colsSet:
                        v = row[model_key]
                        if isinstance(v, (dict, list)):
                            to_insert[db_col] = json.dumps(v)
                        elif isinstance(v, datetime):
                            to_insert[db_col] = v.isoformat()
                        else:
                            to_insert[db_col] = v

                # 2) Map standard prediction/confidence/key_factors/saved_at if present
                if 'prediction' in row:
                    if 'is_laundering' in colsSet:
                        to_insert['is_laundering'] = row['prediction']
                    elif 'prediction' in colsSet:
                        to_insert['prediction'] = row['prediction']
                if 'confidence' in row and 'confidence' in colsSet:
                    to_insert['confidence'] = row['confidence']
                if 'key_factors' in row and 'key_factors' in colsSet:
                    to_insert['key_factors'] = json.dumps(row['key_factors']) if isinstance(row['key_factors'], (list, dict)) else row['key_factors']
                if 'saved_at' in row:
                    if 'timestamp' in colsSet:
                        to_insert['timestamp'] = row['saved_at']
                    elif 'saved_at' in colsSet:
                        to_insert['saved_at'] = row['saved_at']
                    elif 'transaction_date' in colsSet:
                        to_insert['transaction_date'] = row['saved_at']

                # 3) Fallback: attempt to match remaining keys heuristically
                def find_col(name):
                    candidates = [
                        name,
                        name.lower(),
                        name.replace(' ', '_'),
                        name.replace(' ', '_').lower(),
                        name.replace('.', '_').lower(),
                        name.replace(' ', ''),
                        name.replace(' ', '').lower(),
                    ]
                    for c in candidates:
                        if c in colsSet and c not in to_insert:
                            return c
                    return None

                for k, v in row.items():
                    # skip if already mapped
                    already_mapped = any((colsSet).__contains__(c) and to_insert.get(c) is not None for c in [COLUMN_RENAME_MAP.get(k)])
                    if already_mapped:
                        continue
                    col = find_col(k)
                    if not col:
                        continue
                    if isinstance(v, (dict, list)):
                        to_insert[col] = json.dumps(v)
                    elif isinstance(v, datetime):
                        to_insert[col] = v.isoformat()
                    else:
                        to_insert[col] = v

                if not to_insert:
                    return False

                placeholders = ','.join(['%s'] * len(to_insert))
                col_list = ','.join([f"`{c}`" for c in to_insert.keys()])
                sql = f"INSERT INTO transaction ({col_list}) VALUES ({placeholders})"
                cur.execute(sql, tuple(to_insert.values()))
                conn.commit()
                return True
        finally:
            conn.close()
    except Exception as e:
        print(f"[WARN] Failed to insert into DB: {e}", file=sys.stderr)
        return False

# -----------------------
# load model + encoder safely
# -----------------------
def load_ensemble_models():
    """
    Load ensemble models: CatBoost, XGBoost, Random Forest, and Stacking meta-model.
    Returns tuple: (cat_model, xgb_model, rf_model, stacked_model, encoder)
    """
    cat_model = None
    xgb_model = None
    rf_model = None
    stacked_model = None
    encoder = None
    
    # Load base models
    if os.path.exists(CAT_MODEL_PATH):
        # CatBoost models saved as .cbm must be loaded via the CatBoost API
        cat_model = CatBoostClassifier()
        cat_model.load_model(CAT_MODEL_PATH)
    else:
        raise FileNotFoundError(f"CatBoost model not found at {CAT_MODEL_PATH}")
    
    if os.path.exists(XGB_MODEL_PATH):
        xgb_model = joblib.load(XGB_MODEL_PATH)
    else:
        raise FileNotFoundError(f"XGBoost model not found at {XGB_MODEL_PATH}")
    
    if os.path.exists(RF_MODEL_PATH):
        rf_model = joblib.load(RF_MODEL_PATH)
    else:
        raise FileNotFoundError(f"Random Forest model not found at {RF_MODEL_PATH}")
    
    # Load stacking model
    if os.path.exists(STACKED_MODEL_PATH):
        stacked_model = joblib.load(STACKED_MODEL_PATH)
    else:
        raise FileNotFoundError(f"Stacking model not found at {STACKED_MODEL_PATH}")
    
    # Load categorical encoders
    if os.path.exists(ENCODER_PATH):
        encoder = joblib.load(ENCODER_PATH)
    else:
        encoder = None
    
    return cat_model, xgb_model, rf_model, stacked_model, encoder

# -----------------------
# load stacking configuration (base order and threshold)
# -----------------------
def load_stacking_config():
    """
    Load stacking configuration containing base_model_order and decision_threshold.
    Returns (base_model_order, decision_threshold). Defaults used if file missing.
    """
    base_model_order = ["xgb", "rf", "cat"]
    decision_threshold = 0.5
    try:
        if os.path.exists(STACKING_CONFIG_PATH):
            cfg = joblib.load(STACKING_CONFIG_PATH)
            if isinstance(cfg, dict):
                base_model_order = cfg.get("base_model_order", base_model_order)
                decision_threshold = cfg.get("decision_threshold", decision_threshold)
    except Exception:
        # keep defaults on any error
        pass
    return base_model_order, float(decision_threshold)

# -----------------------
# prepare final X for model
# -----------------------
def prepare_features_for_model(df_row: pd.DataFrame, encoder, expected_order=None):
    """
    df_row: single-row DataFrame containing all raw + engineered features and also binned columns.
    encoder: categorical encoder (e.g., dict of LabelEncoders or ColumnTransformer)
    Returns: pandas DataFrame with encoded features ready for ensemble models
    """
    # Debug: Show log-transformed values for features that will be log-binned
    try:
        log_cols = [c for c in LOG_BIN_FEATURES if c in df_row.columns]
        if log_cols:
            df_logview = df_row.copy()
            for c in log_cols:
                df_logview[c] = np.log1p(pd.to_numeric(df_logview[c], errors='coerce').fillna(0))
            _print_feature_block("FEATURES AFTER LOG TRANSFORMATION (BEFORE BINNING)", df_logview, log_cols)
            # Also print raw vs log1p for quick comparison
            for c in log_cols:
                try:
                    raw_val = pd.to_numeric(df_row[c], errors='coerce').fillna(0).values[0]
                except Exception:
                    raw_val = df_row[c].iloc[0] if hasattr(df_row[c], 'iloc') else df_row[c]
                log_val = df_logview[c].iloc[0]
                print(f"[DEBUG] {c}: raw={raw_val} log1p={log_val}", file=sys.stderr)
        else:
            print("[DEBUG] No log-bin features present in row", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Failed to print log transformation debug: {e}", file=sys.stderr)

    # IMPORTANT: Apply binning with selective log scale for features that require it.
    # For features listed in LOG_BIN_FEATURES, we bin log1p(values) using log1p(edges),
    # otherwise we bin on the raw scale.
    # 1) Apply binning to produce *_binned columns
    binned = apply_bins(df_row, bins_config, LOG_BIN_FEATURES)

    # 1.1) Create alias columns to match model's expected feature names
    # The CatBoost model was trained with PascalCase feature names.
    MODEL_FEATURE_ALIASES = {
        'receiving_currency': 'Receiving Currency',
        'payment_currency': 'Payment Currency',
        'payment_format': 'Payment Format',
        'amount_binned': 'Amount_binned',
        'amount_received_binned': 'Amount Received_binned',
    }
    for src, dst in MODEL_FEATURE_ALIASES.items():
        if src in binned.columns and dst not in binned.columns:
            binned[dst] = binned[src]
    # Additional aliasing to match training expectations
    # CatBoost expects 'beneficiary_unique_senders_binned' but we compute
    # 'beneficiary_unique_senders_at_time_of_txn_binned'. Align if needed.
    if (
        'beneficiary_unique_senders_binned' not in binned.columns
        and 'beneficiary_unique_senders_at_time_of_txn_binned' in binned.columns
    ):
        binned['beneficiary_unique_senders_binned'] = binned['beneficiary_unique_senders_at_time_of_txn_binned']
    
    print(f"\n[DEBUG] Columns after binning: {list(binned.columns)}", file=sys.stderr)
    print(f"[DEBUG] Log-binned features: {LOG_BIN_FEATURES}", file=sys.stderr)

    # Debug: show candidate vector right after binning (pre-encoding)
    try:
        candidate_order_pre = [f for f in (categorical_features + numeric_features) if f in binned.columns]
        _print_feature_block("FEATURES AFTER BINNING (CANDIDATE VECTOR)", binned, candidate_order_pre)
    except Exception as e:
        print(f"[ERROR] Failed to print features after binning: {e}", file=sys.stderr)

    # Debug: per-feature bin diagnostics for key binned features
    try:
        for feat in bins_config.keys():
            if feat in binned.columns and f"{feat}_binned" in binned.columns:
                raw_val = pd.to_numeric(df_row[feat], errors="coerce").fillna(0).values[0]
                log_val = None
                if feat in LOG_BIN_FEATURES:
                    log_val = float(np.log1p(raw_val))
                bval = binned[f"{feat}_binned"].astype("Int64").values[0]
                edges = prepare_bins(bins_config[feat])
                # Edges interpretation: for log-binned features, edges are already in log space
                # Determine the bracket
                bracket_low = None
                bracket_high = None
                if pd.notna(bval):
                    idx = int(bval)
                    if idx >= 1 and idx < len(edges):
                        bracket_low = edges[idx - 1]
                        bracket_high = edges[idx]
                print(
                    f"[BIN] {feat}: raw={raw_val}"
                    + (f" log1p={log_val}" if log_val is not None else "")
                    + f" -> bin={bval} range=({bracket_low}, {bracket_high})",
                    file=sys.stderr,
                )
    except Exception:
        pass

    # 2) select categorical and numeric features as per your list
    cat_feats = [f for f in categorical_features if f in binned.columns]
    num_feats = [f for f in numeric_features if f in binned.columns]
    
    print(f"[DEBUG] Categorical features found: {cat_feats}", file=sys.stderr)
    print(f"[DEBUG] Numeric features found: {num_feats}", file=sys.stderr)

    # 3) Encode categorical features using the loaded encoder
    if encoder is not None and len(cat_feats) > 0:
        try:
            # Split categorical features into string-based and binned (numeric) categoricals
            string_cat_feats = []
            binned_cat_feats = []
            for col in cat_feats:
                if col in binned.columns:
                    if pd.api.types.is_string_dtype(binned[col]) or binned[col].dtype == object:
                        string_cat_feats.append(col)
                    else:
                        binned_cat_feats.append(col)

            # If encoder is a dict of LabelEncoders (one per categorical column)
            if isinstance(encoder, dict):
                # Helper to safely encode values, avoiding -1 by mapping unknowns to a known class
                def safe_encode(le, v, col_name=None):
                    try:
                        if v in le.classes_:
                            return le.transform([v])[0]
                        # Case-insensitive match
                        v_str = str(v).strip()
                        v_l = v_str.lower()
                        classes_list = list(getattr(le, 'classes_', []))
                        classes_lower = [str(c).strip().lower() for c in classes_list]
                        # Exact case-insensitive match
                        for idx, cls_l in enumerate(classes_lower):
                            if cls_l == v_l:
                                return le.transform([classes_list[idx]])[0]
                        # Currency-specific normalization: try mapping codes to full names present in classes
                        currency_synonyms = {
                            'usd': ['us dollar', 'u.s. dollar', 'united states dollar'],
                            'eur': ['euro'],
                            'inr': ['indian rupee', 'rupee'],
                            'gbp': ['british pound', 'pound', 'sterling'],
                        }
                        # If column is a currency field, expand search
                        if col_name in ('Receiving Currency', 'Payment Currency', 'receiving_currency', 'payment_currency'):
                            key = v_l
                            # Accept 3-letter codes by lower
                            if len(v_str) == 3 and v_str.isalpha():
                                key = v_l
                            # Search exact synonyms first
                            for code, syns in currency_synonyms.items():
                                if v_l == code or v_l in syns:
                                    # find any class containing any synonym token
                                    search_terms = [code] + syns
                                    for i, cls_l in enumerate(classes_lower):
                                        if any(term in cls_l for term in search_terms):
                                            return le.transform([classes_list[i]])[0]
                            # Fallback: substring match of provided token within classes
                            for i, cls_l in enumerate(classes_lower):
                                if v_l in cls_l:
                                    return le.transform([classes_list[i]])[0]
                        # Generic fallback: map common words to codes if codes exist in classes
                        mapping = {
                            'euro': 'EUR', 'usd': 'USD', 'us dollar': 'USD', 'inr': 'INR', 'rupee': 'INR', 'gbp': 'GBP', 'pound': 'GBP',
                        }
                        if v_l in mapping and mapping[v_l] in classes_list:
                            return le.transform([mapping[v_l]])[0]
                        # Prefer 'Unknown' class if available
                        for i, cls_l in enumerate(classes_lower):
                            if cls_l == 'unknown':
                                return le.transform([classes_list[i]])[0]
                        # Fallback to first class to avoid -1
                        return le.transform([classes_list[0]])[0]
                    except Exception:
                        # As a last resort, return 0
                        return 0

                for col in string_cat_feats:
                    if col in encoder and col in binned.columns:
                        le = encoder[col]
                        binned[col] = binned[col].map(lambda x: safe_encode(le, x, col))
                # For binned categorical features, ensure they are numeric codes (no encoding)
                for col in binned_cat_feats:
                    binned[col] = pd.to_numeric(binned[col], errors='coerce').fillna(1)
            else:
                # If encoder is a ColumnTransformer or similar, avoid transforming here to prevent -1s
                # Keep string categoricals by mapping to clean, standardized values
                for col in string_cat_feats:
                    binned[col] = binned[col].astype(str).str.strip()
                # Ensure binned categoricals are numeric
                for col in binned_cat_feats:
                    binned[col] = pd.to_numeric(binned[col], errors='coerce').fillna(1)
        except Exception as e:
            # Fallback: keep binned categoricals numeric and clean string categoricals
            for col in cat_feats:
                if col in binned.columns:
                    if pd.api.types.is_string_dtype(binned[col]) or binned[col].dtype == object:
                        binned[col] = binned[col].astype(str).str.strip()
                    else:
                        binned[col] = pd.to_numeric(binned[col], errors='coerce').fillna(1)

    # 4) Fill missing values intelligently
    # For numeric features, use 0 as default only if needed
    # Most features should already be calculated
    for col in num_feats:
        if col in binned.columns:
            # Ensure numeric dtype (avoid object strings like "54")
            binned[col] = pd.to_numeric(binned[col], errors='coerce')
            # Only fill if there are actual NaN values
            if binned[col].isna().any():
                # For ratio/percentage features, 0 is meaningful
                if 'ratio' in col.lower() or 'percent' in col.lower():
                    binned[col] = binned[col].fillna(0)
                # For count features, 0 is meaningful
                elif 'count' in col.lower() or 'num' in col.lower():
                    binned[col] = binned[col].fillna(0)
                # For amount/total features, 0 is meaningful
                elif 'amount' in col.lower() or 'total' in col.lower():
                    binned[col] = binned[col].fillna(0)
                # For other numeric features, use median or 0
                else:
                    binned[col] = binned[col].fillna(0)
    
    # For categorical features, fill with sensible defaults (avoid -1)
    for col in cat_feats:
        if col in binned.columns and binned[col].isna().any():
            if pd.api.types.is_string_dtype(binned[col]) or binned[col].dtype == object:
                # Fill strings with 'Unknown'
                binned[col] = binned[col].fillna('Unknown')
            else:
                # For binned numeric categories, use first bin (1)
                binned[col] = binned[col].fillna(1)

    # Debug: print features after categorical encoding (candidate vector)
    try:
        candidate_order = [f for f in (categorical_features + numeric_features) if f in binned.columns]
        _print_feature_block("FEATURES AFTER CATEGORICAL ENCODING (CANDIDATE VECTOR)", binned, candidate_order)
    except Exception as e:
        print(f"[ERROR] Failed to print features after encoding: {e}", file=sys.stderr)

    # Ensure specific known numeric flags are numeric
    for hard_num in ["kyc_score", "is_pep"]:
        if hard_num in binned.columns:
            binned[hard_num] = pd.to_numeric(binned[hard_num], errors='coerce').fillna(0)

    # 5) Select final feature set in the EXACT order the model expects
    # If an explicit expected_order is provided, use it; otherwise default to configured lists
    if not expected_order:
        expected_order = categorical_features + numeric_features
    
    # Only include features that exist in the binned dataframe
    final_feats = [f for f in expected_order if f in binned.columns]
    X_final = binned[final_feats]
    
    print(f"\n[DEBUG] Final feature order for model: {final_feats}", file=sys.stderr)
    print(f"[DEBUG] Feature count: {len(final_feats)} features", file=sys.stderr)

    # Print any missing features compared to expected_order
    missing = [f for f in expected_order if f not in binned.columns]
    if missing:
        print(f"[WARN] Missing features (not present after engineering/binning): {missing}", file=sys.stderr)

    return X_final


def ensemble_predict(X, cat_model, xgb_model, rf_model, stacked_model, base_order=None, decision_threshold: float = 0.5):
    """
    Perform ensemble prediction using base models with simple threshold rule.
    X: prepared features (DataFrame or numpy array)
    Returns: (prediction, confidence, base_predictions)
    
    Rule: If ANY base model gives probability > 0.3, flag as suspicious (return 1)
    """
    # Get predictions from base models
    try:
        # CatBoost prediction
        if hasattr(cat_model, "predict_proba"):
            cat_proba = cat_model.predict_proba(X)
            cat_pred = cat_proba[:, 1] if cat_proba.shape[1] > 1 else cat_proba[:, 0]
        else:
            cat_pred = cat_model.predict(X)
        try:
            cat_val = float(cat_pred[0]) if hasattr(cat_pred, "__len__") else float(cat_pred)
            print(f"[BASE] CatBoost score: {cat_val}", file=sys.stderr)
        except Exception:
            pass
        
        # XGBoost prediction
        if hasattr(xgb_model, "predict_proba"):
            xgb_proba = xgb_model.predict_proba(X)
            xgb_pred = xgb_proba[:, 1] if xgb_proba.shape[1] > 1 else xgb_proba[:, 0]
        else:
            xgb_pred = xgb_model.predict(X)
        try:
            xgb_val = float(xgb_pred[0]) if hasattr(xgb_pred, "__len__") else float(xgb_pred)
            print(f"[BASE] XGBoost score: {xgb_val}", file=sys.stderr)
        except Exception:
            pass
        
        # Random Forest prediction
        if hasattr(rf_model, "predict_proba"):
            rf_proba = rf_model.predict_proba(X)
            rf_pred = rf_proba[:, 1] if rf_proba.shape[1] > 1 else rf_proba[:, 0]
        else:
            rf_pred = rf_model.predict(X)
        try:
            rf_val = float(rf_pred[0]) if hasattr(rf_pred, "__len__") else float(rf_pred)
            print(f"[BASE] RandomForest score: {rf_val}", file=sys.stderr)
        except Exception:
            pass
        
        # Extract individual scores
        cat_score = float(cat_pred[0]) if hasattr(cat_pred, "__len__") else float(cat_pred)
        xgb_score = float(xgb_pred[0]) if hasattr(xgb_pred, "__len__") else float(xgb_pred)
        rf_score = float(rf_pred[0]) if hasattr(rf_pred, "__len__") else float(rf_pred)
        
        print(f"[DEBUG] Base model predictions: CatBoost={cat_score:.4f}, XGBoost={xgb_score:.4f}, RandomForest={rf_score:.4f}", file=sys.stderr)
        
        # Simple threshold rule: If ANY model > 0.3, flag as suspicious
        THRESHOLD = 0.3
        max_score = max(cat_score, xgb_score, rf_score)
        final_pred = 1 if max_score > THRESHOLD else 0
        confidence = max_score
        
        print(f"[DECISION] Max score: {max_score:.4f}, Threshold: {THRESHOLD}, Prediction: {final_pred}", file=sys.stderr)
        
        # Stack base predictions for return value (for compatibility)
        base_map = {"cat": cat_pred, "xgb": xgb_pred, "rf": rf_pred}
        order = base_order or ["xgb", "rf", "cat"]
        base_stack = []
        for name in order:
            if name not in base_map:
                raise RuntimeError(f"Unknown base model in stacking order: {name}")
            base_stack.append(base_map[name])
        base_predictions = np.column_stack(base_stack)
        
        return final_pred, confidence, base_predictions[0]
    
    except Exception as e:
        raise RuntimeError(f"Ensemble prediction failed: {e}")

# -----------------------
# top-level: read stdin JSON, process, predict, print output
# -----------------------
def main():
    print("=== PREDICT.PY VERSION: 2024-12-16-FIX ===", file=sys.stderr)
    print(f"Python executable: {sys.executable}", file=sys.stderr)
    print(f"Working directory: {os.getcwd()}", file=sys.stderr)
    
    raw = sys.stdin.read()
    if not raw:
        print(json.dumps({"error": "no input received"}))
        return

    try:
        input_json = json.loads(raw)
    except Exception as e:
        print(json.dumps({"error": f"invalid json input: {e}"}))
        return

    # 1) Normalize legacy PascalCase keys and trim string values
    LEGACY_KEY_MAP = {
        "From Bank": "from_bank",
        "Account": "account",
        "To Bank": "to_bank_txn",
        "Account.1": "account_1",
        "Amount Received": "amount_received",
        "Receiving Currency": "receiving_currency",
        "Amount": "amount",
        "Payment Currency": "payment_currency",
        "Payment Format": "payment_format",
    }

    normalized_input = {}
    for k, v in input_json.items():
        target_key = LEGACY_KEY_MAP.get(k, k)
        # Trim string values
        if isinstance(v, str):
            v = v.strip()
        normalized_input[target_key] = v

    # 1.5) Map frontend UI keys -> model/db column names (FEATURE_MAP)
    mapped = {}
    for ui_key, value in normalized_input.items():
        if ui_key in FEATURE_MAP:
            mapped_key = FEATURE_MAP[ui_key]
            mapped[mapped_key] = value
        else:
            # keep unknown keys as-is (useful if some optional fields already named as model columns)
            mapped[ui_key] = value

    # Ensure critical identifiers are stripped of whitespace
    for id_key in ("account", "account_1"):
        if id_key in mapped and mapped[id_key] is not None:
            mapped[id_key] = str(mapped[id_key]).strip()

    # Normalize booleans/flags and common categoricals early
    # is_pep: accept diverse inputs like "Yes"/"No", "true"/"false", 1/0
    if 'is_pep' in mapped:
        v = mapped['is_pep']
        try:
            # try numeric
            mapped['is_pep'] = int(float(v))
        except Exception:
            s = str(v).strip().lower()
            mapped['is_pep'] = 1 if s in ('yes', 'true', '1', 'y', 't') else 0

    # Light normalization for currency strings (upper-case codes)
    for cur_key in ('receiving_currency', 'payment_currency'):
        if cur_key in mapped and isinstance(mapped[cur_key], str):
            s = mapped[cur_key].strip()
            if len(s) == 3 and s.isalpha():
                mapped[cur_key] = s.upper()
            else:
                mapped[cur_key] = s

    # 2) Compute engineered features
    print("\n=== INPUT FEATURES ===", file=sys.stderr)
    for key, value in sorted(mapped.items()):
        print(f"{key}: {value}", file=sys.stderr)
    
    mapped = compute_engineered_features(mapped) or mapped
    
    print("\n=== ALL COMPUTED FEATURES ===", file=sys.stderr)
    for key, value in sorted(mapped.items()):
        print(f"{key}: {value}", file=sys.stderr)

    # NOTE: defer saving to CSV until after prediction so we can include
    # prediction, confidence and key_factors in the saved row.

    # 3) Convert single dict to DataFrame row (columns = keys)
    df_row = pd.DataFrame([mapped])


    # 4) Load ensemble models + encoder
    try:
        cat_model, xgb_model, rf_model, stacked_model, encoder = load_ensemble_models()
    except Exception as e:
        tb = traceback.format_exc()
        print(json.dumps({"error": f"failed to load ensemble models: {e}", "trace": tb}))
        return
    # 4.1) Load stacking configuration
    base_order, decision_threshold = load_stacking_config()

    # 5) Prepare final X for models
    try:
        # Determine expected orders from CatBoost and XGBoost if available
        expected_cat = None
        expected_xgb = None
        if hasattr(cat_model, 'feature_names_') and cat_model.feature_names_:
            expected_cat = list(cat_model.feature_names_)
        if hasattr(xgb_model, 'feature_names_') and xgb_model.feature_names_:
            expected_xgb = list(xgb_model.feature_names_)
        # Fallback if XGBoost exposes via booster
        if expected_xgb is None and hasattr(xgb_model, 'get_booster'):
            try:
                booster = xgb_model.get_booster()
                names = getattr(booster, 'feature_names', None)
                if names:
                    expected_xgb = list(names)
            except Exception:
                pass

        # Prefer CatBoost order if present, else XGBoost, else configured list
        expected_order = expected_cat or expected_xgb or (categorical_features + numeric_features)

        # Prepare features with unified expected order for all models
        X = prepare_features_for_model(df_row, encoder, expected_order)
        print("\n=== FINAL FEATURES FOR MODEL ===", file=sys.stderr)
        print(f"Feature columns: {list(X.columns)}", file=sys.stderr)
        print(f"Feature values: {X.values[0].tolist()}", file=sys.stderr)
    except Exception as e:
        tb = traceback.format_exc()
        print(json.dumps({"error": f"failed to prepare features: {e}", "trace": tb}))
        return

    # Optionally save binned features to DB now (after log-preview & unified binning)
    if SAVE_TO_DB:
        try:
            df_binned = apply_bins(df_row, bins_config, LOG_BIN_FEATURES)
            save_row_binned = df_binned.iloc[0].to_dict()
            inserted_binned = insert_prediction_into_db(save_row_binned)
            if inserted_binned:
                print("[INFO] Binned features saved to database", file=sys.stderr)
            else:
                print("[WARN] Failed to save binned features to database", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Error saving binned features: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    else:
        print("[INFO] SAVE_TO_DB=false: Skipping DB save of binned features", file=sys.stderr)

    # 6) Ensemble Predict (4 base models + stacking)
    try:
        prediction, confidence, base_preds = ensemble_predict(X, cat_model, xgb_model, rf_model, stacked_model, base_order, decision_threshold)
        prediction = str(prediction)  # Convert to string for consistent output
    except Exception as e:
        tb = traceback.format_exc()
        print(json.dumps({"error": f"ensemble prediction failed: {e}", "trace": tb}))
        return

    # Debug: print features (as used) after obtaining prediction
    try:
        _print_feature_block("FEATURES USED FOR PREDICTION (POST-PREDICTION ECHO)", X, list(X.columns))
        print(f"[DEBUG] Prediction: {prediction}", file=sys.stderr)
        if 'confidence' in locals() and confidence is not None:
            print(f"[DEBUG] Confidence: {confidence}", file=sys.stderr)
        try:
            # base_preds is a 1D array of base model outputs if available
            if base_preds is not None:
                print(f"[DEBUG] Base model preds [xgb, rf, cat]: {base_preds.tolist()}", file=sys.stderr)
                try:
                    names = ["xgb", "rf", "cat"]
                    for i, n in enumerate(names):
                        if i < len(base_preds):
                            print(f"[DEBUG] {n} score: {float(base_preds[i])}", file=sys.stderr)
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass

    # 7) Optionally extract simple key_factors (placeholder)
    key_factors = []
    try:
        # Simple heuristic: flag amount_to_income_ratio large, or low kyc_score
        atir = mapped.get("amount_to_income_ratio", None)
        if atir is not None and not pd.isna(atir) and atir > 3:
            key_factors.append("High amount-to-income ratio")
        kyc = mapped.get("kyc_score", None)
        if kyc is not None and not pd.isna(kyc) and float(kyc) < 40:
            key_factors.append("Low KYC score")
    except Exception:
        pass

    # 8) Save prediction results to database (disabled unless SAVE_TO_DB=true)
    if SAVE_TO_DB:
        try:
            save_row_prediction = {
                "is_laundering": prediction,
                "prediction": prediction,
                "confidence": confidence if 'confidence' in locals() else None,
                "key_factors": key_factors,
                "saved_at": datetime.now().isoformat()
            }
            print("\n=== SAVING PREDICTION RESULTS TO DATABASE ===", file=sys.stderr)
            inserted_prediction = insert_prediction_into_db(save_row_prediction)
            if inserted_prediction:
                print("[INFO] Prediction results saved to database", file=sys.stderr)
            else:
                print("[WARN] Failed to save prediction results to database", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Failed to save prediction results: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    else:
        print("[INFO] SAVE_TO_DB=false: Skipping DB save of prediction results", file=sys.stderr)

    # 9) Output JSON
    out = {
        "prediction": prediction,
        "confidence": confidence if confidence is not None else None,
        "key_factors": key_factors
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
