"""
Generates synthetic prediction traffic against a running instance of the
Fraud Detection API, for demo/portfolio purposes -- so there's non-trivial
data in logs/predictions.jsonl to show Evidently drift monitoring working
against, without waiting for real production traffic.

DEMO DATA ONLY. Every request sent by this script is tagged with the
X-Prediction-Source: synthetic header, which the API logs into each
record's "source" field. This lets any later analysis (Evidently reports,
manual review) filter this traffic out and treat it as demonstration data,
not a real production distribution. See README.md for the same caveat
stated for anyone reading the repo.

Feature values are sampled from real rows in the training dataset (not
randomly generated) so the traffic reflects realistic feature
combinations and correlations, rather than statistically implausible
random noise.

SIMULATED GROUND TRUTH: because each sampled row comes from the labeled
training dataset, its real isFraud label is known. This script writes
that label to logs/simulated_ground_truth.jsonl, keyed by a request_id
sent as the X-Request-Id header, so scripts/check_performance_decay.py
can later join predictions against outcomes. Real production traffic has
no equivalent -- actual fraud outcomes arrive weeks later via chargebacks,
if they arrive in an automatable form at all. This file exists ONLY to
demonstrate what a decay-check pipeline would do once real labeled
outcomes exist; it is not a substitute for one.

Usage:
    python scripts/generate_synthetic_traffic.py --n 200 --api-url http://127.0.0.1:8000
"""

import argparse
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["TransactionAmt", "C13", "C1", "C14"]
CATEGORICAL_FEATURES = ["card4", "card6", "P_emaildomain"]
TARGET = "isFraud"
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

GROUND_TRUTH_PATH = Path("logs/simulated_ground_truth.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic demo traffic for the fraud API")
    parser.add_argument("--data-path", type=Path, default=Path("ieee_fraud_detection.parquet"))
    parser.add_argument("--n", type=int, default=200, help="Number of synthetic requests to send")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--delay", type=float, default=0.02, help="Seconds between requests")
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}. This script samples "
                                 f"feature combinations from real data rather than inventing them.")

    df = pd.read_parquet(args.data_path, columns=FEATURES + [TARGET])
    sample = df.sample(n=min(args.n, len(df)), random_state=None).reset_index(drop=True)

    endpoint = f"{args.api_url.rstrip('/')}/predict"
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)

    sent, failed = 0, 0
    with open(GROUND_TRUTH_PATH, "a") as gt_file:
        for _, row in sample.iterrows():
            payload = {
                "TransactionAmt": float(row["TransactionAmt"]) if pd.notna(row["TransactionAmt"]) else 0.0,
                "P_emaildomain": row["P_emaildomain"] if pd.notna(row["P_emaildomain"]) else "missing",
                "C1": float(row["C1"]) if pd.notna(row["C1"]) else 0.0,
                "C13": float(row["C13"]) if pd.notna(row["C13"]) else 0.0,
                "C14": float(row["C14"]) if pd.notna(row["C14"]) else 0.0,
                "card4": row["card4"] if pd.notna(row["card4"]) else "missing",
                "card6": row["card6"] if pd.notna(row["card6"]) else "missing",
            }
            request_id = str(uuid.uuid4())
            headers = {"X-Prediction-Source": "synthetic", "X-Request-Id": request_id}
            try:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=5)
                resp.raise_for_status()
                sent += 1
                gt_record = {
                    "request_id": request_id,
                    "true_label": int(row[TARGET]),
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source": "simulated",
                }
                gt_file.write(json.dumps(gt_record) + "\n")
            except requests.RequestException as e:
                failed += 1
                logger.warning(f"Request failed: {e}")
            time.sleep(args.delay)

    logger.info(f"Done. Sent: {sent}, Failed: {failed}. "
                f"Check logs/predictions.jsonl for records with \"source\": \"synthetic\", "
                f"and {GROUND_TRUTH_PATH} for their simulated true labels.")
    if failed > 0:
        logger.warning(f"{failed} requests failed -- is the API running at {args.api_url}?")


if __name__ == "__main__":
    main()
