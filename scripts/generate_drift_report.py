"""
Generates an Evidently drift report comparing:
  - Reference: the held-out test split used at training time, with the
    served model's predicted fraud_probability attached.
  - Current: production prediction logs (logs/predictions.jsonl).

Covers BOTH input feature drift (TransactionAmt, C1, C13, C14, card4,
card6, P_emaildomain) and model OUTPUT drift (fraud_probability). Output
drift matters separately from input drift: the model's score distribution
can shift even when individual input features look stable, if the
*combination* of inputs it's seeing has changed.

By default this filters logs to source == "live" only, since synthetic
demo traffic (see scripts/generate_synthetic_traffic.py) is not real
production behavior and would give a misleading drift signal if mixed
in. Use --include-synthetic to override this for demo purposes -- doing
so is flagged loudly in the report filename and console output so nobody
mistakes a demo run for a real production drift check.

Usage:
    python scripts/generate_drift_report.py --data-path ieee_fraud_detection.parquet
    python scripts/generate_drift_report.py --include-synthetic   # demo only
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["TransactionAmt", "C13", "C1", "C14"]
CATEGORICAL_FEATURES = ["card4", "card6", "P_emaildomain"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "isFraud"
RANDOM_STATE = 42  # must match train.py's split, so "reference" is the same held-out data


def build_reference(data_path: Path, model_path: Path, test_size: float) -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    X = df[FEATURES].copy()
    y = df[TARGET]
    _, X_test, _, _ = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
    X_test[CATEGORICAL_FEATURES] = X_test[CATEGORICAL_FEATURES].fillna("missing")

    model = joblib.load(model_path)
    X_test = X_test.copy()
    X_test["fraud_probability"] = model.predict_proba(X_test)[:, 1]
    return X_test


def build_current(log_path: Path, include_synthetic: bool) -> pd.DataFrame:
    if not log_path.exists():
        raise FileNotFoundError(
            f"No prediction log found at {log_path}. The API needs to have served "
            f"at least some requests before a drift report is possible."
        )
    allowed_sources = {"live", "synthetic"} if include_synthetic else {"live"}
    records = []
    skipped = 0
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("source") not in allowed_sources:
                skipped += 1
                continue
            row = dict(rec["input"])
            row["fraud_probability"] = rec["output"]["fraud_probability"]
            records.append(row)

    if skipped:
        logger.info(f"Excluded {skipped} log rows outside allowed sources ({allowed_sources}).")
    if not records:
        raise ValueError(
            f"No rows matched allowed sources {allowed_sources} in {log_path}. "
            f"Nothing to compare against reference data."
        )
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate a drift report comparing training vs. production data")
    parser.add_argument("--data-path", type=Path, default=Path("ieee_fraud_detection.parquet"))
    parser.add_argument(
        "--model-path", type=Path,
        default=Path("Fraud_Detection/models/Fraud_Detection_Pipeline.pkl"),
    )
    parser.add_argument("--log-path", type=Path, default=Path("logs/predictions.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--test-size", type=float, default=0.2, help="Must match train.py's --test-size")
    parser.add_argument(
        "--include-synthetic", action="store_true",
        help="Include synthetic demo traffic in the drift comparison. NOT representative of "
             "real production behavior -- use only to demonstrate the report format.",
    )
    parser.add_argument(
        "--min-current-rows", type=int, default=30,
        help="Minimum rows required in current data. Below this, Evidently's drift "
             "calculations can be statistically meaningless or crash outright.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building reference data (held-out training split + model output)...")
    reference = build_reference(args.data_path, args.model_path, args.test_size)
    logger.info(f"Reference: {len(reference)} rows")

    logger.info(f"Building current data from {args.log_path} (include_synthetic={args.include_synthetic})...")
    current = build_current(args.log_path, args.include_synthetic)
    logger.info(f"Current: {len(current)} rows")

    if len(current) < args.min_current_rows:
        source_hint = (
            "Not enough live traffic yet."
            if not args.include_synthetic
            else "Not enough rows even with synthetic traffic included -- check logs/predictions.jsonl."
        )
        raise SystemExit(
            f"Only {len(current)} current-data rows available (minimum: {args.min_current_rows}). "
            f"{source_hint} Drift statistics are unreliable or will error out below this size. "
            f"{'Try --include-synthetic to demo the report format.' if not args.include_synthetic else ''}"
        )

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "DEMO-with-synthetic" if args.include_synthetic else "live-only"
    html_path = args.output_dir / f"drift_report_{tag}_{timestamp}.html"
    result.save_html(str(html_path))
    logger.info(f"Saved drift report to {html_path}")

    if args.include_synthetic:
        logger.warning(
            "This report includes SYNTHETIC demo traffic and does NOT represent real "
            "production drift. Do not use it to make actual monitoring decisions."
        )

    print(f"\nDone. Report: {html_path}")


if __name__ == "__main__":
    main()
