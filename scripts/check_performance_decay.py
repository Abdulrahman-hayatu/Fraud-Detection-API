"""
Detects performance decay by comparing metrics computed on recent
predictions (joined against their true labels) to the currently promoted
champion model's recorded training metrics.

GROUND TRUTH LIMITATION: this currently only has ground truth for
synthetic traffic (see scripts/generate_synthetic_traffic.py), because
real fraud outcomes require a feedback loop (chargebacks, manual review)
that doesn't exist in this project there's no real customer base
generating labeled outcomes. This script is structurally correct and
would work identically once real labeled outcomes exist; until then, it
demonstrates the pattern against simulated labels rather than measuring
anything about actual production behavior. Every output makes this
explicit rather than presenting synthetic derived "decay" as real.

Decay is flagged if:
  recall drops by more than --recall-tolerance (absolute) vs. the champion, OR
  roc_auc drops by more than --auc-tolerance (absolute) vs. the champion.
These are configurable, not hidden defaults -- see --help.

Usage:
    python scripts/check_performance_decay.py
"""

import argparse
import json
import logging
from pathlib import Path

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def join_predictions_with_ground_truth(predictions: list, ground_truth: list) -> list:
    gt_by_id = {r["request_id"]: r for r in ground_truth if r.get("request_id")}
    joined = []
    unmatched = 0
    for pred in predictions:
        rid = pred.get("request_id")
        gt = gt_by_id.get(rid)
        if gt is None:
            unmatched += 1
            continue
        joined.append({
            "y_true": gt["true_label"],
            "y_pred": int(pred["output"]["is_fraud"]),
            "y_proba": pred["output"]["fraud_probability"],
        })
    if unmatched:
        logger.info(f"{unmatched} predictions had no matching ground truth record (expected for "
                    f"live traffic, which has no simulated labels).")
    return joined


def main():
    parser = argparse.ArgumentParser(description="Check for performance decay against the champion model")
    parser.add_argument("--predictions-path", type=Path, default=Path("logs/predictions.jsonl"))
    parser.add_argument("--ground-truth-path", type=Path, default=Path("logs/simulated_ground_truth.jsonl"))
    parser.add_argument(
        "--champion-metadata-path", type=Path,
        default=Path("Fraud_Detection/models/model_metadata.json"),
    )
    parser.add_argument("--output-path", type=Path, default=Path("reports/decay_check.json"))
    parser.add_argument("--recall-tolerance", type=float, default=0.05,
                         help="Absolute recall drop vs. champion that triggers a decay flag")
    parser.add_argument("--auc-tolerance", type=float, default=0.03,
                         help="Absolute ROC-AUC drop vs. champion that triggers a decay flag")
    parser.add_argument("--min-samples", type=int, default=200,
                         help="Minimum joined samples required to compute reliable metrics. "
                              "Set higher than you might expect: ROC-AUC is noisy on small samples, and "
                              "testing this script with 200 samples against a model compared to "
                              "itself produced a 0.037 AUC swing from pure sampling variance, enough "
                              "to trip the default 0.03 tolerance with zero real degradation. If you "
                              "lower this, also consider loosening --auc-tolerance accordingly.")
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.champion_metadata_path.exists():
        raise FileNotFoundError(
            f"No champion model metadata at {args.champion_metadata_path}. "
            f"Nothing to compare against. Has a model ever been promoted?"
        )
    with open(args.champion_metadata_path) as f:
        champion = json.load(f)
    champion_metrics = champion["metrics"]

    predictions = load_jsonl(args.predictions_path)
    ground_truth = load_jsonl(args.ground_truth_path)
    joined = join_predictions_with_ground_truth(predictions, ground_truth)

    result = {
        "champion_run_id": champion.get("run_id"),
        "n_joined_samples": len(joined),
        "ground_truth_note": "SIMULATED labels only (synthetic traffic). Not a measurement of real "
                              "production performance. See script docstring.",
    }

    if len(joined) < args.min_samples:
        result["decay_detected"] = False
        result["status"] = "insufficient_data"
        result["message"] = (
            f"Only {len(joined)} joined samples (minimum: {args.min_samples}). "
            f"Not enough labeled data to reliably assess decay."
        )
        with open(args.output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.warning(result["message"])
        print(json.dumps(result, indent=2))
        return

    y_true = [r["y_true"] for r in joined]
    y_pred = [r["y_pred"] for r in joined]
    y_proba = [r["y_proba"] for r in joined]

    current_metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else None,
    }

    recall_drop = champion_metrics["recall"] - current_metrics["recall"]
    auc_drop = (
        champion_metrics["roc_auc"] - current_metrics["roc_auc"]
        if current_metrics["roc_auc"] is not None else 0.0
    )

    decay_detected = (recall_drop > args.recall_tolerance) or (auc_drop > args.auc_tolerance)

    result.update({
        "decay_detected": decay_detected,
        "status": "checked",
        "champion_metrics": {
            "recall": champion_metrics["recall"],
            "roc_auc": champion_metrics["roc_auc"],
        },
        "current_metrics": current_metrics,
        "recall_drop": recall_drop,
        "auc_drop": auc_drop,
        "thresholds": {
            "recall_tolerance": args.recall_tolerance,
            "auc_tolerance": args.auc_tolerance,
        },
    })

    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Decay check: {'DECAY DETECTED' if decay_detected else 'no decay'} "
                f"(recall_drop={recall_drop:.4f}, auc_drop={auc_drop:.4f})")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
