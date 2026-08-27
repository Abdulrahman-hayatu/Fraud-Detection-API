"""
Champion-vs-challenger promotion gate.

Compares a newly trained ("challenger") model's metrics against the
currently-promoted ("champion") model's recorded metrics. Promotes the
challenger ONLY if it meets or beats the champion on BOTH:
  - recall (the priority metric per the team's recall-over-precision
    decision -- see app/config.py)
  - roc_auc (a guard metric, independent of the decision threshold)

Why both, not just recall: recall alone is gameable. A degenerate model
that flags every transaction as fraud gets recall=1.0 trivially, while
being useless in production. ROC-AUC is threshold-independent and
reflects the model's actual ranking ability, so a model can't pass the
gate just by moving its threshold or predicting positive indiscriminately.

Both metrics are read at the fixed DECISION_THRESHOLD (0.50) operating
point recorded in each run's metrics.json -- not the optimal_threshold_
diagnostic, which is informational only and not what's actually served.

NOTE ON CODE DUPLICATION: this script reimplements the small amount of
promotion logic (copy model, write model_metadata.json, dvc add) that
also appears in train.py's --promote path, rather than importing it as a
shared module. This project's scripts have consistently been kept
self-contained (see the repeated NUMERIC_FEATURES/CATEGORICAL_FEATURES
definitions across train.py and the scripts/ files) rather than
introducing a shared-import convention -- this follows that existing
pattern rather than a decision that duplication here is free.

Usage:
    python scripts/promotion_gate.py --challenger-run-id 20260827T103803Z
"""

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_metrics(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{label} metrics not found at {path}")
    with open(path) as f:
        return json.load(f)


def promote(challenger_pickle_path: Path, run_id: str, metrics: dict, canonical_dir: Path) -> Path:
    """Copies the challenger model to the canonical served path and writes
    its metadata. Mirrors train.py's --promote logic (see module docstring
    for why this isn't a shared import)."""
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = canonical_dir / "Fraud_Detection_Pipeline.pkl"

    model = joblib.load(challenger_pickle_path)
    joblib.dump(model, canonical_path)

    metadata_path = canonical_path.with_name("model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "promoted_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "promoted_by": "promotion_gate.py (automated champion/challenger comparison)",
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    try:
        subprocess.run(["dvc", "add", str(canonical_path)], check=True, capture_output=True)
        logger.info(f"Updated DVC pointer for {canonical_path}.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"Could not auto-update DVC tracking ({e}). "
                        f"Run 'dvc add {canonical_path}' manually before committing.")

    return canonical_path


def main():
    parser = argparse.ArgumentParser(description="Champion-vs-challenger promotion gate")
    parser.add_argument("--challenger-run-id", required=True,
                         help="Run ID of the newly trained model (the directory name under models/)")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--champion-metadata-path", type=Path,
                         default=Path("Fraud_Detection/models/model_metadata.json"))
    parser.add_argument("--canonical-dir", type=Path, default=Path("Fraud_Detection/models"))
    parser.add_argument("--decision-output-path", type=Path, default=Path("reports/promotion_decision.json"))
    args = parser.parse_args()

    args.decision_output_path.parent.mkdir(parents=True, exist_ok=True)

    challenger_dir = args.models_dir / args.challenger_run_id
    challenger_metrics_path = challenger_dir / "metrics.json"
    challenger_pickle_path = challenger_dir / "pipeline.pkl"

    if not challenger_pickle_path.exists():
        raise FileNotFoundError(f"Challenger model artifact not found at {challenger_pickle_path}")

    challenger_full = load_metrics(challenger_metrics_path, "Challenger")
    challenger_metrics = challenger_full["metrics"]

    decision = {
        "challenger_run_id": args.challenger_run_id,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not args.champion_metadata_path.exists():
        # Bootstrap case: nothing is currently promoted. First model always
        # gets promoted -- there's no champion to compare against.
        logger.info("No champion currently promoted. Promoting challenger as the first model.")
        canonical_path = promote(challenger_pickle_path, args.challenger_run_id, challenger_metrics, args.canonical_dir)
        decision.update({
            "promoted": True,
            "reason": "bootstrap: no champion existed yet",
            "canonical_path": str(canonical_path),
        })
        with open(args.decision_output_path, "w") as f:
            json.dump(decision, f, indent=2)
        print(json.dumps(decision, indent=2))
        return

    champion = load_metrics(args.champion_metadata_path, "Champion")
    champion_metrics = champion["metrics"]

    challenger_recall = challenger_metrics["recall"]
    challenger_auc = challenger_metrics["roc_auc"]
    champion_recall = champion_metrics["recall"]
    champion_auc = champion_metrics["roc_auc"]

    recall_ok = challenger_recall >= champion_recall
    auc_ok = challenger_auc >= champion_auc
    should_promote = recall_ok and auc_ok

    decision.update({
        "champion_run_id": champion.get("run_id"),
        "champion_metrics": {"recall": champion_recall, "roc_auc": champion_auc},
        "challenger_metrics": {"recall": challenger_recall, "roc_auc": challenger_auc},
        "recall_check_passed": recall_ok,
        "auc_check_passed": auc_ok,
        "promoted": should_promote,
    })

    if should_promote:
        logger.info(f"Challenger {args.challenger_run_id} meets or beats champion "
                     f"{champion.get('run_id')} on both recall and ROC-AUC. Promoting.")
        canonical_path = promote(challenger_pickle_path, args.challenger_run_id, challenger_metrics, args.canonical_dir)
        decision["canonical_path"] = str(canonical_path)
        decision["reason"] = "challenger met or beat champion on recall and roc_auc"
    else:
        failed = []
        if not recall_ok:
            failed.append(f"recall {challenger_recall:.4f} < champion {champion_recall:.4f}")
        if not auc_ok:
            failed.append(f"roc_auc {challenger_auc:.4f} < champion {champion_auc:.4f}")
        reason = "; ".join(failed)
        logger.info(f"Challenger {args.challenger_run_id} did NOT beat champion "
                     f"{champion.get('run_id')}: {reason}. No promotion.")
        decision["reason"] = reason

    with open(args.decision_output_path, "w") as f:
        json.dump(decision, f, indent=2)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
