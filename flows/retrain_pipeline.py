"""
Prefect orchestration for the retrain pipeline.

Retrain trigger policy (as decided): retrain if EITHER data/concept drift
is detected OR performance decay is detected. No data-volume trigger --
there's no real customer base generating the volume that would justify one.

Each stage below calls an already-independently-tested script via
subprocess and reads its JSON output, rather than reimplementing drift/
decay/training/gating logic inside Prefect tasks. Every script already
works correctly standalone (see their own verification history) --
Prefect's job here is sequencing and branching on their results, not
duplicating logic that's already been tested.

DEMO MODE: pass --include-synthetic-for-drift to have the drift check
consider synthetic traffic. Without it, drift/decay checks only look at
real "live" traffic and simulated ground truth, which will likely report
"insufficient data" until real usage accumulates -- that's the honest
behavior, not a bug to work around.

Usage:
    python flows/retrain_pipeline.py
    python flows/retrain_pipeline.py --include-synthetic-for-drift
    python flows/retrain_pipeline.py --force-retrain   # skip drift/decay gate, always retrain
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from prefect import flow, task

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_script(args: list, description: str) -> subprocess.CompletedProcess:
    """Runs a script and returns the completed process. Raises on
    non-zero exit EXCEPT we don't treat a script's own reported negative
    finding (e.g. "no drift") as a failure -- these scripts exit 0 whether
    or not they find something actionable; only a real crash is non-zero."""
    logger.info(f"Running: {description}")
    result = subprocess.run(
        [sys.executable] + args, cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(f"{description} failed:\n{result.stderr}")
        raise RuntimeError(f"{description} exited with code {result.returncode}")
    return result


@task
def check_drift(include_synthetic: bool) -> dict:
    args = ["scripts/generate_drift_report.py"]
    if include_synthetic:
        args.append("--include-synthetic")
    run_script(args, "Drift check")
    signal_path = REPO_ROOT / "reports" / "drift_signal.json"
    with open(signal_path) as f:
        return json.load(f)


@task
def check_decay() -> dict:
    run_script(["scripts/check_performance_decay.py"], "Performance decay check")
    signal_path = REPO_ROOT / "reports" / "decay_check.json"
    with open(signal_path) as f:
        return json.load(f)


@task
def retrain(data_path: str) -> str:
    """Trains a new challenger (no --promote -- the gate decides that)
    and returns its run_id."""
    models_dir = REPO_ROOT / "models"
    before = set(models_dir.glob("*")) if models_dir.exists() else set()

    run_script(["train.py", "--data-path", data_path, "--output-dir", "models"], "Training challenger")

    after = set(models_dir.glob("*"))
    new_dirs = after - before
    if len(new_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one new run directory under models/, found {len(new_dirs)}: {new_dirs}"
        )
    run_id = new_dirs.pop().name
    logger.info(f"New challenger run_id: {run_id}")
    return run_id


@task
def promotion_gate(challenger_run_id: str) -> dict:
    run_script(
        ["scripts/promotion_gate.py", "--challenger-run-id", challenger_run_id],
        "Promotion gate",
    )
    decision_path = REPO_ROOT / "reports" / "promotion_decision.json"
    with open(decision_path) as f:
        return json.load(f)


@flow(name="fraud-detection-retrain-pipeline")
def retrain_pipeline(
    data_path: str = "ieee_fraud_detection.parquet",
    include_synthetic_for_drift: bool = False,
    force_retrain: bool = False,
) -> dict:
    drift_result = check_drift(include_synthetic_for_drift)
    decay_result = check_decay()

    drift_detected = drift_result.get("dataset_drift_detected", False)
    decay_detected = decay_result.get("decay_detected", False)

    logger.info(f"Drift detected: {drift_detected} | Decay detected: {decay_detected} | "
                f"Force retrain: {force_retrain}")

    if not (drift_detected or decay_detected or force_retrain):
        logger.info("No drift, no decay, no force flag. Skipping retrain.")
        return {
            "retrained": False,
            "reason": "no drift or decay detected",
            "drift_result": drift_result,
            "decay_result": decay_result,
        }

    trigger_reason = []
    if drift_detected:
        trigger_reason.append("drift")
    if decay_detected:
        trigger_reason.append("decay")
    if force_retrain:
        trigger_reason.append("forced")
    logger.info(f"Retrain triggered by: {', '.join(trigger_reason)}")

    challenger_run_id = retrain(data_path)
    decision = promotion_gate(challenger_run_id)

    return {
        "retrained": True,
        "trigger_reason": trigger_reason,
        "drift_result": drift_result,
        "decay_result": decay_result,
        "challenger_run_id": challenger_run_id,
        "promotion_decision": decision,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fraud detection retrain pipeline")
    parser.add_argument("--data-path", default="ieee_fraud_detection.parquet")
    parser.add_argument("--include-synthetic-for-drift", action="store_true",
                         help="Let the drift check consider synthetic demo traffic. "
                              "Without this, it only looks at real live traffic.")
    parser.add_argument("--force-retrain", action="store_true",
                         help="Skip the drift/decay gate and always retrain. For testing "
                              "the retrain+promotion-gate path on demand.")
    args = parser.parse_args()

    result = retrain_pipeline(
        data_path=args.data_path,
        include_synthetic_for_drift=args.include_synthetic_for_drift,
        force_retrain=args.force_retrain,
    )
    print(json.dumps(result, indent=2, default=str))
