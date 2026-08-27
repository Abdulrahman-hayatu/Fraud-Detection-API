"""
Prediction logging for the fraud detection API.

Every /predict call is appended as one JSON line to a local log file.
This is the data that later Evidently drift monitoring will compare
against the training distribution -- without it, drift detection has
nothing to measure.

KNOWN LIMITATION -- read before deploying:
If this API runs on a platform with an ephemeral filesystem (e.g. Render's
free/standard tiers), this log file is WIPED on every restart or redeploy.
Local JSONL logging is fine for local development and demonstrating the
pattern, but production drift monitoring needs the log written to
something that survives restarts -- e.g. shipped to S3/a database, or
written to a mounted persistent disk. Treat this as step one of two, not
a finished production logging solution.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_PATH = LOG_DIR / "predictions.jsonl"

# Appends from concurrent requests (FastAPI can handle several at once)
# must not interleave partial writes into the file.
_write_lock = threading.Lock()


def log_prediction(
    request_data: dict, response_data: dict, model_run_id: str,
    source: str = "live", request_id: str = None,
) -> None:
    """Append one prediction record. Never raises -- a logging failure
    should not break the actual prediction response.

    source: "live" for real API traffic, "synthetic" for generated demo/
    load-test traffic (see scripts/generate_synthetic_traffic.py). Tagged
    in the record itself so downstream analysis (e.g. Evidently drift
    reports) can filter synthetic rows out rather than silently mixing
    them with real production data.

    request_id: caller-supplied or server-generated ID. Used by
    scripts/check_performance_decay.py to join synthetic predictions
    against their (simulated) true labels, recorded separately by
    scripts/generate_synthetic_traffic.py. Real production predictions
    have an ID too but nothing to join it against, since this API has no
    feedback loop for real fraud outcomes.
    """
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model_run_id": model_run_id,
        "source": source,
        "input": request_data,
        "output": response_data,
    }
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str)
        with _write_lock:
            with open(LOG_PATH, "a") as f:
                f.write(line + "\n")
    except Exception as e:
        # Logging is best-effort. A disk-full or permissions error here
        # should show up in application logs, not take down /predict.
        logger.error(f"Failed to write prediction log: {e}")
