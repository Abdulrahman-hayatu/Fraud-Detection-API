# Fraud Detection API

![Retrain Pipeline](https://github.com/Abdulrahman-hayatu/Fraud-Detection-API/actions/workflows/retrain.yml/badge.svg)

A FastAPI service that scores transactions for fraud probability using an XGBoost model, wrapped in a full MLOps pipeline: data/model versioning, prediction logging, drift and performance-decay monitoring, experiment tracking, an automated champion-vs-challenger promotion gate, and a scheduled CI/CD retrain workflow.

**Status: portfolio / demonstration project.** The pipeline itself is real, tested, and running (see the badge above), but this API has no real customer traffic behind it yet. Several pieces — described honestly below — currently run against simulated data for demonstration purposes rather than production data. Nothing here claims to be "production-ready" in the sense of having handled real-world load or real-world labeled outcomes.

## API

**POST `/predict`**

| Field | Type | Example |
|---|---|---|
| `TransactionAmt` | float | `215.75` |
| `P_emaildomain` | string | `"gmail.com"` |
| `C1` | float | `1.0` |
| `C13` | float | `305.0` |
| `C14` | float | `420.0` |
| `card4` | string | `"visa"` |
| `card6` | string | `"debit"` |

Response:

```json
{
  "fraud_probability": 0.024,
  "is_fraud": false,
  "risk_level": "Low"
}
```

`is_fraud` is thresholded at 0.50 (see [Model Performance](#model-performance) for why). `risk_level` buckets the probability: Low (<0.30), Medium (0.30–0.70), High (≥0.70).

## Model Performance

Trained on the IEEE-CIS fraud detection dataset (XGBoost, 20% held-out test split, `random_state=42`):

| Metric | @ threshold 0.50 (served) | @ threshold 0.857 (not used) |
|---|---|---|
| Precision | 0.161 | 0.604 |
| Recall | 0.691 | 0.369 |
| F1 | 0.262 | 0.458 |
| ROC-AUC | 0.864 | — |
| PR-AUC | 0.457 | — |

**The 0.50 threshold is a deliberate, reviewed decision, not an oversight.** The precision at 0.161 looks bad in isolation, but the team decided recall matters more here — missing real fraud costs more than reviewing extra false positives. The 0.857 threshold column shows what precision/recall look like at the F1-optimal point instead; it's computed and logged on every training run as a diagnostic, but was explicitly rejected as the serving threshold for the reason above. See `app/config.py` for where this is recorded.

## Project Structure

```
app/                     FastAPI service (main.py, model.py, schemas.py, config.py, prediction_logger.py)
train.py                 Reproducible training script (CLI-driven, no hardcoded paths)
scripts/
  generate_synthetic_traffic.py   Demo traffic generator (see caveats below)
  generate_drift_report.py        Evidently drift monitoring
  check_performance_decay.py      Performance decay detection
  promotion_gate.py               Champion-vs-challenger promotion logic
flows/
  retrain_pipeline.py     Prefect orchestration tying the above together
.github/workflows/
  retrain.yml             Scheduled + manually-triggered CI/CD retrain workflow
```

## Running Locally

**Serve the API:**
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Docs at `http://127.0.0.1:8000/docs`.

**Train a new model:**
```bash
pip install -r requirements-train.txt
python train.py --data-path ieee_fraud_detection.parquet --output-dir models --promote
```
`--promote` overwrites the served model and writes `Fraud_Detection/models/model_metadata.json` (run ID + metrics). Omit it to train without affecting what's live.

Data and the served model are versioned with **DVC** (S3 remote), not committed directly to git:
```bash
dvc pull   # get the current data + model
dvc push   # after training + promoting
```

## MLOps Pipeline

### Prediction logging
Every `/predict` call is appended to `logs/predictions.jsonl`, tagged with the serving model's run ID, a `source` field (`live` vs `synthetic`), and a `request_id`.

**Limitation:** this writes to local disk. On a platform with an ephemeral filesystem (e.g. Render without a persistent disk), this log is wiped on every restart — sufficient to demonstrate the pattern, not sufficient for real production monitoring without shipping logs somewhere durable.

### Synthetic demo traffic
`scripts/generate_synthetic_traffic.py` sends requests built from real feature combinations sampled from the training data (not random noise), tagged `"source": "synthetic"`. Because the sampled rows come from labeled data, their true `isFraud` label is known — this is written to `logs/simulated_ground_truth.jsonl`, keyed by `request_id`.

**This exists only to demonstrate the monitoring pipeline working.** There is no real customer base generating this traffic or these outcomes. Every script that consumes this data says so explicitly and excludes it by default unless a `--include-synthetic` style flag is passed.

### Drift monitoring
`scripts/generate_drift_report.py` (Evidently) compares the training data's held-out test split against recent prediction logs — covering both **input feature drift** (transaction amount, card type, email domain, etc.) and **model output drift** (the predicted fraud probability distribution itself). Produces a human-readable HTML report and a machine-readable `reports/drift_signal.json` for automation to act on. Defaults to real (`live`) traffic only; synthetic traffic must be explicitly opted into.

### Performance decay detection
`scripts/check_performance_decay.py` joins predictions to their (simulated) true labels and compares recall/ROC-AUC against the currently-promoted model's recorded metrics, flagging decay past configurable tolerances.

**Limitation:** ground truth here is simulated-only. Real fraud outcomes require a feedback loop (chargebacks, manual review) that doesn't exist for this project, since there's no real customer base. This script is structurally correct and would work identically against real labeled outcomes — it just doesn't have any yet.

### Experiment tracking
Every training run logs hyperparameters, both metric sets (served-threshold and optimal-threshold-diagnostic), evaluation plots, and the model artifact to **MLflow via DagsHub** — independent of DVC, which remains the sole source of truth for what's actually served.

### Promotion gate (champion vs. challenger)
`scripts/promotion_gate.py` only promotes a newly trained model if it meets or beats the current production model on **both** recall (the priority metric) and ROC-AUC. Recall alone is gameable — a model that flags every transaction as fraud gets recall=1.0 trivially — so ROC-AUC, which is threshold-independent, acts as a guard against that.

### Orchestration & retraining policy
`flows/retrain_pipeline.py` (Prefect) checks drift and decay, and only retrains + runs the promotion gate if **either** fires. There is deliberately no data-volume trigger (e.g. "retrain after N new rows") — with no real customer base, that trigger would never mean anything meaningful yet.

### CI/CD
`.github/workflows/retrain.yml` runs the full flow weekly (Monday 06:00 UTC) or on manual dispatch, with a `demo_mode` toggle (generates synthetic traffic so the drift/decay checks have something to evaluate) and a `force_retrain` override for testing. Only commits back to the repo if an actual promotion occurs. Requires five repository secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (DVC's S3 remote), and `MLFLOW_TRACKING_URI`/`USERNAME`/`PASSWORD` (DagsHub).

## Known Limitations

- No real production traffic — demonstrated with simulated/synthetic data throughout, always clearly labeled as such.
- Ground truth for decay detection is simulated only; no real fraud-outcome feedback loop exists.
- Prediction logs are local-disk only; won't survive a restart on an ephemeral filesystem without further work.
- Small-sample statistics (drift scores, decay metrics) are noisy below a few hundred rows — the pipeline's default minimums account for this, but it's worth knowing if you tune them down.

## Author

**Abdulrahman Hayatu** — ML Engineer / Data Scientist
[GitHub](https://github.com/Abdulrahman-Hayatu) · [LinkedIn](https://linkedin.com/in/abdulrahman-hayatu)
