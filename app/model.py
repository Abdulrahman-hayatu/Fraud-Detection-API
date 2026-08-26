import json
import pandas as pd
from pathlib import Path
import joblib
from app.schemas import FraudRequest
from app.config import DECISION_THRESHOLD
from app.prediction_logger import log_prediction


BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained pipeline ONCE at startup
MODEL_PATH = BASE_DIR / "Fraud_Detection" / "models" / "Fraud_Detection_Pipeline.pkl"
MODEL_METADATA_PATH = MODEL_PATH.with_name("model_metadata.json")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(
        f"Failed to load model at {MODEL_PATH}. This usually means the pickle is "
        f"corrupt or was trained with different library versions than what's "
        f"installed (check requirements.txt pins). Original error: {e}"
    ) from e

# Model version, for tagging prediction logs. Falls back to "unknown" for
# models promoted before model_metadata.json was introduced -- an older
# .pkl shouldn't crash the app on startup, but logs from it won't be
# traceable to a specific training run.
if MODEL_METADATA_PATH.exists():
    with open(MODEL_METADATA_PATH) as f:
        MODEL_RUN_ID = json.load(f).get("run_id", "unknown")
else:
    MODEL_RUN_ID = "unknown"


# Risk level mapping 
def get_risk_level(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    elif probability < 0.70:
        return "Medium"
    else:
        return "High"


# Prediction function (used by API)
def predict_fraud(request: FraudRequest) -> dict:
    # Convert request to DataFrame
    request_dict = request.model_dump()
    input_df = pd.DataFrame([request_dict])

    # Predict probability
    fraud_probability = model.predict_proba(input_df)[:, 1][0]

    is_fraud = fraud_probability >= DECISION_THRESHOLD

    response = {
        "fraud_probability": float(fraud_probability),
        "is_fraud": bool(is_fraud),
        "risk_level": get_risk_level(fraud_probability)
    }

    log_prediction(request_dict, response, MODEL_RUN_ID)

    return response
