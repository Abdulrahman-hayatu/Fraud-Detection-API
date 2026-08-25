import pandas as pd
from pathlib import Path
import joblib
from app.schemas import FraudRequest
from app.config import DECISION_THRESHOLD


BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained pipeline ONCE at startup
MODEL_PATH = BASE_DIR / "Fraud_Detection" / "models" / "Fraud_Detection_Pipeline.pkl"
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
    input_df = pd.DataFrame([request.model_dump()])

    # Predict probability
    fraud_probability = model.predict_proba(input_df)[:, 1][0]

    is_fraud = fraud_probability >= DECISION_THRESHOLD

    return {
        "fraud_probability": float(fraud_probability),
        "is_fraud": bool(is_fraud),
        "risk_level": get_risk_level(fraud_probability)
    }
