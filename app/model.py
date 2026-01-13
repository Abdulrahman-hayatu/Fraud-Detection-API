import pandas as pd
from pathlib import Path
import joblib
from app.schemas import FraudRequest


BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained pipeline ONCE at startup
MODEL_PATH = BASE_DIR / "Fraud_Detection" / "models" / "Fraud_Detection_Pipeline.pkl"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)


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
    input_df = pd.DataFrame([request.dict()])

    # Predict probability
    fraud_probability = model.predict_proba(input_df)[:, 1][0]

    # Decision threshold
    threshold = 0.50
    is_fraud = fraud_probability >= threshold

    return {
        "fraud_probability": float(fraud_probability),
        "is_fraud": bool(is_fraud),
        "risk_level": get_risk_level(fraud_probability)
    }
