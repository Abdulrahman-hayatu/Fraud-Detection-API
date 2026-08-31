from fastapi import FastAPI, Header
from typing import Optional
from app.schemas import FraudRequest, FraudResponse
from app.model import predict_fraud

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using XGBoost",
    version="1.0.0"
)


@app.post("/predict", response_model=FraudResponse)
def predict(
    request: FraudRequest,
    x_prediction_source: Optional[str] = Header(default="live"),
):
    """
    Predict fraud probability for a single transaction.

    x_prediction_source: internal-only header for tagging prediction logs
    (e.g. "synthetic" for demo/load-test traffic vs "live" for real
    requests). Not part of the public API contract. Callers can ignore
    this entirely and it defaults to "live".
    """
    return predict_fraud(request, source=x_prediction_source)
