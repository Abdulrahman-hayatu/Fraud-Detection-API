import uuid
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
    x_request_id: Optional[str] = Header(default=None),
):
    """
    Predict fraud probability for a single transaction.

    x_prediction_source: internal-only header for tagging prediction logs
    (e.g. "synthetic" for demo/load-test traffic vs "live" for real
    requests). Not part of the public API contract -- callers can ignore
    this entirely and it defaults to "live".

    x_request_id: optional caller-supplied ID, logged alongside the
    prediction. Lets a caller (e.g. the synthetic traffic generator, which
    knows the real label for its sampled rows) join predictions against a
    separately-recorded outcome later. Real production traffic has no
    such ID by default -- one is generated server-side so every log line
    still has one, but it has nothing to join against unless the caller
    tracks it themselves.
    """
    request_id = x_request_id or str(uuid.uuid4())
    return predict_fraud(request, source=x_prediction_source, request_id=request_id)
