from fastapi import FastAPI
from app.schemas import FraudRequest, FraudResponse
from app.model import predict_fraud

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using XGBoost",
    version="1.0.0"
)


@app.post("/predict", response_model=FraudResponse)
def predict(request: FraudRequest):
    """
    Predict fraud probability for a single transaction
    """
    return predict_fraud(request)
