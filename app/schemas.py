from pydantic import BaseModel, Field


class FraudRequest(BaseModel):
    TransactionAmt: float = Field(..., example=215.75)
    P_emaildomain: str = Field(..., example="gmail.com")
    C1: float = Field(..., example=1.0)
    C13: float = Field(..., example=305.0)
    C14: float = Field(..., example=420.0)
    card4: str = Field(..., example="visa")
    card6: str = Field(..., example="debit")


class FraudResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str