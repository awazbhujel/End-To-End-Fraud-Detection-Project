"""Request/response schemas.

Field names deliberately match the dataset columns used in the notebook (PaySim),
including its inconsistent spelling: ``oldbalanceOrg`` vs ``newbalanceOrig``.
The API accepts the RAW transaction; all feature engineering (log_amount, hour,
balance differences, type encoding, scaling, ...) happens server-side so that clients
cannot get it wrong.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TransactionType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Sanity cap only (guards against absurd/overflow-prone input); not a modelling constraint.
MAX_AMOUNT = 1e12


class TransactionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",  # unknown/misspelled fields -> 422 instead of being silently ignored
        json_schema_extra={
            "examples": [
                {
                    "step": 1,
                    "type": "TRANSFER",
                    "amount": 181.0,
                    "oldbalanceOrg": 181.0,
                    "newbalanceOrig": 0.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 0.0,
                }
            ]
        },
    )

    step: int = Field(
        ge=1,
        le=743,
        description="PaySim time step: 1 step = 1 hour of the simulated month (1-743).",
    )
    type: TransactionType = Field(description="Transaction type.")
    amount: float = Field(ge=0, le=MAX_AMOUNT, allow_inf_nan=False, description="Transaction amount.")
    oldbalanceOrg: float = Field(
        ge=0, le=MAX_AMOUNT, allow_inf_nan=False, description="Sender balance before the transaction."
    )
    newbalanceOrig: float = Field(
        ge=0, le=MAX_AMOUNT, allow_inf_nan=False, description="Sender balance after the transaction."
    )
    oldbalanceDest: float = Field(
        ge=0, le=MAX_AMOUNT, allow_inf_nan=False, description="Recipient balance before the transaction."
    )
    newbalanceDest: float = Field(
        ge=0, le=MAX_AMOUNT, allow_inf_nan=False, description="Recipient balance after the transaction."
    )

    @field_validator("type", mode="before")
    @classmethod
    def _normalise_type(cls, value):
        """Accept 'transfer ' / 'Transfer' as 'TRANSFER'."""
        return value.strip().upper() if isinstance(value, str) else value


class PredictionResponse(BaseModel):
    prediction: Literal["FRAUD", "LEGITIMATE"]
    is_fraud: bool
    fraud_probability: float = Field(description="Model probability of fraud (0-1).")
    decision_threshold: float = Field(description="Flag as fraud when probability >= this value.")
    model_name: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    preprocessing_loaded: bool
    detail: str | None = None


class RootResponse(BaseModel):
    service: str
    status: str
    version: str
    docs: str
