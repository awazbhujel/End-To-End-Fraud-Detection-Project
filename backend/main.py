"""Fraud Detection API (FastAPI).

Run locally:   uvicorn main:app --reload          (from the backend/ folder)
Run on Render: uvicorn main:app --host 0.0.0.0 --port $PORT

CORS is intentionally NOT enabled: the Streamlit frontend calls this API from its
server (via `requests`), not from the user's browser, so cross-origin rules do not apply.
"""

from __future__ import annotations

import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status

from predictor import (
    ArtifactError,
    FraudPredictor,
    Preprocessing,
    load_model_artifact,
)
from schemas import (
    HealthResponse,
    PredictionResponse,
    RootResponse,
    TransactionRequest,
)

APP_VERSION = "1.0.0"
BASE_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("fraud_api")


# --------------------------------------------------------------------------- #
# Startup: load artifacts once, keep them in memory
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "best_fraud_model_tuned.pkl"))
    pre_path = Path(os.getenv("PREPROCESSING_PATH", BASE_DIR / "model" / "preprocessing.json"))

    app.state.predictor = None
    app.state.model_artifact = None
    app.state.preprocessing = None
    app.state.errors = []

    try:
        app.state.model_artifact = load_model_artifact(model_path)
        logger.info("Model artifact loaded (%s).", app.state.model_artifact["model_name"])
    except ArtifactError as exc:
        app.state.errors.append(str(exc))
        logger.error("Model artifact not loaded: %s", exc)

    try:
        app.state.preprocessing = Preprocessing.from_json(pre_path)
        logger.info("Preprocessing artifact loaded.")
    except ArtifactError as exc:
        app.state.errors.append(str(exc))
        logger.error("Preprocessing artifact not loaded: %s", exc)

    if app.state.model_artifact is not None and app.state.preprocessing is not None:
        try:
            app.state.predictor = FraudPredictor(app.state.model_artifact, app.state.preprocessing)
            logger.info("Predictor ready.")
        except ArtifactError as exc:
            app.state.errors.append(str(exc))
            logger.error("Predictor not ready: %s", exc)

    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Scores a single bank transaction with a trained XGBoost fraud model.",
    version=APP_VERSION,
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Optional API-key protection (disabled unless the API_KEY env var is set)
# --------------------------------------------------------------------------- #
def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("API_KEY")
    if not expected:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", response_model=RootResponse, tags=["status"])
def root() -> RootResponse:
    """Basic API status."""
    return RootResponse(service="Fraud Detection API", status="running", version=APP_VERSION, docs="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Service up but model not ready"}},
    tags=["status"],
)
def health(request: Request, response: Response) -> HealthResponse:
    """Health check: 200 when the model and preprocessing are loaded, otherwise 503."""
    state = request.app.state
    ready = state.predictor is not None
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_loaded=state.model_artifact is not None,
        preprocessing_loaded=state.preprocessing is not None,
        detail=None if ready else " | ".join(state.errors) or "Predictor not initialised.",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(require_api_key)],
    responses={
        401: {"description": "Missing/invalid API key (only when API_KEY is configured)"},
        422: {"description": "Invalid transaction payload"},
        503: {"description": "Model or preprocessing artifact not loaded"},
    },
    tags=["prediction"],
)
def predict(transaction: TransactionRequest, request: Request) -> PredictionResponse:
    """Predict whether a transaction is fraudulent."""
    predictor: FraudPredictor | None = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Check GET /health for details.",
        )
    try:
        result = predictor.predict(transaction.model_dump())
    except ValueError as exc:  # e.g. transaction type not seen in training
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception:  # noqa: BLE001
        # Log the stack trace server-side; never echo internals or the payload to the client.
        logger.exception("Unexpected error while predicting")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal prediction error.")

    # Log the decision only - never amounts, balances or other transaction details.
    logger.info(
        "prediction type=%s result=%s probability=%.4f",
        transaction.type,
        result["prediction"],
        result["fraud_probability"],
    )
    return PredictionResponse(**result)


if __name__ == "__main__":  # pragma: no cover - convenience for `python main.py`
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
