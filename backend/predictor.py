"""Model/preprocessing loading and inference.

This module reproduces, step for step, the preprocessing performed in
``notebooks/Bank_fraud_detection.ipynb`` before the XGBoost model was trained:

    log_amount         = log1p(amount)
    is_high_amount     = amount > 99th percentile of amount (computed on the full dataset)
    hour               = step % 24
    is_night           = hour in {0,1,2,3,4,5,22,23}
    balance_diff_orig  = oldbalanceOrg  - newbalanceOrig
    balance_diff_dest  = newbalanceDest - oldbalanceDest
    type_enc           = LabelEncoder(type)      (alphabetical class order)
    all 9 features     -> RobustScaler (fitted on the training split only)

The notebook did NOT save the fitted RobustScaler, the 99th-percentile value or the
LabelEncoder classes. They are stored in ``preprocessing.json``, which is generated
by ``scripts/export_preprocessing.py`` (see README). Without that file the API
refuses to predict instead of silently returning wrong scores.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("fraud_api.predictor")

# Hours flagged as "night" in the notebook: df['hour'].isin([0,1,2,3,4,5,22,23])
NIGHT_HOURS = frozenset({0, 1, 2, 3, 4, 5, 22, 23})

# Keys the notebook stored in best_fraud_model_tuned.pkl
MODEL_ARTIFACT_KEYS = ("model_name", "model", "threshold", "features")

PREPROCESSING_SCHEMA_VERSION = 1


class ArtifactError(RuntimeError):
    """Raised when the model or preprocessing artifact is missing or invalid."""


# --------------------------------------------------------------------------- #
# Preprocessing artifact
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Preprocessing:
    """Everything the notebook fitted on the training data, in a plain-JSON form."""

    features: list[str]
    scaler_center: np.ndarray  # RobustScaler.center_ (per-feature median)
    scaler_scale: np.ndarray  # RobustScaler.scale_  (per-feature IQR, zeros -> 1)
    amount_p99: float
    type_classes: list[str]  # LabelEncoder.classes_ (index == encoded value)

    @classmethod
    def from_json(cls, path: Path) -> "Preprocessing":
        path = Path(path)
        if not path.is_file():
            raise ArtifactError(
                f"Preprocessing artifact '{path.name}' not found. Generate it with "
                "scripts/export_preprocessing.py (see README, step 'Generate the "
                "preprocessing artifact')."
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("schema_version") != PREPROCESSING_SCHEMA_VERSION:
                raise ValueError(f"unsupported schema_version {raw.get('schema_version')!r}")
            features = [str(f) for f in raw["features"]]
            center = np.asarray(raw["scaler_center"], dtype=float)
            scale = np.asarray(raw["scaler_scale"], dtype=float)
            p99 = float(raw["amount_p99"])
            classes = [str(c) for c in raw["type_classes"]]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"Preprocessing artifact '{path.name}' is invalid: {exc}") from exc

        n = len(features)
        if center.shape != (n,) or scale.shape != (n,):
            raise ArtifactError("Preprocessing artifact: scaler arrays do not match feature count.")
        if not (np.isfinite(center).all() and np.isfinite(scale).all() and math.isfinite(p99)):
            raise ArtifactError("Preprocessing artifact contains non-finite values.")
        if (scale == 0).any():
            raise ArtifactError("Preprocessing artifact contains a zero scale value.")
        if not classes:
            raise ArtifactError("Preprocessing artifact has no transaction type classes.")
        return cls(features, center, scale, p99, classes)


# --------------------------------------------------------------------------- #
# Feature engineering (identical to the notebook)
# --------------------------------------------------------------------------- #
def engineer_features(rows: pd.DataFrame, pre: Preprocessing) -> pd.DataFrame:
    """Raw PaySim columns -> the model's 9 *unscaled* features, in training order.

    ``rows`` must contain: step, type, amount, oldbalanceOrg, newbalanceOrig,
    oldbalanceDest, newbalanceDest.
    """
    unknown = sorted(set(rows["type"]) - set(pre.type_classes))
    if unknown:
        raise ValueError(f"Unknown transaction type(s): {unknown}. Allowed: {pre.type_classes}")

    out = pd.DataFrame(index=rows.index)
    out["step"] = rows["step"].astype(float)
    out["amount"] = rows["amount"].astype(float)
    out["log_amount"] = np.log1p(rows["amount"].astype(float))
    out["is_high_amount"] = (rows["amount"] > pre.amount_p99).astype(int)
    out["hour"] = rows["step"] % 24
    out["is_night"] = out["hour"].isin(NIGHT_HOURS).astype(int)
    out["balance_diff_orig"] = rows["oldbalanceOrg"] - rows["newbalanceOrig"]
    out["balance_diff_dest"] = rows["newbalanceDest"] - rows["oldbalanceDest"]
    type_index = {name: i for i, name in enumerate(pre.type_classes)}
    out["type_enc"] = rows["type"].map(type_index).astype(int)
    return out[pre.features]


def scale_features(features: pd.DataFrame, pre: Preprocessing) -> pd.DataFrame:
    """Equivalent to ``RobustScaler.transform``: (x - center_) / scale_."""
    scaled = (features.to_numpy(dtype=float) - pre.scaler_center) / pre.scaler_scale
    return pd.DataFrame(scaled, columns=pre.features, index=features.index)


def preprocess(rows: pd.DataFrame, pre: Preprocessing) -> pd.DataFrame:
    """Raw transactions -> model-ready (scaled) feature matrix."""
    return scale_features(engineer_features(rows, pre), pre)


# --------------------------------------------------------------------------- #
# Model artifact
# --------------------------------------------------------------------------- #
def load_model_artifact(path: Path) -> dict[str, Any]:
    """Load the dict saved by the notebook with ``joblib.dump``.

    SECURITY: joblib/pickle can execute arbitrary code when loading. Only load
    artifacts you created yourself or fully trust.
    """
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"Model artifact '{path.name}' not found.")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            artifact = joblib.load(path)
        if any("older version of XGBoost" in str(w.message) for w in caught):
            logger.info(
                "Model was pickled with an older XGBoost release than the one installed; it loads "
                "and predicts without errors. For long-term storage consider Booster.save_model()."
            )
    except Exception as exc:  # noqa: BLE001 - surface any unpickling problem clearly
        raise ArtifactError(
            f"Could not load model artifact '{path.name}' ({type(exc).__name__}: {exc}). "
            "Check that xgboost==2.1.4 (see backend/requirements.txt) is installed."
        ) from exc

    if not isinstance(artifact, dict) or any(k not in artifact for k in MODEL_ARTIFACT_KEYS):
        raise ArtifactError(f"Model artifact must be a dict with keys {MODEL_ARTIFACT_KEYS}.")
    if not hasattr(artifact["model"], "predict_proba"):
        raise ArtifactError("Model in artifact has no predict_proba().")
    threshold = float(artifact["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ArtifactError(f"Decision threshold {threshold} is outside (0, 1).")
    artifact["threshold"] = threshold
    artifact["features"] = list(artifact["features"])
    return artifact


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
class FraudPredictor:
    """Holds the trained model + preprocessing and scores single transactions."""

    def __init__(self, model_artifact: dict[str, Any], preprocessing: Preprocessing) -> None:
        if model_artifact["features"] != preprocessing.features:
            raise ArtifactError(
                "Feature mismatch between model and preprocessing artifacts: "
                f"model={model_artifact['features']} preprocessing={preprocessing.features}"
            )
        self.model = model_artifact["model"]
        self.model_name: str = str(model_artifact["model_name"])
        self.threshold: float = model_artifact["threshold"]
        self.features: list[str] = model_artifact["features"]
        self.pre = preprocessing

    def predict(self, transaction: dict[str, Any]) -> dict[str, Any]:
        """Score one raw transaction (dict with the 7 raw PaySim fields)."""
        X = preprocess(pd.DataFrame([transaction]), self.pre)
        probability = float(self.model.predict_proba(X)[0, 1])
        is_fraud = probability >= self.threshold  # same rule as the notebook
        return {
            "prediction": "FRAUD" if is_fraud else "LEGITIMATE",
            "is_fraud": bool(is_fraud),
            "fraud_probability": round(probability, 6),
            "decision_threshold": round(self.threshold, 6),
            "model_name": self.model_name,
        }
