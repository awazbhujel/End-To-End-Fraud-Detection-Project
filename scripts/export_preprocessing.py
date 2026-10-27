#!/usr/bin/env python
"""Re-create the preprocessing artifacts your notebook fitted but did not save.

WHY THIS EXISTS
---------------
The notebook trained the model on features scaled with a ``RobustScaler`` and encoded
with a ``LabelEncoder``, and built ``is_high_amount`` from the 99th percentile of
``amount``. ``best_fraud_model_tuned.pkl`` only contains the model, the threshold and the
feature list, so these three fitted objects are missing. The API cannot produce correct
scores without them.

WHAT IT DOES
------------
It repeats the notebook's preprocessing EXACTLY (same operations, same order, same
``train_test_split(test_size=0.2, stratify=y, random_state=42)``), then writes the
fitted values to ``backend/model/preprocessing.json``.

Because the notebook printed its test-set confusion matrix for the saved model, the
script can PROVE the reproduction is faithful: it scores the reproduced test split with
your model and compares the result to the notebook's printed numbers. The file is only
written if that check passes (or if you pass --skip-verify).

USAGE (from the repository root)
--------------------------------
    python scripts/export_preprocessing.py

``data/Fraud.csv`` must be the same PaySim file used in the notebook (6,362,620 rows).
Run it in an environment that has xgboost==2.1.4 (``pip install -r backend/requirements.txt``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

REPO_ROOT = Path(__file__).resolve().parents[1]

# Feature order saved inside best_fraud_model_tuned.pkl (artifact["features"]).
FEATURES = [
    "step",
    "amount",
    "log_amount",
    "is_high_amount",
    "hour",
    "is_night",
    "balance_diff_orig",
    "balance_diff_dest",
    "type_enc",
]
RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]
NIGHT_HOURS = [0, 1, 2, 3, 4, 5, 22, 23]
SCHEMA_VERSION = 1  # keep in sync with backend/predictor.py

# ---- Reference values printed by the notebook itself (not computed by this script) ----
NOTEBOOK_ROWS = 6_362_620  # cell "Quick EDA": Shape: (6362620, 11)
NOTEBOOK_FRAUD = 8_213  # cell "Quick EDA": isFraud == 1
NOTEBOOK_CONFUSION = {"tp": 1474, "fn": 169, "fp": 3612, "tn": 1267269}  # cell "Testing"
CONFUSION_TOLERANCE = 5  # absorbs float noise from running a different xgboost patch version


# --------------------------------------------------------------------------- #
def prepare(raw: pd.DataFrame):
    """Notebook feature engineering. Returns X, y, amount_p99, fitted LabelEncoder."""
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise SystemExit(f"ERROR: data file is missing required columns: {missing}")

    df = raw[RAW_COLUMNS].copy()

    df["log_amount"] = np.log1p(df["amount"])
    p99 = df["amount"].quantile(0.99)  # computed on the FULL dataset, as in the notebook
    df["is_high_amount"] = (df["amount"] > p99).astype(int)

    df["hour"] = df["step"] % 24
    df["is_night"] = df["hour"].isin(NIGHT_HOURS).astype(int)

    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df.fillna(0, inplace=True)

    le = LabelEncoder()
    df["type_enc"] = le.fit_transform(df["type"])

    X = df[FEATURES]
    y = df["isFraud"].values
    return X, y, float(p99), le


def fit_scaler(X: pd.DataFrame, y: np.ndarray):
    """Same split + scaler fit as the notebook. Returns scaler and the split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols != FEATURES:
        raise SystemExit(f"ERROR: numeric columns {num_cols} != expected features {FEATURES}")
    scaler = RobustScaler()
    scaler.fit(X_train[num_cols])
    return scaler, X_train, X_test, y_train, y_test


def build_artifact(scaler: RobustScaler, le: LabelEncoder, p99: float, extra: dict | None = None) -> dict:
    """Plain-JSON artifact consumed by backend/predictor.py (no pickle needed)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "features": FEATURES,
        "scaler_center": [float(v) for v in scaler.center_],
        "scaler_scale": [float(v) for v in scaler.scale_],  # sklearn already maps zero scale -> 1
        "amount_p99": float(p99),
        "type_classes": [str(c) for c in le.classes_],
        "meta": extra or {},
    }


def verify_against_model(model_path: Path, scaler, X_train, X_test, y_train, y_test) -> bool:
    """Check the reproduction against numbers the notebook printed. Returns True if OK."""
    import joblib
    from sklearn.metrics import confusion_matrix

    try:
        artifact = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! Could not load the model for verification: {type(exc).__name__}: {exc}")
        print("    Install the pinned deps: pip install -r backend/requirements.txt")
        return False

    model, threshold = artifact["model"], float(artifact["threshold"])
    if list(artifact["features"]) != FEATURES:
        print(f"  ! Model feature list differs from expected: {artifact['features']}")
        return False

    ok = True

    # Check 1: the model stores scale_pos_weight = (#legit / #fraud) of the TRAIN split.
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    saved = float(model.get_params()["scale_pos_weight"])
    match = abs(ratio - saved) < 1e-6
    ok &= match
    print(f"  [{'PASS' if match else 'FAIL'}] train-split class ratio {ratio:.6f} vs model's {saved:.6f}")

    # Check 2: confusion matrix on the reproduced test split vs the notebook's printed one.
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[FEATURES]), columns=FEATURES)
    scores = model.predict_proba(X_test_scaled)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, (scores >= threshold).astype(int)).ravel()
    got = {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}
    for key, expected in NOTEBOOK_CONFUSION.items():
        match = abs(got[key] - expected) <= CONFUSION_TOLERANCE
        ok &= match
        print(f"  [{'PASS' if match else 'FAIL'}] {key.upper():>2}: reproduced {got[key]:>8,} | notebook {expected:>8,}")
    return ok


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "Fraud.csv",
        help="PaySim CSV used in the notebook (default: data/Fraud.csv)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "backend" / "model" / "best_fraud_model_tuned.pkl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "backend" / "model" / "preprocessing.json",
    )
    parser.add_argument("--skip-verify", action="store_true", help="Write the file without the model check")
    args = parser.parse_args()

    if not args.data.is_file():
        print(f"ERROR: data file not found: {args.data}")
        return 2

    print(f"Reading {args.data} ...")
    raw = pd.read_csv(args.data, usecols=RAW_COLUMNS)
    print(f"  rows: {len(raw):,} (notebook: {NOTEBOOK_ROWS:,}) | frauds: {int(raw['isFraud'].sum()):,} (notebook: {NOTEBOOK_FRAUD:,})")
    if len(raw) != NOTEBOOK_ROWS or int(raw["isFraud"].sum()) != NOTEBOOK_FRAUD:
        print("  ! WARNING: this is not the same dataset the notebook used; the result may be wrong.")

    print("Re-running the notebook's preprocessing ...")
    X, y, p99, le = prepare(raw)
    scaler, X_train, X_test, y_train, y_test = fit_scaler(X, y)
    print(f"  amount 99th percentile : {p99:,.4f}")
    print(f"  type classes (encoded) : {dict(zip(le.classes_, range(len(le.classes_))))}")
    print(f"  train / test rows      : {len(X_train):,} / {len(X_test):,}")

    if args.skip_verify:
        print("Skipping verification (--skip-verify).")
    else:
        print("Verifying against the notebook's printed results ...")
        if not verify_against_model(args.model, scaler, X_train, X_test, y_train, y_test):
            print("\nVerification FAILED - preprocessing.json was NOT written.")
            print("Most likely cause: the dataset differs from the notebook's data, or the model/xgboost version differs.")
            return 1
        print("  Preprocessing reproduces the notebook's results.")

    import sklearn

    artifact = build_artifact(
        scaler,
        le,
        p99,
        extra={
            "source_rows": int(len(raw)),
            "step_min": int(raw["step"].min()),
            "step_max": int(raw["step"].max()),
            "sklearn_version": sklearn.__version__,
            "verified_against_notebook": not args.skip_verify,
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print(f"Observed step range in data: {artifact['meta']['step_min']}-{artifact['meta']['step_max']} (API accepts 1-743)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
