"""Streamlit frontend for the Fraud Detection API.

Run locally (from the repository root):
    streamlit run frontend/app.py

The backend URL is read from Streamlit secrets (`API_URL`) or the `API_URL` environment
variable. There is deliberately no hard-coded default URL.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="centered")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
HEALTH_TIMEOUT = 8  # seconds; short so the page loads even if the backend is asleep


def get_setting(name: str, default: str | None = None) -> str | None:
    """Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # noqa: BLE001 - no secrets.toml present
        pass
    return os.getenv(name, default)


API_URL = (get_setting("API_URL") or "").strip().rstrip("/")
API_KEY = get_setting("API_KEY")
# Generous default: a sleeping free-tier backend needs ~1 minute to wake up.
REQUEST_TIMEOUT = float(get_setting("REQUEST_TIMEOUT", "60") or 60)
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# Example transactions: rows 0, 2 and 3 of the dataset preview in the notebook,
# with the ground-truth `isFraud` label that the dataset gives them.
PRESETS = {
    "Dataset row 0 - PAYMENT (labelled legitimate)": dict(
        step=1, tx_type="PAYMENT", amount=9839.64, old_org=170136.0, new_org=160296.36, old_dest=0.0, new_dest=0.0
    ),
    "Dataset row 2 - TRANSFER (labelled fraud)": dict(
        step=1, tx_type="TRANSFER", amount=181.0, old_org=181.0, new_org=0.0, old_dest=0.0, new_dest=0.0
    ),
    "Dataset row 3 - CASH_OUT (labelled fraud)": dict(
        step=1, tx_type="CASH_OUT", amount=181.0, old_org=181.0, new_org=0.0, old_dest=21182.0, new_dest=0.0
    ),
}
DEFAULTS = PRESETS["Dataset row 0 - PAYMENT (labelled legitimate)"]
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def load_preset(name: str) -> None:
    for key, value in PRESETS[name].items():
        st.session_state[key] = value


# --------------------------------------------------------------------------- #
# Backend communication
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=20, show_spinner=False)
def check_backend(api_url: str) -> dict:
    """GET /health -> {'state': online|not_ready|waking|unreachable|error, 'detail': str}."""
    try:
        response = requests.get(f"{api_url}/health", timeout=HEALTH_TIMEOUT)
    except requests.exceptions.Timeout:
        return {"state": "waking", "detail": "No response yet - a free-tier backend may be waking up."}
    except requests.exceptions.RequestException:
        return {"state": "unreachable", "detail": "Could not connect. Check API_URL."}

    try:
        body = response.json()
    except ValueError:
        body = {}
    if response.status_code == 200:
        return {"state": "online", "detail": "Model loaded and ready."}
    if response.status_code == 503:
        return {"state": "not_ready", "detail": body.get("detail") or "Model not ready."}
    return {"state": "error", "detail": f"Unexpected response (HTTP {response.status_code})."}


def format_validation_error(response: requests.Response) -> str:
    """Turn FastAPI's 422 body into a short human-readable message."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        return "invalid input."
    if isinstance(detail, list):
        parts = []
        for item in detail:
            field = item.get("loc", ["?"])[-1] if isinstance(item, dict) else "?"
            message = item.get("msg", "invalid value") if isinstance(item, dict) else str(item)
            parts.append(f"{field}: {message}")
        return "; ".join(parts) or "invalid input."
    return str(detail) if detail else "invalid input."


def call_predict(payload: dict) -> tuple[dict | None, str | None]:
    """POST /predict -> (result, error_message). Exactly one of the two is None."""
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        return None, (
            "The backend did not respond in time. If it is hosted on a free tier it may be "
            "starting up (up to ~1 minute) - please try again."
        )
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to the backend. Check that it is running and that API_URL is correct."
    except requests.exceptions.RequestException as exc:
        return None, f"Request failed ({type(exc).__name__})."

    if response.status_code == 200:
        try:
            return response.json(), None
        except ValueError:
            return None, "The backend returned an unreadable response."
    if response.status_code == 422:
        return None, f"The backend rejected the input - {format_validation_error(response)}"
    if response.status_code == 401:
        return None, "The backend rejected the API key. Check the API_KEY secret."
    if response.status_code == 503:
        return None, "The model is not ready on the backend yet. See the status indicator in the sidebar."
    return None, f"Backend error (HTTP {response.status_code})."


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("🛡️ Bank Fraud Detection")
st.caption("Score a single transaction with the trained XGBoost fraud model.")

if not API_URL:
    st.error(
        "**Backend URL not configured.** Set `API_URL` in Streamlit secrets "
        "(`.streamlit/secrets.toml` locally, or *App settings → Secrets* on Streamlit Community Cloud) "
        "or as an environment variable. See the README."
    )
    st.stop()

# ---- Sidebar: connectivity status ----
with st.sidebar:
    st.subheader("Backend status")
    status = check_backend(API_URL)
    if status["state"] == "online":
        st.success(f"🟢 Online\n\n{status['detail']}")
    elif status["state"] in ("waking", "not_ready"):
        st.warning(f"🟡 {status['detail']}")
    else:
        st.error(f"🔴 {status['detail']}")
    st.button("Re-check", on_click=check_backend.clear)
    st.divider()

# ---- Example presets ----
st.markdown("##### Try an example")
cols = st.columns(len(PRESETS))
for col, name in zip(cols, PRESETS):
    col.button(name, on_click=load_preset, args=(name,), use_container_width=True)

# ---- Input form ----
with st.form("transaction_form"):
    st.markdown("##### Transaction details")
    c1, c2 = st.columns(2)
    c1.number_input(
        "Step (hour of the simulated month)",
        min_value=1,
        max_value=743,
        step=1,
        key="step",
        help="PaySim time step: 1 step = 1 hour. The model also derives hour-of-day = step mod 24.",
    )
    c2.selectbox("Transaction type", TRANSACTION_TYPES, key="tx_type")
    st.number_input("Amount", min_value=0.0, step=100.0, format="%.2f", key="amount")

    c3, c4 = st.columns(2)
    c3.number_input("Sender balance before", min_value=0.0, step=100.0, format="%.2f", key="old_org")
    c4.number_input("Sender balance after", min_value=0.0, step=100.0, format="%.2f", key="new_org")
    c5, c6 = st.columns(2)
    c5.number_input("Recipient balance before", min_value=0.0, step=100.0, format="%.2f", key="old_dest")
    c6.number_input("Recipient balance after", min_value=0.0, step=100.0, format="%.2f", key="new_dest")

    submitted = st.form_submit_button("Analyze transaction", type="primary", use_container_width=True)

# ---- Prediction ----
if submitted:
    # Field names match the dataset columns (note the dataset's own spelling: Org vs Orig).
    payload = {
        "step": int(st.session_state["step"]),
        "type": st.session_state["tx_type"],
        "amount": float(st.session_state["amount"]),
        "oldbalanceOrg": float(st.session_state["old_org"]),
        "newbalanceOrig": float(st.session_state["new_org"]),
        "oldbalanceDest": float(st.session_state["old_dest"]),
        "newbalanceDest": float(st.session_state["new_dest"]),
    }
    with st.spinner("Contacting the fraud-detection API... (the first request after idle can take ~1 minute)"):
        result, error = call_predict(payload)

    if error:
        st.error(error)
    else:
        if result["is_fraud"]:
            st.error("🚨 **Flagged as likely FRAUD**")
        else:
            st.success("✅ **Not flagged as fraud**")
        m1, m2 = st.columns(2)
        m1.metric("Fraud probability", f"{result['fraud_probability']:.2%}")
        m2.metric("Decision threshold", f"{result['decision_threshold']:.2%}")
        st.caption(
            f"A transaction is flagged when its fraud probability is at or above the threshold. "
            f"Model: {result['model_name']}."
        )

with st.expander("About this model"):
    st.markdown(
        """
        * **Model:** tuned XGBoost classifier trained on the PaySim dataset (6,362,620 transactions, 8,213 fraudulent).
        * **Results on the notebook's held-out test split** (1,272,524 transactions, 1,643 fraudulent), at the
          saved decision threshold: **recall 0.897**, **precision 0.290**; ROC-AUC 0.9982, PR-AUC 0.8617.
        * The threshold favours catching fraud (high recall), so about 7 in 10 flags are false alarms.
        * PaySim is a *synthetic* dataset - real-world performance would differ.
        """
    )
