from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import pandas as pd
import pickle
from typing import Any, Dict, List

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("CKD_MODEL_PATH", "risk_model.pkl")


def load_model(path: str):
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    # Handle case where model is stored as a dictionary
    if isinstance(loaded, dict):
        return loaded
    return loaded


model_dict = load_model(MODEL_PATH)
# Extract actual model and metadata from dictionary if needed
if isinstance(model_dict, dict):
    model = model_dict.get("model", model_dict)
    encoder = model_dict.get("encoder", None)
    feature_list = model_dict.get("features", None)
else:
    model = model_dict
    encoder = None
    feature_list = None

FEATURE_NAMES = feature_list if feature_list is not None else getattr(model, "feature_names_in_", None)
CLASS_LABELS = getattr(model, "classes_", None)


def _normalize_key(key: str) -> str:
    return key.strip().lower()


def _normalize_value(value: Any) -> Any:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return np.nan
        lowered = stripped.lower()
        if lowered in {"yes", "y", "true"}:
            return 1
        if lowered in {"no", "n", "false"}:
            return 0
        try:
            return float(stripped)
        except ValueError:
            return stripped
    return value


def _prepare_records(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        raise ValueError("No JSON payload provided")

    # Accept single record, list, or {"records": [...]}
    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    elif isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise ValueError("Unsupported payload. Provide an object or list of objects.")

    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("No records provided")

    normalized: List[Dict[str, Any]] = []

    # Allowed features = what the model expects
    allowed_features = set(FEATURE_NAMES) if FEATURE_NAMES is not None else None

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each record must be an object")

        normalized_record = {}

        for key, value in record.items():
            if not key:
                continue

            clean_key = _normalize_key(str(key))

            # If the model has a known feature list, drop unknown keys
            if allowed_features is not None and clean_key not in allowed_features:
                continue

            # Prevent nested dicts or lists from being passed to float()
            if isinstance(value, (dict, list)):
                raise ValueError(f"Feature '{clean_key}' must be a scalar, got {type(value)}")

            normalized_record[clean_key] = _normalize_value(value)

        normalized.append(normalized_record)

    return normalized



def _build_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if FEATURE_NAMES is not None:
        # Ensure all required features are present
        for feature in FEATURE_NAMES:
            if feature not in df.columns:
                df[feature] = np.nan
        # Select only the features the model expects, in the correct order
        df = df[FEATURE_NAMES]
    # Apply encoder if available
    if encoder is not None:
        try:
            df = encoder.transform(df)
        except Exception:
            # If encoder fails, try to use it as a function or skip
            pass
    return df


@app.route("/")
def health():
    return {"status": "ok"}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        records = _prepare_records(request.get_json(force=True, silent=False))
        df = _build_dataframe(records)

        predictions = model.predict(df).tolist()
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(df).tolist()
            except Exception:
                probabilities = None

        results = []
        for idx, record in enumerate(records):
            result: Dict[str, Any] = {
                "input": record,
                "prediction": predictions[idx],
            }
            if probabilities:
                result["probabilities"] = probabilities[idx]
            if CLASS_LABELS is not None:
                result["classes"] = CLASS_LABELS.tolist() if hasattr(CLASS_LABELS, "tolist") else list(CLASS_LABELS)
            results.append(result)

        return jsonify({"results": results})
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500
