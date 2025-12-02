"""
CKD Stage Prediction API with SHAP Explanations

This Flask API serves a machine learning model for predicting Chronic Kidney Disease (CKD) stages.
It includes full support for SHAP (SHapley Additive exPlanations) feature importance analysis.

Endpoints:
    GET  /              - Health check
    GET  /model/info    - Get model metadata and capabilities
    POST /predict       - Predict CKD stages (optionally with SHAP values)
    POST /explain       - Get detailed SHAP explanations for predictions

Model Components:
    - model: Calibrated classifier for stage prediction
    - encoder: OrdinalEncoder for stage labels (G1, G2, G3a, G3b, G4, G5)
    - imputer: SimpleImputer for handling missing values
    - shap_explainer: TreeExplainer for feature importance
    - raw_tree_model: Base LGBMClassifier
    - shap_background: Background data for SHAP

Note: The model must be saved with Python 3.12 due to numba compatibility.
If loading on Python 3.9, you'll need to recreate the model or use Python 3.12.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import pandas as pd
import pickle
import joblib
from typing import Any, Dict, List, Optional
import shap

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("CKD_MODEL_PATH", "risk_model.pkl")


def load_model(path: str):
    """
    Load model saved with joblib.
    Note: If you get numba version errors, the model was likely saved on
    a different Python version. See error message for solutions.
    """
    try:
        # Try standard joblib load
        loaded = joblib.load(path)
    except (TypeError, AttributeError) as e:
        error_msg = str(e)
        if "code()" in error_msg or "numba" in error_msg.lower():
            # Python/numba version mismatch - provide helpful error
            import sys
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            raise RuntimeError(
                f"\n{'='*70}\n"
                f"MODEL LOADING ERROR: Python/Numba Version Mismatch\n"
                f"{'='*70}\n"
                f"The model file '{path}' was saved on Python 3.12, but you're\n"
                f"trying to load it on Python {python_version}.\n\n"
                f"This causes incompatibility with numba-compiled code in the\n"
                f"SHAP explainer (even though the app doesn't use SHAP).\n\n"
                f"SOLUTIONS:\n"
                f"1. Use Python 3.12 to run the app:\n"
                f"   python3.12 -m venv .venv\n"
                f"   source .venv/bin/activate\n"
                f"   pip install -r requirements.txt\n\n"
                f"2. Recreate the model on Python {python_version}:\n"
                f"   Run your training notebook on Python {python_version}\n\n"
                f"3. Use the fix_model.py script (if you have Python 3.12 available):\n"
                f"   python3.12 fix_model.py\n\n"
                f"Original error: {error_msg}\n"
                f"{'='*70}\n"
            ) from e
        else:
            raise
    except Exception as e:
        # Fallback to pickle if joblib fails (unlikely for joblib-saved files)
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)
        except Exception as pickle_err:
            raise RuntimeError(
                f"Failed to load model from {path}. "
                f"Joblib error: {str(e)}. "
                f"Pickle error: {str(pickle_err)}."
            ) from e
    
    # Handle case where model is stored as a dictionary
    if isinstance(loaded, dict):
        return loaded
    return loaded


model_dict = load_model(MODEL_PATH)
# Extract all model components from dictionary
if isinstance(model_dict, dict):
    model = model_dict.get("model", model_dict)
    encoder = model_dict.get("encoder", None)
    imputer = model_dict.get("imputer", None)
    shap_explainer = model_dict.get("shap_explainer", None)
    raw_tree_model = model_dict.get("raw_tree_model", None)
    shap_background = model_dict.get("shap_background", None)
    
    # Optimize SHAP explainer for production (Render free tier optimizations)
    # 1. Reduce background from 2000 → 40 (70% less RAM)
    # 2. Use fast SHAP mode: feature_perturbation="tree_path_dependent" (20-40x faster)
    # 3. Preload explainer once (already done at module level)
    if shap_explainer is not None and shap_background is not None and raw_tree_model is not None:
        try:
            # Reduce background data size (default: 40 samples for optimal speed/accuracy balance)
            max_background_samples = int(os.environ.get("SHAP_MAX_BACKGROUND", "40"))
            if hasattr(shap_background, 'shape') and shap_background.shape[0] > max_background_samples:
                import random
                random.seed(42)  # For reproducibility
                sample_indices = random.sample(
                    range(shap_background.shape[0]),
                    min(max_background_samples, shap_background.shape[0])
                )
                shap_background_reduced = shap_background[sample_indices]
                # Recreate explainer with reduced background + fast mode
                try:
                    shap_explainer = shap.TreeExplainer(
                        raw_tree_model,
                        data=shap_background_reduced,
                        feature_perturbation="tree_path_dependent"  # Fast SHAP mode (20-40x faster)
                    )
                    print(f"Optimized SHAP explainer: reduced background from {shap_background.shape[0]} to {shap_background_reduced.shape[0]} samples with fast mode")
                except Exception as e:
                    print(f"Warning: Could not create fast mode explainer: {e}")
                    # Fallback: try without fast mode
                    try:
                        shap_explainer = shap.TreeExplainer(raw_tree_model, data=shap_background_reduced)
                        print(f"Optimized SHAP explainer: reduced background from {shap_background.shape[0]} to {shap_background_reduced.shape[0]} samples (standard mode)")
                    except Exception as e2:
                        print(f"Warning: Could not recreate SHAP explainer: {e2}")
                        # Keep original explainer
            else:
                # Background is already small, but try to enable fast mode
                try:
                    shap_explainer = shap.TreeExplainer(
                        raw_tree_model,
                        data=shap_background,
                        feature_perturbation="tree_path_dependent"  # Fast SHAP mode
                    )
                    print("Enabled fast SHAP mode on existing explainer")
                except Exception:
                    # If fast mode fails, keep original
                    pass
        except Exception as e:
            print(f"Warning: SHAP optimization check failed: {e}")
            # Continue with original explainer
    # Try different possible keys for feature list
    feature_list = (
        model_dict.get("feature_cols") or 
        model_dict.get("features") or 
        model_dict.get("feature_list")
    )
else:
    model = model_dict
    encoder = None
    imputer = None
    shap_explainer = None
    raw_tree_model = None
    shap_background = None
    feature_list = None

FEATURE_NAMES = feature_list if feature_list is not None else getattr(model, "feature_names_in_", None)
CLASS_LABELS = getattr(model, "classes_", None)

# Store model metadata
MODEL_INFO = {
    "has_shap": shap_explainer is not None,
    "has_imputer": imputer is not None,
    "has_encoder": encoder is not None,
    "feature_count": len(FEATURE_NAMES) if FEATURE_NAMES is not None else None,
    "class_count": len(CLASS_LABELS) if CLASS_LABELS is not None else None,
}


def _get_shap_class_index(pred_stage: str) -> Optional[int]:
    """
    Find the correct SHAP output index for a given CKD stage label.
    This handles the mapping between stage labels (G1, G2, etc.) and class indices.
    """
    if encoder is None or raw_tree_model is None:
        return None
    
    try:
        raw_classes = raw_tree_model.classes_
        stage_labels = encoder.inverse_transform(raw_classes.reshape(-1, 1)).ravel()
        
        matches = np.where(stage_labels == pred_stage)[0]
        if len(matches) > 0:
            return int(matches[0])
        
        # Fallback: find closest match
        stage_order = {"G1": 0, "G2": 1, "G3a": 2, "G3b": 3, "G4": 4, "G5": 5}
        pred_order = stage_order.get(pred_stage, 0)
        distances = [
            abs(pred_order - stage_order.get(lbl, pred_order))
            for lbl in stage_labels
        ]
        return int(np.argmin(distances))
    except Exception:
        return None


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



def _build_dataframe(records: List[Dict[str, Any]], apply_imputer: bool = True) -> pd.DataFrame:
    """
    Build DataFrame from records and apply preprocessing.
    
    Args:
        records: List of normalized record dictionaries
        apply_imputer: Whether to apply imputation (default: True)
    
    Returns:
        Preprocessed DataFrame ready for model prediction
    """
    df = pd.DataFrame(records)
    if FEATURE_NAMES is not None:
        # Ensure all required features are present
        for feature in FEATURE_NAMES:
            if feature not in df.columns:
                df[feature] = np.nan
        # Select only the features the model expects, in the correct order
        df = df[FEATURE_NAMES]
    
    # Apply imputer if available
    if apply_imputer and imputer is not None:
        try:
            # Imputer expects array-like, returns array
            df_array = imputer.transform(df)
            # Convert back to DataFrame to preserve column names
            df = pd.DataFrame(df_array, columns=FEATURE_NAMES, index=df.index)
        except Exception as e:
            # If imputer fails, log but continue
            print(f"Warning: Imputation failed: {e}")
    
    # Note: Encoder is for target labels, not features, so we don't apply it here
    return df


@app.route("/")
def health():
    return {"status": "ok"}


@app.route("/model/info", methods=["GET"])
def model_info():
    """Get information about the loaded model."""
    info = MODEL_INFO.copy()
    info["features"] = FEATURE_NAMES.tolist() if FEATURE_NAMES is not None else None
    info["classes"] = CLASS_LABELS.tolist() if CLASS_LABELS is not None else None
    if encoder is not None and hasattr(encoder, "categories_"):
        info["class_mapping"] = {
            int(i): cat[0] for i, cat in enumerate(encoder.categories_)
        }
    return jsonify(info)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict CKD stages for input records.
    
    Request body can include:
    - records: list of patient records
    - include_shap: boolean to include SHAP explanations (default: false)
    """
    try:
        payload = request.get_json(force=True, silent=False)
        records = _prepare_records(payload)
        include_shap = payload.get("include_shap", False) if isinstance(payload, dict) else False
        
        df = _build_dataframe(records)

        predictions = model.predict(df).tolist()
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(df).tolist()
            except Exception:
                probabilities = None

        # Get SHAP values if requested and available
        shap_values = None
        if include_shap and shap_explainer is not None:
            try:
                # Optimize SHAP computation for production environments
                # Skip additivity check for faster computation
                shap_values_raw = shap_explainer.shap_values(
                    df.values,
                    check_additivity=False  # Skip validation for speed
                )
                
                # Handle different SHAP output formats
                if isinstance(shap_values_raw, list):
                    # List of arrays (one per class)
                    shap_values = [sv.tolist() if hasattr(sv, 'tolist') else sv for sv in shap_values_raw]
                elif isinstance(shap_values_raw, np.ndarray):
                    if len(shap_values_raw.shape) == 3:
                        # 3D array: (samples, features, classes) - convert to list format
                        n_samples, n_features, n_classes = shap_values_raw.shape
                        shap_values = []
                        for class_idx in range(n_classes):
                            class_shap = shap_values_raw[:, :, class_idx].tolist()
                            shap_values.append(class_shap)
                    else:
                        # 2D array: (samples, features) - binary classification
                        shap_values = shap_values_raw.tolist()
                else:
                    shap_values = shap_values_raw
            except Exception as e:
                print(f"Warning: SHAP computation failed: {e}")
                shap_values = None

        results = []
        for idx, record in enumerate(records):
            result: Dict[str, Any] = {
                "input": record,
                "prediction": predictions[idx],
            }
            
            # Add decoded prediction if encoder is available
            if encoder is not None and CLASS_LABELS is not None:
                pred_idx = predictions[idx]
                if isinstance(pred_idx, (int, np.integer)):
                    if hasattr(encoder, "inverse_transform"):
                        try:
                            decoded = encoder.inverse_transform([[pred_idx]])[0][0]
                            result["prediction_label"] = decoded
                        except Exception:
                            pass
            
            if probabilities:
                result["probabilities"] = probabilities[idx]
                # Add probability for predicted class
                if isinstance(predictions[idx], (int, np.integer)):
                    pred_idx = int(predictions[idx])
                    if pred_idx < len(probabilities[idx]):
                        result["confidence"] = float(probabilities[idx][pred_idx])
            
            if CLASS_LABELS is not None:
                result["classes"] = CLASS_LABELS.tolist() if hasattr(CLASS_LABELS, "tolist") else list(CLASS_LABELS)
            
            # Add SHAP values for this record
            if shap_values is not None:
                if isinstance(shap_values, list) and len(shap_values) > 0:
                    # Multi-class: get SHAP values for the predicted class
                    pred_idx = int(predictions[idx])
                    if pred_idx < len(shap_values):
                        record_shap = shap_values[pred_idx][idx]
                        # Create feature importance mapping
                        shap_dict = {}
                        if FEATURE_NAMES is not None:
                            for feat_idx, feat_name in enumerate(FEATURE_NAMES):
                                if feat_idx < len(record_shap):
                                    shap_dict[feat_name] = float(record_shap[feat_idx])
                        result["shap_values"] = shap_dict
                        # Also include all classes' SHAP values
                        result["shap_values_all_classes"] = {
                            f"class_{i}": {
                                FEATURE_NAMES[j]: float(shap_values[i][idx][j])
                                for j in range(len(FEATURE_NAMES))
                            } if i < len(shap_values) and idx < len(shap_values[i]) else {}
                            for i in range(len(shap_values))
                        }
                else:
                    # Binary or single output
                    if idx < len(shap_values):
                        record_shap = shap_values[idx]
                        shap_dict = {}
                        if FEATURE_NAMES is not None:
                            for feat_idx, feat_name in enumerate(FEATURE_NAMES):
                                if feat_idx < len(record_shap):
                                    shap_dict[feat_name] = float(record_shap[feat_idx])
                        result["shap_values"] = shap_dict
            
            results.append(result)

        return jsonify({"results": results})
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/explain", methods=["POST"])
def explain():
    """
    Get SHAP explanations: stage, confidence, and top 3 most relevant features.
    Accepts one or multiple patient records.
    """
    if shap_explainer is None:
        return jsonify({"error": "SHAP explainer not available in model"}), 400
    
    try:
        records = _prepare_records(request.get_json(force=True, silent=False))
        df = _build_dataframe(records)

        # Get predictions first (fast)
        predictions = model.predict(df).tolist()
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(df).tolist()
            except Exception:
                pass

        # Compute SHAP values
        # Note: Render free tier has 30s hard timeout - may fail with many records
        try:
            shap_values_raw = shap_explainer.shap_values(
                df.values,
                check_additivity=False  # Skip validation for speed
            )
        except Exception as shap_err:
            error_msg = str(shap_err)
            if "timeout" in error_msg.lower() or "time" in error_msg.lower():
                return jsonify({
                    "error": "SHAP computation timed out. Try fewer records or set SHAP_MAX_BACKGROUND=5."
                }), 504
            else:
                return jsonify({
                    "error": f"SHAP computation failed: {error_msg}"
                }), 500
        
        # Handle different SHAP output formats
        if isinstance(shap_values_raw, list):
            shap_values = [sv.tolist() if hasattr(sv, 'tolist') else sv for sv in shap_values_raw]
        elif isinstance(shap_values_raw, np.ndarray):
            if len(shap_values_raw.shape) == 3:
                # 3D array: (samples, features, classes) - convert to list format
                n_samples, n_features, n_classes = shap_values_raw.shape
                shap_values = []
                for class_idx in range(n_classes):
                    class_shap = shap_values_raw[:, :, class_idx].tolist()
                    shap_values.append(class_shap)
            else:
                shap_values = shap_values_raw.tolist()
        else:
            shap_values = shap_values_raw

        results = []
        for idx, record in enumerate(records):
            pred_idx = int(predictions[idx])
            
            # Get prediction label (stage)
            stage = None
            if encoder is not None:
                try:
                    stage = encoder.inverse_transform([[pred_idx]])[0][0]
                except Exception:
                    pass
            
            # Get confidence score
            confidence = None
            if probabilities and idx < len(probabilities) and pred_idx < len(probabilities[idx]):
                confidence = float(probabilities[idx][pred_idx])
            
            # Get top 3 most relevant features from SHAP
            top_features = []
            if isinstance(shap_values, list) and pred_idx < len(shap_values) and idx < len(shap_values[pred_idx]):
                pred_shap = shap_values[pred_idx][idx]
                if isinstance(pred_shap, (list, np.ndarray)) and FEATURE_NAMES is not None:
                    # Build feature importance dict
                    feature_importance = {}
                    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
                        if feat_idx < len(pred_shap):
                            val = pred_shap[feat_idx]
                            if isinstance(val, (np.ndarray, np.generic)):
                                val = val.item() if val.size == 1 else float(val)
                            else:
                                val = float(val)
                            feature_importance[feat_name] = val
                    
                    # Sort by absolute importance and get top 3
                    sorted_factors = sorted(
                        feature_importance.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )
                    
                    # Return only top 3 feature names
                    top_features = [feat_name for feat_name, _ in sorted_factors[:3]]
            
            # Return response: stage, confidence, top 3 features
            result = {
                "stage": stage,
                "confidence": confidence,
                "top_features": top_features
            }
            
            results.append(result)

        return jsonify({"results": results})
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as exc:
        return jsonify({"error": f"Explanation failed: {exc}"}), 500
