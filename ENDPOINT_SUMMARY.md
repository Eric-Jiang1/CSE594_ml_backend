# API Endpoints Summary

## Quick Overview

### Current Endpoints

1. **`GET /model/info`** - Get model metadata (features, classes, capabilities)
2. **`POST /predict`** - Main prediction endpoint (with optional SHAP)
3. **`POST /explain`** - Detailed SHAP explanations (similar to predict with SHAP)

---

## Endpoint Comparison

### `/predict` vs `/explain`

| Feature | `/predict` | `/explain` |
|---------|-----------|------------|
| **Purpose** | Primary prediction endpoint | Detailed SHAP explanations |
| **SHAP Values** | Optional (`include_shap: true`) | Always included |
| **Response Format** | Standard prediction format | Includes sorted `feature_importance` array |
| **Use Case** | General predictions | When you need detailed feature importance analysis |

### What `/predict` Returns

**Without SHAP (`include_shap: false` or omitted):**
```json
{
  "results": [{
    "input": {...},
    "prediction": 2,
    "prediction_label": "G3a",
    "probabilities": [0.05, 0.10, 0.35, 0.25, 0.15, 0.10],
    "confidence": 0.35,
    "classes": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
  }]
}
```

**With SHAP (`include_shap: true`):**
```json
{
  "results": [{
    "input": {...},
    "prediction": 2,
    "prediction_label": "G3a",
    "probabilities": [...],
    "confidence": 0.35,
    "classes": [...],
    "shap_values": {
      "sc": 2.5,
      "hemo": -1.2,
      ...
    },
    "shap_values_all_classes": {
      "class_0": {...},
      "class_1": {...},
      ...
    }
  }]
}
```

### What `/explain` Returns

**Summary:** The `/explain` endpoint is similar to `/predict` with `include_shap: true`, but always includes SHAP values and provides additional formatting. It returns all the same fields as `/predict` (prediction, probabilities, confidence) plus `shap_values` (feature importance for the predicted class), `feature_importance` (pre-sorted array of `[feature_name, shap_value]` tuples sorted by absolute importance), and `shap_values_all_classes` (SHAP values for all 6 CKD stages using class labels like "G1", "G2" instead of "class_0", "class_1"). Use this endpoint when you always need SHAP explanations and want the convenience of pre-sorted feature importance. For most use cases, `/predict` with `include_shap: true` is sufficient and more flexible.

---

## Recommendation

**Use `/predict` as your primary endpoint** because:
1. It can do everything `/explain` does (just set `include_shap: true`)
2. More flexible (you can choose whether to include SHAP)
3. Consistent response format
4. Better performance when SHAP isn't needed

**Use `/explain` only if:**
- You always need SHAP values
- You want the pre-sorted `feature_importance` array
- You prefer class labels ("G1") over indices ("class_0") in `shap_values_all_classes`

---

## Summary

**For most use cases, use:**
```
POST /predict
{
  "records": [...],
  "include_shap": true  // Set to true when you need explanations
}
```

This gives you everything you need in one endpoint!

