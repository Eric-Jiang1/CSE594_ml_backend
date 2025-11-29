# CKD Stage Prediction API

A Flask API for predicting Chronic Kidney Disease (CKD) stages with SHAP explanations.

## Requirements

- **Python 3.12** (required for model compatibility)
- All dependencies listed in `requirements.txt`

## Setup

1. **Install Python 3.12** (if not already installed):
   ```bash
   brew install python@3.12
   ```

2. **Create virtual environment**:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

Start the server with gunicorn:

```bash
source .venv/bin/activate
gunicorn app:app
```

The server will start on `http://127.0.0.1:8000`

## API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### `GET /model/info`
Get model metadata and capabilities.

**Response:**
```json
{
  "has_shap": true,
  "has_imputer": true,
  "has_encoder": true,
  "feature_count": 8,
  "class_count": 6,
  "features": ["age", "sex", "sg", "al", "sc", "bu", "hemo", "pcv"],
  "classes": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
  "class_mapping": {
    "0": "G1",
    "1": "G2",
    "2": "G3a",
    "3": "G3b",
    "4": "G4",
    "5": "G5"
  }
}
```

### `POST /predict`
Predict CKD stages for patient records.

**Request Body:**
```json
{
  "records": [
    {
      "age": 55,
      "sex": 0,
      "sg": 1.015,
      "al": 3.8,
      "sc": 2.5,
      "bu": 65,
      "hemo": 10.2,
      "pcv": 31
    }
  ],
  "include_shap": false
}
```

**Response:**
```json
{
  "results": [
    {
      "input": {...},
      "prediction": 2,
      "prediction_label": "G3a",
      "probabilities": [0.1, 0.2, 0.35, 0.2, 0.1, 0.05],
      "confidence": 0.35,
      "classes": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
      "shap_values": {
        "sc": 2.5,
        "hemo": -1.2,
        ...
      }
    }
  ]
}
```

### `POST /explain`
Get detailed SHAP explanations for predictions.

**Request Body:**
```json
{
  "records": [
    {
      "age": 55,
      "sex": 0,
      "sg": 1.015,
      "al": 3.8,
      "sc": 2.5,
      "bu": 65,
      "hemo": 10.2,
      "pcv": 31
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "input": {...},
      "prediction": 2,
      "prediction_label": "G3a",
      "probabilities": [...],
      "confidence": 0.35,
      "shap_values": {
        "sc": 2.5,
        "hemo": -1.2,
        ...
      },
      "feature_importance": [
        ["sc", 2.5],
        ["hemo", -1.2],
        ...
      ],
      "shap_values_all_classes": {
        "G1": {...},
        "G2": {...},
        "G3a": {...},
        ...
      }
    }
  ]
}
```

## Model Features

- **8 Input Features**: age, sex, sg (specific gravity), al (albumin), sc (serum creatinine), bu (blood urea), hemo (hemoglobin), pcv (packed cell volume)
- **6 Output Classes**: G1, G2, G3a, G3b, G4, G5 (CKD stages)
- **SHAP Explanations**: Feature importance for each prediction
- **Missing Value Handling**: Automatic imputation using median values
- **Calibrated Probabilities**: Isotonic calibration for better probability estimates

## Model Components

The model bundle includes:
- `model`: Calibrated classifier (CalibratedClassifierCV)
- `encoder`: OrdinalEncoder for stage labels
- `imputer`: SimpleImputer for missing values
- `shap_explainer`: TreeExplainer for SHAP values
- `raw_tree_model`: Base LGBMClassifier
- `shap_background`: Background data for SHAP

## Environment Variables

- `CKD_MODEL_PATH`: Path to model file (default: `risk_model.pkl`)

## Notes

- The model was trained and saved with Python 3.12, so Python 3.12 is required to load it
- SHAP explanations are computationally expensive; use `include_shap: true` only when needed
- All feature names are case-insensitive and will be normalized
