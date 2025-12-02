# CKD Prediction API - Frontend Developer Documentation

## Base URL
```
http://127.0.0.1:8000
```

## Overview

This API provides endpoints for predicting Chronic Kidney Disease (CKD) stages and obtaining detailed explanations of the predictions using SHAP (SHapley Additive exPlanations) values.

**Key Features:**
- Predicts CKD stage (G1, G2, G3a, G3b, G4, or G5) from patient biomarkers
- Returns prediction probabilities for all stages
- Provides SHAP feature importance explanations (optional)
- Automatically handles missing values using median imputation
- Supports batch predictions (multiple patients at once)

---

## Endpoints Summary

### 1. `GET /model/info`
Get model metadata and configuration.

### 2. `POST /predict` ⭐ **PRIMARY ENDPOINT**
Get predictions with optional SHAP explanations. This is the main endpoint you'll use.

### 3. `POST /explain`
Get detailed SHAP explanations (similar to `/predict` with `include_shap: true`, but with additional formatting).

---

## Detailed Endpoint Documentation

### `GET /model/info`

**Purpose:** Retrieve model metadata, available features, and class mappings.

**Request:**
```http
GET /model/info
```

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

**Use Case:** Call this once on app initialization to understand what features are required and how to map class indices to labels.

---

### `POST /predict` ⭐ **MAIN ENDPOINT**

**Purpose:** Predict CKD stages for one or more patients. Optionally includes SHAP feature importance explanations.

**Request:**
```http
POST /predict
Content-Type: application/json
```

**Request Body Formats:**

**Format 1: Single patient (object)**
```json
{
  "age": 55,
  "sex": 0,
  "sg": 1.015,
  "al": 3.8,
  "sc": 2.5,
  "bu": 65,
  "hemo": 10.2,
  "pcv": 31,
  "include_shap": false
}
```

**Format 2: Single patient in array**
```json
[
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
```

**Format 3: Multiple patients**
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
    },
    {
      "age": 65,
      "sex": 1,
      "sg": 1.010,
      "al": 3.2,
      "sc": 3.0,
      "bu": 80,
      "hemo": 9.5,
      "pcv": 28
    }
  ],
  "include_shap": true
}
```

**Request Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `records` | Array | No* | - | Array of patient records (if using object format) |
| `include_shap` | Boolean | No | `false` | Include SHAP feature importance values |
| Feature fields | Number/String | Partial** | - | Patient biomarker values (see Feature Reference below) |

\* If you send an array directly or a single object, `records` is not needed.
\** You can omit some features; missing values will be imputed automatically.

**Response (without SHAP):**
```json
{
  "results": [
    {
      "input": {
        "age": 55,
        "sex": 0,
        "sg": 1.015,
        "al": 3.8,
        "sc": 2.5,
        "bu": 65,
        "hemo": 10.2,
        "pcv": 31
      },
      "prediction": 2,
      "prediction_label": "G3a",
      "probabilities": [0.05, 0.10, 0.35, 0.25, 0.15, 0.10],
      "confidence": 0.35,
      "classes": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    }
  ]
}
```

**Response (with SHAP - `include_shap: true`):**
```json
{
  "results": [
    {
      "input": {
        "age": 55,
        "sex": 0,
        "sg": 1.015,
        "al": 3.8,
        "sc": 2.5,
        "bu": 65,
        "hemo": 10.2,
        "pcv": 31
      },
      "prediction": 2,
      "prediction_label": "G3a",
      "probabilities": [0.05, 0.10, 0.35, 0.25, 0.15, 0.10],
      "confidence": 0.35,
      "classes": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
      "shap_values": {
        "sc": 2.5,
        "hemo": -1.2,
        "bu": 0.8,
        "al": -0.5,
        "pcv": -0.3,
        "sg": 0.1,
        "age": -0.05,
        "sex": 0.02
      },
      "shap_values_all_classes": {
        "class_0": {
          "age": -0.03,
          "sex": 0.32,
          "sg": 0.01,
          "al": 0.01,
          "sc": 0.01,
          "bu": 0.00,
          "hemo": 0.00,
          "pcv": 0.00
        },
        "class_1": { ... },
        "class_2": { ... },
        "class_3": { ... },
        "class_4": { ... },
        "class_5": { ... }
      }
    }
  ]
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `results` | Array | Array of prediction results (one per input record) |
| `results[].input` | Object | Echo of the input data (normalized) |
| `results[].prediction` | Integer | Predicted class index (0-5) |
| `results[].prediction_label` | String | Predicted stage label ("G1", "G2", "G3a", "G3b", "G4", "G5") |
| `results[].probabilities` | Array[Float] | Probability for each class [G1, G2, G3a, G3b, G4, G5] |
| `results[].confidence` | Float | Probability of the predicted class (0.0-1.0) |
| `results[].classes` | Array[Float] | Class indices [0, 1, 2, 3, 4, 5] |
| `results[].shap_values` | Object | SHAP values for predicted class (only if `include_shap: true`) |
| `results[].shap_values_all_classes` | Object | SHAP values for all classes (only if `include_shap: true`) |

**SHAP Values Explanation:**
- **Positive values**: Feature pushes prediction toward higher (worse) CKD stages
- **Negative values**: Feature pushes prediction toward lower (better) CKD stages
- **Magnitude**: Larger absolute values indicate stronger influence

**Error Responses:**

```json
// 400 Bad Request - Validation error
{
  "error": "No JSON payload provided"
}

// 500 Internal Server Error
{
  "error": "Prediction failed: <error message>"
}
```

---

### `POST /explain`

**Purpose:** Get detailed SHAP explanations with sorted feature importance. This endpoint is similar to `/predict` with `include_shap: true`, but always includes SHAP values and provides additional formatting. The response includes all the same fields as `/predict` (prediction, probabilities, confidence) plus `shap_values` (feature importance for the predicted class), `feature_importance` (pre-sorted array of `[feature_name, shap_value]` tuples sorted by absolute importance), and `shap_values_all_classes` (SHAP values for all 6 CKD stages using class labels like "G1", "G2" instead of "class_0", "class_1"). Use this endpoint when you always need SHAP explanations and want the convenience of pre-sorted feature importance. For most use cases, `/predict` with `include_shap: true` is sufficient and more flexible.

---

## Feature Reference

### Required Features

| Feature | Name | Type | Range/Values | Description |
|---------|------|------|--------------|-------------|
| Age | `age` | Number | 25-85 | Patient age in years |
| Sex | `sex` | Number | 0 or 1 | 0 = Female, 1 = Male |
| Specific Gravity | `sg` | Number | 1.002-1.030 | Urine specific gravity |
| Albumin | `al` | Number | 2.0-5.4 | Serum albumin (g/dL) |
| Serum Creatinine | `sc` | Number | 0.4-7.0 | Serum creatinine (mg/dL) |
| Blood Urea | `bu` | Number | 5-180 | Blood urea nitrogen (mg/dL) |
| Hemoglobin | `hemo` | Number | 6-17 | Hemoglobin (g/dL) |
| Packed Cell Volume | `pcv` | Number | 16-52 | Hematocrit (%) |

### Feature Handling

1. **Case Insensitive**: Feature names are case-insensitive (`Age`, `AGE`, `age` all work)
2. **Missing Values**: You can omit features; they will be automatically imputed with median values
3. **Type Conversion**: 
   - Strings like `"yes"`, `"y"`, `"true"` → `1`
   - Strings like `"no"`, `"n"`, `"false"` → `0`
   - Numeric strings → converted to numbers
   - Empty strings → treated as missing (imputed)

**Example - Partial Input:**
```json
{
  "hemo": 14.5,
  "sc": 1.2,
  "bu": 30.0,
  "pcv": 45.0
}
```
Missing features (`age`, `sex`, `sg`, `al`) will be automatically filled with median values.

---

## CKD Stage Reference

| Stage | Label | Index | Description |
|-------|-------|-------|-------------|
| G1 | "G1" | 0 | Normal or high eGFR (≥90) |
| G2 | "G2" | 1 | Mildly decreased eGFR (60-89) |
| G3a | "G3a" | 2 | Mildly to moderately decreased eGFR (45-59) |
| G3b | "G3b" | 3 | Moderately to severely decreased eGFR (30-44) |
| G4 | "G4" | 4 | Severely decreased eGFR (15-29) |
| G5 | "G5" | 5 | Kidney failure (eGFR <15) |

---

## Usage Examples

### Example 1: Basic Prediction (JavaScript/Fetch)

```javascript
async function predictCKD(patientData) {
  const response = await fetch('http://127.0.0.1:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      records: [patientData],
      include_shap: false
    })
  });
  
  const data = await response.json();
  return data.results[0];
}

// Usage
const patient = {
  age: 55,
  sex: 0,
  sg: 1.015,
  al: 3.8,
  sc: 2.5,
  bu: 65,
  hemo: 10.2,
  pcv: 31
};

const result = await predictCKD(patient);
console.log(`Predicted Stage: ${result.prediction_label}`);
console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
```

### Example 2: Prediction with SHAP Explanations

```javascript
async function predictWithSHAP(patientData) {
  const response = await fetch('http://127.0.0.1:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      records: [patientData],
      include_shap: true
    })
  });
  
  const data = await response.json();
  return data.results[0];
}

const result = await predictWithSHAP(patient);

// Display top contributing features
const topFeatures = Object.entries(result.shap_values)
  .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
  .slice(0, 3);

console.log('Top contributing features:');
topFeatures.forEach(([feature, value]) => {
  const direction = value > 0 ? 'worsens' : 'improves';
  console.log(`  ${feature}: ${value.toFixed(2)} (${direction} CKD)`);
});
```

### Example 3: Batch Predictions

```javascript
async function predictBatch(patients) {
  const response = await fetch('http://127.0.0.1:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      records: patients,
      include_shap: false
    })
  });
  
  const data = await response.json();
  return data.results;
}

const patients = [
  { age: 55, sex: 0, sg: 1.015, al: 3.8, sc: 2.5, bu: 65, hemo: 10.2, pcv: 31 },
  { age: 65, sex: 1, sg: 1.010, al: 3.2, sc: 3.0, bu: 80, hemo: 9.5, pcv: 28 }
];

const results = await predictBatch(patients);
results.forEach((result, idx) => {
  console.log(`Patient ${idx + 1}: ${result.prediction_label} (${(result.confidence * 100).toFixed(1)}%)`);
});
```

### Example 4: Get Model Info

```javascript
async function getModelInfo() {
  const response = await fetch('http://127.0.0.1:8000/model/info');
  const data = await response.json();
  return data;
}

const modelInfo = await getModelInfo();
console.log('Available features:', modelInfo.features);
console.log('Class mapping:', modelInfo.class_mapping);
```

---

## Error Handling

### Common Errors

1. **400 Bad Request**
   - Missing JSON payload
   - Invalid data format
   - Empty records array

2. **500 Internal Server Error**
   - Model loading issues
   - SHAP computation failures
   - Internal processing errors

### Error Response Format

```json
{
  "error": "Error message description"
}
```

### Recommended Error Handling

```javascript
async function predictWithErrorHandling(patientData) {
  try {
    const response = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        records: [patientData],
        include_shap: true
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || `HTTP ${response.status}`);
    }
    
    const data = await response.json();
    return data.results[0];
  } catch (error) {
    console.error('Prediction failed:', error.message);
    // Handle error in UI
    return null;
  }
}
```

---

## Performance Considerations

1. **SHAP Computation**: Including SHAP values (`include_shap: true`) adds computational overhead. Use sparingly for real-time predictions.

2. **Batch Processing**: The API supports batch predictions. Send multiple patients in one request for better performance.

3. **Missing Features**: The API automatically imputes missing values, so you can send partial data. However, more complete data leads to more accurate predictions.

---

## Recommendations for Frontend Implementation

1. **Call `/model/info` on app startup** to get feature list and class mappings
2. **Use `/predict` as the primary endpoint** - it can do everything you need
3. **Set `include_shap: true` only when needed** (e.g., when user clicks "Explain" button)
4. **Handle missing features gracefully** - the API will impute them, but you may want to show warnings
5. **Display probabilities** - show confidence scores and probability distribution
6. **Visualize SHAP values** - use bar charts or waterfall plots to show feature contributions
7. **Cache model info** - don't call `/model/info` on every request

---

## Quick Reference

**Primary Endpoint:** `POST /predict`

**Minimal Request:**
```json
{
  "sc": 2.5,
  "hemo": 10.2
}
```

**Full Request with SHAP:**
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
  "include_shap": true
}
```

**Key Response Fields:**
- `prediction_label`: "G1", "G2", "G3a", "G3b", "G4", or "G5"
- `confidence`: 0.0 to 1.0 (probability of predicted class)
- `probabilities`: Array of 6 probabilities (one per stage)
- `shap_values`: Object mapping feature names to importance values (if `include_shap: true`)

---

## Support

For API issues or questions, contact the backend team or refer to the backend repository documentation.

