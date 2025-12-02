# Fix for Render Free Tier 502 Errors

## Problem
Render's free tier has a **hard 30-second timeout** that cannot be overridden by gunicorn configuration. SHAP computations are taking longer than 30 seconds, causing 502 errors.

## Immediate Solutions

### Option 1: Reduce SHAP Background to Minimum (Recommended)
In Render dashboard, set environment variable:
```
SHAP_MAX_BACKGROUND=5
```

This reduces SHAP background data to just 5 samples, making computation much faster.

### Option 2: Use /predict Instead of /explain
The `/predict` endpoint with `include_shap: true` has the same functionality but may handle errors better:
```json
POST /predict
{
  "records": [{"age": 55, ...}],
  "include_shap": true
}
```

### Option 3: Disable SHAP on Free Tier
If SHAP is not critical, you can:
1. Remove SHAP explainer from model bundle
2. Or catch errors and return predictions without SHAP

### Option 4: Upgrade Render Tier
Paid tiers have longer timeouts (up to 300 seconds) and more resources.

## Current Optimizations Already Applied

1. ✅ Background data reduced to 10 samples (default)
2. ✅ Single record limit for SHAP requests
3. ✅ `check_additivity=False` for faster computation
4. ✅ Gunicorn timeout set to 180s (but Render overrides to 30s)

## Quick Fix Steps

1. **Go to Render Dashboard → Your Service → Environment**
2. **Add/Update these variables:**
   ```
   SHAP_MAX_BACKGROUND=5
   MAX_SHAP_RECORDS=1
   ```
3. **Redeploy the service**

## Testing

After setting `SHAP_MAX_BACKGROUND=5`, test with:
```bash
curl -X POST https://your-app.onrender.com/explain \
  -H "Content-Type: application/json" \
  -d '{"records": [{"age": 55, "sex": 0, "sg": 1.015, "al": 3.8, "sc": 2.5, "bu": 65, "hemo": 10.2, "pcv": 31}]}'
```

If it still times out, SHAP may not be feasible on free tier. Consider using predictions without SHAP or upgrading.

