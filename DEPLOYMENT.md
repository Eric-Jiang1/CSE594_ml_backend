# Deployment Guide for Render

## Issue Fixed: Worker Timeout

The API was experiencing worker timeouts on Render's free tier due to SHAP computations taking longer than 30 seconds.

## Solutions Implemented

### 1. Increased Gunicorn Timeout
- Created `gunicorn_config.py` with 120-second timeout (up from default 30 seconds)
- Configured for single worker to avoid memory issues on free tier

### 2. Optimized SHAP Computation
- Added `check_additivity=False` to skip validation checks (faster computation)
- Reduced background data size automatically if it exceeds 100 samples
- Configurable via `SHAP_MAX_BACKGROUND` environment variable

### 3. Deployment Files
- **Procfile**: Specifies gunicorn command with config
- **render.yaml**: Render service configuration (optional)
- **gunicorn_config.py**: Gunicorn settings

## Deployment Steps

### Option 1: Using Procfile (Recommended)

1. Ensure these files are in your repository:
   - `Procfile`
   - `gunicorn_config.py`
   - `app.py`
   - `requirements.txt`
   - `risk_model.pkl`

2. On Render:
   - Create new Web Service
   - Connect your repository
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --config gunicorn_config.py`
   - Environment: `Python 3`

### Option 2: Using render.yaml

1. Add `render.yaml` to your repository
2. On Render, use "Apply Render Configuration" option

## Environment Variables

Optional environment variables:
- `SHAP_MAX_BACKGROUND`: Maximum background samples for SHAP (default: 100)
- `LOG_LEVEL`: Logging level (default: info)
- `CKD_MODEL_PATH`: Path to model file (default: risk_model.pkl)
- `PORT`: Server port (auto-set by Render)

## Performance Optimization

### For Free Tier:
- SHAP background data is automatically reduced to 100 samples
- Single worker process to avoid memory limits
- Timeout increased to 120 seconds

### For Paid Tiers:
You can increase `SHAP_MAX_BACKGROUND` for more accurate SHAP values:
```bash
SHAP_MAX_BACKGROUND=500
```

## Monitoring

Check Render logs for:
- "Optimized SHAP explainer" message (confirms background reduction)
- Worker timeout errors (should be resolved)
- SHAP computation warnings

## Troubleshooting

### Still Getting Timeouts?

1. **Reduce SHAP background further:**
   ```bash
   SHAP_MAX_BACKGROUND=50
   ```

2. **Disable SHAP entirely for testing:**
   - Don't set `include_shap: true` in requests
   - Or remove SHAP explainer from model bundle

3. **Check memory usage:**
   - Free tier has 512MB RAM limit
   - Monitor in Render dashboard

### SHAP Computation Still Slow?

- The model's background data might be very large
- Consider retraining with smaller background dataset
- Or use approximate SHAP methods

## Notes

- Free tier has limited resources; SHAP computations will be slower
- Consider upgrading to paid tier for better performance
- Predictions without SHAP are very fast (< 1 second)

