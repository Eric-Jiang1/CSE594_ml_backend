from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your model
with open("risk_model.pkl", "rb") as f:
    bundle = pickle.load(f)

    if "pipeline" in bundle:
        model = bundle["pipeline"]
    
    elif "model" in bundle:
        model = bundle["model"]
    
    else:
        raise Exception(f"Could not find model inside keys: {bundle.keys()}")
    

@app.route("/")
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]

        try:
            prob = model.predict_proba(df)[0].tolist()
        except:
            prob = None

        return jsonify({
            "prediction": int(prediction),
            "probabilities": prob
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/debug-model")
def debug_model():
    with open("risk_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return {"type": str(type(bundle)), 
            "keys": list(bundle.keys()) if isinstance(bundle, dict) else "no keys"}

