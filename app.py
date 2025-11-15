from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your model
with open("risk_model.pkl", "rb") as f:
    model = pickle.load(f)

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
