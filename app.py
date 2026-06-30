"""
MaternaAI — Flask Inference API
----------------------------------
Serves the trained XGBoost model with SHAP-based explanations.
Run locally with: python app.py
Deploy free-tier on Render.com or Railway.app

Endpoints:
    POST /predict   -> { risk_level, risk_label, confidence, top_factors, recommendation }
    GET  /health     -> { status: "ok" }
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import joblib
import shap
import numpy as np
import json

app = Flask(__name__)
CORS(app)  # allow requests from your PWA frontend during development

# ---- Load model artifacts once at startup ----
model = xgb.XGBClassifier()
model.load_model("model.json")
scaler = joblib.load("scaler.joblib")

with open("feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

explainer = shap.TreeExplainer(model)

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
RECOMMENDATIONS = {
    0: "Routine follow-up at next scheduled antenatal visit.",
    1: "Closer monitoring recommended. Schedule earlier follow-up.",
    2: "Immediate specialist referral recommended.",
}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Validate required fields
    missing = [c for c in FEATURE_COLS if c not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        x = np.array([[float(data[c]) for c in FEATURE_COLS]])
    except (ValueError, TypeError):
        return jsonify({"error": "All fields must be numeric"}), 400

    x_scaled = scaler.transform(x)

    # Predict
    raw = model.predict(x_scaled)
    probs = raw[0] if raw.ndim > 1 else model.predict_proba(x_scaled)[0]
    risk_class = int(np.argmax(probs))
    confidence = float(np.max(probs))

    # SHAP explanation for this specific prediction
    shap_values = explainer.shap_values(x_scaled)
    if isinstance(shap_values, list):
        class_shap = shap_values[risk_class][0]
    else:
        class_shap = shap_values[0, :, risk_class] if shap_values.ndim == 3 else shap_values[0]

    # Top 3 contributing factors for this patient
    contributions = sorted(
        zip(FEATURE_COLS, class_shap),
        key=lambda t: abs(t[1]),
        reverse=True
    )[:3]

    top_factors = [
        {
            "feature": feat,
            "value": data[feat],
            "impact": "increases risk" if val > 0 else "decreases risk",
            "magnitude": round(float(abs(val)), 3)
        }
        for feat, val in contributions
    ]

    return jsonify({
        "risk_class": risk_class,
        "risk_label": RISK_LABELS[risk_class],
        "confidence": round(confidence, 3),
        "top_factors": top_factors,
        "recommendation": RECOMMENDATIONS[risk_class],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
