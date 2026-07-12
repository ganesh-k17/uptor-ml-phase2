import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "003-07July2026"

if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from mlpregressor_california import get_california_model, predict_california_value

app = Flask(__name__)

# Train once at startup and reuse for all requests
MODEL, SCALER, FEATURE_COLUMNS = get_california_model()


@app.get("/")
def home():
    return jsonify({"message": "Simple API is running", "status": "ok"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "world")
    return jsonify({
        "message": f"Hello, {name}!",
        "received": data,
    })


@app.post("/california-predict")
def california_predict():
    data = request.get_json(silent=True) or {}

    try:
        prediction = predict_california_value(data)
        return jsonify({
            "prediction": float(prediction),
            "model": "mlpregressor_california"
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(debug=False, host="0.0.0.0", port=port)