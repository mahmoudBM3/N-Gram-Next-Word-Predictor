"""Flask web UI for n-gram predictor."""

from __future__ import annotations

import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer
from src.inference.predictor import Predictor
from src.model.ngram_model import NGramModel


def get_env(name: str) -> str:
    """Read an environment variable or raise an explicit KeyError."""
    value = os.getenv(name)
    if value is None or value == "":
        raise KeyError(f"Missing config variable: {name}. Check config/.env.")
    return value


# Load config
load_dotenv(os.path.join("config", ".env"), override=True)

# Initialize Flask app
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Load model and dependencies
try:
    ngram_order = int(get_env("NGRAM_ORDER"))
    unk_threshold = int(get_env("UNK_THRESHOLD"))
    top_k = int(get_env("TOP_K"))
    model_path = get_env("MODEL")
    vocab_path = get_env("VOCAB")

    normalizer = Normalizer()
    model = NGramModel(ngram_order=ngram_order, unk_threshold=unk_threshold)
    model.load(model_path, vocab_path)
    predictor = Predictor(model=model, normalizer=normalizer, ngram_order=ngram_order)
    
    print("✓ Model loaded successfully")
except Exception as exc:
    print(f"ERROR loading model: {exc}")
    raise


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html", top_k=top_k)


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for predictions."""
    data = request.get_json()
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "Input text is empty"}), 400
    
    try:
        predictions = predictor.predict_next(text, top_k)
        return jsonify({"predictions": predictions, "text": text})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("Starting N-Gram Predictor Web UI...")
    print("Open your browser to: http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
