"""
Inference helper — importable from Colab after training.
Usage:
    from src.predict import predict_flow, predict_batch
"""
import numpy as np
import joblib
from pathlib import Path

MODEL_DIR = Path("models")

_cache = {}

def _load():
    if not _cache:
        _cache["model"]    = joblib.load(MODEL_DIR / "ensemble.pkl")
        _cache["scaler"]   = joblib.load(MODEL_DIR / "scaler.pkl")
        _cache["features"] = joblib.load(MODEL_DIR / "feature_names.pkl")
    return _cache["model"], _cache["scaler"], _cache["features"]

def predict_flow(flow_dict: dict) -> dict:
    model, scaler, features = _load()
    x = np.array([[flow_dict.get(f, 0.0) for f in features]])
    x_sc = scaler.transform(x)
    proba = model.predict_proba(x_sc)[0]
    pred  = int(np.argmax(proba))
    return {
        "is_attack":  bool(pred == 1),
        "confidence": round(float(proba[pred]), 4),
        "proba_benign": round(float(proba[0]), 4),
        "proba_attack": round(float(proba[1]), 4),
    }

def predict_batch(flows: list[dict]) -> list[dict]:
    model, scaler, features = _load()
    X  = np.array([[f.get(feat, 0.0) for feat in features] for f in flows])
    Xs = scaler.transform(X)
    probas = model.predict_proba(Xs)
    preds  = np.argmax(probas, axis=1)
    return [{"is_attack": bool(p==1), "confidence": round(float(probas[i][p]),4)}
            for i,p in enumerate(preds)]
