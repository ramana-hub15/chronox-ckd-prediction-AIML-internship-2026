"""REST API endpoints for ChronoX: /api/predict, /api/metrics, /api/history."""

from __future__ import annotations

import json

from flask import Blueprint, current_app, jsonify, request
from flask_login import current_user, login_required

from app.ml.predictor import FEATURE_META, load_artifacts, predict_ckd
from app.ml.preprocessing import FEATURE_NAMES
from app.models import Prediction, db

api_bp = Blueprint("api", __name__, url_prefix="/api")

ALLOWED_SG = {1.005, 1.010, 1.015, 1.020, 1.025}


@api_bp.route("/metrics", methods=["GET"])
def api_metrics():
    """Return model evaluation metrics and dataset summary JSON."""
    metrics = load_artifacts(current_app.config["MODELS_DIR"])["metrics"]
    return jsonify(metrics), 200


@api_bp.route("/history", methods=["GET"])
@login_required
def api_history():
    """Return the authenticated user's prediction history as JSON."""
    band = request.args.get("band")
    model = request.args.get("model")
    limit = request.args.get("limit", type=int)

    q = Prediction.query.filter_by(user_id=current_user.id)
    if band in {"Low", "Moderate", "High"}:
        q = q.filter_by(risk_band=band)
    if model in {"ann", "logistic", "both"}:
        q = q.filter_by(model_used=model)
    q = q.order_by(Prediction.created_at.desc())
    if limit and limit > 0:
        q = q.limit(min(limit, 200))

    items = [p.to_dict() for p in q.all()]
    return jsonify({"count": len(items), "predictions": items}), 200


@api_bp.route("/predict", methods=["POST"])
@login_required
def api_predict():
    """Validate 9 clinical features, run calibrated prediction, and persist record."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON request body is required."}), 400

    errors: dict[str, str] = {}
    cleaned: dict[str, float] = {}

    for feat in FEATURE_NAMES:
        if feat not in payload or payload[feat] is None or payload[feat] == "":
            errors[feat] = f"Missing required parameter '{feat}'."
            continue
        raw = payload[feat]
        if feat == "htn" and isinstance(raw, str):
            low = raw.strip().lower()
            if low in {"yes", "1", "true"}:
                val = 1.0
            elif low in {"no", "0", "false"}:
                val = 0.0
            else:
                errors[feat] = "Hypertension must be 0/1 or yes/no."
                continue
        else:
            try:
                val = float(raw)
            except (TypeError, ValueError):
                errors[feat] = f"Invalid numeric value for '{feat}'."
                continue

        meta = FEATURE_META[feat]
        if val < meta["min"] or val > meta["max"]:
            errors[feat] = f"Value for '{feat}' must be between {meta['min']} and {meta['max']}."
            continue
        cleaned[feat] = val

    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    model_choice = str(payload.get("model_choice", "both")).lower().strip()
    if model_choice not in {"ann", "logistic", "both"}:
        return jsonify({"error": "model_choice must be 'ann', 'logistic', or 'both'."}), 400

    patient_label = str(payload.get("patient_label", "API Screening")).strip()[:120] or "API Screening"
    result = predict_ckd(
        cleaned,
        model_choice=model_choice,
        models_dir=current_app.config["MODELS_DIR"],
    )

    record = Prediction(
        user_id=current_user.id,
        patient_label=patient_label,
        model_used=result["model_choice"],
        age=cleaned["age"],
        bp=cleaned["bp"],
        sg=cleaned["sg"],
        al=cleaned["al"],
        bgr=cleaned["bgr"],
        bu=cleaned["bu"],
        sc=cleaned["sc"],
        hemo=cleaned["hemo"],
        htn=int(cleaned["htn"]),
        risk_score=result["risk_score"],
        ann_score=result["model_scores"]["ann"],
        logistic_score=result["model_scores"]["logistic"],
        risk_band=result["risk_band"],
        stage_hint=result["stage_hint"]["stage"],
        egfr_approx=result["stage_hint"]["egfr_approx"],
        contributions_json=json.dumps(result["contributions"]),
        suggestions_json=json.dumps(result["suggestions"]),
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({"prediction_id": record.id, **result}), 201
