"""Clinical screening, dashboard, analytics, comparison, and history routes."""

from __future__ import annotations

import csv
import json
from io import StringIO

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy import asc, desc

from app.forms import PredictionForm
from app.ml.predictor import FEATURE_META, load_artifacts, predict_ckd
from app.models import Prediction, User, db
from app.utils.pdf_report import build_prediction_pdf

clinical_bp = Blueprint("clinical", __name__)


def _save_prediction_record(
    user_id: int,
    patient_label: str,
    model_choice: str,
    features: dict[str, float],
    result: dict,
) -> Prediction:
    record = Prediction(
        user_id=user_id,
        patient_label=(patient_label or "Clinical Screening").strip(),
        model_used=result["model_choice"],
        age=features["age"],
        bp=features["bp"],
        sg=features["sg"],
        al=features["al"],
        bgr=features["bgr"],
        bu=features["bu"],
        sc=features["sc"],
        hemo=features["hemo"],
        htn=int(features["htn"]),
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
    return record


@clinical_bp.route("/dashboard")
@login_required
def dashboard() -> str:
    """User overview dashboard with KPIs and recent screenings."""
    preds = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .all()
    )
    total_count = len(preds)
    avg_risk = round(sum(p.risk_score for p in preds) / total_count, 1) if total_count else 0.0
    last_risk = round(preds[0].risk_score, 1) if total_count else None
    last_band = preds[0].risk_band if total_count else None
    high_risk_count = sum(1 for p in preds if p.risk_band == "High")

    return render_template(
        "pages/dashboard.html",
        total_count=total_count,
        avg_risk=avg_risk,
        last_risk=last_risk,
        last_band=last_band,
        high_risk_count=high_risk_count,
        recent_predictions=preds[:6],
    )


@clinical_bp.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    """Multi-step CKD risk prediction wizard."""
    form = PredictionForm()
    if form.validate_on_submit():
        features = {
            "age": float(form.age.data),
            "bp": float(form.bp.data),
            "sg": float(form.sg.data),
            "al": float(form.al.data),
            "bgr": float(form.bgr.data),
            "bu": float(form.bu.data),
            "sc": float(form.sc.data),
            "hemo": float(form.hemo.data),
            "htn": float(form.htn.data),
        }
        model_choice = form.model_choice.data
        result = predict_ckd(
            features,
            model_choice=model_choice,
            models_dir=current_app.config["MODELS_DIR"],
        )
        record = _save_prediction_record(
            user_id=current_user.id,
            patient_label=form.patient_label.data,
            model_choice=model_choice,
            features=features,
            result=result,
        )
        flash("CKD risk screening completed.", "success")
        return redirect(url_for("clinical.result", prediction_id=record.id))

    return render_template("pages/predict.html", form=form, feature_meta=FEATURE_META)


@clinical_bp.route("/result/<int:prediction_id>")
@login_required
def result(prediction_id: int) -> str:
    """Display animated risk gauge, stage hint, suggestions, and feature explanation."""
    prediction = db.session.get(Prediction, prediction_id)
    if prediction is None or prediction.user_id != current_user.id:
        abort(404)

    # Recompute stage detail text cleanly
    from app.ml.predictor import get_ckd_stage_hint, get_risk_band

    band_info = get_risk_band(prediction.risk_score)
    stage_info = get_ckd_stage_hint(
        prediction.risk_score, prediction.sc, prediction.age, prediction.al
    )
    return render_template(
        "pages/result.html",
        prediction=prediction,
        band_info=band_info,
        stage_info=stage_info,
    )


@clinical_bp.route("/result/<int:prediction_id>/pdf")
@login_required
def download_pdf(prediction_id: int) -> Response:
    """Download a formatted PDF clinical screening report."""
    prediction = db.session.get(Prediction, prediction_id)
    if prediction is None or prediction.user_id != current_user.id:
        abort(404)

    pdf_bytes = build_prediction_pdf(prediction, current_user)
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="chronox_report_{prediction.id}.pdf"'
        },
    )


@clinical_bp.route("/analytics")
def analytics() -> str:
    """Interactive Chart.js analytics dashboard for ML evaluation and screening trends."""
    metrics = load_artifacts(current_app.config["MODELS_DIR"])["metrics"]

    if current_user.is_authenticated:
        preds = (
            Prediction.query.filter_by(user_id=current_user.id)
            .order_by(Prediction.created_at.asc())
            .all()
        )
        is_demo_preview = False
    else:
        demo = User.query.filter_by(email="demo@chronox.com").first()
        preds = (
            Prediction.query.filter_by(user_id=demo.id)
            .order_by(Prediction.created_at.asc())
            .all()
            if demo
            else []
        )
        is_demo_preview = True

    band_counts = {
        "Low": sum(1 for p in preds if p.risk_band == "Low"),
        "Moderate": sum(1 for p in preds if p.risk_band == "Moderate"),
        "High": sum(1 for p in preds if p.risk_band == "High"),
    }
    history_series = [
        {
            "date": p.created_at.strftime("%b %d"),
            "label": p.patient_label,
            "risk": round(p.risk_score, 1),
            "ann": round(p.ann_score, 1),
            "logistic": round(p.logistic_score, 1),
        }
        for p in preds
    ]

    return render_template(
        "pages/analytics.html",
        metrics=metrics,
        band_counts=band_counts,
        history_series=history_series,
        is_demo_preview=is_demo_preview,
    )


@clinical_bp.route("/comparison")
def comparison() -> str:
    """Side-by-side comparison of Calibrated ANN vs Calibrated Logistic Regression."""
    metrics = load_artifacts(current_app.config["MODELS_DIR"])["metrics"]
    return render_template("pages/comparison.html", metrics=metrics)


@clinical_bp.route("/history")
@login_required
def history() -> str:
    """Searchable, sortable, filterable table of past predictions."""
    query_text = request.args.get("q", "").strip()
    band_filter = request.args.get("band", "").strip()
    model_filter = request.args.get("model", "").strip()
    sort_by = request.args.get("sort", "newest").strip()

    q = Prediction.query.filter_by(user_id=current_user.id)
    if query_text:
        like_term = f"%{query_text}%"
        q = q.filter(
            Prediction.patient_label.ilike(like_term)
            | Prediction.stage_hint.ilike(like_term)
        )
    if band_filter in {"Low", "Moderate", "High"}:
        q = q.filter_by(risk_band=band_filter)
    if model_filter in {"ann", "logistic", "both"}:
        q = q.filter_by(model_used=model_filter)

    order_map = {
        "newest": desc(Prediction.created_at),
        "oldest": asc(Prediction.created_at),
        "risk_desc": desc(Prediction.risk_score),
        "risk_asc": asc(Prediction.risk_score),
    }
    q = q.order_by(order_map.get(sort_by, desc(Prediction.created_at)))
    predictions = q.all()

    return render_template(
        "pages/history.html",
        predictions=predictions,
        query_text=query_text,
        band_filter=band_filter,
        model_filter=model_filter,
        sort_by=sort_by,
    )


@clinical_bp.route("/history/delete/<int:prediction_id>", methods=["POST"])
@login_required
def delete_prediction(prediction_id: int):
    """Delete a prediction owned by the logged-in user."""
    prediction = db.session.get(Prediction, prediction_id)
    if prediction is None or prediction.user_id != current_user.id:
        abort(404)
    db.session.delete(prediction)
    db.session.commit()
    flash("Prediction record deleted.", "info")
    return redirect(url_for("clinical.history"))


@clinical_bp.route("/history/export")
@login_required
def export_history_csv() -> Response:
    """Export all predictions of the logged-in user as CSV."""
    preds = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .all()
    )
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "ID",
            "Created At (UTC)",
            "Label",
            "Model",
            "Risk Score (%)",
            "Risk Band",
            "ANN Score (%)",
            "Logistic Score (%)",
            "Stage Hint",
            "Approx eGFR",
            "Age",
            "BP",
            "Specific Gravity",
            "Albumin",
            "Blood Glucose Random",
            "Blood Urea",
            "Serum Creatinine",
            "Hemoglobin",
            "Hypertension",
        ]
    )
    for p in preds:
        writer.writerow(
            [
                p.id,
                p.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                p.patient_label,
                p.model_used,
                p.risk_score,
                p.risk_band,
                p.ann_score,
                p.logistic_score,
                p.stage_hint,
                p.egfr_approx,
                p.age,
                p.bp,
                p.sg,
                p.al,
                p.bgr,
                p.bu,
                p.sc,
                p.hemo,
                "Yes" if p.htn == 1 else "No",
            ]
        )

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="chronox_history.csv"'},
    )
