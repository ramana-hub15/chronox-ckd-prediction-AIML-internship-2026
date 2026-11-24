"""Inference, risk stratification, CKD stage hint, and explainability for ChronoX."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.ml.preprocessing import FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

FEATURE_META: dict[str, dict[str, Any]] = {
    "age": {"label": "Age", "unit": "yrs", "normal": "18 – 65 yrs", "min": 1, "max": 110},
    "bp": {"label": "Blood Pressure (Diastolic)", "unit": "mmHg", "normal": "60 – 80 mmHg", "min": 40, "max": 200},
    "sg": {"label": "Specific Gravity", "unit": "", "normal": "1.015 – 1.025", "min": 1.005, "max": 1.030},
    "al": {"label": "Albumin (Urine)", "unit": "0–5", "normal": "0 (Trace/None)", "min": 0, "max": 5},
    "bgr": {"label": "Blood Glucose Random", "unit": "mg/dL", "normal": "70 – 140 mg/dL", "min": 20, "max": 600},
    "bu": {"label": "Blood Urea", "unit": "mg/dL", "normal": "10 – 45 mg/dL", "min": 1, "max": 400},
    "sc": {"label": "Serum Creatinine", "unit": "mg/dL", "normal": "0.6 – 1.2 mg/dL", "min": 0.2, "max": 80},
    "hemo": {"label": "Hemoglobin", "unit": "g/dL", "normal": "12.0 – 17.5 g/dL", "min": 2.5, "max": 22},
    "htn": {"label": "Hypertension", "unit": "Yes/No", "normal": "No", "min": 0, "max": 1},
}


@lru_cache(maxsize=4)
def load_artifacts(models_dir: str | Path = MODELS_DIR) -> dict[str, Any]:
    """Load trained joblib pipelines and evaluation metrics JSON."""
    path = Path(models_dir)
    lr_model = joblib.load(path / "logistic_regression.joblib")
    ann_model = joblib.load(path / "ann_mlp.joblib")
    with open(path / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {"logistic": lr_model, "ann": ann_model, "metrics": metrics}


def estimate_egfr_approx(sc: float, age: float) -> float:
    """Compute an approximate screening eGFR (average of male/female CKD-EPI 2021)."""
    sc_safe = max(0.2, float(sc))
    age_safe = max(18.0, float(age))
    # CKD-EPI 2021 creatinine equation without race
    egfr_f = 142 * (min(sc_safe / 0.7, 1.0) ** -0.241) * (max(sc_safe / 0.7, 1.0) ** -1.200) * (0.9938 ** age_safe) * 1.012
    egfr_m = 142 * (min(sc_safe / 0.9, 1.0) ** -0.302) * (max(sc_safe / 0.9, 1.0) ** -1.200) * (0.9938 ** age_safe)
    return round(float((egfr_f + egfr_m) / 2.0), 1)


def get_risk_band(score: float) -> dict[str, str]:
    """Map continuous risk score (0-100) to Low (<30), Moderate (30-60), or High (>60)."""
    if score < 30.0:
        return {"band": "Low", "badge": "success", "color": "#1FB5A8"}
    if score <= 60.0:
        return {"band": "Moderate", "badge": "warning", "color": "#F59E0B"}
    return {"band": "High", "badge": "danger", "color": "#FF6B6B"}


def get_ckd_stage_hint(risk_score: float, sc: float, age: float, al: float) -> dict[str, Any]:
    """Provide a screening-level CKD stage hint from estimated eGFR, albuminuria, and risk score."""
    egfr = estimate_egfr_approx(sc, age)
    if risk_score < 30.0 and egfr >= 90 and al == 0:
        stage = "No CKD Indicators"
        detail = f"Estimated screening eGFR ~{egfr} mL/min/1.73m² with normal albumin."
    elif egfr >= 90:
        stage = "Stage 1 Hint (G1)"
        detail = f"Preserved eGFR (~{egfr} mL/min/1.73m²), but elevated markers warrant monitoring."
    elif egfr >= 60:
        stage = "Stage 2 Hint (G2)"
        detail = f"Mildly reduced eGFR (~{egfr} mL/min/1.73m²); monitor blood pressure and proteinuria."
    elif egfr >= 30:
        stage = "Stage 3 Hint (G3a/G3b)"
        detail = f"Moderately reduced eGFR (~{egfr} mL/min/1.73m²); nephrology evaluation advised."
    elif egfr >= 15:
        stage = "Stage 4 Hint (G4)"
        detail = f"Severely reduced eGFR (~{egfr} mL/min/1.73m²); prompt specialist follow-up needed."
    else:
        stage = "Stage 5 Hint (G5)"
        detail = f"Critically reduced eGFR (~{egfr} mL/min/1.73m²); urgent medical attention required."
    return {"stage": stage, "egfr_approx": egfr, "detail": detail}


def compute_feature_contributions(
    features: dict[str, float],
    models: dict[str, Any],
    model_choice: str,
) -> list[dict[str, Any]]:
    """Compute signed per-feature contributions relative to healthy baseline medians.

    Uses counterfactual probability delta: P(patient) - P(patient with feature_i at healthy median),
    averaged across selected models so explanations work consistently for ANN, Logistic Regression,
    and consensus ('both'). Positive values mean the input pushed risk UP; negative pushed risk DOWN.
    """
    baseline = models["metrics"]["dataset"]["healthy_baseline"]
    row_df = pd.DataFrame([features], columns=FEATURE_NAMES)

    active_keys = ["logistic", "ann"] if model_choice == "both" else [model_choice]
    base_probs = [float(models[k].predict_proba(row_df)[0, 1]) for k in active_keys]
    mean_prob = float(np.mean(base_probs))

    contributions: list[dict[str, Any]] = []
    for feat in FEATURE_NAMES:
        cf = dict(features)
        cf[feat] = float(baseline[feat])
        cf_df = pd.DataFrame([cf], columns=FEATURE_NAMES)
        cf_prob = float(np.mean([float(models[k].predict_proba(cf_df)[0, 1]) for k in active_keys]))
        delta_pct = round((mean_prob - cf_prob) * 100.0, 2)
        contributions.append(
            {
                "feature": feat,
                "label": FEATURE_META[feat]["label"],
                "value": features[feat],
                "baseline": baseline[feat],
                "unit": FEATURE_META[feat]["unit"],
                "impact_pct": delta_pct,
                "direction": "increases_risk" if delta_pct > 0.2 else ("decreases_risk" if delta_pct < -0.2 else "neutral"),
            }
        )

    contributions.sort(key=lambda item: abs(item["impact_pct"]), reverse=True)
    return contributions


def build_diagnostic_suggestions(
    features: dict[str, float],
    risk_band: str,
) -> dict[str, list[str]]:
    """Generate personalized clinical screening suggestions across 4 actionable categories."""
    lifestyle: list[str] = []
    diet: list[str] = []
    nephrologist: list[str] = []
    tests: list[str] = []

    if risk_band == "High":
        nephrologist.append("Schedule an appointment with a nephrologist or primary care physician within 1–2 weeks.")
        nephrologist.append("Avoid NSAIDs (e.g., ibuprofen, naproxen) unless explicitly cleared by your physician.")
        tests.extend([
            "Comprehensive Metabolic Panel (CMP) with repeat Serum Creatinine & eGFR",
            "Urine Albumin-to-Creatinine Ratio (uACR) & full Urinalysis",
            "Renal Ultrasound to assess kidney morphology",
        ])
    elif risk_band == "Moderate":
        nephrologist.append("Book a routine physician review within 4 weeks to confirm lab trends.")
        nephrologist.append("Monitor blood pressure twice weekly and keep a log for your doctor.")
        tests.extend([
            "Repeat Serum Creatinine / eGFR in 4–8 weeks",
            "Spot Urine Albumin-to-Creatinine Ratio (uACR)",
            "HbA1c and Fasting Lipid Panel",
        ])
    else:
        nephrologist.append("Continue annual preventive health checkups; seek care sooner if swelling or urinary changes occur.")
        tests.extend([
            "Routine annual Basic Metabolic Panel (BMP)",
            "Annual urine dipstick or uACR if you have diabetes or hypertension",
        ])

    if features["bp"] >= 90 or features["htn"] == 1:
        lifestyle.append("Aim for 150 minutes/week of moderate aerobic exercise and daily blood pressure tracking.")
        diet.append("Limit sodium intake to under 2,000 mg/day (avoid processed and canned salty foods).")
    else:
        lifestyle.append("Maintain regular physical activity (30 min brisk walking most days) and quality sleep.")

    if features["bgr"] >= 140:
        diet.append("Prioritize low-glycemic whole foods and reduce refined sugars to protect renal microvessels.")
        if "HbA1c" not in " ".join(tests):
            tests.append("HbA1c test to evaluate 3-month glycemic control")

    if features["sc"] > 1.3 or features["bu"] > 45 or features["al"] >= 1:
        diet.append("Moderate excessive animal protein and stay adequately hydrated per your doctor's fluid guidance.")
    else:
        diet.append("Follow a balanced, kidney-friendly diet rich in fresh vegetables, whole grains, and adequate hydration.")

    if features["hemo"] < 12.0:
        tests.append("Complete Blood Count (CBC), Ferritin, and Iron Studies (screen for renal anemia)")
        diet.append("Discuss iron-rich foods or physician-guided supplementation if anemia is confirmed.")

    if features["sg"] <= 1.010:
        lifestyle.append("Discuss urine concentration findings with your clinician during follow-up.")

    return {
        "lifestyle": lifestyle,
        "diet": diet,
        "nephrologist": nephrologist,
        "tests": tests,
    }


def predict_ckd(
    features: dict[str, float],
    model_choice: str = "both",
    models_dir: str | Path = MODELS_DIR,
) -> dict[str, Any]:
    """Run calibrated CKD prediction, risk banding, stage hint, explanations, and advice."""
    choice = model_choice.lower().strip()
    if choice not in {"ann", "logistic", "both"}:
        choice = "both"

    cleaned = {feat: float(features[feat]) for feat in FEATURE_NAMES}
    row_df = pd.DataFrame([cleaned], columns=FEATURE_NAMES)
    models = load_artifacts(models_dir)

    ann_prob = float(models["ann"].predict_proba(row_df)[0, 1])
    lr_prob = float(models["logistic"].predict_proba(row_df)[0, 1])

    ann_score = round(ann_prob * 100.0, 1)
    lr_score = round(lr_prob * 100.0, 1)

    if choice == "ann":
        risk_score = ann_score
        model_label = "Calibrated ANN (MLP)"
    elif choice == "logistic":
        risk_score = lr_score
        model_label = "Calibrated Logistic Regression"
    else:
        risk_score = round(((ann_prob + lr_prob) / 2.0) * 100.0, 1)
        model_label = "Consensus (ANN + Logistic Regression)"

    band_info = get_risk_band(risk_score)
    stage_hint = get_ckd_stage_hint(risk_score, cleaned["sc"], cleaned["age"], cleaned["al"])
    contributions = compute_feature_contributions(cleaned, models, choice)
    suggestions = build_diagnostic_suggestions(cleaned, band_info["band"])

    return {
        "risk_score": risk_score,
        "risk_band": band_info["band"],
        "risk_badge": band_info["badge"],
        "risk_color": band_info["color"],
        "model_choice": choice,
        "model_label": model_label,
        "model_scores": {
            "ann": ann_score,
            "logistic": lr_score,
            "consensus": round(((ann_prob + lr_prob) / 2.0) * 100.0, 1),
        },
        "stage_hint": stage_hint,
        "contributions": contributions,
        "suggestions": suggestions,
        "inputs": cleaned,
    }
