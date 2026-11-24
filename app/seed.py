"""Database seeding for demo user and sample prediction history."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from app.ml.predictor import predict_ckd
from app.models import Prediction, User, db

DEMO_SAMPLES = [
    ("Routine Wellness Check", "both", {"age": 34, "bp": 70, "sg": 1.025, "al": 0, "bgr": 92, "bu": 22, "sc": 0.8, "hemo": 15.1, "htn": 0}, 28),
    ("Annual Checkup #2", "both", {"age": 41, "bp": 80, "sg": 1.020, "al": 0, "bgr": 108, "bu": 29, "sc": 0.9, "hemo": 14.3, "htn": 0}, 23),
    ("Hypertension Follow-Up", "logistic", {"age": 54, "bp": 90, "sg": 1.015, "al": 1, "bgr": 136, "bu": 46, "sc": 1.4, "hemo": 12.6, "htn": 1}, 19),
    ("Diabetic Screening Panel", "ann", {"age": 58, "bp": 80, "sg": 1.015, "al": 2, "bgr": 218, "bu": 52, "sc": 1.6, "hemo": 11.8, "htn": 1}, 15),
    ("Mild Fatigue Workup", "both", {"age": 49, "bp": 80, "sg": 1.020, "al": 1, "bgr": 124, "bu": 38, "sc": 1.2, "hemo": 12.9, "htn": 0}, 12),
    ("Post-Hydration Retest", "both", {"age": 46, "bp": 70, "sg": 1.025, "al": 0, "bgr": 98, "bu": 26, "sc": 0.85, "hemo": 14.8, "htn": 0}, 9),
    ("Elevated Creatinine Alert", "both", {"age": 65, "bp": 100, "sg": 1.010, "al": 3, "bgr": 245, "bu": 88, "sc": 3.2, "hemo": 9.6, "htn": 1}, 6),
    ("Nephrology Referral Pre-Screen", "ann", {"age": 69, "bp": 90, "sg": 1.010, "al": 4, "bgr": 190, "bu": 110, "sc": 4.5, "hemo": 8.9, "htn": 1}, 4),
    ("BP Medication Review", "logistic", {"age": 52, "bp": 80, "sg": 1.015, "al": 1, "bgr": 130, "bu": 42, "sc": 1.3, "hemo": 13.1, "htn": 1}, 2),
    ("Baseline Healthy Profile", "both", {"age": 29, "bp": 70, "sg": 1.025, "al": 0, "bgr": 88, "bu": 20, "sc": 0.75, "hemo": 15.6, "htn": 0}, 0),
]


def seed_demo_data() -> None:
    """Create the demo user and sample prediction records if absent."""
    demo_email = "demo@chronox.com"
    demo_user = User.query.filter_by(email=demo_email).first()
    if demo_user is None:
        demo_user = User(
            full_name="Dr. Maya Patel (Demo)",
            email=demo_email,
            organization="ChronoX Nephrology Clinic",
            theme_preference="light",
        )
        demo_user.set_password("Demo@1234")
        db.session.add(demo_user)
        db.session.commit()

    if Prediction.query.filter_by(user_id=demo_user.id).count() == 0:
        now = datetime.now(timezone.utc)
        for label, model_choice, params, days_ago in DEMO_SAMPLES:
            result = predict_ckd(params, model_choice=model_choice)
            record = Prediction(
                user_id=demo_user.id,
                patient_label=label,
                model_used=model_choice,
                age=params["age"],
                bp=params["bp"],
                sg=params["sg"],
                al=params["al"],
                bgr=params["bgr"],
                bu=params["bu"],
                sc=params["sc"],
                hemo=params["hemo"],
                htn=int(params["htn"]),
                risk_score=result["risk_score"],
                ann_score=result["model_scores"]["ann"],
                logistic_score=result["model_scores"]["logistic"],
                risk_band=result["risk_band"],
                stage_hint=result["stage_hint"]["stage"],
                egfr_approx=result["stage_hint"]["egfr_approx"],
                contributions_json=json.dumps(result["contributions"]),
                suggestions_json=json.dumps(result["suggestions"]),
                created_at=now - timedelta(days=days_ago, hours=(days_ago % 7) + 1),
            )
            db.session.add(record)
        db.session.commit()
