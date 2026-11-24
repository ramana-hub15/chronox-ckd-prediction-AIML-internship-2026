"""SQLAlchemy database models for ChronoX."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """Registered user account with password hashing and UI preferences."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(180), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    organization = db.Column(db.String(120), default="", nullable=False)
    theme_preference = db.Column(db.String(16), default="light", nullable=False)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    predictions = db.relationship(
        "Prediction",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="Prediction.created_at.desc()",
    )

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def __repr__(self) -> str:
        return f"<User {self.email}>"


class Prediction(db.Model):
    """Stored CKD screening prediction and clinical inputs."""

    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    patient_label = db.Column(db.String(120), default="Self Screening", nullable=False)
    model_used = db.Column(db.String(24), default="both", nullable=False)

    # Nine clinical features
    age = db.Column(db.Float, nullable=False)
    bp = db.Column(db.Float, nullable=False)
    sg = db.Column(db.Float, nullable=False)
    al = db.Column(db.Float, nullable=False)
    bgr = db.Column(db.Float, nullable=False)
    bu = db.Column(db.Float, nullable=False)
    sc = db.Column(db.Float, nullable=False)
    hemo = db.Column(db.Float, nullable=False)
    htn = db.Column(db.Integer, nullable=False)

    # Prediction outputs
    risk_score = db.Column(db.Float, nullable=False)
    ann_score = db.Column(db.Float, nullable=False)
    logistic_score = db.Column(db.Float, nullable=False)
    risk_band = db.Column(db.String(24), nullable=False)
    stage_hint = db.Column(db.String(120), nullable=False)
    egfr_approx = db.Column(db.Float, nullable=False)
    contributions_json = db.Column(db.Text, nullable=False)
    suggestions_json = db.Column(db.Text, nullable=False)

    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    user = db.relationship("User", back_populates="predictions")

    @property
    def contributions(self) -> list[dict[str, Any]]:
        return json.loads(self.contributions_json)

    @property
    def suggestions(self) -> dict[str, list[str]]:
        return json.loads(self.suggestions_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "patient_label": self.patient_label,
            "model_used": self.model_used,
            "inputs": {
                "age": self.age,
                "bp": self.bp,
                "sg": self.sg,
                "al": self.al,
                "bgr": self.bgr,
                "bu": self.bu,
                "sc": self.sc,
                "hemo": self.hemo,
                "htn": self.htn,
            },
            "risk_score": self.risk_score,
            "ann_score": self.ann_score,
            "logistic_score": self.logistic_score,
            "risk_band": self.risk_band,
            "stage_hint": self.stage_hint,
            "egfr_approx": self.egfr_approx,
            "contributions": self.contributions,
            "suggestions": self.suggestions,
            "created_at": self.created_at.isoformat(),
        }


class ContactMessage(db.Model):
    """Contact form submissions persisted for review."""

    __tablename__ = "contact_messages"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(180), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
