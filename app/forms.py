"""WTForms definitions with server-side validation for ChronoX."""

from __future__ import annotations

from flask_wtf import FlaskForm
from wtforms import (
    BooleanField,
    FloatField,
    IntegerField,
    PasswordField,
    SelectField,
    StringField,
    SubmitField,
    TextAreaField,
)
from wtforms.validators import (
    DataRequired,
    Email,
    EqualTo,
    Length,
    NumberRange,
    Optional,
)


class RegisterForm(FlaskForm):
    """User registration form."""

    full_name = StringField(
        "Full Name",
        validators=[DataRequired(), Length(min=2, max=120)],
    )
    email = StringField(
        "Email Address",
        validators=[DataRequired(), Email(), Length(max=180)],
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired(), Length(min=8, max=128)],
    )
    confirm_password = PasswordField(
        "Confirm Password",
        validators=[DataRequired(), EqualTo("password", message="Passwords must match.")],
    )
    submit = SubmitField("Create Account")


class LoginForm(FlaskForm):
    """User login form."""

    email = StringField(
        "Email Address",
        validators=[DataRequired(), Email(), Length(max=180)],
    )
    password = PasswordField("Password", validators=[DataRequired()])
    remember = BooleanField("Remember me")
    submit = SubmitField("Sign In")


class PredictionForm(FlaskForm):
    """Nine-parameter clinical CKD screening form."""

    patient_label = StringField(
        "Screening Label",
        default="Clinical Screening",
        validators=[Optional(), Length(max=120)],
    )
    model_choice = SelectField(
        "Prediction Model",
        choices=[
            ("both", "Consensus (ANN + Logistic Regression)"),
            ("ann", "Calibrated ANN (MLP Neural Network)"),
            ("logistic", "Calibrated Logistic Regression"),
        ],
        default="both",
        validators=[DataRequired()],
    )
    age = FloatField(
        "Age (years)",
        default=48,
        validators=[DataRequired(), NumberRange(min=1, max=110)],
    )
    bp = FloatField(
        "Blood Pressure — Diastolic (mmHg)",
        default=80,
        validators=[DataRequired(), NumberRange(min=40, max=200)],
    )
    sg = SelectField(
        "Urine Specific Gravity",
        choices=[
            ("1.005", "1.005 (Very Dilute)"),
            ("1.010", "1.010 (Low)"),
            ("1.015", "1.015 (Borderline)"),
            ("1.020", "1.020 (Normal)"),
            ("1.025", "1.025 (Normal Concentrated)"),
        ],
        default="1.020",
        validators=[DataRequired()],
    )
    al = IntegerField(
        "Urine Albumin (0–5)",
        default=0,
        validators=[NumberRange(min=0, max=5)],
    )
    bgr = FloatField(
        "Blood Glucose Random (mg/dL)",
        default=110,
        validators=[DataRequired(), NumberRange(min=20, max=600)],
    )
    bu = FloatField(
        "Blood Urea (mg/dL)",
        default=32,
        validators=[DataRequired(), NumberRange(min=1, max=400)],
    )
    sc = FloatField(
        "Serum Creatinine (mg/dL)",
        default=1.0,
        validators=[DataRequired(), NumberRange(min=0.2, max=80)],
    )
    hemo = FloatField(
        "Hemoglobin (g/dL)",
        default=14.2,
        validators=[DataRequired(), NumberRange(min=2.5, max=22)],
    )
    htn = SelectField(
        "Hypertension",
        choices=[("0", "No"), ("1", "Yes")],
        default="0",
        validators=[DataRequired()],
    )
    submit = SubmitField("Analyze CKD Risk")


class ProfileForm(FlaskForm):
    """Account profile and theme update form."""

    full_name = StringField(
        "Full Name",
        validators=[DataRequired(), Length(min=2, max=120)],
    )
    organization = StringField(
        "Clinic / Organization",
        validators=[Optional(), Length(max=120)],
    )
    theme_preference = SelectField(
        "Preferred Theme",
        choices=[("light", "Light Clinical Theme"), ("dark", "Dark Mode")],
        validators=[DataRequired()],
    )
    submit = SubmitField("Save Profile")


class PasswordChangeForm(FlaskForm):
    """Change account password form."""

    current_password = PasswordField("Current Password", validators=[DataRequired()])
    new_password = PasswordField(
        "New Password",
        validators=[DataRequired(), Length(min=8, max=128)],
    )
    confirm_password = PasswordField(
        "Confirm New Password",
        validators=[DataRequired(), EqualTo("new_password", message="Passwords must match.")],
    )
    submit = SubmitField("Update Password")


class ContactForm(FlaskForm):
    """Contact inquiry form."""

    name = StringField("Your Name", validators=[DataRequired(), Length(min=2, max=120)])
    email = StringField("Email Address", validators=[DataRequired(), Email(), Length(max=180)])
    subject = StringField("Subject", validators=[DataRequired(), Length(min=3, max=200)])
    message = TextAreaField("Message", validators=[DataRequired(), Length(min=10, max=2000)])
    submit = SubmitField("Send Message")
