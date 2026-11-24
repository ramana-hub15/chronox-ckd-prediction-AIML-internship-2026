"""Authentication and user profile/settings routes."""

from __future__ import annotations

from urllib.parse import urlparse

from flask import Blueprint, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app.extensions import limiter
from app.forms import LoginForm, PasswordChangeForm, ProfileForm, RegisterForm
from app.models import User, db

auth_bp = Blueprint("auth", __name__)


def _is_safe_redirect(target: str | None) -> bool:
    if not target:
        return False
    parsed = urlparse(target)
    return parsed.scheme == "" and parsed.netloc == ""


@auth_bp.route("/register", methods=["GET", "POST"])
@limiter.limit("15 per minute", methods=["POST"])
def register():
    """Handle user registration with password hashing and validation."""
    if current_user.is_authenticated:
        return redirect(url_for("clinical.dashboard"))

    form = RegisterForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        if User.query.filter_by(email=email).first():
            form.email.errors.append("An account with this email already exists.")
            return render_template("auth/register.html", form=form), 400

        user = User(
            full_name=form.full_name.data.strip(),
            email=email,
            theme_preference="light",
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash("Welcome to ChronoX! Your account has been created.", "success")
        return redirect(url_for("clinical.dashboard"))

    return render_template("auth/register.html", form=form)


@auth_bp.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute", methods=["POST"])
def login():
    """Handle user login with rate limiting and remember-me support."""
    if current_user.is_authenticated:
        return redirect(url_for("clinical.dashboard"))

    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        user = User.query.filter_by(email=email).first()
        if user is None or not user.check_password(form.password.data):
            flash("Invalid email or password. Please try again.", "danger")
            return render_template("auth/login.html", form=form), 401

        login_user(user, remember=bool(form.remember.data))
        flash(f"Welcome back, {user.full_name}.", "success")
        next_page = request.args.get("next")
        if _is_safe_redirect(next_page):
            return redirect(next_page)
        return redirect(url_for("clinical.dashboard"))

    return render_template("auth/login.html", form=form)


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    """Sign out the current user."""
    logout_user()
    flash("You have been signed out safely.", "info")
    return redirect(url_for("main.landing"))


@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """Manage profile details, theme preference, and password changes."""
    profile_form = ProfileForm(
        full_name=current_user.full_name,
        organization=current_user.organization,
        theme_preference=current_user.theme_preference,
    )
    password_form = PasswordChangeForm()

    action = request.form.get("action")
    if request.method == "POST" and action == "profile":
        if profile_form.validate_on_submit():
            current_user.full_name = profile_form.full_name.data.strip()
            current_user.organization = (profile_form.organization.data or "").strip()
            current_user.theme_preference = profile_form.theme_preference.data
            db.session.commit()
            flash("Profile settings updated.", "success")
            return redirect(url_for("auth.profile"))

    elif request.method == "POST" and action == "password":
        if password_form.validate_on_submit():
            if not current_user.check_password(password_form.current_password.data):
                password_form.current_password.errors.append("Current password is incorrect.")
            else:
                current_user.set_password(password_form.new_password.data)
                db.session.commit()
                flash("Password changed successfully.", "success")
                return redirect(url_for("auth.profile"))

    return render_template(
        "pages/profile.html",
        profile_form=profile_form,
        password_form=password_form,
    )


@auth_bp.route("/profile/theme", methods=["POST"])
@login_required
def toggle_theme():
    """Persist the user's light/dark theme selection."""
    payload = request.get_json(silent=True) or {}
    theme = payload.get("theme", "light")
    if theme not in {"light", "dark"}:
        return jsonify({"error": "Invalid theme"}), 400
    current_user.theme_preference = theme
    db.session.commit()
    return jsonify({"theme": theme})
