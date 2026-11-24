"""Public, educational, and PWA routes for ChronoX."""

from __future__ import annotations

from flask import Blueprint, Response, current_app, flash, redirect, render_template, send_from_directory, url_for

from app.forms import ContactForm
from app.ml.predictor import load_artifacts
from app.models import ContactMessage, db

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def landing() -> str:
    """Render the ChronoX landing page with live model metrics highlights."""
    metrics = load_artifacts(current_app.config["MODELS_DIR"])["metrics"]
    return render_template("pages/landing.html", metrics=metrics)


@main_bp.route("/education")
def education() -> str:
    """Render CKD education, 5 stages, risk factors, prevention, and FAQs."""
    return render_template("pages/education.html")


@main_bp.route("/about")
def about() -> str:
    """Render the About ChronoX & clinical engineering methodology page."""
    metrics = load_artifacts(current_app.config["MODELS_DIR"])["metrics"]
    return render_template("pages/about.html", metrics=metrics)


@main_bp.route("/contact", methods=["GET", "POST"])
def contact():
    """Render and handle the contact inquiry form."""
    form = ContactForm()
    if form.validate_on_submit():
        msg = ContactMessage(
            name=form.name.data.strip(),
            email=form.email.data.strip().lower(),
            subject=form.subject.data.strip(),
            message=form.message.data.strip(),
        )
        db.session.add(msg)
        db.session.commit()
        flash("Thank you for reaching out. Your message has been recorded.", "success")
        return redirect(url_for("main.contact"))
    return render_template("pages/contact.html", form=form)


@main_bp.route("/privacy")
def privacy() -> str:
    """Render privacy policy and medical disclaimer page."""
    return render_template("pages/privacy.html")


@main_bp.route("/offline")
def offline() -> str:
    """Render PWA offline fallback page."""
    return render_template("pages/offline.html")


@main_bp.route("/manifest.json")
def manifest():
    """Serve the Web App Manifest at root scope."""
    return send_from_directory(
        current_app.static_folder,
        "manifest.json",
        mimetype="application/manifest+json",
    )


@main_bp.route("/sw.js")
def service_worker():
    """Serve the PWA Service Worker at root scope."""
    response = send_from_directory(
        current_app.static_folder,
        "js/sw.js",
        mimetype="application/javascript",
    )
    response.headers["Service-Worker-Allowed"] = "/"
    response.headers["Cache-Control"] = "no-cache"
    return response
