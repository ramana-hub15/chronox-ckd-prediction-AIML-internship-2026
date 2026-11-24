"""Flask application factory for ChronoX – Chronic Kidney Disease Prediction System."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for

from app.extensions import csrf, limiter, login_manager
from app.models import User, db

BASE_DIR = Path(__file__).resolve().parent.parent


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    """Create and configure the ChronoX Flask application."""
    load_dotenv(BASE_DIR / ".env")

    app = Flask(__name__, instance_relative_config=True)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    default_db_path = Path(app.instance_path) / "chronox.db"
    env_db_url = os.getenv("DATABASE_URL", f"sqlite:///{default_db_path.as_posix()}")

    app.config.from_mapping(
        APP_NAME=os.getenv("APP_NAME", "ChronoX"),
        SECRET_KEY=os.getenv("SECRET_KEY", "dev-fallback-key-replace-in-env"),
        SQLALCHEMY_DATABASE_URI=env_db_url,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        MODELS_DIR=str(BASE_DIR / "models"),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "False").lower() == "true",
        REMEMBER_COOKIE_HTTPONLY=True,
        REMEMBER_COOKIE_SAMESITE="Lax",
        RATELIMIT_STORAGE_URI=os.getenv("RATELIMIT_STORAGE_URI", "memory://"),
        RATELIMIT_HEADERS_ENABLED=True,
        WTF_CSRF_TIME_LIMIT=3600,
    )

    if test_config:
        app.config.update(test_config)

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)

    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please sign in to access your ChronoX clinical workspace."
    login_manager.login_message_category = "info"

    @login_manager.user_loader
    def load_user(user_id: str) -> User | None:
        return db.session.get(User, int(user_id))

    @login_manager.unauthorized_handler
    def unauthorized():
        if request.path.startswith("/api/"):
            return jsonify({"error": "Authentication required."}), 401
        return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))

    from app.routes.api import api_bp
    from app.routes.auth import auth_bp
    from app.routes.clinical import clinical_bp
    from app.routes.main import main_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(clinical_bp)
    app.register_blueprint(api_bp)

    @app.errorhandler(404)
    def not_found(error):  # noqa: ARG001
        if request.path.startswith("/api/"):
            return jsonify({"error": "Resource not found."}), 404
        return render_template("errors/404.html"), 404

    @app.errorhandler(429)
    def rate_limited(error):  # noqa: ARG001
        if request.path.startswith("/api/"):
            return jsonify({"error": "Too many requests. Please slow down."}), 429
        return render_template("errors/429.html"), 429

    @app.errorhandler(500)
    def internal_error(error):  # noqa: ARG001
        db.session.rollback()
        if request.path.startswith("/api/"):
            return jsonify({"error": "Internal server error."}), 500
        return render_template("errors/500.html"), 500

    with app.app_context():
        db.create_all()
        if not app.config.get("SKIP_SEED", False):
            from app.seed import seed_demo_data

            seed_demo_data()

    return app
