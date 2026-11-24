"""End-to-end tests for ChronoX authentication, ML inference, and REST APIs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import create_app  # noqa: E402
from app.models import User, db  # noqa: E402


@pytest.fixture()
def app(tmp_path):
    db_file = tmp_path / "test_chronox.db"
    application = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_file.as_posix()}",
            "WTF_CSRF_ENABLED": False,
            "RATELIMIT_ENABLED": False,
            "SKIP_SEED": False,
        }
    )
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()


def _login_demo(client):
    return client.post(
        "/login",
        data={"email": "demo@chronox.com", "password": "Demo@1234"},
        follow_redirects=True,
    )


def test_public_pages_render(client):
    for route in ["/", "/education", "/about", "/contact", "/privacy", "/analytics", "/comparison", "/manifest.json", "/sw.js"]:
        resp = client.get(route)
        assert resp.status_code == 200, f"Failed on {route}"


def test_custom_404_renders(client):
    resp = client.get("/nonexistent-page-xyz")
    assert resp.status_code == 404
    assert b"Page Not Found" in resp.data


def test_register_and_login_flow(client, app):
    reg = client.post(
        "/register",
        data={
            "full_name": "Alex Rivera",
            "email": "alex@example.com",
            "password": "Password123!",
            "confirm_password": "Password123!",
        },
        follow_redirects=True,
    )
    assert reg.status_code == 200
    assert b"Alex Rivera" in reg.data

    with app.app_context():
        user = User.query.filter_by(email="alex@example.com").first()
        assert user is not None
        assert user.password_hash != "Password123!"
        assert user.check_password("Password123!")

    logout = client.post("/logout", follow_redirects=True)
    assert logout.status_code == 200

    bad_login = client.post(
        "/login",
        data={"email": "alex@example.com", "password": "wrongpassword"},
    )
    assert bad_login.status_code == 401


def test_api_requires_auth_for_predict_and_history(client):
    assert client.get("/api/metrics").status_code == 200
    assert client.get("/api/history").status_code == 401
    assert client.post("/api/predict", json={}).status_code == 401


def test_api_predict_and_history_workflow(client):
    login_resp = _login_demo(client)
    assert login_resp.status_code == 200

    # High-risk profile
    high_payload = {
        "patient_label": "Pytest High Risk",
        "model_choice": "both",
        "age": 64,
        "bp": 100,
        "sg": 1.010,
        "al": 3,
        "bgr": 240,
        "bu": 88,
        "sc": 3.5,
        "hemo": 9.2,
        "htn": 1,
    }
    resp = client.post("/api/predict", json=high_payload)
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["risk_band"] == "High"
    assert data["risk_score"] > 60.0
    assert len(data["contributions"]) == 9
    pred_id = data["prediction_id"]

    # Low-risk profile using ANN only
    low_payload = {
        "patient_label": "Pytest Low Risk",
        "model_choice": "ann",
        "age": 32,
        "bp": 70,
        "sg": 1.025,
        "al": 0,
        "bgr": 92,
        "bu": 22,
        "sc": 0.8,
        "hemo": 15.2,
        "htn": 0,
    }
    resp_low = client.post("/api/predict", json=low_payload)
    assert resp_low.status_code == 201
    low_data = resp_low.get_json()
    assert low_data["risk_band"] == "Low"
    assert low_data["risk_score"] < 30.0

    # Validation failure
    bad_resp = client.post("/api/predict", json={"age": 999})
    assert bad_resp.status_code == 400

    # History API includes newly created prediction
    hist = client.get("/api/history?limit=20").get_json()
    ids = [item["id"] for item in hist["predictions"]]
    assert pred_id in ids

    # Result page + PDF export + CSV export + Dashboard
    assert client.get("/dashboard").status_code == 200
    assert client.get(f"/result/{pred_id}").status_code == 200
    pdf_resp = client.get(f"/result/{pred_id}/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.mimetype == "application/pdf"
    csv_resp = client.get("/history/export")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.mimetype
