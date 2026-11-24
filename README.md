# ChronoX – Chronic Kidney Disease Prediction System

ChronoX is a production-quality, mobile-first Flask & Scikit-learn web application that predicts Chronic Kidney Disease (CKD) risk from nine clinical biomarkers using **Calibrated Artificial Neural Network (ANN)** and **Calibrated Logistic Regression** models.

> **Medical Disclaimer:** This tool is for educational and screening support only and is not a substitute for professional medical diagnosis.

---

## Features

- **Dual Calibrated ML Engine:** `MLPClassifier` (ANN 64–32) and `LogisticRegression` wrapped in 5-fold sigmoid `CalibratedClassifierCV` inside leakage-free Scikit-learn `Pipeline`s (median/mode imputation, quantile outlier winsorization, and `StandardScaler`).
- **Continuous Risk Score & Stage Hint:** Outputs 0–100% calibrated risk segmented into **Low (<30%)**, **Moderate (30–60%)**, and **High (>60%)** bands, plus an approximate CKD-EPI screening eGFR stage hint.
- **Per-Biomarker Explanations:** Counterfactual feature contribution analysis showing exactly which inputs raised or lowered risk, visualized with interactive charts.
- **Diagnostic Suggestions & PDF Export:** Personalized lifestyle, renal diet, follow-up lab test, and nephrologist referral guidance, plus one-click ReportLab PDF report generation.
- **Interactive Analytics & Model Comparison:** Chart.js ROC curves, confusion-matrix heatmaps, accuracy/precision/recall/F1 comparisons, permutation importance, risk distribution doughnut, and chronological risk history.
- **Clinical Workspace & History:** Dashboard KPIs, 3-step screening wizard with synced sliders and sample patient presets, searchable/filterable/sortable history table, CSV export, and dark mode persistence.
- **Security & PWA:** Salted Werkzeug password hashing, Flask-WTF CSRF protection, Flask-Limiter login rate limiting, HTTPOnly cookies, Web App Manifest, Service Worker, and offline fallback.

---

## Project Structure

```text
chronox/
  app/
    __init__.py          # Application factory, security config, seeding
    extensions.py        # LoginManager, CSRFProtect, Limiter
    forms.py             # WTForms with server-side validation
    models.py            # SQLAlchemy User, Prediction, ContactMessage models
    seed.py              # Demo user & 10-record history seeder
    ml/
      preprocessing.py   # ClinicalOutlierClipper & feature definitions
      predictor.py       # Inference, calibration scoring, stage hint, explanations
    routes/
      main.py            # Landing, education, about, contact, privacy, PWA
      auth.py            # Register, login, logout, profile/settings
      clinical.py        # Dashboard, predict wizard, result, PDF, analytics, history
      api.py             # REST endpoints (/api/predict, /api/metrics, /api/history)
    utils/
      pdf_report.py      # ReportLab PDF clinical summary generator
    templates/           # Bootstrap 5.3 + Jinja2 responsive templates
    static/              # CSS, JS, PWA manifest, service worker, icons
  data/
    kidney_disease_uci.csv
  models/
    ann_mlp.joblib
    logistic_regression.joblib
    metrics.json
  tests/
    test_chronox.py
  train.py
  run.py
  requirements.txt
  .env.example
  README.md
```

---

## Dataset & Models

- **Dataset:** UCI Chronic Kidney Disease Dataset (OpenML ID `42972`, `n = 400`, 250 CKD / 150 Not-CKD), stored locally at `data/kidney_disease_uci.csv`. If the CSV is absent and the machine is offline, `train.py` automatically falls back to a schema-compatible synthetic CKD generator.
- **Nine Clinical Parameters:**
  1. `age` — Age (years)
  2. `bp` — Blood Pressure, diastolic (mmHg)
  3. `sg` — Urine Specific Gravity (`1.005`–`1.025`)
  4. `al` — Urine Albumin (`0`–`5`)
  5. `bgr` — Blood Glucose Random (mg/dL)
  6. `bu` — Blood Urea (mg/dL)
  7. `sc` — Serum Creatinine (mg/dL)
  8. `hemo` — Hemoglobin (g/dL)
  9. `htn` — Hypertension (`yes`/`no`)

---

## Quick Setup

```bash
cd chronox
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

# (Optional) Retrain models & regenerate metrics.json
python train.py

# Run the application
python run.py
```

Open `http://127.0.0.1:5000` on your laptop, or `http://<LAPTOP-IP>:5000` on a phone connected to the same Wi-Fi/hotspot (Flask listens on `0.0.0.0:5000` by default; find `<LAPTOP-IP>` with `ipconfig`).

### Demo Credentials
- **Email:** `demo@chronox.com`
- **Password:** `Demo@1234`

### Running Tests

```bash
pytest -v
```

### REST API Endpoints

- `GET /api/metrics` — Model evaluation metrics, ROC curves, confusion matrices, and 5-fold CV stats.
- `POST /api/predict` — Authenticated JSON prediction endpoint accepting the 9 clinical fields + optional `model_choice` (`ann`, `logistic`, `both`).
- `GET /api/history` — Authenticated JSON screening history with optional `band`, `model`, and `limit` query parameters.

---

## Screenshots

- **Landing Page & Hero SVG:** Responsive dark-blue/teal gradient hero with animated nephrology SVG and live KPI counters.
- **3-Step Screening Wizard:** Mobile-friendly sliders, normal-range hints, and sample patient presets.
- **Result & Explainability View:** Animated semicircular risk gauge, eGFR stage hint, feature contribution bar chart, and 4-category diagnostic guidance.
- **Analytics Dashboard:** Chart.js ROC curves, confusion-matrix heatmaps, permutation importance, risk distribution doughnut, and chronological risk trendline.

---

## Resume-Ready Project Description

> **ChronoX – Chronic Kidney Disease Prediction System** | *Python, Flask, Scikit-learn, SQLAlchemy, Bootstrap 5, Chart.js*
> - Engineered an end-to-end clinical ML web application predicting Chronic Kidney Disease risk from nine biomarkers using sigmoid-calibrated Artificial Neural Network (`MLPClassifier`) and `LogisticRegression` pipelines (98.75% test accuracy, 1.000 ANN ROC-AUC, 5-fold stratified CV).
> - Built a leakage-free `ColumnTransformer` preprocessing pipeline with median/mode imputation, custom quantile outlier clipping, `StandardScaler`, and counterfactual per-biomarker risk attribution.
> - Developed a secure, mobile-first PWA using Flask Blueprints, SQLite/SQLAlchemy, Werkzeug auth, CSRF/rate-limiting, interactive Chart.js analytics, ReportLab PDF clinical reports, and automated `pytest` coverage.
