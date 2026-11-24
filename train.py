"""Training and evaluation pipeline for ChronoX CKD prediction models."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.ml.preprocessing import (  # noqa: E402
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURE_NAMES,
    ORDINAL_FEATURES,
    ClinicalOutlierClipper,
)

DATA_PATH = BASE_DIR / "data" / "kidney_disease_uci.csv"
MODELS_DIR = BASE_DIR / "models"

FEATURE_LABELS: dict[str, str] = {
    "age": "Age (years)",
    "bp": "Blood Pressure (mmHg)",
    "sg": "Specific Gravity",
    "al": "Albumin (0–5)",
    "bgr": "Blood Glucose Random (mg/dL)",
    "bu": "Blood Urea (mg/dL)",
    "sc": "Serum Creatinine (mg/dL)",
    "hemo": "Hemoglobin (g/dL)",
    "htn": "Hypertension",
}


def generate_synthetic_ckd(n_samples: int = 400, random_state: int = 42) -> pd.DataFrame:
    """Fallback synthetic CKD generator matching the UCI 9-feature schema."""
    rng = np.random.default_rng(random_state)
    n_ckd = int(n_samples * 0.625)
    n_healthy = n_samples - n_ckd

    def make_group(n: int, is_ckd: bool) -> pd.DataFrame:
        if is_ckd:
            age = np.clip(rng.normal(56, 15, n), 12, 88)
            bp = np.clip(rng.normal(82, 14, n), 50, 140)
            sg = rng.choice([1.005, 1.010, 1.015, 1.020], size=n, p=[0.12, 0.45, 0.30, 0.13])
            al = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.14, 0.22, 0.24, 0.22, 0.15, 0.03])
            bgr = np.clip(rng.normal(175, 85, n), 70, 480)
            bu = np.clip(rng.lognormal(4.1, 0.6, n), 18, 250)
            sc = np.clip(rng.lognormal(1.1, 0.75, n), 0.8, 24.0)
            hemo = np.clip(rng.normal(10.6, 2.1, n), 4.5, 15.5)
            htn = rng.choice(["yes", "no"], size=n, p=[0.62, 0.38])
            cls = ["ckd"] * n
        else:
            age = np.clip(rng.normal(45, 14, n), 18, 78)
            bp = np.clip(rng.normal(71, 8, n), 60, 85)
            sg = rng.choice([1.020, 1.025], size=n, p=[0.48, 0.52])
            al = np.zeros(n)
            bgr = np.clip(rng.normal(105, 18, n), 70, 145)
            bu = np.clip(rng.normal(31, 9, n), 12, 50)
            sc = np.clip(rng.normal(0.9, 0.2, n), 0.4, 1.3)
            hemo = np.clip(rng.normal(15.1, 1.3, n), 12.5, 17.8)
            htn = ["no"] * n
            cls = ["notckd"] * n
        return pd.DataFrame(
            {
                "age": np.round(age, 1),
                "bp": np.round(bp / 10.0) * 10.0,
                "sg": sg,
                "al": al,
                "bgr": np.round(bgr, 0),
                "bu": np.round(bu, 1),
                "sc": np.round(sc, 2),
                "hemo": np.round(hemo, 1),
                "htn": htn,
                "classification": cls,
            }
        )

    df = pd.concat([make_group(n_ckd, True), make_group(n_healthy, False)], ignore_index=True)
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def load_dataset() -> tuple[pd.DataFrame, pd.Series, str]:
    """Load and clean the 9-feature UCI CKD dataset (or synthetic fallback)."""
    source = "UCI Chronic Kidney Disease Dataset (OpenML ID 42972, n=400)"
    if DATA_PATH.exists():
        raw_df = pd.read_csv(DATA_PATH)
    else:
        try:
            from sklearn.datasets import fetch_openml

            fetched = fetch_openml(data_id=42972, as_frame=True, parser="auto")
            raw_df = fetched.frame
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            raw_df.to_csv(DATA_PATH, index=False)
        except Exception:
            raw_df = generate_synthetic_ckd()
            source = "Synthetic Clinical CKD Dataset (UCI Schema Compatible, n=400)"

    df = raw_df[FEATURE_NAMES + ["classification"]].copy()

    # Clean hypertension to binary 1/0 while preserving NaNs for pipeline imputation
    def parse_htn(val: object) -> float:
        if pd.isna(val):
            return np.nan
        cleaned = str(val).strip().strip("'\"").lower()
        if cleaned in {"yes", "1", "true"}:
            return 1.0
        if cleaned in {"no", "0", "false"}:
            return 0.0
        return np.nan

    df["htn"] = df["htn"].map(parse_htn)

    # Clean classification labels (UCI dataset contains "'ckdt'" / "ckd\t" artifacts)
    def parse_target(val: object) -> int:
        cleaned = str(val).strip().strip("'\"").lower()
        return 0 if "notckd" in cleaned else 1

    y = df["classification"].map(parse_target).astype(int)
    X = df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    return X, y, source


def build_preprocessor() -> ColumnTransformer:
    """Create the scikit-learn ColumnTransformer with imputation, outlier clipping, and scaling."""
    continuous_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("outlier_clipper", ClinicalOutlierClipper(lower_quantile=0.01, upper_quantile=0.99)),
            ("scaler", StandardScaler()),
        ]
    )
    ordinal_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipe, CONTINUOUS_FEATURES),
            ("ordinal", ordinal_pipe, ORDINAL_FEATURES),
            ("binary", binary_pipe, BINARY_FEATURES),
        ],
        remainder="drop",
    )


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold,
) -> dict:
    """Compute held-out test metrics, ROC curve, confusion matrix, 5-fold CV, and importance."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    perm = permutation_importance(
        model, X_test, y_test, n_repeats=25, random_state=42, scoring="roc_auc"
    )
    importances = {
        feat: round(float(max(0.0, imp)), 4)
        for feat, imp in zip(FEATURE_NAMES, perm.importances_mean)
    }

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr],
        },
        "cv_5fold": {
            "accuracy_mean": round(float(cv_acc.mean()), 4),
            "accuracy_std": round(float(cv_acc.std()), 4),
            "f1_mean": round(float(cv_f1.mean()), 4),
            "f1_std": round(float(cv_f1.std()), 4),
            "roc_auc_mean": round(float(cv_auc.mean()), 4),
            "roc_auc_std": round(float(cv_auc.std()), 4),
        },
        "permutation_importance": importances,
    }


def extract_logistic_coefficients(pipeline: Pipeline) -> dict[str, float]:
    """Average logistic regression coefficients across CalibratedClassifierCV folds."""
    calibrated: CalibratedClassifierCV = pipeline.named_steps["classifier"]
    transformed_names = CONTINUOUS_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES
    coefs = np.mean([cc.estimator.coef_[0] for cc in calibrated.calibrated_classifiers_], axis=0)
    mapping = {name: round(float(val), 4) for name, val in zip(transformed_names, coefs)}
    return {feat: mapping[feat] for feat in FEATURE_NAMES}


def train_and_save() -> dict:
    """Train calibrated Logistic Regression and ANN models, evaluate, and persist artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, dataset_source = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
                    method="sigmoid",
                    cv=5,
                ),
            ),
        ]
    )

    ann_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=0.05,
                        learning_rate_init=0.003,
                        max_iter=1000,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=20,
                        random_state=42,
                    ),
                    method="sigmoid",
                    cv=5,
                ),
            ),
        ]
    )

    lr_pipeline.fit(X_train, y_train)
    ann_pipeline.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_metrics = evaluate_model(lr_pipeline, X, y, X_train, y_train, X_test, y_test, cv)
    lr_metrics["coefficients"] = extract_logistic_coefficients(lr_pipeline)
    lr_metrics["name"] = "Calibrated Logistic Regression"
    lr_metrics["short_name"] = "Logistic Regression"

    ann_metrics = evaluate_model(ann_pipeline, X, y, X_train, y_train, X_test, y_test, cv)
    ann_metrics["name"] = "Calibrated ANN (MLP 64-32)"
    ann_metrics["short_name"] = "ANN (MLP)"

    # Compute healthy reference medians (from non-CKD rows) for counterfactual feature attribution
    healthy_mask = y == 0
    healthy_baseline = {
        feat: round(float(X.loc[healthy_mask, feat].median()), 3) for feat in FEATURE_NAMES
    }

    metrics_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source": dataset_source,
            "total_samples": int(len(X)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "ckd_positive": int((y == 1).sum()),
            "ckd_negative": int((y == 0).sum()),
            "features": FEATURE_NAMES,
            "feature_labels": FEATURE_LABELS,
            "healthy_baseline": healthy_baseline,
        },
        "models": {
            "logistic": lr_metrics,
            "ann": ann_metrics,
        },
    }

    joblib.dump(lr_pipeline, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(ann_pipeline, MODELS_DIR / "ann_mlp.joblib")
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return metrics_payload


if __name__ == "__main__":
    results = train_and_save()
    print("=== ChronoX Training Complete ===")
    print("Dataset:", results["dataset"]["source"])
    for key in ("logistic", "ann"):
        m = results["models"][key]
        print(
            f"{m['name']}: Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, "
            f"Rec={m['recall']:.4f}, F1={m['f1']:.4f}, AUC={m['roc_auc']:.4f}, "
            f"CV5-AUC={m['cv_5fold']['roc_auc_mean']:.4f}±{m['cv_5fold']['roc_auc_std']:.4f}"
        )
