"""
Training script for the Fraud Detection XGBoost pipeline.
"""
# imports
import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from app.config import DECISION_THRESHOLD
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "isFraud"
NUMERIC_FEATURES = ["TransactionAmt", "C13", "C1", "C14"]
CATEGORICAL_FEATURES = ["card4", "card6", "P_emaildomain"]
PRIORITY_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train")

# Load the dataset from a Parquet file and validate required columns
def load_data(data_path: Path) -> pd.DataFrame:
    logger.info(f"Loading dataset from {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Dataset shape: {df.shape}")

    missing = [f for f in PRIORITY_FEATURES + [TARGET] if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df

# Split the dataset into training and testing sets and handle missing values in categorical features
def split_data(df: pd.DataFrame, test_size: float):
    X = df[PRIORITY_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[CATEGORICAL_FEATURES] = X_train[CATEGORICAL_FEATURES].fillna("missing")
    X_test[CATEGORICAL_FEATURES] = X_test[CATEGORICAL_FEATURES].fillna("missing")

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# Build the model training pipeline
def build_pipeline(y_train: pd.Series) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]
    logger.info(f"scale_pos_weight: {scale_pos_weight:.4f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Find the threshold that maximizes F1 score based on the precision-recall curve
def find_best_f1_threshold(y_test, y_proba) -> tuple[float, float]:
    """Return (threshold, f1) that maximizes F1 across the PR curve.
    If no thresholds are found (e.g., if y_test is all one class), return (0.5, 0.0).
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)
    denom = precision_vals[:-1] + recall_vals[:-1]
    f1_vals = np.divide(
        2 * precision_vals[:-1] * recall_vals[:-1], denom,
        out=np.zeros_like(denom), where=denom > 0,
    )
    if len(f1_vals) == 0:
        return 0.5, 0.0
    best_idx = int(np.argmax(f1_vals))
    return float(thresholds[best_idx]), float(f1_vals[best_idx])

# Evaluate the trained pipeline on the test set and compute various metrics
def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    y_proba = pipeline.predict_proba(X_test)[:, 1]
   
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    metrics = {
        "threshold": DECISION_THRESHOLD,
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "n_test_samples": int(len(y_test)),
        "n_positive_test_samples": int(y_test.sum()),
    }

    best_threshold, best_f1 = find_best_f1_threshold(y_test, y_proba)
    y_pred_opt = (y_proba >= best_threshold).astype(int)
    metrics["optimal_threshold_diagnostic"] = {
        "threshold": best_threshold,
        "precision": float(precision_score(y_test, y_pred_opt)),
        "recall": float(recall_score(y_test, y_pred_opt)),
        "f1_score": best_f1,
        "note": "Diagnostic only, NOT the serving threshold. Reviewed 2026-08-23: "
        "the team is deliberately prioritizing recall over precision for fraud "
        "detection (missing fraud costs more than reviewing false positives), so "
        f"the serving threshold stays at {DECISION_THRESHOLD} despite its lower precision.",
    }

    logger.info(f"Evaluation metrics (threshold={DECISION_THRESHOLD}, chosen to prioritize recall):")
    for k, v in metrics.items():
        if k != "optimal_threshold_diagnostic":
            logger.info(f"  {k}: {v}")
    logger.info(
        f"Optimal-F1 threshold diagnostic (not used): threshold={best_threshold:.4f} "
        f"precision={metrics['optimal_threshold_diagnostic']['precision']:.4f} "
        f"recall={metrics['optimal_threshold_diagnostic']['recall']:.4f} "
        f"f1={best_f1:.4f}"
    )

    return metrics, y_pred, y_proba

# Save evaluation plots (confusion matrix, ROC curve, precision-recall curve, feature importance)
def save_plots(y_test, y_pred, y_proba, pipeline: Pipeline, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(run_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(run_dir / "roc_curve.png", bbox_inches="tight")
    plt.close()

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall_vals, precision_vals)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(run_dir / "precision_recall_curve.png", bbox_inches="tight")
    plt.close()

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:20]
    plt.figure(figsize=(8, 6))
    fi.plot(kind="barh")
    plt.title("Top Feature Importances")
    plt.gca().invert_yaxis()
    plt.savefig(run_dir / "feature_importance.png", bbox_inches="tight")
    plt.close()

    logger.info(f"Saved evaluation plots to {run_dir}")

# Main function to orchestrate the training process
def main():
    parser = argparse.ArgumentParser(description="Train the fraud detection pipeline")
    parser.add_argument("--data-path", type=Path, default=Path("ieee_fraud_detection.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Also copy this run's pipeline to the canonical path the API loads from "
        "(Fraud_Detection/models/Fraud_Detection_Pipeline.pkl). Without this flag, "
        "the run is saved but NOT deployed.",
    )
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run ID: {run_id}")

    df = load_data(args.data_path)
    X_train, X_test, y_train, y_test = split_data(df, args.test_size)
    pipeline = build_pipeline(y_train)

    logger.info("Training pipeline...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")

    metrics, y_pred, y_proba = evaluate(pipeline, X_test, y_test)

    model_path = run_dir / "pipeline.pkl"
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved model to {model_path}")

    metadata = {
        "run_id": run_id,
        "trained_at_utc": run_id,
        "data_path": str(args.data_path),
        "n_rows": int(len(df)),
        "test_size": args.test_size,
        "random_state": RANDOM_STATE,
        "features": PRIORITY_FEATURES,
        "metrics": metrics,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metrics to {run_dir / 'metrics.json'}")

    save_plots(y_test, y_pred, y_proba, pipeline, run_dir)

    if args.promote:
        canonical_path = Path("Fraud_Detection/models/Fraud_Detection_Pipeline.pkl")
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, canonical_path)
        logger.info(f"Promoted run {run_id} to {canonical_path} (this is what the API serves).")
        try:
            subprocess.run(["dvc", "add", str(canonical_path)], check=True, capture_output=True)
            logger.info(
                f"Updated DVC pointer for {canonical_path}. "
                f"Commit {canonical_path}.dvc to git and run 'dvc push' to persist the new artifact."
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(
                f"Could not auto-update DVC tracking ({e}). "
                f"Run 'dvc add {canonical_path}' manually before committing."
            )
    else:
        logger.info(
            "Run saved but NOT promoted. Re-run with --promote once you've reviewed "
            f"{run_dir / 'metrics.json'} and decided this run should replace the served model."
        )


if __name__ == "__main__":
    main()
