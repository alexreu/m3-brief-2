import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy import select
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from config.db import SessionLocal
from ml.model import create_nn_model, model_predict, train_model
from ml.preprocessing import inverse_target_transform, prepare_features_and_target
from models.client import Client
from models.loan_information import LoanInformation


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "loan_amount_model.keras"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
LOSS_CURVE_PATH = ARTIFACTS_DIR / "loss_curve.png"
PREDICTIONS_PLOT_PATH = ARTIFACTS_DIR / "predictions_vs_actuals.png"
OUTLIERS_PATH = ARTIFACTS_DIR / "outlier_summary.json"


def load_training_dataframe() -> pd.DataFrame:
    with SessionLocal() as session:
        stmt = (
            select(Client, LoanInformation)
            .join(LoanInformation, LoanInformation.client_id == Client.id)
        )
        rows = session.execute(stmt).all()

    dataset = []
    for client, loan in rows:
        dataset.append(
            {
                "id": client.id,
                "client_id": loan.client_id,
                "first_name": client.first_name,
                "last_name": client.last_name,
                "age": client.age,
                "caf_quotient": float(client.caf_quotient)
                if client.caf_quotient is not None
                else None,
                "estimated_monthly_income": float(loan.estimated_monthly_income)
                if loan.estimated_monthly_income is not None
                else None,
                "credit_history_count": loan.credit_history_count,
                "personal_risk_score": float(loan.personal_risk_score)
                if loan.personal_risk_score is not None
                else None,
                "credit_score": loan.credit_score,
                "monthly_rent": float(loan.monthly_rent) if loan.monthly_rent is not None else None,
                "loan_amount": float(loan.loan_amount) if loan.loan_amount is not None else None,
            }
        )

    df = pd.DataFrame(dataset)
    if df.empty:
        raise ValueError("No training data found in the database.")

    return df


def save_loss_curve(history) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log-space loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()


def save_prediction_diagnostics(y_true, y_pred) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_value = min(float(y_true.min()), float(y_pred.min()))
    max_value = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.title("Predictions vs Actuals")
    plt.xlabel("Actual loan amount")
    plt.ylabel("Predicted loan amount")
    plt.tight_layout()
    plt.savefig(PREDICTIONS_PLOT_PATH)
    plt.close()


def compute_outlier_summary(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    numeric_columns = [
        "age",
        "estimated_monthly_income",
        "credit_history_count",
        "personal_risk_score",
        "credit_score",
        "monthly_rent",
        "caf_quotient",
        "loan_amount",
    ]

    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())
        summary[column] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_count,
        }

    return summary


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    df = load_training_dataframe()
    outlier_summary = compute_outlier_summary(df)
    X, y, preprocessor = prepare_features_and_target(df)
    y_original = df["loan_amount"].to_numpy()

    X_train, X_temp, y_train, y_temp, y_train_original, y_temp_original = train_test_split(
        X, y, y_original, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test, y_val_original, y_test_original = train_test_split(
        X_temp, y_temp, y_temp_original, test_size=0.50, random_state=42
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    input_dim = X_train.shape[1]
    model = create_nn_model(input_dim)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model, history = train_model(
        model,
        X_train_processed,
        y_train,
        X_val=X_val_processed,
        y_val=y_val,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
    )

    log_predictions = model_predict(model, X_test_processed)
    y_pred = inverse_target_transform(log_predictions)
    y_test_actual = y_test_original
    metrics = {
        "train_loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "test_loss": float(mean_squared_error(y_test_actual, y_pred)),
        "test_mae": float(mean_absolute_error(y_test_actual, y_pred)),
        "prediction_count": int(len(y_pred)),
        "target_transform": "log1p",
        "imputer": "KNNImputer(n_neighbors=5, weights='distance')",
    }

    model.save(MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    with OUTLIERS_PATH.open("w", encoding="utf-8") as outlier_file:
        json.dump(outlier_summary, outlier_file, indent=2)

    save_loss_curve(history)
    save_prediction_diagnostics(y_test_actual, y_pred)

    print(json.dumps(metrics, indent=2))
    print(f"Model saved to {MODEL_PATH}")
    print(f"Loss curve saved to {LOSS_CURVE_PATH}")
    print(f"Predictions plot saved to {PREDICTIONS_PLOT_PATH}")
    print(f"Outlier summary saved to {OUTLIERS_PATH}")


if __name__ == "__main__":
    main()
