import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sqlalchemy import select
from sklearn.model_selection import train_test_split

from config.db import SessionLocal
from ml.evaluate import extract_loss_metrics
from ml.model import create_nn_model, train_model
from ml.preprocessing import prepare_features_and_target
from models.client import Client
from models.loan_information import LoanInformation


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "loan_amount_model.keras"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
LOSS_CURVE_PATH = ARTIFACTS_DIR / "loss_curve.png"


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
                "date_of_birth": client.date_of_birth,
                "height_cm": float(client.height_cm) if client.height_cm is not None else None,
                "weight_kg": float(client.weight_kg) if client.weight_kg is not None else None,
                "sex": client.sex,
                "has_sport_license": client.has_sport_license,
                "education_level": client.education_level,
                "region": client.region,
                "is_smoker": client.is_smoker,
                "is_french_national": client.is_french_national,
                "family_status": client.family_status,
                "created_at": client.created_at,
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
                "account_created_at": loan.account_created_at,
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
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    df = load_training_dataframe()
    X, y, preprocessor = prepare_features_and_target(df)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    model = create_nn_model(X_train_processed.shape[1])
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
        verbose=1,
        callbacks=[early_stopping],
    )

    evaluation_result = model.evaluate(X_test_processed, y_test, verbose=0)
    metrics = extract_loss_metrics(history, evaluation_result)

    model.save(MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    save_loss_curve(history)

    print(json.dumps(metrics, indent=2))
    print(f"Model saved to {MODEL_PATH}")
    print(f"Loss curve saved to {LOSS_CURVE_PATH}")


if __name__ == "__main__":
    main()
