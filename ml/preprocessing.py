import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "loan_amount"

ETHICAL_FEATURE_COLUMNS = [
    "age",
    "estimated_monthly_income",
    "credit_history_count",
    "personal_risk_score",
    "credit_score",
    "monthly_rent",
    "caf_quotient",
]


def split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor():
    num_pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer([
        ("num", num_pipeline, ETHICAL_FEATURE_COLUMNS),
    ])


def prepare_features_and_target(df):
    X = df[ETHICAL_FEATURE_COLUMNS].copy()
    y = np.log1p(df[TARGET_COLUMN])
    preprocessor = build_preprocessor()
    return X, y, preprocessor


def inverse_target_transform(values):
    return np.expm1(values)
