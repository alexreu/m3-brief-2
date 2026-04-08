from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "loan_amount"

NUMERICAL_COLS = [
    "age",
    "height_cm",
    "weight_kg",
    "estimated_monthly_income",
    "credit_history_count",
    "personal_risk_score",
    "credit_score",
    "monthly_rent",
]

CATEGORICAL_COLS = [
    "sex",
    "education_level",
    "region",
    "family_status",
]

BOOLEAN_COLS = [
    "has_sport_license",
    "is_smoker",
    "is_french_national",
]

DROP_COLS = [
    "id",
    "client_id",
    "first_name",
    "last_name",
    "date_of_birth",
    "created_at",
    "account_created_at",
]


def split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor():
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipeline, NUMERICAL_COLS),
        ("cat", cat_pipeline, CATEGORICAL_COLS),
        ("bool", "passthrough", BOOLEAN_COLS),
    ])


def prepare_features_and_target(df):
    X = df.drop(columns=DROP_COLS + [TARGET_COLUMN], errors="ignore")
    y = df[TARGET_COLUMN]
    preprocessor = build_preprocessor()
    return X, y, preprocessor
