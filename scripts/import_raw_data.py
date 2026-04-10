from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from config import SessionLocal
from models.client import Client
from models.loan_information import LoanInformation


PROJECT_ROOT = Path(__file__).resolve().parent.parent


RAW_DATA_PATHS = (
    PROJECT_ROOT / "data" / "raw_data.csv",
    PROJECT_ROOT / "data" / "raw_data_2.csv",
)
MAX_MISSING_RATIO = 0.20
TARGET_COLUMN = "montant_pret"

SOURCE_COLUMNS_DEFAULTS = {
    "nom": pd.NA,
    "prenom": pd.NA,
    "age": pd.NA,
    "taille": pd.NA,
    "poids": pd.NA,
    "sexe": pd.NA,
    "sport_licence": pd.NA,
    "niveau_etude": pd.NA,
    "region": pd.NA,
    "smoker": pd.NA,
    "nationalité_francaise": pd.NA,
    "revenu_estime_mois": pd.NA,
    "situation_familiale": pd.NA,
    "historique_credits": pd.NA,
    "risque_personnel": pd.NA,
    "date_creation_compte": pd.NA,
    "score_credit": pd.NA,
    "loyer_mensuel": pd.NA,
    "montant_pret": pd.NA,
    "orientation_sexuelle": pd.NA,
    "quotient_caf": pd.NA,
    "nb_enfants": pd.NA,
}

CSV_TO_DB_COLUMNS = {
    "nom": "last_name",
    "prenom": "first_name",
    "age": "age",
    "taille": "height_cm",
    "poids": "weight_kg",
    "sexe": "sex",
    "sport_licence": "has_sport_license",
    "niveau_etude": "education_level",
    "region": "region",
    "smoker": "is_smoker",
    "nationalité_francaise": "is_french_national",
    "revenu_estime_mois": "estimated_monthly_income",
    "situation_familiale": "family_status",
    "historique_credits": "credit_history_count",
    "risque_personnel": "personal_risk_score",
    "date_creation_compte": "account_created_at",
    "score_credit": "credit_score",
    "loyer_mensuel": "monthly_rent",
    "montant_pret": "loan_amount",
    "orientation_sexuelle": "sexual_orientation",
    "quotient_caf": "caf_quotient",
    "nb_enfants": "child_count",
}

NUMERIC_COLUMNS = [
    "age",
    "taille",
    "poids",
    "revenu_estime_mois",
    "historique_credits",
    "risque_personnel",
    "score_credit",
    "loyer_mensuel",
    "quotient_caf",
    "nb_enfants",
]

BOOLEAN_COLUMNS = [
    "sport_licence",
    "smoker",
    "nationalité_francaise",
]

CATEGORICAL_COLUMNS = [
    "nom",
    "prenom",
    "sexe",
    "niveau_etude",
    "region",
    "situation_familiale",
    "orientation_sexuelle",
]

ESSENTIAL_COLUMNS = [
    "nom",
    "prenom",
    "age",
    "sexe",
    "revenu_estime_mois",
    "risque_personnel",
    "montant_pret",
]

REQUIRED_SOURCE_COLUMNS = ESSENTIAL_COLUMNS + [
    TARGET_COLUMN,
]


def is_missing(value: Any) -> bool:
    return value is None or pd.isna(value)


def parse_bool(value: Any):
    if is_missing(value):
        return None
    lowered = str(value).strip().lower()
    if lowered == "oui":
        return True
    if lowered == "non":
        return False
    return None


def normalize_string(value: Any):
    if is_missing(value):
        return None
    cleaned = str(value).strip()
    return cleaned or None


def to_optional_int(value: Any) -> int | None:
    if is_missing(value):
        return None
    return int(value)


def to_nullable(value: Any):
    if is_missing(value):
        return None
    return value


def describe_target(series: pd.Series) -> dict[str, float | int | None]:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "max": None,
        }

    return {
        "count": int(numeric_series.shape[0]),
        "min": float(numeric_series.min()),
        "median": float(numeric_series.median()),
        "max": float(numeric_series.max()),
    }


def normalize_source_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    normalized = df.copy()
    added_columns: list[str] = []

    for column, default_value in SOURCE_COLUMNS_DEFAULTS.items():
        if column not in normalized.columns:
            normalized[column] = default_value
            added_columns.append(column)

    missing_required_columns = [
        column for column in REQUIRED_SOURCE_COLUMNS if column not in normalized.columns
    ]
    if missing_required_columns:
        raise KeyError(
            "Missing required source columns: " +
            ", ".join(missing_required_columns)
        )

    return normalized, added_columns


def load_raw_dataframes(paths: tuple[Path, ...] = RAW_DATA_PATHS) -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    added_columns_by_file: dict[str, list[str]] = {}

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        raw_df = pd.read_csv(path)
        normalized_df, added_columns = normalize_source_dataframe(raw_df)
        normalized_df["source_file"] = path.name
        frames.append(normalized_df)
        added_columns_by_file[path.name] = added_columns

    combined_df = pd.concat(frames, ignore_index=True)
    metadata = {
        "source_files": [path.name for path in paths],
        "rows_by_file": {path.name: int(len(frame)) for path, frame in zip(paths, frames)},
        "added_optional_columns": added_columns_by_file,
        "total_rows": int(len(combined_df)),
    }
    return combined_df, metadata


def clean_raw_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    cleaned = df.copy()
    cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)

    initial_row_count = int(len(cleaned))
    initial_target_stats = describe_target(cleaned[TARGET_COLUMN])

    missing_ratio = cleaned.isna().mean(axis=1)
    cleaned = cleaned.loc[missing_ratio <= MAX_MISSING_RATIO].copy()
    after_missing_ratio_count = int(len(cleaned))
    cleaned = cleaned.dropna(subset=ESSENTIAL_COLUMNS).copy()
    after_essential_count = int(len(cleaned))

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned[TARGET_COLUMN] = pd.to_numeric(
        cleaned[TARGET_COLUMN], errors="coerce")
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).copy()
    after_target_count = int(len(cleaned))

    for column in BOOLEAN_COLUMNS:
        cleaned[column] = cleaned[column].map(parse_bool)

    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].map(normalize_string)

    cleaned["date_creation_compte"] = pd.to_datetime(
        cleaned["date_creation_compte"], errors="coerce"
    ).dt.date

    imputed_numeric: dict[str, int] = {}
    for column in NUMERIC_COLUMNS:
        missing_count = int(cleaned[column].isna().sum())
        if missing_count:
            imputed_numeric[column] = missing_count
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    imputed_non_numeric: dict[str, int] = {}
    for column in BOOLEAN_COLUMNS + CATEGORICAL_COLUMNS:
        missing_count = int(cleaned[column].isna().sum())
        if missing_count:
            imputed_non_numeric[column] = missing_count
            modes = cleaned[column].mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else None
            cleaned[column] = cleaned[column].fillna(fill_value)

    cleaned["score_credit"] = cleaned["score_credit"].round().astype("Int64")
    cleaned["historique_credits"] = cleaned["historique_credits"].round().astype(
        "Int64")
    cleaned["age"] = cleaned["age"].round().astype("Int64")
    cleaned["loyer_mensuel"] = cleaned["loyer_mensuel"].clip(lower=0)
    cleaned["quotient_caf"] = cleaned["quotient_caf"].clip(lower=0)
    cleaned["nb_enfants"] = cleaned["nb_enfants"].round().clip(
        lower=0).astype("Int64")

    stats = {
        "initial_rows": initial_row_count,
        "rows_after_missing_ratio_filter": after_missing_ratio_count,
        "rows_after_essential_filter": after_essential_count,
        "rows_after_target_filter": after_target_count,
        "rows_removed": initial_row_count - after_target_count,
        "imputed_numeric": imputed_numeric,
        "imputed_non_numeric": imputed_non_numeric,
        "initial_target_stats": initial_target_stats,
        "cleaned_target_stats": describe_target(cleaned[TARGET_COLUMN]),
    }

    return cleaned.rename(columns=CSV_TO_DB_COLUMNS), stats


def row_to_models(row: pd.Series):
    today = date.today()

    client_data = {
        "first_name": row["first_name"],
        "last_name": row["last_name"],
        "age": int(row["age"]),
        "date_of_birth": None,
        "height_cm": to_nullable(row["height_cm"]),
        "weight_kg": to_nullable(row["weight_kg"]),
        "sex": to_nullable(row["sex"]),
        "has_sport_license": to_nullable(row["has_sport_license"]),
        "education_level": to_nullable(row["education_level"]),
        "region": to_nullable(row["region"]),
        "is_smoker": to_nullable(row["is_smoker"]),
        "is_french_national": to_nullable(row["is_french_national"]),
        "family_status": to_nullable(row["family_status"]),
        "sexual_orientation": to_nullable(row["sexual_orientation"]),
        "caf_quotient": to_nullable(row["caf_quotient"]),
        "child_count": to_optional_int(row["child_count"]),
        "created_at": to_nullable(row["account_created_at"]) or today,
    }

    loan_data = {
        "estimated_monthly_income": to_nullable(row["estimated_monthly_income"]),
        "credit_history_count": to_optional_int(row["credit_history_count"]),
        "personal_risk_score": to_nullable(row["personal_risk_score"]),
        "credit_score": to_optional_int(row["credit_score"]),
        "monthly_rent": to_nullable(row["monthly_rent"]),
        "loan_amount": to_nullable(row["loan_amount"]),
        "account_created_at": to_nullable(row["account_created_at"]) or today,
        "created_at": to_nullable(row["account_created_at"]) or today,
    }

    return client_data, loan_data


def import_raw_data(reset: bool = True, paths: tuple[Path, ...] = RAW_DATA_PATHS) -> None:
    raw_df, source_metadata = load_raw_dataframes(paths)
    cleaned_df, cleaning_stats = clean_raw_dataframe(raw_df)
    cleaning_stats["source_metadata"] = source_metadata

    with SessionLocal() as session:
        if reset:
            session.query(LoanInformation).delete()
            session.query(Client).delete()
            session.commit()

        for _, row in cleaned_df.iterrows():
            client_data, loan_data = row_to_models(row)

            client = Client(**client_data)
            session.add(client)
            session.flush()

            loan_data["client_id"] = client.id
            session.add(LoanInformation(**loan_data))

        session.commit()

    print(f"Imported {len(cleaned_df)} cleaned rows into the database.")
    print(f"Cleaning stats: {cleaning_stats}")


if __name__ == "__main__":
    import_raw_data(reset=True)
