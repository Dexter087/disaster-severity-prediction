import os
import re
import numpy as np
import pandas as pd


RAW_DATA_PATH = os.path.join("data", "raw", "emdat.xlsx")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "cleaned_disasters.csv")


def clean_column_name(col):
    """
    Converts column names into lowercase snake_case format.

    Example:
    'Total Deaths' -> 'total_deaths'
    'Disaster Type' -> 'disaster_type'
    """
    col = col.strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = col.strip("_")
    return col


def standardize_columns(df):
    """
    Standardizes all column names.
    """
    df = df.copy()
    df.columns = [clean_column_name(col) for col in df.columns]
    return df


def find_existing_columns(df, possible_columns):
    """
    Finds which columns from a list are actually present in the dataset.
    This makes the code flexible for different EM-DAT export versions.
    """
    return [col for col in possible_columns if col in df.columns]


def create_impact_score(df):
    """
    Creates an impact score using deaths, affected population, and damage.

    impact_score =
        log1p(total_deaths) +
        log1p(total_affected) +
        log1p(total_damage)

    For this EM-DAT export, damage is usually stored as:
        total_damage_000_us
    or:
        total_damage_adjusted_000_us
    """
    df = df.copy()

    if "total_deaths" not in df.columns:
        print("Warning: 'total_deaths' not found. Creating it with value 0.")
        df["total_deaths"] = 0

    if "total_affected" not in df.columns:
        print("Warning: 'total_affected' not found. Creating it with value 0.")
        df["total_affected"] = 0

    damage_candidates = [
        "total_damage_000_us",
        "total_damage_adjusted_000_us",
        "total_damage",
    ]

    damage_col = None
    for col in damage_candidates:
        if col in df.columns:
            damage_col = col
            break

    if damage_col is None:
        print("Warning: No total damage column found. Creating total_damage_for_target with value 0.")
        df["total_damage_for_target"] = 0
    else:
        print(f"Using '{damage_col}' for damage in impact_score.")
        df["total_damage_for_target"] = df[damage_col]

    impact_cols = [
        "total_deaths",
        "total_affected",
        "total_damage_for_target",
    ]

    for col in impact_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["impact_score"] = (
        np.log1p(df["total_deaths"])
        + np.log1p(df["total_affected"])
        + np.log1p(df["total_damage_for_target"])
    )

    return df


def create_severity_class(df):
    """
    Converts impact_score into Low, Medium, and High classes using quantiles.
    """
    df = df.copy()

    low_threshold = df["impact_score"].quantile(0.33)
    high_threshold = df["impact_score"].quantile(0.66)

    def classify(score):
        if score <= low_threshold:
            return "Low Impact"
        elif score <= high_threshold:
            return "Medium Impact"
        else:
            return "High Impact"

    df["severity_class"] = df["impact_score"].apply(classify)

    print("\nSeverity class distribution:")
    print(df["severity_class"].value_counts())

    return df


def create_features(df):
    """
    Creates additional useful machine learning features.
    """
    df = df.copy()

    if "start_year" in df.columns:
        df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")
        df["decade"] = (df["start_year"] // 10 * 10).astype("Int64").astype(str) + "s"

    if "end_year" in df.columns and "start_year" in df.columns:
        df["end_year"] = pd.to_numeric(df["end_year"], errors="coerce")
        df["disaster_duration_years"] = df["end_year"] - df["start_year"]
        df["disaster_duration_years"] = df["disaster_duration_years"].fillna(0)
        df["disaster_duration_years"] = df["disaster_duration_years"].clip(lower=0)

    if "country" in df.columns:
        country_freq = df["country"].value_counts()
        df["country_disaster_frequency"] = df["country"].map(country_freq)

    if "disaster_type" in df.columns:
        type_freq = df["disaster_type"].value_counts()
        df["disaster_type_frequency"] = df["disaster_type"].map(type_freq)

    if "magnitude" in df.columns:
        df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
        df["magnitude_missing"] = df["magnitude"].isna().astype(int)

    return df


def remove_leakage_columns(df):
    """
    Removes columns that directly reveal the final impact.
    These should not be used as model input features.
    """
    df = df.copy()

    leakage_columns = [
    "total_deaths",
    "no_injured",
    "no_affected",
    "no_homeless",
    "total_affected",
    "reconstruction_costs_000_us",
    "reconstruction_costs_adjusted_000_us",
    "insured_damage_000_us",
    "insured_damage_adjusted_000_us",
    "total_damage_000_us",
    "total_damage_adjusted_000_us",
    "total_damage",
    "total_damage_for_target",
    "impact_score"

    ]

    existing_leakage_columns = find_existing_columns(df, leakage_columns)

    df = df.drop(columns=existing_leakage_columns)

    print("\nRemoved leakage columns:")
    print(existing_leakage_columns)

    return df


def select_useful_columns(df):
    """
    Keeps only useful columns for ML if they exist.
    """
    possible_feature_columns = [
        "country",
        "region",
        "continent",
        "disaster_group",
        "disaster_subgroup",
        "disaster_type",
        "disaster_subtype",
        "start_year",
        "start_month",
        "end_year",
        "magnitude",
        "magnitude_scale",
        "decade",
        "disaster_duration_years",
        "country_disaster_frequency",
        "disaster_type_frequency",
        "magnitude_missing",
        "severity_class"
    ]

    existing_columns = find_existing_columns(df, possible_feature_columns)

    print("\nSelected columns:")
    print(existing_columns)

    return df[existing_columns]


def handle_missing_values(df):
    """
    Handles missing values in the cleaned dataset.
    """
    df = df.copy()

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    for col in numerical_cols:
        if col != "severity_class":
            df[col] = df[col].fillna(df[col].median())

    return df


def preprocess():
    print("Loading raw dataset...")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw dataset not found at: {RAW_DATA_PATH}\n"
            "Please place your EM-DAT file in data/raw/ and rename it to emdat.xlsx"
        )

    df = pd.read_excel(RAW_DATA_PATH)

    print(f"Raw dataset shape: {df.shape}")

    df = standardize_columns(df)

    print("\nAvailable columns after cleaning:")
    print(df.columns.tolist())

    df = create_impact_score(df)
    df = create_severity_class(df)
    df = create_features(df)
    df = remove_leakage_columns(df)
    df = select_useful_columns(df)
    df = handle_missing_values(df)

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\nPreprocessing complete.")
    print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"Processed dataset shape: {df.shape}")

    print("\nFinal columns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    preprocess()