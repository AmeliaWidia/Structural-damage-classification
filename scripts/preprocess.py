"""Preprocessing helpers"""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_csv(path):
    """Load a CSV -> DataFrame."""
    return pd.read_csv(path)


def cast_geographic_columns(X, candidates=("district_id", "vdcmun_id", "ward_id")):
    """Ubah kolom ID geografis yang ada ke tipe string di pandas."""
    result = X.copy()
    for col in candidates:
        if col in result.columns:
            result[col] = result[col].astype("string")
    return result


def split_features_target(train, target="damage_grade", id_column="building_id"):
    """Split target and featuressambil mengecualikan identifier dari predictor."""
    if target not in train.columns:
        raise ValueError(f"Target column {target!r} not found.")
    X = train.drop(columns=[target, id_column], errors="ignore").copy()
    y = train[target].copy()
    X = cast_geographic_columns(X)
    return X, y


def make_preprocessor(X, min_frequency=5):
    """Buat preprocessing."""
    categorical_cols = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ], remainder="drop")
    return preprocessor, numeric_cols, categorical_cols


def normalize_grade_labels(values):
    """Convert labels seperti 'Grade 5' jadi integer 5 untuk validasi submission"""
    extracted = pd.Series(values).astype(str).str.extract(r"(\d+)", expand=False)
    if extracted.isna().any():
        raise ValueError("One or more labels do not contain a numeric grade.")
    return extracted.astype(int)
