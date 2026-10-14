"""Feature engineering helpers yang digunakan dlm eda workflow."""

def get_feature_columns(X, target="damage_grade", id_column="building_id"):
    """Return nama fitur tanpa target dan identifier."""
    return [c for c in X.columns if c not in {target, id_column}]

def cast_geographic_columns(df, candidates=("district_id", "vdcmun_id", "ward_id")):
    """Ubah identifier geografis jadi string tanpa ubah input."""
    result = df.copy()
    for col in candidates:
        if col in result.columns:
            result[col] = result[col].astype("string")
    return result
