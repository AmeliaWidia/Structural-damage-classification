def get_feature_columns(X, target="damage_grade", id_column="building_id"):
    return [c for c in X.columns if c not in {target, id_column}]


def cast_geographic_columns(df, candidates=("district_id", "vdcmun_id", "ward_id")):
    result = df.copy()
    for col in candidates:
        if col in result.columns:
            result[col] = result[col].astype("string")
    return result
