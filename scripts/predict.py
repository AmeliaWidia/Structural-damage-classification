from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from scripts.preprocess import normalize_grade_labels


def save_model_bundle(path, bundle):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(bundle, file)
    return path


def load_model_bundle(path):
    with Path(path).open("rb") as file:
        return pickle.load(file)


def predict_labels(model, model_name, X_eval, label_encoder=None):
    predictions = np.asarray(model.predict(X_eval))
    if model_name == "XGBoost":
        if label_encoder is None:
            raise ValueError("LabelEncoder not found in model bundle.")
        predictions = label_encoder.inverse_transform(predictions.astype(int))
    return predictions


def create_submission(sample_submission, test, predictions, target="damage_grade", id_column="building_id"):
    if target not in sample_submission.columns:
        raise ValueError(f"Target column {target!r} not found in sample submission.")
    if len(sample_submission) != len(predictions) or len(test) != len(predictions):
        raise ValueError("Test, template, and predictions have different row counts.")
    submission = sample_submission.copy()
    if id_column in submission.columns and id_column in test.columns:
        left = submission[id_column].astype(str).reset_index(drop=True)
        right = test[id_column].astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError("ID order differs between sample_submission.csv and test.csv.")
    submission[target] = predictions
    return submission


def validate_submission(submission, sample_submission, train, target="damage_grade"):
    if list(submission.columns) != list(sample_submission.columns):
        raise ValueError("Submission columns differ from the sample template.")
    if len(submission) != len(sample_submission):
        raise ValueError("Submission row count differs from the sample template.")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values.")
    valid_classes = set(normalize_grade_labels(train[target]).unique())
    pred_classes = set(normalize_grade_labels(submission[target]).unique())
    if not pred_classes.issubset(valid_classes):
        raise ValueError(f"Invalid predicted classes: {sorted(pred_classes - valid_classes)}")
    return True


def save_submission(submission, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return path