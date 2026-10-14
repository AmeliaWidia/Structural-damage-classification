"""Training script for damage_grade classification. Run: python scripts/train.py"""
import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "damage_grade"
ID_COL = "building_id"
TUNE_BEST_MODEL = True
TUNING_ITERATIONS = 15
CV_SPLITS = 3


def resolve_project_root():
    """Find project root by locating the data directory."""
    current = os.path.abspath(os.getcwd())
    for candidate in [current, os.path.dirname(current), os.path.dirname(os.path.dirname(current))]:
        if os.path.isdir(os.path.join(candidate, "data")):
            return candidate
    raise FileNotFoundError("Folder data/ tidak ditemukan. Jalankan dari root project atau folder scripts/.")


def load_data(data_dir):
    """Load training data, Kaggle test data, and submission template."""
    paths = {
        "train": os.path.join(data_dir, "train.csv"),
        "test": os.path.join(data_dir, "test.csv"),
        "sample": os.path.join(data_dir, "sample_submission.csv"),
    }
    for path in paths.values():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File tidak ditemukan: {path}")
    return (
        pd.read_csv(paths["train"]),
        pd.read_csv(paths["test"]),
        pd.read_csv(paths["sample"]),
    )


def prepare_features(train_df, test_df):
    """Separate features and target, and align train/test columns."""
    if TARGET not in train_df:
        raise KeyError(f"Kolom target '{TARGET}' tidak ada di train.csv.")
    X = train_df.drop(columns=[TARGET, ID_COL], errors="ignore").copy()
    y = train_df[TARGET].copy()
    if y.isna().any():
        raise ValueError(f"Target {TARGET} memiliki nilai kosong.")
    X_kaggle_test = test_df.drop(columns=[ID_COL], errors="ignore").copy()

    # Geographic IDs are categorical, not continuous numerical measurements.
    for col in ["district_id", "vdcmun_id", "ward_id"]:
        if col in X.columns:
            X[col] = X[col].astype("string")
            if col in X_kaggle_test.columns:
                X_kaggle_test[col] = X_kaggle_test[col].astype("string")

    X_kaggle_test = X_kaggle_test.reindex(columns=X.columns)
    return X, y, X_kaggle_test


def build_preprocessor(X_train):
    """Create imputing, scaling, and one-hot encoding transformations."""
    categorical_cols = X_train.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ], remainder="drop")
    print(f"Fitur numerik: {len(numeric_cols)} | kategorikal: {len(categorical_cols)}")
    return preprocessor


def build_models(preprocessor):
    """Create baseline model pipelines."""
    return {
        "LDA": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("svd", TruncatedSVD(n_components=30, random_state=RANDOM_STATE)),
            ("classifier", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        "SVM": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", LinearSVC(C=1.0, class_weight="balanced",
                                     random_state=RANDOM_STATE, max_iter=3000)),
        ]),
        "Logistic Regression": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", LogisticRegression(C=1.0, solver="saga",
                class_weight="balanced", max_iter=500, tol=1e-3,
                random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", DecisionTreeClassifier(max_depth=20, min_samples_leaf=5,
                class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=20,
                min_samples_leaf=2, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=1)),
        ]),
        "Extra Trees": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", ExtraTreesClassifier(n_estimators=100, max_depth=None,
                min_samples_leaf=2, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=1)),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", XGBClassifier(objective="multi:softprob",
                eval_metric="mlogloss", n_estimators=150, max_depth=6,
                learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=RANDOM_STATE, n_jobs=1)),
        ]),
        "LightGBM": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", LGBMClassifier(objective="multiclass",
                n_estimators=150, learning_rate=0.1, num_leaves=31,
                class_weight="balanced", random_state=RANDOM_STATE,
                n_jobs=1, verbosity=-1)),
        ]),
    }


def build_param_distributions():
    """Define hyperparameter distributions for tuning the selected baseline."""
    return {
        "LDA": {"svd__n_components": [20, 30, 50]},
        "SVM": {
            "classifier__C": [0.1, 0.5, 1.0],
            "classifier__class_weight": [None, "balanced"],
            "classifier__max_iter": [3000, 5000],
        },
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1.0],
            "classifier__class_weight": [None, "balanced"],
            "classifier__max_iter": [500, 1000],
        },
        "Decision Tree": {
            "classifier__max_depth": [10, 15, 25, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 5],
            "classifier__class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [15, 25, None],
            "classifier__min_samples_leaf": [1, 2, 5],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__class_weight": [None, "balanced"],
        },
        "Extra Trees": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 20, 30],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2, 5],
            "classifier__max_features": ["sqrt", 1.0],
            "classifier__class_weight": [None, "balanced"],
        },
        "XGBoost": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__max_depth": [3, 4, 6],
            "classifier__min_child_weight": [1, 3, 5],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
        },
        "LightGBM": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__learning_rate": [0.01, 0.03, 0.05],
            "classifier__num_leaves": [15, 31, 50],
            "classifier__max_depth": [-1, 5, 10],
            "classifier__min_child_samples": [10, 20, 30],
            "classifier__reg_lambda": [0, 1, 5],
        },
    }


def fit_predict(model_name, model, X_train, y_train, X_valid, encoder):
    """Fit a baseline and return predictions using original target labels."""
    if model_name == "XGBoost":
        model.fit(X_train, encoder.transform(y_train))
        return encoder.inverse_transform(np.asarray(model.predict(X_valid)).astype(int))
    model.fit(X_train, y_train)
    return model.predict(X_valid)


def predict_validation(model_name, model, X_valid, encoder):
    """Predict validation labels, decoding XGBoost's integer class predictions."""
    if model_name == "XGBoost":
        return encoder.inverse_transform(np.asarray(model.predict(X_valid)).astype(int))
    return model.predict(X_valid)


def main():
    """Compare models, tune the best baseline, evaluate, and save a model bundle."""
    started = time.time()
    root_dir = resolve_project_root()
    data_dir = os.path.join(root_dir, "data")
    model_dir = os.path.join(root_dir, "models")
    report_dir = os.path.join(root_dir, "reports")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    train_df, kaggle_test_df, sample_submission = load_data(data_dir)
    X, y, X_kaggle_test = prepare_features(train_df, kaggle_test_df)

    print("Root project:", root_dir)
    print("Train:", train_df.shape, "| Kaggle test:", kaggle_test_df.shape)
    print("Sample submission:", sample_submission.shape)
    print("Fitur:", X.shape[1], "| Kelas:", sorted(y.astype(str).unique()))

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print("X_train:", X_train.shape, "| X_valid:", X_valid.shape)
    print("X_kaggle_test (khusus submission):", X_kaggle_test.shape)

    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor)
    param_distributions = build_param_distributions()
    encoder = LabelEncoder().fit(y)

    baseline_rows = []
    fitted_models = {}
    print("\n" + "=" * 65 + "\nBASELINE MODEL COMPARISON\n" + "=" * 65)
    for name, model in models.items():
        model_start = time.time()
        print(f"\nTraining {name}...")
        try:
            pred = fit_predict(name, model, X_train, y_train, X_valid, encoder)
            score = f1_score(y_valid, pred, average="macro")
            acc = accuracy_score(y_valid, pred)
            fitted_models[name] = model
            baseline_rows.append({
                "model": name, "macro_f1": score, "accuracy": acc,
                "seconds": time.time() - model_start, "status": "ok"
            })
            print(f"Validation Macro F1: {score:.5f} | Accuracy: {acc:.5f}")
        except Exception as exc:
            baseline_rows.append({
                "model": name, "macro_f1": np.nan, "accuracy": np.nan,
                "seconds": time.time() - model_start, "status": repr(exc)
            })
            print(f"Model gagal: {exc!r}")

    results = pd.DataFrame(baseline_rows).sort_values(
        "macro_f1", ascending=False, na_position="last"
    )
    results_path = os.path.join(report_dir, "baseline_model_results.csv")
    results.to_csv(results_path, index=False)
    print("\nHasil baseline:\n", results.to_string(index=False))

    successful = results.dropna(subset=["macro_f1"])
    if successful.empty:
        raise RuntimeError("Semua model baseline gagal. Periksa error di atas.")

    best_name = str(successful.iloc[0]["model"])
    best_baseline = fitted_models[best_name]
    baseline_f1 = float(successful.iloc[0]["macro_f1"])
    best_model = best_baseline
    best_params = {}
    best_cv_f1 = None

    print(f"\nBest baseline: {best_name} | Macro F1: {baseline_f1:.5f}")

    if TUNE_BEST_MODEL:
        print(f"\nTuning {best_name} dengan RandomizedSearchCV...")
        tune_start = time.time()
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        n_iter = min(
            TUNING_ITERATIONS,
            len(list(ParameterGrid(param_distributions[best_name])))
        )
        search = RandomizedSearchCV(
            estimator=clone(models[best_name]),
            param_distributions=param_distributions[best_name],
            n_iter=n_iter,
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            random_state=RANDOM_STATE,
            verbose=1,
            refit=True,
            error_score="raise",
        )
        try:
            if best_name == "XGBoost":
                search.fit(X_train, encoder.transform(y_train))
            else:
                search.fit(X_train, y_train)

            tuned_model = search.best_estimator_
            tuned_pred = predict_validation(best_name, tuned_model, X_valid, encoder)
            tuned_f1 = f1_score(y_valid, tuned_pred, average="macro")
            best_params = search.best_params_
            best_cv_f1 = float(search.best_score_)
            print("Best parameters:", best_params)
            print(f"Best CV Macro F1: {best_cv_f1:.5f}")
            print(f"Tuned validation Macro F1: {tuned_f1:.5f}")
            if tuned_f1 >= baseline_f1:
                best_model = tuned_model
                print("Dipilih: model hasil tuning.")
            else:
                best_model = best_baseline
                print("Tuning tidak meningkatkan validasi; memakai baseline.")
        except Exception as exc:
            print(f"Tuning gagal: {exc!r}. Memakai baseline.")
            best_model = best_baseline
            best_params = {}
            best_cv_f1 = None
        print(f"Waktu tuning: {(time.time() - tune_start) / 60:.2f} menit")

    final_valid_pred = predict_validation(best_name, best_model, X_valid, encoder)
    final_f1 = f1_score(y_valid, final_valid_pred, average="macro")
    final_accuracy = accuracy_score(y_valid, final_valid_pred)

    print("\n" + "=" * 65 + "\nFINAL VALIDATION EVALUATION\n" + "=" * 65)
    print("Model:", best_name)
    print(f"Macro F1: {final_f1:.5f} | Accuracy: {final_accuracy:.5f}")
    print("\nClassification report:\n",
          classification_report(y_valid, final_valid_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_valid, final_valid_pred))

    # Refit selected model on all labeled data for downstream Kaggle prediction.
    print("\nRefitting selected model on all labeled training rows...")
    final_model = clone(best_model)
    if best_name == "XGBoost":
        final_model.fit(X, encoder.transform(y))
    else:
        final_model.fit(X, y)

    bundle = {
        "model": final_model,
        "model_name": best_name,
        "label_encoder": encoder,
        "feature_columns": X.columns.tolist(),
        "target_name": TARGET,
        "id_column": ID_COL,
        "best_params": best_params,
        "best_cv_macro_f1": best_cv_f1,
        "validation_macro_f1": float(final_f1),
        "validation_accuracy": float(final_accuracy),
        "target_classes": encoder.classes_.tolist(),
    }
    model_path = os.path.join(model_dir, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    summary = pd.DataFrame([{
        "model_name": best_name,
        "validation_macro_f1": final_f1,
        "validation_accuracy": final_accuracy,
        "best_cv_macro_f1": best_cv_f1,
        "best_params": repr(best_params),
        "training_rows": len(X),
        "feature_count": X.shape[1],
    }])
    summary.to_csv(os.path.join(report_dir, "best_model_summary.csv"), index=False)

    print("\nModel tersimpan:", model_path)
    print("Hasil baseline:", results_path)
    print("Summary:", os.path.join(report_dir, "best_model_summary.csv"))
    print(f"Total waktu: {(time.time() - started) / 60:.2f} menit")
    print("Catatan: test.csv tidak digunakan untuk evaluasi; hanya untuk submission.")


if __name__ == "__main__":
    main()
