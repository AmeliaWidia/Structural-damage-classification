import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from scripts.preprocess import make_preprocessor


def build_models(preprocessor, random_state=42):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    return {
        "LDA": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("svd", TruncatedSVD(n_components=30, random_state=random_state)),
            ("classifier", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        ]),
        "SVM": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", LinearSVC(C=1.0, class_weight="balanced", random_state=random_state, max_iter=3000))
        ]),
        "Logistic Regression": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", LogisticRegression(C=1.0, solver="saga", class_weight="balanced", max_iter=300, tol=1e-3, random_state=random_state))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, class_weight="balanced", random_state=random_state))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2, class_weight="balanced", random_state=random_state, n_jobs=-1))
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", n_estimators=150, max_depth=6, learning_rate=0.1,
                                         subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=random_state, n_jobs=1))
        ]),
        "LightGBM": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", LGBMClassifier(objective="multiclass", n_estimators=150, learning_rate=0.1, num_leaves=31, 
                                          class_weight="balanced", random_state=random_state, n_jobs=1, verbosity=-1))
        ])
    }


def predict_labels(model, model_name, X_eval, label_encoder=None):
    predictions = np.asarray(model.predict(X_eval))
    if model_name == "XGBoost":
        if label_encoder is None:
            raise ValueError("LabelEncoder is required for XGBoost predictions.")
        predictions = label_encoder.inverse_transform(predictions.astype(int))
    return predictions


def train_baselines(models, X_train, y_train, X_valid, y_valid, label_encoder=None):
    rows, trained_models, predictions = [], {}, {}
    y_train_xgb = label_encoder.fit_transform(y_train) if label_encoder is not None else None
    for name, model in models.items():
        start = time.time()
        try:
            model.fit(X_train, y_train_xgb if name == "XGBoost" else y_train)
            pred = predict_labels(model, name, X_valid, label_encoder)
            trained_models[name] = model
            predictions[name] = pred
            rows.append({"Model": name, "Macro F1": f1_score(y_valid, pred, average="macro", zero_division=0),
                         "Accuracy": accuracy_score(y_valid, pred), "Training Time (s)": time.time()-start,
                         "Status": "Success"})
        except Exception as exc:
            rows.append({"Model": name, "Macro F1": np.nan, "Accuracy": np.nan,
                         "Training Time (s)": time.time()-start, "Status": f"Failed: {exc}"})
    results = pd.DataFrame(rows).sort_values("Macro F1", ascending=False, na_position="last").reset_index(drop=True)
    return results, trained_models, predictions


def tune_best_baseline(model_name, baseline_model, X_train, y_train, label_encoder=None):
    grids = {
        "LDA": {"svd__n_components": [20, 30]},
        "SVM": {"classifier__C": [0.5, 1.0]},
        "Logistic Regression": {"classifier__C": [0.5, 1.0]},
        "Decision Tree": {"classifier__max_depth": [15, 25], "classifier__min_samples_leaf": [2, 5]},
        "Random Forest": {"classifier__n_estimators": [100], "classifier__max_depth": [15, None],
                          "classifier__min_samples_leaf": [2, 5]},
        "XGBoost": {"classifier__n_estimators": [100, 150], "classifier__max_depth": [4, 6]},
        "LightGBM": {"classifier__n_estimators": [100, 150], "classifier__num_leaves": [31, 50]}
    }
    search = GridSearchCV(clone(baseline_model), grids[model_name], scoring="f1_macro",
                          cv=2, n_jobs=1, verbose=2, refit=True, error_score="raise")
    fit_y = label_encoder.fit_transform(y_train) if model_name == "XGBoost" and label_encoder is not None else y_train
    search.fit(X_train, fit_y)
    return search