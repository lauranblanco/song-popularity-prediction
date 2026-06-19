"""
Modeling for Song Popularity Prediction.

Trains and compares three classifiers (Logistic Regression, Random Forest,
LightGBM) to predict whether a song lands in the top 25% of popularity.

Key design choice to avoid data leakage: the high-cardinality categorical
features (track_genre, artist_main) are target-encoded INSIDE the pipeline
using sklearn's TargetEncoder, which cross-fits the encoding during training.
The popularity column (the basis of the target) is never used as a feature.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_PATH = DATA_DIR / "processed" / "tracks_features.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"

# Numeric predictors: raw audio features + engineered ones. We deliberately
# exclude `popularity` (target leakage) and `duration_ms` (kept as duration_min).
NUMERIC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_min",
    "key", "mode", "time_signature", "explicit_int",
    "dance_energy", "valence_energy", "acoustic_instrumental",
    "n_genres", "artist_track_count",
]
# High-cardinality categoricals -> target encoded inside the pipeline.
CATEGORICAL_FEATURES = ["track_genre", "artist_main"]

TARGET = "is_hit"
RANDOM_STATE = 42


def load_processed(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load the modeling-ready feature dataset."""
    return pd.read_csv(path)


def make_xy(df: pd.DataFrame, numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES):
    """Split the dataframe into the feature matrix X and the target vector y."""
    X = df[list(numeric) + list(categorical)].copy()
    y = df[TARGET].astype(int)
    return X, y


def build_preprocessor(numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES) -> ColumnTransformer:
    """Scale numeric features and target-encode (cross-fitted) the categoricals."""
    transformers = [("num", StandardScaler(), list(numeric))]
    if categorical:
        # TargetEncoder cross-fits the encoding during fit, preventing leakage.
        transformers.append(("cat", TargetEncoder(), list(categorical)))
    return ColumnTransformer(transformers=transformers)


def get_models(pos_weight: float) -> dict:
    """Return the candidate models as a name -> estimator mapping."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        ),
    }


def train_and_evaluate(test_size: float = 0.2,
                       numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES):
    """
    Train all candidate models and report AUC-ROC and F1 on a held-out test set.

    Pass a reduced `categorical` (e.g. ["track_genre"] without "artist_main") to
    run the cold-start scenario: predicting hits without any artist history.

    Returns:
        results: DataFrame with metrics per model.
        fitted: dict of fitted pipelines.
        split: (X_train, X_test, y_train, y_test) for downstream analysis.
    """
    df = load_processed()
    X, y = make_xy(df, numeric, categorical)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    rows, fitted = [], {}

    for name, model in get_models(pos_weight).items():
        pipe = Pipeline([("prep", build_preprocessor(numeric, categorical)), ("clf", model)])
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "model": name,
            "auc_roc": roc_auc_score(y_test, proba),
            "f1": f1_score(y_test, pred),
        })
        fitted[name] = pipe
        print(f"{name:20s}  AUC={rows[-1]['auc_roc']:.3f}  F1={rows[-1]['f1']:.3f}")

    results = pd.DataFrame(rows).sort_values("auc_roc", ascending=False).reset_index(drop=True)
    return results, fitted, (X_train, X_test, y_train, y_test)


def genre_bias_analysis(pipe, X_test, y_test, raw_df, min_samples: int = 50) -> pd.DataFrame:
    """
    Compute test AUC-ROC per genre to expose where the model works best/worst.
    This is the project's differentiator: does it predict pop better than latin?
    """
    proba = pipe.predict_proba(X_test)[:, 1]
    eval_df = X_test.copy()
    eval_df["y_true"] = y_test.values
    eval_df["y_proba"] = proba

    rows = []
    for genre, g in eval_df.groupby("track_genre"):
        if len(g) < min_samples or g["y_true"].nunique() < 2:
            continue
        rows.append({
            "genre": genre,
            "n_test": len(g),
            "hit_rate": g["y_true"].mean(),
            "auc_roc": roc_auc_score(g["y_true"], g["y_proba"]),
        })
    return pd.DataFrame(rows).sort_values("auc_roc", ascending=False).reset_index(drop=True)


def save_best(fitted: dict, results: pd.DataFrame):
    """Persist the best model (by AUC) to models/ with joblib."""
    best_name = results.iloc[0]["model"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "best_model.joblib"
    joblib.dump(fitted[best_name], path)
    print(f"\nBest model: {best_name} -> {path.relative_to(MODELS_DIR.parent)}")
    return best_name


if __name__ == "__main__":
    results, fitted, (Xtr, Xte, ytr, yte) = train_and_evaluate()
    print("\n=== Ranking ===")
    print(results.to_string(index=False))
    save_best(fitted, results)
