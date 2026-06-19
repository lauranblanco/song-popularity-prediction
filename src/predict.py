"""
Inference helpers for the Streamlit app.

Builds a single-row feature frame from manual inputs (matching the training
schema), runs the saved pipeline, and produces a SHAP explanation for the
prediction. Kept separate from the UI so it can be tested in isolation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from model import NUMERIC_FEATURES, CATEGORICAL_FEATURES

MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.joblib"
UNKNOWN_ARTIST = "__UNKNOWN_ARTIST__"  # unseen value -> TargetEncoder default

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_input_row(inputs: dict) -> pd.DataFrame:
    """
    Assemble one row with every column the pipeline expects.

    `inputs` carries the natural audio features, genre, artist info and a few
    structural fields; the engineered interactions are derived here so they stay
    consistent with `features.add_features`.
    """
    row = {
        "danceability": inputs["danceability"],
        "energy": inputs["energy"],
        "loudness": inputs["loudness"],
        "speechiness": inputs["speechiness"],
        "acousticness": inputs["acousticness"],
        "instrumentalness": inputs["instrumentalness"],
        "liveness": inputs["liveness"],
        "valence": inputs["valence"],
        "tempo": inputs["tempo"],
        "duration_min": inputs["duration_min"],
        "key": inputs["key"],
        "mode": inputs["mode"],
        "time_signature": inputs["time_signature"],
        "explicit_int": int(inputs["explicit"]),
        # Engineered interactions (mirror features.add_features)
        "dance_energy": inputs["danceability"] * inputs["energy"],
        "valence_energy": inputs["valence"] * inputs["energy"],
        "acoustic_instrumental": inputs["acousticness"] * inputs["instrumentalness"],
        "n_genres": inputs["n_genres"],
        "artist_track_count": inputs["artist_track_count"],
        "track_genre": inputs["track_genre"],
        "artist_main": inputs["artist_main"],
    }
    return pd.DataFrame([row])[ALL_FEATURES]


def predict_proba(model, X: pd.DataFrame) -> float:
    """Return the hit probability for a single-row input frame."""
    return float(model.predict_proba(X)[0, 1])


def shap_explanation(model, X: pd.DataFrame):
    """
    Compute a SHAP Explanation for the single input row, ready for a waterfall.
    Imported lazily so the rest of the app loads even if SHAP is slow to import.
    """
    import shap

    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    X_t = prep.transform(X)

    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(X_t)
    if isinstance(values, list):       # [class0, class1] on some versions
        values = values[1]
    base = float(np.ravel(explainer.expected_value)[-1])

    return shap.Explanation(
        values=values[0],
        base_values=base,
        data=X_t[0],
        feature_names=ALL_FEATURES,
    )
