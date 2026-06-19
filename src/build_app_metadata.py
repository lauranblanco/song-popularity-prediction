"""
Build a small metadata file the Streamlit app needs at runtime.

Decouples the app from the raw/processed datasets (which are git-ignored): the
app ships only this JSON plus the serialized model, so it can be deployed to
Streamlit Cloud without the full data.
"""

import json
from pathlib import Path

import pandas as pd

import features as F

APP_DATA_PATH = Path(__file__).parent.parent / "app_data" / "metadata.json"

# Audio features the user controls directly (the rest are derived in the app).
NATURAL_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_min",
]

# Brazilian / regional genres flagged as low-reliability in the UI (see Notebook 03).
BRAZILIAN_GENRES = ["brazil", "samba", "pagode", "mpb", "sertanejo", "forro"]


def build(top_artists: int = 800) -> dict:
    df = F.build_dataset(save=False)

    genres = sorted(df["track_genre"].unique().tolist())

    # Top artists by catalog presence -> {name: track_count} for the dropdown.
    artist_counts = (
        df.drop_duplicates("artist_main")
        .nlargest(top_artists, "artist_track_count")
        .set_index("artist_main")["artist_track_count"]
        .astype(int)
        .to_dict()
    )

    defaults = {f: float(round(df[f].median(), 4)) for f in NATURAL_FEATURES}
    defaults["key"] = int(df["key"].median())
    defaults["mode"] = int(df["mode"].median())
    defaults["time_signature"] = int(df["time_signature"].median())

    meta = {
        "genres": genres,
        "artist_counts": artist_counts,
        "feature_defaults": defaults,
        "brazilian_genres": BRAZILIAN_GENRES,
        "hit_threshold": float(df.attrs["hit_threshold"]),
        "hit_rate": float(round(df["is_hit"].mean(), 4)),
    }
    return meta


if __name__ == "__main__":
    meta = build()
    APP_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(APP_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {APP_DATA_PATH.relative_to(APP_DATA_PATH.parent.parent)}")
    print(f"  genres: {len(meta['genres'])}")
    print(f"  artists: {len(meta['artist_counts'])}")
    print(f"  hit_threshold: {meta['hit_threshold']:.0f}")
