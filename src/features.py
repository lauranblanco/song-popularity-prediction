"""
Feature engineering for Song Popularity Prediction.

Pipeline that transforms the raw Kaggle dataset into a modeling-ready dataset:
cleaning, deduplication by track_id (avoids data leakage), derived features and
a binary target variable (top 25% of popularity).

Target-dependent features (target encoding of genre/artist) are NOT computed
here: they are left for the modeling pipeline inside cross-validation to avoid
leakage. Only target-independent features are generated here.
"""

from pathlib import Path
import pandas as pd

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_PATH = DATA_DIR / "raw" / "spotify_tracks.csv"
PROCESSED_PATH = DATA_DIR / "processed" / "tracks_features.csv"

HIT_QUANTILE = 0.75  # top 25% of popularity = "hit"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw Kaggle dataset."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop trivial nulls and the index column if present."""
    df = df.copy()
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
    df = df.dropna(subset=["artists", "track_name", "album_name"])
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    A song can appear in several genres (same track_id repeated). To avoid the
    same song landing in both train and test (data leakage), we keep a single
    row per track_id.

    - primary genre = the most prominent genre the song belongs to (highest mean
      popularity of the genre), used as a proxy for its main category.
    - n_genres = how many genres it appears in (genre-breadth feature).
    """
    df = df.copy()
    df["n_genres"] = df.groupby("track_id")["track_genre"].transform("nunique")
    genre_strength = df.groupby("track_genre")["popularity"].transform("mean")
    df["_genre_strength"] = genre_strength
    df = (
        df.sort_values("_genre_strength", ascending=False)
        .drop_duplicates(subset="track_id", keep="first")
        .drop(columns="_genre_strength")
        .reset_index(drop=True)
    )
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate target-independent derived features."""
    df = df.copy()

    # Main artist (the 'artists' field may list several, separated by ';')
    df["artist_main"] = df["artists"].str.split(";").str[0].str.strip()

    # Audio interactions (capture the combined "sound profile")
    df["dance_energy"] = df["danceability"] * df["energy"]
    df["valence_energy"] = df["valence"] * df["energy"]
    df["acoustic_instrumental"] = df["acousticness"] * df["instrumentalness"]

    # Readable transformations
    df["duration_min"] = df["duration_ms"] / 60_000
    df["explicit_int"] = df["explicit"].astype(int)

    # Artist prolificacy in the catalog (career proxy; target-independent)
    df["artist_track_count"] = df.groupby("artist_main")["track_id"].transform("count")

    return df


def add_target(df: pd.DataFrame, quantile: float = HIT_QUANTILE) -> pd.DataFrame:
    """Create the binary target: hit = top (1 - quantile) of popularity."""
    df = df.copy()
    threshold = df["popularity"].quantile(quantile)
    df["is_hit"] = (df["popularity"] >= threshold).astype(int)
    df.attrs["hit_threshold"] = threshold
    return df


def build_dataset(save: bool = True) -> pd.DataFrame:
    """Full pipeline: raw -> modeling-ready feature dataset."""
    df = load_raw()
    n0 = len(df)
    df = clean(df)
    df = deduplicate(df)
    df = add_features(df)
    df = add_target(df)

    print(f"Saved rows:        {n0:,}")
    print(f"After cleaning+dedup: {len(df):,}  ({n0 - len(df):,} removed)")
    print(f"Hit threshold (P{int(HIT_QUANTILE*100)}): popularity >= {df.attrs['hit_threshold']:.0f}")
    print(f"Hit rate:        {df['is_hit'].mean():.1%}")

    if save:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"\nSaved -> {PROCESSED_PATH.relative_to(DATA_DIR.parent)}")
    return df


if __name__ == "__main__":
    build_dataset()
