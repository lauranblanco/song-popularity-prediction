"""
Last.fm enrichment for Song Popularity Prediction.

Reuses the collector pattern from Project 2 (LATAM Market Report). Since the
dataset has tens of thousands of unique artists, we do NOT enrich all of them
(API rate limits): we enrich a SAMPLE of the most prolific artists in the
catalog, to validate the hypothesis that an artist's real popularity
(Last.fm listeners) is associated with the popularity of their songs.
"""

import os
import time
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.getenv("LASTFM_API_KEY")
BASE_URL = "https://ws.audioscrobbler.com/2.0/"

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ENRICH_PATH = PROCESSED_DIR / "lastfm_artist_stats.csv"


def _get(params: dict) -> dict:
    params.update({"api_key": API_KEY, "format": "json"})
    r = requests.get(BASE_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def get_artist_stats(artist_name: str) -> dict:
    """Return the artist's global listeners and playcount (artist.getInfo)."""
    try:
        data = _get({"method": "artist.getInfo", "artist": artist_name})
        stats = data.get("artist", {}).get("stats", {})
        return {
            "lastfm_listeners": int(stats.get("listeners", 0)),
            "lastfm_playcount": int(stats.get("playcount", 0)),
        }
    except Exception:
        return {"lastfm_listeners": None, "lastfm_playcount": None}


def enrich_top_artists(df: pd.DataFrame, top_n: int = 300, delay: float = 0.25) -> pd.DataFrame:
    """
    Enrich the top_n most frequent artists in the dataset with Last.fm stats.

    Args:
        df: feature dataset (must have columns 'artist_main' and 'artist_track_count').
        top_n: number of artists to enrich (the most prolific ones).
        delay: pause between calls to respect the rate limit.
    """
    if not API_KEY:
        raise RuntimeError("Missing LASTFM_API_KEY in .env")

    top_artists = (
        df.drop_duplicates("artist_main")
        .nlargest(top_n, "artist_track_count")["artist_main"]
        .tolist()
    )

    rows = []
    for i, name in enumerate(top_artists):
        stats = get_artist_stats(name)
        rows.append({"artist_main": name, **stats})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(top_artists)} artists enriched...")
        time.sleep(delay)

    out = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(ENRICH_PATH, index=False)
    print(f"\nSaved -> {ENRICH_PATH.name} ({len(out)} artists)")
    return out


if __name__ == "__main__":
    from features import build_dataset

    df = build_dataset(save=False)
    enrich_top_artists(df, top_n=300)
