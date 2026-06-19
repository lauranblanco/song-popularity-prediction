"""
Song Popularity Predictor — Streamlit app.

Manual feature inputs -> hit probability (top 25% of popularity) + SHAP
explanation. Surfaces the genre-bias caveat so scores are read responsibly.

Run locally:  streamlit run app.py
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
import predict as P  # noqa: E402

ROOT = Path(__file__).parent
META_PATH = ROOT / "app_data" / "metadata.json"

st.set_page_config(page_title="Song Popularity Predictor", page_icon="🎯", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(P.MODEL_PATH)


@st.cache_data
def load_meta():
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


model = load_model()
meta = load_meta()
defaults = meta["feature_defaults"]

# ------------------------------------------------------------------ Sidebar
st.sidebar.header("🎚️ Song features")

genre = st.sidebar.selectbox("Genre", meta["genres"],
                             index=meta["genres"].index("pop") if "pop" in meta["genres"] else 0)

st.sidebar.markdown("**Artist**")
artist_mode = st.sidebar.radio(
    "Artist history", ["🆕 New / unknown artist", "Known artist"],
    label_visibility="collapsed",
)
if artist_mode == "Known artist":
    artist_name = st.sidebar.selectbox("Pick an artist", sorted(meta["artist_counts"].keys()))
    artist_main = artist_name
    artist_track_count = meta["artist_counts"][artist_name]
else:
    artist_main = P.UNKNOWN_ARTIST
    artist_track_count = 1

st.sidebar.divider()
st.sidebar.markdown("**Audio profile**")
danceability = st.sidebar.slider("Danceability", 0.0, 1.0, defaults["danceability"])
energy = st.sidebar.slider("Energy", 0.0, 1.0, defaults["energy"])
valence = st.sidebar.slider("Valence (positivity)", 0.0, 1.0, defaults["valence"])
acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, defaults["acousticness"])
instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, defaults["instrumentalness"])
speechiness = st.sidebar.slider("Speechiness", 0.0, 1.0, defaults["speechiness"])
liveness = st.sidebar.slider("Liveness", 0.0, 1.0, defaults["liveness"])
loudness = st.sidebar.slider("Loudness (dB)", -60.0, 2.0, defaults["loudness"])
tempo = st.sidebar.slider("Tempo (BPM)", 0.0, 220.0, defaults["tempo"])
duration_min = st.sidebar.slider("Duration (min)", 0.5, 10.0, defaults["duration_min"])
explicit = st.sidebar.checkbox("Explicit", value=False)

with st.sidebar.expander("Advanced"):
    key = st.number_input("Key (0–11)", 0, 11, int(defaults["key"]))
    mode = st.selectbox("Mode", [0, 1], index=int(defaults["mode"]),
                        format_func=lambda m: "Major" if m == 1 else "Minor")
    time_signature = st.selectbox("Time signature", [1, 3, 4, 5],
                                  index=[1, 3, 4, 5].index(int(defaults["time_signature"]))
                                  if int(defaults["time_signature"]) in [1, 3, 4, 5] else 2)
    n_genres = st.number_input("Genre breadth (n_genres)", 1, 6, 1)

inputs = {
    "danceability": danceability, "energy": energy, "valence": valence,
    "acousticness": acousticness, "instrumentalness": instrumentalness,
    "speechiness": speechiness, "liveness": liveness, "loudness": loudness,
    "tempo": tempo, "duration_min": duration_min, "explicit": explicit,
    "key": key, "mode": mode, "time_signature": time_signature, "n_genres": n_genres,
    "track_genre": genre, "artist_main": artist_main,
    "artist_track_count": artist_track_count,
}

# ------------------------------------------------------------------ Main
st.title("🎯 Song Popularity Predictor")
st.caption(
    "Estimates the probability that a track lands in the **top 25% of popularity**, "
    "based on its audio features, genre and artist history. "
    "Model: LightGBM · test AUC-ROC ≈ 0.92."
)

X = P.build_input_row(inputs)
proba = P.predict_proba(model, X)
is_hit = proba >= 0.5

col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("Prediction")
    st.metric("Hit probability", f"{proba:.0%}")
    st.progress(proba)
    if is_hit:
        st.success("**Likely a hit** — predicted in the top 25% of popularity.")
    else:
        st.error("**Unlikely to be a hit** — predicted outside the top 25%.")
    st.caption(
        f"Decision threshold 50%. In the catalog, {meta['hit_rate']:.0%} of songs are hits "
        f"(popularity ≥ {meta['hit_threshold']:.0f})."
    )

with col2:
    st.subheader("Why this prediction? (SHAP)")
    with st.spinner("Computing explanation…"):
        expl = P.shap_explanation(model, X)
        import shap
        fig = plt.figure()
        shap.plots.waterfall(expl, max_display=10, show=False)
        st.pyplot(fig, bbox_inches="tight")
        plt.close(fig)
    st.caption("Red pushes toward *hit*, blue pushes toward *no hit*.")

# ------------------------------------------------------------------ Bias caveat
st.divider()
if genre in meta["brazilian_genres"]:
    st.warning(
        f"⚠️ **Low reliability for this genre.** The model is weak on Brazilian/regional "
        f"genres like **{genre}** (per-genre AUC ≈ 0.45–0.62). `popularity` is a *global* "
        f"Spotify score that under-counts regional success, so a locally huge song may score "
        f"low here. Don't use this prediction for regional Brazilian A&R without a "
        f"region-specific target."
    )
else:
    st.info(
        "ℹ️ **Read responsibly.** The model leans heavily on artist & genre signals. "
        "It performs worst on regional Brazilian genres (pagode, samba, mpb…), where global "
        "popularity under-counts local success. See Notebook 03 for the full bias analysis."
    )
