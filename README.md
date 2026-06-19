# 🎯 Song Popularity Prediction Model

🌐 [Leer en Español](README.es.md)

**Author:** Laura Blanco | **Stack:** Python · Pandas · Scikit-learn · LightGBM · SHAP · Streamlit · Joblib · requests
**Live demo:** _pending deployment_

A machine learning model that predicts whether a song will reach the **top 25% of popularity**, based on its *audio features*, genre and artist history. It includes **SHAP** explainability, a **genre-bias analysis**, and a **Streamlit app** to test new songs.

> Project 3 of **Laura Blanco**'s Music Analytics portfolio. Aimed at **A&R, music marketing and streaming** teams: it demonstrates building decision-support tools, not just descriptive analysis.

---

## 🎵 Business problem

A&R and marketing teams need to estimate, *before* investing in promotion, which songs are most likely to become popular. This is framed as **binary classification**:

> Given a song (audio features + genre + artist), will it reach the top quartile of popularity (top 25%)?

---

## 📊 Data

| Source | Use | Size |
|---|---|---|
| **Kaggle — Spotify Tracks Dataset** | Audio features + popularity + genre (model base) | ~114k songs, 114 genres |
| **Last.fm API** (`requests`) | Enrichment over a **sample** of artists: listener counts to validate the artist-popularity signal | top 300 artists |

> The Kaggle dataset is static. Last.fm enrichment is applied to a sample due to API rate limits.

---

## 🔬 Methodology

1. **EDA** ([`01_eda.ipynb`](notebooks/01_eda.ipynb)): popularity distribution, correlations, dominance of genre.
2. **Feature engineering** ([`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb)): dedup by `track_id` (anti-leakage), audio interactions, artist prolificness, Last.fm hypothesis validation.
3. **Modeling** ([`03_modeling.ipynb`](notebooks/03_modeling.ipynb)): Logistic Regression → Random Forest → LightGBM, with cross-fitted *target encoding* of genre/artist (anti-leakage).
4. **Explainability**: SHAP (summary + waterfall).
5. **Bias analysis**: per-genre AUC-ROC + a *cold-start* experiment (no artist history).
6. **App** ([`app.py`](app.py)): Streamlit: manual inputs → hit probability + SHAP waterfall + bias caveat.

---

## 📈 Key results

- **LightGBM: AUC-ROC ≈ 0.92** on test (target > 0.78 ✅), F1 ≈ 0.73.
- **SHAP:** popularity is mostly an attribute of the **artist and the genre**; audio features are secondary.
- **Cold-start** (no artist history): AUC ≈ 0.87 — the tool stays useful for brand-new artists.
- **Documented bias:** the model fails specifically on **Brazilian/regional genres** (pagode, samba, mpb… AUC ≈ 0.45–0.62), while globalized Latin genres (reggaeton, salsa) predict almost perfectly. Cause: `popularity` is a *global* score that under-counts regional success.

---

## 📁 Structure

```
song-popularity-prediction/
├── data/
│   ├── raw/              # Kaggle dataset (not versioned)
│   └── processed/        # features + Last.fm stats (not versioned)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── features.py           # cleaning + feature engineering
│   ├── lastfm_enrich.py      # Last.fm enrichment (sample)
│   ├── model.py              # pipeline, models, evaluation, bias
│   ├── predict.py            # inference + SHAP for the app
│   └── build_app_metadata.py # generates app_data/metadata.json
├── app_data/metadata.json    # genres, artists and defaults (versioned)
├── models/best_model.joblib  # serialized LightGBM (versioned)
├── reports/                  # 13 exported figures
├── app.py                    # Streamlit app
├── requirements.txt
└── README.md
```

---

## 🚀 How to run

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -r requirements.txt

# Reproduce the pipeline (optional; requires the dataset in data/raw/)
python src/features.py                 # builds data/processed/tracks_features.csv
python src/model.py                    # trains and saves models/best_model.joblib

# Launch the app
streamlit run app.py
```

For Last.fm enrichment: `cp .env.example .env` and add your `LASTFM_API_KEY`.

---

## ☁️ Deploy on Streamlit Cloud

The app only needs `app.py`, `src/`, `app_data/metadata.json`, `models/best_model.joblib` and `requirements.txt` (all versioned). Steps:

1. Push the repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io): *New app* → select the repo → main file `app.py`.
3. Deploy. (No secrets required: the model and metadata ship in the repo.)

---

## 🎯 Success KPIs

- [x] AUC-ROC > 0.78 on the test set → **0.92**
- [x] SHAP plots of the most important features
- [x] Limitations and bias section documented
- [ ] App deployed on Streamlit Cloud *(ready locally; push + deploy pending)*

---

## 📌 Status

✅ **Model and app working**. You can access the app with this link: https://song-popularity-prediction-3vpu3x9czymwxyzvp3b3dr.streamlit.app/.

---

*Part of the Music Analytics portfolio · Laura Blanco · 2026*
