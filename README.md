# 🎯 Song Popularity Prediction Model

Modelo de machine learning que predice si una canción alcanzará el **top 25% de popularidad** a partir de sus *audio features*, género e historial del artista. Incluye explicabilidad con **SHAP**, un análisis de **sesgo por género** y una **app en Streamlit** para testear canciones nuevas.

> Proyecto 3 del portafolio de Music Analytics de **Laura Blanco**. Orientado a equipos de **A&R, marketing musical y plataformas de streaming**: demuestra construcción de herramientas de soporte a decisiones, no solo análisis descriptivo.

---

## 🎵 Problema de negocio

Los equipos de A&R y marketing necesitan estimar, *antes* de invertir en promoción, qué canciones tienen mayor probabilidad de volverse populares. Se aborda como **clasificación binaria**:

> Dada una canción (audio features + género + artista), ¿llegará al cuartil superior de popularidad (top 25%)?

---

## 📊 Datos

| Fuente | Uso | Tamaño |
|---|---|---|
| **Kaggle — Spotify Tracks Dataset** | Audio features + popularidad + género (base del modelo) | ~114k canciones, 114 géneros |
| **Last.fm API** (`requests`) | Enriquecimiento sobre una **muestra** de artistas: listener counts como validación de la señal de popularidad del artista | 300 artistas top |

> El dataset de Kaggle es estático. El enriquecimiento con Last.fm se aplica sobre una muestra por límites de rate de la API.

---

## 🔬 Metodología

1. **EDA** ([`01_eda.ipynb`](notebooks/01_eda.ipynb)) — distribución de popularidad, correlaciones, dominancia del género.
2. **Feature engineering** ([`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb)) — dedup por `track_id` (anti-leakage), interacciones de audio, prolificidad del artista, validación de hipótesis con Last.fm.
3. **Modelado** ([`03_modeling.ipynb`](notebooks/03_modeling.ipynb)) — Logistic Regression → Random Forest → LightGBM, con *target encoding* cross-fitted de género/artista (anti-leakage).
4. **Explicabilidad** — SHAP (summary + waterfall).
5. **Análisis de sesgo** — AUC-ROC por género + experimento *cold-start* (sin historial de artista).
6. **App** ([`app.py`](app.py)) — Streamlit: inputs manuales → probabilidad de hit + SHAP waterfall + aviso de sesgo.

---

## 📈 Resultados clave

- **LightGBM: AUC-ROC ≈ 0.92** en test (meta > 0.78 ✅), F1 ≈ 0.73.
- **SHAP:** la popularidad es sobre todo un atributo del **artista y el género**; las audio features son secundarias.
- **Cold-start** (sin historial de artista): AUC ≈ 0.87 — la herramienta sigue siendo útil para artistas nuevos.
- **Sesgo documentado:** el modelo falla específicamente en **géneros brasileños/regionales** (pagode, samba, mpb… AUC ≈ 0.45–0.62), mientras que el latino global (reggaetón, salsa) se predice casi perfecto. Causa: `popularity` es un score *global* que subrepresenta el éxito regional.

---

## 🛠️ Stack

`Python · Pandas · Scikit-learn · LightGBM · SHAP · Streamlit · Joblib · requests`

---

## 📁 Estructura

```
song-popularity-prediction/
├── data/
│   ├── raw/              # dataset Kaggle (no versionado)
│   └── processed/        # features + stats Last.fm (no versionado)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── features.py           # limpieza + feature engineering
│   ├── lastfm_enrich.py      # enriquecimiento Last.fm (muestra)
│   ├── model.py              # pipeline, modelos, evaluación, sesgo
│   ├── predict.py            # inferencia + SHAP para la app
│   └── build_app_metadata.py # genera app_data/metadata.json
├── app_data/metadata.json    # géneros, artistas y defaults (versionado)
├── models/best_model.joblib  # LightGBM serializado (versionado)
├── reports/                  # 13 figuras exportadas
├── app.py                    # app Streamlit
├── requirements.txt
└── README.md
```

---

## 🚀 Cómo ejecutar

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -r requirements.txt

# Reproducir el pipeline (opcional; requiere el dataset en data/raw/)
python src/features.py                 # genera data/processed/tracks_features.csv
python src/model.py                    # entrena y guarda models/best_model.joblib

# Lanzar la app
streamlit run app.py
```

Para el enriquecimiento Last.fm: `cp .env.example .env` y añade tu `LASTFM_API_KEY`.

---

## ☁️ Deploy en Streamlit Cloud

La app solo necesita `app.py`, `src/`, `app_data/metadata.json`, `models/best_model.joblib` y `requirements.txt` (todos versionados). Pasos:

1. Push del repo a GitHub.
2. En [share.streamlit.io](https://share.streamlit.io): *New app* → seleccionar el repo → main file `app.py`.
3. Deploy. (No requiere secrets: el modelo y los metadatos viajan en el repo.)

---

## 🎯 KPIs de éxito

- [x] AUC-ROC > 0.78 en test set → **0.92**
- [x] SHAP plots de las features más importantes
- [x] Sección de limitaciones y sesgos documentada
- [ ] App deployada en Streamlit Cloud *(lista localmente; pendiente push + deploy)*

---

## 📌 Estado

✅ **Modelo y app funcionales** — pendiente únicamente el deploy a Streamlit Cloud.

---

*Parte del portafolio de Music Analytics · Laura Blanco · 2026*
