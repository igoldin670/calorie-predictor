# TDEE Calorie Predictor

A small, reproducible **Total Daily Energy Expenditure** (TDEE) predictor.  
It bundles a scikit-learn pipeline (custom feature engineering → preprocessing → gradient boosting) and a **Streamlit** app so anyone can train a model and get calorie estimates from basic inputs.

---

## Quickstart

    # 1) Create & activate a clean env (recommended)
    python -m venv .venv
    # Windows: .venv\Scripts\activate
    source .venv/bin/activate

    # 2) Install deps
    pip install -r requirements.txt

    # 3) (Optional) Train a model from the sample CSV
    python src/train.py

    # 4) Run the app
    streamlit run src/streamlit_app.py

If you already have a trained model somewhere else, point the app to it:

    export TDEE_MODEL_PATH=/path/to/tdee_pipeline.joblib
    streamlit run src/streamlit_app.py

---

## What’s included

- **Training script** (`src/train.py`)  
  - 80/20 train/test split  
  - 5-fold CV on the training split  
  - Saves `models/tdee_pipeline.joblib` with the *entire* pipeline + a small `meta` dict containing `residual_std` (used by the app to show a ±1σ band).

- **Feature engineering** (`src/features.py`)  
  - Adds BMI/BMR/TDEE baseline features (Mifflin–St Jeor & Katch–McArdle)  
  - Handles messy inputs (coercion, optional fields, activity synonyms)  
  - Implemented as an sklearn transformer (`FeatureBuilder`) so pickling/cloning works anywhere.

- **Streamlit app** (`src/streamlit_app.py`)  
  - Simple form for age/sex/height/weight/activity (+ optional body fat %, RHR, steps)  
  - Displays predicted kcal/day and a quick ±1σ “comfort band”  
  - Includes a tiny “legacy shim” so older models saved with `FunctionTransformer(build_features)` also load.

---

## Project structure

    calorie-predictor/
    ├─ src/
    │  ├─ features.py          # FeatureBuilder transformer (BMI/BMR, etc.)
    │  ├─ train.py             # Train & save pipeline bundle
    │  └─ streamlit_app.py     # Streamlit UI for predictions
    ├─ data/
    │  └─ tdee_data.csv        # Small sample (for quick training)
    ├─ models/
    │  └─ .gitkeep             # Folder for saved models (ignored by git)
    ├─ requirements.txt        # Pinned dependencies
    ├─ README.md               # You are here
    ├─ .gitignore
    └─ LICENSE

---

## Data schema

The training CSV should have these columns (headers are case-sensitive):

| column           | type   | notes |
|------------------|--------|-------|
| `age`            | int    | 16–90 filtered by default |
| `sex`            | str    | `male` / `female` (common aliases like `m`/`f` mapped) |
| `height_cm`      | float  | 120–220 filtered |
| `weight_kg`      | float  | 35–220 filtered |
| `activity_level` | str    | one of: `sedentary`, `light`, `moderate`, `very`, `athlete` (phrases are folded, e.g. “very active” → `very`) |
| `body_fat_pct`   | float? | optional (used for Katch–McArdle) |
| `resting_hr`     | float? | optional |
| `steps_per_day`  | float? | optional |
| `tdee_kcal`      | float  | **target** |

**Minimal example row:**

    age,sex,height_cm,weight_kg,activity_level,body_fat_pct,resting_hr,steps_per_day,tdee_kcal
    28,male,178,72,moderate,,60,8000,2550

---

## How it works

1. **FeatureBuilder** (`src/features.py`) runs first and adds:
   - `height_m`, `bmi`, `bmi2`
   - **Mifflin–St Jeor BMR** and **Katch–McArdle BMR** (if body fat % present)
   - Activity-adjusted TDEE baselines (`msj_tdee`, `km_tdee`) using:  
     `sedentary: 1.20, light: 1.375, moderate: 1.55, very: 1.725, athlete: 1.90`
2. **ColumnTransformer** imputes & scales numerics, imputes & one-hot encodes categoricals.
3. **HistGradientBoostingRegressor** fits the final predictor (with early stopping).
4. Training prints CV/test metrics and saves:  
   `{"pipeline": sklearn_pipeline, "meta": {"residual_std": <float>}}`
5. The app loads that bundle and shows `prediction ± residual_std`.

---

## Configuration

All paths have sensible defaults, but you can override via env vars:

- **`TDEE_CSV`** – input CSV for training  (default: `data/tdee_data.csv`)
- **`TDEE_MODEL_PATH`** – where to save/read the model  (default: `models/tdee_pipeline.joblib`)

Examples:

    TDEE_CSV=/path/to/my.csv python src/train.py
    TDEE_MODEL_PATH=/tmp/model.joblib streamlit run src/streamlit_app.py

---

## Troubleshooting

- **Model not found in the app**  
  Run `python src/train.py` first, or set `TDEE_MODEL_PATH` to an existing `.joblib`.

- **Cross-val cloning error about FeatureBuilder**  
  You’re likely on an older copy. This repo’s `FeatureBuilder` is clone-safe; make sure you’re using *this* `src/features.py`.

- **Prediction failed (column/type mismatch)**  
  Ensure your input row has these raw fields:  
  `age, sex, height_cm, weight_kg, activity_level, body_fat_pct, resting_hr, steps_per_day`

- **Large files rejected by git**  
  Don’t commit `.joblib`. Use Releases or Git LFS.

---

## Deploy (optional)

- **Streamlit Community Cloud**  
  Point it at `src/streamlit_app.py`. If you also host a pre-trained model, download it on startup or store it in a release and set `TDEE_MODEL_PATH` via app secrets.

- **Docker** (sketch)

    FROM python:3.10-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY src/ src/
    COPY models/ models/
    EXPOSE 8501
    CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

---

## Contributing

Issues and PRs welcome! Please:
- keep `features.py` clone-safe (don’t mutate params in `__init__`),
- avoid committing large data/models,
- include a quick note on how you validated changes.

---

## License

MIT — see [LICENSE](./LICENSE).
