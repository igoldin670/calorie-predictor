# src/train.py
import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import sys

# --- make the src directory importable so 'features' is resolvable everywhere ---
_THIS = Path(__file__).resolve()
_SRC_DIR = _THIS.parent
_REPO_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from features import FeatureBuilder  # noqa: E402

# Paths and constants (relative by default; env vars can override)
INPUT_CSV = os.getenv("TDEE_CSV", str((_REPO_ROOT / "data" / "tdee_data.csv").resolve()))
MODEL_PATH = os.getenv("TDEE_MODEL_PATH", str((_REPO_ROOT / "models" / "tdee_pipeline.joblib").resolve()))
TARGET = "tdee_kcal"
RANDOM_SEED = 42

REQUIRED_COLS = {"age", "sex", "height_cm", "weight_kg", "activity_level", "tdee_kcal"}

def load_data(path):
    df = pd.read_csv(path)

    # Normalize col names if "height"/"weight" are used
    rename_map = {}
    if "height" in df.columns and "height_cm" not in df.columns:
        rename_map["height"] = "height_cm"
    if "weight" in df.columns and "weight_kg" not in df.columns:
        rename_map["weight"] = "weight_kg"
    if rename_map:
        df = df.rename(columns=rename_map)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

    # Drop out-of-range values (simple sanity filters)
    df = df[
        df["age"].between(16, 90)
        & df["height_cm"].between(120, 220)
        & df["weight_kg"].between(35, 220)
    ]
    return df

def split_xy(df: pd.DataFrame):
    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET])
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED)

# Engineered numeric and categorical columns (produced by FeatureBuilder)
NUM_COLS = [
    "age","height_cm","weight_kg","bmi","bmi2",
    "body_fat_pct","resting_hr","steps_per_day",
    "msj_tdee","km_tdee"
]
CAT_COLS = ["activity_level","sex"]

def make_pipeline(random_state=RANDOM_SEED):
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]), NUM_COLS),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), CAT_COLS),
        ],
        remainder="drop"
    )

    model = HistGradientBoostingRegressor(
        random_state=random_state,
        learning_rate=0.06,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.10,
        min_samples_leaf=20
    )

    pipe = Pipeline([
        ("feat", FeatureBuilder()),
        ("pre", preprocess),
        ("model", model),
    ])
    return pipe

def train_and_eval(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_xy(df)

    pipe = make_pipeline()

    # Cross-val on the training set
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_mae = -cross_val_score(
        pipe, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error"
    )
    print(f"[CV] MAE: {cv_mae.mean():.1f} ± {cv_mae.std():.1f}")

    # Fit and evaluate
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)
    resid_std = float(np.std(y_test - pred, ddof=1))

    print(f"[TEST] MAE: {mae:.1f}")
    print(f"[TEST] RMSE: {rmse:.1f}")
    print(f"[TEST] R²: {r2:.3f}")
    print(f"[TEST] Residual std: {resid_std:.1f}")

    # Save model + metadata
    model_path = Path(MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "meta": {"residual_std": resid_std}}, model_path)
    print(model_path)

if __name__ == "__main__":
    print(f"Reading: {INPUT_CSV}")
    df = load_data(INPUT_CSV)
    train_and_eval(df)
    print(df[TARGET].describe())
