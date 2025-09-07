#src/streamlit_app.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

#
#paths (defaults work out of the box; env var can override)
#
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "tdee_pipeline.joblib"
MODEL_PATH = Path(os.getenv("TDEE_MODEL_PATH", str(DEFAULT_MODEL))).resolve()

#
#legacy shim: if an old model was saved with FunctionTransformer(build_features)
#joblib wants this function alive at import time. no harm keeping it here.
#
ACTIVITY_MULT = {
    "sedentary": 1.20,
    "light": 1.375,
    "moderate": 1.55,
    "very": 1.725,
    "athlete": 1.90,
}

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("body_fat_pct", "resting_hr", "steps_per_day"):
        if c not in df.columns:
            df[c] = np.nan

    df["height_m"] = df["height_cm"] / 100.0
    df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)
    df["bmi2"] = df["bmi"] ** 2

    base = 10.0 * df["weight_kg"] + 6.25 * df["height_cm"] - 5.0 * df["age"]
    is_male = df["sex"].astype(str).str.lower().eq("male")
    df["msj_bmr"] = base + np.where(is_male, 5.0, -161.0)

    has_bf = df["body_fat_pct"].notna()
    lbm = df["weight_kg"] * (1.0 - df["body_fat_pct"] / 100.0)
    df["km_bmr"] = np.where(has_bf, 370.0 + 21.6 * lbm, np.nan)

    mult = (
        df["activity_level"].astype(str).str.lower()
        .map(ACTIVITY_MULT)
        .fillna(1.20)
    )
    df["msj_tdee"] = df["msj_bmr"] * mult
    df["km_tdee"]  = df["km_bmr"]  * mult
    return df

#
#loading
#
@st.cache_resource
def load_pipeline_and_meta(path: Path):
    """Expect {"pipeline": pipe, "meta": {"residual_std": float}}."""
    obj = joblib.load(path)
    pipe = obj["pipeline"]
    resid_std = float(obj.get("meta", {}).get("residual_std", np.nan))
    return pipe, resid_std

#
#UI
#
st.set_page_config(page_title="TDEE Predictor", page_icon="🔥", layout="centered")
st.title("TDEE Predictor (kcal/day)")
st.caption("Tiny app on top of a HistGradientBoostingRegressor.")

with st.sidebar:
    st.subheader("Model")
    st.write(f"Path:\n`{MODEL_PATH}`")
    if not MODEL_PATH.exists():
        st.error("Model not found. Train it (python src/train.py) or set TDEE_MODEL_PATH.")
        st.stop()
    try:
        pipe, resid_std = load_pipeline_and_meta(MODEL_PATH)
        st.success("Model loaded, all good.")
        if np.isfinite(resid_std):
            st.caption(f"Residual std (test): {resid_std:,.1f} kcal")
    except Exception as e:
        st.error("Could not load the pipeline bundle :(")
        st.exception(e)
        st.stop()

st.markdown("### Your info (we'll guess your burn)")

with st.form("tdee_form", clear_on_submit=False):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", 16, 90, 25, step=1)
        height_cm = st.number_input("Height (cm)", 120, 220, 175, step=1)
        weight_kg = st.number_input("Weight (kg)", 35, 220, 70, step=1)
        sex = st.selectbox("Sex", ["male", "female"], index=0)
    with c2:
        activity_level = st.selectbox("Activity level", ["sedentary","light","moderate","very","athlete"], index=2)
        provide_bf = st.checkbox("Provide body fat %", value=False)
        body_fat_pct = st.number_input("Body fat (%)", 0.0, 70.0, 20.0, step=0.5, disabled=not provide_bf)
        resting_hr = st.number_input("Resting heart rate (bpm)", 30, 220, 60, step=1)
        steps_per_day = st.number_input("Steps per day", 0, 100000, 8000, step=500)
    submitted = st.form_submit_button("Predict")

#bmi readout (not used by model directly, just for vibes)
height_m = height_cm / 100.0
bmi = weight_kg / (height_m ** 2)
st.write(f"**BMI:** {bmi:.1f}")

if submitted:
    #the raw schema the pipeline expects (FeatureBuilder handles the rest)
    X = pd.DataFrame([{
        "age": int(age),
        "sex": str(sex).lower().strip(),
        "height_cm": float(height_cm),
        "weight_kg": float(weight_kg),
        "activity_level": str(activity_level).lower().strip(),
        "body_fat_pct": float(body_fat_pct) if provide_bf else np.nan,
        "resting_hr": float(resting_hr),
        "steps_per_day": float(steps_per_day),
    }])
    try:
        yhat = float(pipe.predict(X)[0])
        st.success(f"Predicted TDEE: **{yhat:,.0f} kcal/day**")
        if np.isfinite(resid_std):
            lo, hi = yhat - resid_std, yhat + resid_std
            st.caption(f"±1σ: {lo:,.0f} – {hi:,.0f} kcal/day")

        with st.expander("peek at engineered features (for this input)"):
            # purely for transparency; the pipeline will engineer these anyway
            st.dataframe(build_features(X.copy()))

    except Exception as e:
        st.error("Prediction failed (probably a column/type issue).")
        st.exception(e)
        st.caption("Expected raw columns: age, sex, height_cm, weight_kg, activity_level, body_fat_pct, resting_hr, steps_per_day")

st.markdown("---")
st.caption("Set `TDEE_MODEL_PATH` if your model lives somewhere else. Default: models/tdee_pipeline.joblib")