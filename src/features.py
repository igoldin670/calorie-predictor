#src/features.py
#kinda small helper module that makes extra columns (BMI etc)
#try to keep it boring so sklearn can pickle/clone it without drama

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["FeatureBuilder", "ACTIVITY_MULT"]

#default activity map (keep keys in lowercase)
ACTIVITY_MULT: Dict[str, float] = {
    "sedentary": 1.20,
    "light": 1.375,
    "moderate": 1.55,
    "very": 1.725,
    "athlete": 1.90,
}

#what we expect to exist before we add fancy stuff
RAW_REQUIRED = ("age", "sex", "height_cm", "weight_kg", "activity_level")
RAW_OPTIONAL = ("body_fat_pct", "resting_hr", "steps_per_day")


def _to_num(sr):
    #if something weird shows up, just make it NaN and move on
    return pd.to_numeric(sr, errors="coerce")


def _canon_activity(s):
    #normalize stuff like "very active" -> "very"
    s = s.astype(str).str.strip().str.lower()
    if s.isin(ACTIVITY_MULT.keys()).all():
        return s
    contains = s.str.contains
    folded = np.where(contains("athlet"), "athlete",
              np.where(contains("very"), "very",
              np.where(contains("moderate"), "moderate",
              np.where(contains("light"), "light",
              np.where(contains("sedent"), "sedentary", s)))))
    return pd.Series(folded, index=s.index)


def _canon_sex(s):
    #normalizer: m/f etc -> male/female
    s = s.astype(str).str.strip().str.lower()
    return s.replace({
        "m": "male", "man": "male", "boy": "male",
        "f": "female", "woman": "female", "girl": "female",
    })


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer that adds engineered features. Nothing wild.
    Keep __init__ super plain so sklearn can clone it (CV needs that).
    """

    def __init__(
        self,
        activity_mult: Optional[Dict[str, float]] = None,  #pass custom map if you want
        default_activity: str = "sedentary",
        strict: bool = False,  #if True, missing required cols raise
    ):
        #IMPORTANT: don't mutate/derive here (sklearn clone rule)
        self.activity_mult = activity_mult
        self.default_activity = default_activity
        self.strict = strict

    def fit(self, X: pd.DataFrame, y: Any = None):
        #nothing to learn here really
        return self

    def transform(self, df):
        df = df.copy()

        #make sure the columns we need are around
        missing_req = [c for c in RAW_REQUIRED if c not in df.columns]
        if missing_req:
            if self.strict:
                raise ValueError(f"Missing required columns: {missing_req}")
            for c in missing_req:
                df[c] = np.nan  #pipeline will impute later

        for c in RAW_OPTIONAL:
            if c not in df.columns:
                df[c] = np.nan

        #tidy text-ish stuff
        df["sex"] = _canon_sex(df["sex"])
        df["activity_level"] = _canon_activity(df["activity_level"])

        #force numerics, anything funky -> NaN
        for c in ("age", "height_cm", "weight_kg", "body_fat_pct", "resting_hr", "steps_per_day"):
            df[c] = _to_num(df[c])

        df.loc[~(df["height_cm"] > 0), "height_cm"] = np.nan
        df.loc[~(df["weight_kg"] > 0), "weight_kg"] = np.nan

        df["height_m"] = df["height_cm"] / 100.0
        df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)
        df["bmi2"] = df["bmi"] ** 2

        #Mifflin–St Jeor
        base = 10.0 * df["weight_kg"] + 6.25 * df["height_cm"] - 5.0 * df["age"]
        df["msj_bmr"] = base + np.where(df["sex"].eq("male"), 5.0, -161.0)

        #Katch–McArdle
        has_bf = df["body_fat_pct"].notna()
        lbm = df["weight_kg"] * (1.0 - (df["body_fat_pct"] / 100.0))
        df["km_bmr"] = np.where(has_bf, 370.0 + 21.6 * lbm, np.nan)

        #pick the map (custom or default), and a safe fallback key
        mult_map = self.activity_mult if self.activity_mult is not None else ACTIVITY_MULT
        fallback = self.default_activity if self.default_activity in mult_map else "sedentary"
        mult = df["activity_level"].map(mult_map).fillna(mult_map[fallback])

        df["msj_tdee"] = df["msj_bmr"] * mult
        df["km_tdee"] = df["km_bmr"] * mult

        return df
