# src/data_prep.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_daily_log(path: str) -> pd.DataFrame:
    """
    Load the daily food log dataset with basic cleaning.
    """
    df = pd.read_csv(path, on_bad_lines="skip")
    # Standardize column names a bit
    df.columns = [c.strip() for c in df.columns]
    return df


def load_food_composition(path: str) -> pd.DataFrame:
    """
    Load the food composition dataset (USDA-like) with vitamins, minerals, etc.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_legacy_nutrients(path: str) -> pd.DataFrame:
    """
    Load the legacy nutrient dataset and clean 't' (trace) values.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Replace 't' (trace) with 0 in numeric columns
    for col in ["Protein", "Fat", "Sat.Fat", "Fiber", "Carbs", "Calories", "Grams"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .replace("t", 0)
                .replace("T", 0)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_macro_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Protein_Ratio, Carb_Ratio, Fat_Ratio based on g of macros.
    """
    df = df.copy()
    macro_sum = (
        df["Protein (g)"].fillna(0)
        + df["Carbohydrates (g)"].fillna(0)
        + df["Fat (g)"].fillna(0)
    )

    macro_sum = macro_sum.replace(0, np.nan)
    df["Protein_Ratio"] = df["Protein (g)"] / macro_sum
    df["Carb_Ratio"]    = df["Carbohydrates (g)"] / macro_sum
    df["Fat_Ratio"]     = df["Fat (g)"] / macro_sum
    df[["Protein_Ratio", "Carb_Ratio", "Fat_Ratio"]] = df[
        ["Protein_Ratio", "Carb_Ratio", "Fat_Ratio"]
    ].fillna(0)

    return df


def add_energy_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calories density (kcal per gram of macros).
    """
    df = df.copy()
    macro_g = (
        df["Protein (g)"].fillna(0)
        + df["Carbohydrates (g)"].fillna(0)
        + df["Fat (g)"].fillna(0)
    )
    macro_g = macro_g.replace(0, np.nan)
    df["Energy_Density_kcal_per_g_macro"] = df["Calories (kcal)"] / macro_g
    df["Energy_Density_kcal_per_g_macro"] = df[
        "Energy_Density_kcal_per_g_macro"
    ].replace([np.inf, -np.inf], np.nan).fillna(df["Energy_Density_kcal_per_g_macro"].median())
    return df


def encode_categoricals(df: pd.DataFrame, cols=("Category", "Meal_Type")):
    """
    Label-encode categorical columns and return df + fitted encoders.
    """
    df = df.copy()
    encoders = {}
    for col in cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def define_healthy_label(df: pd.DataFrame) -> pd.Series:
    """
    WHO/USDA-inspired heuristic label:
      - Sugars < 25g
      - Fat    < 70g
      - Fiber  >= 5g
    Returns a binary Series (1=healthy, 0=unhealthy).
    """
    healthy = (
        (df["Sugars (g)"] < 25) &
        (df["Fat (g)"] < 70) &
        (df["Fiber (g)"] >= 5)
    ).astype(int)
    return healthy


def scale_features(df: pd.DataFrame, numeric_cols):
    """
    Standardize numeric columns and return (scaled_df, scaler).
    """
    df = df.copy()
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace inf/-inf and fill NaNs with column medians.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(df.median(numeric_only=True))



def unify_macro_schema(
    daily_df: pd.DataFrame,
    food_comp_df: pd.DataFrame,
    legacy_df: pd.DataFrame,
):
    """
    Return three dataframes with a unified macro schema:
    columns: Calories, Protein (g), Carbohydrates (g), Fat (g), Fiber (g), Carbs
    Used for cross-dataset EDA.
    """
 
    daily = daily_df.rename(
        columns={
            "Calories (kcal)": "Calories",
            "Carbohydrates (g)": "Carbohydrates (g)",
        }
    )

    food = food_comp_df.copy()
    if "Data.Energy.Kilocalories" in food.columns:
        food["Calories"] = food["Data.Energy.Kilocalories"]
    if "Data.Protein" in food.columns:
        food["Protein (g)"] = food["Data.Protein"]
    if "Data.Carbohydrate" in food.columns:
        food["Carbohydrates (g)"] = food["Data.Carbohydrate"]
    if "Data.Total lipid (fat)" in food.columns:
        food["Fat (g)"] = food["Data.Total lipid (fat)"]


    legacy = legacy_df.rename(
        columns={
            "Calories": "Calories",
            "Protein": "Protein (g)",
            "Carbs": "Carbohydrates (g)",
            "Fat": "Fat (g)",
        }
    )

    return daily, food, legacy
