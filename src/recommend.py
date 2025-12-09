# src/recommend.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_food_profile_matrix(
    df_daily: pd.DataFrame,
    nutrient_cols=None,
    item_col="Food_Item",
):
    """
    Build an average nutrient profile per food item from the daily log.
    """
    if nutrient_cols is None:
        nutrient_cols = [
            "Calories (kcal)",
            "Protein (g)",
            "Carbohydrates (g)",
            "Fat (g)",
            "Fiber (g)",
            "Sugars (g)",
            "Sodium (mg)",
            "Cholesterol (mg)",
        ]

    profiles = df_daily.groupby(item_col)[nutrient_cols].mean()
    profiles = profiles.fillna(profiles.median())
    return profiles


def cosine_similarity_recommender(food_profiles: pd.DataFrame):
    """
    Precompute similarity matrix and return a simple recommend() function.
    """

    similarity_matrix = cosine_similarity(food_profiles.values)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=food_profiles.index,
        columns=food_profiles.index,
    )

    def recommend(food_name: str, top_n=5):
        if food_name not in similarity_df.index:
            raise ValueError(f"{food_name!r} not found in food list.")
        sims = similarity_df.loc[food_name].sort_values(ascending=False)
        return sims.iloc[1 : top_n + 1]

    return similarity_df, recommend
