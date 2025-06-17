import pandas as pd
from typing import Tuple

def split_groups(df: pd.DataFrame, feature: str, group_a, group_b) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into two groups based on feature values."""
    group_a_df = df[df[feature] == group_a]
    group_b_df = df[df[feature] == group_b]
    return group_a_df, group_b_df
