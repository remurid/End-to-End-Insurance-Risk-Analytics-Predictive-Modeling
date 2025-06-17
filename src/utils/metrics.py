import pandas as pd

def compute_claim_frequency(df: pd.DataFrame, group_col: str) -> pd.Series:
    """Proportion of policies with at least one claim per group."""
    return df.groupby(group_col)['ClaimCount'].apply(lambda x: (x > 0).mean())

def compute_claim_severity(df: pd.DataFrame, group_col: str) -> pd.Series:
    """Average claim amount per group, given a claim occurred."""
    return df[df['ClaimCount'] > 0].groupby(group_col)['ClaimAmount'].mean()

def compute_margin(df: pd.DataFrame, group_col: str) -> pd.Series:
    """Margin = TotalPremium - TotalClaims per group."""
    return df.groupby(group_col).apply(lambda x: x['TotalPremium'].sum() - x['TotalClaims'].sum())
