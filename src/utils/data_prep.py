import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a CSV or TXT file."""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.txt'):
        return pd.read_csv(filepath, delimiter='|')
    else:
        raise ValueError("Unsupported file format.")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing: drop duplicates, handle missing values, etc."""
    df = df.drop_duplicates()
    df = df.dropna()
    return df
