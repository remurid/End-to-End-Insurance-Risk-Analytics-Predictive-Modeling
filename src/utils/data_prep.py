import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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

def handle_missing_data(df: pd.DataFrame, strategy: str = 'mean', columns: list = None) -> pd.DataFrame:
    """Impute missing values for specified columns using the given strategy ('mean', 'median', 'most_frequent', 'constant')."""
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    imputer = SimpleImputer(strategy=strategy)
    df[columns] = imputer.fit_transform(df[columns])
    return df

def encode_categorical(df: pd.DataFrame, columns: list = None, method: str = 'onehot') -> pd.DataFrame:
    """Encode categorical columns using one-hot or label encoding."""
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if method == 'onehot':
        df = pd.get_dummies(df, columns=columns, drop_first=True)
    elif method == 'label':
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features relevant to claims and premiums."""
    # Example: Claim frequency, claim ratio, etc. Adjust as needed.
    if 'TotalClaims' in df.columns and 'CalculatedPremiumPerTerm' in df.columns:
        df['ClaimRatio'] = df['TotalClaims'] / (df['CalculatedPremiumPerTerm'] + 1e-6)
    # Add more feature engineering as needed
    return df

def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    """Split the data into train and test sets."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
