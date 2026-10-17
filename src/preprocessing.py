"""
Data preprocessing pipeline for the Wine Quality dataset.

Handles:
- Loading raw data
- Removing duplicates
- Binarizing the target (quality >= 7 is "good")
- Train/test split
- Feature scaling (StandardScaler)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw wine data. Note: UCI dataset uses semicolon separator."""
    return pd.read_csv(filepath, sep=";")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and reset the index."""
    return df.drop_duplicates().reset_index(drop=True)


def binarize_target(df: pd.DataFrame, target_col: str = "quality") -> pd.DataFrame:
    """
    Convert multiclass quality score into a binary target.
    Wines with quality >= 7 are marked as 1 (good), others as 0.
    """
    df = df.copy()
    df["is_good"] = (df[target_col] >= 7).astype(int)
    return df.drop(columns=[target_col])


def split_and_scale(df: pd.DataFrame, target_col: str = "is_good"):
    """
    Split into train/test sets (80/20) and apply standard scaling.
    Stratify ensures train and test have the same class balance.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Re-wrap in DataFrames with original column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Add the target back to the DataFrames
    train_df = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)

    return train_df, test_df
