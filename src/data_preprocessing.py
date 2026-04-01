import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {filepath}")

    if df.empty:
        raise ValueError("Loaded dataset is empty.")
        
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and basic data cleaning for the Telco dataset.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Example for Telco Churn: TotalCharges is sometimes empty string
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        # Simple imputation with median for this cleaning step
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median())
    
    # Drop CustomerID as it's not a feature
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop('customerID', axis=1)
        
    return df_clean

def split_data(
    df: pd.DataFrame, 
    target_column: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the DataFrame into training and testing sets.
    
    Args:
        df: Input cleaned DataFrame.
        target_column: Name of the target variable.
        test_size: Proportion of data to include in the test split.
        random_state: Random seed for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Convert target to numeric if it's categorical (e.g., 'Yes'/'No' to 1/0)
    if y.dtype == 'object':
        y = y.map({'Yes': 1, 'No': 0})
        
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
