import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_estimators: int = 100, 
    max_depth: int = None, 
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Args:
        X: Processed feature matrix.
        y: Target values.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees.
        random_state: Random state for reproducibility.
        
    Returns:
        RandomForestClassifier: Trained model object.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=random_state
    )
    model.fit(X, y)
    return model
