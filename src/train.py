import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline
from src.persistence import save_artifact
from src.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, CATEGORICAL_COLS, NUMERICAL_COLS, MODEL_PATH, PIPELINE_PATH

def train_and_save_model(data_path: str, model_path: str, pipeline_path: str):
    """
    Complete training workflow: Load, split, fit preprocessing, fit model, and save artifacts.
    """
    # 1. Load and Clean
    df = load_data(data_path)
    df_clean = clean_data(df)

    # 2. Split
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        df_clean, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
    )

    # 3. Build & Fit Preprocessing Pipeline ONLY on X_train
    pipeline = build_preprocessing_pipeline(CATEGORICAL_COLS, NUMERICAL_COLS)
    X_train_processed = pipeline.fit_transform(X_train_raw)

    # 4. Train Model
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train_processed, y_train)

    # 5. Save Artifacts
    save_artifact(model, model_path)
    save_artifact(pipeline, pipeline_path)
    
    return model, pipeline, X_test_raw, y_test

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
