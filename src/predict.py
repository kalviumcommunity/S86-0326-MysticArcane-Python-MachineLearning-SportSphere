import pandas as pd

def predict(model, pipeline, new_data: pd.DataFrame) -> pd.Series:
    """
    Generate predictions for new instances using a fitted model and pipeline.
    
    Args:
        model: Trained model object.
        pipeline: Fitted scikit-learn preprocessing pipeline.
        new_data: Raw DataFrame containing feature values.
        
    Returns:
        pd.Series: Model's categorical predictions or probabilities.
    """
    X_processed = pipeline.transform(new_data)
    predictions = model.predict(X_processed)
    return predictions
