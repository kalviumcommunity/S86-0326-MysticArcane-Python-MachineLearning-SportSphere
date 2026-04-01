import pandas as pd
from src.persistence import load_artifact
from src.config import MODEL_PATH, PIPELINE_PATH

def load_inference_artifacts():
    """Load model and pipeline for production use."""
    model = load_artifact(MODEL_PATH)
    pipeline = load_artifact(PIPELINE_PATH)
    return model, pipeline

def predict_on_new_data(input_data: pd.DataFrame):
    """
    Inference core: Load, Transform (No refitting!), Predict.
    """
    model, pipeline = load_inference_artifacts()
    
    # Apply transformation using previously fitted pipeline
    X_processed = pipeline.transform(input_data)
    
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:, 1]

    return pd.DataFrame({
        "prediction": predictions,
        "probability": probabilities
    })

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
