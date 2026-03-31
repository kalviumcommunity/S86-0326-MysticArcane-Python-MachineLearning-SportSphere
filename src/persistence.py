import joblib
import os

def save_artifact(artifact, filepath: str) -> None:
    """
    Save a model or pipeline artifact to disk.
    
    Args:
        artifact: The object to serialize (model, pipeline, etc.).
        filepath: Full path for the saved file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(artifact, filepath)

def load_artifact(filepath: str):
    """
    Load a model or pipeline artifact from disk.
    
    Args:
        filepath: path for the saved file.
        
    Returns:
        The deserialized artifact.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found at {filepath}")
    return joblib.load(filepath)
