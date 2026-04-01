import os
import json
import logging
from src.config import (
    RAW_DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, 
    CATEGORICAL_COLS, NUMERICAL_COLS, MODEL_PATH, PIPELINE_PATH, LOGS_DIR
)
from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline
from src.train import train_model
from src.evaluate import evaluate_model
from src.persistence import save_artifact

# Configure logging
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Orchestrate the end-to-end model training, evaluation, and persistence workflow.
    """
    logger.info("Starting training pipeline.")
    
    # 1. Load Data
    try:
        logger.info(f"Loading data from {RAW_DATA_PATH}")
        df = load_data(RAW_DATA_PATH)
    except FileNotFoundError:
        logger.error(f"Cannot find raw data at {RAW_DATA_PATH}. Ensure the dataset exists.")
        return

    # 2. Preprocess Data
    logger.info("Cleaning data and splitting into train/test sets.")
    df_clean = clean_data(df)
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        df_clean, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
    )

    # 3. Build & Fit Preprocessing Pipeline
    logger.info("Building and fitting feature engineering pipeline.")
    pipeline = build_preprocessing_pipeline(CATEGORICAL_COLS, NUMERICAL_COLS)
    X_train_processed = pipeline.fit_transform(X_train_raw)
    X_test_processed = pipeline.transform(X_test_raw)

    # 4. Train Model
    logger.info("Training the Random Forest model.")
    model = train_model(X_train_processed, y_train, random_state=RANDOM_STATE)

    # 5. Evaluate Model
    logger.info("Evaluating model performance.")
    metrics = evaluate_model(model, X_test_processed, y_test)
    print(f"Model Evaluation Metrics: {json.dumps(metrics, indent=2)}")
    logger.info(f"Metrics: {metrics}")

    # 6. Save Artifacts
    logger.info(f"Saving artifacts to {MODEL_PATH} and {PIPELINE_PATH}.")
    save_artifact(model, MODEL_PATH)
    save_artifact(pipeline, PIPELINE_PATH)
    
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
