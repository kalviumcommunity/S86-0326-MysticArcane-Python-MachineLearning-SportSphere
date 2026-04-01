import os
import json
import logging
from src.config import (
    RAW_DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, 
    CATEGORICAL_COLS, NUMERICAL_COLS, MODEL_PATH, PIPELINE_PATH, LOGS_DIR
)
from src.train import train_and_save_model
from src.evaluate import evaluate_model
from src.predict import predict_on_new_data

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
    Orchestrate the end-to-end model training, evaluation, and inference workflow.
    Ensures clear separation of Training and Inference modes.
    """
    # ====== TRAINING MODE ======
    logger.info("--- TRAINING MODE STARTING ---")
    
    try:
        # Load, Split, Fit (Preprocess & Model), and Save
        model, pipeline, X_test_raw, y_test = train_and_save_model(
            RAW_DATA_PATH, MODEL_PATH, PIPELINE_PATH
        )
        logger.info("Training and persistence successful.")

        # 2. Evaluation using specific evaluation module
        logger.info("Evaluating training results on held-out test data.")
        X_test_processed = pipeline.transform(X_test_raw)
        metrics = evaluate_model(model, X_test_processed, y_test)
        print(f"Post-Training Evaluation Metrics: {json.dumps(metrics, indent=2)}")
        logger.info(f"Training Metrics: {metrics}")

    except Exception as e:
        logger.error(f"Critical error during training: {str(e)}")
        return

    # ====== INFERENCE MODE (SIMULATED) ======
    logger.info("--- INFERENCE MODE STARTING (Simulation) ---")
    
    # In practice, this would run in a different session/API
    # It loads saved artifacts and applies them to new data.
    try:
        # Using a slice of X_test_raw as 'unseen' input
        input_sample = X_test_raw.head(5)
        inference_results = predict_on_new_data(input_sample)
        print("\n--- Inference Sample Results ---")
        print(inference_results)
        logger.info("Inference mode simulation successful.")

    except Exception as e:
        logger.error(f"Critical error during inference: {str(e)}")

    logger.info("End-to-end pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
