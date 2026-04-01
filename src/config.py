import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'telco_churn.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'cleaned_data.csv')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pkl')
PIPELINE_PATH = os.path.join(MODELS_DIR, 'pipeline.pkl')

# ML configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = 'Churn'

# Feature configuration
CATEGORICAL_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
    'PhoneService', 'MultipleLines', 'InternetService', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
