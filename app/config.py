# app/config.py
import os
from dotenv import load_dotenv
from typing import List, Dict

# --- ENVIRONMENT CHECK ---
IS_DOCKER_ENV = os.getenv("AM_I_IN_A_DOCKER_CONTAINER", "false").lower() == "true"

# --- PROJECT AND APP ROOTS ---
APP_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- LOAD .env ---
dotenv_path = os.path.join(PROJECT_ROOT_DIR, '.env')
load_dotenv(dotenv_path=dotenv_path)

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
if not ALPHA_VANTAGE_API_KEY:
    print("WARNING: ALPHA_VANTAGE_API_KEY is not set in .env or environment.")

# --- SUPPORTED TICKERS AND DATA SOURCES ---
TICKERS_CONFIG: Dict[str, Dict[str, str]] = {
    "GSPC": {"source": "yfinance", "api_ticker": "^GSPC", "display_name": "S&P 500 Index"},
    "IBM": {"source": "alpha_vantage", "api_ticker": "IBM", "display_name": "IBM Stock"}
}

# LOCAL PATH AND CONTAINER PATHS
# Local paths
_LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
_LOCAL_MODELS_STORE_DIR = os.path.join(APP_PACKAGE_DIR, "models_store")
_LOCAL_DB_DIR = APP_PACKAGE_DIR

# Container paths
_CONTAINER_APP_CODE_ROOT = "/app/application" # Copied from host ./app code
_CONTAINER_DATA_MOUNT_POINT = "/app/data_volume" # data_volume is mounted here
_CONTAINER_MODELS_STORE_MOUNT_POINT = os.path.join(_CONTAINER_APP_CODE_ROOT, "models_store")
_CONTAINER_DB_FILES_MOUNT_POINT = os.path.join(_CONTAINER_APP_CODE_ROOT, "database_files")

# Choose final paths based on environment
if IS_DOCKER_ENV:
    print("INFO: Config - Running in Docker environment. Using container paths.")
    EFFECTIVE_DATA_DIR = _CONTAINER_DATA_MOUNT_POINT
    EFFECTIVE_MODELS_STORE_DIR = _CONTAINER_MODELS_STORE_MOUNT_POINT
    EFFECTIVE_DB_DIR = _CONTAINER_DB_FILES_MOUNT_POINT
    EFFECTIVE_LSTM_SCALERS_DIR = os.path.join(EFFECTIVE_MODELS_STORE_DIR, "lstm_scalers")
else:
    print("INFO: Config - Running in Local environment. Using local paths.")
    os.makedirs(_LOCAL_DATA_DIR, exist_ok=True)
    os.makedirs(_LOCAL_MODELS_STORE_DIR, exist_ok=True)

    EFFECTIVE_DATA_DIR = _LOCAL_DATA_DIR
    EFFECTIVE_MODELS_STORE_DIR = _LOCAL_MODELS_STORE_DIR
    EFFECTIVE_DB_DIR = _LOCAL_DB_DIR
    EFFECTIVE_LSTM_SCALERS_DIR = os.path.join(EFFECTIVE_MODELS_STORE_DIR, "lstm_scalers")

# Config to final paths
RAW_DATA_PATH_TEMPLATE = os.path.join(EFFECTIVE_DATA_DIR, "{ticker_key}_raw_data.csv")
PROCESSED_DATA_PATH_TEMPLATE = os.path.join(EFFECTIVE_DATA_DIR, "{ticker_key}_processed_data.csv")

KNN_MODEL_NAME_TEMPLATE = "{ticker_key}_knn_model.pkl"
KNN_SCALER_NAME_TEMPLATE = "{ticker_key}_knn_scaler_x.pkl"
XGBOOST_MODEL_NAME_TEMPLATE = "{ticker_key}_xgboost.xgb"
RANDOM_FOREST_MODEL_NAME_TEMPLATE = "{ticker_key}_random_forest.pkl"
LSTM_MODEL_NAME_TEMPLATE = "{ticker_key}_lstm_model.pth"
LSTM_SCALER_NAME_TEMPLATE = "{ticker_key}_lstm_{scaler_type}_scaler.pkl"

DATABASE_NAME = "predictions_database.sqlite"
DATABASE_PATH = os.path.join(EFFECTIVE_DB_DIR, DATABASE_NAME)

def get_raw_data_path(ticker_key: str) -> str:
    return RAW_DATA_PATH_TEMPLATE.format(ticker_key=ticker_key.lower())

def get_processed_data_path(ticker_key: str) -> str:
    return PROCESSED_DATA_PATH_TEMPLATE.format(ticker_key=ticker_key.lower())

def get_model_path(ticker_key: str, model_type: str) -> str:
    filename = ""
    tk_lower = ticker_key.lower()
    if model_type == "xgboost": filename = XGBOOST_MODEL_NAME_TEMPLATE.format(ticker_key=tk_lower)
    elif model_type == "random_forest": filename = RANDOM_FOREST_MODEL_NAME_TEMPLATE.format(ticker_key=tk_lower)
    elif model_type == "lstm": filename = LSTM_MODEL_NAME_TEMPLATE.format(ticker_key=tk_lower)
    elif model_type == "knn": filename = KNN_MODEL_NAME_TEMPLATE.format(ticker_key=tk_lower)
    else: raise ValueError(f"Unsupported model_type for path: {model_type}")
    # Đảm bảo thư mục cha của model tồn tại, đặc biệt khi chạy local
    os.makedirs(EFFECTIVE_MODELS_STORE_DIR, exist_ok=True)
    return os.path.join(EFFECTIVE_MODELS_STORE_DIR, filename)

def get_knn_scaler_path(ticker_key: str) -> str:
    tklower = ticker_key.lower()
    filename = KNN_SCALER_NAME_TEMPLATE.format(ticker_key=tklower)
    base_models_path = EFFECTIVE_MODELS_STORE_DIR
    os.makedirs(base_models_path, exist_ok=True)
    return os.path.join(base_models_path, filename)

def get_lstm_scaler_path(ticker_key: str, scaler_name: str) -> str:
    # scaler_name: ví dụ "features_X_scaler" hoặc "target_y_scaler"
    tk_lower = ticker_key.lower()
    filename = f"{tk_lower}_lstm_{scaler_name}.pkl" # Tên file sẽ là ví dụ: gspc_lstm_features_X_scaler.pkl
    scaler_dir = os.path.join(EFFECTIVE_MODELS_STORE_DIR, "lstm_scalers") # EFFECTIVE_MODELS_STORE_DIR từ logic if/else
    os.makedirs(scaler_dir, exist_ok=True)
    return os.path.join(scaler_dir, filename)

# -- Other Settings --
TARGET_COLUMN = 'Close'
LAG_DAYS_TRADITIONAL = 14
EXPECTED_FEATURES_TRADITIONAL: List[str] = [
    f'{TARGET_COLUMN}_lag_{i}' for i in range(1, LAG_DAYS_TRADITIONAL + 1)
] + [f'{TARGET_COLUMN}_pct_change_1d']


# KNN SETTINGS
EXPECTED_FEATURES_KNN = EXPECTED_FEATURES_TRADITIONAL
KNN_TARGET_COLUMN_NAME = "Price_Direction"

# LSTM SETTINGS
LSTM_SEQUENCE_LENGTH = 20
LSTM_NUM_FEATURES = 4
LSTM_INPUT_FEATURE_COLUMNS = ['Open', 'High', 'Low', TARGET_COLUMN]

MLFLOW_EXPERIMENT_NAME = "StockPrediction_MultiModel_v3"
SCHEDULER_TIMEZONE = 'America/New_York'
DATA_UPDATE_HOUR_ET, DATA_UPDATE_MINUTE_ET = 17, 15
PREDICTION_HOUR_ET, PREDICTION_MINUTE_ET = 17, 45
FASTAPI_BASE_URL_DEFAULT = "http://localhost:8000"
FASTAPI_URL = os.getenv("FASTAPI_URL", FASTAPI_BASE_URL_DEFAULT)

# --- DEBUG PRINTING --
if __name__ == "__main__" or os.getenv("PRINT_CONFIG_ON_LOAD") == "true":
    print(f"--- CONFIG DEBUG INFO (config.py) ---")
    print(f"IS_DOCKER_ENV: {IS_DOCKER_ENV}")
    print(f"PROJECT_ROOT_DIR: {PROJECT_ROOT_DIR}")
    print(f"APP_PACKAGE_DIR: {APP_PACKAGE_DIR}")
    print(f"EFFECTIVE_DATA_DIR: {EFFECTIVE_DATA_DIR}")
    print(f"EFFECTIVE_MODELS_STORE_DIR: {EFFECTIVE_MODELS_STORE_DIR}")
    print(f"EFFECTIVE_DB_DIR: {EFFECTIVE_DB_DIR}")
    print(f"DATABASE_PATH: {DATABASE_PATH}")
    for tk_key in TICKERS_CONFIG.keys():
        print(f"  Paths for {tk_key}:")
        print(f"    Raw Data: {get_raw_data_path(tk_key)}")
        print(f"    Processed Data: {get_processed_data_path(tk_key)}")
        print(f"    XGB Model: {get_model_path(tk_key, 'xgboost')}")
    print(f"--- END CONFIG DEBUG INFO ---")