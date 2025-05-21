# app/scheduler_tasks.py
import time
import requests
from datetime import datetime
import pandas as pd
import os
import sys

# Use relative imports, assuming this is run as part of the 'app' package
try:
    from .config import (
        FASTAPI_URL, TICKERS_CONFIG, TARGET_COLUMN,
        get_raw_data_path # To check if raw data was fetched
    )
    from .data_ingestion import run_ingestion_for_all_tickers # Ingests for all
    # data_processing might not be directly called by scheduler if it's part of training
    # or if API handles feature creation on-the-fly from DB data.
    # However, we need to get latest close prices to update DB.
    from .db_utils import save_actual_prices, update_actual_price_for_prediction
except ImportError:
    print("SCHEDULER_TASKS: Error during relative import. Ensure PYTHONPATH is correct or run as part of a package.")
    # Fallback for potential direct execution or different environment (less ideal)
    # This assumes scripts are in the same directory or PYTHONPATH is set globally
    from config import (
        FASTAPI_URL, TICKERS_CONFIG, TARGET_COLUMN,
        get_raw_data_path
    )
    from data_ingestion import run_ingestion_for_all_tickers
    from db_utils import save_actual_prices, update_actual_price_for_prediction


def daily_data_ingestion_and_db_update_job():
    """
    Scheduled job to:
    1. Ingest fresh raw data for all configured tickers.
    2. Extract latest 'Close' prices from this raw data.
    3. Save these latest 'Close' prices to the 'stock_prices' table in the DB.
    4. Update 'actual_price' in the 'predictions' table for past prediction dates.
    """
    print(f"SCHEDULER_TASK: [{datetime.now()}] Running daily data ingestion and DB update job...")
    
    # 1. Ingest fresh raw data for all tickers
    print(f"SCHEDULER_TASK: Calling run_ingestion_for_all_tickers()...")
    run_ingestion_for_all_tickers() # This function saves raw data to CSV files

    # 2. For each ticker, read its newly ingested raw CSV, extract 'Close', and update DB
    for ticker_key in TICKERS_CONFIG.keys():
        print(f"SCHEDULER_TASK: Processing DB update for {ticker_key}...")
        raw_file_path = get_raw_data_path(ticker_key)
        try:
            if not os.path.exists(raw_file_path):
                print(f"SCHEDULER_TASK: Raw data file {raw_file_path} not found for {ticker_key}. Skipping DB update for this ticker.")
                continue

            # Read the raw data, ensuring Date is parsed
            df_raw_today = pd.read_csv(raw_file_path, parse_dates=['Date'])
            if df_raw_today.empty:
                print(f"SCHEDULER_TASK: Raw data for {ticker_key} is empty. Skipping DB update.")
                continue
            
            df_raw_today.set_index('Date', inplace=True)
            df_raw_today.sort_index(inplace=True)

            # Decide which column to use as 'Close' (prefer 'Adj_Close')
            # TARGET_COLUMN from config is 'Close'. We make this effective_close.
            effective_close_series = None
            if 'Adj_Close' in df_raw_today.columns and not df_raw_today['Adj_Close'].isnull().all():
                effective_close_series = df_raw_today['Adj_Close']
                print(f"SCHEDULER_TASK: Using 'Adj_Close' for DB update for {ticker_key}.")
            elif 'Close' in df_raw_today.columns and not df_raw_today['Close'].isnull().all():
                effective_close_series = df_raw_today['Close']
                print(f"SCHEDULER_TASK: Using 'Close' for DB update for {ticker_key}.")
            else:
                print(f"SCHEDULER_TASK: Neither 'Adj_Close' nor 'Close' found or usable for {ticker_key}. Skipping DB update.")
                continue
            
            # Create a DataFrame suitable for save_actual_prices (index=Date, column='Close')
            df_for_db_update = pd.DataFrame(effective_close_series.rename('Close'))
            df_for_db_update.dropna(subset=['Close'], inplace=True) # Remove any NaN close prices

            if not df_for_db_update.empty:
                # 3. Save these latest 'Close' prices to the 'stock_prices' table
                save_actual_prices(ticker_key, df_for_db_update)

                # 4. Update 'actual_price' in the 'predictions' table for past prediction dates
                # Only update for the dates we just fetched prices for
                for date_val, row_data in df_for_db_update.iterrows():
                    date_str = date_val.strftime('%Y-%m-%d')
                    actual_close = row_data['Close']
                    update_actual_price_for_prediction(ticker_key, date_str, actual_close)
            else:
                print(f"SCHEDULER_TASK: No valid close prices to update in DB for {ticker_key}.")

        except Exception as e:
            print(f"SCHEDULER_TASK: Error processing DB update for {ticker_key}: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"SCHEDULER_TASK: [{datetime.now()}] Daily data ingestion and DB update job finished.")


def daily_prediction_trigger_job():
    """
    Scheduled job to trigger predictions for all configured tickers and relevant models via API.
    """
    print(f"SCHEDULER_TASK: [{datetime.now()}] Running daily prediction trigger job...")
    
    # Define which models to run predictions for (could also be in config)
    models_to_predict_with = ["xgboost", "random_forest", "lstm"]

    for ticker_key in TICKERS_CONFIG.keys():
        for model_type in models_to_predict_with:
            print(f"SCHEDULER_TASK: Triggering prediction for {ticker_key} using {model_type} model...")
            try:
                predict_url = f"{FASTAPI_URL}/predict" # As defined in config.py
                params = {"ticker_key": ticker_key, "model_type": model_type}
                
                response = requests.post(predict_url, params=params, timeout=60) # Added timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                
                prediction_data = response.json()
                print(f"SCHEDULER_TASK: API Prediction successful for {ticker_key} ({model_type}): {prediction_data.get('predicted_price')}")
            
            except requests.exceptions.HTTPError as http_err:
                print(f"SCHEDULER_TASK: HTTP error calling /predict for {ticker_key} ({model_type}): {http_err}")
                if http_err.response is not None:
                    print(f"    Response Status: {http_err.response.status_code}")
                    try: print(f"    Response Body: {http_err.response.json()}")
                    except ValueError: print(f"    Response Body (text): {http_err.response.text}")
            except requests.exceptions.RequestException as req_err:
                print(f"SCHEDULER_TASK: Request error calling /predict for {ticker_key} ({model_type}): {req_err}")
            except Exception as e:
                print(f"SCHEDULER_TASK: Unexpected error triggering prediction for {ticker_key} ({model_type}): {e}")
                import traceback
                traceback.print_exc()
            
            time.sleep(1) # Small delay between API calls, if needed

    print(f"SCHEDULER_TASK: [{datetime.now()}] Daily prediction trigger job finished.")

# Optional: Placeholder for a retraining job
# def weekly_model_retraining_job():
#     print(f"SCHEDULER_TASK: [{datetime.now()}] Triggering weekly model retraining (Placeholder)...")
#     # Logic to:
#     # 1. Ensure latest data is ingested and processed (or use data already in DB/CSVs)
#     # 2. Call a script or functions from app.model_training to retrain models for all tickers
#     #    (e.g., by running `python -m app.model_training`)
#     # 3. This might involve restarting the API server if new model files are generated,
#     #    or the API server could be designed to reload models periodically.
#     try:
#         # Example: Triggering the model_training script as a subprocess
#         # This is a simplified approach; more robust solutions might involve message queues or dedicated task runners.
#         # Ensure the command below correctly executes your training script within the Docker environment if this worker is in Docker
#         # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # app -> project_root
#         # training_script_path = os.path.join(project_root, "app", "model_training.py")
#         # result = subprocess.run(["python", training_script_path], capture_output=True, text=True, check=False)
#         # print(f"Retraining stdout: {result.stdout}")
#         # if result.returncode != 0:
#         #     print(f"Retraining stderr: {result.stderr}")
#         print("Retraining logic needs to be implemented.")
#     except Exception as e:
#         print(f"SCHEDULER_TASK: Error during retraining job: {e}")
#     print(f"SCHEDULER_TASK: [{datetime.now()}] Weekly model retraining finished (Placeholder).")

if __name__ == "__main__":
    # For testing individual jobs directly (ensure DB and models exist as needed)
    print("Testing scheduler tasks directly...")
    
    # Test data ingestion and DB update
    # daily_data_ingestion_and_db_update_job() # This will call AlphaVantage/yFinance

    # Test prediction trigger (assumes API server is running and models/data are ready)
    # Note: FASTAPI_URL needs to be accessible from where this script is run.
    # If running this test locally and API server is in Docker, FASTAPI_URL should be http://localhost:8000
    # If this script itself is in Docker, FASTAPI_URL should be http://stock_api_server:8000
    # print(f"API URL for test: {FASTAPI_URL}")
    # daily_prediction_trigger_job()
    
    print("Finished testing scheduler tasks.")