# app/historical_seeder.py
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys
import numpy as np
from typing import List

from .config import (
    TICKERS_CONFIG, TARGET_COLUMN,
    LAG_DAYS_TRADITIONAL, EXPECTED_FEATURES_TRADITIONAL,
    LSTM_SEQUENCE_LENGTH, LSTM_NUM_FEATURES, LSTM_INPUT_FEATURE_COLUMNS, # Sử dụng biến này
    get_raw_data_path, get_model_path, get_lstm_scaler_path
)
from .model_utils import (
    load_model,
    prepare_features_for_traditional_model,
    prepare_input_sequence_for_lstm,
    make_prediction
)
from .db_utils import init_db, save_actual_prices, save_prediction, update_actual_price_for_prediction

from sklearn.preprocessing import MinMaxScaler # Cần cho LSTM scaling nếu model_utils không export
import torch # Cần cho LSTM input tensor

def populate_all_stock_prices_from_raw_csv():
    # ... (Hàm này giữ nguyên như phiên bản bạn đã xác nhận là OK) ...
    print("SEEDER: Starting to populate 'stock_prices' table from raw CSVs...")
    init_db()

    for ticker_key, config_details in TICKERS_CONFIG.items():
        print(f"SEEDER: Populating stock_prices for {ticker_key}...")
        raw_file_path = get_raw_data_path(ticker_key)
        try:
            if not os.path.exists(raw_file_path):
                print(f"SEEDER: Raw data file {raw_file_path} not found for {ticker_key}. Skipping.")
                continue

            df_raw = pd.read_csv(raw_file_path, parse_dates=['Date'])
            df_raw.dropna(subset=['Date'], inplace=True)
            if df_raw.empty:
                print(f"SEEDER: Raw data for {ticker_key} is empty after NaT Date drop. Skipping.")
                continue
            
            df_raw.set_index('Date', inplace=True)
            df_raw.sort_index(inplace=True)

            effective_close_series = None
            if 'Adj_Close' in df_raw.columns and not df_raw['Adj_Close'].isnull().all():
                effective_close_series = pd.to_numeric(df_raw['Adj_Close'], errors='coerce')
            elif 'Close' in df_raw.columns and not df_raw['Close'].isnull().all():
                effective_close_series = pd.to_numeric(df_raw['Close'], errors='coerce')
            else:
                print(f"SEEDER: No usable 'Close' or 'Adj_Close' for {ticker_key} in stock_prices. Skipping.")
                continue
            
            df_for_db = pd.DataFrame(effective_close_series.rename('Close'))
            df_for_db.dropna(subset=['Close'], inplace=True)

            if not df_for_db.empty:
                save_actual_prices(ticker_key, df_for_db)
            else:
                print(f"SEEDER: No valid close prices for {ticker_key} to populate stock_prices.")
        except Exception as e:
            print(f"SEEDER: Error populating stock_prices for {ticker_key}: {e}")
            import traceback; traceback.print_exc()
    print("SEEDER: Finished populating 'stock_prices' table.")


def seed_historical_predictions_for_all(start_date_str: str, end_date_str: str,
                                        models_to_seed: List[str] = ["xgboost", "random_forest", "lstm"]):
    print(f"SEEDER: Starting historical prediction seeding from {start_date_str} to {end_date_str}...")
    init_db()

    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    for ticker_key, ticker_config in TICKERS_CONFIG.items():
        print(f"\nSEEDER: --- Seeding predictions for Ticker: {ticker_key} ---")

        # 1. Load full historical data for this ticker
        raw_file_path = get_raw_data_path(ticker_key)
        try:
            df_full_history_raw = pd.read_csv(raw_file_path, parse_dates=['Date'])
            df_full_history_raw.dropna(subset=['Date'], inplace=True)
            if df_full_history_raw.empty:
                print(f"SEEDER: Raw data for {ticker_key} is empty. Cannot seed.")
                continue
            df_full_history_raw.set_index('Date', inplace=True)
            df_full_history_raw.sort_index(inplace=True)

            # Prepare df_history_for_features:
            # It must contain TARGET_COLUMN (as effective close)
            # AND all columns listed in LSTM_INPUT_FEATURE_COLUMNS for LSTM.
            df_history_for_features = pd.DataFrame(index=df_full_history_raw.index)

            # Set TARGET_COLUMN (effective close)
            effective_close_col_name_in_raw = None
            if 'Adj_Close' in df_full_history_raw.columns and not df_full_history_raw['Adj_Close'].isnull().all():
                effective_close_col_name_in_raw = 'Adj_Close'
            elif 'Close' in df_full_history_raw.columns: # TARGET_COLUMN is 'Close'
                effective_close_col_name_in_raw = 'Close'
            
            if effective_close_col_name_in_raw:
                df_history_for_features[TARGET_COLUMN] = pd.to_numeric(df_full_history_raw[effective_close_col_name_in_raw], errors='coerce')
            else:
                print(f"SEEDER: No usable base price column for {TARGET_COLUMN} for {ticker_key}. Skipping ticker.")
                continue
            
            # Add/ensure other LSTM input features are present and numeric
            # LSTM_INPUT_FEATURE_COLUMNS is typically ['Open', 'High', 'Low', 'Close']
            missing_lstm_input_cols = []
            for col in LSTM_INPUT_FEATURE_COLUMNS:
                if col in df_full_history_raw.columns:
                    # If col is TARGET_COLUMN, it's already added and converted.
                    # Otherwise, add and convert.
                    if col not in df_history_for_features.columns: # Avoid overwriting TARGET_COLUMN if it's also an input feature
                         df_history_for_features[col] = pd.to_numeric(df_full_history_raw[col], errors='coerce')
                else:
                    print(f"SEEDER: WARNING - LSTM input feature '{col}' not found in raw data for {ticker_key}.")
                    missing_lstm_input_cols.append(col)
                    df_history_for_features[col] = np.nan # Add as NaN if missing

            if missing_lstm_input_cols:
                 print(f"SEEDER: Due to missing columns {missing_lstm_input_cols}, LSTM predictions might be affected or fail for {ticker_key}.")


            # Drop rows if any of the essential features for ANY model are NaN after conversion
            # For traditional models: TARGET_COLUMN must be valid
            # For LSTM: All columns in LSTM_INPUT_FEATURE_COLUMNS must be valid
            cols_to_check_for_na_in_history = list(set([TARGET_COLUMN] + LSTM_INPUT_FEATURE_COLUMNS))
            # Filter out columns not actually present in df_history_for_features before dropna
            cols_to_check_for_na_in_history = [c for c in cols_to_check_for_na_in_history if c in df_history_for_features.columns]

            df_history_for_features.dropna(subset=cols_to_check_for_na_in_history, how='any', inplace=True)

            if df_history_for_features.empty:
                print(f"SEEDER: df_history_for_features for {ticker_key} is empty after NA drop. Skipping ticker.")
                continue
        except Exception as e:
            print(f"SEEDER: Error loading/processing full history for {ticker_key}: {e}")
            import traceback; traceback.print_exc()
            continue

        # 2. Load models
        loaded_models = {}
        # LSTM scalers will be loaded inside prepare_input_sequence_for_lstm or make_prediction in model_utils
        for mt in models_to_seed:
            model = load_model(ticker_key, mt) # From model_utils
            if model:
                loaded_models[mt] = model
            else:
                print(f"SEEDER: Could not load model {mt} for {ticker_key}. It will be skipped.")

        # 3. Iterate through dates and make predictions
        current_pred_date = start_date
        while current_pred_date <= end_date:
            prediction_target_date_str = current_pred_date.strftime('%Y-%m-%d')
            historical_data_cutoff = current_pred_date - pd.Timedelta(days=1)
            
            data_for_current_features = df_history_for_features[df_history_for_features.index <= historical_data_cutoff].copy()

            for model_type, model_obj in loaded_models.items():
                prediction_input = None
                min_rows_needed = 0

                try:
                    if model_type in ["xgboost", "random_forest"]:
                        min_rows_needed = LAG_DAYS_TRADITIONAL + 1
                        if len(data_for_current_features) < min_rows_needed: continue
                        # Traditional models' feature prep uses only TARGET_COLUMN from input df
                        prediction_input = prepare_features_for_traditional_model(data_for_current_features[[TARGET_COLUMN]])
                    
                    elif model_type == "lstm":
                        min_rows_needed = LSTM_SEQUENCE_LENGTH
                        if len(data_for_current_features) < min_rows_needed: continue
                        
                        # Ensure all LSTM_INPUT_FEATURE_COLUMNS are present in data_for_current_features
                        if not all(col in data_for_current_features.columns for col in LSTM_INPUT_FEATURE_COLUMNS):
                            print(f"SEEDER: Skipping LSTM for {prediction_target_date_str} - missing required input columns in data_for_current_features.")
                            continue
                        # prepare_input_sequence_for_lstm expects a DataFrame with columns defined in LSTM_INPUT_FEATURE_COLUMNS
                        prediction_input = prepare_input_sequence_for_lstm(
                            data_for_current_features[LSTM_INPUT_FEATURE_COLUMNS], # Pass only the necessary features
                            ticker_key
                        )
                    
                    if prediction_input is None or \
                       (isinstance(prediction_input, pd.DataFrame) and prediction_input.empty) or \
                       (isinstance(prediction_input, pd.DataFrame) and prediction_input.isnull().values.any()):
                        continue
                    
                    predicted_price = make_prediction(model_obj, model_type, prediction_input, ticker_key=ticker_key)

                    if predicted_price is not None:
                        save_prediction(ticker_key, prediction_target_date_str, predicted_price, model_type)
                        if current_pred_date in df_history_for_features.index: # Check if actual exists for this date
                            actual_close = df_history_for_features.loc[current_pred_date, TARGET_COLUMN]
                            if pd.notna(actual_close):
                                update_actual_price_for_prediction(ticker_key, prediction_target_date_str, float(actual_close))
                except Exception as e_inner:
                    print(f"SEEDER: Error during prediction loop for {ticker_key}, {model_type}, {prediction_target_date_str}: {e_inner}")

            current_pred_date += pd.Timedelta(days=1)
        print(f"SEEDER: --- Finished seeding for Ticker: {ticker_key} ---")
    print("SEEDER: Historical prediction seeding finished.")


if __name__ == "__main__":
    print("===== Running Historical Seeder Script (app/historical_seeder.py) =====")
    
    # Optional: Set AM_I_IN_A_DOCKER_CONTAINER if running locally and config needs it for paths
    # if os.getenv("AM_I_IN_A_DOCKER_CONTAINER") is None:
    #     os.environ["AM_I_IN_A_DOCKER_CONTAINER"] = "false"

    print("\n--- Step 1: Populating 'stock_prices' table ---")
    populate_all_stock_prices_from_raw_csv()

    start_date_for_predictions = "2020-01-01"
    end_date_for_predictions = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"\n--- Step 2: Seeding historical predictions from {start_date_for_predictions} to {end_date_for_predictions} ---")
    seed_historical_predictions_for_all(
        start_date_str=start_date_for_predictions,
        end_date_str=end_date_for_predictions,
        models_to_seed=["lstm"]
    )
    print("\n===== Historical Seeder Script Finished =====")