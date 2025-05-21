# app/data_processing.py
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

from .config import (
    TICKERS_CONFIG, get_raw_data_path, get_processed_data_path,
    TARGET_COLUMN, LAG_DAYS_TRADITIONAL, EXPECTED_FEATURES_TRADITIONAL,
    LSTM_SEQUENCE_LENGTH, LSTM_INPUT_FEATURE_COLUMNS, get_lstm_scaler_path # Use CONTAINER path for saving scalers
)

def create_traditional_features(df_input: pd.DataFrame, target_col: str,
                                lag_days: int, expected_features_list: List[str]) -> pd.DataFrame:
    """
    Creates lag and percentage change features for traditional models (XGBoost, RandomForest).
    df_input: DataFrame with 'Date' as index and sorted, must contain target_col.
    target_col: The name of the column to use for creating lags and as the base for pct_change.
    lag_days: Number of lag features to create.
    expected_features_list: List of feature names that the model expects.
    Returns a DataFrame with features and 'model_target'.
    """
    if target_col not in df_input.columns:
        print(f"ERROR: Target column '{target_col}' not in DataFrame for traditional features.")
        return pd.DataFrame()

    df = df_input.copy() # Work on a copy
    df_features = df[[target_col]].copy() # Start with the target column

    # 1. Create Lag Features
    for i in range(1, lag_days + 1):
        feature_name = f'{target_col}_lag_{i}'
        df_features[feature_name] = df_features[target_col].shift(i)

    # 2. Create Percentage Change Feature
    # (Today's target_col - Yesterday's target_col) / Yesterday's target_col
    # .shift(1) ensures this feature is based on past information for the current day's prediction target.
    pct_change_feature_name = f'{target_col}_pct_change_1d'
    df_features[pct_change_feature_name] = df_features[target_col].pct_change(periods=1).shift(1)

    # 3. Create 'model_target' (next day's target_col value)
    df_features['model_target'] = df_features[target_col].shift(-1)
    
    # 4. Drop rows with NaNs created by shifting (for lags, pct_change, and model_target)
    df_features.dropna(inplace=True)

    # 5. Select only the expected features and the original target column (for reference) and model_target
    # The original TARGET_COLUMN is kept for reference or potential use in analysis/seeding actuals.
    final_columns_to_keep = [target_col] + expected_features_list + ['model_target']
    
    # Ensure all expected columns exist in df_features before selecting
    actual_present_columns = [col for col in final_columns_to_keep if col in df_features.columns]
    if len(actual_present_columns) != len(final_columns_to_keep):
        missing = set(final_columns_to_keep) - set(actual_present_columns)
        print(f"WARNING: Some expected columns were not generated or dropped: {missing}. This might be due to insufficient data for lags.")

    df_processed = df_features[actual_present_columns].copy()
    
    return df_processed
##########################
###########################
###########################
###########################
#############################
############################
##########################
# def create_and_scale_knn_features(df_input: pd.DataFrame, target_col: str,

def prepare_and_scale_for_lstm(df_input: pd.DataFrame, ticker_key: str,
                               feature_cols: List[str], target_col_for_lstm: str) -> tuple[pd.DataFrame, MinMaxScaler, MinMaxScaler]:
    """
    Prepares and scales data for LSTM model training and saves the scalers.
    df_input: DataFrame with 'Date' as index and sorted, containing feature_cols and target_col_for_lstm.
    ticker_key: String key for the ticker (e.g., "GSPC").
    feature_cols: List of column names to be used as input features for LSTM.
    target_col_for_lstm: The single column name that will be predicted by LSTM (usually TARGET_COLUMN).
    Returns:
        - scaled_df: DataFrame containing scaled input features and the scaled target for LSTM.
        - feature_scaler: The fitted scaler for input features.
        - target_scaler: The fitted scaler for the target.
    Returns (None, None, None) on failure.
    """
    print(f"Preparing and scaling LSTM data for {ticker_key}...")
    df = df_input.copy()

    # Validate required columns
    for col in feature_cols + [target_col_for_lstm]:
        if col not in df.columns:
            print(f"ERROR: Required column '{col}' not found in DataFrame for LSTM processing of {ticker_key}.")
            return None, None, None
    
    # 1. Select and scale input features for LSTM
    features_to_scale_df = df[feature_cols].copy()
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_feature_values = feature_scaler.fit_transform(features_to_scale_df)
    
    # 2. Select and scale the target column for LSTM
    target_to_scale_series = df[[target_col_for_lstm]].copy() # Ensure it's a DataFrame for scaler
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target_values = target_scaler.fit_transform(target_to_scale_series)

    # 3. Save scalers
    # Ensure LSTM_SCALERS_DIR_IN_CONTAINER exists (config helper function now handles this)
    # os.makedirs(LSTM_SCALERS_DIR_IN_CONTAINER, exist_ok=True) # Not needed if get_lstm_scaler_path handles it

    feature_scaler_path = get_lstm_scaler_path(ticker_key, "feature")
    target_scaler_path = get_lstm_scaler_path(ticker_key, "target")
    os.makedirs(os.path.dirname(feature_scaler_path), exist_ok=True) # Ensure directory for scaler exists

    with open(feature_scaler_path, 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    print(f"LSTM scalers for {ticker_key} saved to {os.path.dirname(feature_scaler_path)}.")

    # 4. Create a new DataFrame with scaled values
    scaled_df = pd.DataFrame(scaled_feature_values, columns=feature_cols, index=df.index)
    # The target for LSTM training will be the *next day's* scaled target_col_for_lstm.
    # We add it here for convenience, to be used when creating X,y sequences in model_training.
    scaled_df[f'scaled_{target_col_for_lstm}_target'] = np.roll(scaled_target_values.flatten(), -1)
    # Remove the last row because its target is from the "future" (rolled to the beginning)
    scaled_df = scaled_df.iloc[:-1]

    return scaled_df, feature_scaler, target_scaler


def run_processing_for_all_tickers():
    """Runs data processing for all tickers defined in TICKERS_CONFIG."""
    for ticker_key, config_details in TICKERS_CONFIG.items():
        print(f"\n--- Processing data for {ticker_key} ({config_details['api_ticker']}) ---")
        
        raw_file_path = get_raw_data_path(ticker_key)
        try:
            # Read raw data, ensure 'Date' is parsed and set as index for processing
            df_raw = pd.read_csv(raw_file_path, parse_dates=['Date'])
            df_raw.set_index('Date', inplace=True)
            df_raw.sort_index(inplace=True) # Crucial for time series features
        except FileNotFoundError:
            print(f"ERROR: Raw data file not found for {ticker_key} at {raw_file_path}. Run ingestion first.")
            continue
        except Exception as e:
            print(f"ERROR: Could not read raw data for {ticker_key}: {e}")
            continue

        if df_raw.empty:
            print(f"Raw data for {ticker_key} is empty. Skipping processing.")
            continue

        # --- Decide which column to use as the primary price (TARGET_COLUMN) ---
        # Prefer 'Adj_Close', fallback to 'Close' if 'Adj_Close' is not available or all NaN
        # Our TARGET_COLUMN in config is 'Close'. We will effectively make this 'Close'
        # column represent the adjusted price if 'Adj_Close' is usable.
        
        df_to_process = df_raw.copy() # Work on a copy

        if 'Adj_Close' in df_to_process.columns and not df_to_process['Adj_Close'].isnull().all():
            print(f"Using 'Adj_Close' as the base for '{TARGET_COLUMN}' for {ticker_key}.")
            df_to_process[TARGET_COLUMN] = df_to_process['Adj_Close'] # TARGET_COLUMN is 'Close'
        elif 'Close' in df_to_process.columns:
            print(f"Using raw 'Close' as the base for '{TARGET_COLUMN}' for {ticker_key} ('Adj_Close' not available/usable).")
            # TARGET_COLUMN is already 'Close', so no change needed if it exists and Adj_Close doesn't
        else:
            print(f"CRITICAL: Neither 'Adj_Close' nor 'Close' column is suitable for {ticker_key}. Skipping.")
            continue
        
        # Ensure TARGET_COLUMN ('Close') is numeric and drop NaNs in it
        df_to_process[TARGET_COLUMN] = pd.to_numeric(df_to_process[TARGET_COLUMN], errors='coerce')
        df_to_process.dropna(subset=[TARGET_COLUMN], inplace=True)

        if df_to_process.empty:
            print(f"Data for {ticker_key} became empty after ensuring numeric '{TARGET_COLUMN}'. Skipping.")
            continue

        # 1. Process for Traditional Models (XGBoost, RandomForest)
        print(f"Processing traditional features for {ticker_key}...")
        df_traditional_processed = create_traditional_features(
            df_to_process, # Pass DataFrame with 'Date' as index
            target_col=TARGET_COLUMN,
            lag_days=LAG_DAYS_TRADITIONAL,
            expected_features_list=EXPECTED_FEATURES_TRADITIONAL
        )
        if not df_traditional_processed.empty:
            processed_file_path = get_processed_data_path(ticker_key)
            # os.makedirs(os.path.dirname(processed_file_path), exist_ok=True) # get_processed_data_path handles this
            df_traditional_processed.to_csv(processed_file_path, index=True) # Save with Date index
            print(f"Processed data for traditional models ({ticker_key}) saved to: {processed_file_path}")
        else:
            print(f"Failed to process data for traditional models for {ticker_key}.")

        # 2. Process for LSTM Model (Scaling and saving scalers)
        print(f"Preparing and scaling data for LSTM for {ticker_key}...")
        # For LSTM, we use the df_to_process which has the 'Date' index and a clean TARGET_COLUMN.
        # LSTM_INPUT_FEATURE_COLUMNS in config defines what features to scale for LSTM input.
        scaled_lstm_df, feature_scaler, target_scaler = prepare_and_scale_for_lstm(
            df_to_process, # DataFrame with 'Date' index
            ticker_key,
            feature_cols=LSTM_INPUT_FEATURE_COLUMNS,
            target_col_for_lstm=TARGET_COLUMN # LSTM will predict TARGET_COLUMN
        )
        if scaled_lstm_df is not None and not scaled_lstm_df.empty:
            # The scaled_lstm_df itself isn't typically saved as a primary "processed data" file.
            # Instead, the model_training.py script will call prepare_and_scale_for_lstm again
            # to get the scaled data and use the saved scalers for consistency if loading.
            # Or, you could save scaled_lstm_df if you have a specific workflow for it.
            print(f"LSTM data preparation (scaling and scaler saving) done for {ticker_key}.")
            # Example: scaled_lstm_df.to_csv(get_processed_data_path(ticker_key) + "_lstm_scaled.csv")
        else:
            print(f"Failed to prepare or scale data for LSTM for {ticker_key}.")

if __name__ == "__main__":
    run_processing_for_all_tickers()