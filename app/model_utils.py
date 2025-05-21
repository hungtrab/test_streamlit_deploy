# app/model_utils.py
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import torch
import torch.nn as nn # For LSTMRegressor class definition
import os
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Union, Any # For type hinting

from .config import (
    TICKERS_CONFIG, EXPECTED_FEATURES_TRADITIONAL, TARGET_COLUMN,
    LAG_DAYS_TRADITIONAL, # For traditional models
    LSTM_SEQUENCE_LENGTH, LSTM_NUM_FEATURES, LSTM_INPUT_FEATURE_COLUMNS, # For new LSTM
    get_model_path, get_lstm_scaler_path
)

# --- LSTM Model Class Definition (consistent with your training code) ---
class LSTMRegressor(nn.Module):
  def __init__(self, input_size, hidden_size, output_size=1): # output_size default is 1
    super(LSTMRegressor, self).__init__()
    self.hidden_size = hidden_size
    # Assuming 1 LSTM layer as per your training code structure
    # If your training code used more layers or dropout, adjust here.
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    # x shape: (batch_size, seq_length, input_size)
    # Initialize hidden state and cell state
    h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_size, device=x.device).requires_grad_()
    c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_size, device=x.device).requires_grad_()

    out, _ = self.lstm(x, (h0.detach(), c0.detach())) # out shape: (batch_size, seq_length, hidden_size)
    out = self.fc(out[:, -1, :]) # Get output of the last time step: (batch_size, output_size)
    return out

# --- Global Cache for Loaded Models and Scalers ---
_loaded_models_cache: Dict[str, Any] = {}
_loaded_scalers_cache: Dict[str, MinMaxScaler] = {} # Explicitly MinMaxScaler

def _get_cache_key(ticker_key: str, object_type: str, sub_type: str = None) -> str:
    """Helper to create consistent cache keys."""
    key = f"ticker_{ticker_key.upper()}_{object_type}"
    if sub_type:
        key += f"_{sub_type}"
    return key

# --- Model and Scaler Loading ---
def load_model(ticker_key: str, model_type: str) -> Any:
    """Loads a specific model for a given ticker and caches it."""
    cache_key = _get_cache_key(ticker_key, model_type)
    if cache_key in _loaded_models_cache:
        # print(f"MODEL_UTILS: Returning cached model: {cache_key}")
        return _loaded_models_cache[cache_key]

    model_path = get_model_path(ticker_key, model_type)
    model = None
    print(f"MODEL_UTILS: Loading model {model_type} for {ticker_key} from {model_path}...")

    if not os.path.exists(model_path):
        print(f"ERROR (model_utils): Model file not found at {model_path}")
        _loaded_models_cache[cache_key] = None
        return None

    try:
        if model_type == "xgboost":
            model = xgb.XGBRegressor()
            model.load_model(model_path)
        elif model_type == "random_forest":
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_type == "lstm":
            # Instantiate the model first, then load state_dict
            # LSTM_NUM_FEATURES is the input_size for the LSTMModel
            model = LSTMRegressor(input_size=LSTM_NUM_FEATURES, hidden_size=200) # hidden_size should match training
            # Load onto CPU by default for inference, adjust if GPU is available and intended
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval() # Set to evaluation mode
        else:
            raise ValueError(f"Unsupported model_type: {model_type} for ticker {ticker_key}")
        
        _loaded_models_cache[cache_key] = model
        print(f"MODEL_UTILS: Successfully loaded model: {cache_key}")
        return model
    except Exception as e:
        print(f"ERROR (model_utils) loading model {cache_key}: {e}")
        import traceback
        traceback.print_exc()
        _loaded_models_cache[cache_key] = None
        return None

def load_lstm_scaler(ticker_key: str, scaler_name: str) -> MinMaxScaler | None:
    """
    Loads a specific LSTM scaler (e.g., "features_X_scaler", "target_y_scaler") and caches it.
    """
    cache_key = _get_cache_key(ticker_key, "lstm_scaler", scaler_name) # Use scaler_name as sub_type
    if cache_key in _loaded_scalers_cache:
        return _loaded_scalers_cache[cache_key]

    scaler_path = get_lstm_scaler_path(ticker_key, scaler_name) # From config
    scaler = None
    print(f"MODEL_UTILS: Loading LSTM scaler '{scaler_name}' for {ticker_key} from {scaler_path}...")

    if not os.path.exists(scaler_path):
        print(f"ERROR (model_utils): LSTM Scaler file '{scaler_name}' not found at {scaler_path}")
        _loaded_scalers_cache[cache_key] = None
        return None
    
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        _loaded_scalers_cache[cache_key] = scaler
        print(f"MODEL_UTILS: Successfully loaded LSTM scaler: {cache_key}")
        return scaler
    except Exception as e:
        print(f"ERROR (model_utils) loading LSTM scaler {cache_key}: {e}")
        _loaded_scalers_cache[cache_key] = None
        return None

# --- Feature Preparation for Prediction ---
def prepare_features_for_traditional_model(historical_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features for XGBoost/RandomForest prediction from the latest historical data.
    historical_data_df: DataFrame with 'Date' as index, sorted, containing TARGET_COLUMN.
                        Needs at least LAG_DAYS_TRADITIONAL + 1 rows.
    Returns a single-row DataFrame with features.
    """
    if TARGET_COLUMN not in historical_data_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not in historical_data_df for traditional features.")
    
    required_rows = LAG_DAYS_TRADITIONAL + 1
    if len(historical_data_df) < required_rows:
        print(f"Warning (model_utils): Not enough historical data (need {required_rows}, got {len(historical_data_df)}) "
              "to create all traditional features.")
        return pd.DataFrame(columns=EXPECTED_FEATURES_TRADITIONAL)

    features = {}
    latest_data_series = historical_data_df[TARGET_COLUMN].sort_index(ascending=True).tail(required_rows)

    for i in range(1, LAG_DAYS_TRADITIONAL + 1):
        feature_name = f'{TARGET_COLUMN}_lag_{i}'
        if feature_name in EXPECTED_FEATURES_TRADITIONAL:
            features[feature_name] = latest_data_series.iloc[-(i + 1)]

    pct_change_feature_name = f'{TARGET_COLUMN}_pct_change_1d'
    if pct_change_feature_name in EXPECTED_FEATURES_TRADITIONAL:
        if len(latest_data_series) >= 2:
            current_val = latest_data_series.iloc[-1]
            previous_val = latest_data_series.iloc[-2]
            features[pct_change_feature_name] = (current_val - previous_val) / previous_val if previous_val != 0 else 0.0
        else:
            features[pct_change_feature_name] = 0.0

    for feat in EXPECTED_FEATURES_TRADITIONAL:
        if feat not in features: # Ensure all expected features are present
            features[feat] = np.nan
    feature_df = pd.DataFrame([features], columns=EXPECTED_FEATURES_TRADITIONAL)
    return feature_df


def prepare_input_sequence_for_lstm(historical_data_df: pd.DataFrame, ticker_key: str) -> torch.Tensor | None:
    """
    Prepares the most recent sequence of OHL C data for LSTM prediction, scaling it.
    historical_data_df: DataFrame with 'Date' as index, sorted, containing LSTM_OHLC_FEATURE_COLUMNS.
                        Needs at least LSTM_SEQUENCE_LENGTH rows.
    ticker_key: To load the correct 'features_X_scaler'.
    Returns a PyTorch tensor for LSTM input or None on failure.
    """
    missing_cols = [col for col in LSTM_INPUT_FEATURE_COLUMNS if col not in historical_data_df.columns]
    if missing_cols:
        print(f"ERROR (model_utils): Missing LSTM OHLC feature columns in historical_data_df: {missing_cols}")
        return None

    if len(historical_data_df) < LSTM_SEQUENCE_LENGTH:
        print(f"Warning (model_utils): Not enough historical data (need {LSTM_SEQUENCE_LENGTH}, got {len(historical_data_df)}) "
              f"to create LSTM input sequence for {ticker_key}.")
        return None

    # Load the scaler used for X features during training
    features_X_scaler = load_lstm_scaler(ticker_key, "features_X_scaler")
    if features_X_scaler is None:
        print(f"ERROR (model_utils): LSTM 'features_X_scaler' for {ticker_key} not loaded. Cannot prepare sequence.")
        return None

    # Select the last LSTM_SEQUENCE_LENGTH rows and ONLY the LSTM_OHLC_FEATURE_COLUMNS
    # Data should already be sorted by date before calling this function
    sequence_data_df = historical_data_df[LSTM_INPUT_FEATURE_COLUMNS].tail(LSTM_SEQUENCE_LENGTH)
    
    # Ensure columns are in the same order as during scaler fitting (if scaler has feature_names_in_)
    if hasattr(features_X_scaler, 'feature_names_in_') and list(sequence_data_df.columns) != list(features_X_scaler.feature_names_in_):
        print(f"Warning (model_utils): Column order mismatch for LSTM scaling. Reordering. "
              f"Data has: {list(sequence_data_df.columns)}, Scaler expects: {list(features_X_scaler.feature_names_in_)}")
        try:
            sequence_data_df = sequence_data_df[features_X_scaler.feature_names_in_]
        except KeyError:
            print(f"ERROR (model_utils): Cannot reorder columns for LSTM scaling due to missing columns.")
            return None
    
    try:
        # The input to transform should be a 2D array [n_samples, n_features]
        # Here, n_samples is LSTM_SEQUENCE_LENGTH, n_features is LSTM_NUM_FEATURES
        scaled_sequence_np = features_X_scaler.transform(sequence_data_df.values) # Pass NumPy array
    except Exception as e:
        print(f"ERROR (model_utils): Failed to transform data with LSTM 'features_X_scaler' for {ticker_key}: {e}")
        return None
        
    # Reshape for LSTM: (batch_size=1, sequence_length, num_features)
    input_tensor = torch.from_numpy(scaled_sequence_np).float().unsqueeze(0)
    return input_tensor

# --- Prediction Function ---
def make_prediction(model: any, model_type: str, feature_input: any, ticker_key: str = None) -> float | None:
    """Makes a prediction using the loaded model."""
    if model is None:
        print(f"ERROR (model_utils): Model object is None for {model_type} ({ticker_key}), cannot make prediction.")
        return None

    try:
        if model_type in ["xgboost", "random_forest"]:
            if not isinstance(feature_input, pd.DataFrame):
                raise ValueError("Feature input for traditional models must be a Pandas DataFrame.")
            # Ensure columns are in the correct order
            try:
                feature_input_ordered = feature_input[EXPECTED_FEATURES_TRADITIONAL]
            except KeyError:
                raise ValueError(f"Feature input columns mismatch. Expected {EXPECTED_FEATURES_TRADITIONAL}, got {list(feature_input.columns)}")

            if feature_input_ordered.isnull().values.any():
                nan_cols = feature_input_ordered.columns[feature_input_ordered.isnull().any()].tolist()
                print(f"WARNING (model_utils make_prediction): Traditional feature input for {ticker_key} ({model_type}) contains NaNs in {nan_cols}.")
                # XGBoost might handle NaNs by default if not explicitly configured otherwise during training.
                # RandomForest typically does not handle NaNs.
                if model_type == "random_forest":
                     raise ValueError("Random Forest cannot handle NaN features in prediction input. Please clean data.")
            
            prediction_value = model.predict(feature_input_ordered)[0]
        
        elif model_type == "lstm":
            if not isinstance(feature_input, torch.Tensor): # feature_input is the scaled tensor
                raise ValueError("Feature input for LSTM model must be a PyTorch Tensor.")
            if ticker_key is None:
                raise ValueError("ticker_key is required for LSTM prediction to load target scaler.")
                
            target_y_scaler = load_lstm_scaler(ticker_key, "target_y_scaler") # Load the 'y' scaler
            if target_y_scaler is None:
                raise ValueError(f"LSTM 'target_y_scaler' for {ticker_key} not loaded. Cannot inverse transform prediction.")

            model.eval() # Ensure LSTM model is in evaluation mode
            with torch.no_grad():
                predicted_scaled_tensor = model(feature_input) # Input tensor is already scaled
            
            predicted_scaled_np = predicted_scaled_tensor.cpu().numpy()
            # Inverse transform the prediction using the scaler for the target variable ('Close')
            prediction_value = target_y_scaler.inverse_transform(predicted_scaled_np)[0,0]
        else:
            raise ValueError(f"Unsupported model_type for prediction: {model_type}")

        return float(prediction_value)
    except Exception as e:
        print(f"ERROR (model_utils) during make_prediction for {model_type} ({ticker_key}): {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("--- Model Utils Direct Test (Loading Models and Scalers) ---")
    # This test assumes that training has been run and models/scalers exist.
    # Set AM_I_IN_A_DOCKER_CONTAINER for local testing if your config depends on it for paths
    # os.environ["AM_I_IN_A_DOCKER_CONTAINER"] = "false" # Example for local path resolution

    for tk_key in TICKERS_CONFIG.keys():
        print(f"\n--- Testing for Ticker: {tk_key} ---")
        for mt in ["xgboost", "random_forest", "lstm"]:
            model = load_model(tk_key, mt)
            if model: print(f"Successfully loaded {mt} model for {tk_key}.")
            else: print(f"FAILED to load {mt} model for {tk_key}.")
        
        if load_lstm_scaler(tk_key, "features_X_scaler") and load_lstm_scaler(tk_key, "target_y_scaler"):
            print(f"Successfully loaded LSTM scalers (features_X_scaler, target_y_scaler) for {tk_key}.")
        else:
            print(f"FAILED to load one or more LSTM scalers for {tk_key}.")
            
    # Further testing of prepare_features_* and make_prediction would require mock data or
    # running after data_ingestion and db_utils are fully functional to pull data.
    print("--- End Model Utils Direct Test ---")