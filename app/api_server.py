# app/api_server.py
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel # Keep this for request/response models
from typing import List, Dict, Optional, Any # Added Any
from datetime import datetime, timedelta
import pandas as pd
import os

from .config import (
    TICKERS_CONFIG, TARGET_COLUMN, LAG_DAYS_TRADITIONAL, LSTM_SEQUENCE_LENGTH
)
from .model_utils import (
    load_model,
    load_lstm_scaler, # Might not be directly needed by API if make_prediction handles it
    prepare_features_for_traditional_model,
    prepare_input_sequence_for_lstm,
    make_prediction
)
from .db_utils import (
    init_db,
    save_prediction, # Will need to save ticker and model_type
    get_prediction_history, # Will need to filter by ticker
    get_latest_close_prices, # Will need to fetch for a specific ticker
    update_actual_price_for_prediction
)
# For data update trigger (optional)
# from .data_ingestion import run_ingestion_for_all_tickers
# from .data_processing import run_processing_for_all_tickers # If you want to trigger processing

app = FastAPI(title="Stock & Index Price Prediction API v3")

# --- Global Cache for Models (managed by model_utils.load_model) ---
# We use Depends(get_model_dependency) to load/cache models per request or worker

# --- Dependency for loading models ---
async def get_model_dependency(
    ticker_key: str = Query(..., enum=list(TICKERS_CONFIG.keys()), description="Ticker symbol key (e.g., GSPC, IBM)"),
    model_type: str = Query("xgboost", enum=["xgboost", "random_forest", "lstm"], description="Model type for prediction")
) -> Any:
    """FastAPI dependency to load and cache the specified model."""
    model = load_model(ticker_key, model_type)
    if model is None:
        print(f"API_SERVER: Failed to load model {model_type} for {ticker_key} via dependency.")
        raise HTTPException(status_code=503, detail=f"Model {model_type} for {ticker_key} is not available or failed to load.")
    return model

@app.on_event("startup")
async def startup_event():
    print("API_SERVER: Application startup...")
    init_db() # Initialize database tables
    # Pre-load models (optional, can improve first-request latency)
    # print("API_SERVER: Pre-loading models...")
    # for tk_key in TICKERS_CONFIG.keys():
    #     for mt in ["xgboost", "random_forest", "lstm"]:
    #         try:
    #             load_model(tk_key, mt)
    #         except Exception as e:
    #             print(f"API_SERVER: Warning - could not pre-load model {mt} for {tk_key}: {e}")
    print("API_SERVER: Startup complete.")

# --- Pydantic Models for API Request/Response ---
class PredictionRequest(BaseModel):
    ticker_key: str # e.g., "GSPC", "IBM"
    model_type: str # e.g., "xgboost", "lstm"

class PredictionResponseAPI(BaseModel): # Changed name to avoid conflict
    ticker_key: str
    display_name: str
    prediction_date: str
    predicted_price: float
    model_used: str
    message: Optional[str] = None

class HistoryDataPointAPI(BaseModel):
    prediction_date: str
    predicted_price: Optional[float] = None
    actual_price: Optional[float] = None
    model_used: Optional[str] = None # Which model made this prediction
    # ticker_key: str # Optional: if history combines multiple tickers

class HistoryResponseAPI(BaseModel):
    ticker_key: str
    display_name: str
    data: List[HistoryDataPointAPI]

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Stock & Index Price Prediction API v3!"}

@app.post("/predict", response_model=PredictionResponseAPI)
async def predict_next_trading_day_endpoint(
    ticker_key: str = Query(..., enum=list(TICKERS_CONFIG.keys()), description="Ticker symbol key (e.g., GSPC, IBM)"),
    model_type: str = Query("xgboost", enum=["xgboost", "random_forest", "lstm"], description="Model type for prediction"),
    model: Any = Depends(get_model_dependency) # Model is loaded based on ticker_key and model_type
):
    """
    Predicts the next trading day's price for the given ticker using the specified model.
    """
    print(f"API_SERVER: Received prediction request for {ticker_key} using {model_type} model.")
    if model is None: # Should be caught by Depends, but double-check
        raise HTTPException(status_code=503, detail=f"Model {model_type} for {ticker_key} is not available.")

    try:
        # 1. Get latest historical prices for the ticker from DB
        # Determine how many days of data are needed based on the model type
        days_needed_for_features = 0
        if model_type in ["xgboost", "random_forest"]:
            days_needed_for_features = LAG_DAYS_TRADITIONAL + 1 # +1 for current day to calc lags for next
        elif model_type == "lstm":
            days_needed_for_features = LSTM_SEQUENCE_LENGTH
        else: # Should not happen due to Query enum
            raise HTTPException(status_code=400, detail=f"Invalid model_type: {model_type}")

        # Fetch a bit more data just in case of non-trading days or missing data points
        historical_prices_df = get_latest_close_prices(ticker_key, days=days_needed_for_features + 15)

        if historical_prices_df.empty or len(historical_prices_df) < days_needed_for_features:
            print(f"API_SERVER: Not enough data for {ticker_key}. Need {days_needed_for_features}, Got {len(historical_prices_df)}")
            print(f"DEBUG DB Data for {ticker_key}:\n{historical_prices_df.to_string() if not historical_prices_df.empty else 'Empty DataFrame'}")
            raise HTTPException(status_code=404,
                                 detail=f"Not enough historical data in DB for {ticker_key} (need at least {days_needed_for_features} "
                                        f"days, got {len(historical_prices_df)}) to make prediction with {model_type}.")
        
        # Rename 'close_price' from DB to TARGET_COLUMN ('Close') for feature prep functions
        feature_base_df = historical_prices_df.rename(columns={'close_price': TARGET_COLUMN})

        # 2. Prepare features based on model type
        prediction_input_features = None
        if model_type in ["xgboost", "random_forest"]:
            prediction_input_features = prepare_features_for_traditional_model(feature_base_df)
        elif model_type == "lstm":
            prediction_input_features = prepare_input_sequence_for_lstm(feature_base_df, ticker_key)
        
        if prediction_input_features is None or \
           (isinstance(prediction_input_features, pd.DataFrame) and prediction_input_features.empty) or \
           (isinstance(prediction_input_features, pd.DataFrame) and prediction_input_features.isnull().values.any()): # Check for NaNs too
            
            nan_info = ""
            if isinstance(prediction_input_features, pd.DataFrame) and not prediction_input_features.empty:
                nan_cols = prediction_input_features.columns[prediction_input_features.isnull().any()].tolist()
                if nan_cols: nan_info = f" Features with NaN: {nan_cols}."
            
            print(f"API_SERVER: Failed to create valid features for {ticker_key} with {model_type}.{nan_info}")
            print(f"DEBUG Feature Input:\n{prediction_input_features.to_string() if isinstance(prediction_input_features, pd.DataFrame) else 'Not a DataFrame'}")
            raise HTTPException(status_code=500, detail=f"Could not create valid input features for {model_type} on {ticker_key}.{nan_info}")

        # 3. Make prediction
        predicted_price = make_prediction(model, model_type, prediction_input_features, ticker_key=ticker_key)

        if predicted_price is None: # make_prediction might return None on error
            raise HTTPException(status_code=500, detail=f"Prediction failed for {ticker_key} with {model_type}.")

        # 4. Determine prediction target date (next trading day)
        last_known_date = historical_prices_df.index.max()
        prediction_target_dt = last_known_date
        days_to_add = 1
        while True: # Simple way to find next weekday
            prediction_target_dt = last_known_date + pd.Timedelta(days=days_to_add)
            if prediction_target_dt.weekday() < 5: # Monday=0, ..., Friday=4
                break
            days_to_add += 1
        prediction_date_str = prediction_target_dt.strftime("%Y-%m-%d")

        # 5. Save prediction to DB
        save_prediction(ticker_key, prediction_date_str, predicted_price, model_type)

        return PredictionResponseAPI(
            ticker_key=ticker_key,
            display_name=TICKERS_CONFIG[ticker_key]["display_name"],
            prediction_date=prediction_date_str,
            predicted_price=predicted_price,
            model_used=model_type,
            message=f"Prediction by {model_type} for {TICKERS_CONFIG[ticker_key]['display_name']}."
        )
    except HTTPException as http_exc:
        print(f"API_SERVER: HTTPException for {ticker_key} ({model_type}): {http_exc.detail}")
        raise http_exc
    except ValueError as ve:
        print(f"API_SERVER: ValueError for {ticker_key} ({model_type}): {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"API_SERVER: Unexpected error for {ticker_key} ({model_type}): {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/history", response_model=HistoryResponseAPI)
async def get_history_endpoint(
    ticker_key: str = Query(..., enum=list(TICKERS_CONFIG.keys()), description="Ticker to fetch history for"),
    days_limit: int = Query(90, ge=1, le=730, description="Number of past days of predictions to retrieve")
):
    """Retrieves prediction history for a specific ticker."""
    print(f"API_SERVER: Received history request for {ticker_key}, limit {days_limit} days.")
    try:
        history_df = get_prediction_history(ticker_key, limit=days_limit) # db_utils needs to accept ticker_key
        
        response_data_points = []
        if not history_df.empty:
            for _, row in history_df.iterrows():
                # Combine actual_price (from predictions table) and historical_actual (from stock_prices table)
                final_actual_price = row['actual_price'] if pd.notna(row['actual_price']) else row['historical_actual']
                
                response_data_points.append(HistoryDataPointAPI(
                    prediction_date=row['prediction_date'].strftime('%Y-%m-%d'),
                    predicted_price=float(row['predicted_price']) if pd.notna(row['predicted_price']) else None,
                    model_used=row['model_used'],
                    actual_price=float(final_actual_price) if pd.notna(final_actual_price) else None
                ))
        
        return HistoryResponseAPI(
            ticker_key=ticker_key,
            display_name=TICKERS_CONFIG[ticker_key]["display_name"],
            data=response_data_points
        )
    except Exception as e:
        print(f"API_SERVER: Error fetching prediction history for {ticker_key}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching prediction history for {ticker_key}.")

# Placeholder for data update trigger if needed by scheduler/admin
# @app.post("/admin/trigger-data-update-all")
# async def trigger_full_data_pipeline():
#     print("API_SERVER: Received request to trigger full data pipeline.")
#     try:
#         # This should ideally be an async task or background job
#         run_ingestion_for_all_tickers() # From data_ingestion.py
#         run_processing_for_all_tickers() # From data_processing.py (updates processed CSVs)
        
#         # After processing, worker's daily_data_update_job should pick up new CSV data to update DB.
#         # Or, add logic here to directly update stock_prices table from new processed CSVs.
        
#         return {"message": "Full data ingestion and processing pipeline triggered."}
#     except Exception as e:
#         print(f"API_SERVER: Error triggering full data pipeline: {e}")
#         raise HTTPException(status_code=500, detail=str(e))