# app/data_ingestion.py
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import time
import os
from datetime import datetime
from .config import ALPHA_VANTAGE_API_KEY, TICKERS_CONFIG, get_raw_data_path

# Start date for fetching data. yfinance can often go very far back.
# Alpha Vantage free 'full' is ~20 years.
FETCH_START_DATE = "2000-01-01" # Fetch all available data starting from this date

def fetch_yfinance_data(api_ticker: str, start_date: str = FETCH_START_DATE) -> pd.DataFrame:
    """Fetches all available historical data using yfinance from start_date to the current date."""
    print(f"Fetching data for {api_ticker} using yfinance (from {start_date})...")
    try:
        data = yf.download(api_ticker, start=start_date, progress=False, auto_adjust=False, actions=False)
        # auto_adjust=False to get raw Open, High, Low, Close, and a separate Adj Close
        # actions=False to not download dividends and splits columns, we might not need them for now.
        if data.empty:
            print(f"No data returned for {api_ticker} from yfinance for period starting {start_date}.")
            return pd.DataFrame()
        
        data.reset_index(inplace=True)
        
        # yfinance column names are usually 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
        # We will prioritize 'Close' for our TARGET_COLUMN as per config.
        # If 'Adj Close' exists, we'll keep it as 'Adj_Close' for potential later use or comparison.
        # If it doesn't, we might create one from 'Close' or just proceed without it.
        
        column_rename_map = {}
        if 'Adj Close' in data.columns: # yfinance specific column name
            column_rename_map['Adj Close'] = 'Adj_Close'
        
        if column_rename_map:
            data.rename(columns=column_rename_map, inplace=True)

        # Standard set of columns we are interested in
        desired_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if 'Adj_Close' in data.columns: # Add Adj_Close if it was successfully fetched/renamed
            desired_cols.insert(5, 'Adj_Close')

        # Filter to keep only desired columns that actually exist
        data = data[[col for col in desired_cols if col in data.columns]].copy()
        
        data['Date'] = pd.to_datetime(data['Date'])
        data.sort_values(by='Date', ascending=True, inplace=True) # Ensure chronological order
        
        print(f"Successfully fetched {len(data)} rows for {api_ticker} from yfinance.")
        return data
    except Exception as e:
        print(f"Error fetching data for {api_ticker} from yfinance: {e}")
        return pd.DataFrame()

def fetch_alpha_vantage_data(api_ticker: str, start_date: str = FETCH_START_DATE,
                               max_retries: int = 3, initial_delay: int = 65) -> pd.DataFrame:
    """
    Fetches historical data using Alpha Vantage (TIME_SERIES_DAILY for free tier).
    Filters data from the specified start_date.
    """
    print(f"Fetching data for {api_ticker} using Alpha Vantage (TIME_SERIES_DAILY, from {start_date})...")
    if not ALPHA_VANTAGE_API_KEY:
        print("ERROR: Alpha Vantage API key not configured.")
        return pd.DataFrame()

    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    attempt = 0
    current_delay = initial_delay
    while attempt < max_retries:
        try:
            print(f"Attempt {attempt + 1}/{max_retries} for {api_ticker} (AV TIME_SERIES_DAILY)...")
            # Use get_daily for free tier (no direct adjusted close, but gives OHL C V)
            data, meta_data = ts.get_daily(symbol=api_ticker, outputsize="full")
            
            if data.empty:
                print(f"No data for {api_ticker} (AV). Retrying in {current_delay}s...")
            else:
                data.reset_index(inplace=True)
                data.rename(columns={
                    'date': 'Date',
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close', # This is the unadjusted close
                    '5. volume': 'Volume'
                }, inplace=True)
                
                # For Alpha Vantage get_daily, 'Adj_Close' is not provided.
                # We will primarily use 'Close'. If 'Adj_Close' is needed later,
                # it would require fetching split/dividend data and calculating,
                # or using a paid AV endpoint that provides it.
                # For now, we'll ensure 'Close' is present.
                
                desired_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                data = data[[col for col in desired_cols if col in data.columns]].copy()

                data['Date'] = pd.to_datetime(data['Date'])
                data.sort_values(by='Date', ascending=True, inplace=True)

                # Filter data from the specified start_date
                start_datetime = pd.to_datetime(start_date)
                data = data[data['Date'] >= start_datetime].copy()

                if data.empty:
                    print(f"No data found for {api_ticker} (AV) on or after {start_date}.")
                    return pd.DataFrame()

                print(f"Successfully fetched and filtered {len(data)} rows for {api_ticker} from Alpha Vantage.")
                return data
        except Exception as e:
            error_message = str(e).lower()
            if "premium endpoint" in error_message or "invalid api call" in error_message or "api key" in error_message:
                print(f"API Error for {api_ticker} (AV) (will not retry): {e}")
                return pd.DataFrame()
            # Handle common AV rate limit message
            if "our standard api call frequency is" in error_message or "thank you for using alpha vantage" in error_message:
                 print(f"Alpha Vantage rate limit likely hit for {api_ticker}: {e}")
            else:
                print(f"Error for {api_ticker} (AV): {e}. Retrying in {current_delay}s...")
        attempt += 1
        if attempt < max_retries:
            time.sleep(current_delay)
            current_delay = min(current_delay * 2, 360) # Increase delay, max 6 minutes for AV
    print(f"Failed to fetch data for {api_ticker} (AV) after {max_retries} attempts.")
    return pd.DataFrame()

def run_ingestion_for_all_tickers(start_date_to_fetch: str = FETCH_START_DATE):
    """Runs data ingestion for all tickers defined in TICKERS_CONFIG."""
    for ticker_key, config_details in TICKERS_CONFIG.items():
        print(f"\n--- Ingesting data for {ticker_key} ({config_details['api_ticker']}) from {start_date_to_fetch} ---")
        df_raw = pd.DataFrame()
        api_ticker_name = config_details['api_ticker']

        if config_details['source'] == "yfinance":
            df_raw = fetch_yfinance_data(api_ticker_name, start_date=start_date_to_fetch)
        elif config_details['source'] == "alpha_vantage":
            df_raw = fetch_alpha_vantage_data(api_ticker_name, start_date=start_date_to_fetch)
        else:
            print(f"WARNING: Unknown data source '{config_details['source']}' for {ticker_key}")
            continue

        if not df_raw.empty:
            # Ensure 'Close' column exists, which is our primary TARGET_COLUMN
            if 'Close' not in df_raw.columns:
                print(f"CRITICAL: 'Close' column missing for {ticker_key} after fetch. Skipping save.")
                continue

            # If 'Adj_Close' is missing after fetch, and 'Close' exists, we can choose to
            # create 'Adj_Close' from 'Close' or just proceed. For simplicity, we'll proceed
            # and data_processing will use TARGET_COLUMN ('Close').
            if 'Adj_Close' not in df_raw.columns:
                 print(f"Note: 'Adj_Close' column is not available for {ticker_key}. Will proceed using 'Close'.")

            raw_file_path = get_raw_data_path(ticker_key) # Uses new config helper
            # os.makedirs(os.path.dirname(raw_file_path), exist_ok=True) # get_raw_data_path now handles this
            df_raw.to_csv(raw_file_path, index=False)
            print(f"Raw data for {ticker_key} saved to: {raw_file_path}")
        else:
            print(f"Failed to fetch raw data for {ticker_key}.")

if __name__ == "__main__":
    run_ingestion_for_all_tickers() # Will use FETCH_START_DATE from this file