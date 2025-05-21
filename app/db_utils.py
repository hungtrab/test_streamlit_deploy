# app/db_utils.py
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os # For os.path.dirname if needed for DATABASE_PATH directly here

# Import DATABASE_PATH from config.py
try:
    from .config import DATABASE_PATH, TICKERS_CONFIG # TICKERS_CONFIG might be useful for validation
except ImportError: # Fallback for direct script execution (e.g. if __name__ == "__main__")
    # This assumes config.py is in the same directory if run directly
    # For robust direct execution, you might need to adjust sys.path
    from config import DATABASE_PATH, TICKERS_CONFIG


def get_db_connection():
    """Establishes a connection to the SQLite database."""
    # detect_types allows SQLite to understand more Python types (like datetime if stored as TIMESTAMP)
    conn = sqlite3.connect(DATABASE_PATH, timeout=10.0, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row # Access columns by name
    return conn

def init_db():
    """Initializes the database tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table to store historical actual closing prices for each ticker
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            ticker_key TEXT NOT NULL,
            date TEXT NOT NULL,         -- Format YYYY-MM-DD HH:MM:SS or YYYY-MM-DD
            close_price REAL NOT NULL,
            PRIMARY KEY (ticker_key, date)
        )
    ''')

    # Table to store predictions made by models for each ticker
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker_key TEXT NOT NULL,
            prediction_date TEXT NOT NULL, -- The date FOR WHICH the prediction is made (YYYY-MM-DD)
            predicted_price REAL NOT NULL,
            model_used TEXT NOT NULL,      -- e.g., "xgboost", "lstm"
            actual_price REAL,             -- Actual price for prediction_date, updated later
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- When the prediction record was inserted/updated
            UNIQUE (ticker_key, prediction_date, model_used) -- A ticker can have one prediction per model per day
        )
    ''')
    conn.commit()
    conn.close()
    print(f"DB_UTILS: Database initialized/checked at {DATABASE_PATH}")

def save_actual_prices(ticker_key: str, df_prices: pd.DataFrame):
    """
    Saves or updates actual closing prices for a given ticker.
    df_prices: DataFrame with 'Date' (DatetimeIndex) as index and a 'Close' column.
    """
    if df_prices.empty:
        print(f"DB_UTILS: No prices provided for {ticker_key} to save.")
        return

    if not isinstance(df_prices.index, pd.DatetimeIndex):
        print(f"ERROR (DB_UTILS): df_prices for {ticker_key} must have a DatetimeIndex.")
        return
    if 'Close' not in df_prices.columns:
        print(f"ERROR (DB_UTILS): df_prices for {ticker_key} must have a 'Close' column.")
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    
    saved_count = 0
    try:
        for date_val, row in df_prices.iterrows():
            if pd.isna(date_val):
                print(f"ERROR (DB_UTILS): Invalid date value for {ticker_key}. Skipping.")
                continue

            date_str = date_val.strftime('%Y-%m-%d') # Store date as YYYY-MM-DD string for simplicity
            close_price = row['Close']
            
            if pd.isna(close_price): # Skip NaN prices
                continue

            try:
                cursor.execute("""
                    INSERT INTO stock_prices (ticker_key, date, close_price)
                    VALUES (?, ?, ?)
                    ON CONFLICT(ticker_key, date) DO UPDATE SET
                        close_price=excluded.close_price
                """, (ticker_key, date_str, close_price))
                # Check if a row was changed (inserted or updated)
                if cursor.rowcount > 0: # This check might not be perfectly reliable for ON CONFLICT with all SQLite versions
                                    # but gives a general idea. A more robust way is to query before insert/update.
                    # A simpler way to track is not implemented here to keep it concise
                    pass # Assume success for now
                saved_count +=1 # Count every attempt as a save/update for simplicity of logging
            except Exception as e:
                print(f"ERROR (DB_UTILS): Failed to save/update price for {ticker_key} on {date_str}: {e}")
        
        conn.commit()
    except Exception as e_main:
        print(f"ERROR (DB_UTILS): Failed to save prices for {ticker_key}: {e_main}")
        conn.rollback() # Rollback in case of any error during the loop
    finally:
        conn.close()
    print(f"DB_UTILS: Attempted to save/update {saved_count} price records for {ticker_key}.")


def save_prediction(ticker_key: str, prediction_date_str: str, predicted_price: float, model_used: str):
    """Saves or updates a prediction for a given ticker, date, and model."""
    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT INTO predictions (ticker_key, prediction_date, predicted_price, model_used)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker_key, prediction_date, model_used) DO UPDATE SET
                predicted_price=excluded.predicted_price,
                created_at=CURRENT_TIMESTAMP
        """, (ticker_key, prediction_date_str, predicted_price, model_used))
        conn.commit()
        print(f"DB_UTILS: Saved/Updated prediction for {ticker_key} on {prediction_date_str} using {model_used}.")
    except Exception as e:
        print(f"ERROR (DB_UTILS): Failed to save prediction for {ticker_key} on {prediction_date_str}: {e}")
    finally:
        conn.close()

def update_actual_price_for_prediction(ticker_key: str, date_str: str, actual_price: float):
    """Updates the actual_price in the predictions table for records where it's NULL."""
    conn = get_db_connection()
    try:
        # Update all model predictions for that ticker and date
        cursor = conn.execute("""
            UPDATE predictions SET actual_price = ?
            WHERE ticker_key = ? AND prediction_date = ? AND actual_price IS NULL
        """, (actual_price, ticker_key, date_str))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"DB_UTILS: Updated actual price for {cursor.rowcount} predictions for {ticker_key} on {date_str}.")
    except Exception as e:
        print(f"ERROR (DB_UTILS): Failed to update actual price for predictions of {ticker_key} on {date_str}: {e}")
    finally:
        conn.close()

def get_prediction_history(ticker_key: str, limit: int = 100) -> pd.DataFrame:
    """Retrieves prediction history for a specific ticker, joined with actual prices."""
    conn = get_db_connection()
    df = pd.DataFrame() # Default empty DataFrame
    try:
        # Join predictions with stock_prices on ticker_key and date
        query = """
            SELECT
                p.prediction_date,
                p.predicted_price,
                p.model_used,
                p.actual_price, -- This is the actual price updated AT THE TIME OF PREDICTION if available then
                sp.close_price as historical_actual -- This is the historical close price from stock_prices table
            FROM predictions p
            LEFT JOIN stock_prices sp
                ON p.ticker_key = sp.ticker_key AND p.prediction_date = sp.date
            WHERE p.ticker_key = ?
            ORDER BY p.prediction_date DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(ticker_key, limit))
        if 'prediction_date' in df.columns and not df.empty:
            df['prediction_date'] = pd.to_datetime(df['prediction_date'])
    except Exception as e:
        print(f"ERROR (DB_UTILS): Failed to get prediction history for {ticker_key}: {e}")
    finally:
        conn.close()
    return df

def get_latest_close_prices(ticker_key: str, days: int = 30) -> pd.DataFrame:
    """Retrieves the latest 'days' of actual closing prices for a specific ticker."""
    conn = get_db_connection()
    df = pd.DataFrame()
    try:
        query = """
            SELECT date, close_price
            FROM stock_prices
            WHERE ticker_key = ?
            ORDER BY date DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(ticker_key, days))
        if 'date' in df.columns and not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(ascending=True, inplace=True) # Sort chronologically for feature engineering
        elif not df.empty and 'date' not in df.columns:
             print(f"DB_UTILS: 'date' column missing in get_latest_close_prices for {ticker_key}.")
    except Exception as e:
        print(f"ERROR (DB_UTILS): Failed to get latest close prices for {ticker_key}: {e}")
    finally:
        conn.close()
    return df

if __name__ == "__main__":
    print("Running DB Utils direct test (initializing DB)...")
    # This will create the DB file and tables if they don't exist
    # at the DATABASE_PATH defined in config.py (when config.py is in the same dir or PYTHONPATH is set)
    init_db()

    # Example: Test saving and fetching some data (requires config.py to be accessible)
    if "GSPC" in TICKERS_CONFIG:
        print("\nTesting with GSPC...")
        # Create dummy price data for GSPC
        test_dates = pd.to_datetime([datetime.now() - timedelta(days=i) for i in range(5, 0, -1)])
        test_prices = pd.DataFrame({
            'Close': [100.0, 101.0, 100.5, 102.0, 101.5]
        }, index=test_dates)
        save_actual_prices("GSPC", test_prices)
        
        # Fetch latest prices for GSPC
        latest_gspc = get_latest_close_prices("GSPC", days=10)
        print("Latest GSPC prices from DB:")
        print(latest_gspc.to_string())

        # Save a dummy prediction
        today_str = datetime.now().strftime('%Y-%m-%d')
        save_prediction("GSPC", today_str, 103.0, "xgboost_test")
        
        # Update actual for that prediction
        update_actual_price_for_prediction("GSPC", today_str, 102.5)

        # Get history
        history_gspc = get_prediction_history("GSPC", limit=5)
        print("\nGSPC Prediction History from DB:")
        print(history_gspc.to_string())

    print("\nDB Utils direct test finished.")