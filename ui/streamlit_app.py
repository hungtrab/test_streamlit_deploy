# ui/streamlit_app.py
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# --- Configuration ---
# Determine API URL. Prioritize environment variable, then default to localhost.
# When deploying Streamlit separately (e.g., on Streamlit Community Cloud or another server),
# you MUST set the FASTAPI_URL_FOR_UI environment variable to the public URL of your FastAPI backend.
# If running Streamlit in Docker on the same Docker network as the API server,
# you can use the service name (e.g., "http://stock_api_server:8000").
DEFAULT_API_URL = "http://localhost:8000" # Default for local Docker Compose setup
FASTAPI_URL = os.getenv("FASTAPI_URL_FOR_UI", DEFAULT_API_URL)

# Get ticker configurations. In a real app, this might also come from an API endpoint
# or be hardcoded/read from a shared config if Streamlit app is part of the same project.
# For simplicity, we'll hardcode them here, matching your app.config.TICKERS_CONFIG
# This should ideally be kept in sync with your backend's config.
TICKERS_DISPLAY_CONFIG = {
    "GSPC": {"display_name": "S&P 500 Index"},
    "IBM": {"display_name": "IBM Stock"}
}
AVAILABLE_TICKER_KEYS = list(TICKERS_DISPLAY_CONFIG.keys())
AVAILABLE_MODELS = ["xgboost", "random_forest", "lstm"]

st.set_page_config(layout="wide", page_title="Stock & Index Price Prediction Dashboard")

st.title("📈 Stock & Index Price Prediction Dashboard")
st.markdown("---")

# --- Sidebar for Ticker Selection ---
st.sidebar.header("🎯 Ticker Selection")
selected_ticker_key = st.sidebar.selectbox(
    "Select Ticker:",
    options=AVAILABLE_TICKER_KEYS,
    format_func=lambda x: TICKERS_DISPLAY_CONFIG[x]["display_name"], # Show display name
    key="ticker_selector_main"
)
selected_ticker_display_name = TICKERS_DISPLAY_CONFIG[selected_ticker_key]["display_name"]

# --- Section 1: Latest Prediction ---
st.header(f"🔮 Latest Prediction for {selected_ticker_display_name}")

col1_pred, col2_pred_model = st.columns([3,2])

with col2_pred_model:
    latest_pred_model_type = st.selectbox(
        "Select Model for Latest Prediction:",
        options=AVAILABLE_MODELS,
        index=0, # Default to xgboost
        key=f"latest_pred_model_selector_{selected_ticker_key}" # Key includes ticker to reset on ticker change
    )

if col2_pred_model.button(f"Get Latest Prediction ({latest_pred_model_type.replace('_',' ').title()})", key="get_latest_pred_button"):
    with st.spinner(f"Fetching latest prediction for {selected_ticker_display_name} using {latest_pred_model_type}..."):
        try:
            predict_url = f"{FASTAPI_URL}/predict"
            params = {"ticker_key": selected_ticker_key, "model_type": latest_pred_model_type}
            response = requests.post(predict_url, params=params, timeout=20) # Added timeout
            response.raise_for_status()
            prediction_data = response.json()
            
            pred_date_obj = datetime.strptime(prediction_data.get("prediction_date"), '%Y-%m-%d')
            
            with col1_pred:
                st.metric(
                    label=f"Predicted Close for {selected_ticker_display_name} on {pred_date_obj.strftime('%d/%m/%Y')}",
                    value=f"${prediction_data.get('predicted_price'):,.2f}" # Added comma for thousands
                )
                st.caption(f"Model used: {prediction_data.get('model_used', 'N/A').replace('_',' ').title()}")
                if prediction_data.get("message"):
                    st.info(prediction_data.get("message"))

        except requests.exceptions.HTTPError as http_err:
            with col1_pred:
                st.error(f"API Error: {http_err.response.status_code}")
                try: st.error(f"Details: {http_err.response.json().get('detail', http_err.response.text)}")
                except: st.error(f"Details: {http_err.response.text}")
        except requests.exceptions.RequestException as req_err:
            with col1_pred:
                st.error(f"Request Error: Could not connect to API at {FASTAPI_URL}. Details: {req_err}")
        except Exception as e:
            with col1_pred:
                st.error(f"An unexpected error occurred: {e}")

st.markdown("---")

# --- Section 2: Historical Predictions Chart & Table ---
st.header(f"📊 Historical Predictions for {selected_ticker_display_name}")

# Function to load history (cached)
@st.cache_data(ttl=300) # Cache data for 5 minutes
def load_prediction_history_from_api(ticker_key_for_api: str, days_limit: int = 365):
    print(f"UI: Fetching history for {ticker_key_for_api}, limit {days_limit} days from {FASTAPI_URL}")
    try:
        history_url = f"{FASTAPI_URL}/history"
        params = {"ticker_key": ticker_key_for_api, "days_limit": days_limit}
        response = requests.get(history_url, params=params, timeout=30)
        response.raise_for_status()
        history_api_response = response.json() # Expects {"ticker_key": "...", "display_name": "...", "data": [...]}
        
        history_data = history_api_response.get("data", [])
        if not history_data:
            return pd.DataFrame(columns=['prediction_date', 'predicted_price', 'actual_price', 'model_used']) # Return empty with expected columns
        
        df = pd.DataFrame(history_data)
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        df.set_index('prediction_date', inplace=True)
        df.sort_index(ascending=True, inplace=True)
        return df
    except requests.exceptions.HTTPError as http_err:
        st.error(f"API Error loading history: {http_err.response.status_code}")
        try: st.error(f"Details: {http_err.response.json().get('detail', http_err.response.text)}")
        except: st.error(f"Details: {http_err.response.text}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as req_err:
        st.error(f"Request Error loading history: Could not connect to API at {FASTAPI_URL}. Details: {req_err}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading prediction history: {e}")
        return pd.DataFrame()

# Load history for the selected ticker
days_of_history = st.slider("Days of History to Display:", min_value=30, max_value=730, value=180, step=30, key=f"history_days_slider_{selected_ticker_key}")
history_df = load_prediction_history_from_api(selected_ticker_key, days_limit=days_of_history)

if not history_df.empty:
    # Allow filtering by model for the chart
    unique_models_in_history = sorted(history_df['model_used'].dropna().unique())
    if not unique_models_in_history: # Handle case where no models have predictions yet
        unique_models_in_history = AVAILABLE_MODELS # Fallback to all available models for selection
        
    selected_models_for_chart = st.multiselect(
        "Show predictions from models:",
        options=unique_models_in_history,
        default=unique_models_in_history, # Default to show all models that have data
        key=f"history_model_filter_{selected_ticker_key}"
    )

    # Prepare data for chart
    chart_df = history_df[history_df['model_used'].isin(selected_models_for_chart) | history_df['model_used'].isnull()] # Keep actuals

    if not chart_df.empty:
        fig = go.Figure()

        # Plot Actual Prices (only once)
        # 'actual_price' in history_df is the combined one (prediction's actual or stock_price's historical_actual)
        actual_prices_series = chart_df['actual_price'].dropna().sort_index().drop_duplicates() # Drop duplicates for a clean line
        if not actual_prices_series.empty:
            fig.add_trace(go.Scatter(
                x=actual_prices_series.index,
                y=actual_prices_series,
                mode='lines+markers',
                name='Actual Price',
                line=dict(color='deepskyblue', width=2),
                marker=dict(size=4)
            ))

        # Plot Predicted Prices for each selected model
        for model_name in selected_models_for_chart:
            model_pred_series = chart_df[chart_df['model_used'] == model_name]['predicted_price'].dropna()
            if not model_pred_series.empty:
                # Assign distinct colors/styles
                color = 'orange' if 'xgboost' in model_name else 'lightgreen' if 'random_forest' in model_name else 'violet'
                dash = 'solid' if 'xgboost' in model_name else 'dashdot' if 'random_forest' in model_name else 'dot'
                
                fig.add_trace(go.Scatter(
                    x=model_pred_series.index,
                    y=model_pred_series,
                    mode='lines',
                    name=f'Predicted ({model_name.replace("_"," ").title()})',
                    line=dict(color=color, dash=dash, width=1.5)
                ))
        
        fig.update_layout(
            title=f'Historical Comparison: Actual vs. Predicted Prices for {selected_ticker_display_name}',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            legend_title_text='Legend',
            height=500,
            hovermode="x unified" # Better hover experience
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display Table Data
        st.subheader("Historical Data Table (Filtered)")
        display_table_df = chart_df[['model_used', 'predicted_price', 'actual_price']].copy()
        display_table_df.rename(columns={
            'model_used': 'Model Used',
            'predicted_price': 'Predicted Price',
            'actual_price': 'Actual Price'
        }, inplace=True)
        # Format for display
        st.dataframe(display_table_df.sort_index(ascending=False).style.format({
            "Predicted Price": "${:,.2f}",
            "Actual Price": "${:,.2f}"
        }, na_rep="N/A"), height=300)

    else:
        st.info(f"No historical prediction data to display for {selected_ticker_display_name} with the selected model filters.")
else:
    st.info(f"No prediction history available for {selected_ticker_display_name} for the last {days_of_history} days.")

st.markdown("---")
# --- Admin/Manual Actions (Optional) ---
# st.sidebar.header("⚙️ Admin Actions")
# if st.sidebar.button("Trigger Full Data Update & Processing"):
#     with st.spinner("Requesting data pipeline trigger..."):
#         try:
#             # This endpoint would need to be implemented in api_server.py
#             trigger_url = f"{FASTAPI_URL}/admin/trigger-data-update-all"
#             response = requests.post(trigger_url, timeout=10) # Short timeout for trigger
#             response.raise_for_status()
#             st.sidebar.success(response.json().get("message", "Data pipeline trigger request sent!"))
#             st.experimental_rerun() # Rerun to refresh data after a delay
#         except Exception as e:
#             st.sidebar.error(f"Failed to trigger data pipeline: {e}")

st.sidebar.info("Developed as an MLOps Demo.")