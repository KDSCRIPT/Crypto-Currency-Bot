import streamlit as st
import pandas as pd
from data_fecther import get_dataset
from predictor import make_future_forecasts
import tensorflow as tf
import requests
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from scipy import stats
import numpy as np

# Define trading parameters
st.title("Crypto Trading Bot")

# Input form
st.sidebar.header("Trading Parameters")

cryptocurrency = st.sidebar.selectbox("Trading Pair", ["BTC", "ETH", "LTC"])
short_period = st.sidebar.slider("Short-term SMA Period", min_value=1, max_value=30, value=10)
long_period = st.sidebar.slider("Long-term SMA Period", min_value=5, max_value=50, value=30)

# Calculate the Simple Moving Average (SMA)
def calculate_sma(data, period):
    return data['close'].rolling(window=period).mean()

# Trading strategy using MAE as a metric
def trading_strategy(data):
    if data['short_sma'] > data['long_sma']:
        return 'Buy'
    elif data['short_sma'] < data['long_sma']:
        return 'Sell'
    else:
        return 'Hold'

start_date = st.date_input("Start Date", datetime.datetime(2018, 1, 1))
end_date = st.date_input("End Date", datetime.datetime(2023,1,1))

# Convert date objects to datetime.datetime objects
start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time())

current_date = datetime.datetime.now()
if end_date > current_date:
    st.error("End Date cannot be beyond the current date.")
if (end_date - start_date).days < 30:
    st.error("Date range must be at least 30 days for the bot to have sufficient data for prediction.")
else:
    get_dataset(start_date, end_date)  # Updates the dataset

    # Main content
    st.write(f"Trading Pair: {cryptocurrency}")
    st.write(f"Short-term SMA Period: {short_period}")
    st.write(f"Long-term SMA Period: {long_period}")

    historical_data = pd.read_csv(f"{cryptocurrency}_historical_data.csv")
    historical_data['timestamp'] = pd.to_datetime(historical_data['time'])

    # Calculate the SMAs
    historical_data['short_sma'] = calculate_sma(historical_data, short_period)
    historical_data['long_sma'] = calculate_sma(historical_data, long_period)

    # Implement trading strategy
    historical_data['signal'] = historical_data.apply(trading_strategy, axis=1)

    # Display the backtesting results
    st.header("Backtesting Results")
    st.dataframe(historical_data[['timestamp', 'close', 'signal']])

    # Calculate Mean Absolute Error (MAE) using your model
    # Ensure you have the actual and predicted values for your model
    model_name = f"{cryptocurrency}_model.h5"
    model = tf.keras.models.load_model(model_name)
    actual_prices = historical_data["close"]  # Replace with your actual prices
    make_forecasts = make_future_forecasts(actual_prices, model)
    st.header("Bitcoin Price Forecast for the Next Seven Days")
    date_ranges = [end_date + datetime.timedelta(days=i) for i in range(1, 8)]
    date_ranges = [date.date() for date in date_ranges]
    forecast_data = {'Date': date_ranges, f'Forecasted Price of {cryptocurrency}': make_forecasts}
    forecast_df = pd.DataFrame(forecast_data)
    st.table(forecast_df)

    # Calculate performance metrics
    cumulative_return = (historical_data['close'] / historical_data['close'].iloc[0] - 1) * 100

    # Add a header for performance metrics
    st.header("Bot Performance Metrics")

    # Label for the line chart
    st.write("Cumulative Return Over Time")

    # Display the line chart
    st.line_chart(cumulative_return, use_container_width=True)
