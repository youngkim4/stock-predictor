import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

current_dir = Path(__file__).resolve().parent

# Load stock data
data_file = current_dir.parent / 'data' / 'AAPL_data.csv'
data = pd.read_csv(data_file)

# Function to calculate RSI
def compute_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Feature Engineering: Add new features like moving averages, RSI, etc.
data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
data['SMA_200'] = data['Close'].rolling(window=200).mean()  # 200-day Simple Moving Average
data['RSI'] = compute_rsi(data['Close'])  # Same RSI function as in training

# Drop any NaN values initially
data.dropna(inplace=True)

# Use the same feature set as in training
features = ['Close', 'SMA_50', 'SMA_200', 'RSI']
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler on initial data
scaled_data = scaler.fit_transform(data[features])

# Prepare the most recent data for prediction (e.g., the last 60 days)
time_step = 60
recent_data = scaled_data[-time_step:]
recent_data = recent_data.reshape((1, time_step, len(features)))

# Load the saved LSTM model
model_file = current_dir.parent / 'models' / 'lstm_model.keras'
model = load_model(model_file)

# Predict future stock prices
days_to_predict = 30  # Specify the number of days you want to predict
predicted_prices = []

for i in range(days_to_predict):
    # Predict the next day's price (only the 'Close' price is predicted)
    next_day_price = model.predict(recent_data)
    next_day_price = next_day_price.reshape(1, 1, 1)  # The shape is [1, 1, 1] since we only predict the 'Close' price

    # Inverse transform to get the predicted 'Close' price in the original scale
    previous_features = recent_data[:, -1, 1:].reshape(1, 1, -1)  # Reshape to [1, 1, 3]
    predicted_price_actual = scaler.inverse_transform(
        np.concatenate([next_day_price.reshape(1, -1), previous_features.reshape(1, -1)], axis=1)
    )[:, 0]

    predicted_prices.append(predicted_price_actual[0])  # Append predicted price to list

    # Update 'recent_data' with the new predicted price
    new_close_price = predicted_price_actual[0]  # Use the actual scale price

    # Manually calculate SMA_50, SMA_200, and RSI for the predicted value
    if len(data) >= 50:
        sma_50 = data['Close'].iloc[-50:].mean()  # Last 50 days for SMA_50
    else:
        sma_50 = data['Close'].mean()  # Use the available data for SMA_50

    if len(data) >= 200:
        sma_200 = data['Close'].iloc[-200:].mean()  # Last 200 days for SMA_200
    else:
        sma_200 = data['Close'].mean()  # Use the available data for SMA_200

    # Compute RSI based on the last 14 days (or less if not enough data)
    rsi = compute_rsi(pd.concat([data['Close'], pd.Series([new_close_price])]))

    # Append the predicted 'Close' price and the calculated SMA_50, SMA_200, RSI to the original data
    new_row = pd.DataFrame({'Close': [new_close_price], 'SMA_50': [sma_50], 'SMA_200': [sma_200], 'RSI': [rsi.iloc[-1]]})
    data = pd.concat([data, new_row], ignore_index=True)

    # Rescale the updated data without dropping NaNs
    scaled_data = scaler.fit_transform(data[features])

    # Update recent_data for the next prediction
    recent_data = scaled_data[-time_step:].reshape((1, time_step, len(features)))

# Print the predicted prices
predicted_prices_str = ', '.join([f"{price:.2f}" for price in predicted_prices])
print(f"Predicted prices for the next {days_to_predict} days: {predicted_prices_str}")
