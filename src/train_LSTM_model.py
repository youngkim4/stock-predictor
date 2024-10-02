import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.losses import Huber  # Import the Huber loss


# Load stock data
current_dir = Path(__file__).resolve().parent
data_file = current_dir.parent/ 'data' / 'AAPL_data.csv'
data = pd.read_csv(data_file)

# Feature Engineering: Add new features like moving averages, RSI, etc.
data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
data['SMA_200'] = data['Close'].rolling(window=200).mean() # 200-day Simple Moving Average

# Function to calculate RSI
def compute_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Add RSI as a feature
data['RSI'] = compute_rsi(data['Close'])

# Drop any rows with NaN values that may have been introduced by rolling functions
data.dropna(inplace=True)

# Normalize the data to a range (0, 1)
features = ['Close', 'SMA_50', 'SMA_200', 'RSI']

print(f"Data shape after dropping NaNs: {data.shape}")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])

# Create sequences of past 60 days' prices to predict the next day
def create_sequences(data, time_step=60):
    X = []
    y = []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i])  # The past 60 days
        y.append(data[i, 0])              # The next day’s price
    return np.array(X), np.array(y)

# Create sequences for the LSTM
time_step = 60
X, y = create_sequences(scaled_data, time_step)

# Reshape X to fit the LSTM input requirements: [samples, time steps, features]
print(f"Scaled data shape: {scaled_data.shape}")
X = X.reshape((X.shape[0], X.shape[1], len(features)))

# Split the data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(Input(shape=(X_train.shape[1], len(features))))  # Define input layer
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))  # Add dropout to prevent overfitting
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=50))  # Add more neurons for more complexity
model.add(Dense(units=1))   # Final output layer to predict the next price

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

print(f"Training data shape: {X_train.shape}")

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Evaluate the model on test data
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

# Save the model
model.save(current_dir.parent / 'models' / 'lstm_model.keras')

print("LSTM model trained and saved to '../models/lstm_model.keras'")