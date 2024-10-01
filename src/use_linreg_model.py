import joblib
import pandas as pd
from pathlib import Path

current_dir = Path(__file__).resolve().parent

model_file = current_dir.parent/ 'models' / 'linreg_model.pk1'

# Load the saved Linear Regression model
model = joblib.load(model_file)

# Load stock data
data_file = current_dir.parent/ 'data' / 'AAPL_data.csv'
data = pd.read_csv(data_file)

# Prepare the most recent data point for prediction
recent_data = data[['Open', 'High', 'Low', 'Volume']].iloc[-1].values.reshape(1, -1)

# Number of days to predict
days_to_predict = 10  # Change this number to predict any number of future days
predicted_prices = []

# Predict future stock prices for multiple days
for i in range(days_to_predict):
    # Predict the next day's closing price
    recent_data_df = pd.DataFrame(recent_data, columns=['Open', 'High', 'Low', 'Volume'])
    next_day_price = model.predict(recent_data_df)
    predicted_prices.append(next_day_price[0])

    # Update the recent_data for the next prediction
    # We assume the predicted price becomes the "Open" price for the next day
    recent_data = [[next_day_price[0], recent_data[0][1], recent_data[0][2], recent_data[0][3]]]

# Print all predicted prices
print(f"Predicted closing prices for the next {days_to_predict} days: {predicted_prices}")