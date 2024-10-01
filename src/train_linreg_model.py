import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import joblib

current_dir = Path(__file__).resolve().parent

data_file = current_dir.parent/ 'data' / 'AAPL_data.csv'
data = pd.read_csv(data_file)

X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

model_file = current_dir.parent/ 'models' / 'linreg_model.pk1'
# Save the trained model to the models folder
joblib.dump(model, model_file)

# Optionally, calculate MAE on the training set (for reference)
y_train_pred = model.predict(X)
mae = mean_absolute_error(y, y_train_pred)
print(f"Training MAE: {mae}")
