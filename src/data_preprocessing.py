from pathlib import Path
import yfinance as yf
import pandas as pd

current_dir = Path(__file__).resolve().parent

def download_stock_data(ticker, start_date, end_date):
    # Define the file path
    data = yf.download(ticker, start=start_date, end=end_date)
    file_path = current_dir.parent/ 'data' / f'{ticker}_data.csv'

    # Ensure the directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the data
    data.to_csv(file_path)
    print(f"Data saved to: {file_path}")
    return data

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-09-01"
    stock_data = download_stock_data(ticker, start_date, end_date)
    print(stock_data.head())