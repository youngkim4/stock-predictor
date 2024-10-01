import yfinance as yf
import pandas as pd
from pathlib import Path
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    file_path = Path('..') / 'data' / f'{ticker}_data.csv'
    data.to_csv(file_path)
    return data

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-09-01"
    stock_data = download_stock_data(ticker, start_date, end_date)
    print(stock_data.head)