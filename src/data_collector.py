
import yfinance as yf
import pandas as pd
import os

def collect_stock_data():
    ticker = "^HSI"

    print("Downloading Hang Seng Index data...")

    data = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        print("No stock data received.")
        return None

    os.makedirs("data", exist_ok=True)

    data.to_csv("data/hang_seng_data.csv")

    print("Stock data downloaded successfully!")
    print(data.tail())

    return data


if __name__ == "__main__":
    collect_stock_data()
