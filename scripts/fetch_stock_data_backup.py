import yfinance as yf
import pandas as pd
import os

# Ensure data directory exists
os.makedirs("../data", exist_ok=True)

def download_stock_data():
    """Downloads stock data for MSFT (technical analysis) and A (sentiment analysis)"""
    print("⏳ Downloading stock data...")
    
    # Microsoft data (for technical analysis)
    msft = yf.download("MSFT", start="2020-01-01", end="2021-01-01")
    msft.to_csv("../data/MSFT_historical_data.csv")
    print("✅ Microsoft (MSFT) data saved to data/MSFT_historical_data.csv")
    
    # Agilent data (for sentiment analysis)
    a = yf.download("A", start="2020-01-01", end="2021-01-01")
    a.to_csv("../data/A_historical_data.csv")
    print("✅ Agilent (A) data saved to data/A_historical_data.csv")

if __name__ == "__main__":
    download_stock_data()
