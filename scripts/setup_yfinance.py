import yfinance as yf

# Download Agilent Technologies stock data
a = yf.download("A", start="2020-01-01", end="2021-01-01")
a.to_csv("data/A_historical_data.csv")

