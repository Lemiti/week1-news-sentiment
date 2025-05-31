import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import talib
import os

# === Configuration ===
DATA_PATH = "../data/MSFT_historical_data.csv"
PLOT_DIR = "../notebooks"
os.makedirs(PLOT_DIR, exist_ok=True)

# === Load Data ===
print("[INFO] Loading stock price data...")
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# === Ensure columns exist ===
required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
if not all(col in df.columns for col in required_cols):
    raise ValueError("Missing required OHLCV columns in CSV")

# === 1. Simple Moving Average (SMA) ===
df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Close Price', alpha=0.6)
plt.plot(df.index, df['SMA_20'], label='SMA 20')
plt.plot(df.index, df['SMA_50'], label='SMA 50')
plt.title("MSFT - Simple Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/msft_sma.png")
plt.close()

# === 2. RSI (Relative Strength Index) ===
df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

plt.figure(figsize=(10, 4))
plt.plot(df.index, df['RSI'], label='RSI', color='orange')
plt.axhline(70, linestyle='--', color='red', label='Overbought')
plt.axhline(30, linestyle='--', color='green', label='Oversold')
plt.title("MSFT - Relative Strength Index (RSI)")
plt.xlabel("Date")
plt.ylabel("RSI")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/msft_rsi.png")
plt.close()

# === 3. MACD (Moving Average Convergence Divergence) ===
df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'],
                                                            fastperiod=12,
                                                            slowperiod=26,
                                                            signalperiod=9)

plt.figure(figsize=(10, 4))
plt.plot(df.index, df['MACD'], label='MACD')
plt.plot(df.index, df['MACD_Signal'], label='Signal')
plt.bar(df.index, df['MACD_Hist'], label='Histogram', alpha=0.3)
plt.title("MSFT - MACD Indicator")
plt.xlabel("Date")
plt.ylabel("MACD")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/msft_macd.png")
plt.close()

print("[✅] All indicators calculated and plots saved to notebooks/")

