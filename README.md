# 🧠 Week 1 Challenge: News Sentiment Analysis for Stock Prediction

## 🚀 Project Overview

This repository contains the solution for **Week 1 of the 10 Academy AI Challenge**, implementing a pipeline to analyze the relationship between financial news sentiment and stock price movements of Agilent Technologies (NYSE: A).

Key components:
- Sentiment analysis of financial headlines using TextBlob
- Stock price data processing from Yahoo Finance
- Correlation analysis between news sentiment and price returns
- Visualization of sentiment-return relationships

## 📊 Key Findings

- Achieved correlation coefficient of [X] between sentiment scores and daily returns
- Discovered [Y] pattern in sentiment distribution
- Identified [Z] relationship between news volume and price volatility

## 🛠️ Implementation

### Data Pipeline
1. **News Data**: Processed raw analyst ratings from `raw_analyst_ratings.csv`
2. **Price Data**: Fetched historical prices using `yfinance` (2020-2021)
3. **Sentiment Analysis**: Calculated polarity scores for headlines
4. **Correlation**: Computed daily sentiment-return correlations

### How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download stock data
python scripts/fetch_stock_data.py

# 3. Run technical analysis (eg. MSFT indicators)
python scripts/technical_analysis.py

# 4. Run sentiment analysis (GOOG, AMZN, AAPL stock)
python scripts/task3_sentiment_analysis.py
```
# 📁 Repository Structure

```plain text
.
├── data/
│   ├── raw_analyst_ratings.csv    # News headlines
│   ├── AAPL_historical_data.csv      # Stock price data for AAPL
│   ├── AMZN_historical_data.csv      # Stock price data for AMZN
│   ├── GOOG_historical_data.csv      # Stock price data for GOOG
│   ├── TSLA_historical_data.csv      # Stock price data for TSLA
│   └── META_historical_data.csv      # Stock price data for META
├── notebooks/
│   ├── \*.png            # images
│   └── README.md
├── scripts/
│   ├── eda.py
│   ├── fetch_stock_data.py         # Data download
│   ├── setup_nlp.py
│   ├── technical_analysis.py 
│   └── task3_sentiment_analysis.py # Main analysis
├── images/
│   └── sentiment_vs_return.png    # Output visualization
├── requirements.txt
└── README.md
```

- **Technical Analysis**: Implements SMA, RSI, and MACD indicators for MSFT, 
- **Sentiment Analysis**: Correlates news sentiment with GOOG, AMZN, TSLA, AAPL stock returns
