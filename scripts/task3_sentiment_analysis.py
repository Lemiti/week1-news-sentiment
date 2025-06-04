# scripts/task3_sentiment_analysis.py

import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import timedelta

# === Config ===
NEWS_PATH = "/content/drive/MyDrive/10Acadamy/raw_analyst_ratings.csv"
PRICE_PATH = "/content/drive/MyDrive/10Acadamy/AAPL_historical_data.csv"
PLOT_DIR = "/content/drive/MyDrive/10Acadamy/images"
os.makedirs(PLOT_DIR, exist_ok=True)

# === 1. Load & Preprocess News Data ===
def load_news_data():
    """Load and preprocess news data with sentiment analysis"""
    news_df = pd.read_csv(NEWS_PATH)
    
    # Normalize dates and filter valid entries
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce', utc=True).dt.tz_convert(None).dt.normalize()
    news_df = news_df.dropna(subset=['date'])
    
    # Sentiment analysis with neutral classification
    news_df['sentiment'] = news_df['headline'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    news_df['sentiment_class'] = np.where(
        news_df['sentiment'] > 0.1, 'positive',
        np.where(news_df['sentiment'] < -0.1, 'negative', 'neutral')
    )
    
    return news_df

# === 2. Load & Process Stock Data ===
def load_price_data():
    """Load and preprocess stock price data"""
    price_df = pd.read_csv(PRICE_PATH, skiprows=[1, 2])
    price_df.columns = price_df.columns.str.strip()
    
    # Date handling and cleaning
    price_df['date'] = pd.to_datetime(price_df['Date'], errors='coerce')
    price_df = price_df.dropna(subset=['date'])
    
    # Numeric conversion
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_cols:
        if col in price_df.columns:
            price_df[col] = pd.to_numeric(price_df[col], errors='coerce')
    
    # Calculate returns
    price_df.sort_values('date', inplace=True)
    price_df['return'] = price_df['Close'].pct_change()
    price_df['next_day_return'] = price_df['return'].shift(-1)  # For lag analysis
    price_df['stock'] = 'A'
    
    return price_df[['date', 'stock', 'return', 'next_day_return']]

# === 3. Sentiment Aggregation ===
def aggregate_sentiment(news_df):
    """Calculate daily aggregated sentiment"""
    daily_sentiment = news_df.groupby(['date', 'stock']).agg({
        'sentiment': ['mean', 'count'],
        'sentiment_class': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'neutral'
    }).reset_index()
    
    daily_sentiment.columns = ['date', 'stock', 'avg_sentiment', 'article_count', 'dominant_sentiment']
    return daily_sentiment

# === 4. Correlation Analysis ===
def analyze_correlations(merged_df):
    """Calculate and visualize correlations"""
    # Same-day correlation
    corr = merged_df['avg_sentiment'].corr(merged_df['return'])
    print(f"📈 Same-day correlation: {corr:.4f}")
    
    # Next-day correlation (predictive power)
    next_day_corr = merged_df['avg_sentiment'].corr(merged_df['next_day_return'])
    print(f"🔮 Next-day correlation: {next_day_corr:.4f}")
    
    # Sentiment class vs returns
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=merged_df, x='dominant_sentiment', y='return')
    plt.title("Same-Day Returns by Sentiment Class")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=merged_df, x='dominant_sentiment', y='next_day_return')
    plt.title("Next-Day Returns by Sentiment Class for AAPL")
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/sentiment_class_analysis_AAPL.png")
    plt.close()
    
    return corr, next_day_corr

# === Main Execution ===
if __name__ == "__main__":
    # Data loading
    news_df = load_news_data()
    price_df = load_price_data()
    daily_sentiment = aggregate_sentiment(news_df)
    
    # Merge with forward-fill for trading days without news
    merged = pd.merge(price_df, daily_sentiment, on=['date', 'stock'], how='left')
    merged[['avg_sentiment', 'article_count']] = merged.groupby('stock')[
        ['avg_sentiment', 'article_count']].fillna(0)
    merged['dominant_sentiment'] = merged['dominant_sentiment'].fillna('neutral')
    
    # Analysis
    corr, next_day_corr = analyze_correlations(merged)
    
    # Scatter plot (original visualization)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged, x='avg_sentiment', y='return', 
                   hue='dominant_sentiment', palette={'positive':'green', 'negative':'red', 'neutral':'gray'})
    plt.title(f"Sentiment vs Return (r = {corr:.2f})")
    plt.xlabel("Average Daily Sentiment")
    plt.ylabel("Daily Stock Return For AAPL")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/sentiment_vs_return_AAPL.png")
    plt.show()
