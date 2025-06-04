from textblob import TextBlob
import pandas as pd

# Load news
news_df = pd.read_csv("data/news.csv")
news_df['date'] = pd.to_datetime(news_df['date'])

# Add sentiment score
news_df['sentiment'] = news_df['headline'].apply(lambda h: TextBlob(str(h)).sentiment.polarity)

daily_sentiment = news_df.groupby(['date', 'stock'])['sentiment'].mean().reset_index()
daily_sentiment.columns = ['date', 'stock', 'avg_sentiment']

stock_df = pd.read_csv("data/AAPL_historical_data.csv")
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
stock_df.sort_values('Date', inplace=True)
stock_df['return'] = stock_df['Close'].pct_change()
stock_df = stock_df[['Date', 'return']].rename(columns={'Date': 'date'})
stock_df['stock'] = 'AAPL'

merged_df = pd.merge(daily_sentiment, stock_df, on=['date', 'stock'])

corr = merged_df['avg_sentiment'].corr(merged_df['return'])
print(f"Pearson Correlation: {corr:.4f}")

sns.scatterplot(x='avg_sentiment', y='return', data=merged_df)
plt.title(f'Sentiment vs Return (r={corr:.2f})')
plt.xlabel('Average Daily Sentiment')
plt.ylabel('Daily Stock Return')
plt.tight_layout()
plt.savefig("notebooks/sentiment_vs_return.png")
plt.show()


for stock in merged_df['stock'].unique():
    temp = merged_df[merged_df['stock'] == stock]
    corr = temp['avg_sentiment'].corr(temp['return'])
    print(f"{stock}: correlation = {corr:.3f}")

