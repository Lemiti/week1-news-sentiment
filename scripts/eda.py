# scripts/eda.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob

# === CONFIGURATION ===
DATA_PATH = "../data/raw_analyst_ratings.csv"  # You can change this to any other stock CSV
PLOT_DIR = "../notebooks"  #Place where the plots to be saved

os.makedirs(PLOT_DIR, exist_ok=True)

# === LOAD DATA ===
print("[INFO] Loading data...")
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# === BASIC EDA ===
print("\n[INFO] Basic Statistics")
print(df.describe(include='all'))

# === 1. Headline Length Analysis ===
if 'headline' in df.columns:
    print("[INFO] Analyzing headline lengths...")
    df['headline_length'] = df['headline'].str.len()
    
    plt.figure(figsize=(8, 4))
    sns.histplot(df['headline_length'], bins=50, kde=True)
    plt.title("Headline Length Distribution")
    plt.xlabel("Length of Headline")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/headline_length_distribution.png")
    plt.close()

# === 2. Publisher Analysis ===
if 'publisher' in df.columns:
    print("[INFO] Analyzing publishers...")
    publisher_counts = df['publisher'].value_counts().head(20)
    
    plt.figure(figsize=(10, 5))
    publisher_counts.plot(kind='bar')
    plt.title("Top 20 Publishers by Article Count")
    plt.ylabel("Number of Articles")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/top_publishers.png")

# === 3. Time Series Analysis ===
print("[INFO] Plotting article counts by date...")
if 'date' in df.columns:
    daily_counts = df['date'].dt.date.value_counts().sort_index()
    
    plt.figure(figsize=(12, 4))
    daily_counts.plot()
    plt.title("Articles Published Per Day")
    plt.xlabel("Date")
    plt.ylabel("Number of Articles")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/articles_per_day.png")
    plt.close()

    df['hour'] = df['date'].dt.hour
    plt.figure(figsize=(8, 4))
    sns.countplot(x='hour', data=df)
    plt.title("News Publication by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/article_publication_by_hour.png")
    plt.close()

# === 4. Topic Modeling (Keyword Extraction) ===
if 'headline' in df.columns:
    print("[INFO] Generating word cloud from headlines...")
    text = " ".join(df['headline'].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, stopwords=STOPWORDS).generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Common Words in Headlines")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/headline_wordcloud.png")
    plt.close()

# === 5. Publisher Domain Analysis (Optional) ===
if 'publisher' in df.columns:
    print("[INFO] Checking for publisher email domains...")
    df['domain'] = df['publisher'].str.extract(r'@([\w\.-]+)')
    domain_counts = df['domain'].value_counts().dropna().head(10)

    if not domain_counts.empty:
        plt.figure(figsize=(10, 4))
        domain_counts.plot(kind='bar')
        plt.title("Top Publisher Email Domains")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/top_publisher_email_domains.png")
        plt.close()
