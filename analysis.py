import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

df = pd.read_csv("amazon_reviews.csv") 

def get_sentiment_score(text):
    return TextBlob(str(text)).sentiment.polarity

df['sentiment_score'] = df['reviewerName'].apply(get_sentiment_score)

def get_sentiment_label(score):
    if score >= 0:
        return 'Positive'
    else:
        return 'Negative'

df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)

label_counts = df['sentiment_label'].value_counts()


plt.figure(figsize=(6, 4))
plt.bar(label_counts.index, label_counts.values, color=['green', 'red'])
plt.title("Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()


