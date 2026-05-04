from flask import Flask, request, render_template
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_sentiment_score(text):
    return TextBlob(str(text)).sentiment.polarity

def get_sentiment_label(score):
    return "Positive" if score >= 0 else "Negative"


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']

    if not file:
        return "No file uploaded"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath)

    text_column = "reviewText" if "reviewText" in df.columns else df.columns[0]

    df['sentiment_score'] = df[text_column].apply(get_sentiment_score)
    df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)

    label_counts = df['sentiment_label'].value_counts()

    chart_path = os.path.join("static", "chart.png")
    plt.figure(figsize=(6,4))
    plt.bar(label_counts.index, label_counts.values)
    plt.title("Sentiment Analysis")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return render_template(
        "result.html",
        positive=label_counts.get("Positive", 0),
        negative=label_counts.get("Negative", 0),
        chart="chart.png"
    )
chart_path = os.path.join("static", "chart.png")
os.makedirs(os.path.join(app.root_path, "static"), exist_ok=True)
plt.savefig(chart_path)

if __name__ == "__main__":
    app.run(debug=True)
