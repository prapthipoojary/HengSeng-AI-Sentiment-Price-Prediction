import sys
import os

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import pandas as pd
from src.sentiment_analyzer import analyze_sentiment
import os


def analyze_all_news():

    file_path = "data/financial_news.csv"

    if not os.path.exists(file_path):
        print("News file not found!")
        return

    df = pd.read_csv(file_path)

    if df.empty or "headline" not in df.columns:
        print("No valid news headlines found!")
        return

    results = []

    print("Starting AI news sentiment analysis...")

    for headline in df["headline"].dropna():

        result = analyze_sentiment(str(headline))

        results.append({
            "headline": headline,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"]
        })

        print("\nNews:", headline)
        print("Sentiment:", result["sentiment"])
        print("Confidence:", result["confidence"], "%")

    if results:

        output = pd.DataFrame(results)

        output.to_csv(
            "data/sentiment_results.csv",
            index=False
        )

        print("\n----- SENTIMENT SUMMARY -----")
        print(output["sentiment"].value_counts())

        print("\nResults saved successfully!")


if __name__ == "__main__":
    analyze_all_news()
