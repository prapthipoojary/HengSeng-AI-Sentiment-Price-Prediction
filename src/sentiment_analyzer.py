
from transformers import pipeline
from functools import lru_cache

@lru_cache(maxsize=1)
def load_finbert():
    print("Loading FinBERT...")
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert"
    )


def analyze_sentiment(news):

    analyzer = load_finbert()
    result = analyzer(news)[0]

    return {
        "news": news,
        "sentiment": result["label"],
        "confidence": round(result["score"] * 100, 2)
    }
