from src.data_collector import collect_stock_data
from src.price_predictor import predict_stock_price
from src.news_collector import collect_news
from src.news_sentiment import analyze_all_news


def main():
    print("=" * 55)
    print(" HANG SENG AI SENTIMENT & PRICE PREDICTION AGENT")
    print("=" * 55)

    print("\nSTEP 1: Collecting stock market data")
    collect_stock_data()

    print("\nSTEP 2: Predicting stock prices")
    predict_stock_price()

    print("\nSTEP 3: Collecting financial news")
    collect_news()

    print("\nSTEP 4: Analyzing news using FinBERT")
    analyze_all_news()

    print("\n" + "=" * 55)
    print(" AI AGENT EXECUTION COMPLETED!")
    print("=" * 55)


if __name__ == "__main__":
    main()