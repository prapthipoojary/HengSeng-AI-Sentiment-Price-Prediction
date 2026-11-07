
import yfinance as yf
import pandas as pd
import os


def collect_news():

    print("Fetching Hang Seng financial news...")

    ticker = yf.Ticker("^HSI")

    try:
        news_items = ticker.news

        headlines = []

        for item in news_items:

            content = item.get("content", item)

            title = content.get("title")

            if title:
                headlines.append({
                    "headline": title,
                    "publisher": content.get("provider", {}).get(
                        "displayName", "Yahoo Finance"
                    )
                })

        if not headlines:
            print("No news headlines found.")
            return []

        os.makedirs("data", exist_ok=True)

        df = pd.DataFrame(headlines)
        df.to_csv("data/financial_news.csv", index=False)

        print(f"Collected {len(df)} news headlines.")

        print(df.head(10).to_string(index=False))

        return headlines

    except Exception as e:
        print("News collection failed:", e)
        return []


if __name__ == "__main__":
    collect_news()
