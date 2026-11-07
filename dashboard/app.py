
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import json
import joblib

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.news_collector import collect_news
from src.news_sentiment import analyze_all_news
from src.sentiment_analyzer import analyze_sentiment

st.set_page_config(
    page_title="Heng Seng AI | Market Intelligence",
    page_icon="📈",
    layout="wide"
)

# ---------------- DESIGN ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: #101d35;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.hero {
    background: linear-gradient(115deg, #10213f, #174b83);
    padding: 30px;
    border-radius: 16px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 30px;
    font-weight: 800;
}

.hero p {
    color: #d9e8fa;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 13px;
    border: 1px solid #e7ebf2;
    min-height: 125px;
}

.metric-label {
    color: #718096;
    font-size: 12px;
    font-weight: 700;
}

.metric-value {
    color: #172b4d;
    font-size: 26px;
    font-weight: 800;
    margin-top: 10px;
}

.metric-note {
    color: #718096;
    font-size: 12px;
    margin-top: 5px;
}

.section-title {
    color: #172b4d;
    font-size: 21px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 12px;
}

.footer {
    text-align: center;
    color: #8290a5;
    padding: 25px;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- DATA LOADING ----------------

@st.cache_data(ttl=300)
def load_stock_data():
    path = os.path.join(PROJECT_ROOT, "data", "hang_seng_data.csv")

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    except Exception:
        df = pd.read_csv(path, index_col=0)

    df = df.loc[:, ~df.columns.duplicated()]
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])

    return df.sort_index()


def load_csv(filename):
    path = os.path.join(PROJECT_ROOT, "data", filename)

    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def load_metrics():
    path = os.path.join(PROJECT_ROOT, "data", "model_metrics.json")

    if os.path.exists(path):
        try:
            with open(path, "r") as file:
                return json.load(file)
        except Exception:
            return {}

    return {}


stock = load_stock_data()
news = load_csv("financial_news.csv")
sentiments = load_csv("sentiment_results.csv")
evaluation = load_csv("model_evaluation.csv")
metrics = load_metrics()


# ---------------- PREDICTION ----------------

predicted_price = None

try:
    model_path = os.path.join(
        PROJECT_ROOT, "models", "hang_seng_model.pkl"
    )

    if os.path.exists(model_path) and not stock.empty:
        model = joblib.load(model_path)

        data = stock.copy()

        data["Daily_Return"] = data["Close"].pct_change()
        data["MA5"] = data["Close"].rolling(5).mean()
        data["MA10"] = data["Close"].rolling(10).mean()
        data["MA20"] = data["Close"].rolling(20).mean()
        data["Volatility"] = data["Daily_Return"].rolling(5).std()

        features = [
            "Open", "High", "Low", "Close", "Volume",
            "Daily_Return", "MA5", "MA10", "MA20", "Volatility"
        ]

        latest = data[features].dropna().iloc[[-1]]
        predicted_price = float(model.predict(latest)[0])

except Exception as error:
    st.sidebar.error(f"Prediction loading error: {error}")


# ---------------- SENTIMENT ----------------

market_score = 50
positive_count = 0
negative_count = 0
neutral_count = 0

if not sentiments.empty and "sentiment" in sentiments.columns:

    sentiments["sentiment"] = (
        sentiments["sentiment"].astype(str).str.lower()
    )

    positive_count = int(
        (sentiments["sentiment"] == "positive").sum()
    )

    negative_count = int(
        (sentiments["sentiment"] == "negative").sum()
    )

    neutral_count = int(
        (sentiments["sentiment"] == "neutral").sum()
    )

    if "confidence" in sentiments.columns:

        values = sentiments["sentiment"].map({
            "positive": 1,
            "neutral": 0,
            "negative": -1
        }).fillna(0)

        weights = pd.to_numeric(
            sentiments["confidence"], errors="coerce"
        ).fillna(0) / 100

        if weights.sum() > 0:
            weighted = (values * weights).sum() / weights.sum()
            market_score = round(50 + 50 * weighted)


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("## 📈 HENG SENG AI")
    st.caption("MARKET INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Market Analysis",
            "Model Performance",
            "News & Sentiment",
            "AI Analyzer"
        ]
    )

    st.divider()

    st.markdown("### ⚙️ Controls")

    if st.button("🔄 Refresh Financial News", use_container_width=True):

        with st.spinner("Collecting and analyzing financial news..."):
            collect_news()
            analyze_all_news()
            st.cache_data.clear()
            st.rerun()

    st.caption("Data: Yahoo Finance")
    st.caption("Sentiment Model: FinBERT")
    st.caption("Prediction Model: Random Forest")


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>Heng Seng AI Market Intelligence</h1>
    <p>AI-powered financial news sentiment analysis and stock price prediction</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# OVERVIEW
# =====================================================

if page == "Overview":

    st.markdown("### Market Overview")

    if stock.empty:
        st.warning("Stock data unavailable. Run python main.py first.")
        st.stop()

    latest_close = float(stock["Close"].iloc[-1])

    previous_close = (
        float(stock["Close"].iloc[-2])
        if len(stock) > 1 else latest_close
    )

    daily_change = latest_close - previous_close

    daily_percent = (
        daily_change / previous_close * 100
        if previous_close else 0
    )

    prediction_text = (
        f"{predicted_price:,.2f}"
        if predicted_price is not None else "Not available"
    )

    columns = st.columns(4)

    cards = [
        ("LATEST HANG SENG CLOSE", f"{latest_close:,.2f}",
         f"{daily_percent:+.2f}% daily change"),

        ("AI PREDICTED NEXT CLOSE", prediction_text,
         "Random Forest estimate"),

        ("AI MARKET SENTIMENT", f"{market_score}/100",
         "FinBERT news indicator"),

        ("NEWS ANALYZED", str(len(sentiments)),
         "Financial headlines")
    ]

    for col, card in zip(columns, cards):

        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{card[0]}</div>
                <div class="metric-value">{card[1]}</div>
                <div class="metric-note">{card[2]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Hang Seng Index Performance</div>',
        unsafe_allow_html=True
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stock.index,
        y=stock["Close"],
        mode="lines",
        name="Closing Price",
        line=dict(color="#2364aa", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(35,100,170,0.08)"
    ))

    fig.update_layout(
        height=370,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Index Value",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:

        st.markdown("### News Sentiment Distribution")

        sentiment_counts = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Count": [
                positive_count,
                negative_count,
                neutral_count
            ]
        })

        fig2 = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            hole=0.58,
            color="Sentiment",
            color_discrete_map={
                "Positive": "#16a36a",
                "Negative": "#e05252",
                "Neutral": "#e5a12b"
            }
        )

        fig2.update_layout(
            height=310,
            template="plotly_white"
        )

        st.plotly_chart(fig2, use_container_width=True)

    with right:

        st.markdown("### AI Market Sentiment")

        st.metric("Sentiment Indicator", f"{market_score}/100")
        st.progress(market_score / 100)

        if market_score > 60:
            st.success("Positive news sentiment")
        elif market_score < 40:
            st.error("Negative news sentiment")
        else:
            st.info("Neutral / balanced news sentiment")

        st.caption(
            "Experimental news sentiment indicator. "
            "It is not a guaranteed market forecast."
        )


# =====================================================
# MARKET ANALYSIS
# =====================================================

elif page == "Market Analysis":

    st.markdown("### 📊 Market Analysis")

    if stock.empty:
        st.warning("No stock data available.")
        st.stop()

    period = st.selectbox(
        "Select historical view",
        ["1 Month", "3 Months", "6 Months", "1 Year", "All Data"]
    )

    days = {
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365
    }

    view = stock.copy()

    if period != "All Data":
        view = view.tail(days[period])

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=view.index,
        open=view["Open"],
        high=view["High"],
        low=view["Low"],
        close=view["Close"],
        name="Hang Seng"
    ))

    fig.update_layout(
        title="Hang Seng Index - Candlestick Chart",
        height=500,
        template="plotly_white",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Historical Market Data")

    st.dataframe(
        stock.tail(20).sort_index(ascending=False),
        use_container_width=True
    )

    if predicted_price is not None:

        current = float(stock["Close"].iloc[-1])
        change = predicted_price - current
        percentage = change / current * 100 if current else 0

        st.markdown("### AI Price Prediction")

        a, b, c = st.columns(3)

        a.metric("Latest Close", f"{current:,.2f}")
        b.metric("Predicted Close", f"{predicted_price:,.2f}")
        c.metric("Predicted Change", f"{percentage:+.2f}%")


# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.markdown("### 🧠 AI Model Performance Evaluation")

    st.write(
        "Evaluation of the Random Forest model using a chronological "
        "80% training and 20% testing split."
    )

    if not metrics or evaluation.empty:

        st.warning(
            "Model evaluation files not found. "
            "Run python src/price_predictor.py first."
        )

    else:

        mae = metrics.get("MAE", 0)
        rmse = metrics.get("RMSE", 0)
        r2 = metrics.get("R2_Score", 0)

        a, b, c = st.columns(3)

        a.metric("Mean Absolute Error", f"{mae:,.2f} points")
        b.metric("Root Mean Squared Error", f"{rmse:,.2f} points")
        c.metric("R² Score", f"{r2:.4f}")

        st.markdown("### Dataset Information")

        d, e = st.columns(2)

        d.metric(
            "Training Samples",
            metrics.get("Training_Samples", 0)
        )

        e.metric(
            "Testing Samples",
            metrics.get("Testing_Samples", 0)
        )

        st.markdown("### Actual vs Predicted Closing Prices")

        evaluation["Date"] = pd.to_datetime(
            evaluation["Date"], errors="coerce"
        )

        evaluation["Actual_Close"] = pd.to_numeric(
            evaluation["Actual_Close"], errors="coerce"
        )

        evaluation["Predicted_Close"] = pd.to_numeric(
            evaluation["Predicted_Close"], errors="coerce"
        )

        evaluation = evaluation.dropna(
            subset=["Date", "Actual_Close", "Predicted_Close"]
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=evaluation["Date"],
            y=evaluation["Actual_Close"],
            mode="lines",
            name="Actual Closing Price",
            line=dict(color="#2364aa", width=2)
        ))

        fig.add_trace(go.Scatter(
            x=evaluation["Date"],
            y=evaluation["Predicted_Close"],
            mode="lines",
            name="Predicted Closing Price",
            line=dict(color="#e39a35", width=2, dash="dash")
        ))

        fig.update_layout(
            height=430,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Closing Price",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Evaluation Results Table")

        st.dataframe(
            evaluation.sort_values("Date", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "MAE and RMSE measure prediction errors in index points. "
            "R² measures how well the model explains variation in the "
            "test data. A negative R² is possible and indicates poor "
            "performance relative to a simple mean prediction baseline."
        )

        st.caption(
            "These results measure historical test performance. "
            "They do not guarantee future prediction accuracy."
        )


# =====================================================
# NEWS AND SENTIMENT
# =====================================================

elif page == "News & Sentiment":

    st.markdown("### 📰 Financial News Intelligence")

    if not sentiments.empty:

        display = sentiments.copy()

        if "confidence" in display.columns:
            display["confidence"] = pd.to_numeric(
                display["confidence"], errors="coerce"
            )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

    elif not news.empty:

        st.dataframe(
            news,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No news available. Use Refresh Financial News in the sidebar."
        )

    st.markdown("### Sentiment Summary")

    a, b, c = st.columns(3)

    a.metric("Positive", positive_count)
    b.metric("Negative", negative_count)
    c.metric("Neutral", neutral_count)


# =====================================================
# AI ANALYZER
# =====================================================

elif page == "AI Analyzer":

    st.markdown("### 🤖 AI Financial News Analyzer")

    st.write(
        "Enter a financial headline to analyze its sentiment using FinBERT."
    )

    headline = st.text_area(
        "Financial news headline",
        placeholder=(
            "Example: Asian stocks rise as technology shares gain..."
        )
    )

    if st.button("Analyze with FinBERT", type="primary"):

        if headline.strip():

            with st.spinner("Analyzing headline with FinBERT..."):

                result = analyze_sentiment(headline.strip())

            sentiment = result["sentiment"].lower()
            confidence = result["confidence"]

            st.markdown("### Analysis Result")

            a, b = st.columns(2)

            a.metric("Predicted Sentiment", sentiment.upper())
            b.metric("Model Confidence", f"{confidence:.2f}%")

            st.progress(min(confidence / 100, 1.0))

            if sentiment == "positive":
                st.success("Positive financial sentiment.")

            elif sentiment == "negative":
                st.error("Negative financial sentiment.")

            else:
                st.info("Neutral financial sentiment.")

        else:

            st.warning("Please enter a headline first.")


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
HENG SENG AI MARKET INTELLIGENCE
<br>
Powered by FinBERT, Random Forest, Python and Streamlit
<br><br>
Academic AI project. Predictions and sentiment scores are experimental
and should not be treated as financial advice.
</div>
""", unsafe_allow_html=True)
