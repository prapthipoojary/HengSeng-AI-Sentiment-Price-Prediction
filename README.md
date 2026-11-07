# Heng Seng AI – Sentiment & Price Prediction Agent

## Project Overview
Heng Seng AI is an AI-powered financial market analysis system that collects Hang Seng Index historical data and financial news, analyses news sentiment using FinBERT, and predicts the next closing price using a Random Forest machine learning model.

## Problem Statement
Financial market analysis requires examining large amounts of market data and financial news. This project aims to simplify the process through automated data collection, AI-based sentiment analysis, and machine learning-based price prediction.

## Key Features
- Historical Hang Seng Index data collection.
- Financial news headline collection.
- Positive, negative, and neutral sentiment classification using FinBERT.
- Next-day closing price prediction using Random Forest.
- Model evaluation using MAE, RMSE, and R² score.
- Interactive Streamlit dashboard.
- Manual financial headline analysis.

## Technologies Used
- Python
- Pandas and NumPy
- Yahoo Finance (yfinance)
- Hugging Face Transformers
- FinBERT
- Scikit-learn
- Random Forest Regressor
- Streamlit
- Plotly
- Joblib

## Project Structure
```text
HengSeng-Sentiment-Agent/
├── data/
├── models/
├── src/
├── dashboard/
├── main.py
├── requirements.txt
└── README.md
```

## How to Run

1. Install Python.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the complete pipeline:

```bash
python main.py
```

6. Launch the dashboard:

```bash
python -m streamlit run dashboard/app.py
```

## Model Evaluation
The Random Forest model uses a chronological training and testing split.

Current evaluation results:
- MAE: 240.83 points
- RMSE: 310.59 points
- R² Score: 0.8823

## Limitations
- Financial market predictions are estimates and are not guaranteed.
- News sentiment is currently analysed separately from the price prediction model.
- The system is an academic prototype and is not connected to live trading.

## Disclaimer
This project is developed for educational and internship purposes only. It does not provide financial advice.