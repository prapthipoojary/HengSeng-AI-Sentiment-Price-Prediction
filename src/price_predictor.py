
import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def predict_stock_price():

    print("\nLoading Hang Seng stock data...")

    file_path = "data/hang_seng_data.csv"

    if not os.path.exists(file_path):
        print("Stock data file not found. Run main.py first.")
        return

    # Load stock data
    data = pd.read_csv(
        file_path,
        header=[0, 1],
        index_col=0
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.loc[:, ~data.columns.duplicated()]

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(col in data.columns for col in required):
        print("Required stock columns are missing.")
        print("Available columns:", data.columns.tolist())
        return

    data = data[required].apply(
        pd.to_numeric,
        errors="coerce"
    )

    data = data.dropna(subset=required)
    data = data.sort_index()

    if len(data) < 100:
        print("Not enough historical data for reliable evaluation.")
        return

    # Feature engineering
    data["Daily_Return"] = data["Close"].pct_change()
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA10"] = data["Close"].rolling(10).mean()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["Volatility"] = data["Daily_Return"].rolling(5).std()

    # Next-day closing price target
    data["Target"] = data["Close"].shift(-1)

    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Daily_Return",
        "MA5",
        "MA10",
        "MA20",
        "Volatility"
    ]

    dataset = data.dropna(
        subset=features + ["Target"]
    )

    if len(dataset) < 50:
        print("Not enough usable data for model training.")
        return

    X = dataset[features]
    y = dataset["Target"]

    # Chronological 80/20 split
    split = int(len(dataset) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    print("\nTraining Random Forest model...")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Test predictions
    predictions = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    # Save actual vs predicted values
    evaluation = pd.DataFrame({
        "Date": X_test.index.astype(str),
        "Actual_Close": y_test.values,
        "Predicted_Close": predictions
    })

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    evaluation.to_csv(
        "data/model_evaluation.csv",
        index=False
    )

    # Save evaluation metrics
    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2_Score": float(r2),
        "Training_Samples": int(len(X_train)),
        "Testing_Samples": int(len(X_test))
    }

    with open(
        "data/model_metrics.json",
        "w"
    ) as file:
        json.dump(metrics, file, indent=4)

    # Train and save final model on all available feature rows
    final_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X, y)

    joblib.dump(
        final_model,
        "models/hang_seng_model.pkl"
    )

    # Predict next closing price using latest available features
    latest_features = data[features].dropna().iloc[[-1]]

    next_price = final_model.predict(
        latest_features
    )[0]

    latest_close = float(data["Close"].iloc[-1])

    print("\n========== MODEL PERFORMANCE ==========")

    print("Mean Absolute Error (MAE):", round(mae, 2))
    print("Root Mean Squared Error (RMSE):", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\n========== PRICE PREDICTION ==========")

    print("Latest Closing Price:", round(latest_close, 2))
    print("Predicted Next Closing Price:", round(float(next_price), 2))

    print("\nEvaluation saved to data/model_evaluation.csv")
    print("Metrics saved to data/model_metrics.json")
    print("Model saved to models/hang_seng_model.pkl")

    print("\nMODEL EVALUATION COMPLETED!")


if __name__ == "__main__":
    predict_stock_price()
