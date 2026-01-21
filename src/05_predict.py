# File: 05_predict.py
# Description:
# This script loads a pre-trained model (by default an XGBoost model)
# and generates predictions on a provided test dataset.
# Predictions are saved to `results/predictions.csv`.

import pandas as pd
import joblib
import argparse

def load_model(model_path="models/xgb_model.pkl"):
    """
    Load a pre-trained model from a specified path.

    Parameters:
        model_path (str): File path to the saved model (pickle format).

    Returns:
        Trained model object.
    """
    return joblib.load(model_path)

def predict(model, X):
    """
    Use the trained model to generate predictions.

    Parameters:
        model: Trained model object.
        X (pd.DataFrame): Feature matrix for prediction.

    Returns:
        Predicted labels (array-like).
    """
    return model.predict(X)

if __name__ == "__main__":
    # Command-line interface for specifying input path
    parser = argparse.ArgumentParser(description="Generate predictions using a trained model.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/processed/X_test.csv", 
        help="Path to the input test data CSV file."
    )
    args = parser.parse_args()

    # Load test features
    X_test = pd.read_csv(args.input)

    # Load pre-trained model
    model = load_model()

    # Generate predictions
    y_pred = predict(model, X_test)

    # Save predictions to CSV
    pd.Series(y_pred).to_csv("results/predictions.csv", index=False)
    print("Predictions saved to results/predictions.csv")
