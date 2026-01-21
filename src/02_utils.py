# File: 02_utils.py
# Description:
# This module provides utility functions for:
# - Saving evaluation metrics (CSV + JSON)
# - Loading processed or raw data
# - Logging experiment runs for reproducibility and auditing

import pandas as pd
import os
import json

def save_results(metrics: dict, model_name: str, out_dir='results/tables/'):
    """
    Saves the evaluation metrics for a given model to both CSV and JSON formats.

    Parameters:
        metrics (dict): A dictionary of metric names and values (e.g., F1, ROC_AUC).
        model_name (str): The name of the model used for naming the output files.
        out_dir (str): Directory path to store result files.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save as a single-row CSV for tabular tracking
    df = pd.DataFrame([metrics])
    csv_path = os.path.join(out_dir, f"{model_name}_metrics.csv")
    df.to_csv(csv_path, index=False)

    # Save as JSON for human-readable logging or further parsing
    json_path = os.path.join(out_dir, f"{model_name}_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Results saved for model: {model_name}")

def load_data(processed=True):
    """
    Loads either preprocessed or raw dataset depending on the flag.

    Parameters:
        processed (bool): If True, loads data from 'data/processed'. If False, loads raw data.

    Returns:
        X_train, X_test, y_train, y_test: If processed=True
        X, y, None, None: If processed=False (entire raw data)
    """
    if processed:
        # Load pre-split and scaled data
        X_train = pd.read_csv('data/processed/X_train.csv')
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
        y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
        return X_train, X_test, y_train, y_test
    else:
        # Load raw data and split features/labels
        df = pd.read_csv('data/raw/creditcard.csv')
        y = df['Class']
        X = df.drop(columns='Class')
        return X, y, None, None

def log_experiment(name, parameters, metrics, out_dir='results/logs/'):
    """
    Logs the configuration and performance of a model run to a JSON file.

    Parameters:
        name (str): Name or identifier for the model/experiment.
        parameters (dict): Dictionary of model hyperparameters or config values.
        metrics (dict): Dictionary of performance metrics (e.g., F1, ROC_AUC).
        out_dir (str): Directory to save the log file.
    """
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{name}_log.json")

    log = {
        "model": name,
        "parameters": parameters,
        "metrics": metrics
    }

    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)

    print(f"Experiment log saved to {log_path}")
