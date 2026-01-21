# File: 01_preprocess.py
# Description:
# This script performs the preprocessing pipeline for the credit card fraud dataset.
# It loads the data, applies standard scaling, splits the dataset into train and test sets,
# and saves both the transformed data and the fitted scaler object for reproducibility.

import pandas as pd
import numpy as np
import os
import sys
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def log(msg):
    """
    Utility logging function to prefix messages with a timestamp.
    Useful for tracking progress during script execution.
    """
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_data(path="data/raw/creditcard.csv", sample=False):
    """
    Loads the credit card dataset from the specified path.
    Optionally samples 10% of the dataset for faster experimentation.

    Parameters:
        path (str): Path to the CSV file.
        sample (bool): Whether to sample 10% of the dataset.

    Returns:
        df (DataFrame): Loaded (and optionally sampled) DataFrame.
    """
    log("Loading raw data...")
    if not os.path.exists(path):
        log(f"File not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    log(f"Data loaded with shape: {df.shape}")

    if sample:
        df = df.sample(frac=0.1, random_state=42)
        log(f"Sampled 10% of the data: New shape {df.shape}")

    return df

def preprocess(df):
    """
    Preprocesses the dataset:
    - Splits into features and labels.
    - Applies StandardScaler to features.
    - Performs stratified train/test split.

    Parameters:
        df (DataFrame): Raw dataset including 'Class' column.

    Returns:
        X_train, X_test, y_train, y_test: Scaled train-test splits.
        scaler: Fitted StandardScaler object.
    """
    log("Starting preprocessing...")
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    log("Preprocessing complete.")
    return X_train, X_test, y_train, y_test, scaler

def save_data(X_train, X_test, y_train, y_test, scaler,
              data_dir="data/processed", scaler_path="models/scaler.pkl"):
    """
    Saves the processed data splits and fitted scaler to disk.

    Parameters:
        X_train, X_test, y_train, y_test: Numpy arrays to save as CSV.
        scaler: Fitted StandardScaler object.
        data_dir (str): Directory to save processed CSVs.
        scaler_path (str): Path to save the scaler object (.pkl).
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    log("Saving processed datasets and scaler object...")
    pd.DataFrame(X_train).to_csv(f"{data_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{data_dir}/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv(f"{data_dir}/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(f"{data_dir}/y_test.csv", index=False)
    joblib.dump(scaler, scaler_path)
    log(f"Data saved to '{data_dir}' and scaler saved to '{scaler_path}'.")

if __name__ == "__main__":
    # Optionally activate sampling mode with `--sample` argument
    sample = "--sample" in sys.argv
    log("Starting preprocessing pipeline...")

    # Load, preprocess, and save data
    df = load_data(sample=sample)
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    save_data(X_train, X_test, y_train, y_test, scaler)

    log("Preprocessing pipeline finished successfully.")
