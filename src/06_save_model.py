# File: 06_save_model.py
# Description:
# This script trains an XGBoost model on preprocessed training data
# and saves the trained model to disk for later inference.

import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split

# Load processed training features and labels
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# Initialize and train XGBoost classifier
# Note: 'use_label_encoder=False' avoids warnings related to label encoding
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, "models/xgb_model.pkl")
print("XGBoost model saved at models/xgb_model.pkl")
