# File: run_all_experiments.py
# Description:
# This script evaluates multiple classification models across different data balancing strategies
# (Baseline, SMOTE, and Class Weighting) over multiple runs and saves metrics for analysis.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import pandas as pd
import numpy as np
import os
from src import models  # ensure __init__.py exists in src/

# === Create output directories ===
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# === Define all model variants ===
def get_models(random_state):
    return [
        ("lr", LogisticRegression(max_iter=1000, random_state=random_state)),
        ("dt", DecisionTreeClassifier(random_state=random_state)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ("knn", KNeighborsClassifier()),
        ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)),
        ("lgbm", LGBMClassifier(random_state=random_state))
    ]

# === Number of repeated experiment runs ===
n_runs = 2
results = []

# === Main Experiment Loop ===
for run in range(1, n_runs + 1):
    print(f"\n=== Run {run} ===")

    # Reload and split data for each run to ensure variability
    data = pd.read_csv('data/raw/creditcard.csv')
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42 + run
    )

    for model_tag, base_model in get_models(random_state=42 + run):

        # Determine if model supports class weight parameters
        supports_class_weight = 'class_weight' in base_model.get_params()
        supports_scale_pos_weight = 'scale_pos_weight' in base_model.get_params()

        # Define which sampling/weighting variants to test
        variants = ["Baseline", "SMOTE"]
        if supports_class_weight or supports_scale_pos_weight:
            variants.append("class_weight")

        for variant in variants:
            # Assign weighting scheme
            weighting = "Balanced" if variant == "class_weight" else "None"
            params = base_model.get_params().copy()

            # Apply class or scale_pos_weighting if supported
            if 'class_weight' in params:
                params['class_weight'] = 'balanced' if variant == "class_weight" else None
            if 'scale_pos_weight' in params:
                scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
                params['scale_pos_weight'] = scale_weight if variant == "class_weight" else 1

            # Rebuild model with modified parameters
            model_instance = base_model.__class__(**params)

            # Logging
            log_filename = f"{model_tag}_{variant.lower()}_run{run}.log"
            log_path = os.path.join("logs", log_filename)
            with open(log_path, "w") as log_file:
                def log(msg: str):
                    print(msg)
                    log_file.write(msg + "\n")

                log(f"\n===== Training {model_tag.upper()} (Sampling={variant if variant != 'class_weight' else 'None'}, Weighting={weighting}) =====")
                
                use_smote = variant == "SMOTE"

                # Evaluate model with enhanced metric output
                precision, recall, f1, roc_auc = models.evaluate_model(
                    model_instance, X_train, y_train, X_test, y_test, use_smote
                )

                log(f"Precision: {precision:.4f}")
                log(f"Recall: {recall:.4f}")
                log(f"F1 Score: {f1:.4f}")
                log(f"ROC-AUC Score: {roc_auc:.4f}")
                log("=" * 30)

            # Collect results for aggregation
            results.append({
                "Model": model_tag.upper(),
                "Sampling": variant if variant != "class_weight" else "None",
                "Weighting": weighting,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC_AUC": roc_auc,
                "Run": run
            })

# === Save final aggregated metrics to CSV ===
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("results/metrics.csv", index=False)
print("[INFO] Saved metrics to results/metrics.csv")
