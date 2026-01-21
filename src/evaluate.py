# File: evaluate.py
# Description:
# This module provides evaluation tools for:
# - Threshold tuning for classification
# - Printing performance metrics
# - Plotting precision-recall curves and confusion matrices

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    precision_recall_curve  
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_threshold(y_true, y_probs, thresholds=None):
    """
    Finds the optimal threshold that maximizes F1 score.

    Parameters:
        y_true (array-like): Ground truth binary labels.
        y_probs (array-like): Predicted probabilities for the positive class.
        thresholds (list or array): Thresholds to evaluate. Defaults to 0.1–0.9 with 0.05 step.

    Returns:
        best_thresh (float): Threshold that gave the highest F1 score.
        best_f1 (float): Corresponding best F1 score.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)

    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1

def print_metrics(y_true, y_pred, y_probs):
    """
    Prints classification metrics including F1, ROC-AUC, and PR-AUC.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted class labels.
        y_probs (array-like): Predicted probabilities for positive class.
    """
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"F1-score: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_probs):.4f}")
    print(f"PR-AUC: {average_precision_score(y_true, y_probs):.4f}")

def plot_pr_curve(y_true, y_probs, model_name, out_dir='results/plots/'):
    """
    Saves a precision-recall curve for a given model.

    Parameters:
        y_true (array-like): True labels.
        y_probs (array-like): Predicted probabilities.
        model_name (str): Name of the model (used in plot title and filename).
        out_dir (str): Output directory for saving the plot.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure()
    plt.plot(recall, precision, label=f'{model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision–Recall Curve ({model_name})')
    plt.legend()
    plt.grid(True)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{model_name}_pr_curve.png")
    plt.close()

def plot_conf_matrix(y_true, y_pred, model_name, out_dir='results/plots/'):
    """
    Saves a confusion matrix heatmap for a given model.

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted class labels.
        model_name (str): Model identifier for title and filename.
        out_dir (str): Directory to save the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix – {model_name}')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{model_name}_conf_matrix.png")
    plt.close()
