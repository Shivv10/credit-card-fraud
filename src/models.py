# File: models.py
# Description:
# This module defines the core evaluation logic for training and evaluating classifiers.
# It includes support for optional SMOTE, threshold tuning, and logging of key metrics.

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from src import evaluate  # from 04_evaluate.py

def evaluate_model(model, X_train, y_train, X_test, y_test, use_smote=False):
    """
    Trains and evaluates a classification model with optional SMOTE.

    Parameters:
        model: A scikit-learn-compatible classifier.
        X_train (DataFrame): Training feature set.
        y_train (Series): Training labels.
        X_test (DataFrame): Test feature set.
        y_test (Series): Test labels.
        use_smote (bool): Whether to apply SMOTE on training data.

    Returns:
        precision (float): Precision score.
        recall (float): Recall score.
        f1 (float): F1 score.
        roc_auc (float): ROC-AUC score.
    """

    # === Apply SMOTE if requested ===
    if use_smote:
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    # === Train the model ===
    model.fit(X_train_res, y_train_res)

    # === Predict probabilities and apply optimal threshold ===
    y_probs = model.predict_proba(X_test)[:, 1]
    threshold, _ = evaluate.evaluate_threshold(y_test, y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    # === Compute metrics ===
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probs)

    # === Optional: plot precision-recall and confusion matrix ===
    # model_name = type(model).__name__
    # evaluate.plot_pr_curve(y_test, y_probs, model_name)
    # evaluate.plot_conf_matrix(y_test, y_pred, model_name)

    return precision, recall, f1, roc_auc
