# Credit Card Fraud Detection

This project implements a full machine learning pipeline to detect fraudulent credit card transactions. It covers preprocessing, modeling, evaluation, sampling techniques, and visualization of results.

---

## Project Structure

```
credit-card-fraud/
│
├── data/                    # Raw and processed datasets
├── models/                  # Saved model files
├── results/
│   ├── plots/               # Output plots
│   └── tables/              # Summary tables
│
├── src/                     # Source scripts
│   ├── 01_preprocessing.py
│   ├── 02_sampling.py
│   ├── models.py
│   ├── 04_evaluate.py
│   ├── 05_train.py
│   ├── 06_predict.py
│   ├── 07_plot_results.py
│   └── utils.py
│
├── run_all_experiments.py   # Main pipeline runner
├── requirements.txt         # Dependencies
└── README.md
```

---

## Setup

### 1. Create and activate virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```
pip install pandas scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn
```

---

## Running the Full Pipeline

### Important Note on Testing and Reproducibility

The accompanying report is based on a Monte Carlo cross-validation using `n_runs = 10` (meaning the entire pipeline was run 10 times with different random seeds).  
This was essential for producing statistically robust stability analysis (e.g., F1-score variance shown in box plots).

The submitted code (`run_all_experiments.py`) is set to `n_runs = 2` (Line 36) for quicker verification.

⚠️ **Warning:** Changing `n_runs` back to **10** will require ~60 minutes or more depending on hardware.

---

## Execution Steps

### 1. Clean up (Recommended)

```
bash clean_env.sh
```

### 2. Run the pipeline

```
python3 run_all_experiments.py
```

---

## Generating Plots and Tables

To visualize model performance using the generated `results/metrics.csv`:

```
python src/07_plot_results.py
```

### Output Files

**Plots (results/plots/):**
- bar_F1.png
- bar_ROC_AUC.png
- heatmap_F1.png
- heatmap_ROC_AUC.png
- radar_top_models.png

**Tables (tables/):**
- metrics_summary.csv
- top_5_F1.csv (terminal)
- top_5_ROC_AUC.csv (terminal)

---

## Cleaning the Environment

To remove all generated artifacts (models, results, tables, logs):

```
bash clean_env.sh
```

Run this from the project root (`credit-card-fraud/`).

---

## Key Features

- Handles imbalanced data using:
  - SMOTE  
  - Class weighting  
- Evaluates models using:
  - F1 Score  
  - ROC AUC  
  - Precision  
  - Recall  
- Generates precision–recall curves and confusion matrices  
- Saves metrics across multiple runs for robust statistical analysis  

---

## Models Supported

- Logistic Regression (LR)
- Decision Tree (DT)
- Random Forest (RF)
- K-Nearest Neighbors (KNN)
- XGBoost (XGB)
- LightGBM (LGBM)

---

## Results & Conclusion

This study conducted a rigorous comparative analysis using **Stratified Monte Carlo cross-validation ($N=10$ runs)** to ensure statistical stability. The evaluation focused on **F1-Score** and **ROC-AUC** rather than standard accuracy, avoiding the "Accuracy Paradox" common in imbalanced datasets.

### Key Findings

* **Gradient Boosting Dominance:** Tree-based ensemble methods (**XGBoost** and **LightGBM**) significantly outperformed linear baselines. They achieved the highest mean F1-scores ($\approx 0.86$) and ROC-AUC ($\approx 0.98$), validating their status as the state-of-the-art for tabular fraud detection.
* **The "Precision Collapse" in Linear Models:** While **SMOTE** successfully improved Recall (detection rate) across all models, it caused a catastrophic drop in Precision for Logistic Regression ($0.80 \rightarrow 0.44$)[cite: 86, 429]. Linear decision boundaries failed to handle the synthetic noise introduced by SMOTE, resulting in excessive false alarms.
* **Stability & Robustness:** The **SMOTE + XGBoost** configuration demonstrated the highest stability with minimal variance ($\sigma_{F1} \approx 0.01$) across random seeds, making it the most reliable candidate for production deployment.

### Performance Summary (Average of 10 Runs)

| Model | Strategy | Recall | Precision | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | SMOTE | **0.81** | **0.90** | **0.86** |
| **Random Forest** | Baseline | 0.81 | 0.91 | 0.86 |
| **LightGBM** | SMOTE | 0.80 | 0.87 | 0.83 |
| **Logistic Regression** | Baseline | 0.71 | 0.80 | 0.75 |
| **Logistic Regression** | SMOTE | 0.84 | 0.44 | 0.58 |


### Conclusion

The empirical results confirm that while resampling techniques like SMOTE are powerful tools for increasing fraud detection rates (Recall), they must be paired with non-linear classifiers (like XGBoost) to maintain high Precision. The study validates that a **Gradient Boosting framework combined with SMOTE** offers the optimal balance of detection capability and operational efficiency for credit card fraud systems.

---

## Notes

- Metrics are stored in `results/metrics.csv`
- Add `__init__.py` inside `src/` if module imports fail
