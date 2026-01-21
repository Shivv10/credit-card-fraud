# File: 07_plot_results.py
# Description:
# This script reads aggregated evaluation metrics from results/metrics.csv,
# produces several comparative visualizations, and writes summary tables.
#
# It expects the CSV to contain at least:
#   Model, Sampling, Weighting, Precision, Recall, F1, ROC_AUC, Run
 
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 
# -------------------------------------------------------------------
# Setup: directories and visual style
# -------------------------------------------------------------------
os.makedirs("results/plots", exist_ok=True)
os.makedirs("tables", exist_ok=True)
sns.set_style("whitegrid")
 
 
def main():
    # ----------------------------------------------------------------
    # 1. Load metrics and normalize column names
    # ----------------------------------------------------------------
    metrics_path = "results/metrics.csv"
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Could not find {metrics_path}")
 
    df = pd.read_csv(metrics_path)
    print(f"[INFO] Loaded {metrics_path}")
    print(df.head())
 
    # Standardize metric column names if there are small variations
    rename_map = {
        "F1 Score": "F1",
        "F1-Score": "F1",
        "ROC AUC": "ROC_AUC",
        "ROC-AUC": "ROC_AUC",
        "AUC": "ROC_AUC",
        "Precision Score": "Precision",
        "Recall Score": "Recall",
    }
    df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns},
        inplace=True,
    )
 
    # Ensure all required columns exist
    required_cols = ["Model", "Sampling", "Weighting", "Run"]
    metric_cols = ["Precision", "Recall", "F1", "ROC_AUC"]
    missing = [c for c in required_cols + metric_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in metrics.csv: {missing}")
 
    # Convert metric columns to numeric
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce")
 
    # Ensure categorical columns are strings and handle missing values
    for col in ["Model", "Sampling", "Weighting"]:
        df[col] = df[col].astype(str).fillna("None")
 
    if df[metric_cols].isnull().values.any():
        print(
            "Warning: Missing values detected in metric columns. "
            "They will be ignored where necessary."
        )
 
    # ----------------------------------------------------------------
    # 2. Aggregate metrics across runs (mean over Run)
    # ----------------------------------------------------------------
    group_cols = ["Model", "Sampling", "Weighting"]
    agg_df = df.groupby(group_cols)[metric_cols].mean(numeric_only=True).reset_index()
    print("\n[INFO] Aggregated metrics (mean over runs):")
    print(agg_df.head())
 
    # ----------------------------------------------------------------
    # 3. Bar plots: F1 and ROC_AUC by Model and Sampling
    # ----------------------------------------------------------------
    for metric in ["F1", "ROC_AUC"]:
        if metric not in agg_df.columns:
            continue
 
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=agg_df, x="Model", y=metric, hue="Sampling")
 
        ax.set_title(f"{metric} Score by Model and Sampling")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
 
        # Annotate bars with values
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(
                    f"{height:.2f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
 
        # Only show legend if there is more than one sampling strategy
        if agg_df["Sampling"].nunique() > 1:
            ax.legend(title="Sampling")
        else:
            ax.get_legend().remove()
 
        plt.tight_layout()
        out_path = f"results/plots/bar_{metric}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[Saved] {out_path}")
 
    # ----------------------------------------------------------------
    # 4. Heatmaps: F1 and ROC_AUC for Model vs (Sampling_Weighting)
    # ----------------------------------------------------------------
    for metric in ["F1", "ROC_AUC"]:
        if metric not in agg_df.columns:
            continue
 
        heat_df = agg_df.copy()
        heat_df["Sampling_Weighting"] = heat_df["Sampling"] + "_" + heat_df["Weighting"]
        pivot = heat_df.pivot(
            index="Model", columns="Sampling_Weighting", values=metric
        )
 
        # Check if there is any non-NaN data to plot
        if pivot.dropna(how="all").empty:
            print(
                f"[Warning] Skipping heatmap for {metric} — no valid data available."
            )
            continue
 
        plt.figure(figsize=(max(8, 1.5 * len(pivot.columns)), 5))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title(f"{metric} Score Heatmap")
        plt.xlabel("Sampling_Weighting")
        plt.ylabel("Model")
        plt.tight_layout()
        out_path = f"results/plots/heatmap_{metric}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[Saved] {out_path}")
 
    # ----------------------------------------------------------------
    # 5. Radar chart: Top 3 configurations by F1
    #    Shows multi-metric profile (Precision, Recall, F1, ROC_AUC)
    # ----------------------------------------------------------------
    radar_metrics = ["Precision", "Recall", "F1", "ROC_AUC"]
    if all(m in agg_df.columns for m in radar_metrics):
        top_df = agg_df.dropna(subset=["F1"]).nlargest(3, "F1")
        top_df = top_df.dropna(subset=radar_metrics)
 
        if not top_df.empty:
            num_metrics = len(radar_metrics)
            angles = [
                n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)
            ]
            angles += angles[:1]
 
            plt.figure(figsize=(6, 6))
            ax = plt.subplot(111, polar=True)
 
            for _, row in top_df.iterrows():
                values = [row[m] for m in radar_metrics]
                values += values[:1]  # close the circle
                label = (
                    f"{row['Model']} "
                    f"(Sampling={row['Sampling']}, Weighting={row['Weighting']})"
                )
                ax.plot(angles, values, label=label)
                ax.fill(angles, values, alpha=0.25)
 
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(radar_metrics)
            ax.set_ylim(0, 1)
            ax.set_title("Top 3 Configurations by F1 (Radar Chart)", y=1.08)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            plt.tight_layout()
            out_path = "results/plots/radar_top_models.png"
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"[Saved] {out_path}")
        else:
            print(
                "[Notice] Radar chart skipped: no valid top configurations "
                "with all metrics present."
            )
    else:
        print(
            "[Notice] Radar chart skipped: some required metrics are missing "
            f"from metrics.csv: {radar_metrics}"
        )
 
    # ----------------------------------------------------------------
    # 6. Stability plots: boxplots of F1 and ROC_AUC across runs
    # ----------------------------------------------------------------
    for metric in ["F1", "ROC_AUC"]:
        if metric not in df.columns:
            continue
 
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(data=df, x="Model", y=metric, hue="Sampling")
        ax.set_title(f"{metric} Stability Across Runs by Model and Sampling")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
 
        if df["Sampling"].nunique() > 1:
            ax.legend(title="Sampling")
        else:
            ax.get_legend().remove()
 
        plt.tight_layout()
        out_path = f"results/plots/box_{metric}_stability.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[Saved] {out_path}")
 
    # ----------------------------------------------------------------
    # 7. Save summary tables
    # ----------------------------------------------------------------
    summary_path = "tables/metrics_summary.csv"
    agg_df.to_csv(summary_path, index=False)
    print(f"[Saved] {summary_path}")
 
    for metric in ["F1", "ROC_AUC"]:
        if metric in agg_df.columns:
            top5 = agg_df.dropna(subset=[metric]).nlargest(5, metric)
            out_path = f"tables/top_5_{metric}.csv"
            top5.to_csv(out_path, index=False)
            print(f"[Saved] {out_path}")
 
    print("All plots and tables successfully generated.")
 
 
if __name__ == "__main__":
    main()