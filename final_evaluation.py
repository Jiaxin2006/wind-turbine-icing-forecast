#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Course-report evaluation helper.

This script does not retrain models.  It reads the saved prediction files from
the main ensemble run and rebuilds the tables used in the final report.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_rel, wilcoxon
except Exception:  # scipy is available in the current environment, keep fallback for portability
    ttest_rel = None
    wilcoxon = None


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "final_results"
MAIN_PRED_PATH = ROOT / "output_ot_full_temp_wind" / "wind_model_output_with_OT_predictions_and_ensembles.csv"
GRID_PATH = ROOT / "out_cnn_lstm_grid_search_revised" / "grid_search_results.csv"
CLUSTER_METRICS_PATH = ROOT / "out_cnn_lstm_cluster_1" / "clustering_classification_metrics.csv"

MODEL_COLUMNS = {
    "RandomForest": "pred_rf",
    "SVR": "pred_svr",
    "CNN": "pred_cnn",
    "LSTM": "pred_lstm",
    "Transformer": "pred_tr",
    "Stacking_RidgeCV": "OT_pred_Ensemble_meta",
    "NNLS_weighted": "OT_pred_Ensemble_nnls",
}

PAIRWISE_TESTS = [
    ("RandomForest", "SVR"),
    ("RandomForest", "Stacking_RidgeCV"),
    ("SVR", "Stacking_RidgeCV"),
    ("RandomForest", "CNN"),
    ("Stacking_RidgeCV", "CNN"),
]


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-9
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def mase(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float("nan")
    naive = np.mean(np.abs(np.diff(y_true)))
    if naive < 1e-12:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / naive)


def metrics_for(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = math.sqrt(mse)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE_percent": safe_mape(y_true, y_pred),
        "sMAPE_percent": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred),
        "Bias": float(np.mean(y_pred - y_true)),
        "MedianAE": float(np.median(np.abs(err))),
    }


def pvalue_pair(y_a, y_b, test_name):
    if test_name == "t" and ttest_rel is not None:
        return float(ttest_rel(y_a, y_b, nan_policy="omit").pvalue)
    if test_name == "wilcoxon" and wilcoxon is not None:
        delta = np.asarray(y_a) - np.asarray(y_b)
        if np.allclose(delta, 0):
            return 1.0
        return float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)

    delta = np.asarray(y_a, dtype=float) - np.asarray(y_b, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) < 2:
        return float("nan")

    # Fallbacks avoid adding one more dependency just for the report tables.
    # With n ~= 8k, the normal approximation is fine for a course-level check.
    if test_name == "t":
        sd = np.std(delta, ddof=1)
        if sd < 1e-12:
            return 1.0
        z = abs(np.mean(delta) / (sd / math.sqrt(len(delta))))
        return float(math.erfc(z / math.sqrt(2.0)))

    wins = int(np.sum(delta > 0))
    losses = int(np.sum(delta < 0))
    n = wins + losses
    if n == 0:
        return 1.0
    z = abs(wins - n / 2.0) / math.sqrt(n / 4.0)
    return float(math.erfc(z / math.sqrt(2.0)))


def build_main_tables():
    df = pd.read_csv(MAIN_PRED_PATH)
    y_true = pd.to_numeric(df["OT"], errors="coerce").to_numpy(dtype=float)

    rows = []
    errors = {}
    for model, col in MODEL_COLUMNS.items():
        pred = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(y_true) & np.isfinite(pred)
        rows.append({
            "Model": model,
            "PredictionColumn": col,
            "N": int(keep.sum()),
            **metrics_for(y_true[keep], pred[keep]),
        })
        errors[model] = np.abs(y_true[keep] - pred[keep])

    metrics_df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    sig_rows = []
    for left, right in PAIRWISE_TESTS:
        err_left = errors[left]
        err_right = errors[right]
        n = min(len(err_left), len(err_right))
        err_left = err_left[:n]
        err_right = err_right[:n]
        sig_rows.append({
            "ModelA": left,
            "ModelB": right,
            "N": int(n),
            "MeanAbsErr_A": float(np.mean(err_left)),
            "MeanAbsErr_B": float(np.mean(err_right)),
            "B_minus_A": float(np.mean(err_right - err_left)),
            "Paired_t_p": pvalue_pair(err_left, err_right, "t"),
            "Nonparametric_test": "Wilcoxon" if wilcoxon is not None else "Sign test fallback",
            "Nonparametric_p": pvalue_pair(err_left, err_right, "wilcoxon"),
        })
    sig_df = pd.DataFrame(sig_rows)
    return metrics_df, sig_df


def build_extra_tables():
    extras = []

    if GRID_PATH.exists():
        grid = pd.read_csv(GRID_PATH)
        if not grid.empty and "MAE" in grid.columns:
            best = grid.loc[grid["MAE"].idxmin()]
            extras.append({
                "Experiment": "Best revised CNN-LSTM search",
                "Source": str(GRID_PATH.relative_to(ROOT)),
                "Model": best.get("MODEL_TYPE", "unknown"),
                "MAE": float(best["MAE"]),
                "RMSE": float(best.get("RMSE", np.nan)),
                "R2": float(best.get("R2", np.nan)),
                "Note": "Different saved run/test window; report as supplementary, not paired with the main table.",
            })

    if CLUSTER_METRICS_PATH.exists():
        cluster = pd.read_csv(CLUSTER_METRICS_PATH)
        for _, row in cluster.iterrows():
            extras.append({
                "Experiment": str(row["Method"]),
                "Source": str(CLUSTER_METRICS_PATH.relative_to(ROOT)),
                "Model": "KMeans consistency",
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
                "Note": (
                    f"Accuracy={row['Accuracy']:.3f}, F1={row['F1 score']:.3f}, "
                    f"ARI={row['ARI']:.3f}, NMI={row['NMI']:.3f}"
                ),
            })

    return pd.DataFrame(extras)


def write_figures(metrics_df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip figures: matplotlib is not available ({exc})")
        return []

    plot_df = metrics_df[metrics_df["Model"] != "NNLS_weighted"].copy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(plot_df))
    width = 0.38
    ax.bar(x - width / 2, plot_df["MAE"], width, label="MAE")
    ax.bar(x + width / 2, plot_df["RMSE"], width, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Model"], rotation=25, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("Main test error comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "main_error_comparison.png", dpi=180)
    plt.close(fig)

    pred_df = pd.read_csv(MAIN_PRED_PATH)
    show = pred_df.iloc[:500].copy()
    show["time"] = pd.to_datetime(show["time"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(show["time"], show["OT"], label="True OT", linewidth=1.2)
    ax.plot(show["time"], show["pred_svr"], label="SVR", linewidth=1.0)
    ax.plot(show["time"], show["pred_rf"], label="RandomForest", linewidth=1.0)
    ax.plot(show["time"], show["OT_pred_Ensemble_meta"], label="Stacking", linewidth=1.0)
    ax.set_title("Prediction excerpt on the test split")
    ax.set_ylabel("OT")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "prediction_excerpt.png", dpi=180)
    plt.close(fig)
    return ["main_error_comparison.png", "prediction_excerpt.png"]


def write_markdown(metrics_df, sig_df, extra_df, figures):
    main_cols = ["Model", "N", "MAE", "RMSE", "sMAPE_percent", "MASE", "Bias"]
    sig_cols = ["ModelA", "ModelB", "MeanAbsErr_A", "MeanAbsErr_B", "B_minus_A", "Paired_t_p", "Nonparametric_test", "Nonparametric_p"]

    def md_table(df, cols, float_digits=4):
        if df.empty:
            return "_No rows._"
        rows = []
        rows.append("| " + " | ".join(cols) + " |")
        rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in df[cols].iterrows():
            cells = []
            for col in cols:
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    if np.isnan(val):
                        cells.append("")
                    else:
                        cells.append(f"{val:.{float_digits}f}")
                else:
                    cells.append(str(val))
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join(rows)

    lines = [
        "# Final Evaluation Tables",
        "",
        "These tables are generated from saved prediction files. No model is retrained here.",
        "",
        "## Main Unified Test Table",
        "",
        md_table(metrics_df, main_cols, float_digits=4),
        "",
        "Notes:",
        "",
        "- Lower MAE/RMSE/sMAPE/MASE is better.",
        "- `Bias` is prediction minus ground truth; positive values mean over-prediction on average.",
        "- `NNLS_weighted` is kept for transparency but should be discussed as a failed ensemble attempt.",
        "",
        "Generated figures:" if figures else "Generated figures: skipped because matplotlib is not available in this Python environment.",
        "",
        "\n".join(f"- `{name}`" for name in figures) if figures else "",
        "",
        "## Paired Error Tests",
        "",
        md_table(sig_df, sig_cols, float_digits=6),
        "",
        "Interpretation: `B_minus_A < 0` means ModelB has lower absolute error than ModelA on average. The nonparametric column uses Wilcoxon when scipy is available; otherwise it uses a paired sign-test fallback.",
        "",
        "## Supplementary Experiments",
        "",
        md_table(extra_df, list(extra_df.columns), float_digits=4) if not extra_df.empty else "_No supplementary table found._",
        "",
    ]
    (OUT_DIR / "report_tables.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    metrics_df, sig_df = build_main_tables()
    extra_df = build_extra_tables()

    metrics_df.to_csv(OUT_DIR / "final_metrics.csv", index=False)
    sig_df.to_csv(OUT_DIR / "significance_tests.csv", index=False)
    extra_df.to_csv(OUT_DIR / "supplementary_results.csv", index=False)
    figures = write_figures(metrics_df)
    write_markdown(metrics_df, sig_df, extra_df, figures)

    print(f"Wrote final tables to {OUT_DIR.relative_to(ROOT)}")
    print(metrics_df[["Model", "MAE", "RMSE", "sMAPE_percent", "MASE"]].to_string(index=False))


if __name__ == "__main__":
    main()
