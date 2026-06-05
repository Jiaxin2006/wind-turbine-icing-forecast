#!/usr/bin/env python3
"""
bootstrap_cv_eval.py — COURSE_COVERAGE Step 4: E2 + E3/T2

两部分内容：
  E2  显式 TimeSeriesSplit(5-fold) 交叉验证
      — 对 RF/SVR/GBM/KNN/BayesianRidge 在训练集上做时序 5-fold CV
      — 估计模型在不同时间段上的泛化误差，与测试集结果对比
  E3/T2  Bootstrap 置信区间
      — 来源 1: baselines_ext.py 输出 (RF/GBM/AdaBoost/KNN/BayesianRidge/SVR/Stacking)
      — 来源 2: final_evaluation.py 输出 (CNN/LSTM/Transformer/RF_main/SVR_main/Ensemble_meta/Ensemble_nnls)
      — 对测试集预测结果做 B=2000 次 bootstrap
      — 给出 MAE / RMSE 的 95% CI，量化点估计的不确定性
      — Appendix：展示 Bootstrap 分布直方图

无 torch 依赖。
运行：python3 bootstrap_cv_eval.py
输出目录：output_bootstrap_cv/
"""

import math, random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 42
random.seed(SEED); np.random.seed(SEED)

OUT_DIR = Path("output_bootstrap_cv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── 读取数据 ────────────────────────────────────────
_COL_MAP = {"time": "统计时间", "OT": "OT",
            "exog_temp": "Exogenous1", "exog_wind": "Exogenous2"}

def _load(path):
    df = pd.read_excel(path)
    rev = {v: k for k, v in _COL_MAP.items()}
    df = df.rename(columns={o: n for o, n in rev.items() if o in df.columns})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    for c in ["OT","exog_temp","exog_wind"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.interpolate(limit=5).bfill().ffill()

print("Reading data...")
df = _load("标注的数据-#67_1.xlsx")

LAGS = [1,2,3,6,12]; ROLL = 3
df["temp_roll_3"] = df["exog_temp"].rolling(ROLL, min_periods=1).mean()
df["wind_roll_3"] = df["exog_wind"].rolling(ROLL, min_periods=1).mean()
for lag in LAGS:
    df[f"OT_lag_{lag}"] = df["OT"].shift(lag)
    df[f"temp_lag_{lag}"] = df["exog_temp"].shift(lag)
    df[f"wind_lag_{lag}"] = df["exog_wind"].shift(lag)
df = df.dropna().reset_index(drop=True)

FEAT = (["exog_temp","exog_wind","temp_roll_3","wind_roll_3"]
        + [f"OT_lag_{l}" for l in LAGS]
        + [f"temp_lag_{l}" for l in LAGS]
        + [f"wind_lag_{l}" for l in LAGS])

n = len(df)
test_sz  = int(n * 0.20)
val_sz   = int(n * 0.10)
train_sz = n - test_sz - val_sz

# ───────────────── E2: TimeSeriesSplit 5-fold CV on train set ─────────────
print("\n" + "="*65)
print("E2 — TimeSeriesSplit(5-fold) Cross-Validation on training set")
print("="*65)
print(f"  Train size: {train_sz}  |  Val: {val_sz}  |  Test: {test_sz}")

train_df = df.iloc[:train_sz].reset_index(drop=True)
X_train  = train_df[FEAT].values
y_train  = train_df["OT"].values

tscv = TimeSeriesSplit(n_splits=5)

MODELS_CV = {
    "RF":           RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1),
    "GBM":          GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                              subsample=0.8, random_state=SEED),
    "KNN":          Pipeline([("sc", StandardScaler()),
                               ("knn", KNeighborsRegressor(n_neighbors=10, weights="distance"))]),
    "BayesianRidge":Pipeline([("sc", StandardScaler()), ("br", BayesianRidge(max_iter=300))]),
    "SVR":          Pipeline([("sc", StandardScaler()),
                               ("svr", SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale"))]),
}

cv_records = []
for name, model in MODELS_CV.items():
    maes = []
    rmses = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
        Xv,  yv  = X_train[val_idx], y_train[val_idx]
        model.fit(Xtr, ytr)
        pv = model.predict(Xv)
        mae  = mean_absolute_error(yv, pv)
        rmse = math.sqrt(mean_squared_error(yv, pv))
        maes.append(mae); rmses.append(rmse)
        print(f"  {name:15s} fold {fold+1}: MAE={mae:.2f}  RMSE={rmse:.2f}")
    rec = dict(Model=name,
               CV_MAE_mean=np.mean(maes),   CV_MAE_std=np.std(maes),
               CV_RMSE_mean=np.mean(rmses),  CV_RMSE_std=np.std(rmses))
    cv_records.append(rec)
    print(f"  → {name:15s} CV MAE={rec['CV_MAE_mean']:.2f}±{rec['CV_MAE_std']:.2f}  "
          f"RMSE={rec['CV_RMSE_mean']:.2f}±{rec['CV_RMSE_std']:.2f}")

df_cv = pd.DataFrame(cv_records)
df_cv.to_csv(OUT_DIR / "cv_results.csv", index=False)
print(f"\nSaved to {OUT_DIR}/cv_results.csv")

# ───────────────── E3/T2: Bootstrap 置信区间 ─────────────────────────────
# 来源 1: baselines_ext.py 输出（传统基线）
# 来源 2: final_evaluation.py 输出（序列模型 + 集成）
PRED_CSV_BASE = Path("output_ot_extended") / "test_predictions.csv"
PRED_CSV_MAIN = Path("output_ot_full_temp_wind") / "wind_model_output_with_OT_predictions_and_ensembles.csv"

print("\n" + "="*65)
print("E3/T2 — Bootstrap Confidence Intervals (B=2000)")
print("="*65)

def _run_bootstrap(y_true, pred_dict, B, rng):
    """Return list of CI records and distributions dict."""
    N = len(y_true)
    records = []
    dists   = {}
    for name, y_hat in pred_dict.items():
        mae_b  = np.empty(B)
        rmse_b = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, N, size=N)
            yt, yp = y_true[idx], y_hat[idx]
            mae_b[b]  = mean_absolute_error(yt, yp)
            rmse_b[b] = math.sqrt(mean_squared_error(yt, yp))
        lo_mae,  hi_mae  = np.percentile(mae_b,  [2.5, 97.5])
        lo_rmse, hi_rmse = np.percentile(rmse_b, [2.5, 97.5])
        mae_obs  = mean_absolute_error(y_true, y_hat)
        rmse_obs = math.sqrt(mean_squared_error(y_true, y_hat))
        records.append(dict(
            Model=name,
            MAE=mae_obs,   MAE_CI_lo=lo_mae,   MAE_CI_hi=hi_mae,
            RMSE=rmse_obs, RMSE_CI_lo=lo_rmse, RMSE_CI_hi=hi_rmse,
        ))
        dists[name] = (mae_b, rmse_b)
        print(f"  {name:25s} MAE={mae_obs:.2f} 95%CI=[{lo_mae:.2f},{hi_mae:.2f}]  "
              f"RMSE={rmse_obs:.2f} 95%CI=[{lo_rmse:.2f},{hi_rmse:.2f}]")
    return records, dists

B   = 2000
rng = np.random.default_rng(SEED)
boot_records      = []
boot_distributions = {}

# ── 来源 1: 传统基线 ──────────────────────────────────────────────────────
if PRED_CSV_BASE.exists():
    base_df = pd.read_csv(PRED_CSV_BASE)
    y_true_base = base_df["OT_true"].values
    preds_base  = {col[5:]: base_df[col].values
                   for col in base_df.columns if col.startswith("pred_")}
    print(f"\n--- Source 1: Classical baselines ({PRED_CSV_BASE}) ---")
    recs, dists = _run_bootstrap(y_true_base, preds_base, B, rng)
    boot_records.extend(recs)
    boot_distributions.update(dists)
else:
    print(f"  [WARN] {PRED_CSV_BASE} not found")

# ── 来源 2: 序列模型 + 集成 ────────────────────────────────────────────────
if PRED_CSV_MAIN.exists():
    main_df   = pd.read_csv(PRED_CSV_MAIN)
    y_true_main = main_df["OT"].values
    # Column name → display name mapping for sequence models and ensembles
    SEQ_COL_MAP = {
        "pred_cnn":             "CNN",
        "pred_lstm":            "LSTM",
        "pred_tr":              "Transformer",
        "OT_pred_Ensemble_meta": "Ensemble_meta",
        "OT_pred_Ensemble_nnls": "Ensemble_nnls",
    }
    preds_seq = {disp: main_df[col].values
                 for col, disp in SEQ_COL_MAP.items()
                 if col in main_df.columns}
    print(f"\n--- Source 2: Sequence models + Ensembles ({PRED_CSV_MAIN.name}) ---")
    recs, dists = _run_bootstrap(y_true_main, preds_seq, B, rng)
    boot_records.extend(recs)
    boot_distributions.update(dists)
else:
    print(f"  [WARN] {PRED_CSV_MAIN} not found — sequence model CIs skipped")

if boot_records:
    df_boot = pd.DataFrame(boot_records)
    df_boot.to_csv(OUT_DIR / "bootstrap_ci.csv", index=False)
    print(f"\nSaved to {OUT_DIR}/bootstrap_ci.csv")

# ─────────────── 图 1: CV 误差折线图（均值±标准差）─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
names_cv = df_cv["Model"].tolist()
x = np.arange(len(names_cv))

for ax, col_m, col_s, ylabel, title in zip(
    axes,
    ["CV_MAE_mean", "CV_RMSE_mean"], ["CV_MAE_std", "CV_RMSE_std"],
    ["MAE (W)", "RMSE (W)"],
    ["Cross-Validation MAE (5-fold TimeSeriesSplit)", "Cross-Validation RMSE (5-fold TimeSeriesSplit)"]
):
    means = df_cv[col_m].values
    stds  = df_cv[col_s].values
    bars  = ax.bar(x, means, yerr=stds, capsize=5, color="#90CAF9", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(names_cv, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.bar_label(bars, labels=[f"{m:.0f}±{s:.0f}" for m, s in zip(means, stds)],
                 padding=4, fontsize=8)

plt.suptitle("E2: 5-fold TimeSeriesSplit CV — training set only", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "cv_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# 图 2: CV vs 测试集 MAE 对比（合理性检验：CV 不应过于乐观或悲观）
if boot_records:
    # 取测试集 MAE 的点估计与 CV 均值对比
    test_mae_map = {r["Model"]: r["MAE"] for r in boot_records}
    both = [(r["Model"], r["CV_MAE_mean"], test_mae_map.get(r["Model"], None))
            for r in cv_records if r["Model"] in test_mae_map]
    both = [(n, c, t) for n, c, t in both if t is not None]
    if both:
        names_b, cv_m, test_m = zip(*both)
        x2 = np.arange(len(names_b)); w = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        b1 = ax.bar(x2 - w/2, cv_m,   w, label="CV MAE (train)",   color="#42A5F5", alpha=0.85)
        b2 = ax.bar(x2 + w/2, test_m, w, label="Test MAE (holdout)", color="#EF5350", alpha=0.85)
        ax.set_xticks(x2); ax.set_xticklabels(names_b, rotation=15, ha="right")
        ax.set_ylabel("MAE (W)"); ax.legend()
        ax.set_title("CV MAE vs Test MAE — 检验 CV 是否高估/低估测试性能")
        ax.bar_label(b1, fmt="%.0f", padding=2, fontsize=8)
        ax.bar_label(b2, fmt="%.0f", padding=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "cv_vs_test_mae.png", dpi=150)
        plt.close()

# 图 3: Bootstrap 分布直方图（每个模型一个 panel）
if boot_distributions:
    n_models = len(boot_distributions)
    fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 7))
    if n_models == 1:
        axes = np.array(axes).reshape(2, 1)

    for col_idx, (name, (mae_b, rmse_b)) in enumerate(boot_distributions.items()):
        rec = next(r for r in boot_records if r["Model"] == name)
        for row_idx, (data, obs, lo, hi, label) in enumerate([
            (mae_b,  rec["MAE"],  rec["MAE_CI_lo"],  rec["MAE_CI_hi"],  "MAE (W)"),
            (rmse_b, rec["RMSE"], rec["RMSE_CI_lo"], rec["RMSE_CI_hi"], "RMSE (W)"),
        ]):
            ax = axes[row_idx, col_idx]
            ax.hist(data, bins=50, color="#90CAF9", edgecolor="white", alpha=0.8)
            ax.axvline(obs, color="red", lw=2, label=f"Obs={obs:.1f}")
            ax.axvline(lo,  color="navy", lw=1.5, ls="--", label=f"95% CI")
            ax.axvline(hi,  color="navy", lw=1.5, ls="--")
            ax.set_xlabel(label, fontsize=8)
            ax.set_title(f"{name}\n[{lo:.1f}, {hi:.1f}]", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("Bootstrap count")
            ax.legend(fontsize=7)

    plt.suptitle("E3/T2: Bootstrap Distribution of MAE and RMSE (B=2000)\n"
                 "Dashed lines = 95% Confidence Interval", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bootstrap_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

# ─────────────── 终端摘要 ──────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY: Cross-Validation Results")
print("="*65)
print(df_cv[["Model","CV_MAE_mean","CV_MAE_std","CV_RMSE_mean","CV_RMSE_std"
             ]].to_string(index=False, float_format="%.2f"))

if boot_records:
    print("\n" + "="*65)
    print("SUMMARY: Bootstrap 95% Confidence Intervals (Test set)")
    print("="*65)
    print(df_boot[["Model","MAE","MAE_CI_lo","MAE_CI_hi","RMSE","RMSE_CI_lo","RMSE_CI_hi"
                   ]].to_string(index=False, float_format="%.2f"))

print(f"\nOutput files in {OUT_DIR}/:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}")
