#!/usr/bin/env python3
"""
stacking_comparison.py — Holdout Blending vs OOF Stacking 对比

比较两种 meta-feature 生成策略，配合三种元学习器：

策略 A — Holdout Blending（当前 baselines_ext.py 做法）:
  - 在 train_train（80% of train）上训练基模型
  - 在 meta_holdout（20% of train）上预测，生成元特征
  - 元学习器只能看到 20% 训练数据（约 5,845 行）

策略 B — OOF (Out-of-Fold) Stacking（更标准的 A10）:
  - 对整个训练集做 TimeSeriesSplit(5)
  - 每折用此前数据训练基模型，当前折生成 OOF 预测
  - OOF 预测拼接后等长整个训练集（约 29,225 行）
  - 元学习器获得 5× 更多的元训练数据

元学习器（三种对比）:
  - RidgeCV: L2 正则，权重连续，自动 CV 选 alpha
  - LassoCV: L1 正则，稀疏权重，能将弱基模型权重压为 0
  - ElasticNetCV: L1+L2 混合，兼顾稀疏性和稳定性

基模型: RF, GBM, KNN, BayesianRidge, SVR（与 baselines_ext.py 完全对齐）

无 torch 依赖。
运行：python3 stacking_comparison.py
输出目录：output_stacking_compare/

OOF 失败定量分析（分布 + 前/后 20% 预测质量 + Ridge 消融）：
  python3 oof_meta_shift_analysis.py  →  output_stacking_compare/oof_shift_*
"""

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge, RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = Path("output_stacking_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────── 数据加载（与 baselines_ext.py 相同） ─────────────
_COL_MAP = {"time": "统计时间", "OT": "OT",
            "exog_temp": "Exogenous1", "exog_wind": "Exogenous2"}

def _load(path):
    df = pd.read_excel(path)
    rev = {v: k for k, v in _COL_MAP.items()}
    df = df.rename(columns={o: n for o, n in rev.items() if o in df.columns})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    for c in ["OT", "exog_temp", "exog_wind"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.interpolate(limit=5).bfill().ffill()

def _metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    smape = float(np.mean(np.abs(y_true-y_pred)/(np.abs(y_true)+np.abs(y_pred)+1e-9))*100)
    return dict(MAE=mae, RMSE=rmse, R2=r2, sMAPE=smape)

print("Reading data...")
df = _load("标注的数据-#67_1.xlsx")

LAGS = [1, 2, 3, 6, 12]
df["temp_roll_3"] = df["exog_temp"].rolling(3, min_periods=1).mean()
df["wind_roll_3"] = df["exog_wind"].rolling(3, min_periods=1).mean()
for lag in LAGS:
    df[f"OT_lag_{lag}"]   = df["OT"].shift(lag)
    df[f"temp_lag_{lag}"] = df["exog_temp"].shift(lag)
    df[f"wind_lag_{lag}"] = df["exog_wind"].shift(lag)
df = df.dropna().reset_index(drop=True)
print(f"  Rows: {len(df)}")

FEAT = (["exog_temp","exog_wind","temp_roll_3","wind_roll_3"]
        + [f"OT_lag_{l}" for l in LAGS]
        + [f"temp_lag_{l}" for l in LAGS]
        + [f"wind_lag_{l}" for l in LAGS])

n = len(df)
test_sz  = int(n * 0.20)
val_sz   = int(n * 0.10)
train_sz = n - test_sz - val_sz

train_df = df.iloc[:train_sz].reset_index(drop=True)
val_df   = df.iloc[train_sz: train_sz+val_sz].reset_index(drop=True)
test_df  = df.iloc[train_sz+val_sz:].reset_index(drop=True)
print(f"  Train {train_sz} | Val {val_sz} | Test {test_sz}")

# Holdout split
meta_end  = int(train_sz * 0.8)
tt_df     = train_df.iloc[:meta_end].reset_index(drop=True)   # train_train
meta_df   = train_df.iloc[meta_end:].reset_index(drop=True)   # meta_holdout

# scaler fit only on train_train to prevent leakage
sc = StandardScaler().fit(tt_df[FEAT].values)
X_tt   = sc.transform(tt_df[FEAT].values);   y_tt   = tt_df["OT"].values
X_meta = sc.transform(meta_df[FEAT].values); y_meta = meta_df["OT"].values
X_val  = sc.transform(val_df[FEAT].values);  y_val  = val_df["OT"].values
X_test = sc.transform(test_df[FEAT].values); y_test = test_df["OT"].values

# Full train set (train_train + meta_holdout + val) for final base model refit
X_train_full = np.vstack([X_tt, X_meta, X_val])
y_train_full = np.concatenate([y_tt, y_meta, y_val])

# ───────────────────── 定义基模型 ────────────────────────────────────────────
BASE_MODELS = {
    "RF":  RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
    "GBM": GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                      learning_rate=0.1, subsample=0.8, random_state=SEED),
    "KNN": KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=1),
    "BR":  BayesianRidge(max_iter=300),
}
# SVR requires separate unscaled fit (uses its own internal Pipeline scaler for grid search)
# For simplicity, include SVR with fixed best params (from baselines_ext results)
SVR_PARAMS = {"C": 10, "epsilon": 0.1, "gamma": "scale"}

META_LEARNERS = {
    "RidgeCV":      RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]),
    "LassoCV":      LassoCV(cv=5, max_iter=3000, random_state=SEED),
    "ElasticNetCV": ElasticNetCV(cv=5, max_iter=3000, random_state=SEED),
}
BASE_NAMES = list(BASE_MODELS.keys()) + ["SVR"]

# ───────────── 训练基模型（在全量 train+val 上），得到 test 预测 ───────────────
print("\nFitting final base models on full train+val for test predictions...")

# SVR (already tuned, use fixed params)
svr_pipe = Pipeline([("sc", StandardScaler()), ("svr", SVR(kernel="rbf", **SVR_PARAMS))])
svr_pipe.fit(np.vstack([tt_df[FEAT].values, meta_df[FEAT].values, val_df[FEAT].values]),
             y_train_full)

final_models = {}
for name, model in BASE_MODELS.items():
    from copy import deepcopy
    m = deepcopy(model)
    m.fit(X_train_full, y_train_full)
    final_models[name] = m
final_models["SVR"] = svr_pipe

test_preds_base = {}
for name, m in final_models.items():
    if name == "SVR":
        test_preds_base[name] = m.predict(test_df[FEAT].values)
    else:
        test_preds_base[name] = m.predict(X_test)

# Individual base model metrics (reference)
print("\n=== Base Model Test Metrics ===")
for name, preds in test_preds_base.items():
    m = _metrics(y_test, preds)
    print(f"  {name:15s} MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}")

records = []
for name, preds in test_preds_base.items():
    m = _metrics(y_test, preds)
    records.append({"Strategy": "Base", "MetaLearner": name, **m})

# ─────────────── 策略 A: Holdout Blending ─────────────────────────────────
print("\n" + "="*65)
print("Strategy A — Holdout Blending  (meta features from 20% holdout)")
print("="*65)
print(f"  train_train: {len(tt_df)} rows  |  meta_holdout: {len(meta_df)} rows")

# Train base models on train_train, predict on meta_holdout
holdout_meta_preds = {}
for name, model in BASE_MODELS.items():
    from copy import deepcopy
    m = deepcopy(model)
    m.fit(X_tt, y_tt)
    holdout_meta_preds[name] = m.predict(X_meta)
# SVR on meta_holdout
svr_pipe_tt = Pipeline([("sc", StandardScaler()), ("svr", SVR(kernel="rbf", **SVR_PARAMS))])
svr_pipe_tt.fit(tt_df[FEAT].values, y_tt)
holdout_meta_preds["SVR"] = svr_pipe_tt.predict(meta_df[FEAT].values)

X_holdout_meta = np.column_stack([holdout_meta_preds[n] for n in BASE_NAMES])
X_test_stack   = np.column_stack([test_preds_base[n] for n in BASE_NAMES])

for ml_name, ml_cls in META_LEARNERS.items():
    from copy import deepcopy
    meta_learner = deepcopy(ml_cls)
    meta_learner.fit(X_holdout_meta, y_meta)
    stack_pred = meta_learner.predict(X_test_stack)
    m = _metrics(y_test, stack_pred)

    coefs = meta_learner.coef_ if hasattr(meta_learner, "coef_") else None
    nonzero = int((coefs != 0).sum()) if coefs is not None else None
    label = f"A_{ml_name}"
    records.append({"Strategy": "Holdout_Blending", "MetaLearner": ml_name, **m})
    print(f"  {ml_name:15s}: MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}",
          end="")
    if coefs is not None:
        print(f"  nonzero_coefs={nonzero}/{len(coefs)}", end="")
        print(f"  coefs={dict(zip(BASE_NAMES, [f'{c:.3f}' for c in coefs]))}")
    else:
        print()

# ─────────────── 策略 B: OOF Stacking ────────────────────────────────────
print("\n" + "="*65)
print("Strategy B — OOF Stacking  (meta features from 5-fold TimeSeriesSplit)")
print("="*65)

# Combine train+val as the "training pool" for OOF; then test is truly held out
X_trainval = X_train_full   # shape (train+val, features)
y_trainval = y_train_full
trainval_df = pd.concat([train_df, val_df], ignore_index=True)

tscv = TimeSeriesSplit(n_splits=5)
oof_preds = {name: np.zeros(len(X_trainval)) for name in BASE_NAMES}

print(f"  trainval: {len(X_trainval)} rows — 5-fold OOF")
for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_trainval)):
    Xtr, ytr = X_trainval[tr_idx], y_trainval[tr_idx]
    Xvl       = X_trainval[val_idx]
    tr_rows   = trainval_df.iloc[tr_idx]
    vl_rows   = trainval_df.iloc[val_idx]

    for name, model in BASE_MODELS.items():
        from copy import deepcopy
        m = deepcopy(model)
        m.fit(Xtr, ytr)
        oof_preds[name][val_idx] = m.predict(Xvl)

    # SVR fold (needs raw features for its internal scaler)
    svr_fold = Pipeline([("sc", StandardScaler()), ("svr", SVR(kernel="rbf", **SVR_PARAMS))])
    svr_fold.fit(tr_rows[FEAT].values, ytr)
    oof_preds["SVR"][val_idx] = svr_fold.predict(vl_rows[FEAT].values)

    mae_fold_rf = mean_absolute_error(y_trainval[val_idx], oof_preds["RF"][val_idx])
    print(f"  Fold {fold_idx+1}: val_size={len(val_idx)}  RF_fold_MAE={mae_fold_rf:.2f}")

X_oof_meta  = np.column_stack([oof_preds[n] for n in BASE_NAMES])
X_test_oof_stack = X_test_stack  # same test preds from final-retrained models

for ml_name, ml_cls in META_LEARNERS.items():
    from copy import deepcopy
    meta_learner = deepcopy(ml_cls)
    meta_learner.fit(X_oof_meta, y_trainval)
    stack_pred = meta_learner.predict(X_test_oof_stack)
    m = _metrics(y_test, stack_pred)

    coefs = meta_learner.coef_ if hasattr(meta_learner, "coef_") else None
    nonzero = int((coefs != 0).sum()) if coefs is not None else None
    records.append({"Strategy": "OOF_Stacking", "MetaLearner": ml_name, **m})
    print(f"  {ml_name:15s}: MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}",
          end="")
    if coefs is not None:
        print(f"  nonzero_coefs={nonzero}/{len(coefs)}", end="")
        print(f"  coefs={dict(zip(BASE_NAMES, [f'{c:.3f}' for c in coefs]))}")
    else:
        print()

# ─────────────── Lasso 稀疏性分析 ────────────────────────────────────────────
print("\n" + "="*65)
print("LassoCV Sparsity Analysis — which base models get zeroed out?")
print("="*65)
for strategy, X_meta_mat, y_meta_mat in [
    ("Holdout", X_holdout_meta, y_meta),
    ("OOF",     X_oof_meta,     y_trainval)
]:
    lasso = LassoCV(cv=5, max_iter=3000, random_state=SEED)
    lasso.fit(X_meta_mat, y_meta_mat)
    print(f"\n  [{strategy}] alpha={lasso.alpha_:.4f}")
    for bname, coef in zip(BASE_NAMES, lasso.coef_):
        status = "ZERO" if coef == 0 else f"{coef:+.4f}"
        print(f"    {bname:8s}: {status}")

# ─────────────── 保存结果 ────────────────────────────────────────────────────
df_res = pd.DataFrame(records)
df_res.to_csv(OUT_DIR / "stacking_comparison.csv", index=False)
print(f"\nResults saved to {OUT_DIR}/stacking_comparison.csv")

# ─────────────── 汇总打印 ────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY — All methods sorted by MAE")
print("="*65)
df_show = df_res[df_res["Strategy"] != "Base"].copy()
df_show["Label"] = df_show["Strategy"] + " + " + df_show["MetaLearner"]
df_show = df_show.sort_values("MAE")
print(df_show[["Label","MAE","RMSE","R2"]].to_string(index=False, float_format="%.3f"))

# ─────────────── 图: 各策略 MAE 对比 ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
COLORS_STRAT = {"Holdout_Blending": "#1565C0", "OOF_Stacking": "#E65100"}
HATCH_ML = {"RidgeCV": "", "LassoCV": "//", "ElasticNetCV": "xx"}

for ax, metric in zip(axes, ["MAE", "RMSE"]):
    x = np.arange(len(META_LEARNERS))
    w = 0.35
    strat_labels = list(COLORS_STRAT.keys())
    for i, strat in enumerate(strat_labels):
        vals = [df_res[(df_res["Strategy"]==strat) & (df_res["MetaLearner"]==ml)][metric].values[0]
                for ml in META_LEARNERS]
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=strat.replace("_"," "),
                      color=COLORS_STRAT[strat], alpha=0.8,
                      edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    # Add individual base model reference lines
    base_maes = {n: df_res[(df_res["Strategy"]=="Base") & (df_res["MetaLearner"]==n)][metric].values[0]
                 for n in ["RF","SVR","GBM"]}
    for bname, bval in base_maes.items():
        ax.axhline(bval, ls="--", lw=1, color="gray", alpha=0.7)
    ax.annotate(f"RF={base_maes['RF']:.1f}", xy=(len(META_LEARNERS)-0.5, base_maes["RF"]),
                fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(list(META_LEARNERS.keys()))
    ax.set_ylabel(metric)
    ax.set_title(f"Stacking Strategy Comparison — {metric}")
    ax.legend()

plt.suptitle("Holdout Blending vs OOF Stacking × Meta-Learner", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "stacking_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {OUT_DIR}/stacking_comparison.png")
print(f"\nOutput files in {OUT_DIR}/:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}")
