#!/usr/bin/env python3
"""
baselines_ext.py — 补充基线实验（COURSE_COVERAGE.md Step 1）

新增方法：
  A13  GradientBoostingRegressor（GBDT，Boosting 串行集成）
  A13  AdaBoostRegressor（弱学习器加权集成）
  A3   KNeighborsRegressor（非参数，距离加权）
  A2   BayesianRidge（贝叶斯线性回归，MAP 估计）

与 ensemble.py 完全相同的数据/切分/特征工程，保证可以和主表对比。
结果写入 output_ot_extended/，不改动 output_ot_full_temp_wind/。

运行：python3 baselines_ext.py
依赖：sklearn / pandas / numpy / openpyxl（无 torch 依赖）
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge, RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

_COL_MAP = {
    "time": "统计时间", "OT": "OT",
    "exog_temp": "Exogenous1", "exog_wind": "Exogenous2",
}

def _load_dataframe(data_path):
    """Inline of core.load_dataframe — no torch dependency."""
    df = pd.read_excel(data_path)
    rev = {v: k for k, v in _COL_MAP.items()}
    df = df.rename(columns={o: n for o, n in rev.items() if o in df.columns})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    for c in ["OT", "exog_temp", "exog_wind"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.interpolate(limit=5).bfill().ffill()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = Path("output_ot_extended")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 与 ensemble.py 完全相同的超参设置，保证数据口径一致
LAGS = [1, 2, 3, 6, 12]
ROLL_WINDOW = 3
KEEP_EXOG_LAGS = True
VAL_RATIO = 0.10
TEST_RATIO = 0.20
DATA_PATH = "标注的数据-#67_1.xlsx"

# ─────────────────────────────── 数据准备 ────────────────────────────────────
print("Reading data...")
df = _load_dataframe(DATA_PATH)

df["temp_roll_3"] = df["exog_temp"].rolling(ROLL_WINDOW, min_periods=1).mean()
df["wind_roll_3"] = df["exog_wind"].rolling(ROLL_WINDOW, min_periods=1).mean()
for lag in LAGS:
    df[f"OT_lag_{lag}"] = df["OT"].shift(lag)
    if KEEP_EXOG_LAGS:
        df[f"temp_lag_{lag}"] = df["exog_temp"].shift(lag)
        df[f"wind_lag_{lag}"] = df["exog_wind"].shift(lag)
df = df.dropna().reset_index(drop=True)
print(f"Rows after feature engineering: {len(df)}")

n = len(df)
test_size = int(n * TEST_RATIO)
val_size = int(n * VAL_RATIO)
train_size = n - test_size - val_size
train_df = df.iloc[:train_size].reset_index(drop=True)
val_df   = df.iloc[train_size: train_size + val_size].reset_index(drop=True)
test_df  = df.iloc[train_size + val_size:].reset_index(drop=True)
print(f"Train / Val / Test: {len(train_df)} / {len(val_df)} / {len(test_df)}")

# stacking meta-holdout (80/20 inside train)
train_train_end = int(len(train_df) * 0.8)
train_train_df = train_df.iloc[:train_train_end].reset_index(drop=True)
meta_holdout_df = train_df.iloc[train_train_end:].reset_index(drop=True)

feat_cols = (
    ["exog_temp", "exog_wind", "temp_roll_3", "wind_roll_3"]
    + [f"OT_lag_{l}" for l in LAGS]
    + ([f"temp_lag_{l}" for l in LAGS] + [f"wind_lag_{l}" for l in LAGS]
       if KEEP_EXOG_LAGS else [])
)

# scaler fit only on train_train to avoid leakage
scaler = StandardScaler().fit(train_train_df[feat_cols].values)
X_tt   = scaler.transform(train_train_df[feat_cols].values);  y_tt  = train_train_df["OT"].values
X_meta = scaler.transform(meta_holdout_df[feat_cols].values); y_meta = meta_holdout_df["OT"].values
X_val  = scaler.transform(val_df[feat_cols].values);          y_val  = val_df["OT"].values
X_test = scaler.transform(test_df[feat_cols].values);         y_test = test_df["OT"].values

def metrics(y_true, y_pred, name=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    eps  = 1e-9
    smape = float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100)
    mase  = float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true - y_true.mean()) + eps)))
    bias  = float(np.mean(y_pred - y_true))
    if name:
        print(f"  {name:30s} MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  sMAPE={smape:.1f}%")
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
            "sMAPE(%)": smape, "MASE": mase, "Bias": bias}

# ─────────────────────────── 模型训练与评估 ──────────────────────────────────
# 每个模型在 train_train 上训练，在 meta_holdout 上看验证，在 test 上最终评估
# 最终模型用 train_train + meta_holdout + val 重训（与 ensemble.py 一致）

def retrain_full(model_cls, **kwargs):
    """用全量训练+验证集重训，然后返回 test 预测。"""
    X_full = np.vstack([X_tt, X_meta, X_val])
    y_full = np.concatenate([y_tt, y_meta, y_val])
    m = model_cls(**kwargs)
    m.fit(X_full, y_full)
    return m

records_test   = []   # test-set metrics for all models
records_meta   = []   # meta-holdout predictions for stacking
meta_preds     = {}   # name -> meta-holdout pred array (stacking input)
test_preds     = {}   # name -> test pred array

print("\n=== Training baseline models ===")

# ── A12  RandomForest（Bagging baseline，与主表对齐） ──────────────────────
print("[RF]  RandomForest (Bagging, A11/A12)")
rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
rf.fit(X_tt, y_tt)
meta_preds["RF"] = rf.predict(X_meta)
m_full = retrain_full(RandomForestRegressor, n_estimators=200, random_state=SEED, n_jobs=-1)
test_preds["RF"] = m_full.predict(X_test)
records_test.append(metrics(y_test, test_preds["RF"], "RF"))

# ── A13  GradientBoosting（GBDT，Boosting）────────────────────────────────
print("[GBM] GradientBoostingRegressor (Boosting, A13)")
gbm = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                                 subsample=0.8, random_state=SEED)
gbm.fit(X_tt, y_tt)
meta_preds["GBM"] = gbm.predict(X_meta)
m_gbm = retrain_full(GradientBoostingRegressor, n_estimators=200, max_depth=4,
                      learning_rate=0.1, subsample=0.8, random_state=SEED)
test_preds["GBM"] = m_gbm.predict(X_test)
records_test.append(metrics(y_test, test_preds["GBM"], "GBM"))

# ── A13  AdaBoost（弱学习器加权集成）─────────────────────────────────────
print("[ADA] AdaBoostRegressor (Boosting, A13)")
from sklearn.tree import DecisionTreeRegressor
ada = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=100, learning_rate=0.5, random_state=SEED
)
ada.fit(X_tt, y_tt)
meta_preds["AdaBoost"] = ada.predict(X_meta)
m_ada = retrain_full(
    AdaBoostRegressor,
    estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=100, learning_rate=0.5, random_state=SEED
)
test_preds["AdaBoost"] = m_ada.predict(X_test)
records_test.append(metrics(y_test, test_preds["AdaBoost"], "AdaBoost"))

# ── A3   KNN（非参数，距离加权）──────────────────────────────────────────
print("[KNN] KNeighborsRegressor (non-parametric, A3)")
knn = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn.fit(X_tt, y_tt)
meta_preds["KNN"] = knn.predict(X_meta)
m_knn = retrain_full(KNeighborsRegressor, n_neighbors=10, weights="distance", n_jobs=-1)
test_preds["KNN"] = m_knn.predict(X_test)
records_test.append(metrics(y_test, test_preds["KNN"], "KNN"))

# ── A2   BayesianRidge（MAP 估计，贝叶斯线性回归）───────────────────────
print("[BAY] BayesianRidge (Bayesian/MAP, A2/T1)")
bay = BayesianRidge(max_iter=300)
bay.fit(X_tt, y_tt)
meta_preds["BayesianRidge"] = bay.predict(X_meta)
m_bay = retrain_full(BayesianRidge, max_iter=300)
test_preds["BayesianRidge"] = m_bay.predict(X_test)
records_test.append(metrics(y_test, test_preds["BayesianRidge"], "BayesianRidge"))

# ── SVR（引入作对照，与主实验结果校验）────────────────────────────────
print("[SVR] SVR (grid search, for cross-check)")
tscv = TimeSeriesSplit(n_splits=4)
pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
pg   = {"svr__C": [1, 10, 50], "svr__epsilon": [0.1, 0.5], "svr__gamma": ["scale"]}
gs   = GridSearchCV(pipe, pg, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
gs.fit(train_train_df[feat_cols].values, y_tt)
best_svr = gs.best_estimator_
meta_preds["SVR"] = best_svr.predict(meta_holdout_df[feat_cols].values)
X_comb = np.vstack([train_train_df[feat_cols].values, meta_holdout_df[feat_cols].values, val_df[feat_cols].values])
y_comb = np.concatenate([y_tt, y_meta, y_val])
best_svr.fit(X_comb, y_comb)
test_preds["SVR"] = best_svr.predict(test_df[feat_cols].values)
records_test.append(metrics(y_test, test_preds["SVR"], "SVR"))

# ─────────────────────────────── Stacking ────────────────────────────────────
# meta features: predictions on the meta-holdout split
meta_X = np.column_stack([meta_preds[k] for k in meta_preds])
meta_y = y_meta
test_X_stack = np.column_stack([test_preds[k] for k in meta_preds])
base_names = list(meta_preds.keys())

print("\n=== Stacking — RidgeCV (L2) ===")
stacker_ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
stacker_ridge.fit(meta_X, meta_y)
print(f"  alpha: {stacker_ridge.alpha_:.4f}")
print(f"  coefs: { {k: f'{v:.3f}' for k,v in zip(base_names, stacker_ridge.coef_)} }")
stack_pred_ridge = stacker_ridge.predict(test_X_stack)
test_preds["Stacking_RidgeCV"] = stack_pred_ridge
records_test.append(metrics(y_test, stack_pred_ridge, "Stacking_RidgeCV"))

print("\n=== Stacking — LassoCV (L1, sparse) ===")
stacker_lasso = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5, max_iter=3000)
stacker_lasso.fit(meta_X, meta_y)
nonzero = int((stacker_lasso.coef_ != 0).sum())
print(f"  alpha: {stacker_lasso.alpha_:.4f}  nonzero coefs: {nonzero}/{len(base_names)}")
print(f"  coefs: { {k: f'{v:.3f}' for k,v in zip(base_names, stacker_lasso.coef_)} }")
stack_pred_lasso = stacker_lasso.predict(test_X_stack)
test_preds["Stacking_LassoCV"] = stack_pred_lasso
records_test.append(metrics(y_test, stack_pred_lasso, "Stacking_LassoCV"))

print("\n=== Stacking — ElasticNetCV (L1+L2) ===")
stacker_en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5, max_iter=3000)
stacker_en.fit(meta_X, meta_y)
nonzero_en = int((stacker_en.coef_ != 0).sum())
print(f"  alpha: {stacker_en.alpha_:.4f}  l1_ratio: {stacker_en.l1_ratio_:.2f}  "
      f"nonzero: {nonzero_en}/{len(base_names)}")
print(f"  coefs: { {k: f'{v:.3f}' for k,v in zip(base_names, stacker_en.coef_)} }")
stack_pred_en = stacker_en.predict(test_X_stack)
test_preds["Stacking_ElasticNet"] = stack_pred_en
records_test.append(metrics(y_test, stack_pred_en, "Stacking_ElasticNet"))

# ─────────────────────────────── 保存结果 ────────────────────────────────────
df_results = pd.DataFrame(records_test)
df_results.to_csv(OUT_DIR / "baselines_ext_metrics.csv", index=False)
print(f"\nResults saved to {OUT_DIR}/baselines_ext_metrics.csv")
print(df_results[["Model","MAE","RMSE","R2","sMAPE(%)"]].to_string(index=False))

# 保存各模型在 test 上的预测
pred_df = pd.DataFrame({"OT_true": y_test})
for k, v in test_preds.items():
    pred_df[f"pred_{k}"] = v
pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

# ─────────────────────────── 对比图 ──────────────────────────────────────────
# 误差柱状图（MAE）
df_plot = df_results.sort_values("MAE")
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#2196F3" if "GBM" in m or "AdaBoost" in m
          else "#4CAF50" if "KNN" in m
          else "#FF9800" if "Bayes" in m
          else "#9C27B0" if "Stack" in m
          else "#757575"
          for m in df_plot["Model"]]
bars = ax.bar(df_plot["Model"], df_plot["MAE"], color=colors, edgecolor="white", linewidth=0.5)
ax.set_ylabel("MAE (W)")
ax.set_title("Baselines Extended — Test MAE Comparison")
ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
ax.tick_params(axis="x", rotation=30)
# legend
from matplotlib.patches import Patch
legend_handles = [
    Patch(color="#2196F3", label="A13 Boosting"),
    Patch(color="#4CAF50", label="A3 KNN"),
    Patch(color="#FF9800", label="A2 BayesianRidge"),
    Patch(color="#9C27B0", label="Stacking (A10)"),
    Patch(color="#757575", label="Baseline"),
]
ax.legend(handles=legend_handles, fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "baselines_ext_mae.png", dpi=150)
plt.close()

# 前 500 步预测曲线对比（核心方法）
fig, ax = plt.subplots(figsize=(14, 4))
n_show = min(500, len(y_test))
ax.plot(range(n_show), y_test[:n_show], "k-", lw=1.2, label="True OT", alpha=0.8)
for name, style in [("GBM","#2196F3"), ("KNN","#4CAF50"), ("BayesianRidge","#FF9800"), ("SVR","#757575")]:
    ax.plot(range(n_show), test_preds[name][:n_show], "-", color=style, lw=0.8, alpha=0.7, label=name)
ax.set_xlabel("Test sample index"); ax.set_ylabel("OT (W)")
ax.set_title("Prediction Excerpt — Extended Baselines")
ax.legend(fontsize=8, ncol=5)
plt.tight_layout()
plt.savefig(OUT_DIR / "baselines_ext_excerpt.png", dpi=150)
plt.close()

print(f"\nDone. Figures saved to {OUT_DIR}/")
