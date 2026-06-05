#!/usr/bin/env python3
"""
stacking_fair_pool.py

Fair ensemble comparison under one shared base-model pool.

The previous report mixed several ensemble results that used different base
model pools. This script keeps the pool fixed when comparing meta learners and
meta-feature strategies:

  Main pool:     RF + GBM + BayesianRidge + SVR
  Extended pool: Main pool + KNN + AdaBoost

For each pool, it evaluates:
  - Holdout stacking/blending
  - OOF stacking with TimeSeriesSplit, using only rows that truly receive OOF
    predictions
  - RidgeCV / LassoCV / ElasticNetCV / NNLS as meta learners

Outputs are written to output_stacking_fair/.
"""


from pathlib import Path as _Path
import os as _os
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
_os.chdir(_PROJECT_ROOT)

from collections import OrderedDict
from copy import deepcopy
import math
from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import nnls
try:
    from scipy.stats import ttest_rel, wilcoxon
except Exception:
    ttest_rel = None
    wilcoxon = None

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = Path("output_stacking_fair")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = "标注的数据-#67_1.xlsx"
LAGS = [1, 2, 3, 6, 12]
TEST_RATIO = 0.20
VAL_RATIO = 0.10

SVR_PARAMS = dict(kernel="rbf", C=50, epsilon=0.1, gamma="scale")

_COL_MAP = {
    "time": "统计时间",
    "OT": "OT",
    "exog_temp": "Exogenous1",
    "exog_wind": "Exogenous2",
}


def load_dataframe(path):
    df = pd.read_excel(path)
    rev = {v: k for k, v in _COL_MAP.items()}
    df = df.rename(columns={old: new for old, new in rev.items() if old in df.columns})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    for col in ["OT", "exog_temp", "exog_wind"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.interpolate(limit=5).bfill().ffill()


def add_features(df):
    df = df.copy()
    df["temp_roll_3"] = df["exog_temp"].rolling(3, min_periods=1).mean()
    df["wind_roll_3"] = df["exog_wind"].rolling(3, min_periods=1).mean()
    for lag in LAGS:
        df[f"OT_lag_{lag}"] = df["OT"].shift(lag)
        df[f"temp_lag_{lag}"] = df["exog_temp"].shift(lag)
        df[f"wind_lag_{lag}"] = df["exog_wind"].shift(lag)
    return df.dropna().reset_index(drop=True)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    smape = float(np.mean(2.0 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100.0)
    return dict(
        MAE=float(mae),
        RMSE=float(rmse),
        R2=float(r2_score(y_true, y_pred)),
        sMAPE=float(smape),
        Bias=float(np.mean(y_pred - y_true)),
    )


def make_model(name):
    if name == "RF":
        return Pipeline([
            ("sc", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1)),
        ])
    if name == "GBM":
        return Pipeline([
            ("sc", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=SEED,
            )),
        ])
    if name == "BayesianRidge":
        return Pipeline([
            ("sc", StandardScaler()),
            ("model", BayesianRidge(max_iter=300)),
        ])
    if name == "SVR":
        return Pipeline([
            ("sc", StandardScaler()),
            ("model", SVR(**SVR_PARAMS)),
        ])
    if name == "KNN":
        return Pipeline([
            ("sc", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=1)),
        ])
    if name == "AdaBoost":
        return Pipeline([
            ("sc", StandardScaler()),
            ("model", AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=3),
                n_estimators=100, learning_rate=0.5, random_state=SEED,
            )),
        ])
    raise KeyError(name)


def fit_meta(method, x_meta, y_meta, x_test):
    if method == "RidgeCV":
        learner = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        learner.fit(x_meta, y_meta)
        pred = learner.predict(x_test)
        return pred, learner.coef_, float(learner.intercept_)

    if method == "LassoCV":
        learner = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5,
                          max_iter=5000, random_state=SEED)
        learner.fit(x_meta, y_meta)
        pred = learner.predict(x_test)
        return pred, learner.coef_, float(learner.intercept_)

    if method == "ElasticNetCV":
        learner = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5,
                               max_iter=5000, random_state=SEED)
        learner.fit(x_meta, y_meta)
        pred = learner.predict(x_test)
        return pred, learner.coef_, float(learner.intercept_)

    if method == "NNLS":
        weights, _ = nnls(x_meta, y_meta)
        pred = x_test @ weights
        return pred, weights, 0.0

    raise KeyError(method)


def bootstrap_ci(y_true, predictions, b=2000):
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    rows = []
    for name, pred in predictions.items():
        mae_dist = np.empty(b)
        rmse_dist = np.empty(b)
        for i in range(b):
            idx = rng.integers(0, n, size=n)
            mae_dist[i] = mean_absolute_error(y_true[idx], pred[idx])
            rmse_dist[i] = math.sqrt(mean_squared_error(y_true[idx], pred[idx]))
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_true, pred),
            "MAE_CI_lo": np.percentile(mae_dist, 2.5),
            "MAE_CI_hi": np.percentile(mae_dist, 97.5),
            "RMSE": math.sqrt(mean_squared_error(y_true, pred)),
            "RMSE_CI_lo": np.percentile(rmse_dist, 2.5),
            "RMSE_CI_hi": np.percentile(rmse_dist, 97.5),
        })
    return pd.DataFrame(rows)


def pvalue_pair(a, b, kind):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if kind == "t" and ttest_rel is not None:
        return float(ttest_rel(a, b, nan_policy="omit").pvalue)
    if kind == "wilcoxon" and wilcoxon is not None:
        delta = a - b
        if np.allclose(delta, 0):
            return 1.0
        return float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
    delta = a - b
    sd = np.std(delta, ddof=1)
    if sd < 1e-12:
        return 1.0
    z = abs(np.mean(delta) / (sd / math.sqrt(len(delta))))
    return float(math.erfc(z / math.sqrt(2.0)))


def paired_tests(y_true, predictions, pairs):
    rows = []
    errors = {name: np.abs(y_true - pred) for name, pred in predictions.items()}
    for left, right in pairs:
        if left not in errors or right not in errors:
            continue
        err_left = errors[left]
        err_right = errors[right]
        rows.append({
            "ModelA": left,
            "ModelB": right,
            "MeanAbsErr_A": float(np.mean(err_left)),
            "MeanAbsErr_B": float(np.mean(err_right)),
            "B_minus_A": float(np.mean(err_right - err_left)),
            "Paired_t_p": pvalue_pair(err_left, err_right, "t"),
            "Nonparametric_test": "Wilcoxon" if wilcoxon is not None else "normal approximation",
            "Nonparametric_p": pvalue_pair(err_left, err_right, "wilcoxon"),
        })
    return pd.DataFrame(rows)


def label(pool_name, strategy, method):
    return f"{pool_name}_{strategy}_{method}"


print("Reading data...")
df = add_features(load_dataframe(DATA_PATH))

feature_cols = (
    ["exog_temp", "exog_wind", "temp_roll_3", "wind_roll_3"]
    + [f"OT_lag_{lag}" for lag in LAGS]
    + [f"temp_lag_{lag}" for lag in LAGS]
    + [f"wind_lag_{lag}" for lag in LAGS]
)

n = len(df)
test_size = int(n * TEST_RATIO)
val_size = int(n * VAL_RATIO)
train_size = n - test_size - val_size

train_df = df.iloc[:train_size].reset_index(drop=True)
val_df = df.iloc[train_size: train_size + val_size].reset_index(drop=True)
test_df = df.iloc[train_size + val_size:].reset_index(drop=True)
trainval_df = pd.concat([train_df, val_df], ignore_index=True)

meta_start = int(len(train_df) * 0.8)
train_train_df = train_df.iloc[:meta_start].reset_index(drop=True)
meta_df = train_df.iloc[meta_start:].reset_index(drop=True)

X_tt = train_train_df[feature_cols].values
y_tt = train_train_df["OT"].values
X_meta_raw = meta_df[feature_cols].values
y_meta = meta_df["OT"].values
X_trainval = trainval_df[feature_cols].values
y_trainval = trainval_df["OT"].values
X_test = test_df[feature_cols].values
y_test = test_df["OT"].values

print(f"Rows: total={len(df)} train={len(train_df)} val={len(val_df)} test={len(test_df)}")
print(f"Holdout: train_train={len(train_train_df)} meta_holdout={len(meta_df)}")

base_names = ["RF", "GBM", "BayesianRidge", "SVR", "KNN", "AdaBoost"]
pools = OrderedDict([
    ("StrongPool", ["RF", "GBM", "BayesianRidge", "SVR"]),
    ("ExtendedPool", ["RF", "GBM", "BayesianRidge", "SVR", "KNN", "AdaBoost"]),
])
meta_methods = ["RidgeCV", "LassoCV", "ElasticNetCV", "NNLS"]

print("\nFitting final base models on train+val...")
final_test_preds = {}
base_rows = []
for name in base_names:
    model = make_model(name)
    model.fit(X_trainval, y_trainval)
    pred = model.predict(X_test)
    final_test_preds[name] = pred
    rec = {"Model": name, **metrics(y_test, pred)}
    base_rows.append(rec)
    print(f"  {name:15s} MAE={rec['MAE']:.2f} RMSE={rec['RMSE']:.2f} R2={rec['R2']:.4f}")

pd.DataFrame(base_rows).to_csv(OUT_DIR / "base_metrics.csv", index=False)

print("\nBuilding holdout meta features...")
holdout_preds = {}
for name in base_names:
    model = make_model(name)
    model.fit(X_tt, y_tt)
    holdout_preds[name] = model.predict(X_meta_raw)
    print(f"  {name:15s} meta MAE={mean_absolute_error(y_meta, holdout_preds[name]):.2f}")

print("\nBuilding corrected OOF meta features...")
oof_preds = {name: np.full(len(y_trainval), np.nan) for name in base_names}
tscv = TimeSeriesSplit(n_splits=5)
for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_trainval), 1):
    print(f"  Fold {fold}: train={len(tr_idx)} val={len(val_idx)}")
    for name in base_names:
        model = make_model(name)
        model.fit(X_trainval[tr_idx], y_trainval[tr_idx])
        oof_preds[name][val_idx] = model.predict(X_trainval[val_idx])

oof_mask = np.ones(len(y_trainval), dtype=bool)
for name in base_names:
    oof_mask &= np.isfinite(oof_preds[name])
print(f"  OOF rows used for meta learner: {oof_mask.sum()} / {len(oof_mask)}")

records = []
coef_records = []
prediction_df = pd.DataFrame({"OT_true": y_test})
for name, pred in final_test_preds.items():
    prediction_df[f"pred_{name}"] = pred

for pool_name, pool in pools.items():
    print(f"\n=== {pool_name}: {', '.join(pool)} ===")

    x_holdout = np.column_stack([holdout_preds[name] for name in pool])
    x_oof = np.column_stack([oof_preds[name][oof_mask] for name in pool])
    y_oof = y_trainval[oof_mask]
    x_test_stack = np.column_stack([final_test_preds[name] for name in pool])

    for strategy, x_meta_for_strategy, y_meta_for_strategy in [
        ("Holdout", x_holdout, y_meta),
        ("OOF", x_oof, y_oof),
    ]:
        for method in meta_methods:
            pred, coefs, intercept = fit_meta(method, x_meta_for_strategy, y_meta_for_strategy, x_test_stack)
            rec = {
                "Pool": pool_name,
                "Strategy": strategy,
                "MetaLearner": method,
                "BaseModels": "+".join(pool),
                "MetaRows": len(y_meta_for_strategy),
                **metrics(y_test, pred),
            }
            records.append(rec)
            out_label = label(pool_name, strategy, method)
            prediction_df[f"pred_{out_label}"] = pred
            for base_name, coef in zip(pool, coefs):
                coef_records.append({
                    "Pool": pool_name,
                    "Strategy": strategy,
                    "MetaLearner": method,
                    "BaseModel": base_name,
                    "Coefficient": float(coef),
                    "Intercept": intercept,
                })
            print(f"  {strategy:8s} {method:12s} MAE={rec['MAE']:.2f} RMSE={rec['RMSE']:.2f} "
                  f"R2={rec['R2']:.4f}")

results_df = pd.DataFrame(records).sort_values(["Pool", "MAE"]).reset_index(drop=True)
coef_df = pd.DataFrame(coef_records)
results_df.to_csv(OUT_DIR / "stacking_fair_results.csv", index=False)
coef_df.to_csv(OUT_DIR / "stacking_fair_coefficients.csv", index=False)
prediction_df.to_csv(OUT_DIR / "stacking_fair_predictions.csv", index=False)

report_preds = {
    "RF": final_test_preds["RF"],
    "GBM": final_test_preds["GBM"],
    "BayesianRidge": final_test_preds["BayesianRidge"],
    "SVR": final_test_preds["SVR"],
}
for strategy in ["Holdout", "OOF"]:
    for method in meta_methods:
        out_label = label("StrongPool", strategy, method)
        report_preds[out_label] = prediction_df[f"pred_{out_label}"].values

print("\nBootstrap CIs for main fair-pool models...")
ci_df = bootstrap_ci(y_test, report_preds, b=2000)
ci_df.to_csv(OUT_DIR / "stacking_fair_bootstrap_ci.csv", index=False)

pairs = [
    ("RF", "SVR"),
    ("SVR", "StrongPool_Holdout_RidgeCV"),
    ("SVR", "StrongPool_Holdout_LassoCV"),
    ("SVR", "StrongPool_OOF_RidgeCV"),
    ("StrongPool_Holdout_RidgeCV", "StrongPool_OOF_RidgeCV"),
    ("StrongPool_Holdout_NNLS", "StrongPool_OOF_NNLS"),
]
sig_df = paired_tests(y_test, report_preds, pairs)
sig_df.to_csv(OUT_DIR / "stacking_fair_significance_tests.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
plot_df = results_df[results_df["Pool"] == "StrongPool"].copy()
plot_df["Label"] = plot_df["Strategy"] + "+" + plot_df["MetaLearner"]
plot_df = plot_df.sort_values("MAE")
bars = ax.bar(plot_df["Label"], plot_df["MAE"], color="#4E79A7")
ax.axhline(mean_absolute_error(y_test, final_test_preds["SVR"]), color="#E15759", ls="--", lw=1.5,
           label="SVR")
ax.axhline(mean_absolute_error(y_test, final_test_preds["RF"]), color="#59A14F", ls="--", lw=1.5,
           label="RF")
ax.set_ylabel("MAE (W)")
ax.set_title("Fair Stacking Comparison on the Same Strong Base Pool")
ax.tick_params(axis="x", rotation=30)
ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "stacking_fair_strong_pool_mae.png", dpi=150)
plt.close()

print(f"\nSaved outputs to {OUT_DIR}/")
print(results_df.to_string(index=False, float_format="%.4f"))
