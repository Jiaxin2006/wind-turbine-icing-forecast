#!/usr/bin/env python3
"""
oof_meta_shift_analysis.py — OOF 失败原因的定量分析

对比 train+val 池内「前 20% / 后 20% / 全量」的：
  1) 目标与特征分布（OT、温度、风速）
  2) TimeSeriesSplit OOF 各折基模型预测 MAE
  3) OOF 元特征 vs 全量重训模型在同一段上的预测质量差异
  4) Holdout meta_holdout 与 OOF 末段、测试段的对齐程度

运行：python3 oof_meta_shift_analysis.py
输出：output_stacking_compare/oof_shift_*.csv, oof_shift_*.png
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = Path("output_stacking_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_COL_MAP = {
    "time": "统计时间",
    "OT": "OT",
    "exog_temp": "Exogenous1",
    "exog_wind": "Exogenous2",
}


def _load(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    rev = {v: k for k, v in _COL_MAP.items()}
    df = df.rename(columns={o: n for o, n in rev.items() if o in df.columns})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    for c in ["OT", "exog_temp", "exog_wind"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.interpolate(limit=5).bfill().ffill()


def _metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _segment_indices(n: int, frac: float = 0.2) -> dict[str, slice]:
    k = int(n * frac)
    return {
        "first_20pct": slice(0, k),
        "last_20pct": slice(n - k, n),
        "full": slice(0, n),
    }


def _dist_row(series: pd.Series, label: str, segment: str) -> dict:
    x = series.values.astype(float)
    return {
        "variable": label,
        "segment": segment,
        "n": len(x),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _ks_vs_full(full: np.ndarray, part: np.ndarray) -> tuple[float, float]:
    stat, p = stats.ks_2samp(part, full)
    return float(stat), float(p)


print("Reading data...")
df = _load("标注的数据-#67_1.xlsx")

LAGS = [1, 2, 3, 6, 12]
df["temp_roll_3"] = df["exog_temp"].rolling(3, min_periods=1).mean()
df["wind_roll_3"] = df["exog_wind"].rolling(3, min_periods=1).mean()
for lag in LAGS:
    df[f"OT_lag_{lag}"] = df["OT"].shift(lag)
    df[f"temp_lag_{lag}"] = df["exog_temp"].shift(lag)
    df[f"wind_lag_{lag}"] = df["exog_wind"].shift(lag)
df = df.dropna().reset_index(drop=True)

FEAT = (
    ["exog_temp", "exog_wind", "temp_roll_3", "wind_roll_3"]
    + [f"OT_lag_{l}" for l in LAGS]
    + [f"temp_lag_{l}" for l in LAGS]
    + [f"wind_lag_{l}" for l in LAGS]
)

n = len(df)
test_sz = int(n * 0.20)
val_sz = int(n * 0.10)
train_sz = n - test_sz - val_sz

train_df = df.iloc[:train_sz].reset_index(drop=True)
val_df = df.iloc[train_sz : train_sz + val_sz].reset_index(drop=True)
test_df = df.iloc[train_sz + val_sz :].reset_index(drop=True)

meta_end = int(train_sz * 0.8)
tt_df = train_df.iloc[:meta_end].reset_index(drop=True)
meta_df = train_df.iloc[meta_end:].reset_index(drop=True)

sc = StandardScaler().fit(tt_df[FEAT].values)
X_tt = sc.transform(tt_df[FEAT].values)
y_tt = tt_df["OT"].values
X_meta = sc.transform(meta_df[FEAT].values)
y_meta = meta_df["OT"].values
X_val = sc.transform(val_df[FEAT].values)
y_val = val_df["OT"].values
X_test = sc.transform(test_df[FEAT].values)
y_test = test_df["OT"].values

X_train_full = np.vstack([X_tt, X_meta, X_val])
y_train_full = np.concatenate([y_tt, y_meta, y_val])
trainval_df = pd.concat([train_df, val_df], ignore_index=True)
n_tv = len(trainval_df)

BASE_MODELS = {
    "RF": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
    "GBM": GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=SEED,
    ),
    "KNN": KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=1),
    "BR": BayesianRidge(max_iter=300),
}
SVR_PARAMS = {"C": 10, "epsilon": 0.1, "gamma": "scale"}
BASE_NAMES = list(BASE_MODELS.keys()) + ["SVR"]

# ── 1) 数据分布：前 20% vs 后 20% vs 全量 (train+val) ─────────────────────
print("\n[1] Distribution: first 20% vs last 20% vs full (train+val)")
dist_rows = []
segments = _segment_indices(n_tv, 0.2)
for seg_name, sl in segments.items():
    sub = trainval_df.iloc[sl]
    for var in ["OT", "exog_temp", "exog_wind"]:
        dist_rows.append(_dist_row(sub[var], var, seg_name))

dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(OUT_DIR / "oof_shift_distribution.csv", index=False)

ks_rows = []
full_ot = trainval_df["OT"].values
full_temp = trainval_df["exog_temp"].values
full_wind = trainval_df["exog_wind"].values
for seg_name in ("first_20pct", "last_20pct"):
    sl = segments[seg_name]
    sub = trainval_df.iloc[sl]
    for var, full_arr in [
        ("OT", full_ot),
        ("exog_temp", full_temp),
        ("exog_wind", full_wind),
    ]:
        d, p = _ks_vs_full(full_arr, sub[var].values)
        ks_rows.append(
            {
                "segment": seg_name,
                "variable": var,
                "KS_statistic": d,
                "KS_pvalue": p,
            }
        )
ks_df = pd.DataFrame(ks_rows)
ks_df.to_csv(OUT_DIR / "oof_shift_ks_test.csv", index=False)

# ── 2) OOF 预测 + 各折训练规模 ─────────────────────────────────────────────
print("[2] OOF fold-wise prediction quality")
tscv = TimeSeriesSplit(n_splits=5)
oof_preds = {name: np.zeros(n_tv) for name in BASE_NAMES}
fold_rows = []

for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train_full)):
    Xtr, ytr = X_train_full[tr_idx], y_train_full[tr_idx]
    Xvl = X_train_full[val_idx]
    tr_rows = trainval_df.iloc[tr_idx]
    vl_rows = trainval_df.iloc[val_idx]
    yvl = y_train_full[val_idx]

    fold_preds = {}
    for name, model in BASE_MODELS.items():
        m = deepcopy(model)
        m.fit(Xtr, ytr)
        pred = m.predict(Xvl)
        oof_preds[name][val_idx] = pred
        fold_preds[name] = pred

    svr_fold = Pipeline(
        [("sc", StandardScaler()), ("svr", SVR(kernel="rbf", **SVR_PARAMS))]
    )
    svr_fold.fit(tr_rows[FEAT].values, ytr)
    oof_preds["SVR"][val_idx] = svr_fold.predict(vl_rows[FEAT].values)
    fold_preds["SVR"] = oof_preds["SVR"][val_idx]

    row = {
        "fold": fold_idx + 1,
        "train_rows": len(tr_idx),
        "val_rows": len(val_idx),
        "train_frac_of_tv": len(tr_idx) / n_tv,
    }
    for name in BASE_NAMES:
        m = _metrics(yvl, fold_preds[name])
        row[f"{name}_MAE"] = m["MAE"]
        row[f"{name}_R2"] = m["R2"]
    fold_rows.append(row)

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUT_DIR / "oof_shift_fold_metrics.csv", index=False)

# ── 3) 全量重训模型 vs OOF，按时间段切片 ───────────────────────────────────
print("[3] Full-retrain vs OOF by time segment")
final_models = {}
for name, model in BASE_MODELS.items():
    m = deepcopy(model)
    m.fit(X_train_full, y_train_full)
    final_models[name] = m

svr_pipe = Pipeline([("sc", StandardScaler()), ("svr", SVR(kernel="rbf", **SVR_PARAMS))])
svr_pipe.fit(
    np.vstack([tt_df[FEAT].values, meta_df[FEAT].values, val_df[FEAT].values]),
    y_train_full,
)
final_models["SVR"] = svr_pipe

full_preds = {}
for name, m in final_models.items():
    if name == "SVR":
        full_preds[name] = m.predict(trainval_df[FEAT].values)
    else:
        full_preds[name] = m.predict(X_train_full)

# Holdout blending preds on meta_holdout
holdout_preds = {}
for name, model in BASE_MODELS.items():
    m = deepcopy(model)
    m.fit(X_tt, y_tt)
    holdout_preds[name] = m.predict(X_meta)
svr_tt = Pipeline([("sc", StandardScaler()), ("svr", SVR(kernel="rbf", **SVR_PARAMS))])
svr_tt.fit(tt_df[FEAT].values, y_tt)
holdout_preds["SVR"] = svr_tt.predict(meta_df[FEAT].values)

seg_pred_rows = []
for seg_name, sl in segments.items():
    idx = np.arange(n_tv)[sl]
    y_seg = y_train_full[sl]
    for name in BASE_NAMES:
        oof_m = _metrics(y_seg, oof_preds[name][sl])
        full_m = _metrics(y_seg, full_preds[name][sl])
        seg_pred_rows.append(
            {
                "segment": seg_name,
                "model": name,
                "source": "OOF",
                **oof_m,
                "MAE_ratio_vs_full": oof_m["MAE"] / full_m["MAE"]
                if full_m["MAE"] > 1e-9
                else np.nan,
            }
        )
        seg_pred_rows.append(
            {
                "segment": seg_name,
                "model": name,
                "source": "FullRetrain",
                **full_m,
                "MAE_ratio_vs_full": 1.0,
            }
        )

# meta_holdout 段（与 holdout blending 对齐）
for name in BASE_NAMES:
    mh_sl = slice(meta_end, train_sz)
    hm = _metrics(y_meta, holdout_preds[name])
    fm = _metrics(y_meta, full_preds[name][mh_sl])
    seg_pred_rows.append(
        {
            "segment": "meta_holdout_only",
            "model": name,
            "source": "HoldoutBlend",
            **hm,
            "MAE_ratio_vs_full": hm["MAE"] / fm["MAE"] if fm["MAE"] > 1e-9 else np.nan,
        }
    )
    seg_pred_rows.append(
        {
            "segment": "meta_holdout_only",
            "model": name,
            "source": "FullRetrain_same_rows",
            **fm,
            "MAE_ratio_vs_full": 1.0,
        }
    )

seg_pred_df = pd.DataFrame(seg_pred_rows)
seg_pred_df.to_csv(OUT_DIR / "oof_shift_segment_pred_quality.csv", index=False)

# ── 4) 元特征统计：OOF 前 20% vs 后 20% vs Holdout vs Test ───────────────
print("[4] Meta-feature (base pred) distribution shift")
meta_stat_rows = []
contexts = {
    "OOF_first_20pct": oof_preds,
    "OOF_last_20pct": oof_preds,
    "OOF_full": oof_preds,
    "FullRetrain_trainval": full_preds,
    "Holdout_meta": None,
    "FullRetrain_test": None,
}

for ctx_label, pred_dict in [
    ("OOF_first_20pct", oof_preds),
    ("OOF_last_20pct", oof_preds),
    ("OOF_full", oof_preds),
    ("FullRetrain_trainval", full_preds),
]:
    sl = segments["first_20pct"] if "first" in ctx_label else (
        segments["last_20pct"] if "last" in ctx_label else segments["full"]
    )
    for name in BASE_NAMES:
        p = pred_dict[name][sl]
        meta_stat_rows.append(
            {
                "context": ctx_label,
                "base_model": name,
                "pred_mean": float(np.mean(p)),
                "pred_std": float(np.std(p)),
                "pred_median": float(np.median(p)),
                "residual_std": float(np.std(y_train_full[sl] - p)),
            }
        )

for name in BASE_NAMES:
    p = holdout_preds[name]
    meta_stat_rows.append(
        {
            "context": "Holdout_meta",
            "base_model": name,
            "pred_mean": float(np.mean(p)),
            "pred_std": float(np.std(p)),
            "pred_median": float(np.median(p)),
            "residual_std": float(np.std(y_meta - p)),
        }
    )
    if name == "SVR":
        tp = svr_pipe.predict(test_df[FEAT].values)
    else:
        tp = final_models[name].predict(X_test)
    meta_stat_rows.append(
        {
            "context": "FullRetrain_test",
            "base_model": name,
            "pred_mean": float(np.mean(tp)),
            "pred_std": float(np.std(tp)),
            "pred_median": float(np.median(tp)),
            "residual_std": float(np.std(y_test - tp)),
        }
    )

meta_stat_df = pd.DataFrame(meta_stat_rows)
meta_stat_df.to_csv(OUT_DIR / "oof_shift_meta_feature_stats.csv", index=False)

# KS: OOF RF meta vs test RF meta
oof_rf = oof_preds["RF"]
test_rf = final_models["RF"].predict(X_test)
ks_meta = []
for seg_name, sl in segments.items():
    if seg_name == "full":
        continue
    d, p = stats.ks_2samp(oof_rf[sl], test_rf)
    ks_meta.append(
        {
            "comparison": f"OOF_RF_{seg_name}_vs_test",
            "KS_statistic": d,
            "KS_pvalue": p,
        }
    )
d, p = stats.ks_2samp(holdout_preds["RF"], test_rf)
ks_meta.append(
    {
        "comparison": "Holdout_RF_meta_vs_test",
        "KS_statistic": d,
        "KS_pvalue": p,
    }
)
d, p = stats.ks_2samp(full_preds["RF"][segments["last_20pct"]], test_rf)
ks_meta.append(
    {
        "comparison": "FullRetrain_RF_last20pct_vs_test",
        "KS_statistic": d,
        "KS_pvalue": p,
    }
)
pd.DataFrame(ks_meta).to_csv(OUT_DIR / "oof_shift_meta_ks.csv", index=False)

# ── 5) 消融：仅用 OOF 后 20% 训 Ridge vs 全量 OOF vs Holdout ───────────────
print("[5] RidgeCV ablation on OOF subsets")
X_oof = np.column_stack([oof_preds[n] for n in BASE_NAMES])
X_test_stack = np.column_stack(
    [
        final_models[n].predict(X_test) if n != "SVR" else svr_pipe.predict(test_df[FEAT].values)
        for n in BASE_NAMES
    ]
)
X_holdout_meta = np.column_stack([holdout_preds[n] for n in BASE_NAMES])

ablation_rows = []
for label, sl_fit, X_fit, y_fit in [
    ("OOF_all", slice(0, n_tv), X_oof, y_train_full),
    ("OOF_last_20pct_only", segments["last_20pct"], X_oof[segments["last_20pct"]], y_train_full[segments["last_20pct"]]),
    ("OOF_first_20pct_only", segments["first_20pct"], X_oof[segments["first_20pct"]], y_train_full[segments["first_20pct"]]),
    ("Holdout_meta", slice(0, len(y_meta)), X_holdout_meta, y_meta),
]:
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(X_fit, y_fit)
    pred = ridge.predict(X_test_stack)
    ablation_rows.append({"meta_train_set": label, "n_meta_rows": len(y_fit), **_metrics(y_test, pred)})

pd.DataFrame(ablation_rows).to_csv(OUT_DIR / "oof_shift_ridge_ablation.csv", index=False)

# ── 6) 图表 ────────────────────────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, var, title in zip(
    axes,
    ["OT", "exog_temp", "exog_wind"],
    ["OT (kW)", "Temperature", "Wind speed"],
):
    for seg_name, color in [
        ("first_20pct", "#E65100"),
        ("last_20pct", "#1565C0"),
        ("full", "#9E9E9E"),
    ]:
        sl = segments[seg_name]
        x = trainval_df[var].iloc[sl].values
        ax.hist(x, bins=40, alpha=0.45 if seg_name != "full" else 0.25, density=True, label=seg_name, color=color)
    ax.set_title(title)
    ax.legend(fontsize=7)
fig.suptitle("Train+Val distribution: first 20% vs last 20% vs full", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "oof_shift_distribution_hist.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
x_fold = fold_df["fold"].values
ax.plot(x_fold, fold_df["RF_MAE"], "o-", label="RF", color="#2E7D32")
ax.plot(x_fold, fold_df["GBM_MAE"], "s-", label="GBM", color="#1565C0")
ax.plot(x_fold, fold_df["SVR_MAE"], "^-", label="SVR", color="#C62828")
ax2 = ax.twinx()
ax2.bar(x_fold, fold_df["train_rows"] / 1000, alpha=0.2, color="gray", label="train size (k)")
ax.set_xlabel("TimeSeriesSplit fold (expanding window)")
ax.set_ylabel("Validation MAE (W)")
ax2.set_ylabel("Training rows (×1000)")
ax.set_title("OOF fold: training size grows → validation MAE degrades")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUT_DIR / "oof_shift_fold_mae.png", dpi=150, bbox_inches="tight")
plt.close(fig)

rf_seg = seg_pred_df[seg_pred_df["model"] == "RF"]
labels = ["first_20pct", "last_20pct", "meta_holdout_only", "full"]
sources = ["OOF", "FullRetrain", "HoldoutBlend"]
fig, ax = plt.subplots(figsize=(9, 4))
width = 0.25
xpos = np.arange(len(labels))
for i, src in enumerate(sources):
    sub = rf_seg[(rf_seg["source"] == src) & (rf_seg["segment"].isin(labels))]
    mae_vals = []
    for lab in labels:
        row = sub[sub["segment"] == lab]
        mae_vals.append(row["MAE"].iloc[0] if len(row) else np.nan)
    ax.bar(xpos + i * width, mae_vals, width, label=src)
ax.set_xticks(xpos + width)
ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("MAE (W)")
ax.set_title("RF prediction quality by segment & training regime")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "oof_shift_rf_segment_mae.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 打印摘要
print("\n=== Summary ===")
print(dist_df.pivot_table(index="segment", columns="variable", values="mean").round(2))
print("\nFold RF MAE:", fold_df[["fold", "train_rows", "RF_MAE"]].to_string(index=False))
early = seg_pred_df[(seg_pred_df["model"] == "RF") & (seg_pred_df["segment"] == "first_20pct")]
print("\nRF MAE first 20% — OOF vs FullRetrain:")
print(early[["source", "MAE"]].to_string(index=False))
print("\nRidge ablation on test:")
print(pd.DataFrame(ablation_rows)[["meta_train_set", "n_meta_rows", "MAE", "R2"]].to_string(index=False))
print(f"\nOutputs written to {OUT_DIR}/")
