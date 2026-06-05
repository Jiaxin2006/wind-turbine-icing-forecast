#!/usr/bin/env python3
"""
paper_protocol_eval.py — 序列结构扩展实验的统一评估口径

与 cnn_lstm_grid_search.py 的设置一致：
  - 特征：exog_temp, exog_wind, OT_prev（单步滞后）
  - 划分：按时间 70% / 10% / 20%（train/val/test）
  - 输入 StandardScaler、目标 StandardScaler（仅在 train 上 fit）
  - 经典模型在缩放后的表格特征上训练；指标在原始 OT 尺度上计算

输出：
  output_paper_protocol/paper_protocol_metrics.csv
  output_paper_protocol/paper_protocol_comparison.md

深度模型（MLP/CNN/LSTM/CNN_LSTM 等）取自已有 exp_results.csv（网格搜索最优 run），
本脚本仅重跑/校验 RandomForest 与 SVR。

运行：python3 validation/paper_protocol_eval.py
"""


from pathlib import Path as _Path
import os as _os
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
_os.chdir(_PROJECT_ROOT)

import math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
np.random.seed(SEED)

OUT_DIR = Path("output_paper_protocol")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_RATIO = 0.10
TEST_RATIO = 0.20

# 课程报告中使用的序列扩展模型参照结果
MAIN_TEX_REFERENCE = [
    ("RandomForest", 69.049231, 123.591377, 0.979569),
    ("SVR", 96.370367, 141.738074, 0.973128),
    ("LSTM", 73.828310, 145.826290, 0.9768005),
    ("CNN1D", 242.774150, 392.165360, 0.7942627),
    ("Transformer", 230.419400, 397.737020, 0.7883752),
    ("MLP", 73.628830, 141.951540, 0.9780175),
    ("CNN", 72.052780, 139.010650, 0.9789226),
    ("CNN_LSTM", 67.562816, 135.511973, 0.9799663),
    ("CNN_MLP", 82.948150, 151.150230, 0.9694397),
    ("LSTM_MLP", 81.964860, 145.571420, 0.9716540),
    ("LSTM_CNN_MLP", 81.117920, 149.151760, 0.9702425),
    ("CNN_LSTM_MLP", 81.844850, 147.712750, 0.9708139),
]

EXP_RESULTS_CSV = Path("exp_results.csv")


def load_data():
    df_raw = pd.read_excel("标注的数据-#67_1.xlsx")
    col_map = {}
    for k, cands in {
        "time": ["time", "统计时间"],
        "OT": ["OT"],
        "exog_temp": ["Exogenous1", "exog_temp", "temperature"],
        "exog_wind": ["Exogenous2", "exog_wind", "wind_speed"],
    }.items():
        for c in cands:
            if c in df_raw.columns:
                col_map[k] = c
                break
    df = df_raw.rename(columns={v: k for k, v in col_map.items()})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    for c in ["OT", "exog_temp", "exog_wind"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.interpolate(limit=5).bfill().ffill()
    df["OT_prev"] = df["OT"].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df


def metrics_raw(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return dict(MAE=mae, RMSE=rmse, R2=r2)


print("Loading data (paper protocol)...")
df = load_data()
n = len(df)
test_size = int(n * TEST_RATIO)
val_size = int(n * VAL_RATIO)
train_size = n - test_size - val_size

train_idx = np.arange(0, train_size)
val_idx = np.arange(train_size, train_size + val_size)
test_idx = np.arange(train_size + val_size, n)

print(f"  N={n}  train={train_size}  val={val_size}  test={test_size}")

feat_cols = ["exog_temp", "exog_wind", "OT_prev"]
X_train = df.iloc[train_idx][feat_cols].values
X_val = df.iloc[val_idx][feat_cols].values
X_test = df.iloc[test_idx][feat_cols].values
y_train = df.iloc[train_idx]["OT"].values
y_val = df.iloc[val_idx]["OT"].values
y_test = df.iloc[test_idx]["OT"].values

sc_x = StandardScaler().fit(X_train)
sc_y = StandardScaler().fit(y_train.reshape(-1, 1))

X_train_s = sc_x.transform(X_train)
X_val_s = sc_x.transform(X_val)
X_test_s = sc_x.transform(X_test)
# 经典模型：在缩放特征上训练，预测再 inverse 目标（此处直接预测 y，不 scale y）
# 深度模型用 scaled y；RF/SVR 通常对原始 y 回归，与网格中 sklearn 对照一致用原始 y

records = []

# ── Random Forest ──────────────────────────────────────────────────────────
print("\n[RF] RandomForest...")
rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1)
rf.fit(X_train_s, y_train)
pred_rf = rf.predict(X_test_s)
m_rf = metrics_raw(y_test, pred_rf)
records.append(dict(Protocol="paper", Source="rerun", Model="RandomForest", **m_rf))
print(f"  MAE={m_rf['MAE']:.4f}  RMSE={m_rf['RMSE']:.4f}  R2={m_rf['R2']:.6f}")

# ── SVR (TimeSeriesSplit grid on train) ─────────────────────────────────────
print("\n[SVR] GridSearch + refit train+val...")
tscv = TimeSeriesSplit(n_splits=4)
pipe = Pipeline([("sc", StandardScaler()), ("svr", SVR(kernel="rbf"))])
param_grid = {
    "svr__C": [0.1, 1, 10, 50],
    "svr__epsilon": [0.1, 0.5, 1.0],
    "svr__gamma": ["scale", "auto"],
}
gs = GridSearchCV(pipe, param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=1)
gs.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
best_svr = gs.best_estimator_
print(f"  best_params: {gs.best_params_}")
pred_svr = best_svr.predict(X_test)
m_svr = metrics_raw(y_test, pred_svr)
records.append(dict(Protocol="paper", Source="rerun", Model="SVR", **m_svr))
print(f"  MAE={m_svr['MAE']:.4f}  RMSE={m_svr['RMSE']:.4f}  R2={m_svr['R2']:.6f}")

# ── 深度模型：来自 exp_results.csv ───────────────────────────────────────────
if EXP_RESULTS_CSV.exists():
    print("\n[Deep] Loading exp_results.csv (paper grid best runs)...")
    exp = pd.read_csv(EXP_RESULTS_CSV)
    name_map = {
        "mlp": "MLP",
        "cnn": "CNN",
        "lstm": "LSTM",
        "cnn_lstm": "CNN_LSTM",
        "cnn_mlp": "CNN_MLP",
        "lstm_mlp": "LSTM_MLP",
        "lstm_cnn_mlp": "LSTM_CNN_MLP",
        "cnn_lstm_mlp": "CNN_LSTM_MLP",
    }
    for _, row in exp.iterrows():
        mt = str(row["MODEL_TYPE"])
        label = name_map.get(mt, mt)
        records.append(dict(
            Protocol="paper",
            Source="exp_results.csv",
            Model=label,
            MAE=float(row["MAE"]),
            RMSE=float(row["RMSE"]),
            R2=float(row["R2"]),
        ))
else:
    print(f"  [WARN] {EXP_RESULTS_CSV} not found — deep models omitted")

# ── 参照结果列 ─────────────────────────────────────────────────────────────
for name, mae, rmse, r2 in MAIN_TEX_REFERENCE:
    records.append(dict(
        Protocol="paper",
        Source="reference",
        Model=name,
        MAE=mae,
        RMSE=rmse,
        R2=r2,
    ))

df_out = pd.DataFrame(records)
df_out.to_csv(OUT_DIR / "paper_protocol_metrics.csv", index=False)

# 对比表：rerun vs reference
rerun = df_out[df_out["Source"] == "rerun"]
main_ref = df_out[df_out["Source"] == "reference"].drop_duplicates("Model")
lines = [
    "# Paper protocol metrics\n",
    f"Split: train={train_size}, val={val_size}, test={test_size}\n",
    f"Features: {feat_cols}\n\n",
    "## Rerun (this script) vs reference\n\n",
    "| Model | MAE (rerun) | MAE (ref) | RMSE (rerun) | RMSE (ref) |\n",
    "|-------|-------------|-----------|--------------|------------|\n",
]
for model in ["RandomForest", "SVR"]:
    r = rerun[rerun["Model"] == model]
    m = main_ref[main_ref["Model"] == model]
    if len(r) and len(m):
        lines.append(
            f"| {model} | {r['MAE'].values[0]:.2f} | {m['MAE'].values[0]:.2f} | "
            f"{r['RMSE'].values[0]:.2f} | {m['RMSE'].values[0]:.2f} |\n"
        )

lines.append("\n## All models (exp_results + reference)\n\n")
pivot = df_out[df_out["Source"].isin(["rerun", "exp_results.csv", "reference"])].pivot_table(
    index="Model", columns="Source", values="MAE", aggfunc="first"
)
lines.append(pivot.to_string())
lines.append("\n")

(OUT_DIR / "paper_protocol_comparison.md").write_text("".join(lines), encoding="utf-8")

print("\n" + "=" * 60)
print("SUMMARY — paper protocol (prediction layer only)")
print("=" * 60)
show = df_out[df_out["Source"].isin(["rerun", "exp_results.csv"])].sort_values("MAE")
print(show[["Model", "Source", "MAE", "RMSE", "R2"]].to_string(index=False, float_format="%.4f"))
print(f"\nSaved: {OUT_DIR}/paper_protocol_metrics.csv")
