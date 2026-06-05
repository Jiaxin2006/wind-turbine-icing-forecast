"""
基础模型对比：RandomForest、SVR、LSTM、CNN1D、Transformer。
读取 Excel，按时间切分 train/val/test（7:1:2）。
"""

import random
import math
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader

from core import load_dataframe, metric_table, SeqDataset

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

OUT_DIR = Path("output_ot_models-new")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VAL_RATIO = 0.1

TEST_RATIO = 0.20
SEQ_LEN = 12
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据（列名映射、排序、数值化、插值见 core.load_dataframe）
data_path = "标注的数据-#67_1.xlsx"
df = load_dataframe(data_path, required=["OT", "exog_temp", "exog_wind"])

# 滚动均值 + 滞后特征（经典模型和后续分析会用到）
df["temp_roll_3"] = df["exog_temp"].rolling(3, min_periods=1).mean()
df["wind_roll_3"] = df["exog_wind"].rolling(3, min_periods=1).mean()

lags = [1, 2, 3, 6, 12]
for lag in lags:
    df[f"OT_lag_{lag}"] = df["OT"].shift(lag)
    df[f"temp_lag_{lag}"] = df["exog_temp"].shift(lag)
    df[f"wind_lag_{lag}"] = df["exog_wind"].shift(lag)

df = df.dropna().reset_index(drop=True)

# 按时间顺序切分，避免未来信息泄露
n = len(df)
test_size = int(n * TEST_RATIO)
val_size = int(n * VAL_RATIO)
train_df = df.iloc[:-test_size - val_size].copy()
val_df = df.iloc[-test_size - val_size:-test_size].copy()
test_df = df.iloc[-test_size:].copy()

# 经典模型只用温度、风速和 OT 的一阶滞后
feature_cols_classical = ["exog_temp", "exog_wind", "OT_lag_1"]
X_train_cl = train_df[feature_cols_classical].values
X_val_cl = val_df[feature_cols_classical].values
X_test_cl = test_df[feature_cols_classical].values
y_train = train_df["OT"].values
y_val = val_df["OT"].values
y_test = test_df["OT"].values

# scaler 只在训练集上 fit
scaler_cl = StandardScaler()
X_train_cl_s = scaler_cl.fit_transform(X_train_cl)
X_val_cl_s = scaler_cl.transform(X_val_cl)
X_test_cl_s = scaler_cl.transform(X_test_cl)

# 无监督聚类：KMeans 只在训练集上 fit，再 predict 全量数据
cluster_features = ["exog_temp", "exog_wind"]
scaler_cluster = StandardScaler()
Xc_train = scaler_cluster.fit_transform(train_df[cluster_features].values)
kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=20)
kmeans.fit(Xc_train)
labels = kmeans.predict(scaler_cluster.transform(df[cluster_features].values))
df["cluster"] = labels

train_cluster_labels = kmeans.predict(scaler_cluster.transform(train_df[cluster_features].values))
cluster_summary = (
    train_df.assign(cluster=train_cluster_labels)
    .groupby("cluster")[["exog_temp", "exog_wind"]]
    .mean()
)
# 温度更低的簇标为 icing_or_off
cluster_low = cluster_summary["exog_temp"].idxmin()
label_map = {cluster_low: "icing_or_off", 1 - cluster_low: "normal"}
df["cluster_label"] = df["cluster"].map(label_map)

results = {}

# RandomForest 基线
rf = RandomForestRegressor(n_estimators=10, random_state=SEED, n_jobs=1)
rf.fit(X_train_cl_s, y_train)
rf_pred = rf.predict(X_test_cl_s)
results["RandomForest"] = metric_table(y_test, rf_pred)
joblib.dump((rf, scaler_cl, feature_cols_classical), OUT_DIR / "rf_model.joblib")

# SVR：TimeSeriesSplit 网格搜索，再用 train+val 重训后在 test 上评估
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {"C": [1], "epsilon": [0.1], "gamma": ["scale", "auto"]}
pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
grid = GridSearchCV(
    pipe,
    {
        "svr__C": param_grid["C"],
        "svr__epsilon": param_grid["epsilon"],
        "svr__gamma": param_grid["gamma"],
    },
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=1,
)
grid.fit(X_train_cl, y_train)  # 传入原始特征，Pipeline 内部做 scaling
best_svr_pipe = grid.best_estimator_

X_train_plus_val = np.vstack([X_train_cl, X_val_cl])
y_train_plus_val = np.concatenate([y_train, y_val])
best_svr_pipe.fit(X_train_plus_val, y_train_plus_val)
svr_pred = best_svr_pipe.predict(X_test_cl)
joblib.dump((best_svr_pipe, feature_cols_classical), OUT_DIR / "svr_model.joblib")
results["SVR"] = metric_table(y_test, svr_pred)


# SeqDataset 来自 core.py
# 序列模型只用温度和风速，scaler 同样只在训练集 fit
feature_cols_seq = ["exog_temp", "exog_wind"]
scaler_seq = StandardScaler()
df_seq_scaled_vals = scaler_seq.fit(train_df[feature_cols_seq].values).transform(df[feature_cols_seq].values)
df_seq_scaled = df.copy()
df_seq_scaled[feature_cols_seq] = df_seq_scaled_vals

train_size = len(train_df)
train_dataset = SeqDataset(df_seq_scaled, 0, train_size - 1, SEQ_LEN, feature_cols_seq, "OT")
val_dataset = SeqDataset(df_seq_scaled, train_size, train_size + val_size - 1, SEQ_LEN, feature_cols_seq, "OT")
test_dataset = SeqDataset(df_seq_scaled, train_size + val_size, n - 1, SEQ_LEN, feature_cols_seq, "OT")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


def train_model_torch(model, optimizer, train_loader, val_loader, epochs=EPOCHS, device=DEVICE):
    """按验证集 loss 保存最优权重。"""
    criterion = nn.MSELoss()
    model.to(device)
    best_state = None
    best_loss = float("inf")
    for _ in range(epochs):
        model.train()
        tloss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tloss += loss.item() * x.size(0)
        tloss /= len(train_loader.dataset)
        if val_loader is None:
            val_loss = tloss
        else:
            model.eval()
            vloss = 0.0
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv = xv.to(device)
                    yv = yv.to(device).unsqueeze(1)
                    outv = model(xv)
                    vloss += criterion(outv, yv).item() * xv.size(0)
            val_loss = vloss / len(val_loader.dataset)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_torch(model, loader, device=DEVICE):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds.append(out.cpu().numpy())
            trues.append(y.numpy())
    preds = np.vstack(preds).ravel()
    trues = np.hstack(trues).ravel()
    return trues, preds


class LSTMReg(nn.Module):
    def __init__(self, input_dim, hid=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hid, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hid, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        return self.fc(out)


class CNN1DReg(nn.Module):
    def __init__(self, input_dim, hid=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hid, hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(hid, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Conv1d 需要 (B, feat, seq_len)
        out = self.conv(x)
        return self.fc(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        divterm = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * divterm)
        pe[:, 1::2] = torch.cos(pos * divterm)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class TransformerReg(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_enc(x)
        out = self.transformer(x)
        out = out[:, -1, :]
        return self.fc(out)


# 训练 LSTM / CNN / Transformer 三个序列模型
input_dim = len(feature_cols_seq)
lstm = LSTMReg(input_dim=input_dim, hid=64)
lstm = train_model_torch(lstm, torch.optim.Adam(lstm.parameters(), lr=LR), train_loader, val_loader, epochs=EPOCHS)
y_true_lstm, y_pred_lstm = predict_torch(lstm, test_loader)
results["LSTM"] = metric_table(y_true_lstm, y_pred_lstm)
torch.save({"model_state": lstm.state_dict(), "scaler": scaler_seq, "feat": feature_cols_seq}, OUT_DIR / "lstm.pt")

cnn = CNN1DReg(input_dim=input_dim, hid=64)
cnn = train_model_torch(cnn, torch.optim.Adam(cnn.parameters(), lr=LR), train_loader, val_loader, epochs=EPOCHS)
y_true_cnn, y_pred_cnn = predict_torch(cnn, test_loader)
results["CNN1D"] = metric_table(y_true_cnn, y_pred_cnn)
torch.save({"model_state": cnn.state_dict(), "scaler": scaler_seq, "feat": feature_cols_seq}, OUT_DIR / "cnn1d.pt")

transformer = TransformerReg(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
transformer = train_model_torch(
    transformer, torch.optim.Adam(transformer.parameters(), lr=LR), train_loader, val_loader, epochs=EPOCHS
)
y_true_tr, y_pred_tr = predict_torch(transformer, test_loader)
results["Transformer"] = metric_table(y_true_tr, y_pred_tr)
torch.save(
    {"model_state": transformer.state_dict(), "scaler": scaler_seq, "feat": feature_cols_seq},
    OUT_DIR / "transformer.pt",
)

# 汇总各模型在测试集上的预测
test_rows = df.iloc[-test_size:].reset_index(drop=True).copy()
test_rows["OT_pred_RandomForest"] = rf_pred
test_rows["OT_pred_SVR"] = svr_pred

# 序列模型输出按全局时间索引对齐到 test_rows；对不齐的位置用 RF 预测回填
seq_pred_indices = list(range(len(train_df), len(train_df) + len(y_pred_lstm)))
for name, arr in [("LSTM", y_pred_lstm), ("CNN1D", y_pred_cnn), ("Transformer", y_pred_tr)]:
    colname = f"OT_pred_{name}"
    test_rows[colname] = np.nan
    for offset, pred in enumerate(arr):
        global_idx = seq_pred_indices[offset]
        if global_idx >= len(df) - test_size:
            local_idx = global_idx - (len(df) - test_size)
            if 0 <= local_idx < len(test_rows):
                test_rows.at[local_idx, colname] = float(pred)

for name in ["LSTM", "CNN1D", "Transformer"]:
    col = f"OT_pred_{name}"
    test_rows[col] = test_rows[col].fillna(test_rows["OT_pred_RandomForest"])

out_csv = OUT_DIR / "wind_model_output_with_OT_predictions.csv"
test_rows.to_csv(out_csv, index=False)

metrics_df = pd.DataFrame(results).T
metrics_df.to_csv(OUT_DIR / "model_metrics.csv")

# 保存对比图：真值 vs 预测、残差分布、MAE 柱状图、聚类 PCA
plt.rcParams.update({"axes.grid": True})
time_vals = test_rows["time"]
for model_name in results:
    pred_col = f"OT_pred_{model_name}"
    if pred_col not in test_rows.columns:
        continue
    plt.figure(figsize=(12, 4))
    plt.plot(time_vals, test_rows["OT"].values, label="OT_true")
    plt.plot(time_vals, test_rows[pred_col].values, label=f"OT_pred_{model_name}", alpha=0.8)
    plt.xlabel("time")
    plt.ylabel("OT")
    plt.title(f"OT: true vs predicted ({model_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ot_true_vs_pred_{model_name}.png")
    plt.close()

for model_name in results:
    pred_col = f"OT_pred_{model_name}"
    res = test_rows["OT"].values - test_rows[pred_col].values
    plt.figure(figsize=(6, 4))
    plt.hist(res, bins=50)
    plt.title(f"Residuals Histogram ({model_name})")
    plt.xlabel("Residual (OT_true - OT_pred)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"residuals_{model_name}.png")
    plt.close()

plt.figure(figsize=(8, 5))
mae_vals = [results[k]["MAE"] for k in results]
plt.bar(list(results.keys()), mae_vals)
plt.ylabel("MAE")
plt.title("Model comparison (MAE)")
plt.tight_layout()
plt.savefig(OUT_DIR / "model_comparison_mae.png")
plt.close()

# PCA 可视化用的是训练集 fit 过的 scaler_cluster
Xc_all = scaler_cluster.transform(df[cluster_features].values)
pca = PCA(n_components=2, random_state=SEED)
pc = pca.fit_transform(Xc_all)
plt.figure(figsize=(7, 5))
plt.scatter(pc[:, 0], pc[:, 1], c=(df["cluster"] == cluster_low).astype(int), s=8, cmap="coolwarm")
plt.title("Clustering PCA projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(OUT_DIR / "cluster_pca.png")
plt.close()

print("\nModels and OT metrics:")
for k, v in results.items():
    print(
        f"{k}: MAE={v['MAE']:.2f}, MSE={v['MSE']:.2f}, RMSE={v['RMSE']:.2f}, "
        f"NRMSE={v['NRMSE']:.2f}, R2={v['R2']:.2f}, MAPE={v['MAPE']:.2f}, "
        f"SMAPE={v['SMAPE']:.2f}, MASE={v['MASE']:.2f}"
    )

print(f"\nSaved predictions: {out_csv}")
print(f"Saved metrics: {OUT_DIR / 'model_metrics.csv'}")
print(f"Saved artifacts and plots: {OUT_DIR}")
print("\nMetrics table:")
print(metrics_df)
