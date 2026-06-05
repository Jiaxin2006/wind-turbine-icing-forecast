#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full revised script:
- Temperature-only clustering (configurable to disable clustering).
- Expanded model search: MLP / LSTM / CNN / CNN+LSTM / CNN+LSTM+Attention(+optional Transformer).
- Target scaling with scaler_y; features scaling on train only.
- Robust training: early stopping, ReduceLROnPlateau, weight decay, optional peak oversampling, asymmetric loss.
- Complete metrics on ORIGINAL scale: MAE/MSE/RMSE/NRMSE/R2/MAPE(%)/sMAPE(%)/MASE.
- JSON-safe serialization to avoid int32-key errors.
"""

# =========================
# Imports & global settings
# =========================
import os, math, random, json, time, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

OUT_DIR = Path("out_cnn_lstm_grid_search_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =========================
# Small utilities & metrics
# =========================
def json_safe(obj):
    """Recursively map numpy/torch types to Python builtins so json.dump won't fail."""
    import numpy as _np
    try:
        import torch as _torch
    except Exception:
        _torch = None

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if _np is not None and isinstance(obj, _np.ndarray):
        return json_safe(obj.tolist())
    if _np is not None and isinstance(obj, (_np.integer,)):
        return int(obj)
    if _np is not None and isinstance(obj, (_np.floating,)):
        return float(obj)
    if _torch is not None and _torch.is_tensor(obj):
        return json_safe(obj.detach().cpu().tolist())
    return obj

def mape_np(y, yhat):
    eps = 1e-9
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.mean(np.abs((y - yhat) / (np.abs(y) + eps))) * 100.0)

def smape_np(y, yhat):
    eps = 1e-9
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.mean(2.0 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + eps)) * 100.0)

def mase_np(y_true, y_pred, insample):
    """MASE: MAE(pred) / MAE(naive one-step) computed on training (insample)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    insample = np.asarray(insample).ravel()
    if len(insample) < 2:
        return float("nan")
    mae_naive = np.mean(np.abs(insample[1:] - insample[:-1]))
    mae_model = np.mean(np.abs(y_true - y_pred))
    return float(mae_model / mae_naive) if mae_naive > 0 else float("nan")

def full_metrics(y_true_raw, y_pred_raw, train_y_raw_for_mase):
    """Compute full set of metrics on original scale."""
    y = np.asarray(y_true_raw).ravel()
    p = np.asarray(y_pred_raw).ravel()
    mse = mean_squared_error(y, p)
    rmse = math.sqrt(mse)
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "NRMSE": float(rmse / (np.mean(np.abs(y)) + 1e-9)),
        "R2": float(r2_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "MAPE(%)": mape_np(y, p),
        "sMAPE(%)": smape_np(y, p),
        "MASE": mase_np(y, p, train_y_raw_for_mase),
    }

# =========================
# Data reading & preprocessing
# =========================
print("Reading data...")
data_path = "标注的数据-#67_1.xlsx"  # <- edit if needed
df_raw = pd.read_excel(data_path)

# --- Column name normalization (robust to headers in Chinese/English) ---
col_candidates = {
    'time': ['time','timestamp','date','统计时间'],
    'OT': ['OT','ot','efficiency','output','目标','目标值'],
    'exog_temp': ['Exogenous1','temperature','temp','外温','温度'],
    'exog_wind': ['Exogenous2','wind_speed','wind','风速']
}
cols = {}
for k, cands in col_candidates.items():
    for c in cands:
        if c in df_raw.columns:
            cols[k] = c; break
if 'time' not in cols:
    for c in df_raw.columns:
        if np.issubdtype(df_raw[c].dtype, np.datetime64):
            cols['time'] = c; break
if set(['OT','exog_temp']).difference(cols):
    raise ValueError(f"Missing essential columns. Found map: {cols}, df cols: {list(df_raw.columns)}")

df = df_raw.rename(columns={v:k for k,v in cols.items()})
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
for c in ['OT','exog_temp'] + ([ 'exog_wind'] if 'exog_wind' in df.columns else []):
    df[c] = pd.to_numeric(df[c], errors='coerce')

# --- Basic gap filling; create lag feature of target ---
df = df.interpolate(limit=5).ffill().bfill()
df['OT_prev'] = df['OT'].shift(1)
df = df.dropna().reset_index(drop=True)
print("Rows after preprocess:", len(df))
if len(df) < 200:
    print("Warning: dataset is small; prefer simpler models in the search space.")

# =========================
# Split (chronological)
# =========================
BASE = {
    "SEQ_LEN": 32,
    "VAL_RATIO": 0.10,
    "TEST_RATIO": 0.20,

    # training
    "EPOCHS": 80,
    "BATCH_SIZE": 16,
    "LR": 3e-4,
    "WEIGHT_DECAY": 1e-4,
    "EARLY_STOPPING_PATIENCE": 7,
    "LR_FACTOR": 0.5,
    "SCHED_PATIENCE": 3,

    # model defaults
    "DROPOUT": 0.1,
    "CNN_CHANNELS": 16,
    "CNN_KERNEL": 3,
    "LSTM_HID": 128,
    "TRANS_DMODEL": 128,
    "NUM_HEADS": 4,
    "NUM_TRANS_LAYERS": 1,

    # other tricks
    "HUBER_DELTA": 1.0,
    "USE_CLUSTER": True
}

# =========================
# Train/Val/Test split
# =========================
n = len(df)
test_size = int(n * BASE["TEST_RATIO"])
val_size  = int(n * BASE["VAL_RATIO"])
train_size = n - test_size - val_size
if train_size <= BASE["SEQ_LEN"] + 5:
    raise ValueError("Not enough training data; reduce SEQ_LEN or adjust split ratios.")

train_end = n - test_size - val_size
train_idx = np.arange(0, train_end)
val_idx   = np.arange(train_end, train_end + val_size)
test_idx  = np.arange(train_end + val_size, n)

print("Train/Val/Test sizes:", train_size, val_size, test_size)

# =========================
# Scalers (fit on TRAIN only)
# =========================
feat_cols = ['exog_temp'] + (['exog_wind'] if 'exog_wind' in df.columns else []) + ['OT_prev']

scaler_inputs = StandardScaler().fit(df.iloc[train_idx][feat_cols].values)
joblib.dump(scaler_inputs, OUT_DIR/'scaler_inputs.joblib')

scaler_y = StandardScaler().fit(df.iloc[train_idx][['OT']].values)
joblib.dump(scaler_y, OUT_DIR/'scaler_y.joblib')

# Make a scaled copy of df
df_scaled = df.copy()
df_scaled[feat_cols] = scaler_inputs.transform(df[feat_cols].values)

# =========================
# Clustering (temperature only)
# =========================
'''USE_CLUSTER = BASE["USE_CLUSTER"]
N_CLUSTERS  = params["N_CLUSTERS"]

if USE_CLUSTER:
    cluster_cols = ['exog_temp','exog_wind', 'OT_prev']  # only use temperature for clustering
    scaler_cluster = StandardScaler().fit(df.iloc[train_idx][cluster_cols].values)

    Xc_train = scaler_cluster.transform(df.iloc[train_idx][cluster_cols].values)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=20).fit(Xc_train)

    Xc_full = scaler_cluster.transform(df[cluster_cols].values)
    clusters = kmeans.predict(Xc_full)

    # attach cluster to both df and df_scaled
    df['cluster'] = clusters
    df_scaled['cluster'] = clusters

    joblib.dump(scaler_cluster, OUT_DIR/'scaler_cluster.joblib')
    joblib.dump(kmeans, OUT_DIR/'kmeans.joblib')

    print("Clustering enabled. Unique clusters:", sorted(np.unique(clusters)))
    print("Train cluster counts:\n", df.loc[train_idx, 'cluster'].value_counts().sort_index())
    print("Val   cluster counts:\n", df.loc[val_idx, 'cluster'].value_counts().sort_index())
    print("Test  cluster counts:\n", df.loc[test_idx, 'cluster'].value_counts().sort_index())

else:
    df['cluster'] = -1
    df_scaled['cluster'] = -1
    print("Clustering disabled. Training one global model.")
'''

# =========================
# Dataset class
# =========================
class ClusterSeqDataset(Dataset):
    """Sequences whose every time step belongs to the same cluster."""
    def __init__(self, df_scaled, start_idx, end_idx, seq_len, feat_cols, cluster_id, scaler_y):
        self.df = df_scaled; self.start = start_idx; self.end = end_idx
        self.seq_len = seq_len; self.feat_cols = feat_cols
        self.cluster_id = cluster_id; self.scaler_y = scaler_y
        self.valid_idx = []
        for i in range(self.start, self.end - self.seq_len):
            block = self.df.iloc[i: i + self.seq_len + 1]
            if (block['cluster'].values == cluster_id).all():
                self.valid_idx.append(i)

    def __len__(self): return len(self.valid_idx)

    def __getitem__(self, idx):
        i0 = self.valid_idx[idx]
        seq = self.df.iloc[i0: i0 + self.seq_len][self.feat_cols].values.astype(np.float32)
        label_idx = i0 + self.seq_len
        y_raw = float(self.df.iloc[label_idx]['OT'])
        y_scaled = float(self.scaler_y.transform([[y_raw]])[0,0])
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32), int(label_idx)

class SeqDataset(Dataset):
    """General sequential dataset without cluster constraint."""
    def __init__(self, df_scaled, start_idx, end_idx, seq_len, feat_cols, scaler_y):
        self.df = df_scaled; self.start = start_idx; self.end = end_idx
        self.seq_len = seq_len; self.feat_cols = feat_cols; self.scaler_y = scaler_y
        self.n = max(0, (self.end - self.start + 1) - self.seq_len)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        i0 = self.start + idx
        seq = self.df.iloc[i0: i0 + self.seq_len][self.feat_cols].values.astype(np.float32)
        label_idx = i0 + self.seq_len
        y_raw = float(self.df.iloc[label_idx]['OT'])
        y_scaled = float(self.scaler_y.transform([[y_raw]])[0,0])
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32), int(label_idx)

# =========================
# Model zoo: lightweight to complex
# =========================
class MLPBaseline(nn.Module):
    """Simple MLP on flattened sequence."""
    def __init__(self, seq_len, feat_dim, hidden=64, dropout=0.1, hetero=False):
        super().__init__()
        self.hetero = hetero
        out_dim = 2 if hetero else 1
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len*feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        out = self.net(x)
        if self.hetero:
            mu, logvar = out[:,0], out[:,1].clamp(-10,10)
            return mu, logvar
        return out.squeeze(1)

class LSTMOnly(nn.Module):
    """Single-layer LSTM + MLP head."""
    def __init__(self, feat_dim, hid=128, dropout=0.1, hetero=False):
        super().__init__()
        self.hetero = hetero
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hid, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        out_dim = 2 if hetero else 1
        self.head = nn.Sequential(nn.Linear(hid, 64), nn.ReLU(), nn.Linear(64, out_dim))

    def forward(self, x):
        h, _ = self.lstm(x); h = self.drop(h[:, -1, :])
        out = self.head(h)
        if self.hetero:
            mu, logvar = out[:,0], out[:,1].clamp(-10,10)
            return mu, logvar
        return out.squeeze(1)

class CNNOnly(nn.Module):
    """Two 1D conv layers + pooling + MLP head."""
    def __init__(self, feat_dim, channels=16, k=3, dropout=0.1, hetero=False):
        super().__init__()
        self.hetero = hetero
        self.conv1 = nn.Conv1d(feat_dim, channels, k, padding=k//2)
        self.conv2 = nn.Conv1d(channels, channels, k, padding=k//2)
        self.act = nn.ReLU(); self.drop = nn.Dropout(dropout)
        out_dim = 2 if hetero else 1
        self.head = nn.Sequential(nn.Linear(channels, 64), nn.ReLU(), nn.Linear(64, out_dim))

    def forward(self, x):
        c = x.permute(0,2,1)
        c = self.act(self.conv1(c))
        c = self.act(self.conv2(c))
        c = self.drop(c)
        # global average pooling over time
        g = torch.mean(c, dim=-1)
        out = self.head(g)
        if self.hetero:
            mu, logvar = out[:,0], out[:,1].clamp(-10,10)
            return mu, logvar
        return out.squeeze(1)

class CNN_LSTM_Attn(nn.Module):
    """CNN -> LSTM -> (optionally Attention/Transformer) -> head."""
    def __init__(self, feat_dim, channels=16, k=3, lstm_hid=128,
                 d_model=128, nhead=4, n_trans=1, dropout=0.1, hetero=False):
        super().__init__()
        self.hetero = hetero
        self.conv1 = nn.Conv1d(feat_dim, channels, k, padding=k//2)
        self.conv2 = nn.Conv1d(channels, channels, k, padding=k//2)
        self.act = nn.ReLU(); self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(channels, lstm_hid, num_layers=1, batch_first=True)
        self.proj = nn.Linear(lstm_hid, d_model) if lstm_hid != d_model else nn.Identity()
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        if n_trans > 0:
            enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*2, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_trans)
        else:
            self.encoder = None
        out_dim = 2 if hetero else 1
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, out_dim))

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        c = x.permute(0,2,1)
        c = self.act(self.conv1(c)); c = self.act(self.conv2(c)); c = self.drop(c)
        c = c.permute(0,2,1)
        h, _ = self.lstm(c); h = self.drop(h)
        z = self.proj(h)
        z, _ = self.mha(z, z, z, need_weights=False); z = self.drop(z)
        if self.encoder is not None:
            z = self.encoder(z)
        last = z[:, -1, :]
        out = self.head(last)
        if self.hetero:
            mu, logvar = out[:,0], out[:,1].clamp(-10,10)
            return mu, logvar
        return out.squeeze(1)

class LSTM_CNN_MLP(nn.Module):
    """LSTM并行CNN特征提取 + MLP融合的组合模型"""
    def __init__(self, feat_dim, lstm_hid=128, channels=16, k=3, mlp_hidden=64, dropout=0.1, hetero=False):
        super().__init__()
        self.hetero = hetero
        
        # LSTM分支
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=lstm_hid, num_layers=1, batch_first=True)
        self.lstm_drop = nn.Dropout(dropout)
        
        # CNN分支
        self.conv1 = nn.Conv1d(feat_dim, channels, k, padding=k//2)
        self.conv2 = nn.Conv1d(channels, channels, k, padding=k//2)
        self.cnn_drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
        # MLP分支（对最后一个时间步的原始特征）
        self.mlp_branch = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden)
        )
        
        # 特征融合层
        fusion_input = lstm_hid + channels + mlp_hidden
        out_dim = 2 if hetero else 1
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_input // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input // 2, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape
        
        # LSTM分支：提取序列时序特征
        lstm_out, _ = self.lstm(x)
        lstm_feat = self.lstm_drop(lstm_out[:, -1, :])  # 取最后时刻输出
        
        # CNN分支：提取局部模式特征
        cnn_input = x.permute(0, 2, 1)  # (batch, feat, seq)
        cnn_out = self.act(self.conv1(cnn_input))
        cnn_out = self.act(self.conv2(cnn_out))
        cnn_out = self.cnn_drop(cnn_out)
        # 全局平均池化
        cnn_feat = torch.mean(cnn_out, dim=-1)  # (batch, channels)
        
        # MLP分支：处理最新时刻的特征
        mlp_input = x[:, -1, :]  # 取最后一个时间步
        mlp_feat = self.mlp_branch(mlp_input)
        
        # 特征融合
        fused = torch.cat([lstm_feat, cnn_feat, mlp_feat], dim=1)
        out = self.fusion(fused)
        
        if self.hetero:
            mu, logvar = out[:, 0], out[:, 1].clamp(-10, 10)
            return mu, logvar
        return out.squeeze(1)


class Advanced_LSTM_CNN_MLP(nn.Module):
    """更高级的组合模型：包含注意力机制和残差连接"""
    def __init__(self, feat_dim, lstm_hid=128, channels=16, k=3, mlp_hidden=64, dropout=0.1, hetero=False):
        super().__init__()
        self.hetero = hetero
        
        # 多层LSTM
        self.lstm1 = nn.LSTM(feat_dim, lstm_hid, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_hid, lstm_hid, batch_first=True)
        
        # 残差CNN
        self.conv1 = nn.Conv1d(feat_dim, channels, k, padding=k//2)
        self.conv2 = nn.Conv1d(channels, channels, k, padding=k//2)
        self.conv3 = nn.Conv1d(channels, channels, k, padding=k//2)
        self.conv_res = nn.Conv1d(feat_dim, channels, 1)  # 1x1 conv for residual
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(lstm_hid, num_heads=4, batch_first=True)
        
        # 增强MLP分支
        self.mlp_branch = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden * 2, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden)
        )
        
        # 融合网络
        fusion_input = lstm_hid + channels + mlp_hidden
        out_dim = 2 if hetero else 1
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_input),
            nn.LayerNorm(fusion_input),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input, fusion_input // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input // 2, out_dim)
        )
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 多层LSTM with attention
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out2, lstm_out2, lstm_out2)
        lstm_feat = attn_out[:, -1, :]
        
        # 残差CNN
        cnn_input = x.permute(0, 2, 1)
        identity = self.conv_res(cnn_input)
        
        cnn_out = self.act(self.conv1(cnn_input))
        cnn_out = self.act(self.conv2(cnn_out))
        cnn_out = self.conv3(cnn_out)
        cnn_out = cnn_out + identity  # 残差连接
        cnn_out = self.act(cnn_out)
        cnn_feat = torch.mean(cnn_out, dim=-1)
        
        # 增强MLP
        mlp_input = x[:, -1, :]
        mlp_feat = self.mlp_branch(mlp_input)
        
        # 融合
        fused = torch.cat([lstm_feat, cnn_feat, mlp_feat], dim=1)
        out = self.fusion(fused)
        
        if self.hetero:
            mu, logvar = out[:, 0], out[:, 1].clamp(-10, 10)
            return mu, logvar
        return out.squeeze(1)

class CNN_LSTM_MLP(nn.Module):
    """CNN特征提取 + LSTM序列建模 + MLP最终处理的组合模型"""
    def __init__(self, feat_dim, lstm_hid=128, channels=16, k=3, mlp_hidden=64, dropout=0.1, hetero=False):
        super().__init__()
        self.hetero = hetero
        
        # CNN分支：先进行特征提取
        self.conv1 = nn.Conv1d(feat_dim, channels, k, padding=k//2)
        self.conv2 = nn.Conv1d(channels, channels, k, padding=k//2)
        self.cnn_drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
        # LSTM分支：对CNN提取的特征进行序列建模
        self.lstm = nn.LSTM(input_size=channels, hidden_size=lstm_hid, num_layers=1, batch_first=True)
        self.lstm_drop = nn.Dropout(dropout)
        
        # MLP分支：对原始最后时刻特征的直接处理
        self.mlp_branch = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden)
        )
        
        # 特征融合层
        fusion_input = lstm_hid + mlp_hidden
        out_dim = 2 if hetero else 1
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_input // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input // 2, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape
        
        # CNN特征提取
        cnn_input = x.permute(0, 2, 1)  # (batch, feat, seq)
        cnn_out = self.act(self.conv1(cnn_input))
        cnn_out = self.act(self.conv2(cnn_out))
        cnn_out = self.cnn_drop(cnn_out)
        
        # 将CNN输出转换为LSTM输入格式
        lstm_input = cnn_out.permute(0, 2, 1)  # (batch, seq, channels)
        
        # LSTM序列建模
        lstm_out, _ = self.lstm(lstm_input)
        lstm_feat = self.lstm_drop(lstm_out[:, -1, :])  # 取最后时刻输出
        
        # MLP分支：处理原始最新时刻的特征
        mlp_input = x[:, -1, :]  # 取最后一个时间步
        mlp_feat = self.mlp_branch(mlp_input)
        
        # 特征融合
        fused = torch.cat([lstm_feat, mlp_feat], dim=1)
        out = self.fusion(fused)
        
        if self.hetero:
            mu, logvar = out[:, 0], out[:, 1].clamp(-10, 10)
            return mu, logvar
        return out.squeeze(1)


class CNN_MLP(nn.Module):
    """CNN特征提取 + MLP处理的双分支组合模型"""
    def __init__(self, feat_dim, channels=16, k=3, mlp_hidden=64, dropout=0.1, hetero=False):
        super().__init__()
        self.hetero = hetero
        
        # CNN分支：提取局部时序模式
        self.conv1 = nn.Conv1d(feat_dim, channels, k, padding=k//2)
        self.conv2 = nn.Conv1d(channels, channels, k, padding=k//2)
        self.conv3 = nn.Conv1d(channels, channels, k, padding=k//2)
        self.cnn_drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
        # MLP分支1：处理全局平均池化后的CNN特征
        self.mlp_cnn_branch = nn.Sequential(
            nn.Linear(channels, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden)
        )
        
        # MLP分支2：处理最后时刻的原始特征
        self.mlp_raw_branch = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden)
        )
        
        # 特征融合层
        fusion_input = mlp_hidden + mlp_hidden  # 两个MLP分支的输出
        out_dim = 2 if hetero else 1
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_input // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input // 2, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape
        
        # CNN分支：提取时序模式特征
        cnn_input = x.permute(0, 2, 1)  # (batch, feat, seq)
        cnn_out = self.act(self.conv1(cnn_input))
        cnn_out = self.act(self.conv2(cnn_out))
        cnn_out = self.act(self.conv3(cnn_out))
        cnn_out = self.cnn_drop(cnn_out)
        
        # 全局平均池化
        cnn_feat = torch.mean(cnn_out, dim=-1)  # (batch, channels)
        mlp_cnn_feat = self.mlp_cnn_branch(cnn_feat)
        
        # MLP分支：处理最新时刻的原始特征
        mlp_raw_input = x[:, -1, :]  # 取最后一个时间步
        mlp_raw_feat = self.mlp_raw_branch(mlp_raw_input)
        
        # 特征融合
        fused = torch.cat([mlp_cnn_feat, mlp_raw_feat], dim=1)
        out = self.fusion(fused)
        
        if self.hetero:
            mu, logvar = out[:, 0], out[:, 1].clamp(-10, 10)
            return mu, logvar
        return out.squeeze(1)


class LSTM_MLP(nn.Module):
    """LSTM序列建模 + MLP特征处理的双分支组合模型"""
    def __init__(self, feat_dim, lstm_hid=128, mlp_hidden=64, dropout=0.1, hetero=False):
        super().__init__()
        self.hetero = hetero
        
        # LSTM分支：提取序列时序特征
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=lstm_hid, num_layers=1, batch_first=True)
        self.lstm_drop = nn.Dropout(dropout)
        
        # MLP分支1：处理LSTM的全序列输出的平均
        self.mlp_lstm_branch = nn.Sequential(
            nn.Linear(lstm_hid, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden)
        )
        
        # MLP分支2：直接处理最后时刻的原始特征
        self.mlp_raw_branch = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden)
        )
        
        # MLP分支3：处理输入序列的统计特征（均值、方差等）
        self.mlp_stat_branch = nn.Sequential(
            nn.Linear(feat_dim * 3, mlp_hidden),  # mean, std, last
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden)
        )
        
        # 特征融合层
        fusion_input = mlp_hidden + mlp_hidden + mlp_hidden  # 三个MLP分支
        out_dim = 2 if hetero else 1
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_input // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input // 2, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape
        
        # LSTM分支：提取序列时序特征
        lstm_out, _ = self.lstm(x)
        lstm_last = self.lstm_drop(lstm_out[:, -1, :])  # 最后时刻
        mlp_lstm_feat = self.mlp_lstm_branch(lstm_last)
        
        # MLP分支1：处理最新时刻的原始特征
        mlp_raw_input = x[:, -1, :]  # 取最后一个时间步
        mlp_raw_feat = self.mlp_raw_branch(mlp_raw_input)
        
        # MLP分支2：处理序列统计特征
        x_mean = torch.mean(x, dim=1)  # 序列均值
        x_std = torch.std(x, dim=1)    # 序列标准差
        x_last = x[:, -1, :]          # 最后时刻值
        stat_input = torch.cat([x_mean, x_std, x_last], dim=1)
        mlp_stat_feat = self.mlp_stat_branch(stat_input)
        
        # 特征融合
        fused = torch.cat([mlp_lstm_feat, mlp_raw_feat, mlp_stat_feat], dim=1)
        out = self.fusion(fused)
        
        if self.hetero:
            mu, logvar = out[:, 0], out[:, 1].clamp(-10, 10)
            return mu, logvar
        return out.squeeze(1)

# =========================
# Loss helpers
# =========================
def log_cosh_loss(pred, target):
    return torch.mean(torch.log(torch.cosh(pred - target + 1e-12)))

def hetero_nll(mu, logvar, y):
    inv = torch.exp(-logvar)
    return (0.5 * inv * (y - mu)**2 + 0.5 * logvar).mean()

def asym_weight(diff, under_w):
    return torch.where(diff < 0, under_w, 1.0)

# =========================
# Search space (now with MODEL_TYPE)
# =========================
param_grid = {
    "MODEL_TYPE": [
                   "cnn_lstm_attn","cnn_lstm_mlp"
    ],   # <- expanded
    "SEQ_LEN": [2,4],

    # common training
    "LR": [3e-4],
    "BATCH_SIZE": [16],
    "WEIGHT_DECAY": [0.0],
    "DROPOUT": [0.0],

    # model sizes
    "CNN_CHANNELS": [16],
    "CNN_KERNEL": [3],
    "LSTM_HID": [128],
    "MLP_HIDDEN": [96],
    "NUM_TRANS_LAYERS": [1,2],   # only used for cnn_lstm_attn

    # loss configs
    "LOSS_TYPE": ["smape"], # "mae", "huber", 
    "USE_HET": [True],
    "USE_ASYM": [True],
    "ASYM_W": [1.0],

    # data tricks
    "OVERSAMPLE_PEAKS": [True],
    "N_CLUSTERS": [1,2,3,4,5]
}
max_runs = 40  # hard cap

# =========================
# Train + Eval per run (per cluster)
# =========================
def build_model(model_type, seq_len, feat_dim, p):
    het = p["USE_HET"]
    if model_type == "mlp":
        return MLPBaseline(seq_len, feat_dim, hidden=64, dropout=p["DROPOUT"], hetero=het)
    if model_type == "lstm":
        return LSTMOnly(feat_dim, hid=p["LSTM_HID"], dropout=p["DROPOUT"], hetero=het)
    if model_type == "cnn":
        return CNNOnly(feat_dim, channels=p["CNN_CHANNELS"], k=p["CNN_KERNEL"], dropout=p["DROPOUT"], hetero=het)
    if model_type == "cnn_lstm":
        return CNN_LSTM_Attn(feat_dim, channels=p["CNN_CHANNELS"], k=p["CNN_KERNEL"],
                             lstm_hid=p["LSTM_HID"], d_model=p["LSTM_HID"], nhead=4,
                             n_trans=0, dropout=p["DROPOUT"], hetero=het)
    if model_type == "cnn_lstm_attn":
        return CNN_LSTM_Attn(feat_dim, channels=p["CNN_CHANNELS"], k=p["CNN_KERNEL"],
                             lstm_hid=p["LSTM_HID"], d_model=p["LSTM_HID"], nhead=4,
                             n_trans=p["NUM_TRANS_LAYERS"], dropout=p["DROPOUT"], hetero=het)
    elif model_type == "lstm_cnn_mlp":
        return LSTM_CNN_MLP(feat_dim, lstm_hid=p["LSTM_HID"], 
                           channels=p["CNN_CHANNELS"], k=p["CNN_KERNEL"],
                           mlp_hidden=p.get("MLP_HIDDEN", 64), 
                           dropout=p["DROPOUT"], hetero=het)
    
    elif model_type == "advanced_lstm_cnn_mlp":
        return Advanced_LSTM_CNN_MLP(feat_dim, lstm_hid=p["LSTM_HID"],
                                    channels=p["CNN_CHANNELS"], k=p["CNN_KERNEL"],
                                    mlp_hidden=p.get("MLP_HIDDEN", 64),
                                    dropout=p["DROPOUT"], hetero=het)
    elif model_type == "cnn_lstm_mlp":
        return CNN_LSTM_MLP(feat_dim, lstm_hid=p["LSTM_HID"],
                           channels=p["CNN_CHANNELS"], k=p["CNN_KERNEL"],
                           mlp_hidden=p.get("MLP_HIDDEN", 64),
                           dropout=p["DROPOUT"], hetero=het)
    
    elif model_type == "cnn_mlp":
        return CNN_MLP(feat_dim, channels=p["CNN_CHANNELS"], k=p["CNN_KERNEL"],
                      mlp_hidden=p.get("MLP_HIDDEN", 64),
                      dropout=p["DROPOUT"], hetero=het)
    
    elif model_type == "lstm_mlp":
        return LSTM_MLP(feat_dim, lstm_hid=p["LSTM_HID"],
                       mlp_hidden=p.get("MLP_HIDDEN", 64),
                       dropout=p["DROPOUT"], hetero=het)
    
    
    raise ValueError(f"Unknown MODEL_TYPE {model_type}")

def compute_loss(pred_or_mu, target, params, logvar=None):
    loss_type = params["LOSS_TYPE"]
    if logvar is not None:
        # heteroscedastic: wrap base loss in Gaussian NLL
        if loss_type == "mae":
            base = torch.abs(pred_or_mu - target)
        elif loss_type == "huber":
            base = nn.functional.smooth_l1_loss(pred_or_mu, target, reduction='none')
        elif loss_type == "logcosh":
            base = torch.log(torch.cosh(pred_or_mu - target + 1e-12))
        elif loss_type == "mape":
            base = torch.abs((pred_or_mu - target) / (target + 1e-6))
        elif loss_type == "smape":
            base = 2.0 * torch.abs(pred_or_mu - target) / (torch.abs(pred_or_mu) + torch.abs(target) + 1e-6)
        else:  # default to MSE-like
            base = (pred_or_mu - target)**2
        if params["USE_ASYM"]:
            w = asym_weight(pred_or_mu - target, params["ASYM_W"]).to(pred_or_mu.device)
            base = w * base
        return (0.5 * torch.exp(-logvar) * base + 0.5 * logvar).mean()
    else:
        if loss_type == "mae":
            return torch.abs(pred_or_mu - target).mean()
        if loss_type == "huber":
            return nn.functional.smooth_l1_loss(pred_or_mu, target)
        if loss_type == "logcosh":
            return log_cosh_loss(pred_or_mu, target)
        if loss_type == "mape":
            return torch.mean(torch.abs((pred_or_mu - target) / (target + 1e-6)))
        if loss_type == "smape":
            return torch.mean(2.0 * torch.abs(pred_or_mu - target) / (torch.abs(pred_or_mu) + torch.abs(target) + 1e-6))
        # default: SmoothL1 ~ robust MSE
        return nn.SmoothL1Loss()(pred_or_mu, target)

def train_and_eval(run_id, params):
    """Train per-cluster models and evaluate on test; save predictions and metrics."""
    cfg = BASE.copy(); cfg.update(params)

    # indices
    train_start, train_end = 0, train_size - 1
    val_start, val_end     = train_size, train_size + val_size - 1
    test_start, test_end   = train_size + val_size, n - 1

    # prepare loaders for each cluster id
    cluster_ids = sorted(df['cluster'].unique())
    train_loaders, val_loaders = {}, {}
    peak_thr = {}

    for cid in cluster_ids:
        tr_ds = ClusterSeqDataset(df_scaled, train_start, train_end, cfg["SEQ_LEN"], feat_cols, cid, scaler_y)
        va_ds = ClusterSeqDataset(df_scaled, val_start, val_end, cfg["SEQ_LEN"], feat_cols, cid, scaler_y)

        # if too few sequences, skip this cluster
        if len(tr_ds) < 8 or len(va_ds) < 4:
            print(f"[Run {run_id}] Skip cluster {cid}: too few sequences (train {len(tr_ds)}, val {len(va_ds)})")
            continue

        if cfg["OVERSAMPLE_PEAKS"]:
            ys_tr_raw = np.array([df.iloc[i + cfg["SEQ_LEN"]]['OT'] for i in tr_ds.valid_idx])
            thr = np.percentile(ys_tr_raw, 95)
            peak_thr[cid] = float(thr)
            weights = np.where(ys_tr_raw > thr, 3.0, 1.0).astype(np.float32)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            tr_loader = DataLoader(tr_ds, batch_size=cfg["BATCH_SIZE"], sampler=sampler, drop_last=False)
        else:
            tr_loader = DataLoader(tr_ds, batch_size=cfg["BATCH_SIZE"], shuffle=True, drop_last=False)
            peak_thr[cid] = None

        va_loader = DataLoader(va_ds, batch_size=cfg["BATCH_SIZE"], shuffle=False, drop_last=False)
        train_loaders[cid] = tr_loader; val_loaders[cid] = va_loader

    # global test loader (no cluster constraint; we pick model by last input index's cluster)
    test_loader = DataLoader(SeqDataset(df_scaled, test_start, test_end, cfg["SEQ_LEN"], feat_cols, scaler_y),
                             batch_size=cfg["BATCH_SIZE"], shuffle=False, drop_last=False)

    run_dir = OUT_DIR / f"run_{run_id:03d}"
    run_dir.mkdir(exist_ok=True, parents=True)
    with open(run_dir/"config.json", "w") as f:
        json.dump(json_safe({"params": params, "cfg": cfg}), f, indent=2)

    # train per-cluster
    models, history = {}, {}
    feat_dim = len(feat_cols)
    for cid in cluster_ids:
        if cid not in train_loaders:  # skipped small cluster
            continue

        model = build_model(cfg["MODEL_TYPE"], cfg["SEQ_LEN"], feat_dim, cfg).to(DEVICE)

        # init final bias from train label (scaled) for stability
        tr_loader = train_loaders[cid]
        y_train_scaled = []
        for i in tr_loader.dataset.valid_idx:
            y_train_scaled.append(float(scaler_y.transform([[df.iloc[i + cfg["SEQ_LEN"]]['OT']]])[0,0]))
        train_mean_scaled = float(np.mean(y_train_scaled)) if len(y_train_scaled)>0 else 0.0
        with torch.no_grad():
            # heuristic: try to access last linear in head to set bias
            for m in model.modules():
                if isinstance(m, nn.Linear) and m.out_features in (1,2):
                    if cfg["USE_HET"] and m.out_features == 2:
                        m.bias.data[0].fill_(train_mean_scaled)  # mu
                        m.bias.data[1].fill_(0.0)                # logvar start small
                    elif m.out_features == 1:
                        m.bias.data.fill_(train_mean_scaled)
                    break

        opt = optim.Adam(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=cfg["LR_FACTOR"],
                                                         patience=cfg["SCHED_PATIENCE"])

        best_val = float("inf"); best_state = None; es = 0
        tr_losses, va_losses = [], []

        for ep in range(1, BASE["EPOCHS"]+1):
            # --- train ---
            model.train(); s_loss = 0.0; s_cnt = 0
            for xb, yb, _ in train_loaders[cid]:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                opt.zero_grad()
                if cfg["USE_HET"]:
                    mu, logvar = model(xb)
                    loss = compute_loss(mu, yb, cfg, logvar=logvar)
                else:
                    out = model(xb)
                    loss = compute_loss(out, yb, cfg)
                loss.backward(); opt.step()
                s_loss += float(loss.item()) * xb.size(0); s_cnt += xb.size(0)
            tr_epoch = s_loss / max(1, s_cnt); tr_losses.append(tr_epoch)

            # --- validate ---
            model.eval(); v_loss = 0.0; v_cnt = 0
            with torch.no_grad():
                for xb, yb, _ in val_loaders[cid]:
                    xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                    if cfg["USE_HET"]:
                        mu, logvar = model(xb)
                        loss_v = compute_loss(mu, yb, cfg, logvar=logvar)  # 🔑 和 train 保持一致
                    else:
                        out = model(xb)
                        loss_v = compute_loss(out, yb, cfg) 

                    v_loss += float(loss_v.item()) * xb.size(0); v_cnt += xb.size(0)
            va_epoch = v_loss / max(1, v_cnt); va_losses.append(va_epoch)
            scheduler.step(va_epoch)

            print(f"[Run {run_id}] C{cid} Ep {ep:02d} train={tr_epoch:.6f} val={va_epoch:.6f}")

            if va_epoch < best_val - 1e-9:
                best_val = va_epoch; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; es = 0
            else:
                es += 1
                if es >= BASE["EARLY_STOPPING_PATIENCE"]:
                    print(f"[Run {run_id}] Early stop C{cid} at ep {ep}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        models[cid] = model
        history[cid] = {"train": tr_losses, "val": va_losses}

        # plot loss
        plt.figure(figsize=(7,4))
        plt.plot(range(1, len(tr_losses)+1), tr_losses, label="train")
        plt.plot(range(1, len(va_losses)+1), va_losses, label="val")
        plt.title(f"Run{run_id} Cluster{cid} Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        plt.tight_layout(); plt.savefig(run_dir/f"loss_run{run_id}_cluster{cid}.png"); plt.close()

        torch.save(model.state_dict(), run_dir/f"model_run{run_id}_cluster{cid}.pt")

    # --- Test: dispatch by last-step cluster id ---
    preds_scaled, trues_raw, label_idxs = [], [], []
    with torch.no_grad():
        for xb, yb, idxs in test_loader:
            for i in range(xb.size(0)):
                label_idx = int(idxs[i]); last_in_idx = label_idx - 1
                cid = int(df.loc[last_in_idx, 'cluster'])
                if cid not in models:  # skipped small cluster -> fallback: use any available model
                    if len(models) == 0:
                        continue
                    cid = list(models.keys())[0]
                model = models[cid].to(DEVICE).eval()
                x1 = xb[i:i+1].to(DEVICE)
                if cfg["USE_HET"]:
                    mu, _ = model(x1); pred_s = float(mu.detach().cpu().numpy().ravel()[0])
                else:
                    pred_s = float(model(x1).detach().cpu().numpy().ravel()[0])
                preds_scaled.append(pred_s)
                trues_raw.append(float(df.loc[label_idx, 'OT']))
                label_idxs.append(label_idx)

    if len(preds_scaled) == 0:
        metrics = {k: float("nan") for k in ["MAE","MSE","RMSE","NRMSE","R2","MAPE(%)","sMAPE(%)","MASE"]}
        preds_inv = []
    else:
        preds_inv = scaler_y.inverse_transform(np.array(preds_scaled).reshape(-1,1)).ravel()
        # MASE denominator uses train raw OT (chronological)
        train_y_raw_for_mase = train_df['OT'].values
        metrics = full_metrics(trues_raw, preds_inv, train_y_raw_for_mase)

    # save predictions & metrics
    out_pred = pd.DataFrame({
        "time": df.loc[label_idxs, 'time'].values if len(label_idxs)>0 else [],
        "OT_true": trues_raw,
        "OT_pred": preds_inv
    })
    out_pred.to_csv(run_dir/"test_predictions.csv", index=False)
    with open(run_dir/"metrics.json", "w") as f:
        json.dump(json_safe({"metrics_test": metrics, "params": params, "history": history}), f, indent=2)

    # summary row
    summary = {
        "run_id": run_id,
        "MODEL_TYPE": cfg["MODEL_TYPE"],
        "SEQ_LEN": cfg["SEQ_LEN"],
        "LR": cfg["LR"],
        "BATCH_SIZE": cfg["BATCH_SIZE"],
        "WEIGHT_DECAY": cfg["WEIGHT_DECAY"],
        "DROPOUT": cfg["DROPOUT"],
        "CNN_CHANNELS": cfg["CNN_CHANNELS"],
        "CNN_KERNEL": cfg["CNN_KERNEL"],
        "LSTM_HID": cfg["LSTM_HID"],
        "NUM_TRANS_LAYERS": cfg["NUM_TRANS_LAYERS"],
        "USE_HET": cfg["USE_HET"],
        "LOSS_TYPE": cfg["LOSS_TYPE"],
        "USE_ASYM": cfg["USE_ASYM"],
        "ASYM_W": cfg["ASYM_W"],
        "OVERSAMPLE_PEAKS": cfg["OVERSAMPLE_PEAKS"],
        "N_CLUSTERS": cfg["N_CLUSTERS"],
        **metrics
    }
    return summary

# =========================
# Build/run grid (randomly sub-sample if too many)
# =========================
all_keys = list(param_grid.keys())
all_combos = list(itertools.product(*[param_grid[k] for k in all_keys]))
print(f"Total combos: {len(all_combos)}")
if len(all_combos) > max_runs:
    random.seed(SEED)
    all_combos = random.sample(all_combos, max_runs)
    print(f"Sampled to {len(all_combos)} (max_runs={max_runs})")

results = []
for rid, combo in enumerate(all_combos):
    params = dict(zip(all_keys, combo))
    # =========================
    # Final splits (with cluster preserved)
    # =========================

    N_CLUSTERS = params["N_CLUSTERS"]
    cluster_cols = ['exog_temp','exog_wind', 'OT_prev']  # only use temperature for clustering
    scaler_cluster = StandardScaler().fit(df.iloc[train_idx][cluster_cols].values)

    Xc_train = scaler_cluster.transform(df.iloc[train_idx][cluster_cols].values)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=20).fit(Xc_train)

    Xc_full = scaler_cluster.transform(df[cluster_cols].values)
    clusters = kmeans.predict(Xc_full)

    # attach cluster to both df and df_scaled
    df['cluster'] = clusters
    df_scaled['cluster'] = clusters

    joblib.dump(scaler_cluster, OUT_DIR/'scaler_cluster.joblib')
    joblib.dump(kmeans, OUT_DIR/'kmeans.joblib')

    print("Clustering enabled. Unique clusters:", sorted(np.unique(clusters)))
    print("Train cluster counts:\n", df.loc[train_idx, 'cluster'].value_counts().sort_index())
    print("Val   cluster counts:\n", df.loc[val_idx, 'cluster'].value_counts().sort_index())
    print("Test  cluster counts:\n", df.loc[test_idx, 'cluster'].value_counts().sort_index())

    train_df = df_scaled.iloc[train_idx].reset_index(drop=True)
    val_df   = df_scaled.iloc[val_idx].reset_index(drop=True)
    test_df  = df_scaled.iloc[test_idx].reset_index(drop=True)

    for name, sub in [('train', train_df), ('val', val_df), ('test', test_df)]:
        assert 'cluster' in sub.columns, f"{name}_df lost 'cluster' column!"

    # seed per run
    random.seed(SEED + rid); np.random.seed(SEED + rid); torch.manual_seed(SEED + rid)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + rid)
    print(f"\n==== START RUN {rid} :: {params}")
    try:
        summary = train_and_eval(rid, params)
        results.append(summary)
        pd.DataFrame(results).to_csv(OUT_DIR/"grid_search_results_partial.csv", index=False)
    except Exception as e:
        print(f"[RUN {rid}] failed with error: {e}")

df_res = pd.DataFrame(results)
df_res.to_csv(OUT_DIR/"grid_search_results.csv", index=False)
print("Saved grid summary to:", OUT_DIR/"grid_search_results.csv")

# =========================
# Simple post-run plot
# =========================
if not df_res.empty:
    plt.figure(figsize=(8,5))
    for m in sorted(df_res["MODEL_TYPE"].unique()):
        sub = df_res[df_res["MODEL_TYPE"] == m]
        means = sub.groupby("SEQ_LEN")["MAE"].mean().sort_index()
        plt.plot(means.index.astype(str), means.values, marker='o', label=m)
    plt.xlabel("SEQ_LEN"); plt.ylabel("Test MAE"); plt.title("MAE by MODEL_TYPE & SEQ_LEN")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR/"summary_model_seq_mae.png"); plt.close()
    print("Saved:", OUT_DIR/"summary_model_seq_mae.png")

print("Done. All outputs are under:", OUT_DIR)
