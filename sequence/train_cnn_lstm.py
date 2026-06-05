#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustered CNN->LSTM->Attention with proper Train/Val per cluster, LR scheduler, EarlyStopping,
and optional heteroscedastic/asymmetric loss + peak oversampling.

Inputs: time, exog_temp, exog_wind
Features used: exog_temp, exog_wind, OT_prev  (可自行扩展)
Target: OT

Outputs:
 - scaler, cluster scaler/kmeans
 - per-cluster best weights
 - train/val loss曲线
 - 测试集预测CSV/图 & 指标
"""


from pathlib import Path as _Path
import os as _os
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
_os.chdir(_PROJECT_ROOT)

import os, math, random, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim

# ------------------ CONFIG ------------------
# 默认超参来自网格搜索最优 run（out_cnn_lstm_grid_search_revised/run_024, MAE≈67.5W）。
# 本脚本 N_CLUSTERS=1 用于聚类/embedding 单模型；run_024 原用 N_CLUSTERS=2 分工况训练。
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
N_CLUSTERS = 1
OUT_DIR = Path(f"out_cnn_lstm_cluster_{N_CLUSTERS}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据/模型超参（≈ run_024: cnn_lstm_attn）
SEQ_LEN = 8
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 40
EARLY_STOPPING_PATIENCE = 7
DROPOUT_RATE = 0.05
WEIGHT_DECAY = 0.0

CNN_CHANNELS = 32
CNN_KERNEL = 3
LSTM_HID = 64
TRANS_DMODEL = 64            # grid search: d_model = LSTM_HID
NUM_HEADS = 4
NUM_TRANSFORMER_LAYERS = 0

VAL_RATIO = 0.10
TEST_RATIO = 0.20

# 训练策略/损失
USE_HETEROSCEDASTIC = True
USE_ASYMMETRIC_LOSS = False
ASYM_UNDER_WEIGHT = 1.5
OVERSAMPLE_PEAKS = False
PEAK_PERCENTILE = 95
PEAK_WEIGHT_ALPHA = 2.0

PATIENCE = 3
LR_FACTOR = 0.5

LOSS_TYPE = "mae"            # 与 run_024 一致；train/val 共用同一 loss
HUBER_DELTA = 1.0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
plt.rcParams.update({"axes.grid": True})

# ------------------ Utils ------------------
def mape(true, pred):
    true = np.array(true).ravel(); pred = np.array(pred).ravel()
    eps = 1e-9
    return np.mean(np.abs((true - pred) / (np.abs(true) + eps))) * 100.0

def metrics_dict(y_true, y_pred):
    y_true = np.array(y_true).ravel(); y_pred = np.array(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mape_v = mape(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE(%)": mape_v}

# ------------------ READ & PREPROCESS ------------------
print("Reading data...")
data_path = "标注的数据-#67_1.xlsx"
df_raw = pd.read_excel(data_path)

col_candidates = { 'time': ['time','timestamp','date','统计时间'],
                   'OT': ['OT'],
                   'exog_temp': ['exog_temp','Exogenous1','temperature'],
                   'exog_wind': ['exog_wind','Exogenous2','wind_speed'] }

cols = {}
for target, names in col_candidates.items():
    for n in names:
        if n in df_raw.columns:
            cols[target] = n
            break

if 'time' not in cols:
    for c in df_raw.columns:
        if np.issubdtype(df_raw[c].dtype, np.datetime64):
            cols['time'] = c; break
if 'time' not in cols:
    raise ValueError("No time column found. Edit col_candidates or header.")

for req in ['OT','exog_temp','exog_wind']:
    if req not in cols:
        if req in df_raw.columns:
            cols[req] = req
        else:
            found = False
            for c in df_raw.columns:
                l = c.lower()
                if 'ot' in l and req=='OT':
                    cols['OT']=c; found=True; break
                if 'temp' in l and req=='exog_temp':
                    cols['exog_temp']=c; found=True; break
                if 'wind' in l and req=='exog_wind':
                    cols['exog_wind']=c; found=True; break
            if not found:
                raise ValueError(f"Column for {req} not found.")

df = df_raw.rename(columns={ cols[k]: k for k in cols })
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# 数值化/插补
for c in ['exog_temp','exog_wind','OT']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.interpolate(limit=5).ffill().bfill()

# 仅保留需要列 & 构造 OT_prev
df = df[['time','exog_temp','exog_wind','OT']].copy()
df['OT_prev'] = df['OT'].shift(1)
df = df.dropna().reset_index(drop=True)

feat_cols = ['exog_temp','exog_wind','OT_prev']  # 如需增强特征，可自行扩展

print("Rows after preprocess:", len(df))
if len(df) < 200:
    print("Warning: very small dataset; model may overfit or underperform.")

# ------------------ Train/Val/Test split ------------------
n = len(df)
test_size = int(n * TEST_RATIO)
val_size = int(n * VAL_RATIO)
train_size = n - test_size - val_size
if train_size <= SEQ_LEN + 5:
    raise ValueError("Not enough training data for chosen SEQ_LEN.")

train_df = df.iloc[:train_size].reset_index(drop=True)
val_df   = df.iloc[train_size: train_size + val_size].reset_index(drop=True)
test_df  = df.iloc[train_size + val_size:].reset_index(drop=True)

print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))

# ------------------ Clustering (fit on TRAIN to avoid leakage) ------------------
cluster_features = ['exog_temp','exog_wind','OT_prev']
scaler_cluster = StandardScaler().fit(train_df[cluster_features].values)
Xc_train = scaler_cluster.transform(train_df[cluster_features].values)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=20).fit(Xc_train)

# 给全数据打cluster标签（使用train拟合的scaler/kmeans）
Xc_full = scaler_cluster.transform(df[cluster_features].values)
labels_full = kmeans.predict(Xc_full)
df['cluster'] = labels_full

# 持久化聚类对象
joblib.dump(scaler_cluster, OUT_DIR/'scaler_cluster.joblib')
joblib.dump(kmeans, OUT_DIR/'kmeans.joblib')

def assign_cluster_label_from_raw(temp, wind, prev):
    X_new = np.array([[temp, wind, prev]], dtype=float)
    Xs = scaler_cluster.transform(X_new)
    return int(kmeans.predict(Xs)[0])

# ------------------ FEATURE / TARGET SCALING (fit on TRAIN) ------------------
scaler = StandardScaler().fit(train_df[feat_cols].values)
scaler_y = StandardScaler().fit(train_df[['OT']].values)
joblib.dump(scaler, OUT_DIR/'scaler_inputs.joblib')
joblib.dump(scaler_y, OUT_DIR/'scaler_y.joblib')

df_scaled = df.copy()
df_scaled[feat_cols] = scaler.transform(df[feat_cols].values)

# ------------------ Datasets ------------------
class ClusterSeqDataset(Dataset):
    """
    序列窗口需完全处于同一 cluster。
    返回: (seq_scaled, y_scaled, label_idx)
    """
    def __init__(self, df_scaled, start_idx, end_idx, seq_len, feat_cols, cluster_id, scaler_y):
        self.df = df_scaled
        self.start = start_idx
        self.end = end_idx
        self.seq_len = seq_len
        self.feat_cols = feat_cols
        self.cluster_id = cluster_id
        self.scaler_y = scaler_y

        self.valid_idx = []
        for i in range(self.start, self.end - self.seq_len + 1):
            block = self.df.iloc[i: i + self.seq_len + 1]
            if (block['cluster'].values == cluster_id).all():
                self.valid_idx.append(i)

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        i0 = self.valid_idx[idx]
        seq = self.df.iloc[i0: i0 + self.seq_len][self.feat_cols].values.astype(np.float32)
        label_idx = i0 + self.seq_len
        y_raw = float(self.df.iloc[label_idx]['OT'])
        y = float(self.scaler_y.transform([[y_raw]])[0, 0])
        return seq, np.float32(y), int(label_idx)

class SeqDataset(Dataset):
    """ 测试集整体（不按cluster），方便逐样本选对应模型 """
    def __init__(self, df_scaled, start_idx, end_idx, seq_len, feat_cols, scaler_y):
        self.df = df_scaled
        self.start = start_idx
        self.end = end_idx
        self.seq_len = seq_len
        self.feat_cols = feat_cols
        self.scaler_y = scaler_y
        self.n = max(0, (self.end - self.start + 1) - self.seq_len)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i0 = self.start + idx
        seq = self.df.iloc[i0: i0 + self.seq_len][self.feat_cols].values.astype(np.float32)
        label_idx = i0 + self.seq_len
        y_raw = float(self.df.iloc[label_idx]['OT'])
        y = float(self.scaler_y.transform([[y_raw]])[0, 0])
        return seq, np.float32(y), int(label_idx)

# 构建 per-cluster 的 train/val loader
train_start, train_end = 0, train_size - 1
val_start, val_end     = train_size, train_size + val_size - 1
test_start, test_end   = train_size + val_size, n - 1

train_loaders, val_loaders = {}, {}
peak_thresholds = {}

for cluster_id in range(N_CLUSTERS):
    train_ds = ClusterSeqDataset(df_scaled, train_start, train_end, SEQ_LEN, feat_cols, cluster_id, scaler_y)
    val_ds   = ClusterSeqDataset(df_scaled, val_start,   val_end,   SEQ_LEN, feat_cols, cluster_id, scaler_y)

    # 过采样峰值（基于 y label 的分位数）
    if OVERSAMPLE_PEAKS and len(train_ds) > 0:
        ys_train = np.array([df.iloc[i + SEQ_LEN]['OT'] for i in train_ds.valid_idx])
        thr = np.percentile(ys_train, PEAK_PERCENTILE)
        peak_thresholds[cluster_id] = float(thr)
        sample_weights = np.where(ys_train > thr, 1.0 + PEAK_WEIGHT_ALPHA, 1.0).astype(np.float32)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        peak_thresholds[cluster_id] = None

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    train_loaders[cluster_id] = train_loader
    val_loaders[cluster_id]   = val_loader

# 测试 loader（整体）
test_loader = DataLoader(SeqDataset(df_scaled, test_start, test_end, SEQ_LEN, feat_cols, scaler_y),
                         batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ------------------ Model ------------------
class CNN_LSTM_Attention(nn.Module):
    """与 cnn_lstm_grid_search.py 中 CNN_LSTM_Attn 对齐（双卷积 + LSTM + MHA）。"""
    def __init__(self, feat_dim=3, cnn_channels=32, cnn_kernel=3, lstm_hid=64,
                 d_model=64, nhead=4, num_transformer_layers=0, dropout_rate=0.05,
                 heteroscedastic=False):
        super().__init__()
        self.hetero = heteroscedastic

        self.conv1 = nn.Conv1d(feat_dim, cnn_channels, cnn_kernel, padding=cnn_kernel // 2)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, cnn_kernel, padding=cnn_kernel // 2)
        self.act = nn.ReLU()
        self.dropout_cnn = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(cnn_channels, lstm_hid, num_layers=1, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout_rate)
        self.proj_to_d = nn.Linear(lstm_hid, d_model) if lstm_hid != d_model else nn.Identity()

        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout_rate)

        if num_transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=d_model * 2, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                             num_layers=num_transformer_layers)
        else:
            self.transformer_encoder = None

        final_out_dim = 2 if heteroscedastic else 1
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, final_out_dim),
        )

    def forward(self, x):
        c = x.permute(0, 2, 1)
        c = self.act(self.conv1(c))
        c = self.act(self.conv2(c))
        c = self.dropout_cnn(c)
        c = c.permute(0, 2, 1)

        lstm_out, _ = self.lstm(c)
        lstm_out = self.dropout_lstm(lstm_out)
        tr_in = self.proj_to_d(lstm_out)

        attn_out, _ = self.mha(tr_in, tr_in, tr_in, need_weights=False)
        attn_out = self.dropout_attn(attn_out)
        tr_out = self.transformer_encoder(attn_out) if self.transformer_encoder else attn_out
        last = tr_out[:, -1, :]
        out = self.fc(last)
        if self.hetero:
            return out[:, 0], out[:, 1].clamp(-10, 10)
        return out.squeeze(1)

# ------------------ Loss ------------------
def compute_loss(pred_or_mu, target, logvar=None):
    """train/val 共用；与 grid search 的 compute_loss 一致。"""
    if logvar is not None:
        if LOSS_TYPE == "mae":
            base = torch.abs(pred_or_mu - target)
        elif LOSS_TYPE == "huber":
            base = nn.functional.smooth_l1_loss(pred_or_mu, target, reduction='none')
        elif LOSS_TYPE == "mape":
            base = torch.abs((pred_or_mu - target) / (target + 1e-6))
        elif LOSS_TYPE == "smape":
            base = 2.0 * torch.abs(pred_or_mu - target) / (
                torch.abs(pred_or_mu) + torch.abs(target) + 1e-6)
        else:
            base = (pred_or_mu - target) ** 2
        if USE_ASYMMETRIC_LOSS:
            w = torch.where(pred_or_mu - target < 0, ASYM_UNDER_WEIGHT, 1.0)
            base = w * base
        return (0.5 * torch.exp(-logvar) * base + 0.5 * logvar).mean()

    if LOSS_TYPE == "mae":
        return torch.abs(pred_or_mu - target).mean()
    if LOSS_TYPE == "huber":
        return nn.functional.smooth_l1_loss(pred_or_mu, target)
    if LOSS_TYPE == "mape":
        return torch.mean(torch.abs((pred_or_mu - target) / (target + 1e-6)))
    if LOSS_TYPE == "smape":
        return torch.mean(2.0 * torch.abs(pred_or_mu - target) / (
            torch.abs(pred_or_mu) + torch.abs(target) + 1e-6))
    return nn.SmoothL1Loss()(pred_or_mu, target)

def init_output_bias(model, train_loader):
    """用训练集 scaled label 均值初始化输出层 bias。"""
    ys = []
    for i in train_loader.dataset.valid_idx:
        ys.append(float(scaler_y.transform([[df.iloc[i + SEQ_LEN]['OT']]])[0, 0]))
    mean_y = float(np.mean(ys)) if ys else 0.0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.out_features in (1, 2):
                if USE_HETEROSCEDASTIC and m.out_features == 2:
                    m.bias.data[0].fill_(mean_y)
                    m.bias.data[1].fill_(0.0)
                elif m.out_features == 1:
                    m.bias.data.fill_(mean_y)
                break

# ------------------ Train per cluster w/ Val ------------------
models = {}
history = {}

for cluster_id in range(N_CLUSTERS):
    feat_dim = len(feat_cols)
    model = CNN_LSTM_Attention(feat_dim=feat_dim, cnn_channels=CNN_CHANNELS, cnn_kernel=CNN_KERNEL,
                               lstm_hid=LSTM_HID, d_model=TRANS_DMODEL, nhead=NUM_HEADS,
                               num_transformer_layers=NUM_TRANSFORMER_LAYERS, dropout_rate=DROPOUT_RATE,
                               heteroscedastic=USE_HETEROSCEDASTIC).to(DEVICE)
    init_output_bias(model, train_loader)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=LR_FACTOR, patience=PATIENCE)

    train_loader = train_loaders[cluster_id]
    val_loader   = val_loaders[cluster_id]

    best_val = float('inf'); best_state = None; es_cnt = 0
    train_losses, val_losses = [], []

    print(f"\n=== Training cluster {cluster_id} ===")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    for ep in range(1, EPOCHS+1):
        # ----- train -----
        model.train()
        t_loss, t_cnt = 0.0, 0
        for seq, y, _ in train_loader:
            seq = seq.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad()
            if USE_HETEROSCEDASTIC:
                mu, logvar = model(seq)
                loss = compute_loss(mu, y, logvar=logvar)
            else:
                out = model(seq)
                loss = compute_loss(out, y)
            loss.backward()
            opt.step()
            t_loss += loss.item() * seq.size(0); t_cnt += seq.size(0)
        train_epoch = t_loss / max(1, t_cnt)
        train_losses.append(train_epoch)

        # ----- val -----
        model.eval()
        v_loss, v_cnt = 0.0, 0
        with torch.no_grad():
            for seq, y, _ in val_loader:
                seq = seq.to(DEVICE); y = y.to(DEVICE)
                if USE_HETEROSCEDASTIC:
                    mu, logvar = model(seq)
                    loss_v = compute_loss(mu, y, logvar=logvar)
                else:
                    out = model(seq)
                    loss_v = compute_loss(out, y)
                v_loss += loss_v.item() * seq.size(0); v_cnt += seq.size(0)
        val_epoch = v_loss / max(1, v_cnt) if v_cnt>0 else train_epoch
        val_losses.append(val_epoch)

        scheduler.step(val_epoch)
        print(f"[Cluster {cluster_id}] Ep {ep:03d} | train={train_epoch:.6f} | val={val_epoch:.6f}")

        # early stopping
        if val_epoch < best_val - 1e-9:
            best_val = val_epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            es_cnt = 0
            torch.save(best_state, OUT_DIR / f'best_state_cluster{cluster_id}.pt')
        else:
            es_cnt += 1
            if es_cnt >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping cluster {cluster_id} at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    models[cluster_id] = model

    # save final (+ embedding_analysis 期望的文件名)
    torch.save(model.state_dict(), OUT_DIR / f"model_cnn_lstm_att_final_{cluster_id}.pt")
    torch.save(model.state_dict(), OUT_DIR / f"model_run0_cluster{cluster_id}.pt")
    history[cluster_id] = {'train': train_losses, 'val': val_losses}

    # plot loss
    plt.figure(figsize=(8,4))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='train')
    if len(val_losses) > 0:
        plt.plot(range(1, len(val_losses)+1), val_losses, label='val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Cluster {cluster_id} Train/Val Loss')
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / f'loss_cluster{cluster_id}.png'); plt.close()

# ------------------ Evaluate on test ------------------
print("\n=== Testing ===")
if len(test_loader) == 0:
    raise ValueError("Test loader has no data — check split or SEQ_LEN!")

models = {k: v.eval() for k, v in models.items()}

preds, trues, label_indices = [], [], []

with torch.no_grad():
    for seq, y, idxs in test_loader:
        # 对每个样本，用“最后一个输入时间点”的 cluster 选模型
        for i in range(seq.size(0)):
            label_idx = int(idxs[i].item())
            last_input_idx = label_idx - 1  # 序列最后一个输入对应的全局索引
            cluster_id = int(df.loc[last_input_idx, 'cluster'])
            model = models.get(cluster_id, None)
            if model is None:
                continue

            x = seq[i:i+1].to(DEVICE)
            if USE_HETEROSCEDASTIC:
                mu, logvar = model(x)
                mu_scaled = mu.cpu().numpy().ravel()
            else:
                mu_scaled = model(x).cpu().numpy().ravel()
            out = scaler_y.inverse_transform(mu_scaled.reshape(-1, 1)).ravel()

            preds.append(out)
            trues.append(np.array([float(df.iloc[label_idx]['OT'])]))
            label_indices.append(label_idx)

pred = np.concatenate(preds)
true = np.concatenate(trues)
label_indices = np.array(label_indices)

# metrics
metrics_test = metrics_dict(true, pred)
print("Test metrics:", metrics_test)

# save predictions
test_times = df.loc[label_indices, 'time'].values
out_df = pd.DataFrame({'time': test_times, 'OT_true': true, 'OT_pred': pred})
out_csv = OUT_DIR/'test_predictions_cnn_lstm_att.csv'
out_df.to_csv(out_csv, index=False)
print("Saved test predictions to:", out_csv)

# plot
plt.figure(figsize=(12,4))
plt.plot(test_times, true, label='OT_true')
plt.plot(test_times, pred, label='OT_pred', alpha=0.85)
plt.xlabel('time'); plt.ylabel('OT'); plt.title('OT: true vs predicted')
plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR/'ot_true_vs_pred.png'); plt.close()

# save metrics & config
pd.DataFrame(metrics_test, index=['CNN_LSTM_Att']).T.to_csv(OUT_DIR/'test_metrics.csv')
config = {
    "source": "grid_search run_024 defaults (out_cnn_lstm_grid_search_revised)",
    "SEQ_LEN": SEQ_LEN, "BATCH_SIZE": BATCH_SIZE, "LR": LR, "EPOCHS": EPOCHS,
    "WEIGHT_DECAY": WEIGHT_DECAY, "LOSS_TYPE": LOSS_TYPE,
    "CNN_CHANNELS": CNN_CHANNELS, "LSTM_HID": LSTM_HID, "TRANS_DMODEL": TRANS_DMODEL,
    "NUM_HEADS": NUM_HEADS, "NUM_TRANSFORMER_LAYERS": NUM_TRANSFORMER_LAYERS,
    "DROPOUT_RATE": DROPOUT_RATE, "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
    "VAL_RATIO": VAL_RATIO, "TEST_RATIO": TEST_RATIO, "N_CLUSTERS": N_CLUSTERS,
    "USE_HETEROSCEDASTIC": USE_HETEROSCEDASTIC, "USE_ASYMMETRIC_LOSS": USE_ASYMMETRIC_LOSS,
    "ASYM_UNDER_WEIGHT": ASYM_UNDER_WEIGHT, "OVERSAMPLE_PEAKS": OVERSAMPLE_PEAKS,
    "PEAK_PERCENTILE": PEAK_PERCENTILE, "PEAK_WEIGHT_ALPHA": PEAK_WEIGHT_ALPHA,
}
with open(OUT_DIR/'model_config.json','w') as f:
    json.dump(config, f, indent=2)
print("All outputs saved in:", OUT_DIR)
