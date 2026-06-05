#!/usr/bin/env python3
"""
cnn_lstm_attention_ot.py

End-to-end model: CNN -> LSTM -> MultiHeadAttention -> Dense
Inputs: exog_temp, exog_wind (time series)
Target: OT

Outputs:
 - model weights (best by val loss)
 - training/validation loss plot
 - test predictions CSV (time, OT_true, OT_pred)
 - metrics (MAE, MSE, RMSE, MAPE) printed and saved

Requirements: numpy, pandas, scikit-learn, matplotlib, torch, joblib
"""

import os
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import time

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ------------------ CONFIG ------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path("out_cnn_lstm_att-cluster-1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Data / model hyperparams
SEQ_LEN = 16          # number of timesteps in input sequence
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 80    # 40步训一轮：40-135，80-115，120-110
EARLY_STOPPING_PATIENCE = 7
DROPOUT_RATE = 0.1  # Dropout比例

CNN_CHANNELS = 16      # conv channels
CNN_KERNEL = 3
LSTM_HID = 128
TRANS_DMODEL = 128     # must match LSTM_HID or project to this
NUM_HEADS = 4
NUM_TRANSFORMER_LAYERS = 1

VAL_RATIO = 0.00
TEST_RATIO = 0.20

PATIENCE = 3        # 学习率衰减等待轮数
LR_FACTOR = 0.5     # 学习率衰减倍数

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

plt.rcParams.update({"axes.grid": True})

# ------------------ UTIL ------------------
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

# require minimal columns
if 'time' not in cols:
    # try to detect a datetime-like column
    for c in df_raw.columns:
        if np.issubdtype(df_raw[c].dtype, np.datetime64):
            cols['time'] = c; break
if 'time' not in cols:
    raise ValueError("No time column found. Edit col_candidates or CSV header.")

for req in ['OT','exog_temp','exog_wind']:
    if req not in cols:
        # try direct name
        if req in df_raw.columns:
            cols[req] = req
        else:
            # attempt some common names already in df columns
            found = False
            for c in df_raw.columns:
                if c.lower().find('ot')>=0 and req=='OT':
                    cols['OT']=c; found=True; break
                if 'temp' in c.lower() and req=='exog_temp':
                    cols['exog_temp']=c; found=True; break
                if 'wind' in c.lower() and req=='exog_wind':
                    cols['exog_wind']=c; found=True; break
            if not found:
                raise ValueError(f"Column for {req} not found. Please update column names.")

# rename selected columns to canonical names
df = df_raw.rename(columns={ cols[k]: k for k in cols })
# parse time
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# numeric conversions
df['exog_temp'] = pd.to_numeric(df['exog_temp'], errors='coerce')
df['exog_wind'] = pd.to_numeric(df['exog_wind'], errors='coerce')
df['OT'] = pd.to_numeric(df['OT'], errors='coerce')

# interpolate small gaps and forward/backfill
df = df.interpolate(limit=5).ffill().bfill()

# keep only the necessary columns (user requested only temp & wind as inputs)
df = df[['time','exog_temp','exog_wind','OT']].copy()
df['OT_prev'] = df['OT'].shift(1)
feat_cols = ['exog_temp','exog_wind','OT_prev']
df = df.dropna().reset_index(drop=True)

# simple sanity
print("Rows after read & fill:", len(df))
if len(df) < 200:
    print("Warning: very small dataset; model may overfit or not train well.")

# ----------------- Clustering (icing/off) -----------------
cluster_features = feat_cols
X_cluster = df[cluster_features].values
scaler_cluster = StandardScaler()
Xc_scaled = scaler_cluster.fit_transform(X_cluster)
kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=20)

def assign_cluster_label(temp, wind, prev):
    X_new = scaler_cluster.transform(np.array([[temp, wind, prev]], dtype=float))
    cluster_id = int(kmeans.predict(X_new)[0])
    return cluster_id

labels = kmeans.fit_predict(Xc_scaled)
df['cluster'] = labels
cluster_summary = df.groupby('cluster')[feat_cols].mean()
cluster_low = cluster_summary['exog_temp'].idxmin()
label_map = {cluster_low: 'icing_or_off', 1-cluster_low: 'normal'}
df['cluster_label'] = df['cluster'].map(label_map)

# ------------------ FEATURE SCALING ------------------
# We'll scale inputs (temp & wind) using StandardScaler fit on train portion (we'll split first)
n = len(df)
test_size = int(n * TEST_RATIO)
val_size = int(n * VAL_RATIO)
train_size = n - test_size - val_size

if train_size <= SEQ_LEN + 5:
    raise ValueError("Not enough training data for chosen SEQ_LEN. Reduce SEQ_LEN or test/val ratios.")

train_df = df.iloc[:train_size].reset_index(drop=True)
val_df = df.iloc[train_size: train_size + val_size].reset_index(drop=True)
test_df = df.iloc[train_size + val_size : ].reset_index(drop=True)

print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))

scaler = StandardScaler().fit(train_df[feat_cols].values)
# store scaler
joblib.dump(scaler, OUT_DIR/'scaler_inputs.joblib')

# apply scaling to full df for sequence construction
df_scaled = df.copy()
df_scaled[feat_cols] = scaler.transform(df[feat_cols].values)

# ------------------ SEQ DATASET ------------------
class SeqDataset(Dataset):
    """
    returns (seq, y, time_idx)
    seq: shape (seq_len, feat_dim)
    y: scalar OT at index (start+seq_len)
    time_idx: the global index of the label (useful for alignment)
    """
    def __init__(self, df_scaled, start_idx, end_idx, seq_len, feat_cols):
        self.df = df_scaled
        self.start = start_idx
        self.end = end_idx
        self.seq_len = seq_len
        self.feat_cols = feat_cols
        # number of windows: (end - start + 1) - seq_len
        self.n = max(0, (self.end - self.start + 1) - self.seq_len)
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        idx0 = self.start + idx
        seq = self.df.iloc[idx0: idx0 + self.seq_len][self.feat_cols].values.astype(np.float32)
        label_idx = idx0 + self.seq_len
        y = self.df.iloc[label_idx]['OT'].astype(np.float32)
        return seq, y, int(label_idx)

def make_dataset_by_cluster(df_scaled, start_idx, end_idx, seq_len, feat_cols, cluster_id):
    # 返回只包含该cluster数据的dataset
    cluster_idx = df_scaled.iloc[start_idx:end_idx+1][df_scaled['cluster'] == cluster_id].index
    # 找出能构成完整序列的索引
    valid_idx = []
    for i in cluster_idx:
        if i + seq_len < end_idx+1 and all(df_scaled.iloc[i:i+seq_len+1]['cluster'] == cluster_id):
            valid_idx.append(i)
    class ClusterSeqDataset(Dataset):
        def __len__(self):
            return len(valid_idx)
        def __getitem__(self, idx):
            idx0 = valid_idx[idx]
            seq = df_scaled.iloc[idx0: idx0 + seq_len][feat_cols].values.astype(np.float32)
            y = df_scaled.iloc[idx0 + seq_len]['OT'].astype(np.float32)
            return seq, y, int(idx0 + seq_len)
    return ClusterSeqDataset()

'''
# create loaders for train / val / test using sequence windows wholly inside each split
train_dataset = SeqDataset(df_scaled, 0, train_size - 1, SEQ_LEN, feat_cols)
val_dataset = SeqDataset(df_scaled, train_size, train_size + val_size - 1, SEQ_LEN, feat_cols)
test_dataset = SeqDataset(df_scaled, train_size + val_size, n - 1, SEQ_LEN, feat_cols)

print("Seq sizes (train,val,test):", len(train_dataset), len(val_dataset), len(test_dataset))
if len(test_dataset) == 0:
    raise ValueError("Test dataset has 0 sequence samples. Adjust SEQ_LEN or split sizes.")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
'''

train_loaders = {}
# val_loader = DataLoader(SeqDataset(df_scaled, train_size, train_size + val_size - 1, SEQ_LEN, feat_cols), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader = DataLoader(SeqDataset(df_scaled, train_size + val_size, train_size + val_size + test_size - 1, SEQ_LEN, feat_cols), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
for cluster_id in [0,1]:
    train_loaders[cluster_id] = DataLoader(make_dataset_by_cluster(df_scaled, 0, train_size - 1, SEQ_LEN, feat_cols, cluster_id),
                                           batch_size=BATCH_SIZE, shuffle=True)


# ------------------ MODEL: CNN -> LSTM -> Attention -> FC ------------------
class CNN_LSTM_Attention(nn.Module):
    def __init__(self, feat_dim=3, cnn_channels=64, cnn_kernel=3, lstm_hid=128,
                 d_model=128, nhead=4, num_transformer_layers=1, out_dim=1, dropout_rate = 0.3):
        super().__init__()
        # CNN branch
        self.conv1 = nn.Conv1d(in_channels=feat_dim, out_channels=cnn_channels, kernel_size=cnn_kernel, padding=cnn_kernel//2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=cnn_kernel, padding=cnn_kernel//2)
        self.dropout_cnn = nn.Dropout(dropout_rate)
        # project conv output (per time step) to lstm input dim if needed
        # We'll apply conv across sequence (input shape for conv: B, feat, seq_len)
        # to feed LSTM we need (B, seq_len, conv_channels) -> permute accordingly
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hid, num_layers=1, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout_rate)
        # project LSTM hidden dim to transformer d_model if mismatch
        if lstm_hid != d_model:
            self.proj_to_d = nn.Linear(lstm_hid, d_model)
        else:
            self.proj_to_d = None
        # MultiHeadAttention expects (seq_len, batch, d_model) or use TransformerEncoder; here we use MultiheadAttention
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout_rate)
        # optional small transformer encoder stack for extra processing
        if num_transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward= d_model*2, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        else:
            self.transformer_encoder = None
        # final MLP
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        """
        x: (B, seq_len, feat_dim)
        returns: (B, 1)
        """
        # conv expects (B, feat_dim, seq_len)
        c = x.permute(0,2,1)                  # B, feat, seq_len
        c = self.act(self.conv1(c))           # B, cnn_channels, seq_len
        # c = self.pool(c)                      # B, cnn_channels, 1
        c = self.act(self.conv2(c))           # B, cnn_channels, seq_len
        c = self.dropout_cnn(c)               # B, cnn_channels, seq_len
        # to LSTM: (B, seq_len, cnn_channels)
        c = c.permute(0,2,1)
        lstm_out, _ = self.lstm(c)            # B, seq_len, lstm_hid
        lstm_out = self.dropout_lstm(lstm_out)
        # project to d_model if needed
        if self.proj_to_d is not None:
            tr_in = self.proj_to_d(lstm_out)  # B, seq_len, d_model
        else:
            tr_in = lstm_out                  # assume dims match d_model
        # Multi-head attention: use queries=keys=values = tr_in (self-attention)
        # nn.MultiheadAttention with batch_first=True accepts (B, seq_len, d_model)
        attn_out, attn_weights = self.mha(tr_in, tr_in, tr_in, need_weights=True)  # B, seq_len, d_model
        attn_out = self.dropout_attn(attn_out)
        if self.transformer_encoder is not None:
            tr_out = self.transformer_encoder(attn_out)  # B, seq_len, d_model
        else:
            tr_out = attn_out
        # pool/truncate: use last time-step representation
        last = tr_out[:, -1, :]               # B, d_model
        out = self.fc(last)                   # B, 1
        return out.squeeze(1)                 # B

# ------------------ TRAIN LOOP ------------------
models = {}
for cluster_id in [0,1]:
    print(f'Training model for cluster {cluster_id}:')
    model = CNN_LSTM_Attention(feat_dim=3, cnn_channels=CNN_CHANNELS, cnn_kernel=CNN_KERNEL,
                            lstm_hid=LSTM_HID, d_model=TRANS_DMODEL, nhead=NUM_HEADS,
                            num_transformer_layers=NUM_TRANSFORMER_LAYERS, dropout_rate=DROPOUT_RATE).to(DEVICE)


    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    criterion = nn.SmoothL1Loss()  # 损失函数（比MSE更稳健）
    # 学习率调度器（验证集loss连续PATIENCE轮不下降 -> lr乘LR_FACTOR）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=LR_FACTOR, 
                                                    patience=PATIENCE)

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    train_losses = []
    # val_losses = []

    start_time = time.time()
    print("Starting training...")
    for ep in range(1, EPOCHS+1):
        model.train()
        t_loss = 0.0
        count = 0
        for seq, y, _ in train_loaders[cluster_id]:
            seq = seq.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)
            opt.zero_grad()
            out = model(seq).unsqueeze(1)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            t_loss += loss.item() * seq.size(0)
            count += seq.size(0)
        train_epoch_loss = t_loss / max(1, count)
        train_losses.append(train_epoch_loss)

        # # validation
        # model.eval()
        # v_loss = 0.0
        # vcount = 0
        # cluster_ids = []
        # with torch.no_grad():
        #     for seq, y, _ in val_loader:
        #         seq = seq.to(DEVICE)
        #         y = y.to(DEVICE).unsqueeze(1)
        #         # get cluster label for each sequence in the batch
        #         cluster_labels = [int(assign_cluster_label(s[-1,0], s[-1,1])) for s in seq.cpu().numpy()]
        #         cluster_ids.extend(cluster_labels)
        #         # predict using the corresponding model
        #         # predict using the corresponding model and accumulate predictions and truths
        #         for i, cluster_id in enumerate(cluster_labels):
        #             model = models[cluster_id]
        #             out = model(seq[[i]].to(DEVICE)).cpu().numpy().ravel()
        #             v_loss += criterion(torch.tensor(out).unsqueeze(0), y[i].unsqueeze(0)).item()
        #             vcount += 1
        # val_epoch_loss = v_loss / max(1, vcount)
        # val_losses.append(val_epoch_loss)
        # scheduler.step(val_epoch_loss)
        scheduler.step(train_epoch_loss)

        # print(f"Epoch {ep:03d} | train_loss={train_epoch_loss:.6f} | val_loss={val_epoch_loss:.6f}")
        print(f"Epoch {ep:03d} | train_loss={train_epoch_loss:.6f}")
        # # early stopping
        # if val_epoch_loss < best_val_loss - 1e-9:
        #     best_val_loss = val_epoch_loss
        #     best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        #     patience = 0
        #     # save intermediate best
        #     torch.save(best_state, OUT_DIR/'best_state.pt')
        # else:
        #     patience += 1
        #     if patience >= EARLY_STOPPING_PATIENCE:
        #         print(f"Early stopping at epoch {ep}")
        #         break
        if train_epoch_loss < best_val_loss - 1e-9:
            best_val_loss = train_epoch_loss
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            patience = 0
            # save intermediate best
            torch.save(best_state, OUT_DIR/'best_state.pt')
        else:
            patience += 1

    elapsed = time.time() - start_time
    print(f"Training_{cluster_id} finished in {elapsed:.1f}s; best val loss = {best_val_loss:.6f}")

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
        models[cluster_id] = model

    # save final model
    torch.save(model.state_dict(), OUT_DIR/'model_cnn_lstm_att_final_{cluster_id}.pt')
def predict_for_new_data(temp, wind, seq):
    cluster_id = kmeans.predict([[temp, wind]])[0]
    model = models[cluster_id]
    model.eval()
    with torch.no_grad():
        pred = model(seq.to(DEVICE))
    return pred.cpu().numpy()
# ------------------ Plot train/val loss ------------------
plt.figure(figsize=(8,4))
plt.plot(range(1, len(train_losses)+1), train_losses, label='train_loss')
# plt.plot(range(1, len(val_losses)+1), val_losses, label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.title('Training & Validation Loss')
plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR/'loss_curve.png'); plt.close()

# ------------------ Evaluate on test ------------------
# model.eval()
# preds = []
# trues = []
# times_idx = []
# cluster_ids = []
# with torch.no_grad():
#     for seq, y, label_idx in test_loader:
#         seq = seq.to(DEVICE)
#         # get cluster label for each sequence in the batch
#         cluster_labels = [assign_cluster_label(s[-1,0], s[-1,1], s[-1,2]) for s in seq.cpu().numpy()]
#         cluster_ids.extend(cluster_labels)
#         # predict using the corresponding model
#         for i, cluster_id in enumerate(cluster_labels):
#             model = models[cluster_id]
#             out = model(seq[[i]].to(DEVICE)).cpu().numpy().ravel()
#             preds.append(out)
#             trues.append(y[i].cpu().numpy().ravel())
#             times_idx.append(label_idx[i].cpu().numpy().ravel())
# pred = np.concatenate(preds)
# true = np.concatenate(trues)
# label_indices = np.concatenate(times_idx)
model.eval()
preds = []
trues = []
times_idx = []
cluster_ids = []
if len(test_loader) == 0:
    raise ValueError("Test loader has no data — check dataset split!")


with torch.no_grad():
    for seq, y, label_idx in test_loader:
        seq = seq.to(DEVICE)

        # 计算每个样本的聚类标签
        cluster_labels = [assign_cluster_label(s[-1,0], s[-1,1], s[-1,2]) for s in seq.cpu().numpy()]
        cluster_ids.extend(cluster_labels)

        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id not in models:
                print(f"Warning: No model for cluster {cluster_id}, skipping sample {i}")
                continue

            sub_model = models[cluster_id]
            out = sub_model(seq[[i]]).cpu().numpy().ravel()
            preds.append(out)
            trues.append(y[i].cpu().numpy().ravel())
            times_idx.append(label_idx[i].cpu().numpy().ravel())

if not preds:
    raise RuntimeError("Test loop produced no predictions! Check your clustering and test data.")

pred = np.concatenate(preds)
true = np.concatenate(trues)
label_indices = np.concatenate(times_idx)



metrics_test = metrics_dict(true, pred)
print("Test metrics:", metrics_test)

# save predictions with times
test_times = df.loc[label_indices, 'time'].values
out_df = pd.DataFrame({
    'time': test_times,
    'OT_true': true,
    'OT_pred': pred
})
out_csv = OUT_DIR/'test_predictions_cnn_lstm_att.csv'
out_df.to_csv(out_csv, index=False)
print("Saved test predictions to:", out_csv)

pred_col = f'OT_pred'
plt.figure(figsize=(12,4))
plt.plot(test_times, true, label='OT_true')
plt.plot(test_times, pred, label=f'OT_pred', alpha=0.8)
plt.xlabel('time')
plt.ylabel('OT')
plt.title(f'OT: true vs predicted)')
plt.legend()
plt.tight_layout()
fname = OUT_DIR / f"ot_true_vs_pred.png"
plt.savefig(fname)
plt.close()

# save metrics
pd.DataFrame(metrics_test, index=['CNN_LSTM_Att']).T.to_csv(OUT_DIR/'test_metrics.csv')
print("Saved metrics to:", OUT_DIR/'test_metrics.csv')

# ------------------ Save model config & README ------------------
config = {
    "SEQ_LEN": SEQ_LEN, "BATCH_SIZE": BATCH_SIZE, "LR": LR, "EPOCHS": EPOCHS,
    "CNN_CHANNELS": CNN_CHANNELS, "LSTM_HID": LSTM_HID, "TRANS_DMODEL": TRANS_DMODEL,
    "NUM_HEADS": NUM_HEADS, "NUM_TRANSFORMER_LAYERS": NUM_TRANSFORMER_LAYERS, "DROPOUT_RATE": DROPOUT_RATE,"EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE
}
import json
with open(OUT_DIR/'model_config.json','w') as f:
    json.dump(config, f, indent=2)
print("Saved config to model_config.json")

print("All outputs saved in:", OUT_DIR)
