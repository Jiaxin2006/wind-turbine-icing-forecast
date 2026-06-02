"""
wind_ot_model_compare.py

- Reads a CSV file with turbine measurements.
- Predicts OT column using multiple models:
    RandomForest, SVR, LSTM, 1D-CNN, Transformer
- Performs unsupervised clustering (icing/off) like before.
- Saves plots (no Chinese labels) and CSV with predictions and metrics.
Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn torch torchvision tqdm lightgbm
"""

import os
import random
import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")

# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------- CONFIG -----------------
OUT_DIR = Path("output_ot_models-new")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VAL_RATIO = 0.1  

# Map your column names if they differ
col_map = {
    "time": "统计时间",
    "OT": "OT",
    "exog_temp": "Exogenous1",
    "exog_wind": "Exogenous2",
    "gen_speed": "平均发电机转速(rpm)",
    "I_A": "平均网侧A相电流(A)",
    "I_B": "平均网侧B相电流(A)",
    "I_C": "平均网侧C相电流(A)",
    "V_A": "平均网侧A相电压(V)",
    "V_B": "平均网侧B相电压(V)",
    "V_C": "平均网侧C相电压(V)",
}

POWER_PF = 0.95
TEST_RATIO = 0.20
SEQ_LEN = 12         # sequence length for LSTM/CNN/Transformer (e.g., 12 minutes)
BATCH_SIZE = 32
EPOCHS = 20          # reduce if CPU-only
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6   # for MAPE denom

# ----------------- UTILITIES -----------------
def metric_table(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    nrmse = rmse / np.var(y_true) + 1e-9
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + EPS))) * 100.0
    smape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + EPS))*100.0)
    mase = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true - y_true.mean()) + EPS)))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "NRMSE": nrmse, "R2": r2, "MAPE": mape, "SMAPE": smape, "MASE": mase}

# ----------------- READ & PREP -----------------
# 读取数据
data_path = "标注的数据-#67_1.xlsx"
df = pd.read_excel(data_path)
# rename columns
reverse_map = {v: k for k, v in col_map.items()}
df = df.rename(columns={orig: new for orig, new in reverse_map.items() if orig in df.columns})

if "time" in df.columns:
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
else:
    raise ValueError("Time column not found. Check col_map mapping for 'time'.")

# check required columns
required = ['OT','exog_temp','exog_wind']
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing required column in input CSV: {c}")

# numeric
num_cols = required
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.interpolate(limit=5).fillna(method='bfill').fillna(method='ffill')

# estimate active power as before (kW)
# features: raw & lags & rolling
df['temp_roll_3'] = df['exog_temp'].rolling(3, min_periods=1).mean()
df['wind_roll_3'] = df['exog_wind'].rolling(3, min_periods=1).mean()

# create lag features (for classical models)
lags = [1,2,3,6,12]
for lag in lags:
    df[f'OT_lag_{lag}'] = df['OT'].shift(lag)
    df[f'temp_lag_{lag}'] = df['exog_temp'].shift(lag)
    df[f'wind_lag_{lag}'] = df['exog_wind'].shift(lag)

df = df.dropna().reset_index(drop=True)  # drop initial rows lost to lagging

# ----------------- Train/Test split (time-based) -----------------
n = len(df)
test_size = int(n * TEST_RATIO)
val_size = int(n * VAL_RATIO)
train_df = df.iloc[:-test_size-val_size].copy()
val_df = df.iloc[-test_size - val_size:-test_size].copy()
test_df = df.iloc[-test_size:].copy()

# FEATURES for classical models
# feature_cols_classical = ['exog_temp','exog_wind','temp_roll_3','wind_roll_3'] + \
#                          [f'OT_lag_{l}' for l in lags] + [f'temp_lag_{l}' for l in lags] + [f'wind_lag_{l}' for l in lags]

feature_cols_classical = ['exog_temp','exog_wind','OT_lag_1']
X_train_cl = train_df[feature_cols_classical].values
X_val_cl  = val_df[feature_cols_classical].values
X_test_cl  = test_df[feature_cols_classical].values
y_train = train_df['OT'].values
y_val  = val_df['OT'].values
y_test  = test_df['OT'].values

scaler_cl = StandardScaler()
X_train_cl_s = scaler_cl.fit_transform(X_train_cl)
X_val_cl_s  = scaler_cl.transform(X_val_cl)     
X_test_cl_s  = scaler_cl.transform(X_test_cl)

# ----------------- Clustering (icing/off) -----------------
cluster_features = ['exog_temp','exog_wind']
scaler_cluster = StandardScaler()
Xc_train = scaler_cluster.fit_transform(train_df[cluster_features].values)
kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=20)
kmeans.fit(Xc_train)
labels = kmeans.predict(scaler_cluster.transform(df[cluster_features].values))
df['cluster'] = labels

train_cluster_labels = kmeans.predict(scaler_cluster.transform(train_df[cluster_features].values))
cluster_summary = (
    train_df.assign(cluster=train_cluster_labels)
    .groupby('cluster')[['exog_temp','exog_wind']]
    .mean()
)
cluster_low = cluster_summary['exog_temp'].idxmin()
label_map = {cluster_low: 'icing_or_off', 1-cluster_low: 'normal'}
df['cluster_label'] = df['cluster'].map(label_map)

# ----------------- CLASSICAL MODELS -----------------
results = {}
# 
# RandomForest
rf = RandomForestRegressor(n_estimators=10, random_state=SEED, n_jobs=-1)
rf.fit(X_train_cl_s, y_train)
rf_pred = rf.predict(X_test_cl_s)
results['RandomForest'] = metric_table(y_test, rf_pred)
joblib.dump((rf, scaler_cl, feature_cols_classical), OUT_DIR / "rf_model.joblib")

# # SVR (Support Vector Regression) - use RBF kernel
# svr = SVR(kernel='rbf', C=10.0, gamma='scale')
# svr.fit(X_train_cl_s, y_train)
# svr_pred = svr.predict(X_test_cl_s)
# results['SVR'] = metric_table(y_test, svr_pred)
# joblib.dump((svr, scaler_cl, feature_cols_classical), OUT_DIR / "svr_model.joblib")

# + VAL tuning, grid-search 在 train（使用 TimeSeriesSplit），并用 val 或最后把最佳模型用 train+val 合并重训练再评估 test
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {'C':[1], 'epsilon':[0.1], 'gamma':['scale','auto']}
pipe = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf'))])
grid = GridSearchCV(pipe, {'svr__C':param_grid['C'], 'svr__epsilon':param_grid['epsilon'], 'svr__gamma':param_grid['gamma']},
                    cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
grid.fit(X_train_cl, y_train)   # NOTE: pass raw X_train_cl (pipe will scale)
best_svr_pipe = grid.best_estimator_

# Optionally retrain best_svr on train+val for final model
X_train_plus_val = np.vstack([X_train_cl, X_val_cl])
y_train_plus_val = np.concatenate([y_train, y_val])
best_svr_pipe.fit(X_train_plus_val, y_train_plus_val)
svr_pred = best_svr_pipe.predict(X_test_cl)
joblib.dump((best_svr_pipe, feature_cols_classical), OUT_DIR/"svr_model.joblib")

results['SVR'] = metric_table(y_test, svr_pred)
# (Optional) LightGBM could be added here if installed

# ----------------- SEQUENCE DATASETS FOR NN MODELS -----------------
class SeqDataset(Dataset):
    def __init__(self, df_full, idx_start, idx_end, seq_len, feature_cols, target_col='OT'):
        self.df = df_full
        self.start = idx_start
        self.end = idx_end
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col
    def __len__(self):
        return max(0, self.end - self.start - self.seq_len + 1)
    def __getitem__(self, idx):
        idx0 = self.start + idx
        seq = self.df.iloc[idx0: idx0 + self.seq_len][self.feature_cols].values.astype(np.float32)
        y = self.df.iloc[idx0 + self.seq_len][self.target_col].astype(np.float32)
        return seq, y

# features for sequence models
feature_cols_seq = ['exog_temp','exog_wind']
scaler_seq = StandardScaler()
df_seq_scaled_vals = scaler_seq.fit(train_df[feature_cols_seq].values).transform(df[feature_cols_seq].values)
df_seq_scaled = df.copy()
df_seq_scaled[feature_cols_seq] = df_seq_scaled_vals

# indices
train_seq_start = 0
train_size = len(train_df)
train_seq_end   = train_size - 1
val_seq_start   = train_size
val_seq_end     = train_size + val_size - 1
test_seq_start  = train_size + val_size
test_seq_end    = n - 1

train_dataset = SeqDataset(df_seq_scaled, train_seq_start, train_seq_end, SEQ_LEN, feature_cols_seq, 'OT')
val_dataset   = SeqDataset(df_seq_scaled, val_seq_start, val_seq_end, SEQ_LEN, feature_cols_seq, 'OT')
test_dataset  = SeqDataset(df_seq_scaled, test_seq_start, test_seq_end, SEQ_LEN, feature_cols_seq, 'OT')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ----------------- NEURAL NETWORK MODELS -----------------
# Common training loop
def train_model_torch(model, optimizer, train_loader, val_loader, epochs=EPOCHS, device=DEVICE):
    criterion = nn.MSELoss()
    model.to(device)
    best_state = None
    best_loss = float('inf')
    for ep in range(epochs):
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
        # optional val
        if val_loader is None:
            val_loss = tloss
        else:
            model.eval()
            vloss = 0.0
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv = xv.to(device); yv = yv.to(device).unsqueeze(1)
                    outv = model(xv)
                    vloss += criterion(outv, yv).item() * xv.size(0)
            val_loss = vloss / len(val_loader.dataset)
        # save best
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
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

# LSTM regressor
class LSTMReg(nn.Module):
    def __init__(self, input_dim, hid=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hid, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hid, 32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x):
        # x: (B, seq_len, feat)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last step
        return self.fc(out)

# 1D CNN regressor
class CNN1DReg(nn.Module):
    def __init__(self, input_dim, hid=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hid, hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(hid, 32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x):
        # x: (B, seq_len, feat) -> conv needs (B, feat, seq_len)
        x = x.permute(0,2,1)
        out = self.conv(x)
        return self.fc(out)

# Simple Transformer encoder regressor
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        divterm = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * divterm)
        pe[:, 1::2] = torch.cos(pos * divterm)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerReg(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x):
        # x: (B, seq_len, feat)
        x = self.input_fc(x)
        x = self.pos_enc(x)
        out = self.transformer(x)  # (B, seq_len, d_model)
        out = out[:, -1, :]  # last token
        return self.fc(out)

# Train LSTM
input_dim = len(feature_cols_seq)
lstm = LSTMReg(input_dim=input_dim, hid=64)
optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
lstm = train_model_torch(lstm, optimizer, train_loader, val_loader, epochs=EPOCHS)
y_true_lstm, y_pred_lstm = predict_torch(lstm, test_loader)
results['LSTM'] = metric_table(y_true_lstm, y_pred_lstm)
torch.save({'model_state': lstm.state_dict(), 'scaler': scaler_seq, 'feat': feature_cols_seq}, OUT_DIR/"lstm.pt")

# Train CNN1D
cnn = CNN1DReg(input_dim=input_dim, hid=64)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
cnn = train_model_torch(cnn, optimizer, train_loader, val_loader, epochs=EPOCHS)
y_true_cnn, y_pred_cnn = predict_torch(cnn, test_loader)
results['CNN1D'] = metric_table(y_true_cnn, y_pred_cnn)
torch.save({'model_state': cnn.state_dict(), 'scaler': scaler_seq, 'feat': feature_cols_seq}, OUT_DIR/"cnn1d.pt")

# Train Transformer
transformer = TransformerReg(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
optimizer = torch.optim.Adam(transformer.parameters(), lr=LR)
transformer = train_model_torch(transformer, optimizer, train_loader, val_loader, epochs=EPOCHS)
y_true_tr, y_pred_tr = predict_torch(transformer, test_loader)
results['Transformer'] = metric_table(y_true_tr, y_pred_tr)
torch.save({'model_state': transformer.state_dict(), 'scaler': scaler_seq, 'feat': feature_cols_seq}, OUT_DIR/"transformer.pt")

# ----------------- AGGREGATE PREDICTIONS AND SAVE -----------------
# For classical models we have predictions aligned to test_df rows
# For NN sequence models, test_dataset starts at index len(train_df)-SEQ_LEN relative to df; produce indices mapping

# 1) Create a DataFrame for the test set rows used for evaluation (time index)
test_rows = df.iloc[-test_size:].reset_index(drop=True).copy()
# For classical preds:
test_rows['OT_pred_RandomForest'] = rf_pred
test_rows['OT_pred_SVR'] = svr_pred

# For sequence models:
# Build predictions where sequence models produced predictions: they start at index (len(train_df)-SEQ_LEN) ... up to len(df)-SEQ_LEN-1 etc.
# We already used test_dataset built from df_seq_scaled with start=len(train_df)-SEQ_LEN
# Let's create array of timestamps corresponding to sequence model outputs:
seq_start_idx = len(train_df)  # prediction at time index seq_start_idx ... corresponds to df.iloc[seq_start_idx: ...]
# But our test_dataset yields samples for idx0 in [len(train_df)-SEQ_LEN, len(df)-SEQ_LEN-1]
# After careful inspection, simpler approach: align by time index using values from predict_torch results length
len_seq_preds = len(y_pred_lstm)
# The time indices for sequence preds start at index (len(train_df)) and go forward len_seq_preds steps
seq_pred_indices = list(range(len(train_df), len(train_df) + len_seq_preds))
# But test_rows correspond to last test_size rows of df. Let's build a mapping:
for name, arr in [('LSTM', y_pred_lstm), ('CNN1D', y_pred_cnn), ('Transformer', y_pred_tr)]:
    colname = f'OT_pred_{name}'
    # init with NaNs
    test_rows[colname] = np.nan
    # Fill in overlapping indices
    # For each seq_pred index i -> global df index idx -> if idx falls in the final test_size window, place it
    for offset, pred in enumerate(arr):
        global_idx = seq_pred_indices[offset]
        # if this global_idx is within the last test_size indices:
        if global_idx >= len(df) - test_size:
            local_idx = global_idx - (len(df) - test_size)  # index into test_rows
            if 0 <= local_idx < len(test_rows):
                test_rows.at[local_idx, colname] = float(pred)

# For any remaining NaNs (if sequence model produced fewer preds than test rows), we can leave NaN or fill with classical RF
# Fill NaNs with RF predictions as fallback for continuity
for name in ['LSTM', 'CNN1D', 'Transformer']:
    col = f'OT_pred_{name}'
    test_rows[col] = test_rows[col].fillna(test_rows['OT_pred_RandomForest'])

# Save predictions into main output CSV
out_csv = OUT_DIR / "wind_model_output_with_OT_predictions.csv"
test_rows.to_csv(out_csv, index=False)

# Save metrics table
metrics_df = pd.DataFrame(results).T
metrics_df.to_csv(OUT_DIR / "model_metrics.csv")

# ----------------- PLOTTING -----------------
sns.set_style("whitegrid")

# 1) Plot true vs predicted for each model (OT) over test period and save
time_vals = test_rows['time']
for model_name in results.keys():
    pred_col = f'OT_pred_{model_name}' if model_name not in ['RandomForest','SVR'] else f'OT_pred_{model_name}'
    if pred_col not in test_rows.columns:
        continue
    plt.figure(figsize=(12,4))
    plt.plot(time_vals, test_rows['OT'].values, label='OT_true')
    plt.plot(time_vals, test_rows[pred_col].values, label=f'OT_pred_{model_name}', alpha=0.8)
    plt.xlabel('time')
    plt.ylabel('OT')
    plt.title(f'OT: true vs predicted ({model_name})')
    plt.legend()
    plt.tight_layout()
    fname = OUT_DIR / f"ot_true_vs_pred_{model_name}.png"
    plt.savefig(fname)
    plt.close()

# 2) Residual histograms
for model_name in results.keys():
    pred_col = f'OT_pred_{model_name}'
    res = test_rows['OT'].values - test_rows[pred_col].values
    plt.figure(figsize=(6,4))
    plt.hist(res, bins=50)
    plt.title(f'Residuals Histogram ({model_name})')
    plt.xlabel('Residual (OT_true - OT_pred)')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"residuals_{model_name}.png")
    plt.close()

# 3) Model comparison bar chart (MAE)
plt.figure(figsize=(8,5))
mae_vals = [results[k]['MAE'] for k in results.keys()]
plt.bar(list(results.keys()), mae_vals)
plt.ylabel('MAE')
plt.title('Model comparison (MAE)')
plt.tight_layout()
plt.savefig(OUT_DIR / "model_comparison_mae.png")
plt.close()

# 4) Save clustering scatter PCA plot
pca = PCA(n_components=2, random_state=SEED)
pc = pca.fit_transform(Xc_scaled)
plt.figure(figsize=(7,5))
plt.scatter(pc[:,0], pc[:,1], c=(df['cluster']==cluster_low).astype(int), s=8, cmap='coolwarm')
plt.title('Clustering PCA projection')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.tight_layout()
plt.savefig(OUT_DIR / "cluster_pca.png")
plt.close()

# ----------------- PRINT & EXPLAIN OUTPUT -----------------
print("\n--- Models and OT metrics ---")
for k, v in results.items():
    # {"MAE": mae, "MSE": mse, "RMSE": rmse, "NRMSE": nrmse, "R2": r2, "MAPE": mape, "SMAPE": smape, "MASE": mase}
    print(f"{k}: MAE={v['MAE']:.2f}, MSE={v['MSE']:.2f}, RMSE={v['RMSE']:.2f}, NRMSE={v['NRMSE']:.2f}, R2={v['R2']:.2f}, MAPE={v['MAPE']:.2f}, SMAPE={v['SMAPE']:.2f}, MASE={v['MASE']:.2f}")

print("\n--- Model artifacts & plots ---")

print(f"\nSaved test predictions and other fields to: {out_csv}")
print(f"Saved model metrics to: {OUT_DIR/'model_metrics.csv'}")
print(f"Saved model artifacts & plots to: {OUT_DIR}")

# Explain columns in output CSV
explanation = f"""
Output CSV columns (file: {out_csv.name}):

- time: timestamp for the test row.
- OT: true target value (ground truth) for OT.
- OT_pred_RandomForest: predicted OT from RandomForest baseline.
- OT_pred_SVR: predicted OT from SVR model.
- OT_pred_LSTM: predicted OT from LSTM model (sequence model).
- OT_pred_CNN1D: predicted OT from 1D-CNN model (sequence model).
- OT_pred_Transformer: predicted OT from Transformer-based model (sequence model).
- (Other original columns): exog_temp, exog_wind, gen_speed, P_est_kW, cluster, cluster_label, etc.

Notes:
- Sequence model predictions (LSTM/CNN/Transformer) were aligned to the test timestamps as best as possible; if any sequence-model predictions did not map exactly to a test row, they were filled by RandomForest predictions as fallback.
- All plots are saved as PNG files in the output directory.
- Model metrics (MAE, MSE, RMSE, MAPE) for OT are in model_metrics.csv.
"""
print(explanation)

# Print OT metrics summary (a concise table)
print("Concise OT metrics table:")
print(metrics_df)

# End
