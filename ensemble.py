#!/usr/bin/env python3
"""
主实验：仅用温度/风速特征预测 OT。
基模型 RF + SVR(网格搜索) + CNN/LSTM/Transformer，再做 RidgeCV stacking 和 NNLS 加权集成，
最后抽取序列模型 embedding 做 PCA+KMeans 聚类。结果保存到 OUT_DIR。
"""

import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from scipy.optimize import nnls

import torch
from torch.utils.data import DataLoader

from core import (
    load_dataframe,
    metrics_dict,
    SeqDataset,
    LSTMReg,
    CNN1DReg,
    TransformerReg,
    train_torch_model,
    predict_torch,
)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
OUT_DIR = Path("output_ot_full_temp_wind")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEEP_EXOG_LAGS = True        # 是否加入温度/风速的滞后特征
LAGS = [1,2,3,6,12]          # 滞后步数（采样间隔 1 分钟，即对应分钟）
ROLL_WINDOW = 3
VAL_RATIO = 0.10
TEST_RATIO = 0.20
SEQ_LEN = 24
BATCH_SIZE = 16
EPOCHS_NN = 40
EPOCHS_CNN_TUNE = 24
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({"axes.grid": True})

# 读取数据（列名映射、排序、数值化、插值见 core.load_dataframe）
print("Reading data...")
data_path = "标注的数据-#67_1.xlsx"
df = load_dataframe(data_path, required=["OT", "exog_temp", "exog_wind"])

# 特征工程：仅温度/风速的滚动均值与滞后
df['temp_roll_3'] = df['exog_temp'].rolling(ROLL_WINDOW, min_periods=1).mean()
df['wind_roll_3'] = df['exog_wind'].rolling(ROLL_WINDOW, min_periods=1).mean()

for lag in LAGS:
    df[f'OT_lag_{lag}'] = df['OT'].shift(lag)
    if KEEP_EXOG_LAGS:
        df[f'temp_lag_{lag}'] = df['exog_temp'].shift(lag)
        df[f'wind_lag_{lag}'] = df['exog_wind'].shift(lag)

df = df.dropna().reset_index(drop=True)
print(f"Rows after feature engineering: {len(df)}")

# 按时间顺序切分 train/val/test，避免未来信息泄露
n = len(df)
test_size = int(n * TEST_RATIO)
val_size = int(n * VAL_RATIO)
train_size = n - test_size - val_size
if train_size <= SEQ_LEN + 10:
    raise ValueError("Not enough data for specified SEQ_LEN and splits. Reduce SEQ_LEN or test/val ratios.")

train_df = df.iloc[:train_size].reset_index(drop=True)
val_df   = df.iloc[train_size: train_size + val_size].reset_index(drop=True)
test_df  = df.iloc[train_size + val_size : ].reset_index(drop=True)
print(f"Train / Val / Test sizes: {len(train_df)} / {len(val_df)} / {len(test_df)}")

# 在 train 内部再切 train_train + meta_holdout，用于生成 stacking 的 meta 数据
train_train_end = int(len(train_df) * 0.8)
train_train_df = train_df.iloc[:train_train_end].reset_index(drop=True)
meta_holdout_df = train_df.iloc[train_train_end:].reset_index(drop=True)
print(f"train_train / meta_holdout sizes: {len(train_train_df)} / {len(meta_holdout_df)}")

# 经典模型用滚动+滞后特征；序列模型只用温度、风速
feature_cols_classical = ['exog_temp','exog_wind','temp_roll_3','wind_roll_3'] + [f'OT_lag_{l}' for l in LAGS]
if KEEP_EXOG_LAGS:
    feature_cols_classical += [f'temp_lag_{l}' for l in LAGS] + [f'wind_lag_{l}' for l in LAGS]

feature_cols_seq = ['exog_temp','exog_wind']

# scaler 只在 train_train 上 fit，避免泄露
scaler_cl = StandardScaler().fit(train_train_df[feature_cols_classical].values)
X_train_cl = scaler_cl.transform(train_train_df[feature_cols_classical].values)
y_train_tt = train_train_df['OT'].values
X_meta_hold = scaler_cl.transform(meta_holdout_df[feature_cols_classical].values)
y_meta_hold = meta_holdout_df['OT'].values
X_val_cl = scaler_cl.transform(val_df[feature_cols_classical].values)
y_val = val_df['OT'].values
X_test_cl = scaler_cl.transform(test_df[feature_cols_classical].values)
y_test = test_df['OT'].values

# 序列特征同样只在 train_train 上 fit scaler
scaler_seq = StandardScaler().fit(train_train_df[feature_cols_seq].values)
df_seq_scaled = df.copy()
df_seq_scaled[feature_cols_seq] = scaler_seq.transform(df[feature_cols_seq].values)

# SeqDataset / 序列模型 / 训练循环均来自 core.py
# train_train 用于训练基模型，meta_holdout 用作早停/选择的验证集
train_train_seq_loader = DataLoader(SeqDataset(df_seq_scaled, 0, train_train_end-1, SEQ_LEN, feature_cols_seq), batch_size=BATCH_SIZE, shuffle=True)
meta_seq_loader = DataLoader(SeqDataset(df_seq_scaled, train_train_end-1, train_size-1, SEQ_LEN, feature_cols_seq), batch_size=BATCH_SIZE, shuffle=False)

# 经典模型：RF 直接训练，SVR 用 TimeSeriesSplit 网格搜索
print("\nTraining RandomForest on train_train...")
rf_tt = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1)
rf_tt.fit(X_train_cl, y_train_tt)

print("GridSearch SVR (TimeSeriesSplit) on train_train...")
tscv = TimeSeriesSplit(n_splits=4)
pipe = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf'))])
param_grid_pipe = {'svr__C':[0.1,1,10,50], 'svr__epsilon':[0.1,0.5,1.0], 'svr__gamma':['scale','auto']}
grid = GridSearchCV(pipe, param_grid_pipe, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1)
grid.fit(train_train_df[feature_cols_classical].values, train_train_df['OT'].values)
pd.DataFrame(grid.cv_results_).to_csv(OUT_DIR/'svr_grid_results.csv', index=False)
best_svr_pipe = grid.best_estimator_
print("Best SVR params:", grid.best_params_)

# 用 train_train+meta_holdout+val 合并重训最佳 SVR
X_combined = np.vstack([train_train_df[feature_cols_classical].values, meta_holdout_df[feature_cols_classical].values, val_df[feature_cols_classical].values])
y_combined = np.concatenate([train_train_df['OT'].values, meta_holdout_df['OT'].values, val_df['OT'].values])
best_svr_pipe.fit(X_combined, y_combined)

joblib.dump(rf_tt, OUT_DIR/'rf_traintrain.joblib')
joblib.dump(best_svr_pipe, OUT_DIR/'svr_best_traintrain.joblib')

# CNN 小网格：在 train_train 上训练、meta_holdout 上选择
print("\nTuning CNN on train_train (validate on meta_holdout)...")
cnn_grid = [dict(hid=32,kernel=3,lr=1e-3), dict(hid=64,kernel=3,lr=1e-3), dict(hid=64,kernel=5,lr=5e-4)]
best_cnn = None; best_mse = float('inf'); best_cfg = None
for cfg in cnn_grid:
    model = CNN1DReg(input_dim=len(feature_cols_seq), hid=cfg['hid'], kernel_size=cfg['kernel'])
    model = train_torch_model(model, train_train_seq_loader, val_loader=meta_seq_loader, epochs=EPOCHS_CNN_TUNE, lr=cfg['lr'], device=DEVICE, early_stopping=3)
    _, preds_meta = predict_torch(model, meta_seq_loader)
    true_meta = meta_holdout_df['OT'].values[:len(preds_meta)]
    if len(true_meta) == 0:
        continue
    mse = mean_squared_error(true_meta, preds_meta[:len(true_meta)])
    print("cfg", cfg, "meta_mse", mse)
    if mse < best_mse:
        best_mse = mse; best_cnn = model; best_cfg = cfg

print("Best CNN cfg:", best_cfg)
# retrain best CNN on combined (train_train + meta_holdout + val)
combined_seq_loader = DataLoader(SeqDataset(df_seq_scaled, 0, train_size + val_size -1, SEQ_LEN, feature_cols_seq), batch_size=BATCH_SIZE, shuffle=True)
final_cnn = CNN1DReg(input_dim=len(feature_cols_seq), hid=best_cfg['hid'], kernel_size=best_cfg['kernel'])
final_cnn = train_torch_model(final_cnn, combined_seq_loader, val_loader=None, epochs=EPOCHS_NN, lr=best_cfg['lr'], device=DEVICE)

# train LSTM and Transformer similarly (train on train_train, val on meta_holdout, then retrain on combined)
print("\nTraining LSTM (select on meta_holdout, then retrain combined)...")
lstm_model = LSTMReg(input_dim=len(feature_cols_seq), hid=64)
lstm_model = train_torch_model(lstm_model, train_train_seq_loader, val_loader=meta_seq_loader, epochs=EPOCHS_NN, device=DEVICE)
lstm_final = LSTMReg(input_dim=len(feature_cols_seq), hid=64)
lstm_final = train_torch_model(lstm_final, combined_seq_loader, val_loader=None, epochs=EPOCHS_NN, device=DEVICE)

print("\nTraining Transformer (select on meta_holdout, then retrain combined)...")
tr_model = TransformerReg(input_dim=len(feature_cols_seq), d_model=64)
tr_model = train_torch_model(tr_model, train_train_seq_loader, val_loader=meta_seq_loader, epochs=EPOCHS_NN, device=DEVICE)
tr_final = TransformerReg(input_dim=len(feature_cols_seq), d_model=64)
tr_final = train_torch_model(tr_final, combined_seq_loader, val_loader=None, epochs=EPOCHS_NN, device=DEVICE)

# 用各基模型在 meta_holdout 上的预测作为 stacking 的输入特征
print("\nBuilding meta features on meta_holdout...")
meta_df = meta_holdout_df.copy()
# classical predictions using rf_tt & best_svr_pipe (trained on train_train earlier)
meta_df['pred_rf'] = rf_tt.predict(scaler_cl.transform(meta_df[feature_cols_classical].values))
meta_df['pred_svr'] = best_svr_pipe.predict(meta_df[feature_cols_classical].values)
# nn preds on meta_seq_loader (from selection-stage models)
_, preds_cnn_meta = predict_torch(best_cnn, meta_seq_loader)
_, preds_lstm_meta = predict_torch(lstm_model, meta_seq_loader)
_, preds_tr_meta = predict_torch(tr_model, meta_seq_loader)
Lm = len(meta_df)
meta_df['pred_cnn'] = preds_cnn_meta[:Lm] if len(preds_cnn_meta)>=Lm else np.pad(preds_cnn_meta, (0,Lm-len(preds_cnn_meta)),'edge')
meta_df['pred_lstm'] = preds_lstm_meta[:Lm] if len(preds_lstm_meta)>=Lm else np.pad(preds_lstm_meta, (0,Lm-len(preds_lstm_meta)),'edge')
meta_df['pred_tr'] = preds_tr_meta[:Lm] if len(preds_tr_meta)>=Lm else np.pad(preds_tr_meta, (0,Lm-len(preds_tr_meta)),'edge')

meta_features = ['pred_rf','pred_svr','pred_cnn','pred_lstm','pred_tr']
X_meta_feats = meta_df[meta_features].values
y_meta_vals = meta_df['OT'].values

# train RidgeCV meta learner
meta_learner = RidgeCV(alphas=[0.01,0.1,1.0,10.0]).fit(X_meta_feats, y_meta_vals)
joblib.dump(meta_learner, OUT_DIR/'meta_learner_ridgecv.joblib')

# 用全部训练数据重训基模型，得到最终预测
print("\nRefitting base models on combined training set for final predictions...")
X_all_train = np.vstack([train_train_df[feature_cols_classical].values, meta_holdout_df[feature_cols_classical].values, val_df[feature_cols_classical].values])
y_all_train = np.concatenate([train_train_df['OT'].values, meta_holdout_df['OT'].values, val_df['OT'].values])

rf_final = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1)
rf_final.fit(scaler_cl.transform(X_all_train), y_all_train)
best_svr_pipe.fit(X_all_train, y_all_train)
# final NN models (final_cnn, lstm_final, tr_final) already retrained on combined

# 测试集预测
print("Predicting on test set...")
pred_rf_test = rf_final.predict(scaler_cl.transform(test_df[feature_cols_classical].values))
pred_svr_test = best_svr_pipe.predict(test_df[feature_cols_classical].values)

# For sequence models, build sliding windows for test portion and predict
# We'll build sequence windows ending at indices train_size+val_size ... n-1
seq_window_end_indices = list(range(train_size + val_size, n))  # these correspond to target indices
# Create arrays of sequences for these end indices
seqs = []
seq_times = []
for end_idx in seq_window_end_indices:
    start_idx = end_idx - SEQ_LEN
    if start_idx < 0:
        continue
    arr = df_seq_scaled.iloc[start_idx: end_idx][feature_cols_seq].values.astype(np.float32)
    seqs.append(arr)
    seq_times.append(df.iloc[end_idx]['time'])
if len(seqs) == 0:
    raise ValueError("No sequence windows for test; check SEQ_LEN and splits.")
X_seqs = np.stack(seqs, axis=0)  # shape (m, seq_len, feat)
# create DataLoader for predictions
pred_dataset = torch.utils.data.TensorDataset(torch.tensor(X_seqs))
pred_loader = DataLoader(pred_dataset, batch_size=BATCH_SIZE, shuffle=False)
def model_predict_array_torch(model, X_array):
    model = model.to(DEVICE)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X_array.shape[0], BATCH_SIZE):
            xb = torch.tensor(X_array[i:i+BATCH_SIZE]).to(DEVICE)
            out = model(xb)
            if isinstance(out, tuple): out = out[0]
            preds.append(out.cpu().numpy().ravel())
    if len(preds)==0:
        return np.array([])
    return np.concatenate(preds)

preds_cnn_test = model_predict_array_torch(final_cnn, X_seqs)
preds_lstm_test = model_predict_array_torch(lstm_final, X_seqs)
preds_tr_test = model_predict_array_torch(tr_final, X_seqs)

# Now align sequence preds to test_df by time using merge_asof
seq_pred_df = pd.DataFrame({'time': seq_times, 'pred_cnn': preds_cnn_test, 'pred_lstm': preds_lstm_test, 'pred_tr': preds_tr_test})
# Merge into test_df by nearest time backward
test_rows = test_df.copy().sort_values('time').reset_index(drop=True)
seq_pred_df = seq_pred_df.sort_values('time').reset_index(drop=True)
merged = pd.merge_asof(test_rows, seq_pred_df, on='time', direction='backward')
pred_rf_test = np.asarray(pred_rf_test).ravel()
if len(pred_rf_test) != len(merged):
    # if lengths differ, try to broadcast or slice appropriately (here we raise to catch error)
    raise ValueError("pred_rf_test length does not match merged rows")

# if some NaNs remain, forward fill then fallback to RF
merged['pred_cnn'] = merged['pred_cnn'].ffill().fillna(pd.Series(pred_rf_test, index=merged.index))
merged['pred_lstm'] = merged['pred_lstm'].ffill().fillna(pd.Series(pred_rf_test, index=merged.index))
merged['pred_tr']   = merged['pred_tr'].ffill().fillna(pd.Series(pred_rf_test, index=merged.index))

merged['pred_rf'] = pred_rf_test
merged['pred_svr'] = pred_svr_test

# meta ensemble prediction
X_test_meta = merged[['pred_rf','pred_svr','pred_cnn','pred_lstm','pred_tr']].values
merged['OT_pred_Ensemble_meta'] = meta_learner.predict(X_test_meta)

# NNLS weighted ensemble with column scaling (prevent scale dominance)
col_std = X_meta_feats.std(axis=0, keepdims=True) + 1e-12
X_meta_scaled = X_meta_feats / col_std
w, _ = nnls(X_meta_scaled, y_meta_vals)
if w.sum() > 0:
    w = w / (w.sum() + 1e-12)
X_test_scaled = X_test_meta / col_std
merged['OT_pred_Ensemble_nnls'] = X_test_scaled.dot(w)
np.savetxt(OUT_DIR/'ensemble_weights_nnls.txt', w)

# 汇总指标并保存预测、模型权重
models_metrics = {}
models_metrics['RandomForest'] = metrics_dict(merged['OT'].values, merged['pred_rf'].values)
models_metrics['SVR'] = metrics_dict(merged['OT'].values, merged['pred_svr'].values)
models_metrics['CNN'] = metrics_dict(merged['OT'].values, merged['pred_cnn'].values)
models_metrics['LSTM'] = metrics_dict(merged['OT'].values, merged['pred_lstm'].values)
models_metrics['Transformer'] = metrics_dict(merged['OT'].values, merged['pred_tr'].values)
models_metrics['Ensemble_meta'] = metrics_dict(merged['OT'].values, merged['OT_pred_Ensemble_meta'].values)
models_metrics['Ensemble_nnls'] = metrics_dict(merged['OT'].values, merged['OT_pred_Ensemble_nnls'].values)

pd.DataFrame(models_metrics).T.to_csv(OUT_DIR/'final_metrics.csv')
merged.to_csv(OUT_DIR/'wind_model_output_with_OT_predictions_and_ensembles.csv', index=False)
joblib.dump(rf_final, OUT_DIR/'rf_final.joblib')
joblib.dump(best_svr_pipe, OUT_DIR/'svr_final.joblib')
torch.save(final_cnn.state_dict(), OUT_DIR/'cnn_final.pt')
torch.save(lstm_final.state_dict(), OUT_DIR/'lstm_final.pt')
torch.save(tr_final.state_dict(), OUT_DIR/'transformer_final.pt')

# 对全量数据滑窗，抽取每个序列模型倒数第二层的 embedding
def extract_embeddings(model, df_scaled, feat_cols, seq_len, batch_size=BATCH_SIZE):
    model = model.to(DEVICE); model.eval()
    seqs = []
    times = []
    for end_idx in range(seq_len, len(df_scaled)):
        start_idx = end_idx - seq_len
        seqs.append(df_scaled.iloc[start_idx:end_idx][feat_cols].values.astype(np.float32))
        times.append(df_scaled.iloc[end_idx]['time'])
    if len(seqs) == 0:
        return None
    X = np.stack(seqs, axis=0)
    embs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.tensor(X[i:i+batch_size]).to(DEVICE)
            out = model(xb, return_embedding=True)
            emb = out[1].cpu().numpy()
            embs.append(emb)
    embs = np.vstack(embs)
    emb_df = pd.DataFrame(embs, columns=[f'emb_{i}' for i in range(embs.shape[1])])
    emb_df['time'] = times
    return emb_df

print("\nExtracting embeddings for CNN/LSTM/Transformer and saving...")
emb_cnn = extract_embeddings(final_cnn, df_seq_scaled, feature_cols_seq, SEQ_LEN)
emb_lstm = extract_embeddings(lstm_final, df_seq_scaled, feature_cols_seq, SEQ_LEN)
emb_tr = extract_embeddings(tr_final, df_seq_scaled, feature_cols_seq, SEQ_LEN)
if emb_cnn is not None:
    emb_cnn.to_csv(OUT_DIR/'embeddings_cnn.csv', index=False)
if emb_lstm is not None:
    emb_lstm.to_csv(OUT_DIR/'embeddings_lstm.csv', index=False)
if emb_tr is not None:
    emb_tr.to_csv(OUT_DIR/'embeddings_transformer.csv', index=False)

# 对原始输入和各 embedding 分别做 PCA+KMeans
print("PCA + KMeans on inputs and on embeddings (if available)...")
X_inputs = df[['exog_temp','exog_wind']].values
sc_inputs = StandardScaler().fit(X_inputs)
Xp = sc_inputs.transform(X_inputs)
pc_inputs = PCA(n_components=2, random_state=SEED).fit_transform(Xp)
kmeans_inputs = KMeans(n_clusters=2, random_state=SEED, n_init=20).fit(pc_inputs)
df['pc1_inputs'] = pc_inputs[:,0]; df['pc2_inputs'] = pc_inputs[:,1]; df['cluster_inputs'] = kmeans_inputs.labels_
pd.DataFrame({'time': df['time'].values, 'pc1': pc_inputs[:,0], 'pc2': pc_inputs[:,1], 'cluster': kmeans_inputs.labels_}).to_csv(OUT_DIR/'embedding_inputs_pca.csv', index=False)
plt.figure(figsize=(7,5)); plt.scatter(pc_inputs[:,0], pc_inputs[:,1], c=kmeans_inputs.labels_, s=8, cmap='tab10'); plt.title('Inputs PCA cluster (temp & wind)'); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.tight_layout(); plt.savefig(OUT_DIR/'pca_inputs_cluster.png'); plt.close()

# embeddings clustering (each if exists)
def pca_kmeans_and_plot(emb_df, name):
    if emb_df is None:
        return
    Xemb = emb_df[[c for c in emb_df.columns if c.startswith('emb_')]].values
    Xemb_s = StandardScaler().fit_transform(Xemb)
    pc = PCA(n_components=2, random_state=SEED).fit_transform(Xemb_s)
    k = KMeans(n_clusters=2, random_state=SEED, n_init=20).fit(pc)
    emb_df['pc1'] = pc[:,0]; emb_df['pc2'] = pc[:,1]; emb_df['cluster_kmeans'] = k.labels_
    emb_df.to_csv(OUT_DIR/f'embedding_{name}_pca_cluster.csv', index=False)
    plt.figure(figsize=(7,5)); plt.scatter(pc[:,0], pc[:,1], c=k.labels_, s=8, cmap='tab10'); plt.title(f'{name} embedding PCA cluster'); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.tight_layout(); plt.savefig(OUT_DIR/f'{name}_embed_pca_cluster.png'); plt.close()

pca_kmeans_and_plot(emb_cnn, 'cnn')
pca_kmeans_and_plot(emb_lstm, 'lstm')
pca_kmeans_and_plot(emb_tr, 'transformer')

# 真值 vs 预测、残差、各模型 MAE 对比图
print("Saving final plots...")
time_vals = merged['time']
plt.figure(figsize=(12,4))
plt.plot(time_vals, merged['OT'].values, label='OT_true')
plt.plot(time_vals, merged['pred_rf'].values, label='RF')
plt.plot(time_vals, merged['OT_pred_Ensemble_meta'].values, label='Ensemble_meta', alpha=0.9)
plt.xlabel('time'); plt.ylabel('OT'); plt.legend(); plt.title('OT: true vs RF vs Ensemble_meta')
plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(OUT_DIR/'ot_true_vs_rf_vs_ensemble.png'); plt.close()

# residual hist
res = merged['OT'].values - merged['OT_pred_Ensemble_meta'].values
plt.figure(figsize=(6,4)); plt.hist(res, bins=50); plt.title('Residuals Ensemble_meta'); plt.tight_layout(); plt.savefig(OUT_DIR/'residuals_ensemble_meta.png'); plt.close()

# model MAE comparison bar
plt.figure(figsize=(9,5))
names = list(models_metrics.keys())
maes = [models_metrics[n]['MAE'] for n in names]
plt.bar(names, maes)
plt.ylabel('MAE'); plt.title('Model comparison (MAE)'); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(OUT_DIR/'model_comparison_mae.png'); plt.close()

print("\nDone. Outputs saved to:", OUT_DIR)
print("Final metrics:")
print(pd.DataFrame(models_metrics).T)

# End of script
