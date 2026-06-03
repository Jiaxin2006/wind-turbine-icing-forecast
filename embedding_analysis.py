"""
embedding_analysis.py — CNN-LSTM-Attention embedding 聚类可视化与一致性分析。

从已训练模型中抽取 test 集 embedding，对 真实标签 / embedding / 预测值 分别做 KMeans，
计算三者之间的 ARI/NMI/F1 一致性，输出 clustering_classification_metrics.csv
（被 final_evaluation.py 读取）和若干对比散点图。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core import load_dataframe, SeqDatasetWithIdx

sns.set(style="whitegrid")

# ---------- CONFIG ----------
OUT_DIR = Path("out_cnn_lstm_cluster_1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = "标注的数据-#67_1.xlsx"
SEQ_LEN    = 4       # must match training
BATCH_SIZE = 16
VAL_RATIO  = 0.10
TEST_RATIO = 0.20
N_CLUSTERS = 2

# 模型结构超参（必须与 train_cnn_lstm.py / cnn_lstm_grid_search.py 训练时一致）
CNN_CHANNELS         = 16
CNN_KERNEL           = 3
LSTM_HID             = 128
TRANS_DMODEL         = 128
NUM_HEADS            = 4
NUM_TRANSFORMER_LAYERS = 0
DROPOUT_RATE         = 0.0
USE_HETEROSCEDASTIC  = True   # True: model outputs (mu, logvar, emb)

GLOBAL_MODEL_PATH = OUT_DIR / "model_run0_cluster0.pt"
FALLBACK_MODEL    = OUT_DIR / "model_cnn_lstm_att_final_0.pt"
SCALER_PATH       = OUT_DIR / "scaler_inputs.joblib"

EMB_NPY          = OUT_DIR / "test_embeddings.npy"
META_CSV         = OUT_DIR / "test_embeddings_meta.csv"
SCATTER_PNG      = OUT_DIR / "scatter_ot_temp_by_emb_cluster.png"
PRED_SCATTER_PNG = OUT_DIR / "scatter_ot_temp_by_pred_cluster.png"
COMPARISON_PNG   = OUT_DIR / "clustering_comparison.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load & preprocess ----------
print("Reading data...")
df = load_dataframe(DATA_PATH)
df['OT_prev'] = df['OT'].shift(1)
df = df.dropna().reset_index(drop=True)

feat_cols = [c for c in ['exog_temp', 'exog_wind', 'OT_prev'] if c in df.columns]

# ---------- Scaler (must use same scaler as training) ----------
if SCALER_PATH.exists():
    scaler_inputs = joblib.load(SCALER_PATH)
    df_scaled = df.copy()
    df_scaled[feat_cols] = scaler_inputs.transform(df[feat_cols].values)
else:
    scaler_inputs = StandardScaler().fit(df[feat_cols].values)
    df_scaled = df.copy()
    df_scaled[feat_cols] = scaler_inputs.transform(df[feat_cols].values)
    joblib.dump(scaler_inputs, SCALER_PATH)
    print(f"[WARN] scaler not found, fit on full df and saved to {SCALER_PATH}")

# ---------- Test split (must match training logic) ----------
n = len(df_scaled)
test_size  = int(n * TEST_RATIO)
val_size   = int(n * VAL_RATIO)
train_size = n - test_size - val_size
test_start = train_size + val_size
test_end   = n - 1

test_ds     = SeqDatasetWithIdx(df_scaled, test_start, test_end, SEQ_LEN, feat_cols)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
print("Test samples:", len(test_ds))

# ---------- Model (CNN-LSTM-Attention, must match train_cnn_lstm.py / cnn_lstm_grid_search.py) ----------
class CNN_LSTM_Attention(nn.Module):
    def __init__(self, feat_dim=3, cnn_channels=64, cnn_kernel=3, lstm_hid=128,
                 d_model=128, nhead=4, num_transformer_layers=0, dropout_rate=0.0,
                 heteroscedastic=False):
        super().__init__()
        self.hetero = heteroscedastic
        self.conv1 = nn.Conv1d(in_channels=feat_dim, out_channels=cnn_channels,
                               kernel_size=cnn_kernel, padding=cnn_kernel//2)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels,
                               kernel_size=cnn_kernel, padding=cnn_kernel//2)
        self.act = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels,
                               kernel_size=cnn_kernel, padding=cnn_kernel//2)
        self.dropout_cnn = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hid,
                            num_layers=1, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout_rate)
        if lstm_hid != d_model:
            self.proj_to_d = nn.Linear(lstm_hid, d_model)
        else:
            self.proj_to_d = None
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout_rate)
        if num_transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                                       dim_feedforward=d_model*2,
                                                       batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                             num_layers=num_transformer_layers)
        else:
            self.transformer_encoder = None
        final_out_dim = 2 if USE_HETEROSCEDASTIC else 1
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, final_out_dim)
        )
    def forward(self, x):
        c = x.permute(0,2,1)
        c = self.act(self.conv1(c))
        c = self.act(self.conv2(c))
        c = self.act(self.conv3(c))
        c = self.dropout_cnn(c)
        c = c.permute(0,2,1)
        lstm_out, _ = self.lstm(c)
        lstm_out = self.dropout_lstm(lstm_out)
        if self.proj_to_d is not None:
            tr_in = self.proj_to_d(lstm_out)
        else:
            tr_in = lstm_out
        attn_out, _ = self.mha(tr_in, tr_in, tr_in, need_weights=False)
        attn_out = self.dropout_attn(attn_out)
        tr_out = self.transformer_encoder(attn_out) if self.transformer_encoder else attn_out
        last = tr_out[:, -1, :]         # embedding we want
        out = self.fc(last)
        if self.hetero:
            mu = out[:,0]; logvar = out[:,1].clamp(-10,10)
            return mu, logvar, last  # 返回embedding用于聚类
        else:
            return out.squeeze(1), last  # 返回embedding用于聚类

# ---------- Load model weights ----------
model_path = GLOBAL_MODEL_PATH if GLOBAL_MODEL_PATH.exists() else (FALLBACK_MODEL if FALLBACK_MODEL.exists() else None)
if model_path is None:
    raise FileNotFoundError(f"No global model found at {GLOBAL_MODEL_PATH} or fallback {FALLBACK_MODEL}. Please provide a trained global model.")

print("Loading model from:", model_path)
feat_dim = len(feat_cols)
model = CNN_LSTM_Attention(feat_dim=feat_dim, cnn_channels=CNN_CHANNELS, cnn_kernel=CNN_KERNEL,
                           lstm_hid=LSTM_HID, d_model=TRANS_DMODEL, nhead=NUM_HEADS,
                           num_transformer_layers=NUM_TRANSFORMER_LAYERS, dropout_rate=DROPOUT_RATE,
                           heteroscedastic=USE_HETEROSCEDASTIC)
state = torch.load(model_path, map_location=DEVICE)
# if state is a dict of tensors (state_dict saved), load it; else if saved as raw state, try both
if isinstance(state, dict) and any(k.startswith('conv1') or k.startswith('fc') for k in state.keys()):
    model.load_state_dict(state)
else:
    # maybe the saved file is a raw dict with 'model' key or similar
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    elif 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        try:
            model.load_state_dict(state)
        except Exception as e:
            print("Warning: could not directly load state_dict; attempting direct assignment may fail.")
            raise e

model.to(DEVICE).eval()

# ----------------- Collect embeddings and predictions -----------------
embeddings = []
predictions = []
true_labels = []
meta_label_idxs = []

with torch.no_grad():
    for xb, yb, idxs in test_loader:
        xb = xb.to(DEVICE)
        
        if USE_HETEROSCEDASTIC:
            mu, logvar, emb = model(xb)
            pred = mu.cpu().numpy()
        else:
            pred, emb = model(xb)
            pred = pred.cpu().numpy()
        
        embeddings.append(emb.cpu().numpy())
        predictions.extend(pred.tolist())
        true_labels.extend(yb.numpy().tolist())
        meta_label_idxs.extend([int(i) for i in idxs.numpy().tolist()])

if len(embeddings) == 0:
    raise RuntimeError("No embeddings were extracted. Check model architecture.")

embeddings = np.vstack(embeddings)  # shape (N_test, d)
predictions = np.array(predictions)
true_labels = np.array(true_labels)

print("Embeddings shape:", embeddings.shape)
print("Predictions shape:", predictions.shape)

# ----------------- 保存数据 -----------------
np.save(EMB_NPY, embeddings)
pd.DataFrame({
    'label_idx': meta_label_idxs,
    'true_label': true_labels,
    'prediction': predictions
}).to_csv(META_CSV, index=False)

# ----------------- KMeans聚类分析 (纯聚类方法，不使用threshold) -----------------
print("\n" + "="*60)
print("KMEANS CLUSTERING ANALYSIS")
print("="*60)

# 1. 对真实标签进行KMeans聚类
true_reshaped = true_labels.reshape(-1, 1)
kmeans_true = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20).fit(true_reshaped)
true_cluster_labels = kmeans_true.labels_

# 2. 基于embeddings的聚类
kmeans_emb = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20).fit(embeddings)
emb_cluster_labels = kmeans_emb.labels_

# 3. 基于predictions的聚类
pred_reshaped = predictions.reshape(-1, 1)  
kmeans_pred = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20).fit(pred_reshaped)
pred_cluster_labels = kmeans_pred.labels_

print(f"True labels clustering centers: {kmeans_true.cluster_centers_.flatten()}")
print(f"Embedding clustering centers shape: {kmeans_emb.cluster_centers_.shape}")
print(f"Prediction clustering centers: {kmeans_pred.cluster_centers_.flatten()}")

# 统计各聚类的样本分布
print(f"\nTrue labels cluster distribution: {np.bincount(true_cluster_labels)}")
print(f"Embedding cluster distribution: {np.bincount(emb_cluster_labels)}")  
print(f"Prediction cluster distribution: {np.bincount(pred_cluster_labels)}")

# ----------------- 聚类一致性分析 -----------------
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, classification_report

def cluster_consistency_metrics(labels1, labels2, name1, name2):
    """计算两个聚类结果的一致性指标"""
    ari = adjusted_rand_score(labels1, labels2)
    nmi = normalized_mutual_info_score(labels1, labels2)
    
    # 计算聚类准确率 (通过最佳匹配)
    from scipy.optimize import linear_sum_assignment
    
    cm = confusion_matrix(labels1, labels2)
    row_ind, col_ind = linear_sum_assignment(-cm)  # 匈牙利算法找最佳匹配
    accuracy = cm[row_ind, col_ind].sum() / len(labels1)
    
    return {
        'comparison': f"{name1} vs {name2}",
        'ARI': ari,
        'NMI': nmi, 
        'Accuracy': accuracy,
        'n_samples': len(labels1)
    }

def compute_clustering_classification_metrics(true_labels, pred_labels, method_name):
    """
    计算聚类结果的分类指标 (类似二分类指标，但用于聚类)
    使用匈牙利算法找到最佳标签匹配，然后计算分类指标
    """
    from scipy.optimize import linear_sum_assignment
    
    # 构建混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 使用匈牙利算法找最佳匹配
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # 重新映射pred_labels以匹配true_labels
    label_mapping = dict(zip(col_ind, row_ind))
    mapped_pred_labels = np.array([label_mapping[label] for label in pred_labels])
    
    # 重新计算混淆矩阵
    final_cm = confusion_matrix(true_labels, mapped_pred_labels)
    
    # 对于K=2的情况，提取TP, FP, TN, FN
    if final_cm.shape == (2, 2):
        tn, fp, fn, tp = final_cm.ravel()
    else:
        # 对于K>2的情况，计算总体指标
        tp = np.diag(final_cm).sum()
        fp = final_cm.sum(axis=0) - np.diag(final_cm)
        fn = final_cm.sum(axis=1) - np.diag(final_cm)
        tn = final_cm.sum() - (tp + fp.sum() + fn.sum())
        
        # 转换为标量
        fp = fp.sum()
        fn = fn.sum()
    
    # 计算指标
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # ARI和NMI作为额外指标
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    metrics = {
        'Method': method_name,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (TPR)': recall,
        'F1 score': f1,
        'False Positive Rate (FPR)': fpr,
        'False Negative Rate (FNR)': fnr,
        'ARI': ari,
        'NMI': nmi,
        'Number of test samples': len(true_labels),
        'Label_mapping': label_mapping
    }
    
    return metrics

# 计算聚类一致性
consistency_results = []
consistency_results.append(cluster_consistency_metrics(true_cluster_labels, emb_cluster_labels, 'True', 'Embedding'))
consistency_results.append(cluster_consistency_metrics(true_cluster_labels, pred_cluster_labels, 'True', 'Prediction'))  
consistency_results.append(cluster_consistency_metrics(emb_cluster_labels, pred_cluster_labels, 'Embedding', 'Prediction'))

# 计算聚类分类指标 (以真实聚类为ground truth)
emb_cluster_metrics = compute_clustering_classification_metrics(true_cluster_labels, emb_cluster_labels, 'Embedding Clustering')
pred_cluster_metrics = compute_clustering_classification_metrics(true_cluster_labels, pred_cluster_labels, 'Prediction Clustering')

# 输出聚类一致性结果
print("\n" + "="*60)
print("CLUSTERING CONSISTENCY RESULTS")
print("="*60)

for result in consistency_results:
    print(f"\n{result['comparison']}:")
    print(f"  Adjusted Rand Index (ARI): {result['ARI']:.4f}")
    print(f"  Normalized Mutual Info (NMI): {result['NMI']:.4f}")  
    print(f"  Clustering Accuracy: {result['Accuracy']:.4f}")
    print(f"  Samples: {result['n_samples']}")

# 输出聚类分类指标
print("\n" + "="*60)
print("CLUSTERING CLASSIFICATION METRICS")
print("="*60)
print("(Using True Clustering as Ground Truth)")

for metrics in [emb_cluster_metrics, pred_cluster_metrics]:
    print(f"\n{metrics['Method']}:")
    print(f"  TP: {metrics['TP']}")
    print(f"  FP: {metrics['FP']}")  
    print(f"  TN: {metrics['TN']}")
    print(f"  FN: {metrics['FN']}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall (TPR): {metrics['Recall (TPR)']:.4f}")
    print(f"  F1 score: {metrics['F1 score']:.4f}")
    print(f"  False Positive Rate (FPR): {metrics['False Positive Rate (FPR)']:.4f}")
    print(f"  False Negative Rate (FNR): {metrics['False Negative Rate (FNR)']:.4f}")
    print(f"  ARI: {metrics['ARI']:.4f}")
    print(f"  NMI: {metrics['NMI']:.4f}")
    print(f"  Number of test samples: {metrics['Number of test samples']}")
    print(f"  Label mapping: {metrics['Label_mapping']}")

# 保存聚类分类指标结果
classification_df = pd.DataFrame([emb_cluster_metrics, pred_cluster_metrics])
classification_df.to_csv(OUT_DIR / "clustering_classification_metrics.csv", index=False)

# 保存聚类一致性结果
consistency_df = pd.DataFrame(consistency_results)
consistency_df.to_csv(OUT_DIR / "clustering_consistency_metrics.csv", index=False)

# ----------------- 准备可视化数据 -----------------
label_idxs = np.array(meta_label_idxs, dtype=int)
ot_true_vals = df.loc[label_idxs, 'OT'].values
temp_vals = df.loc[label_idxs, 'exog_temp'].values

# 构建完整的可视化DataFrame
plot_df = pd.DataFrame({
    'label_idx': label_idxs,
    'OT_true': ot_true_vals,
    'OT_pred': predictions,
    'exog_temp': temp_vals,
    'true_cluster': true_cluster_labels,
    'emb_cluster': emb_cluster_labels,
    'pred_cluster': pred_cluster_labels
})

# ----------------- 可视化对比 (纯聚类视角) -----------------
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
palette = sns.color_palette("tab10", n_colors=N_CLUSTERS)

# 1. 真实标签的聚类结果
ax1 = axes[0, 0]
for c in range(N_CLUSTERS):
    sub = plot_df[plot_df['true_cluster']==c]
    ax1.scatter(sub['OT_true'], sub['exog_temp'], s=20, alpha=0.7, 
               label=f'True Cluster {c} (μ={kmeans_true.cluster_centers_[c,0]:.0f}W)', color=palette[c])
ax1.set_xlabel("OT True (W)")
ax1.set_ylabel("Temperature")
ax1.set_title("True Labels KMeans Clustering")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Embedding聚类结果
ax2 = axes[0, 1]
for c in range(N_CLUSTERS):
    sub = plot_df[plot_df['emb_cluster']==c]
    ax2.scatter(sub['OT_true'], sub['exog_temp'], s=20, alpha=0.7, 
               label=f'Emb Cluster {c}', color=palette[c])
ax2.set_xlabel("OT True (W)")
ax2.set_ylabel("Temperature")
ax2.set_title(f"Embedding Clustering (ARI vs True: {consistency_results[0]['ARI']:.3f})")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 预测值聚类结果
ax3 = axes[1, 0]
for c in range(N_CLUSTERS):
    sub = plot_df[plot_df['pred_cluster']==c]
    ax3.scatter(sub['OT_true'], sub['exog_temp'], s=20, alpha=0.7, 
               label=f'Pred Cluster {c} (μ={kmeans_pred.cluster_centers_[c,0]:.0f}W)', color=palette[c])
ax3.set_xlabel("OT True (W)")
ax3.set_ylabel("Temperature")
ax3.set_title(f"Prediction Clustering (ARI vs True: {consistency_results[1]['ARI']:.3f})")
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 真实值 vs 预测值散点图
ax4 = axes[1, 1]
# 按真实聚类着色
for c in range(N_CLUSTERS):
    sub = plot_df[plot_df['true_cluster']==c]
    ax4.scatter(sub['OT_true'], sub['OT_pred'], s=20, alpha=0.7, 
               label=f'True Cluster {c}', color=palette[c])
# 添加y=x参考线
min_val = min(plot_df['OT_true'].min(), plot_df['OT_pred'].min())
max_val = max(plot_df['OT_true'].max(), plot_df['OT_pred'].max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
ax4.set_xlabel("OT True (W)")
ax4.set_ylabel("OT Predicted (W)")
ax4.set_title("True vs Predicted Values")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(COMPARISON_PNG, dpi=300, bbox_inches='tight')
plt.close()

# 单独保存预测聚类散点图
plt.figure(figsize=(8,6))
for c in range(N_CLUSTERS):
    sub = plot_df[plot_df['pred_cluster']==c]
    center_val = kmeans_pred.cluster_centers_[c,0]
    plt.scatter(sub['OT_true'], sub['exog_temp'], s=20, alpha=0.8, 
               label=f'Pred_cluster_{c} (μ={center_val:.0f}W)', color=palette[c])
plt.xlabel("OT (true power)")
plt.ylabel("Temperature (exog_temp)")
plt.title(f"Prediction-based clustering (KMeans={N_CLUSTERS}) — points colored by cluster")
plt.legend()
plt.tight_layout()
plt.savefig(PRED_SCATTER_PNG, dpi=300, bbox_inches='tight')
plt.close()

# 保存完整数据（cluster_eval.py 也从这里读取）
plot_df.to_csv(OUT_DIR / "full_clustering_analysis.csv", index=False)

print(f"\n保存的文件:")
print(f"- 嵌入向量: {EMB_NPY}")
print(f"- 元数据: {META_CSV}")
print(f"- 对比可视化: {COMPARISON_PNG}")
print(f"- 预测聚类图: {PRED_SCATTER_PNG}")
print(f"- 聚类分类指标: {OUT_DIR}/clustering_classification_metrics.csv")
print(f"- 聚类一致性指标: {OUT_DIR}/clustering_consistency_metrics.csv")
print(f"- 完整分析数据: {OUT_DIR}/full_clustering_analysis.csv")

print(f"\n使用KMeans(k={N_CLUSTERS})对真实标签、预测值、embedding分别聚类")
print("完成聚类分类指标分析!")

# 打印聚类中心对比
print(f"\n聚类中心对比:")
print(f"真实标签聚类中心: {kmeans_true.cluster_centers_.flatten()}")
print(f"预测值聚类中心: {kmeans_pred.cluster_centers_.flatten()}")
print(f"聚类中心差异: {np.abs(kmeans_true.cluster_centers_.flatten() - kmeans_pred.cluster_centers_.flatten())}")

# 创建类似论文表格的输出
print(f"\n" + "="*60)
print("TABLE: CLUSTERING CLASSIFICATION RESULTS")
print("="*60)

print("\nEmbedding Clustering Results:")
emb_m = emb_cluster_metrics
print(f"TP: {emb_m['TP']}")
print(f"FP: {emb_m['FP']}")
print(f"TN: {emb_m['TN']}")
print(f"FN: {emb_m['FN']}")
print(f"Accuracy: {emb_m['Accuracy']:.4f}")
print(f"Precision: {emb_m['Precision']:.4f}")
print(f"Recall (TPR): {emb_m['Recall (TPR)']:.4f}")
print(f"F1 score: {emb_m['F1 score']:.4f}")
print(f"False Positive Rate (FPR): {emb_m['False Positive Rate (FPR)']:.4f}")
print(f"False Negative Rate (FNR): {emb_m['False Negative Rate (FNR)']:.4f}")
print(f"Number of test samples: {emb_m['Number of test samples']}")
print(f"Clustering method: KMeans(K={N_CLUSTERS})")

print("\nPrediction Clustering Results:")
pred_m = pred_cluster_metrics
print(f"TP: {pred_m['TP']}")
print(f"FP: {pred_m['FP']}")
print(f"TN: {pred_m['TN']}")
print(f"FN: {pred_m['FN']}")
print(f"Accuracy: {pred_m['Accuracy']:.4f}")
print(f"Precision: {pred_m['Precision']:.4f}")
print(f"Recall (TPR): {pred_m['Recall (TPR)']:.4f}")
print(f"F1 score: {pred_m['F1 score']:.4f}")
print(f"False Positive Rate (FPR): {pred_m['False Positive Rate (FPR)']:.4f}")
print(f"False Negative Rate (FNR): {pred_m['False Negative Rate (FNR)']:.4f}")
print(f"Number of test samples: {pred_m['Number of test samples']}")
print(f"Clustering method: KMeans(K={N_CLUSTERS})")
