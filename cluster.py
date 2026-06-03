#!/usr/bin/env python3
# eval_emb_cluster_shutdown.py
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
sns.set(style="whitegrid")

OUT_DIR = "out_cnn_lstm_cluster_1"   # 根据你的路径修改
CSV_PATH = os.path.join(OUT_DIR, "emb_cluster_vs_ot_temp.csv")
EMB_PATHS = [os.path.join(OUT_DIR, "test_embeddings.npy"),
             os.path.join(OUT_DIR, "embeddings.npy"),
             os.path.join(OUT_DIR, "embeddings_test.npy")]

THRESH = 1000.0  # OT threshold: <1000 -> should close (positive class)
K3_OUTPUT = os.path.join(OUT_DIR, "emb_cluster_k3_scatter.png")
BINARY_SCATTER = os.path.join(OUT_DIR, "emb_cluster_binary_confusion.png")
METRICS_JSON = os.path.join(OUT_DIR, "emb_cluster_binary_metrics.json")
CONF_CSV = os.path.join(OUT_DIR, "emb_cluster_confusion.csv")
K3_SUMMARY = os.path.join(OUT_DIR, "emb_cluster_k3_summary.csv")

# 1. load csv
df = pd.read_csv(CSV_PATH)
required_cols = ['OT_true','exog_temp','emb_cluster']
for c in required_cols:
    if c not in df.columns:
        raise RuntimeError(f"Required column {c} not found in {CSV_PATH}. Found columns: {df.columns.tolist()}")

# create true binary label: 1 = should_close (OT_true < THRESH)
df['should_close_true'] = (df['OT_true'] < THRESH).astype(int)

# 2. map existing emb_cluster (K=2) -> binary prediction by majority voting
mapping = {}
for c in sorted(df['emb_cluster'].unique()):
    sub = df[df['emb_cluster']==c]
    # majority of 'should_close_true'
    maj = int(sub['should_close_true'].mode().iat[0]) if len(sub)>0 else 0
    mapping[c] = maj
print("Cluster -> predicted label (majority):", mapping)
df['should_close_pred'] = df['emb_cluster'].map(mapping).astype(int)

# 3. compute confusion matrix and metrics
y_true = df['should_close_true'].values
y_pred = df['should_close_pred'].values
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn)>0 else None
precision = tp / (tp + fp) if (tp+fp)>0 else 0.0
recall = tp / (tp + fn) if (tp+fn)>0 else 0.0
f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
fpr = fp / (fp + tn) if (fp+tn)>0 else 0.0
fnr = fn / (fn + tp) if (fn+tp)>0 else 0.0

metrics = {
    "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    "Accuracy": float(accuracy), "Precision": float(precision),
    "Recall": float(recall), "F1": float(f1),
    "FPR": float(fpr), "FNR": float(fnr),
    "Cluster_to_label_mapping": mapping,
    "Threshold_W": float(THRESH),
    "N_samples": int(len(df))
}
# --- make JSON-safe: convert numpy types to native Python types ---
def to_jsonable(x):
    import numpy as _np
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, _np.integer):
        return int(x)
    if isinstance(x, _np.floating):
        return float(x)
    if isinstance(x, _np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        # convert keys to str (JSON requires string keys)
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    try:
        return str(x)
    except Exception:
        return None

metrics_jsonable = to_jsonable(metrics)
print("Binary metrics:", json.dumps(metrics_jsonable, indent=2))

with open(METRICS_JSON, "w") as f:
    json.dump(metrics_jsonable, f, indent=2)

df.to_csv(CONF_CSV, index=False)
print("Saved confusion CSV to:", CONF_CSV)
print("Saved metrics JSON to:", METRICS_JSON)

# 4. scatter plot: OT_true (x) vs exog_temp (y), colored by emb_cluster; mark predicted shutdown clusters
plt.figure(figsize=(8,6))
palette = sns.color_palette("tab10", n_colors=len(df['emb_cluster'].unique()))
for c in sorted(df['emb_cluster'].unique()):
    sub = df[df['emb_cluster']==c]
    label = f"cluster_{c} (pred={'CLOSE' if mapping[c]==1 else 'KEEP'})"
    plt.scatter(sub['OT_true'], sub['exog_temp'], s=20, alpha=0.8, label=label, color=palette[int(c%10)])
plt.axvline(THRESH, color='k', linestyle='--', label=f"threshold={THRESH}W")
plt.xlabel("OT_true (W)"); plt.ylabel("Temperature (°C)")
plt.legend(markerscale=2, fontsize='small')
plt.title("OT_true vs Temperature colored by embedding-cluster (K=2) — cluster majority predicts shutdown")
plt.tight_layout()
plt.savefig(BINARY_SCATTER, dpi=300, bbox_inches='tight')
plt.close()
print("Saved binary scatter to:", BINARY_SCATTER)

# 5. If K=2 poor, try K=3 on embeddings (preferred) or fallback to OT/temp 2D
emb = None
for p in EMB_PATHS:
    if os.path.exists(p):
        try:
            emb = np.load(p)
            print("Loaded embeddings from", p)
            break
        except Exception as e:
            print("Failed to load embeddings from", p, ":", e)
if emb is None:
    print("No embedding file found. Will perform KMeans(3) on (OT_true, exog_temp) instead.")
    X_k3 = df[['OT_true','exog_temp']].values
else:
    X_k3 = emb  # if embedding rows correspond to df rows order; otherwise ensure alignment

# run kmeans k=3
k3 = KMeans(n_clusters=3, random_state=42, n_init=20).fit(X_k3)
labels_k3 = k3.labels_
# if emb used, ensure labels_k3 aligns with df rows (embedding order must match)
if emb is not None and len(labels_k3) == len(df):
    df['k3_cluster'] = labels_k3
elif emb is None:
    df['k3_cluster'] = labels_k3  # using OT/temp -> aligned by df order
else:
    # mismatch length: fallback to subset or reindex
    print("Warning: k3 labels length mismatch; applying k3 by truncation/padding.")
    L = min(len(labels_k3), len(df))
    df['k3_cluster'] = np.concatenate([labels_k3[:L], np.full(len(df)-L, -1, dtype=int)])

# 6. summarize k3 clusters in OT/temp 2D and save
summary = []
for c in sorted(df['k3_cluster'].unique()):
    sub = df[df['k3_cluster']==c]
    summary.append({
        'k3_cluster': int(c),
        'count': int(len(sub)),
        'OT_mean': float(sub['OT_true'].mean()),
        'OT_std': float(sub['OT_true'].std()),
        'temp_mean': float(sub['exog_temp'].mean()),
        'temp_std': float(sub['exog_temp'].std()),
        'should_close_ratio': float(sub['should_close_true'].mean())
    })
pd.DataFrame(summary).to_csv(K3_SUMMARY, index=False)
print("Saved K=3 summary to:", K3_SUMMARY)

# 7. plot K=3 scatter (OT vs temp) colored by k3 cluster, annotate cluster centers
plt.figure(figsize=(8,6))
for c in sorted(df['k3_cluster'].unique()):
    sub = df[df['k3_cluster']==c]
    plt.scatter(sub['OT_true'], sub['exog_temp'], s=20, alpha=0.8, label=f"k3_{c}")
# plot centroids
centroids = KMeans(n_clusters=3, random_state=42, n_init=20).fit(X_k3).cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], c='k', s=120, marker='X', label='centroids')
plt.axvline(THRESH, color='k', linestyle='--')
plt.xlabel("OT_true (W)"); plt.ylabel("Temperature (°C)")
plt.title("K=3 clustering (on embeddings if available; else on OT/temp)")
plt.legend(markerscale=2, fontsize='small')
plt.tight_layout()
plt.savefig(K3_OUTPUT, dpi=300, bbox_inches='tight')
plt.close()
print("Saved K=3 scatter to:", K3_OUTPUT)

print("Done. Results and plots saved in:", OUT_DIR)
