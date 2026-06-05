#!/usr/bin/env python3
"""
cluster_agglomerative.py — COURSE_COVERAGE Step 2: A8 凝聚式层次聚类

在已保存的 CNN-LSTM-Attention embedding 上，用 AgglomerativeClustering（ward /
complete / average 三种连接策略）与 KMeans(k=2) 做横向对比。

目的：
  1. 验证工况分离结构是否对聚类算法选择鲁棒（若 ARI 一致 → 结论可信）
  2. 通过树状图（dendrogram）展示凝聚式聚类的层次过程
  3. 补充课程考点 A8（AgglomerativeClustering）

无 torch 依赖 —— 直接读取已保存的 .npy / .csv 文件。
运行：python3 clustering/cluster_agglomerative.py
输出目录：output_cluster_agglomerative/
"""


from pathlib import Path as _Path
import os as _os
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
_os.chdir(_PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment

# ─────────────────────────── 路径配置 ───────────────────────────────────────
SRC_DIR = Path("out_cnn_lstm_cluster_1")
OUT_DIR = Path("output_cluster_agglomerative")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_NPY  = SRC_DIR / "test_embeddings.npy"
META_CSV = SRC_DIR / "test_embeddings_meta.csv"

# ─────────────────────────── 加载数据 ───────────────────────────────────────
print("Loading pre-saved embeddings and metadata...")
embeddings  = np.load(EMB_NPY)           # (N, 128) — CNN-LSTM-Attention hidden states
meta        = pd.read_csv(META_CSV)      # columns: label_idx, true_label, prediction
true_labels = meta["true_label"].values  # 真实 OT 值（连续）
predictions = meta["prediction"].values  # 模型预测 OT 值（连续）
N = len(true_labels)
print(f"  N={N}, embedding dim={embeddings.shape[1]}")

# ─────────────────────────── 工具函数 ───────────────────────────────────────
def best_match_metrics(ref_labels, pred_labels):
    """
    用匈牙利算法找最优标签排列后计算准确率 / F1 / ARI / NMI。
    无监督聚类的标签是任意编号，必须先对齐再算分类指标。
    """
    cm = confusion_matrix(ref_labels, pred_labels)
    row_idx, col_idx = linear_sum_assignment(-cm)
    mapping = {c: r for c, r in zip(col_idx, row_idx)}
    mapped = np.array([mapping[l] for l in pred_labels])

    final_cm = confusion_matrix(ref_labels, mapped)
    tn, fp, fn, tp = final_cm.ravel()
    accuracy  = (tp + tn) / N
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    ari = adjusted_rand_score(ref_labels, pred_labels)
    nmi = normalized_mutual_info_score(ref_labels, pred_labels)
    return dict(TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
                Accuracy=accuracy, Precision=precision, Recall=recall,
                F1=f1, ARI=ari, NMI=nmi)

# ─────────────────── 参考基准：KMeans 对真实 OT 值聚类 ─────────────────────
# 与 embedding_analysis.py 完全一致的方式建立 ground truth
print("\nBuilding reference clustering (KMeans on true OT values)...")
true_reshaped   = true_labels.reshape(-1, 1)
kmeans_true     = KMeans(n_clusters=2, random_state=42, n_init=20).fit(true_reshaped)
true_cluster    = kmeans_true.labels_
centers_true    = kmeans_true.cluster_centers_.flatten()
print(f"  KMeans true OT centers: {centers_true}")
print(f"  Cluster counts: {np.bincount(true_cluster)}")

# ─────────────────── 四种聚类方法 × 两种特征 ──────────────────────────────
# 特征 A：128 维 embedding（标准化后）
# 特征 B：1 维预测值（与 embedding_analysis.py 一致）
scaler_emb  = StandardScaler().fit(embeddings)
emb_scaled  = scaler_emb.transform(embeddings)
pred_scaled = predictions.reshape(-1, 1)  # 预测值本身就是 1D，不再标准化（与原脚本一致）

CONFIGS = [
    # (name, ClusterClass, kwargs)
    ("KMeans",                 KMeans,               dict(n_clusters=2, random_state=42, n_init=20)),
    ("Agglomerative-ward",     AgglomerativeClustering, dict(n_clusters=2, linkage="ward")),
    ("Agglomerative-complete", AgglomerativeClustering, dict(n_clusters=2, linkage="complete")),
    ("Agglomerative-average",  AgglomerativeClustering, dict(n_clusters=2, linkage="average")),
]

print("\n" + "="*70)
print("CLUSTERING COMPARISON (feature = embedding, reference = true-OT cluster)")
print("="*70)

records_emb  = []  # embedding 特征的结果
records_pred = []  # 预测值特征的结果

for name, Cls, kwargs in CONFIGS:
    # ── embedding 特征 ──
    model = Cls(**kwargs)
    emb_cluster = model.fit_predict(emb_scaled)
    m = best_match_metrics(true_cluster, emb_cluster)
    m["Method"] = name
    m["Feature"] = "Embedding"
    records_emb.append(m)
    print(f"  {name:30s}  ARI={m['ARI']:.4f}  NMI={m['NMI']:.4f}  "
          f"Acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}")

print("\n" + "="*70)
print("CLUSTERING COMPARISON (feature = prediction, reference = true-OT cluster)")
print("="*70)

for name, Cls, kwargs in CONFIGS:
    # ── 预测值特征 ──
    model = Cls(**kwargs)
    pred_cluster = model.fit_predict(pred_scaled)
    m = best_match_metrics(true_cluster, pred_cluster)
    m["Method"] = name
    m["Feature"] = "Prediction"
    records_pred.append(m)
    print(f"  {name:30s}  ARI={m['ARI']:.4f}  NMI={m['NMI']:.4f}  "
          f"Acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}")

# ─────────────────────────── 保存结果表 ─────────────────────────────────────
df_results = pd.DataFrame(records_emb + records_pred)
df_results.to_csv(OUT_DIR / "cluster_agglomerative_metrics.csv", index=False)
print(f"\nSaved metrics to {OUT_DIR}/cluster_agglomerative_metrics.csv")

# ─────────────────── 图 1：ARI/NMI 对比柱状图 ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
method_labels = [c[0] for c in CONFIGS]
x = np.arange(len(method_labels))
width = 0.35

for ax, records, feat_name in zip(axes, [records_emb, records_pred], ["Embedding", "Prediction"]):
    aris = [r["ARI"] for r in records]
    nmis = [r["NMI"] for r in records]
    bars1 = ax.bar(x - width/2, aris, width, label="ARI",  color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, nmis, width, label="NMI",  color="#FF9800", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(method_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Clustering quality (feature: {feat_name})\nvs true-OT KMeans reference")
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=7)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=7)
    ax.axhline(0.8, ls="--", color="gray", lw=0.8, alpha=0.6, label="ARI=0.8 line")

plt.tight_layout()
plt.savefig(OUT_DIR / "ari_nmi_comparison.png", dpi=150)
plt.close()

# ─────────────────── 图 2：树状图（dendrogram）────────────────────────────
# 对 embedding 做层次聚类并画树状图（理解凝聚式聚类过程用）
# 数据量大时只采子集展示
print("\nBuilding dendrogram on embedding subsample...")
DENDRO_N = 500  # 子采样数量（过多会很慢）
rng = np.random.default_rng(42)
idx_sub = rng.choice(N, size=min(DENDRO_N, N), replace=False)
sub_emb = emb_scaled[idx_sub]

# ward linkage 的树状图
Z = linkage(sub_emb, method="ward")
fig, ax = plt.subplots(figsize=(14, 4))
dendrogram(
    Z, ax=ax,
    truncate_mode="lastp",   # 只显示最后 p 个合并节点
    p=30,
    leaf_rotation=90,
    leaf_font_size=8,
    color_threshold=Z[-2, 2],  # 在最后一次合并前切割 → 恰好 k=2
    show_contracted=True,
)
ax.set_title(f"Hierarchical Clustering Dendrogram (Ward linkage)\n"
             f"on CNN-LSTM-Attention embedding subsample (N={min(DENDRO_N, N)})\n"
             f"Dashed line shows cut point for k=2")
ax.set_xlabel("Sample index (truncated)")
ax.set_ylabel("Ward distance (merge cost)")
# 画切割线
cut_height = (Z[-1, 2] + Z[-2, 2]) / 2
ax.axhline(cut_height, ls="--", color="red", lw=1.5, label=f"k=2 cut (h≈{cut_height:.1f})")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "dendrogram_ward.png", dpi=150)
plt.close()

# ─────────────────── 图 3：PCA 2D 散点（KMeans vs Agglomerative-ward）───────
print("PCA projection for visualization...")
pca = PCA(n_components=2, random_state=42)
emb_2d = pca.fit_transform(emb_scaled)
var_exp = pca.explained_variance_ratio_

# 重新跑一次以拿到完整标签
km_labels   = KMeans(n_clusters=2, random_state=42, n_init=20).fit_predict(emb_scaled)
agg_labels  = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(emb_scaled)
true_c_labels = true_cluster   # reference

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
palette = ["#2196F3", "#E91E63"]
titles = ["True OT KMeans (reference)", "KMeans(k=2) on Embedding", "Agglomerative-ward(k=2)"]
label_sets = [true_c_labels, km_labels, agg_labels]

for ax, lbl, title in zip(axes, label_sets, titles):
    for c, color in enumerate(palette):
        mask = lbl == c
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   s=5, alpha=0.5, color=color, label=f"Cluster {c}")
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
    ax.set_title(title, fontsize=9)
    ax.legend(markerscale=3, fontsize=8)

plt.suptitle("PCA(2D) of CNN-LSTM-Attention embedding — clustering comparison", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────── 图 4：结果总结热力图 ───────────────────────────────────
pivot_ari = df_results.pivot(index="Method", columns="Feature", values="ARI")
pivot_f1  = df_results.pivot(index="Method", columns="Feature", values="F1")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, data, title, fmt in zip(
    axes,
    [pivot_ari, pivot_f1],
    ["Adjusted Rand Index (ARI)\nvs True-OT Reference", "F1 Score\nvs True-OT Reference"],
    [".3f", ".3f"]
):
    values = data.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(data.shape[1]), data.columns)
    ax.set_yticks(range(data.shape[0]), data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = values[i, j]
            label = "" if np.isnan(val) else format(val, fmt)
            ax.text(j, i, label, ha="center", va="center", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("")

plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_summary.png", dpi=150)
plt.close()

# ─────────────────── 终端摘要 ───────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
cols = ["Method", "Feature", "ARI", "NMI", "Accuracy", "F1"]
print(df_results[cols].to_string(index=False, float_format="%.4f"))

print(f"\nOutput files:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f}")
