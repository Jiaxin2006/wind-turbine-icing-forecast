#!/usr/bin/env python3
"""
cluster_kmedoids.py — COURSE_COVERAGE Step 3: A6 K-Medoids (PAM)

从零实现 PAM（Partitioning Around Medoids），并与 KMeans 在三种场景下对比：
  1. 正常 embedding 聚类（重现 ARI 基准）
  2. 加入合成离群点后，KMeans 中心漂移 vs K-Medoids 中心稳定（鲁棒性实验）
  3. 1D 预测值聚类（纯数值，直觉验证）

无 torch 依赖，直接读取预存的 .npy / .csv。
运行：python3 clustering/cluster_kmedoids.py
输出目录：output_cluster_kmedoids/
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

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# ─────────────────────────── 路径配置 ───────────────────────────────────────
SRC_DIR = Path("out_cnn_lstm_cluster_1")
OUT_DIR = Path("output_cluster_kmedoids")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── 加载数据 ───────────────────────────────────────
print("Loading pre-saved embeddings and metadata...")
embeddings  = np.load(SRC_DIR / "test_embeddings.npy")   # (N, 128)
meta        = pd.read_csv(SRC_DIR / "test_embeddings_meta.csv")
true_vals   = meta["true_label"].values
predictions = meta["prediction"].values
N = len(true_vals)
print(f"  N={N}, embedding dim={embeddings.shape[1]}")

# ─────────────────────────── PAM 实现 ───────────────────────────────────────
def kmedoids_pam(X: np.ndarray, k: int, max_iter: int = 50,
                 random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    K-Medoids PAM（Partitioning Around Medoids）

    与 KMeans 的核心区别：
    - KMeans 聚类中心 = 簇内所有点的均值（可能是空间中不存在的「虚」点）
    - K-Medoids 聚类中心 = 簇内使总距离最小的「真实」数据点（medoid）

    这使得 K-Medoids：
    ① 对离群点更鲁棒（medoid 只能从实际数据中选，无法被极端值远远拉走）
    ② 可以使用任意距离度量（不限欧氏）
    ③ 结果更可解释（中心本身就是真实样本）

    时间复杂度：O(k * n_c^2) 每轮，n_c = 最大簇的大小。
    大数据集建议先 PCA 降维。
    """
    rng = np.random.default_rng(random_state)
    n   = len(X)

    # 初始化：随机选 k 个点作为 medoid（索引）
    medoid_idx = rng.choice(n, size=k, replace=False).tolist()

    for _iter in range(max_iter):
        # ── Assignment ────────────────────────────────────────────────────
        # 每个点分配给距离最近的 medoid
        medoid_X = X[medoid_idx]                                  # (k, d)
        dists_to_medoids = cdist(X, medoid_X, metric="euclidean") # (N, k)
        labels = np.argmin(dists_to_medoids, axis=1)              # (N,)

        # ── Update ─────────────────────────────────────────────────────
        # 对每个簇，从成员中选出使「到簇内所有点距离之和」最小的点
        changed = False
        new_medoid_idx = list(medoid_idx)
        for c in range(k):
            cluster_idx = np.where(labels == c)[0]
            if len(cluster_idx) == 0:
                continue
            cluster_X = X[cluster_idx]                          # (n_c, d)
            D_c = cdist(cluster_X, cluster_X, metric="euclidean") # (n_c, n_c)
            total_dists = D_c.sum(axis=1)
            best_local_pos = np.argmin(total_dists)
            best_global_idx = cluster_idx[best_local_pos]
            if best_global_idx != medoid_idx[c]:
                changed = True
            new_medoid_idx[c] = best_global_idx
        medoid_idx = new_medoid_idx
        if not changed:
            print(f"    PAM converged at iteration {_iter + 1}")
            break

    return np.array(medoid_idx), labels


def best_match_metrics(ref: np.ndarray, pred: np.ndarray) -> dict:
    """匈牙利算法对齐后计算 ARI / NMI / Accuracy / F1。"""
    cm = confusion_matrix(ref, pred)
    r, c = linear_sum_assignment(-cm)
    mapping = {cv: rv for cv, rv in zip(c, r)}
    mapped = np.array([mapping[l] for l in pred])
    fcm = confusion_matrix(ref, mapped)
    tn, fp, fn, tp = fcm.ravel()
    acc = (tp + tn) / len(ref)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return dict(TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
                Accuracy=acc, Precision=prec, Recall=rec, F1=f1,
                ARI=adjusted_rand_score(ref, pred),
                NMI=normalized_mutual_info_score(ref, pred))


# ─────────────── 参考基准：KMeans on true OT（与之前一致）──────────────────
print("\nBuilding reference (KMeans on true OT)...")
km_true = KMeans(n_clusters=2, random_state=42, n_init=20).fit(true_vals.reshape(-1, 1))
true_cluster = km_true.labels_
print(f"  Centers: {km_true.cluster_centers_.flatten()}")
print(f"  Counts:  {np.bincount(true_cluster)}")

# ─────────────── 特征：PCA-10 压缩 embedding（加速 PAM 的标准做法）─────────
print("\nPCA(10) on embedding...")
scaler = StandardScaler()
emb_sc = scaler.fit_transform(embeddings)
pca    = PCA(n_components=10, random_state=42)
emb10  = pca.fit_transform(emb_sc)
print(f"  Explained variance (PC1-10): {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ─────────────── 实验 A：正常数据 KMeans vs K-Medoids ────────────────────
print("\n" + "="*60)
print("Experiment A: Normal data — KMeans vs K-Medoids (embedding PCA-10)")
print("="*60)

print("[KMeans] fitting...")
km_emb  = KMeans(n_clusters=2, random_state=42, n_init=20).fit(emb10)
km_pred = km_emb.labels_
m_km    = best_match_metrics(true_cluster, km_pred)
print(f"  KMeans     ARI={m_km['ARI']:.4f}  Acc={m_km['Accuracy']:.4f}  F1={m_km['F1']:.4f}")
km_centers = km_emb.cluster_centers_  # (2, 10) — 虚点

print("[K-Medoids PAM] fitting (may take ~10-30s on PCA-10)...")
medoid_idx, pam_pred = kmedoids_pam(emb10, k=2, random_state=42)
m_pam = best_match_metrics(true_cluster, pam_pred)
print(f"  K-Medoids  ARI={m_pam['ARI']:.4f}  Acc={m_pam['Accuracy']:.4f}  F1={m_pam['F1']:.4f}")
print(f"  Medoid global indices: {medoid_idx}")
pam_centers = emb10[medoid_idx]  # (2, 10) — 真实数据点

# 中心距原点的距离（验证 medoid 是真实点）
print(f"  ||KMeans center 0||={np.linalg.norm(km_centers[0]):.3f}  "
      f"||KMedoids center 0||={np.linalg.norm(pam_centers[0]):.3f}")

# ─────────────── 实验 B：离群点鲁棒性 ──────────────────────────────────────
# 注入 50 个极端离群点（OT 方向极远），对比 KMeans 和 K-Medoids 的中心漂移
print("\n" + "="*60)
print("Experiment B: Outlier robustness demo (embedding PCA-10 + 50 outliers)")
print("="*60)

N_OUT = 50
rng = np.random.default_rng(0)
# 离群点放在第一个 PC 的极远处（5 × 标准差之外）
outlier_offset = np.zeros((N_OUT, 10))
outlier_offset[:, 0] = 8.0   # 沿 PC1 方向推出 8σ
outliers = outlier_offset + rng.normal(0, 0.1, size=(N_OUT, 10))

emb10_noisy      = np.vstack([emb10, outliers])
true_noisy       = np.concatenate([true_cluster, np.zeros(N_OUT, dtype=int)])  # 标记为簇 0

print(f"  Data points: {N} + {N_OUT} outliers = {len(emb10_noisy)}")

# KMeans on noisy data
km_noisy = KMeans(n_clusters=2, random_state=42, n_init=20).fit(emb10_noisy)
# K-Medoids on noisy data
medoid_idx_noisy, _ = kmedoids_pam(emb10_noisy, k=2, random_state=42)

# 比较 PC1 轴上的聚类中心（离群点沿 PC1 方向）
km_c0_pc1    = km_emb.cluster_centers_[:, 0]        # 无离群点时
km_noisy_pc1 = km_noisy.cluster_centers_[:, 0]      # 加离群点后
pam_c0_pc1   = emb10[medoid_idx, 0]                 # 无离群点时（medoid 的 PC1）
pam_noisy_pc1 = emb10_noisy[medoid_idx_noisy, 0]   # 加离群点后（medoid 的 PC1）

km_shift  = np.abs(np.sort(km_noisy_pc1)   - np.sort(km_c0_pc1)).max()
pam_shift = np.abs(np.sort(pam_noisy_pc1)  - np.sort(pam_c0_pc1)).max()
print(f"  KMeans center max shift along PC1:    {km_shift:.4f}")
print(f"  K-Medoids center max shift along PC1: {pam_shift:.4f}")
print(f"  KMeans is {km_shift/max(pam_shift,1e-6):.1f}× more sensitive to outliers")

# ─────────────── 实验 C：1D 预测值 ──────────────────────────────────────
print("\n" + "="*60)
print("Experiment C: 1D prediction values — KMeans vs K-Medoids")
print("="*60)

pred1d = predictions.reshape(-1, 1)
km1d = KMeans(n_clusters=2, random_state=42, n_init=20).fit(pred1d)
m_km1d = best_match_metrics(true_cluster, km1d.labels_)
print(f"  KMeans    centers={km1d.cluster_centers_.flatten()}  "
      f"ARI={m_km1d['ARI']:.4f}")

medoid_1d_idx, pam1d_labels = kmedoids_pam(pred1d, k=2, random_state=42)
m_pam1d = best_match_metrics(true_cluster, pam1d_labels)
pam1d_centers = pred1d[medoid_1d_idx].flatten()
print(f"  K-Medoids medoids={pam1d_centers}  "
      f"ARI={m_pam1d['ARI']:.4f}")
print(f"  Medoids are actual prediction values: {pam1d_centers}")

# ─────────────── 保存结果 ──────────────────────────────────────────────────
records = [
    {"Experiment": "Normal (Emb PCA-10)", "Method": "KMeans",    **m_km},
    {"Experiment": "Normal (Emb PCA-10)", "Method": "K-Medoids", **m_pam},
    {"Experiment": "Normal (Pred 1D)",    "Method": "KMeans",    **m_km1d},
    {"Experiment": "Normal (Pred 1D)",    "Method": "K-Medoids", **m_pam1d},
    {"Experiment": "Outlier (Emb PCA-10)", "Method": "KMeans center shift",
     "ARI": float(km_shift), "NMI": 0, "Accuracy": 0, "F1": 0,
     "TP": 0, "FP": 0, "TN": 0, "FN": 0,
     "Precision": 0, "Recall": 0},
    {"Experiment": "Outlier (Emb PCA-10)", "Method": "K-Medoids center shift",
     "ARI": float(pam_shift), "NMI": 0, "Accuracy": 0, "F1": 0,
     "TP": 0, "FP": 0, "TN": 0, "FN": 0,
     "Precision": 0, "Recall": 0},
]
df_res = pd.DataFrame(records)
df_res.to_csv(OUT_DIR / "kmedoids_metrics.csv", index=False)

# ─────────────── 图 1：PCA-2D 散点（KMeans vs K-Medoids 中心位置）─────────
pca2d = PCA(n_components=2, random_state=42)
emb2d = pca2d.fit_transform(emb_sc)

km2d_centers  = pca2d.transform(scaler.transform(
    scaler.inverse_transform(
        np.pad(km_emb.cluster_centers_, ((0,0),(0, emb_sc.shape[1]-10)), mode='constant')
    )[:, :emb_sc.shape[1]]
))

# Simpler: project cluster centers from PCA-10 back to original, then to PCA-2
# Actually easier: just fit KMeans in PCA-2 directly for visualization
km_viz  = KMeans(n_clusters=2, random_state=42, n_init=20).fit(emb2d)
medoid_viz_idx, pam_viz_labels = kmedoids_pam(emb2d, k=2, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
pca_var = PCA(n_components=2, random_state=42).fit(emb_sc).explained_variance_ratio_
titles  = ["KMeans(k=2) — center is mean (may be virtual)",
           "K-Medoids PAM(k=2) — center is a real data point ★"]
label_sets = [km_viz.labels_, pam_viz_labels]
palette = ["#2196F3", "#E91E63"]

for ax, lbl, title, centers, center_label in zip(
    axes, label_sets, titles,
    [km_viz.cluster_centers_, emb2d[medoid_viz_idx]],
    ["cluster mean", "medoid (real point)"]
):
    for c, color in enumerate(palette):
        mask = lbl == c
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                   s=4, alpha=0.4, color=color, label=f"Cluster {c}")
    ax.scatter(centers[:, 0], centers[:, 1],
               s=200, marker="*", color="black", zorder=10, label=center_label)
    if title.startswith("K-Medoids"):
        # 圈出 medoid 点
        ax.scatter(centers[:, 0], centers[:, 1],
                   s=400, facecolors="none", edgecolors="black", linewidths=2, zorder=9)
    ax.set_xlabel(f"PC1 ({pca_var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_var[1]*100:.1f}%)")
    ax.set_title(title, fontsize=9)
    ax.legend(markerscale=3, fontsize=8)

plt.suptitle("KMeans vs K-Medoids on CNN-LSTM-Attention Embedding (PCA-2D viz)", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "kmeans_vs_kmedoids_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────── 图 2：鲁棒性对比（离群点前后中心位置）────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

pca1d_axis = 0  # 沿 PC1 展示（离群点方向）

for ax, method_name, c_before, c_after in zip(
    axes,
    ["KMeans", "K-Medoids"],
    [np.sort(km_c0_pc1), np.sort(pam_c0_pc1)],
    [np.sort(km_noisy_pc1), np.sort(pam_noisy_pc1)],
):
    x = [0, 1]
    for ci, (b, a) in enumerate(zip(c_before, c_after)):
        ax.plot(x, [b, a], "-o", label=f"Cluster {ci}", lw=2)
    ax.axhline(8.0, ls="--", color="red", alpha=0.7, label="Outlier direction (PC1≈8)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Before outliers", "After outliers"])
    ax.set_ylabel("Cluster center PC1 value")
    ax.set_title(f"{method_name}: center shift = {np.abs(c_after - c_before).max():.3f}")
    ax.legend(fontsize=8)
    ax.set_ylim(-5, 10)

plt.suptitle(f"Outlier Robustness: 50 extreme outliers injected along PC1\n"
             f"KMeans shift / K-Medoids shift = {km_shift/max(pam_shift,1e-6):.1f}×")
plt.tight_layout()
plt.savefig(OUT_DIR / "outlier_robustness.png", dpi=150)
plt.close()

# ─────────────── 图 3：1D 预测值的 KMeans vs K-Medoids ──────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(predictions, bins=80, color="#90CAF9", alpha=0.7, label="Prediction distribution")
km_c = sorted(km1d.cluster_centers_.flatten())
pm_c = sorted(pam1d_centers)
for v, color, lbl in [(km_c[0], "#1565C0", f"KMeans c0={km_c[0]:.0f}W"),
                       (km_c[1], "#1976D2", f"KMeans c1={km_c[1]:.0f}W"),
                       (pm_c[0], "#B71C1C", f"K-Medoids m0={pm_c[0]:.0f}W (real)"),
                       (pm_c[1], "#D32F2F", f"K-Medoids m1={pm_c[1]:.0f}W (real)")]:
    ax.axvline(v, color=color, lw=2, ls="--" if "KMeans" in lbl else "-", label=lbl)
ax.set_xlabel("Predicted OT (W)")
ax.set_ylabel("Count")
ax.set_title("1D Prediction Clustering: KMeans centers (virtual) vs K-Medoids medoids (real)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "1d_pred_clustering.png", dpi=150)
plt.close()

# ─────────────── 终端摘要 ────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nExperiment A — Normal data (Embedding PCA-10):")
print(f"  KMeans     ARI={m_km['ARI']:.4f}  Acc={m_km['Accuracy']:.4f}  F1={m_km['F1']:.4f}")
print(f"  K-Medoids  ARI={m_pam['ARI']:.4f}  Acc={m_pam['Accuracy']:.4f}  F1={m_pam['F1']:.4f}")
print(f"\nExperiment B — Outlier robustness (50 outliers along PC1):")
print(f"  KMeans center shift:    {km_shift:.4f}")
print(f"  K-Medoids center shift: {pam_shift:.4f}")
print(f"  Ratio:                  {km_shift/max(pam_shift,1e-6):.2f}×")
print(f"\nExperiment C — 1D Prediction:")
print(f"  KMeans    centers={sorted(km1d.cluster_centers_.flatten())}  ARI={m_km1d['ARI']:.4f}")
print(f"  K-Medoids medoids={sorted(pam1d_centers.tolist())}  ARI={m_pam1d['ARI']:.4f}")
print(f"\nOutput files saved to {OUT_DIR}/")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}")
