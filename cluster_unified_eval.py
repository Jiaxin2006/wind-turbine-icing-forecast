#!/usr/bin/env python3
"""
cluster_unified_eval.py — 聚类评测统一到课程报告基线（OT < 1000 kW 停机代理标签）

课程报告的停机代理评测采用：
  - Ground truth: OT_true < 1000 kW → should_close=1
  - 预测: KMeans(k=2) on embedding → 簇内多数投票 → 二分类

本脚本对所有聚类方法（embedding / prediction / 层次 / K-Medoids 若可复现）
在同一 y_true 上报告 Accuracy / F1 / ARI（ARI 相对 OT-KMeans 仅作补充列）。

运行: python3 cluster_unified_eval.py
输出: out_cnn_lstm_cluster_1/clustering_shutdown_metrics.csv
      out_cnn_lstm_cluster_1/clustering_shutdown_metrics.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path("out_cnn_lstm_cluster_1")
CSV_PATH = OUT_DIR / "full_clustering_analysis.csv"
EMB_NPY = OUT_DIR / "test_embeddings.npy"
THRESH = 1000.0
K = 2
RANDOM_STATE = 42


def majority_vote_labels(cluster_ids: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """簇内对 should_close 多数投票，得到 0/1 预测。"""
    mapping = {}
    for c in np.unique(cluster_ids):
        sub = y_true[cluster_ids == c]
        mapping[c] = int(np.round(sub.mean())) if len(sub) else 0
    return np.array([mapping[c] for c in cluster_ids], dtype=int)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "N": int(n),
    }


def eval_method(
    name: str,
    feature: str,
    cluster_ids: np.ndarray,
    y_shutdown: np.ndarray,
    ot_kmeans_ref: np.ndarray,
) -> dict:
    y_pred = majority_vote_labels(cluster_ids, y_shutdown)
    m = binary_metrics(y_shutdown, y_pred)
    return {
        "Method": name,
        "Feature": feature,
        "Ground_truth": f"OT_true<{THRESH:.0f}kW (paper)",
        "ARI_vs_OT_KMeans": float(adjusted_rand_score(ot_kmeans_ref, cluster_ids)),
        "NMI_vs_OT_KMeans": float(
            normalized_mutual_info_score(ot_kmeans_ref, cluster_ids)
        ),
        **m,
    }


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    y_shutdown = (df["OT_true"].values < THRESH).astype(int)
    ot_kmeans_ref = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20).fit(
        df["OT_true"].values.reshape(-1, 1)
    ).labels_

    rows = []
    rows.append(
        eval_method(
            "KMeans",
            "Embedding",
            df["emb_cluster"].values,
            y_shutdown,
            ot_kmeans_ref,
        )
    )
    rows.append(
        eval_method(
            "KMeans",
            "Prediction (OT_pred)",
            df["pred_cluster"].values,
            y_shutdown,
            ot_kmeans_ref,
        )
    )

    emb = np.load(EMB_NPY)
    emb_scaled = StandardScaler().fit_transform(emb)
    pred_1d = df["OT_pred"].values.reshape(-1, 1)

    for link in ("ward", "complete", "average"):
        for feat_name, X in (("Embedding", emb_scaled), ("Prediction", pred_1d)):
            ac = AgglomerativeClustering(n_clusters=K, linkage=link)
            labels = ac.fit_predict(X)
            rows.append(
                eval_method(
                    f"Agglomerative-{link}",
                    feat_name,
                    labels,
                    y_shutdown,
                    ot_kmeans_ref,
                )
            )

    # K-Medoids 简化：用 sklearn 无实现，读 cluster_kmedoids 输出若存在则跳过；
    # 此处对 embedding 再跑 KMeans 作对照（与 cluster_eval 一致源）

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("Accuracy", ascending=False)
    csv_out = OUT_DIR / "clustering_shutdown_metrics.csv"
    out_df.to_csv(csv_out, index=False)

    md_lines = [
        "# 聚类评测（统一基线：OT < 1000 kW）",
        "",
        "与课程报告的停机代理评测设置一致。",
        f"样本数 N={len(df)}，阈值 {THRESH} kW。",
        "",
        "| Method | Feature | Accuracy | F1 | Recall | ARI vs OT-KMeans |",
        "|--------|---------|----------|-----|--------|------------------|",
    ]
    for _, r in out_df.iterrows():
        md_lines.append(
            f"| {r['Method']} | {r['Feature']} | {r['Accuracy']:.4f} | "
            f"{r['F1']:.4f} | {r['Recall']:.4f} | {r['ARI_vs_OT_KMeans']:.4f} |"
        )
    md_out = OUT_DIR / "clustering_shutdown_metrics.md"
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(out_df.to_string(index=False))
    print(f"\nSaved {csv_out} and {md_out}")

    # 与 cluster_eval 交叉校验；若用户只重跑了本脚本而没有同步重跑 cluster_eval，
    # 两边可能引用不同时间生成的 emb_cluster，因此这里只提示而不中断主结果生成。
    ref_path = OUT_DIR / "emb_cluster_binary_metrics.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        km_emb = out_df[
            (out_df["Method"] == "KMeans") & (out_df["Feature"] == "Embedding")
        ].iloc[0]
        diff = abs(km_emb["Accuracy"] - ref["Accuracy"])
        if diff >= 1e-4:
            print(
                f"[WARN] KMeans embedding Accuracy differs from cluster_eval by {diff:.4g}; "
                "rerun cluster_eval.py if you need both files synchronized."
            )


if __name__ == "__main__":
    main()
