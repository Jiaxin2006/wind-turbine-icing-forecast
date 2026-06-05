#!/usr/bin/env python3
"""
ot_threshold_rationale.py — 验证 OT<1000 kW 停机阈值与真实功率 KMeans 二分的关系

对测试集 OT_true 做 KMeans(k=2)（与 embedding_analysis.py 中 OT-KMeans 参照一致），
比较固定阈值 1000 kW 与数据驱动分界（两簇质心中点）的一致程度。

运行：python3 clustering/ot_threshold_rationale.py
输出：output_cluster_threshold/ot_threshold_rationale.csv（及终端摘要）
"""


from pathlib import Path as _Path
import os as _os
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
_os.chdir(_PROJECT_ROOT)

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score

OUT_DIR = Path("output_cluster_threshold")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = Path("out_cnn_lstm_cluster_1/full_clustering_analysis.csv")
THRESH = 1000.0
SEED = 42


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    ot = df["OT_true"].values.astype(float)
    n = len(ot)

    km = KMeans(n_clusters=2, random_state=SEED, n_init=20)
    labels = km.fit_predict(ot.reshape(-1, 1))
    c_low, c_high = sorted(km.cluster_centers_.ravel())
    boundary = (c_low + c_high) / 2.0
    low_label = 0 if km.cluster_centers_[0, 0] < km.cluster_centers_[1, 0] else 1
    km_low = (labels == low_label).astype(int)

    rule_1000 = (ot < THRESH).astype(int)
    acc = accuracy_score(km_low, rule_1000)
    ari = adjusted_rand_score(km_low, rule_1000)

    band = ot[(ot >= THRESH) & (ot < boundary)]
    flip_n = int((rule_1000 != (ot < boundary).astype(int)).sum())

    scan_rows = []
    for t in range(800, 1201, 50):
        r = (ot < t).astype(int)
        scan_rows.append(
            {
                "threshold_W": t,
                "acc_vs_kmeans_low": accuracy_score(km_low, r),
                "ari_vs_kmeans_low": adjusted_rand_score(km_low, r),
            }
        )
    scan_df = pd.DataFrame(scan_rows)

    summary = pd.DataFrame(
        [
            {
                "N_test": n,
                "centroid_low_W": c_low,
                "centroid_high_W": c_high,
                "kmeans_midpoint_W": boundary,
                "midpoint_minus_1000_W": boundary - THRESH,
                "OT_median_W": float(np.median(ot)),
                "pct_OT_below_1000": 100.0 * (ot < THRESH).mean(),
                "pct_kmeans_low_cluster": 100.0 * km_low.mean(),
                "acc_1000_vs_kmeans": acc,
                "ari_1000_vs_kmeans": ari,
                "n_in_band_1000_to_midpoint": len(band),
                "pct_in_band_1000_to_midpoint": 100.0 * len(band) / n,
                "n_flip_1000_vs_midpoint": flip_n,
            }
        ]
    )

    summary.to_csv(OUT_DIR / "ot_threshold_rationale.csv", index=False)
    scan_df.to_csv(OUT_DIR / "ot_threshold_scan.csv", index=False)

    print("=== OT threshold vs KMeans(k=2) on test OT_true ===")
    print(summary.to_string(index=False))
    print("\nThreshold scan (800–1200 W, step 50):")
    print(scan_df.to_string(index=False))
    print(f"\nSaved: {OUT_DIR}/ot_threshold_rationale.csv")


if __name__ == "__main__":
    main()
