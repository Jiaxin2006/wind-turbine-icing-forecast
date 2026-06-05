#!/usr/bin/env python3
"""
hmm_condition.py — COURSE_COVERAGE Step 5: A14/A15 HMM

用 Gaussian HMM 对 (OT, 环境温度) 双变量序列拟合隐状态模型，
把「高发电 / 低发电 / 潜在结冰异常」当成隐马尔可夫状态。

实验内容：
  1. 拟合 K=2 和 K=3 两个 HMM，观察 BIC 选模
  2. 展示 K=3 的状态序列与 OT/温度时序叠加图
  3. 分析每个隐状态的均值和协方差，对应实际物理含义
  4. 与 KMeans(k=3) 聚类结果对比（ARI/NMI）
  5. 转移矩阵分析：状态间切换概率

无 torch 依赖。
运行：python3 hmm_condition.py
输出目录：output_hmm/
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
try:
    from hmmlearn.hmm import GaussianHMM
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: hmmlearn. Run `pip install -r requirements.txt` first.") from exc

SEED = 42
np.random.seed(SEED)
OUT_DIR = Path("output_hmm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────── 读取数据 ─────────────────────────────────────
_COL_MAP = {"time": "统计时间", "OT": "OT",
            "exog_temp": "Exogenous1", "exog_wind": "Exogenous2"}

def _load(path):
    df = pd.read_excel(path)
    rev = {v: k for k, v in _COL_MAP.items()}
    df = df.rename(columns={o: n for o, n in rev.items() if o in df.columns})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    for c in ["OT", "exog_temp", "exog_wind"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.interpolate(limit=5).bfill().ffill()

print("Reading data...")
df = _load("标注的数据-#67_1.xlsx")
print(f"  Total samples: {len(df)}")

# 使用全量时序（不切分 train/test），HMM 是无监督生成模型
# 观测维度：(OT, exog_temp)，标准化后输入
OBS_COLS = ["OT", "exog_temp"]
X_raw = df[OBS_COLS].values.astype(float)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)          # shape (N, 2)
lengths = [len(X)]                        # 整条时序作为一个序列

# ──────────────────────────── 1. BIC 选模 ──────────────────────────────────
print("\n" + "="*60)
print("1. HMM Model Selection via BIC (K=2,3,4)")
print("="*60)

bic_records = []
fitted_hmms = {}
for K in [2, 3, 4]:
    model = GaussianHMM(
        n_components=K,
        covariance_type="full",
        n_iter=200,
        tol=1e-4,
        random_state=SEED,
    )
    model.fit(X, lengths)
    log_lik = model.score(X, lengths)
    # BIC = -2 * log_likelihood + n_params * ln(N)
    # Gaussian HMM full cov: n_params = K*(K-1) + K + K*d + K*d*(d+1)/2
    d = X.shape[1]
    n_trans = K * (K - 1)          # transition (rows sum to 1, so K*(K-1) free)
    n_init  = K - 1                # initial distribution
    n_mean  = K * d
    n_cov   = K * d * (d + 1) // 2
    n_params = n_trans + n_init + n_mean + n_cov
    N        = len(X)
    bic = -2 * log_lik + n_params * math.log(N)
    bic_records.append(dict(K=K, LogLik=log_lik, n_params=n_params, BIC=bic))
    fitted_hmms[K] = model
    print(f"  K={K}: log_lik={log_lik:.1f}  n_params={n_params}  BIC={bic:.1f}")

df_bic = pd.DataFrame(bic_records)
df_bic.to_csv(OUT_DIR / "bic_scores.csv", index=False)
best_K = df_bic.loc[df_bic["BIC"].idxmin(), "K"]
print(f"\n  Best K by BIC: {best_K}")

# 绘 BIC 折线图
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df_bic["K"], df_bic["BIC"], "o-", color="#1565C0", lw=2, markersize=8)
ax.set_xlabel("Number of hidden states K")
ax.set_ylabel("BIC")
ax.set_title("HMM Model Selection — BIC vs K")
ax.axvline(best_K, ls="--", color="red", lw=1.5, label=f"Best K={best_K}")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "bic_curve.png", dpi=150)
plt.close()

# ──────────────────────────── 2. 用 K=3 分析状态 ─────────────────────────────
K_MAIN = 3
model3 = fitted_hmms[K_MAIN]
states3 = model3.predict(X, lengths)    # Viterbi 最优状态序列

print("\n" + "="*60)
print(f"2. K={K_MAIN} HMM — State Statistics (original scale)")
print("="*60)

# 将 HMM 均值反标准化，还原到原始量纲
means_orig  = scaler.inverse_transform(model3.means_)   # (K, 2)
counts = [(states3 == k).sum() for k in range(K_MAIN)]

state_info = []
for k in range(K_MAIN):
    m_ot, m_temp = means_orig[k]
    n = counts[k]
    pct = 100.0 * n / len(states3)
    print(f"  State {k}: OT_mean={m_ot:.1f}W  Temp_mean={m_temp:.1f}°C  N={n} ({pct:.1f}%)")
    state_info.append(dict(State=k, OT_mean=m_ot, Temp_mean=m_temp, Count=n, Pct=pct))

pd.DataFrame(state_info).to_csv(OUT_DIR / "state_stats.csv", index=False)

# 转移矩阵
print("\n  Transition Matrix (row=from, col=to):")
trans = model3.transmat_
for i in range(K_MAIN):
    row_str = "  ".join(f"{trans[i,j]:.3f}" for j in range(K_MAIN))
    print(f"    State {i} -> [{row_str}]")
pd.DataFrame(trans, columns=[f"to_{k}" for k in range(K_MAIN)],
             index=[f"from_{k}" for k in range(K_MAIN)]
            ).to_csv(OUT_DIR / "transition_matrix.csv")

# ──────────────────────────── 3. 时序可视化 ───────────────────────────────────
# 取全量时序的前 5000 点可视化（约 3.5 天）
VIS_N = min(5000, len(df))
t_idx = np.arange(VIS_N)
PALETTE = ["#1565C0", "#EF6C00", "#2E7D32"]
state_colors = [PALETTE[s] for s in states3[:VIS_N]]

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

# OT 时序 + 状态背景
ax = axes[0]
for k in range(K_MAIN):
    mask = states3[:VIS_N] == k
    ax.scatter(t_idx[mask], X_raw[:VIS_N][mask, 0],
               s=3, color=PALETTE[k], alpha=0.6, label=f"State {k}")
ax.set_ylabel("OT (W)")
ax.set_title(f"K={K_MAIN} HMM — State Sequence (first {VIS_N} steps)")
ax.legend(markerscale=4, fontsize=9)

# 温度时序 + 状态着色
ax = axes[1]
for k in range(K_MAIN):
    mask = states3[:VIS_N] == k
    ax.scatter(t_idx[mask], X_raw[:VIS_N][mask, 1],
               s=3, color=PALETTE[k], alpha=0.6)
ax.set_ylabel("Ambient Temp (°C)")

# 隐状态序列
ax = axes[2]
ax.plot(t_idx, states3[:VIS_N], color="#37474F", lw=0.8)
ax.set_yticks(range(K_MAIN))
ax.set_yticklabels([f"S{k}" for k in range(K_MAIN)])
ax.set_ylabel("Hidden State")
ax.set_xlabel("Time step (1 min intervals)")

plt.tight_layout()
plt.savefig(OUT_DIR / "hmm_state_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()

# ──────────────────────────── 4. 状态分布散点图 ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for k in range(K_MAIN):
    mask = states3 == k
    ax.scatter(X_raw[mask, 1], X_raw[mask, 0],
               s=3, alpha=0.3, color=PALETTE[k], label=f"State {k} (n={counts[k]})")
    # 绘制状态均值标记
    ax.scatter([means_orig[k, 1]], [means_orig[k, 0]],
               s=200, marker="*", color=PALETTE[k], edgecolors="black", zorder=5)

ax.set_xlabel("Ambient Temp (°C)")
ax.set_ylabel("OT (W)")
ax.set_title(f"HMM K={K_MAIN}: State Distribution in (Temp, OT) Space\n(stars = state means)")
ax.legend(markerscale=4)
plt.tight_layout()
plt.savefig(OUT_DIR / "hmm_scatter.png", dpi=150)
plt.close()

# ──────────────────────────── 5. 与 KMeans 对比 ──────────────────────────────
print("\n" + "="*60)
print(f"4. HMM vs KMeans (K={K_MAIN}) — Consistency")
print("="*60)

kmeans = KMeans(n_clusters=K_MAIN, random_state=SEED, n_init=10)
km_labels = kmeans.fit_predict(X)

ari = adjusted_rand_score(states3, km_labels)
nmi = normalized_mutual_info_score(states3, km_labels, average_method="arithmetic")
print(f"  ARI  = {ari:.4f}  (1=perfect, 0=random)")
print(f"  NMI  = {nmi:.4f}  (1=perfect, 0=random)")

pd.DataFrame([dict(ARI=ari, NMI=nmi)]).to_csv(OUT_DIR / "hmm_vs_kmeans.csv", index=False)

# 混淆矩阵热力图（HMM rows, KMeans cols）
conf = np.zeros((K_MAIN, K_MAIN), dtype=int)
for h, m in zip(states3, km_labels):
    conf[h, m] += 1

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(conf, cmap="Blues")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(K_MAIN), [f"KMeans {k}" for k in range(K_MAIN)])
ax.set_yticks(range(K_MAIN), [f"HMM {k}" for k in range(K_MAIN)])
for i in range(K_MAIN):
    for j in range(K_MAIN):
        ax.text(j, i, str(conf[i, j]), ha="center", va="center", color="black")
ax.set_xlabel("KMeans label")
ax.set_ylabel("HMM state")
ax.set_title(f"HMM vs KMeans Assignment\nARI={ari:.3f}  NMI={nmi:.3f}")
plt.tight_layout()
plt.savefig(OUT_DIR / "hmm_vs_kmeans_conf.png", dpi=150)
plt.close()

# ──────────────────────────── 6. K=2 vs K=3 对比 ─────────────────────────────
print("\n" + "="*60)
print("5. K=2 HMM — State Statistics")
print("="*60)
model2 = fitted_hmms[2]
states2 = model2.predict(X, lengths)
means2_orig = scaler.inverse_transform(model2.means_)
for k in range(2):
    m_ot, m_temp = means2_orig[k]
    n = (states2 == k).sum()
    print(f"  State {k}: OT_mean={m_ot:.1f}W  Temp_mean={m_temp:.1f}°C  N={n}")

# ─────────────────────────── 终端摘要 ────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nBIC scores:")
print(df_bic[["K", "BIC"]].to_string(index=False))
print(f"\nBest K = {best_K}")
print("\nK=3 state means (original scale):")
for info in state_info:
    print(f"  State {info['State']}: OT={info['OT_mean']:.1f}W  Temp={info['Temp_mean']:.1f}°C  "
          f"({info['Pct']:.1f}%)")
print(f"\nHMM vs KMeans consistency: ARI={ari:.4f}  NMI={nmi:.4f}")
print(f"\nOutput files in {OUT_DIR}/:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}")
