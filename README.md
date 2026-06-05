# 风机结冰相关工况下的时间序列预测与集成建模研究

> 《机器学习概论》课程大实验 · 清华大学计算机系  
> 完整报告见 `[Course_Report.md](Course_Report.md)`

---

## 背景与任务

- **数据**：国网冀北电力有限公司承德供电公司，2024 年 2 月全月，约 41,760 条分钟级记录。字段：时间戳、温度（`exog_temp`）、风速（`exog_wind`）、运行功率（`OT`，单位 W）。主数据文件 `标注的数据-#67_1.xlsx`（本地放置，不入库）。
- **预测任务**：利用温度/风速、`OT` 滞后项与滚动统计，预测下一时刻 `OT`。
- **工况分析**：无逐分钟人工结冰标签；聚类用于工况划分与停机决策支持，**主评测**为 `OT_true < 1000 kW` 停机代理（与报告 §3.6.9 一致），**不得**与 OT-KMeans 参照分群的匈牙利 Accuracy 混比。

---

## 代码结构（仓库内脚本）

```
code/
├── core.py                     # 共用：数据加载、指标、SeqDataset、NN 模型、训练循环
│
├── ── 预测主链路 ──────────────────────────────────────────
├── ensemble.py                 # 混合池探索：RF+SVR+CNN/LSTM/Transformer → RidgeCV stacking
├── baselines_ext.py            # 传统基线 + Holdout Blending stacking（RF/GBM/KNN/BR/SVR）
├── model.py                    # 基础模型单独对比（RF/SVR/LSTM/CNN/Transformer）
├── RandomForest.py             # 多输出 RF 基线（探索用）
├── final_evaluation.py         # 读预测 CSV → 统一指标表 + 配对检验 + 图表
│
├── ── 序列模型 ────────────────────────────────────────────
├── train_cnn_lstm.py           # CNN-LSTM-Attention，分工况训练
├── cnn_lstm_grid_search.py     # CNN-LSTM 超参网格搜索
├── cnn_lstm_attention_ot.py    # 早期单次版本（历史存档）
│
├── ── 集成 / Stacking 分析 ────────────────────────────────
├── oof_meta_shift_analysis.py  # Holdout vs OOF 元特征分布与 Ridge 消融（§5.11.5）
│
├── ── 聚类与工况分析 ──────────────────────────────────────
├── embedding_analysis.py       # 抽 embedding → KMeans → ARI/NMI（补充参照）
├── cluster_eval.py             # 停机基线（OT<1000 kW）二分类评估
├── ot_threshold_rationale.py     # 1000 kW 与 OT-KMeans 分界一致性验证（§3.6.9.0）
├── cluster_agglomerative.py    # AgglomerativeClustering vs KMeans
├── cluster_kmedoids.py         # 手写 PAM K-Medoids 对比
│
├── ── 可靠性验证 ──────────────────────────────────────────
├── bootstrap_cv_eval.py        # TimeSeriesSplit(5) CV + Bootstrap 置信区间
│
├── ── 参考工具（路径硬编码） ──────────────────────────────
├── evaluate.py
├── visualize.py
│
├── Course_Report.md            # 课程报告
├── Proposal.md
└── requirements.txt
```

> 产物目录（`output_*`、`out_*`、`final_results/` 等）由 `.gitignore` 排除，运行脚本后在本地生成。

---

## 快速复现

### 环境

```bash
pip install -r requirements.txt
```

需本地准备 `标注的数据-#67_1.xlsx`（与 `core.py` / 各脚本读取路径一致）。

### 预测层（§4）

```bash
# 混合池 stacking 探索（RF/SVR/CNN/LSTM/Transformer）
python3 ensemble.py
python3 final_evaluation.py

# 传统基线 + Holdout RidgeCV stacking
python3 baselines_ext.py

# 仅汇总已有预测文件
python3 final_evaluation.py
```

### 集成策略分析（Holdout vs OOF，§5.11）

```bash
python3 oof_meta_shift_analysis.py
# → output_stacking_compare/oof_shift_*.csv, oof_shift_*.png
```

### 聚类与停机评测（§5）

```bash
python3 train_cnn_lstm.py          # → out_cnn_lstm_cluster_1/
python3 embedding_analysis.py      # ARI/NMI vs OT-KMeans（补充）
python3 cluster_eval.py            # 停机基线 Accuracy/F1
python3 cluster_agglomerative.py   # 层次聚类对比
python3 cluster_kmedoids.py        # K-Medoids 对比
```

### 交叉验证与 Bootstrap（§4.6）

```bash
python3 bootstrap_cv_eval.py
# → output_bootstrap_cv/cv_results.csv, bootstrap_ci.csv
```

---

## 主要结果

### 预测层：基础模型（测试集 N = 8,349，§4.1–4.2）


| 模型                | MAE (W)   | RMSE (W) | R²    | 说明                 |
| ----------------- | --------- | -------- | ----- | ------------------ |
| **SVR**           | **48.88** | 81.32    | 0.991 | 第一梯队，MAE 最低        |
| **Random Forest** | 49.23     | 86.47    | 0.990 | 与 SVR 接近           |
| **GBM**           | 50.19     | 83.25    | 0.991 | `baselines_ext.py` |
| **BayesianRidge** | 74.09     | 110.69   | 0.984 |                    |
| **KNN**           | 82.16     | 129.35   | 0.978 |                    |
| **AdaBoost**      | 187.50    | 221.46   | 0.934 | 补充基线               |
| LSTM              | 221.35    | 382.49   | 0.977 | 无强 OT 滞后通道         |
| Transformer       | 225.56    | 385.80   | 0.976 |                    |
| CNN               | 257.78    | 407.62   | 0.970 |                    |


**序列扩展**（不同测试窗口，不参与 paired test）：CNN-LSTM-Attention MAE **67.47 W**；CNN-LSTM MAE **67.56 W**（§4.3）。

### 集成层（§4.4 口径说明）


| 实验             | 脚本                           | 基模型池               | 策略                | 代表 MAE (W) | RMSE (W)  |
| -------------- | ---------------------------- | ------------------ | ----------------- | ---------- | --------- |
| 混合池探索          | `ensemble.py`                | RF+SVR+CNN+LSTM+TR | Holdout + RidgeCV | 49.59      | **79.05** |
| 强基线池           | `baselines_ext.py`           | RF+GBM+KNN+BR+SVR  | Holdout + RidgeCV | 51.75      | 79.87     |
| OOF 全量（对照失败）   | `oof_meta_shift_analysis.py` | 同上                 | OOF + RidgeCV     | ~176       | ~203      |
| OOF 仅后 20%（消融） | 同上                           | 同上                 | OOF 子集 + RidgeCV  | ~51        | —         |
| NNLS（失败对照）     | `ensemble.py`                | 混合池                | 非负加权              | ~947       | —         |


**结论**：

- 所有 stacking 变体的 **MAE 均未稳定低于 SVR**；Holdout stacking 主要 **降低 RMSE**（缓解大误差）。
- **OOF 在时序 Expanding-Window 下失败**：meta 特征按时间混入「小样本折 / 大样本折」预测，与测试时全量重训基模型分布不一致；仅用 OOF 时间后 20% 训二层可恢复至 ~51 W（见 §5.11.5）。
- 主流程采用 **Holdout Blending + RidgeCV**（`ensemble.py` / `baselines_ext.py`）。

### 聚类与停机决策（N = 8,347，**主评测 OT < 1000 kW**，§5）


| 方法                     | 特征            | Accuracy   | F1         | ARI vs OT-KMeans |
| ---------------------- | ------------- | ---------- | ---------- | ---------------- |
| **KMeans**             | **Embedding** | **94.67%** | **95.45%** | 0.815            |
| KMeans                 | Prediction    | 93.96%     | 94.92%     | 0.815            |
| Agglomerative-complete | Embedding     | 89.82%     | 91.80%     | 0.718            |
| Agglomerative-ward     | Embedding     | 86.43%     | 89.40%     | 0.619            |
| Agglomerative-average  | Embedding     | 88.57%     | 89.06%     | 0.500            |
| K-Medoids (PCA-10)     | Embedding     | —          | —          | 0.542（补充）        |


> **94.67%** 表示 embedding 聚类与停机代理规则的一致性，**不是**结冰检测真值。旧版 0.952 为相对 `true_cluster` 的匈牙利对齐值，勿与主 Accuracy 并列。

### 统计检验与置信区间（§4.6）

- **配对检验**：RF vs SVR t-test p≈0.44（MAE 差异小）；SVR vs Holdout stacking 约 +2.4 W（显著但工程差距小）。
- **CV**：RF CV MAE 65.8±16.1 W vs 测试 49.4 W（expanding-window 早期折更悲观）。
- **Bootstrap**：SVR/RF/GBM MAE 95% CI 宽约 2.5–3 W，相互重叠。

详见 `Course_Report.md` §4.6 与 `bootstrap_cv_eval.py` 输出。

---

## 实验阶段与报告章节


| 阶段  | 内容                  | 报告章节        | 主要脚本                                                            |
| --- | ------------------- | ----------- | --------------------------------------------------------------- |
| 1   | 传统回归基线              | §4.1        | `baselines_ext.py`, `model.py`                                  |
| 2   | 序列模型与 NAS           | §4.2–4.3    | `ensemble.py`, `train_cnn_lstm.py`, `cnn_lstm_grid_search.py`   |
| 3   | Stacking / 集成       | §4.4, §5.11 | `ensemble.py`, `baselines_ext.py`, `oof_meta_shift_analysis.py` |
| 4   | 聚类与工况               | §5          | `embedding_analysis.py`, `cluster_`*                            |
| 5   | CV / Bootstrap / 检验 | §4.6        | `bootstrap_cv_eval.py`, `final_evaluation.py`                   |


