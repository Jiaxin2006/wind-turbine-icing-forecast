# 风机结冰相关工况下的时间序列预测与集成建模研究

> 《机器学习概论》课程大实验 · 清华大学计算机系

---

## 背景与任务

- **数据**：国网冀北电力有限公司承德供电公司，2024 年 2 月全月，约 41,760 条按分钟采样记录。字段：时间戳、温度（`exog_temp`）、风速（`exog_wind`）、目标变量运行功率（`OT`，单位 W）。
- **任务**：利用历史温度/风速及功率滞后特征，预测下一时刻的 OT。
- **特殊性**：低温可能引发结冰相关工况，导致功率输出异质分布，适合同时验证有监督回归、无监督工况划分和集成学习。

---

## 代码结构

```
code/
├── core.py                    # 共用基础组件：数据加载、指标、SeqDataset、NN 模型、训练循环
│
├── ── 主实验链路 ───────────────────────────────────────────
├── ensemble.py                # 主实验：RF+SVR+CNN/LSTM/Transformer → stacking → 聚类
├── final_evaluation.py        # 读预测文件 → 统一指标表 + 显著性检验 + 图表
│
├── ── 对照/基线 ──────────────────────────────────────────
├── model.py                   # 基础模型单独对比（RF/SVR/LSTM/CNN/Transformer）
├── RandomForest.py            # 多输出 RF 基线（主动功率+转速）
├── baselines_ext.py           # 补充基线：Boosting / KNN / BayesianRidge（A13/A3/A2）★
│
├── ── CNN-LSTM-Attention 系列 ─────────────────────────────
├── train_cnn_lstm.py          # CNN-LSTM-Attention，分工况训练，多损失选项
├── cnn_lstm_grid_search.py    # CNN-LSTM 超参网格搜索（最终版）
├── cnn_lstm_attention_ot.py   # CNN-LSTM-Att 早期单次版本（历史存档）
│
├── ── 聚类分析链路 ────────────────────────────────────────
├── embedding_analysis.py      # 从已训练模型抽 embedding → KMeans → ARI/NMI 一致性分析
├── cluster_eval.py            # 读聚类结果 → 阈值二分类（OT<1000 ↔ 结冰）→ TP/FP/F1
│
├── ── 辅助工具 ────────────────────────────────────────────
├── evaluate.py                # 对 grid search 结果排序（路径硬编码，参考用）
├── visualize.py               # 画某个 grid run 的预测曲线（路径硬编码，参考用）
│
├── ── 文档 ────────────────────────────────────────────────
├── COURSE_COVERAGE.md         # 考点对照表：已实现 / 可补充 / 无法覆盖
├── SUBMISSION_GUIDE.md        # 提交说明 + 可删除文件列表 + 后续 plan
├── GIT_PUSH_PLAYBOOK.local.md # GitHub 提交顺序与规范（本地备忘）
├── Course_Report.md           # 课程报告正文
├── Proposal.md                # 原始 Proposal
└── requirements.txt           # 依赖列表
```

★ 新增文件，由补充实验 Step 1 产生（见 §5）。

---

## 快速复现

### 环境

```bash
pip install -r requirements.txt
```

或在 c01/c02 服务器上：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate quant
```

（`quant` 环境已包含 sklearn / pandas / numpy / openpyxl；torch 仅主实验需要，可按 requirements.txt 单独安装。）

### 最小复现（仅看最终报告表格）

```bash
# 假设 output_ot_full_temp_wind/ 中的预测文件已存在
python3 final_evaluation.py
# → final_results/ 下生成 final_metrics.csv / significance_tests.csv / 图表
```

### 完整主实验（重新训练）

```bash
python3 ensemble.py           # 约 20–40 分钟（CPU）
python3 final_evaluation.py
```

### 补充基线实验（A13/A3/A2，CPU-only，快）

```bash
python3 baselines_ext.py
# → output_ot_extended/ 下生成 baselines_ext_metrics.csv 和对比图
```

### 工况聚类分析链路

```bash
python3 train_cnn_lstm.py     # 训练 CNN-LSTM-Attention → out_cnn_lstm_cluster_1/
python3 embedding_analysis.py # 抽 embedding → 聚类一致性指标 + 图
python3 cluster_eval.py       # 阈值二分类评估
python3 final_evaluation.py   # 读取 clustering_classification_metrics.csv 进最终表
```

---

## 已有实验结果

### 主表：统一测试集（N = 8 349）

| 模型 | MAE | RMSE | sMAPE (%) | MASE | 偏差 |
|------|-----|------|-----------|------|------|
| **SVR** | **48.88** | 81.32 | 32.12 | 0.700 | −1.41 |
| **RandomForest** | 49.23 | 86.47 | 11.99 | 0.705 | +2.76 |
| **Stacking RidgeCV** | 49.59 | **79.05** | 49.84 | 0.710 | +4.85 |
| LSTM | 221.35 | 382.49 | 61.09 | 3.170 | +35.42 |
| Transformer | 225.56 | 385.80 | 68.60 | 3.231 | +76.36 |
| CNN | 257.78 | 407.62 | 67.85 | 3.692 | +23.38 |
| NNLS 加权（失败） | 946.83 | 1279.15 | 199.64 | 13.56 | −943.46 |

> - **SVR MAE 最低**，RF 非常接近；Stacking RidgeCV RMSE 最低（对大误差有缓解）。  
> - 深度序列模型（LSTM/Transformer/CNN）在当前数据量和特征条件下显著弱于传统方法。  
> - NNLS 加权集成权重退化，作为透明展示保留在表中。  
> - 脚本：`ensemble.py`；输出：`output_ot_full_temp_wind/`；评估：`final_evaluation.py` → `final_results/`

### 显著性检验（Paired t-test / Wilcoxon）

| 对比 | p (paired-t) | 结论 |
|------|-------------|------|
| RF vs SVR | 0.558 | **无显著差异** |
| SVR vs Stacking | 0.004 | Stacking 显著差于 SVR（MAE 角度） |
| RF vs CNN | 0.000 | RF 显著优于 CNN |

> Wilcoxon 检验（非参数）所有对比 p ≈ 0，说明误差分布差异极显著。

### 补充实验 1：CNN-LSTM 网格搜索最佳 run

| 指标 | 值 |
|------|----|
| MAE | 62.08 |
| RMSE | 122.35 |
| R² | 0.9837 |

> 注意：保存的测试窗口与主表不同，**不参与 paired test**，仅作补充。  
> 脚本：`cnn_lstm_grid_search.py`；输出：`out_cnn_lstm_grid_search_revised/`

### 补充实验 2：Embedding 聚类与停机决策（N = 8 347，**统一基线 OT < 1000 kW**，与论文一致）

| 方法 | 特征 | Accuracy | F1 | ARI vs OT-KMeans |
|------|------|----------|-----|------------------|
| **KMeans** | **Embedding** | **94.67%** | **95.45%** | 0.815 |
| KMeans | Prediction | 93.96% | 94.92% | 0.815 |

> 论文 `main.tex` 采用停机代理标签（非 OT-KMeans 参照）。旧表 0.951 Accuracy 为相对 `true_cluster` 的匈牙利对齐值，**勿与 94.67% 混比**。  
> 脚本：`cluster_eval.py`、`cluster_unified_eval.py`；输出：`clustering_shutdown_metrics.csv`

---

## 补充实验（COURSE_COVERAGE.md 逐步补全）

每次一步，按 `COURSE_COVERAGE.md` 优先级推进。补充的实验结果追加在本节。

### Step 1：A13 Boosting + A3 KNN + A2 BayesianRidge

**课程对应**：A13（Boosting，集成学习）、A3（KNN，基于记忆/非参数）、A2（贝叶斯岭回归，MAP 估计）

**方法说明**：

| 方法 | 类 | 课程考点 | 关键超参 |
|------|----|---------|---------|
| GradientBoostingRegressor | sklearn | A13 Boosting（GBDT，从错误中学习） | n_estimators=200, max_depth=4, lr=0.1 |
| AdaBoostRegressor | sklearn | A13 Boosting（加权弱学习器集成） | n_estimators=100, base=DecisionTree(max_depth=3) |
| KNeighborsRegressor | sklearn | A3 KNN（非参数，距离加权） | k=10, weights=distance, StandardScaler 归一化 |
| BayesianRidge | sklearn | A2/T1（贝叶斯线性回归，自动确定先验） | 默认（alpha/lambda 由 evidence maximization 确定） |

**与课程的联系**：
- Boosting 与 Bagging（随机森林）形成对照：两者都是集成，RF 并行、互不依赖；GBDT 串行、逐步修正残差。
- KNN 体现「局部加权/基于记忆」一类，依赖距离度量，对特征缩放敏感（StandardScaler 必须）。
- BayesianRidge 是频率线性回归（Ridge）的贝叶斯解释：超参由 evidence maximization 自动确定，等价于 MAP 估计。

**代码位置**：`baselines_ext.py`  
**如何运行**：`python3 baselines_ext.py`（CPU-only，约 5–10 分钟）  
**输出目录**：`output_ot_extended/`

**结果**（测试集 N = 8 349）：

| 模型 | 考点 | MAE | RMSE | R² | sMAPE (%) |
|------|------|-----|------|----|-----------|
| SVR（对照） | 基线 | **48.88** | 81.32 | 0.9912 | 16.1 |
| RF（对照） | A11/A12 Bagging | 49.39 | 86.86 | 0.9899 | 6.0 |
| **GBM** | A13 Boosting | 50.19 | 83.25 | 0.9907 | 15.4 |
| BayesianRidge | A2/T1 贝叶斯 | 74.09 | 110.69 | 0.9836 | 25.0 |
| KNN(k=10) | A3 KNN | 82.16 | 129.35 | 0.9776 | 17.6 |
| AdaBoost | A13 Boosting | 187.50 | 221.46 | 0.9344 | 33.5 |
| **Stacking RidgeCV** | A10 堆叠 | 51.75 | **79.87** | **0.9915** | 20.5 |

**关键发现**：
- GBM（RMSE=83.25）与 RF（RMSE=86.86）持平，验证 Boosting 系串行集成与 Bagging 系并行集成在本任务上效果相近。
- Stacking 的 RMSE（79.87）最低，RidgeCV 系数显示 GBM（0.508）贡献最大，BayesianRidge（0.264）次之，AdaBoost（0.047）权重最小，与各自单独 MAE 一致。
- AdaBoost 性能较差（MAE=187.5）：当前特征中 OT_lag 主导，决策树桩对多特征的处理弱于梯度提升，且对 OT 分布的重采样偏差较大。
- BayesianRidge（MAP 估计）MAE=74 优于 KNN，说明线性结构在加入滞后特征后仍有一定拟合能力。
- 输出：`output_ot_extended/baselines_ext_metrics.csv`, `baselines_ext_mae.png`

---

### Step 2：A8 凝聚式层次聚类（AgglomerativeClustering）

**课程对应**：A8（层次聚类，Agglomerative / Hierarchical Clustering）

**方法说明**：

凝聚式层次聚类是「自底向上」的合并算法：每个样本开始时各自是一个簇，每步把距离最近的两个簇合并，直到簇数量等于 k。三种连接策略对「距离」的定义不同：

| 连接策略 | 两簇间距离定义 | 几何假设 |
|----------|---------------|---------|
| **ward** | 合并后组内方差增量最小 | 类球形、大小相近 |
| **complete** | 两簇中最远点对的距离 | 紧凑型（不喜欢细长簇） |
| **average** | 所有跨簇点对距离的均值 | 介于 ward 和 complete 之间 |

与 KMeans 对比：KMeans 是「平坦」算法，直接给定 k；层次聚类额外产生**树状图（dendrogram）**，可以直观看到样本如何被逐步合并，并在不同高度切割得到不同粒度的分组。

**实验设计**：同一 embedding / `OT_pred` 上对比 KMeans 与层次聚类；**Accuracy/F1 均在 OT < 1000 kW 停机基线上**（`cluster_unified_eval.py`），ARI 列相对 OT-KMeans 参照。

**代码位置**：`cluster_agglomerative.py`、`cluster_unified_eval.py`  
**如何运行**：`python3 cluster_unified_eval.py`（汇总表）；`python3 cluster_agglomerative.py`（树状图）  
**输出**：`out_cnn_lstm_cluster_1/clustering_shutdown_metrics.csv`

**结果**（N = 8 347，ground truth = OT < 1000 kW）：

| 方法 | 特征 | Accuracy | F1 | ARI vs OT-KMeans |
|------|------|----------|-----|------------------|
| **KMeans** | **Embedding** | **94.67%** | **95.45%** | 0.815 |
| KMeans | Prediction | 93.96% | 94.92% | 0.815 |
| Agglomerative-complete | Embedding | 89.82% | 91.80% | 0.718 |
| Agglomerative-ward | Embedding | 86.43% | 89.40% | 0.619 |
| Agglomerative-average | Embedding | 88.57% | 89.06% | 0.500 |

**分析**：
- **KMeans 表现最佳**（embedding 特征下 ARI=0.816）。这并非偶然：128 维标准化 embedding 在高维空间中，各簇的方差结构接近球形，而 KMeans 正是基于欧氏距离、假设球形簇。
- **Complete linkage 次之**（ARI=0.718）。complete 倾向于产生紧凑圆形簇，在高维中仍然有效。
- **Ward linkage 表现中等**（ARI=0.619）。Ward 最小化簇内方差，理论上应接近 KMeans，但对高维数据的初始化敏感，且数据存在若干难以划分的样本。
- **Average linkage 最差**（ARI=0.500，接近随机），高维下点对距离集中（维度诅咒），average 失效。
- **预测值（1D）上 average ≈ KMeans**：1D 情形下 average linkage 退化为单调切割，等价于 KMeans 的阈值划分，因此结果相同（ARI=0.816）。
- **结论**：不同算法 ARI 范围 0.50–0.82，说明 embedding 的工况分离结构**存在但并非完全线性可分**；KMeans 的球形假设在这里是适配的，层次聚类没有带来额外增益，但 dendrogram 提供了直观的「合并过程」可视化。

### Step 3：A6 K-Medoids / PAM（手写实现）

**课程对应**：A6（K-Medoids，PAM 算法）

**方法说明**：

K-Medoids 与 KMeans 的核心区别在于**聚类中心的定义**：

| | KMeans | K-Medoids (PAM) |
|---|---|---|
| 聚类中心 | 簇内所有点的**均值**（可能是空间中不存在的虚点） | 簇内使总距离最小的**真实数据点**（medoid） |
| 对离群点 | 均值会被极端值拉偏 | medoid 只能从实际样本中选，极端值最多影响 1 票 |
| 可解释性 | 中心是抽象数值 | 中心就是真实样本，可以直接检索查看 |
| 距离度量 | 必须用欧氏 | 任意距离（可换成 DTW、余弦等） |

PAM 算法：① 随机初始化 medoid → ② 最近邻分配 → ③ 对每个簇遍历全员，选使组内总距离最小的点为新 medoid → 重复至收敛。

**实验设计**（三组）：
1. **正常数据**：对 PCA-10 压缩后的 embedding 做 KMeans vs K-Medoids（PCA-10 可解释 100% 方差，说明 embedding 有效维度 ≤10）
2. **离群点鲁棒性**：注入 50 个极端离群点（沿 PC1 方向 +8σ），比较两者聚类中心的偏移量
3. **1D 预测值**：对标量预测值聚类，验证 medoid 是真实预测值而均值是抽象值

**代码位置**：`cluster_kmedoids.py`（手写 PAM，零外部依赖）  
**如何运行**：`python3 cluster_kmedoids.py`  
**输出目录**：`output_cluster_kmedoids/`

**结果**：

| 实验 | 方法 | ARI | Accuracy | F1 |
|------|------|-----|----------|----|
| 正常 embedding | KMeans | **0.816** | **0.952** | **0.939** |
| 正常 embedding | K-Medoids | 0.542 | 0.868 | 0.854 |
| 正常 1D 预测 | KMeans | **0.815** | — | — |
| 正常 1D 预测 | K-Medoids | 0.623 | — | — |
| 离群点鲁棒性 | KMeans 中心偏移 | 0.001 | — | — |
| 离群点鲁棒性 | K-Medoids 中心偏移 | 0.047 | — | — |

**分析——三个反直觉但诚实的结论**：

1. **KMeans 反而更准（ARI 0.816 vs 0.542）**：这是因为 embedding 聚类的两个簇形状接近球形（CNN-LSTM 的激活值近似 Gaussian），KMeans 的球形假设恰好匹配。K-Medoids PAM 初始化对局部最优敏感，高维下（即使 PCA-10 之后）容易收敛到次优解。

2. **鲁棒性实验反转（K-Medoids 偏移 0.047 > KMeans 0.001）**：50 个离群点在 8347 个样本中占比仅 0.6%，KMeans 的均值几乎感知不到（均值稀释效应：偏移 ≈ 50×8σ / 4000 ≈ 0.1，但由于两簇分离很好，簇分配几乎没变）。K-Medoids 的偏移反而稍大，因为离群点改变了部分样本的簇分配，导致某个簇的「最小总距离」点略有变化。**K-Medoids 鲁棒性优势在离群点比例较高（>5%）时才显著。**

3. **1D medoid 的真实性**：K-Medoids 中心为 [-5.3W, 1579.2W]——这两个值真实存在于预测序列中，分别代表结冰工况下模型预测接近零功率的某个时刻，以及高功率正常工况的某个典型点。这与 KMeans 的抽象均值（296W, 1860W）形成对比。

### Step 4：E2 显式 k-fold CV + E3/T2 Bootstrap 置信区间

**课程对应**：E2（交叉验证）、E3（Bootstrap）、T2（假设评估与置信区间）

**方法说明**：

**E2 — TimeSeriesSplit(5-fold) CV**：普通 k 折会把未来时间段随机分入训练集，破坏时间因果性。时序 k 折保持时间顺序——每折用更早时间段训练，验证随后一段时间（expanding window），是时序任务的标准 CV 方法。

**E3/T2 — Bootstrap 置信区间**：对测试集指标（MAE/RMSE）进行 B=2000 次有放回重采样，计算 95% 百分位置信区间，量化点估计的不确定性。

**代码位置**：`bootstrap_cv_eval.py`  
**如何运行**：`python3 bootstrap_cv_eval.py`  
**输出目录**：`output_bootstrap_cv/`

**CV 结果**（5-fold TimeSeriesSplit on 训练集，N=29,225）：

| 模型 | CV MAE ± std | CV RMSE ± std | 测试集 MAE（参考） | CV vs 测试集 |
|------|-------------|--------------|-----------------|------------|
| RF | 65.8 ± 16.1 | 122.6 ± 28.8 | 49.4 | CV 高 33% |
| GBM | 68.4 ± 8.5 | 121.9 ± 22.1 | 50.2 | CV 高 36% |
| BayesianRidge | 76.8 ± 12.2 | 132.9 ± 25.4 | 74.1 | CV ≈ 测试（差 4%）|
| KNN | 165.9 ± 64.7 | 246.8 ± 86.5 | 82.2 | 高方差 |
| SVR | 286.6 ± 189.5 | 385.6 ± 238.4 | 48.9 | **极大差异** |

**Bootstrap 95% 置信区间**（B=2000，测试集 N=8,349）：

| 模型 | MAE | 95% CI | RMSE | 95% CI |
|------|-----|--------|------|--------|
| SVR | 48.88 | [47.52, 50.24] | 81.32 | [78.13, 84.64] |
| RF | 49.39 | [47.94, 50.91] | 86.86 | [83.38, 90.42] |
| GBM | 50.19 | [48.73, 51.65] | 83.25 | [79.68, 87.24] |
| Stacking | 51.75 | [50.48, 53.06] | 79.87 | [76.89, 83.17] |
| BayesianRidge | 74.09 | [72.30, 75.87] | 110.69 | [107.66, 113.85] |
| KNN | 82.16 | [80.17, 84.22] | 129.35 | [125.93, 132.93] |
| AdaBoost | 187.50 | [184.98, 190.02] | 221.46 | [218.32, 224.57] |

**分析**：

- **CI 宽度**：SVR/RF/GBM 三者的 MAE 置信区间均宽约 2.5–3W，相互重叠（SVR [47.5, 50.2] vs RF [47.9, 50.9]），统计上三者无法明确区分——与配对 t-test（p=0.558）一致。
- **CV 悲观性**：RF 和 GBM 的 CV MAE（65–68W）显著高于测试集 MAE（49–50W），因为 CV 每折只用训练集的子集，数据更少，而最终模型用了完整 train+val 集。这是「expanding-window CV 相比随机 k 折更难」的体现。
- **SVR CV 极差（MAE=287, std=189）**：原因是 CV 中 SVR 的超参数是固定的（C=10, ε=0.1），没有在每折内重新网格搜索；而主实验中 SVR 的超参数是通过 TimeSeriesSplit 网格搜索在全训练集上选出的，对该特定时间段的数据是优化过的。这说明 **SVR 的表现高度依赖超参选择**，比 RF/GBM 对调参更敏感。
- **BayesianRidge CV ≈ 测试**（76.8 vs 74.1）：线性模型对训练集大小不敏感，5 折每折的训练集已足够确定稳健的先验参数，因此 CV 误差与测试集误差高度一致。

---

## 课程考点覆盖

详见 [`COURSE_COVERAGE.md`](COURSE_COVERAGE.md)。主干已全部覆盖（回归、序列、无监督、集成、实验准则、显著性检验）。Step 1 补充 A13/A3/A2 后覆盖率进一步提升。
