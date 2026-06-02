# 电机结冰聚类模型-实验报告

模型开发总共分为三个阶段：基础模型结构比较；meta-learner 混合基模型进行预测；结合CNN、LSTM、Transformer的模型特性将其结合为模型中不同的层，开发出能具有广泛适应性和泛化能力的基础模型。

数据： 

 [标注的数据-#67_1.xlsx](标注的数据-#67_1.xlsx) 



## 模型结构比较

`model.py`

**参与选取的模型结构如下：**

- RandomForest （随机森林法，Benchmark）
- SVR （向量机）
- CNN (学习局部特征)
- LSTM （序列模型）
- Transformer （Attention）

见`model.py`，结果存储在`output_ot_models` (不含val) 和 `output_ot_models-new` （含有val）。其中训练集: 验证集: 测试集 = 7: 1: 2。文件夹中含有预测效果展示和对应指标。

可以发现，SVR和RandomForest达到了可比的效果，而剩下的三种模型架构在实验效果上均达不到实用标准。考虑到CNN是捕捉局部特性的图模型，LSTM用于捕捉序列特征，而Transformer（即attention方法）在多种任务上具有泛化性，我考虑使用meta-learner尝试开发性能更好的模型。



## META-Learning

`ensemble.py`

**脚本主要步骤**：

1. 基于 **温度（exog_temp）和风速（exog_wind）** 做特征工程（rolling、lag）——**不使用电流/电压/功率**；
3. 时间感知切分：`train` / `val` / `test`，并在 `train` 内再切分 `train_train`（用于训练基础模型并产生 oof 风格 meta 数据）与 `meta_holdout`（用于训练 meta learner）；
4. 经典模型：RandomForest（baseline）和 SVR（在 `train_train` 上做 `TimeSeriesSplit` 网格搜索）；
5. 序列模型：CNN / LSTM / Transformer。先在 `train_train` 上训练并以 `meta_holdout` 做验证（早停/选择超参），选好的模型再用 `train+val+meta`（合并）重训为最终模型；
6. 在 `meta_holdout` 上用各基模型的预测构造 meta 特征，训练 `RidgeCV` meta learner（stacking）；
7. 将所有基模型在 `test` 上预测，meta learner 给出 stacking 预测；同时用 `nnls`（非负最小二乘）得到加权 ensemble 作为备用；
8. 抽取序列模型的 penultimate-layer embeddings（每个滑窗一个 embedding），并对 inputs 与 embeddings 做 PCA + KMeans 聚类并保存图/CSV；

------

### 经典模型（RF、SVR）

- **RandomForest (`rf_tt`)**：在 `train_train` 上训练（作为 baseline）。最终在 `train+val+meta` 重训练得到 `rf_final` 用于 test。
- **SVR 网格搜索**：
  - 在 `train_train` 上用 `TimeSeriesSplit(n_splits=4)` 做 `GridSearchCV`。
  - Pipeline 里含 scaling（`StandardScaler`）与 `SVR(kernel='rbf')`；这个 `Pipeline` 在每个 fold 内会在 train fold 上 fit scaler，避免未来信息泄露。
  - 参数网格： `C`、`epsilon`、`gamma`（可以扩展）。
  - 找到 best 后，会可选地在 `train+val+meta`上重训练以做最终预测。

------

### 序列模型（CNN/LSTM/Transformer）

- **数据输入**：把 `feature_cols_seq`（现在只包含 `exog_temp`,`exog_wind`）按滑窗生成 `(seq_len, feat)` 的输入，预测窗口之后的 OT。
- **SeqDataset**：接受 `start_idx,end_idx`（inclusive），`seq_len`，返回 `seq` 与 `y`。长度计算：`(end-start+1) - seq_len`。因此每个样本对应窗口 `[idx0: idx0+seq_len-1]`，label 是 `idx0+seq_len` 行的 OT（窗口最后之后的那一行）。
- **训练流程**：
  1. 在 `train_train_seq_loader`（train_train）上训练候选模型，`meta_seq_loader`（meta_holdout）上验证并早停（`early_stopping`）——这样可比较各超参的 meta performance；
  2. 选定最好的 CNN 配置（`best_cnn`），然后把 `train_train + meta_holdout + val`（合并成 `combined_seq_loader`）上重训练一个最终模型 `final_cnn`（以便在 test 上使用）；
  3. LSTM 和 Transformer 做类似流程：先用 train_train/select on meta_holdout，然后 retrain on combined。
- **返回 embedding**：模型的 `forward(..., return_embedding=True)` 可以返回 penultimate-layer embedding（脚本定义了这一点）。后面脚本会用这个 embedding 做 PCA+KMeans。

------

### Meta learner（Stacking）与 NNLS 加权

- **Meta（RidgeCV）**：
  - 在 `meta_holdout` 上，用 `pred_rf`, `pred_svr`, `pred_cnn`, `pred_lstm`, `pred_tr` 作为特征，`OT` 作为标签训练 `RidgeCV`（带 L2 正则）。结果保存为 `meta_learner_ridgecv.joblib`。
  - 在 test 上，用同样顺序的 base predictions 构建 `X_test_meta` 并预测 ensemble。
  - 优点：Ridge 能稳定训练并防止过拟合，特别是基模型数量不多时。
- **NNLS（非负最小二乘）**：
  - 把 meta_holdout 的特征列按列标准差缩放（避免尺度主导），然后用 `scipy.optimize.nnls` 估计非负权重 `w`（使得 `X_scaled @ w ≈ y`）。
  - 将权重归一化（和为 1）并在 test 上应用（同样缩放）。
  - 这是一个带“正权约束”的简单加权 ensemble，比直接最小二乘更鲁棒（不至于出现巨大的负/正权重）。
  - RidgeCV 提供训练好的线性组合（可能含负权），但 NNLS 强制非负且可解释（权重代表每个基模型贡献），可作为 sanity-check。

------

### embeddings 抽取与聚类

- `extract_embeddings(model, df_seq_scaled, feat_cols, seq_len)`：
  - 在整个数据集上按滑窗提取 embedding（每个窗口对应一个 embedding，时间戳为窗口的最后时间点）；
  - 将 embedding 存为 CSV（`embeddings_cnn.csv`、`embeddings_lstm.csv`、`embeddings_transformer.csv`）。
- 对 embeddings（和输入的 temp/wind）分别做：
  - `StandardScaler` → `PCA(n_components=2)` → `KMeans(n_clusters=2)`；
  - 保存 PCA 坐标 CSV 与聚类图（PNG）。

------

### 输出文件（都保存在 `OUT_DIR`）

主要输出（文件名）及含义：

- `wind_model_output_with_OT_predictions_and_ensembles.csv`：test 行（time、原始 exog、OT）以及每个基模型预测列 `pred_rf`、`pred_svr`、`pred_cnn`、`pred_lstm`、`pred_tr`、`OT_pred_Ensemble_meta`（stacking）和 `OT_pred_Ensemble_nnls`（NNLS）。
- `final_metrics.csv` 或 `models_metrics_table.csv`：每个模型在 test 上的 MAE、MSE、RMSE、MAPE。
- `svr_grid_results.csv` / `svr_grid_results_full.csv`：SVR 网格搜索的 CV 结果（参数组合和得分）。
- `rf_final.joblib`, `svr_final.joblib`：已训练好的最终 classical 模型。
- `cnn_final.pt`, `lstm_final.pt`, `transformer_final.pt`：最终 NN 模型的权重（PyTorch state dict）。
- `embeddings_cnn.csv`, `embeddings_lstm.csv`, `embeddings_transformer.csv`：penultimate embeddings（含 time 列）。
- `pca_inputs_cluster.png`, `cnn_embed_pca_cluster.png` 等一系列图片（PCA 聚类、预测对比、残差直方图、模型对比 MAE）。
- `ensemble_weights_nnls.txt`：NNLS 得到的权重向量。

![OT_ensemble_compare](/Users/hanjiaxin/Desktop/电机系研究/ecotrans-main/output_ot_ensembles/OT_ensemble_compare.png)

![comparison_mae](/Users/hanjiaxin/Desktop/电机系研究/ecotrans-main/output_ot_ensembles/comparison_mae.png)

CNN、LSTM、Transformer分别提取了数据中不同维度的信息，结合meta-learner的思想，将它们作为模型中的不同 layer，combine出一个mix版本的新模型



## CNN-LSTM-ATTn

`mix.py`

- CNN → LSTM 负责短期和中期时序建模。
- Attention 负责长距离依赖关系。

------

**输入（温度、风速序列） → CNN → LSTM → Attention → Dense 输出**

```
1. CNN层
   - 作用：提取短期的局部变化模式（如风速短时变化的波形）。
   - 相当于在时间序列上做一个滑动窗口的特征抽取。
   - 核心参数：kernel size（3~5）。

2. LSTM层
   - 作用：建模时间依赖关系（短期到中期趋势）。
   - LSTM 在风速和温度的周期变化（昼夜、天气系统）中表现很好。

3. Attention层（Transformer风格的 Multi-Head Attention）
   - 作用：捕捉全局依赖（长时间间隔的影响）。
   - 例如：前一段时间的温度变化可能影响今天的风机效率。

4. Dense 全连接层
   - 作用：整合前面提取的特征并输出功率预测值。
```

------

## **示意结构**

```
Input (T, 2)
   ↓
1D CNN (filters=64, kernel=3, activation=ReLU)
   ↓
LSTM (units=128, return_sequences=True)
   ↓
Multi-Head Attention (heads=4)
   ↓
Flatten
   ↓
Dense(64, activation=ReLU)
   ↓
Dense(1)  →  OT预测
```

预测效果不够理想——NEW 变化

**Dropout 加在 CNN、LSTM、Attention 层**，减少过拟合。

**SmoothL1Loss** 替代 MSE，鲁棒性更好。

**ReduceLROnPlateau** 自动调节学习率。

且，考虑到结冰与非结冰时风机效率不同，新建一个“**聚类+预测**”工作流。先根据训练集建立聚类模型，分别训练“**fault和non-fault**”时的模型，并进行参数搜索，保存最佳模型；再将测试集聚类后选择合适的模型进行预测。

模型的参数搜索见 `param_search.py`。








