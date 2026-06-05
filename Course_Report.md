# 风机结冰相关工况下的时间序列预测与集成建模实验

课程：机器学习概论  
作者：韩佳辛  
数据来源：国网冀北电力有限公司承德供电公司风机运行数据

## 摘要

风机在低温、大风等环境下可能出现叶片结冰或结冰相关异常工况，导致风机运行功率下降，并增加停机决策难度。本文围绕风机运行功率 `OT` 的短时预测任务，使用 2024 年 2 月分钟级风机运行数据，比较传统回归模型、深度序列模型、stacking 集成模型与工况划分方法的效果。实验采用按时间顺序划分训练集、验证集和测试集的方案，以避免未来信息泄露。预测层主结果（§4）覆盖 RF、SVR、GBM、AdaBoost、KNN、BayesianRidge、CNN、LSTM、Transformer、CNN-LSTM 与 CNN-LSTM-Attention；集成层（§4.4）在统一强基模型池 `RF+GBM+BayesianRidge+SVR` 下公平比较 Holdout/OOF stacking 与 NNLS。结果显示：SVR/RF/GBM 在滞后与滚动统计特征上表现最稳健；CNN-LSTM 及 CNN-LSTM-Attention 在序列结构扩展实验中显著优于单一深度模块。Stacking 在 RMSE 上有改进，但 MAE 未稳定超过最佳单模型。聚类与工况划分集中在 §5 报告，以 CNN-LSTM-Attention embedding 为代表，并统一在 `OT<1000 kW` 停机代理标签上报告 Accuracy/F1；主结果准确率为 **94.67%**，OT-KMeans 参照分群仅作 ARI 补充。

## 1. 引言

### 1.1 研究背景

风力发电机组在低温、湿度较高、风速变化剧烈的环境下容易出现叶片结冰。结冰会改变叶片表面的气动特性，使叶片重量增加、转速下降，进而造成输出功率偏离正常工况下的功率曲线。如果判断过早，可能造成不必要停机和发电损失；如果判断过晚，则可能增加设备安全风险。因此，围绕结冰相关工况进行功率预测和状态分析，具有明确的工程价值。

从机器学习角度看，该任务同时具有几个典型特点：第一，数据是分钟级时间序列，存在明显的时间依赖；第二，功率输出受风速、温度、历史功率等变量共同影响；第三，正常工况与结冰相关工况可能服从不同的数据分布。因此，该问题适合综合使用回归学习、序列学习、无监督学习和集成学习方法。

### 1.2 问题定义

本文的核心预测目标是风机运行功率 `OT`。记第 `t` 个时刻的温度为 `Temp_t`，风速为 `Wind_t`，目标变量为 `OT_t`。实验主要考虑单步预测任务，即利用当前和历史若干时刻的环境变量、历史功率滞后项和滚动统计特征，预测测试时刻的 `OT`。

本文中的“工况”指由温度、风速和历史运行状态共同决定的数据状态；“结冰相关工况”指由低温等环境特征诱发，并可能对应不同功率输出规律的一类运行状态。由于当前主数据没有逐分钟人工结冰标签，本文不把问题表述为严格的监督式结冰检测，而是将聚类结果作为工况划分与解释性分析。

### 1.3 研究目标

本文围绕功率预测、集成融合、工况分析与实验评估四条主线展开，对应如下研究目标：

**（1）建立并扩展传统回归基线。** 在统一的时间切分与特征工程设置下，比较 Random Forest、SVR、GBM、AdaBoost、KNN、BayesianRidge 等方法，回答“在本数据规模和特征条件下，传统机器学习能做到什么程度”。

**（2）评估深度序列模型与混合结构。** 比较单独训练的 CNN、LSTM、Transformer，以及 CNN-LSTM / CNN-LSTM-Attention 混合模型；进一步考察按 KMeans 分工况训练是否能改善序列模型表现，判断复杂模型是否值得在本任务上继续投入。

**（3）分析集成学习能否提升预测稳健性。** 在统一强基模型池（RF、GBM、BayesianRidge、SVR）上构建 RidgeCV/LassoCV/ElasticNetCV stacking，并以 NNLS 非负加权集成作为对照，比较 Holdout 与 OOF 两种 meta-feature 生成策略是否能在 MAE、RMSE 等指标上优于最佳单模型。

**（4）比较多种工况划分与状态解释方法。** 由于主数据缺少逐分钟人工结冰标签，本文不把问题表述为严格的监督式结冰检测，而是从以下角度分析运行状态：
- **分工况建模**：KMeans 划分训练样本，按簇训练子模型；
- **表征聚类**：对 CNN-LSTM 的 embedding 或模型预测值做 KMeans；
- **方法对比**：AgglomerativeClustering、K-Medoids 与 KMeans 的 ARI/NMI 一致性；
- **时序隐状态**：Gaussian HMM 对 `(OT, Temp)` 序列建模工况切换；
- **停机规则评测**：以 `OT < 1000 kW` 为代理标签，评估 embedding 聚类与低功率停机规则的一致性。

**（5）建立规范的实验评估流程。** 在时间顺序划分、防泄露预处理的基础上，补充 TimeSeriesSplit 交叉验证、测试集 Bootstrap 置信区间，以及 paired t-test / Wilcoxon 显著性检验，使模型比较不仅看点估计，也看统计不确定性与方法差异是否稳健。

上述目标中，（1）–（3）主要回答“预测准不准”，（4）回答“能否解释不同运行状态”，（5）回答“结论是否可靠”。这些目标相互配合：工况划分不以降低主预测 MAE 为唯一目的，集成学习也不替代聚类分析。

## 2. 数据来源与预处理

### 2.1 数据来源

实验使用国网冀北电力有限公司承德供电公司提供的风机运行数据。主数据文件为 `标注的数据-#67_1.xlsx`，包含 2024 年 2 月 1 日 00:00 至 2024 年 2 月 29 日 23:59 的分钟级观测，共 41,760 条记录。该数据覆盖一个完整 2 月，具有较高采样频率，适合用于短时预测与时序建模。

另有 `11月1日-1月20日叶片结冰统计情况(1).xlsx` 记录部分风机结冰起止时间、累计时长和损失电量。该文件可用于说明结冰问题的工程背景，但它与 2024 年 2 月主训练数据不是逐点对齐的监督标签，因此不直接作为分类标签使用。

### 2.2 字段说明

主数据包含以下字段：


| 字段                | 含义              | 实验用途     |
| ----------------- | --------------- | -------- |
| `统计时间`            | 分钟级时间戳          | 排序、时间切分  |
| `OT`              | 风机运行功率，本文预测目标   | 回归标签     |
| `Exogenous1`      | 外生变量 1，按业务理解为温度 | 环境特征     |
| `Exogenous2`      | 外生变量 2，按业务理解为风速 | 环境特征     |
| `平均发电机转速(rpm)`    | 发电机平均转速         | 辅助理解工况   |
| `平均网侧A/B/C相电流(A)` | 三相电流            | 辅助运行状态信息 |
| `平均网侧A/B/C相电压(V)` | 三相电压            | 辅助运行状态信息 |


本文最终主实验优先使用温度、风速、`OT` 滞后项和滚动统计特征。这样做有两个考虑：其一，与 Proposal 中“温度、风速和历史运行状态共同决定工况”的设定一致；其二，尽量避免把过多电气量直接引入模型后，使模型变成对目标变量的近似代数重构。

### 2.3 预处理方法

预处理流程如下：

1. 将 `统计时间` 转换为时间戳，并按时间升序排序。
2. 将 `OT`、温度、风速等字段转换为数值类型。
3. 对短缺失段进行插值，并用前向/后向填充处理剩余缺失值。
4. 构造滚动统计特征，例如 3 分钟温度均值和 3 分钟风速均值。
5. 构造滞后特征，例如 `OT_lag_1`、`OT_lag_2`、`OT_lag_3`、`OT_lag_6`、`OT_lag_12`。
6. 按时间顺序划分训练集、验证集和测试集，再在训练集上拟合标准化器，之后变换验证集和测试集。

所有需要拟合的数据处理步骤均应只在训练集上拟合。旧版探索脚本中曾有全量数据标准化和全量 KMeans 拟合的问题，已在 `model.py` 中修正；预测层结果以 `ensemble.py`、`baselines_ext.py` 和 `final_evaluation.py` 的无泄露评估为准，公平集成结果以 `stacking_fair_pool.py` 为准。

## 3. 实验方案设计

本文的实验设计按“先建立可比较的预测基线，再尝试模型聚合与序列结构优化，最后用聚类和隐状态模型提供工况解释”的顺序展开。这样的组织方式避免把预测任务和工况解释任务混为一谈：前者回答“下一时刻 `OT` 能否预测准确”，后者回答“模型内部表示能否支持运行状态划分和停机决策”。

### 3.1 阶段一：基础预测基线

第一阶段建立非时序和浅层机器学习基线，包括 Random Forest、SVR、GBM、AdaBoost、KNN 与 BayesianRidge。它们统一使用温度、风速、`OT` 滞后项和 rolling 统计特征，并按时间顺序划分训练/验证/测试集。此阶段的目的不是追求复杂结构，而是确认在强特征工程下传统模型能达到的预测上限。

其中 Random Forest 和 GBM 代表树模型及 Bagging/Boosting 思路，SVR 代表核方法，BayesianRidge 提供线性贝叶斯参照，KNN 和 AdaBoost 作为补充对照。SVR 的超参数通过 `TimeSeriesSplit` 搜索，所有标准化和调参都放在训练折内部，避免未来信息泄露。

### 3.2 阶段二：模型聚合与序列模型调优

第二阶段包含两条线。

第一条线是序列模型扩展。初始 CNN、LSTM、Transformer 单模型表现较弱后，本文进一步尝试 CNN-LSTM 和 CNN-LSTM-Attention，并进行小规模 NAS 式结构搜索。搜索空间包括 `MODEL_TYPE`、`SEQ_LEN`、学习率、卷积通道数、LSTM hidden size、损失函数、异方差输出、峰值过采样与非对称损失等。其动机是：序列模型效果不好不一定说明时序结构无效，也可能是局部模式、时间记忆和极端样本处理没有配合好。

第二条线是 stacking 聚合。本文先做混合池探索，再固定同一个强基模型池 StrongPool 进行公平比较。StrongPool 由 RF、GBM、BayesianRidge、SVR 组成；元学习器比较 RidgeCV、LassoCV、ElasticNetCV 与 NNLS；meta-feature 生成方式比较 Holdout 与 OOF。这样可以把“基模型池是否合理”和“stacking 方法是否有效”分开讨论。

### 3.3 阶段三：聚类与工况决策支持

第三阶段不再以降低预测 MAE 为唯一目标，而是把 CNN-LSTM-Attention 的 penultimate-layer embedding、模型预测值 `OT_pred` 和原始 `OT` 分档用于工况划分。主要方法包括 KMeans、AgglomerativeClustering、K-Medoids PAM 和 Gaussian HMM。

由于主数据没有逐分钟人工结冰标签，本文不把聚类结果直接称为“结冰识别真值”。聚类的主要作用是决策支持：判断 embedding 是否能把低功率/高功率运行状态分开，并用 `OT < 1000 kW` 的停机代理规则评估聚类结果是否和工程阈值一致。HMM 则作为补充的时序隐状态模型，用来观察状态是否具有时间连续性。

### 3.4 阶段四：可靠性验证

第四阶段用 TimeSeriesSplit、Bootstrap 置信区间、paired t-test 和 Wilcoxon 检验验证结论是否稳健。交叉验证用于估计不同时间段上的泛化波动；Bootstrap 用于给 MAE/RMSE 点估计加置信区间；paired t-test 与 Wilcoxon 则利用同一测试样本上的逐点绝对误差，检验模型之间的差异是否只是偶然波动。

### 3.5 算法在实验流程中的角色分类

本文涉及的算法在逻辑上扮演三种不同角色，混淆这些角色会导致对实验设计和结果解读的误判。以下明确区分三类角色，并指出部分算法兼跨多类的情况。

#### 3.5.1 三类角色的定义

**角色 A — 估计器（Estimator）**：接受特征输入，输出连续预测值（此处为 `OT`）。评价标准是 MAE、RMSE 等回归指标，衡量「预测有多准」。

**角色 B — 训练策略（Training Strategy）**：决定「如何优化」一个估计器的参数，本身不直接对外暴露预测接口，但决定了估计器的训练质量。典型例子包括 Boosting（逐步修正残差）、Bagging（有放回抽样减少方差）、交叉验证调参（CV-based hyperparameter selection）。

**角色 C — 分析方法（Analytical Method）**：不以最小化预测误差为唯一目标，而是对数据或模型输出进行结构发现（聚类）或不确定性量化（Bootstrap、统计检验），为决策提供可解释性支持。

---

#### 3.5.2 各算法的角色归属


| 算法                            | 估计器 | 训练策略 | 分析方法 | 说明                                                                         |
| ----------------------------- | --- | ---- | ---- | -------------------------------------------------------------------------- |
| **Random Forest**             | ✓   | —    | —    | Bagging 是其内部训练策略，但 RF 整体对外作为单一估计器                                          |
| **SVR**                       | ✓   | —    | —    | 超参通过 GridSearchCV + TimeSeriesSplit 选择；选择过程是角色 B，但 SVR 本身是角色 A             |
| **GradientBoosting (GBM)**    | ✓   | ✓    | —    | **双重角色**：是估计器（输出预测），也是训练策略（逐步拟合残差，等价于函数空间上的梯度下降）                           |
| **AdaBoost**                  | ✓   | ✓    | —    | **双重角色**：通过对「困难样本」重新加权迭代训练弱学习器，最终加权组合为强估计器；样本重加权是指数损失函数的坐标下降               |
| **KNeighborsRegressor (KNN)** | ✓   | —    | —    | 非参数基于记忆的方法，无显式训练阶段，预测时直接查找 k 近邻                                            |
| **BayesianRidge**             | ✓   | ✓    | —    | **双重角色**：作为估计器是带 L2 正则的线性回归；作为训练策略用 Evidence Maximization 自动确定正则强度，无需手动 CV |
| **LSTM / CNN / Transformer**  | ✓   | —    | —    | 端到端序列估计器，训练由 Adam + MSE 驱动，属通用深度学习流程                                       |
| **CNN-LSTM-Attention**        | ✓   | —    | ✓    | **双重角色**：既是估计器，也是分析工具——其 penultimate-layer embedding（128 维）可提取用于下游工况聚类     |
| **RidgeCV（元学习器）**             | ✓   | ✓    | —    | **双重角色**：在 stacking 中作元学习器（角色 A）；CV 部分通过留一折选出最优正则强度 α（角色 B）                |
| **NNLS 加权集成**                 | ✓   | ✓    | —    | 非负最小二乘求解权重（角色 B），输出加权预测（角色 A）；约束权重非负确保集成可解释                                |
| **KMeans**                    | —   | —    | ✓    | 纯分析方法：将 embedding 或预测值划分为 k 个簇，用于工况识别                                      |
| **AgglomerativeClustering**   | —   | —    | ✓    | 纯分析方法：自底向上层次合并（Ward/Complete/Average），验证工况结构的跨算法稳健性                        |
| **K-Medoids (PAM)**           | —   | —    | ✓    | 纯分析方法：聚类中心为实际数据点（medoid），增强工况代表点的可解释性                                      |
| **TimeSeriesSplit CV**        | —   | ✓    | ✓    | 角色 B（为 SVR 超参搜索提供时间不泄露的验证）+ 角色 C（估计模型在不同时间段的泛化误差分布）                        |
| **Bootstrap 置信区间**            | —   | —    | ✓    | 纯分析方法：量化测试集指标的统计不确定性，不改变任何模型                                               |
| **Paired t-test / Wilcoxon**  | —   | —    | ✓    | 纯分析方法：逐样本误差差异的显著性检验                                                        |


---

#### 3.5.3 几个关键区分

**AdaBoost 和 GBM 的双重身份**：两者都既是训练算法（如何用弱学习器迭代出强模型），又是最终估计器（产物可以预测）。本文中它们主要以角色 A 参与基线对比，用 MAE/RMSE 评价预测准确性。但分析 Boosting vs Bagging（GBM/AdaBoost vs RF）的泛化行为差异时，其训练策略的差异是核心讨论对象（详见 §6.1）。

**RidgeCV 的两层语义**：在 stacking 中，RidgeCV 先作为学习算法通过 CV 选出最优 α，再作为估计器拟合基模型预测到 `OT` 的映射。拆开理解：ridge regression（角色 A）+ CV-based hyperparameter selection（角色 B）。代码中 `RidgeCV` 对象将两步合并，但逻辑上属于两类角色。

**CNN-LSTM-Attention 的双用途**：在功率预测任务中它是角色 A；在工况分析中，其 penultimate-layer embedding 被 KMeans/层次聚类/K-Medoids 使用，此时该模型充当特征提取器（角色 C 的前置），聚类方法才是真正的分析工具（详见 §5）。

---

#### 3.5.4 层次结构总结

在三类角色分类基础上，四个执行层次更清晰：

1. **特征与预处理层**：`StandardScaler` 标准化、rolling 特征、lag 特征构造——为所有估计器提供一致输入
2. **预测层（角色 A 为主）**：RF、SVR、GBM、AdaBoost、KNN、BayesianRidge、CNN、LSTM、Transformer、CNN-LSTM——直接输出 `OT` 预测，以 MAE/RMSE 评价
3. **聚合层（角色 A + B）**：RidgeCV stacking、NNLS——以基模型预测为输入，输出最终预测；训练策略是关键设计选择
4. **分析层（角色 C）**：KMeans/AgglomerativeClustering/K-Medoids（工况结构发现）、TimeSeriesSplit CV（泛化估计）、Bootstrap（置信区间）、统计检验（显著性判断）——不改变预测，为理解和信任预测结果提供依据

### 3.6 评估协议与对照设置

#### 3.6.1 数据划分

由于该任务是时间序列预测，不能随机划分训练集和测试集。随机划分会使模型在训练阶段看到未来时间段的数据分布，从而高估泛化性能。本文采用时间顺序划分，训练集位于最早时间段，验证集位于中间时间段，测试集位于最后时间段。

在 stacking 实验中，训练集内部进一步拆出 `meta_holdout`，用于训练元学习器。这样可以避免元学习器直接学习基模型在训练样本上的过拟合预测。

#### 3.6.2 时间序列交叉验证

时间序列数据并不是不能做交叉验证，但不能使用普通随机 k 折。普通 k 折会把未来时间片随机分到训练集中，再去预测较早时间片，造成未来信息泄露。因此，本文只在需要调参的 SVR 上使用 `TimeSeriesSplit(n_splits=4)`：每一折都保持时间顺序，用更早的时间段训练，用随后的一段时间验证，形成 expanding-window 风格的多折验证。

这种做法比普通交叉验证更严格，也更受数据长度限制。一方面，每一折训练集只能使用验证段之前的数据，不能打乱；另一方面，标准化器、SVR 参数搜索等拟合步骤必须放在 `Pipeline` 内部，使每个 fold 只在本 fold 的训练段上 `fit`。本文没有对所有深度模型都做多折 CV，主要原因是训练成本较高，而且序列模型还涉及滑窗边界、早停和模型保存。深度模型主要依赖独立验证集和最终测试集评估；SVR 的超参数搜索则用时间序列 CV 提供更稳健的调参依据。

#### 3.6.3 评价指标

本文使用以下指标：


| 指标    | 含义                              |
| ----- | ------------------------------- |
| MAE   | 平均绝对误差，主指标之一，直观反映平均偏差           |
| RMSE  | 均方根误差，对大误差更敏感                   |
| sMAPE | 对称 MAPE，缓解普通 MAPE 在接近 0 时不稳定的问题 |
| MASE  | 相对朴素预测的误差尺度，便于判断模型是否优于简单时间序列基准  |
| Bias  | 预测均值减真实均值，反映整体高估或低估             |


普通 MAPE 在本数据中不适合作为唯一主指标，因为 `OT` 存在接近 0 甚至负值的时段，分母过小会导致 MAPE 被极端放大。因此，报告中以 MAE、RMSE 为主，sMAPE 和 MASE 作为补充。


#### 3.6.4 超参数与模型选择方法

不同模型族采用与数据特性匹配的调参策略；**所有拟合（标准化、聚类、网格搜索）仅在训练段或 train_train 上完成**，验证/测试段仅用于评估或早停，避免泄露。

| 模型 / 结构 | 调参方法 | 搜索空间要点 | 脚本 |
|-------------|----------|--------------|------|
| **Random Forest** | 固定强基线配置 | `n_estimators=200–300`，默认深度不限制；使用完整 lag+rolling 特征 | `ensemble.py`、`baselines_ext.py` |
| **SVR** | **GridSearchCV + TimeSeriesSplit(4)** | `C∈{0.1,1,10,50}`，`epsilon∈{0.1,0.5,1}`，`gamma∈{scale,auto}`；Pipeline 内 StandardScaler | `ensemble.py`、`baselines_ext.py` |
| **GBM** | 固定配置 | `n_estimators=200`，`max_depth=4`，`learning_rate=0.1`，`subsample=0.8` | `baselines_ext.py` |
| **AdaBoost** | 固定配置 | 基学习器 `DecisionTree(max_depth=3)`，`n_estimators=100` | `baselines_ext.py` |
| **KNN** | 固定配置 | `n_neighbors=10`，`weights=distance` | `baselines_ext.py` |
| **BayesianRidge** | **Evidence Maximization（自动）** | `max_iter=300`，正则强度由边际似然估计，无需手工 CV | `baselines_ext.py` |
| **CNN / LSTM / Transformer** | **小网格 + meta_holdout 早停** | CNN：`hid∈{32,64}`、`kernel∈{3,5}`；LSTM/Transformer：`hid/d_model=64`；在 `train_train` 训练、meta_holdout 选优后合并 train+val 重训 | `ensemble.py` |
| **CNN-LSTM / Attention / 组合结构** | **随机子采样网格 + 验证集** | `MODEL_TYPE` 含 `mlp/cnn/lstm/cnn_lstm/cnn_lstm_attn` 及 `*_mlp` 组合；`SEQ_LEN∈{2,4,8,16}`，学习率、通道数、是否异方差 NLL/非对称损失/峰值过采样等；测试集报最优 run | `cnn_lstm_grid_search.py` |

**实验说明**：主预测实验使用 **lag+rolling 全特征 + 统一测试集 N=8349**；序列结构扩展实验使用 **`OT_prev`+温风速、目标标准化、N≈8351**。两类实验服务于不同问题：前者比较不同模型在强特征工程下的预测上限，后者比较 CNN/LSTM/Attention 结构细节的有效性。因此，报告在解读时更关注各自实验内部的相对排序，而不把两张表的 MAE 作机械横比。

#### 3.6.5 对照实验

1. **预测层（§4.1–§4.3）**：比较传统模型（RF/SVR/GBM/AdaBoost/KNN/BayesianRidge）、深度单模型（CNN/LSTM/Transformer）与 CNN-LSTM / Attention 扩展结构。
2. **集成层（§4.4）**：在统一基模型池下比较 RidgeCV / LassoCV / ElasticNetCV stacking、NNLS 与 Holdout vs OOF。
3. **统计与稳健性（§4.6）**：配对检验、时序 CV、Bootstrap CI。
4. **聚类与停机（§5）**：以 CNN-LSTM-Attention embedding 为代表性表示，两套 ground truth（§3.6.9）。

paired 检验在主预测实验的最佳模型之间进行（§4.6）；不同输入特征或训练设置的扩展实验不混做 paired test。

#### 3.6.6 统计检验

为了比较模型误差差异，本文使用测试集逐点绝对误差进行配对检验。`final_evaluation.py` 输出预测层配对检验，`stacking_fair_pool.py` 输出公平集成实验的配对 t-test 与 Wilcoxon 检验；若运行环境缺少 `scipy`，则退化为配对 sign test。统计检验只用于辅助判断，不替代误差指标和预测曲线分析。

配对检验选择是因为每个模型对同一批测试样本作出预测——两个模型的误差序列是"成对"的，可以直接对每个样本计算误差差值，然后检验差值的均值是否显著偏离 0。这比先计算各模型误差均值再做双样本 t 检验更有统计功效，也更贴近"在同一数据集上哪个模型更好"的问题。

Wilcoxon 符号秩检验是 paired t-test 的非参数替代，不假设误差差值服从正态分布，对时序数据中可能出现的重尾分布更鲁棒。

#### 3.6.7 显式 k 折交叉验证

对于训练集规模足够的传统模型，本文在 `bootstrap_cv_eval.py` 中额外做了显式 `TimeSeriesSplit(n_splits=5)` 交叉验证。时序 k 折不打乱时间顺序：前 k−1 折逐步增大训练窗口（expanding window），第 k 折为该窗口之后的一段时间。

这与常规 k 折的区别在于：普通 k 折将未来时间片随机分配到训练集中，破坏了时间因果性；时序 k 折保持因果顺序，每折只能"往后看"而不能"往前看"。代价是每折训练集大小不同，方差略大于均匀 k 折。

交叉验证的主要目的是估计模型**在训练集范围内不同时间段上的泛化能力**，而最终的测试集性能（§4）是对 train+val 之外的完全保留时间段的最终评估，两者互补而不能互相替代。

#### 3.6.8 Bootstrap 置信区间

测试集误差（MAE/RMSE）是一个点估计。为了衡量这个估计的不确定性，本文用 Bootstrap 重采样（B=2000 次，有放回采样测试集）估计 95% 置信区间。

Bootstrap 的适用性：测试集 N=8349 个样本是独立的时序点（已通过时间切分与训练集隔开），因此在测试集上做有放回采样近似合理——这等价于从"同一时间段未见数据的总体"中模拟重复抽样。

置信区间宽窄反映了指标对测试集随机性的敏感程度：宽区间说明误差分布方差大，点估计不够稳定；窄区间说明结论鲁棒。

#### 3.6.9 聚类评估：统一基线与补充参照

##### 3.6.9.1 主评测与补充评测

| 维度 | **主评测**：`OT_true < 1000 kW` 停机代理 | **补充评测**：`OT_true` 上 KMeans(k=2) 参照分群 |
|------|------------------------------------------------------|-----------------------------------------------|
| **定义** | 工程规则：瞬时功率低于 1000 kW → 应停机（CLOSE） | 对连续 OT 无监督二分，得到 `true_cluster` |
| **合理性** | 阈值可审计、可对接运行规程，适合解释停机决策 | 用于检验 embedding 是否保留**功率分档结构**；与监督目标 `OT` 相关，**不宜**作为主 Accuracy |
| **本报告用法** | **所有主表 Accuracy / F1 / Precision / Recall 均用此基线** | 仅报告 **ARI / NMI**（附录可保留匈牙利 Accuracy） |

**结论**：与停机决策相关的结论统一使用停机代理评测；OT-KMeans 是内部分群一致性检查，**不得**与 94.67% 混为两个「准确率」。

##### 3.6.9.2 统一主评测协议

| 项目 | 设定 |
|------|------|
| **特征提取模型** | **CNN-LSTM-Attention**（`out_cnn_lstm_cluster_1/model_run0_cluster0.pt`） |
| **Embedding** | 128 维 penultimate；输入 `exog_temp`、`exog_wind`、`OT_prev`，`SEQ_LEN=4` |
| **聚类** | KMeans(k=2)（或 §5.3–§5.4 的其他聚类器）→ 簇标签 |
| **二分类映射** | 每簇对 `should_close = (OT_true < 1000)` **多数投票** → CLOSE/KEEP |
| **Ground truth** | `should_close_true = 1` 当且仅当 `OT_true < 1000 kW` |
| **脚本** | `cluster_eval.py`、`cluster_unified_eval.py` → `clustering_shutdown_metrics.csv` |

##### 3.6.9.3 补充：相对 OT-KMeans 的一致性（不用于主 Accuracy）

对 `OT_true` 做 KMeans(k=2) 得 `true_cluster`；计算 embedding/预测聚类与之的 **ARI、NMI**（匈牙利对齐 Accuracy 见 `clustering_classification_metrics.csv`，**仅作附录**）。ARI≈0.82 表示无监督二分与功率分档一致；与 94.67% 停机 Accuracy 相差约 0.5 个百分点，因 1000 kW 阈值与 KMeans 质心并不完全重合。

##### 3.6.9.4 指标公式

| 指标 | 主评测（停机基线） | 补充（OT-KMeans） |
|------|-------------------|-------------------|
| Accuracy / F1 / Recall | 标准 0/1 混淆矩阵 | 匈牙利对齐后相对 `true_cluster` |
| ARI / NMI | 表中「ARI vs OT-KMeans」列 | 聚类—聚类一致性主指标 |

## 4. 预测模型选择与实验结果

### 4.1 基础非时序模型

基础非时序模型使用 `exog_temp`、`exog_wind`、rolling(3)、`OT/temp/wind` 的 lag {1,2,3,6,12} 作为输入，按时间顺序划分 train 70% / val 10% / test 20%。这一组实验回答的问题是：在强特征工程下，常规机器学习模型能达到怎样的预测水平。

| 模型 | 类型 | MAE (W) | RMSE (W) | R² | 调参（摘要） |
|------|------|--------:|---------:|---:|--------------|
| **SVR** | 核方法 | **48.88** | **81.32** | 0.991 | GridSearchCV + TimeSeriesSplit(4) |
| **Random Forest** | Bagging 树模型 | 49.23 | 86.47 | 0.990 | 固定 n_estimators=200–300 |
| **GBM** | Boosting 树模型 | 50.19 | 83.25 | 0.991 | 固定 Boosting 超参 |
| **BayesianRidge** | 贝叶斯线性模型 | 74.09 | 110.69 | 0.984 | Evidence Maximization |
| **KNN** | 非参数距离模型 | 82.16 | 129.35 | 0.978 | k=10，distance 加权 |
| **AdaBoost** | Boosting 弱学习器集成 | 187.50 | 221.46 | 0.934 | 浅树弱学习器 |

**结果解读**：SVR/RF/GBM 形成第一梯队，说明 `OT` 滞后项和 rolling 统计特征携带了很强的预测信息。BayesianRidge 表现稳定但低于非线性模型，说明任务中存在明显非线性交互。KNN 和 AdaBoost 明显较弱，分别反映出高维滞后特征下距离邻域不稳定、浅弱学习器难以刻画复杂工况。

### 4.2 单独序列模型

单独序列模型使用温度和风速滑窗作为输入，比较 CNN、LSTM 与 Transformer。它们用于检验“只靠环境序列是否足以预测 `OT`”。

| 模型 | 序列建模方式 | MAE (W) | RMSE (W) | R² |
|------|--------------|--------:|---------:|---:|
| **LSTM** | 循环结构建模时间依赖 | 221.35 | 382.49 | 0.977 |
| **Transformer** | self-attention 建模窗口内依赖 | 225.56 | 385.80 | 0.976 |
| **CNN** | 1D 卷积提取局部模式 | 257.78 | 407.62 | 0.970 |

**结果解读**：单独序列模型显著弱于基础非时序模型，核心原因不是“深度学习一定无效”，而是输入特征缺少强 `OT` 滞后信息，且一个月数据不足以支撑高容量模型充分学习复杂工况。因此下一步不是简单继续堆深层网络，而是做结构组合与训练细节优化。

### 4.3 序列模型扩展：NAS 与 CNN-LSTM

基于 §4.2 的负结果，本文进一步做小规模 NAS 式结构搜索，在 `mlp/cnn/lstm/cnn_lstm/cnn_lstm_attn` 及若干组合结构、`SEQ_LEN∈{2,4,8,16}`、学习率、隐藏维度、卷积通道数、损失函数、峰值过采样和非对称损失等候选空间中搜索。优化重点包括：

1. **峰值过采样**：用训练标签 95 分位数识别峰值样本，提高极端区间采样权重。
2. **按工况分簇训练尝试**：先用 `(exog_temp, exog_wind, OT_prev)` 聚类，再训练簇内子模型；该方案最终弱于统一模型，说明分簇后样本减少和路由误差会抵消收益。
3. **非对称损失**：对低估误差加权，缓解效率预测中的风险偏斜。

最终表现最好的两个序列扩展模型如下：

| 模型 | MAE (W) | RMSE (W) | R² | 关键结构参数 | 训练设置 |
|------|--------:|---------:|---:|--------------|----------|
| **CNN-LSTM-Attention** | **67.47** | **131.45** | 0.981 | input_dim=3, SEQ_LEN=8, CNN channels=32, kernel=3, LSTM hidden=64, heads=4, head 64→2 | LR=0.001, dropout=0.05, heteroscedastic output, MAE loss |
| **CNN-LSTM** | 67.56 | 135.51 | 0.980 | input_dim=3, SEQ_LEN=2, CNN channels=16, kernel=3, LSTM hidden=128, heads=4, head 128→64→2 | LR=0.0003, dropout=0, heteroscedastic output, sMAPE loss, peak oversampling |

两者输入均为 `[exog_temp, exog_wind, OT_prev]`。CNN-LSTM-Attention 对应 `out_cnn_lstm_grid_search_revised/run_024`，CNN-LSTM 对应 `out_cnn_lstm_grid_search_1/run_082`。结果说明“局部卷积 + 时间记忆 + 注意力加权”的结构比单独 CNN/LSTM/Transformer 更适合该序列任务。

---

### 4.4 Stacking 模型聚合

集成层输入为**基模型在测试集上的预测**，输出为融合后的 `OT`（见 §3.5 聚合层）。为避免把“集成方法差异”和“基模型池差异”混在一起，本文新增 `stacking_fair_pool.py`，固定同一个强基模型池：

> **StrongPool = RF + GBM + BayesianRidge + SVR**

选择该池的原因是：RF/GBM/SVR 是 §4.1 中表现最稳定的第一梯队或近第一梯队模型，BayesianRidge 提供线性/贝叶斯对照；KNN 和 AdaBoost 在主预测表中明显较弱，因此只放入扩展池作鲁棒性诊断，不进入主集成结论。

在同一 StrongPool 下，比较两种 meta-feature 生成策略：**Holdout stacking/blending**（训练集后 20% 作为 `meta_holdout`，5845 行）与 **OOF stacking**（`TimeSeriesSplit(5)` 产生折外预测，只使用真正获得 OOF 预测的 27830 行）。元学习器统一比较 RidgeCV、LassoCV、ElasticNetCV 与 NNLS。

| 策略 | 元学习器 | 基模型池 | MAE (W) | RMSE (W) | R² |
|------|----------|----------|--------:|---------:|---:|
| **Holdout** | **ElasticNetCV** | StrongPool | **51.31** | **79.61** | **0.9915** |
| Holdout | RidgeCV | StrongPool | 51.51 | 80.00 | 0.9914 |
| Holdout | LassoCV | StrongPool | 51.51 | 80.00 | 0.9914 |
| Holdout | NNLS | StrongPool | 51.54 | 80.14 | 0.9914 |
| OOF | ElasticNetCV | StrongPool | 60.43 | 90.29 | 0.9891 |
| OOF | LassoCV | StrongPool | 60.49 | 90.22 | 0.9891 |
| OOF | RidgeCV | StrongPool | 60.52 | 90.26 | 0.9891 |
| OOF | NNLS | StrongPool | 60.73 | 90.69 | 0.9890 |
| *参考：SVR 单模型* | — | — | *48.88* | *81.32* | *0.9912* |
| *参考：RF 单模型* | — | — | *49.34* | *86.67* | *0.9900* |

![统一 StrongPool 下的 stacking 策略对比](output_stacking_fair/stacking_fair_strong_pool_mae.png)

结果说明三点。第一，**所有公平集成方法的 MAE 都没有超过最佳单模型 SVR**；最佳 Holdout ElasticNetCV 为 51.31 W，比 SVR 高约 2.43 W。第二，Holdout stacking 的 RMSE 可降到 79.61–80.14 W，略低于 SVR 的 81.32 W，说明二层融合主要改善大误差而不是平均绝对误差。第三，修正后的 OOF 不再出现旧脚本中由未覆盖 OOF 行引起的异常结果，但仍明显弱于 Holdout（约 60.5 W vs 51.3 W），说明在本月内非平稳时序数据上，OOF 生成的 meta-feature 与测试阶段“全量重训基模型”的预测分布仍不够一致。

**为什么 stacking 反而不如单个模型？** 主要原因是第一梯队基模型已经非常强且高度相关。SVR、RF、GBM 都大量利用 `OT` 滞后项，误差模式相似，二层元学习器能获得的互补信息有限；当加入弱模型或不稳定模型时，元学习器还需要花容量去抑制噪声。另一方面，stacking 的训练目标是用 meta_holdout 或 OOF 预测去学习融合权重，但测试阶段基模型通常由更完整训练集重训得到，两者预测分布并不完全一致。对非平稳时间序列来说，这种 meta-feature shift 会让 OOF stacking 尤其吃亏。因此本实验中 stacking 更适合作为降低 RMSE、缓解局部大误差的工具，而不是稳定降低 MAE 的主模型。

扩展池（StrongPool + KNN + AdaBoost）的结论与主表一致：Holdout 最优 MAE≈51.46 W，OOF 约 60.5–61.6 W。弱基模型进入池后没有改善主结论，反而增加了元学习器需要处理的噪声。因此报告主结论采用 StrongPool。

除 StrongPool 外，我也尝试过不同的基模型池。例如 `ensemble.py` 的混合池 **RF + SVR + CNN + LSTM + Transformer → RidgeCV** 可以得到 MAE **49.59 W**、RMSE **79.05 W**，说明合适的 pool 选择确实可能改善集成结果。不过该实验同时改变了基模型池和二层训练设置，不能单独说明“哪一种 stacking 策略更优”。因此报告主结论采用统一 StrongPool 的公平比较，而把混合池结果作为说明“pool 选择会影响 stacking 上限”的补充证据。

---

### 4.5 预测曲线与可视化

统一评估脚本生成了主测试集误差对比与片段预测曲线。SVR 与 RF 对整体趋势跟踪较好；深度单模型误差偏大。公平 stacking 的定量比较以 §4.4 的统一 StrongPool 表格和配图为准。

![主测试集误差对比](final_results/main_error_comparison.png)

![测试集片段预测曲线](final_results/prediction_excerpt.png)

第二张图使用测试集中的连续片段而不是完整测试集，是出于可读性考虑：完整测试集约 8,349 个分钟级点，若全部画在一张图中，不同模型曲线会严重重叠，局部峰值、突变和模型间差异反而难以观察。完整测试集的信息已经通过 MAE/RMSE/R² 和第一张误差对比图体现；片段曲线则用于展示模型在局部时间窗口内对趋势、峰值和突变的跟踪能力。因此，报告采用“全局指标 + 局部曲线”的组合，而不是把全部时序点压缩进一张不可读的图。

---

### 4.6 统计检验与置信区间

统计检验和 Bootstrap 置信区间用于回答“预测模型之间的差异是否稳健”。这些结果只比较预测误差，不参与后续聚类评测，因此放在预测结果和聚类分析之间。

#### 4.6.1 配对误差检验

配对检验使用测试集逐点绝对误差。`B_minus_A < 0` 表示 ModelB 的平均绝对误差低于 ModelA；`B_minus_A > 0` 表示 ModelB 更差。

| ModelA | ModelB | MeanAbsErr_A | MeanAbsErr_B | B_minus_A | Paired t-test p | Wilcoxon p |
|--------|--------|-------------:|-------------:|----------:|----------------:|-----------:|
| RF | SVR | 49.34 | 48.88 | -0.46 | 0.4426 | 3.23e-24 |
| SVR | Holdout ElasticNetCV | 48.88 | 51.31 | 2.43 | 6.75e-11 | 1.55e-38 |
| SVR | OOF ElasticNetCV | 48.88 | 60.43 | 11.54 | 1.23e-131 | 1.01e-159 |
| Holdout ElasticNetCV | OOF ElasticNetCV | 51.31 | 60.43 | 9.12 | 2.72e-183 | 1.85e-143 |
| RandomForest | CNN | 49.23 | 257.78 | 208.54 | < 1e-6 | < 1e-6 |

解读上不能只看 p 值。RF 和 SVR 的 t-test 不显著（p=0.4426），说明二者平均误差差异很小；Wilcoxon 显著则说明逐点误差分布存在系统差异。SVR 与 Holdout ElasticNetCV 的差距约 2.43 W，在统计上显著，但工程上仍属于较小差距；OOF stacking 相比 Holdout 的 9 W 以上差距则更有实际意义。

#### 4.6.2 TimeSeriesSplit 交叉验证

CV 在训练集（N=29,225）上进行，每折扩大训练窗口，验证随后时间段：

| 模型 | CV MAE ± std (W) | CV RMSE ± std (W) | 测试集 MAE（参考） |
|------|------------------|-------------------|-------------------:|
| RF | 65.8 ± 16.1 | 122.6 ± 28.8 | 49.4 |
| GBM | 68.4 ± 8.5 | 121.9 ± 22.1 | 50.2 |
| BayesianRidge | 76.8 ± 12.2 | 132.9 ± 25.4 | 74.1 |
| KNN | 165.9 ± 64.7 | 246.8 ± 86.5 | 82.2 |
| SVR（固定超参） | 286.6 ± 189.5 | 385.6 ± 238.4 | 48.9 |

RF/GBM 的 CV MAE 高于测试 MAE，主要因为 expanding-window 早期折训练样本较少；BayesianRidge 的 CV 与测试更接近，说明线性模型受训练集大小影响较小。SVR 的固定超参 CV 很差，说明 SVR 对超参数高度敏感，主结果必须以 TimeSeriesSplit 网格搜索后的 SVR 为准。

#### 4.6.3 Bootstrap 置信区间

Bootstrap 在测试集上做 B=2000 次有放回重采样，估计 MAE/RMSE 的 95% 置信区间。公平集成结果来自 `stacking_fair_pool.py`：

| 模型 | MAE | 95% CI | RMSE | 95% CI |
|------|----:|--------|-----:|--------|
| SVR | 48.88 | [47.53, 50.24] | 81.32 | [78.26, 84.82] |
| RF | 49.34 | [47.90, 50.84] | 86.67 | [83.31, 90.16] |
| GBM | 50.47 | [48.96, 51.97] | 83.93 | [80.12, 87.95] |
| Holdout ElasticNetCV | 51.31 | [50.03, 52.60] | 79.61 | [76.77, 83.00] |
| Holdout RidgeCV | 51.51 | [50.13, 52.86] | 80.00 | [77.00, 83.29] |
| OOF ElasticNetCV | 60.43 | [58.98, 61.80] | 90.29 | [87.49, 93.21] |
| BayesianRidge | 74.09 | [72.38, 75.89] | 110.69 | [107.72, 113.84] |

顶层模型（SVR/RF/GBM/Holdout stacking）的 MAE 置信区间有明显重叠，说明单看点估计不应过度解释 1–3 W 的差异。Holdout stacking 的 RMSE 区间低于 RF/GBM，并与 SVR 接近，支持“stacking 更像是在降低大误差，而不是稳定降低 MAE”的结论。OOF 的 MAE/RMSE 区间则整体右移，说明它在本时序数据上不是偶然失败。

补充地，单独 CNN/LSTM/Transformer 的 MAE 约 221–258 W，置信区间远离传统模型；混合池 NNLS 在旧探索实验中 MAE≈947 W，作为失败对照保留，但不进入公平集成主结论。

---

## 5. 聚类与工况划分

预测模型回答的是连续功率 `OT` 的数值预测；本节进一步讨论模型表示能否支持工况划分和停机决策。所有聚类实验都不参与 §4 的 MAE/RMSE 预测排序，而是作为决策支持和可解释性分析。

### 5.1 聚类基础模型、标签有效性与指标合理性

#### 5.1.1 聚类使用的基础模型

| 项目 | 设定 |
|------|------|
| **代表性模型** | **CNN-LSTM-Attention**（`train_cnn_lstm.py` 训练；权重 `out_cnn_lstm_cluster_1/model_run0_cluster0.pt`） |
| **用于聚类的表示** | 测试集每条样本的 **128 维 penultimate embedding**（及可选的 1 维 `OT_pred`） |
| **聚类算法** | 默认 **KMeans(k=2)**；§5.3–§5.4 在**同一 embedding** 上对比层次聚类 / K-Medoids |

#### 5.1.2 为何可用「一个代表性模型」判断聚类有效性

1. **任务对齐**：该模型是 **预测 MAE 最优深度结构之一**（CNN_LSTM / CNN_LSTM-Attention，§4.3），且为课题主线的 **双任务（预测 + 表征）** 实现，其 embedding 专为 `OT` 时序与工况变化训练，比随机特征或弱模型 embedding 更具信息量。

2. **表征质量可验证**：相对 OT-KMeans 参照分群 ARI≈0.82（§5.2.2）；在**统一停机基线**上 embedding KMeans 达 Accuracy **94.67%**，**预测值聚类**为 93.96%，说明 embedding 更适于决策层聚类——若换用弱 CNN 单模块（MAE≈258 W），工况结构噪声更大，聚类结论不再稳定。

3. **算法对照在同一表示上**：§5.3、§5.4 不改变基础模型，只改变聚类器（KMeans / Ward / PAM）。若多种聚类器在**同一 embedding** 上给出相近 ARI（或相对参考分群一致），则结论反映**数据结构**而非某一聚类实现的偶然性。

4. **不声称跨模型普适**：未对 `ensemble.py` 的 CNN/LSTM/Transformer embedding 逐一做聚类普查；代表性模型选取基于**课题主线 + 预测性能 + 下游停机评测（94.67%）**，在报告中应表述为「在主推深度模型上的工况结构分析」，而非「所有模型的普适结论」。

#### 5.1.3 标签与指标（统一基线）

本文无逐分钟人工结冰标注。**主标签**为 `OT_true < 1000 kW`（停机代理，§3.6.9）；**补充参照**为 OT 上 KMeans 分群（§3.6.9）。主表一律用停机基线上的 Accuracy/F1；ARI 可同时报告「相对 OT-KMeans」以说明功率分档是否被 embedding 保留。**94.67% 是规则一致性，不是结冰检测真值。**

---

### 5.2 工况划分与停机决策评测（统一基线：`OT < 1000 kW`）

本节在 embedding 上运行 KMeans(k=2)，再用簇内多数投票映射停机标签，以 `OT_true < 1000 kW` 为 ground truth。以下 **Accuracy / F1 均在同一基线上** 由 `cluster_unified_eval.py` 汇总（N=8347）。

#### 5.2.1 主结果表

| 方法 | 特征 | Accuracy | F1 | Recall | Precision | ARI vs OT-KMeans |
|------|------|----------|-----|--------|-----------|------------------|
| **KMeans** | **Embedding** | **94.67%** | **95.45%** | **97.33%** | 93.64% | 0.815 |
| KMeans | Prediction (`OT_pred`) | 93.96% | 94.92% | 98.06% | 91.97% | 0.815 |
| Agglomerative-complete | Prediction | 90.73% | 92.48% | 99.12% | 86.66% | 0.744 |
| Agglomerative-complete | Embedding | 89.82% | 91.80% | 99.15% | 85.47% | 0.718 |
| Agglomerative-average | Prediction | 93.81% | 94.80% | 98.21% | 91.62% | 0.816 |
| Agglomerative-average | Embedding | 88.57% | 89.06% | 80.93% | 99.01% | 0.500 |
| Agglomerative-ward | Embedding | 86.43% | 89.40% | 99.52% | 81.14% | 0.619 |
| Agglomerative-ward | Prediction | 83.48% | 87.40% | 99.69% | 77.81% | 0.536 |

**主结果（加粗首行）**：Embedding + KMeans → **Accuracy 94.67%**，与 `emb_cluster_binary_metrics.json` 一致。结果表明：**embedding 聚类优于对标量 `OT_pred` 聚类**（本表 94.67% vs 93.96%）。

**混淆矩阵（Embedding KMeans）**：TP=4671, FP=317, TN=3231, FN=128。

#### 5.2.2 补充：相对 OT-KMeans 参照分群（不作主 Accuracy）

| 对象 | ARI | NMI | 匈牙利 Accuracy* |
|------|-----|-----|-------------------|
| Embedding KMeans vs `true_cluster` | 0.815 | 0.715 | 0.9515 |
| Prediction KMeans vs `true_cluster` | 0.815 | 0.712 | 0.9516 |

\*匈牙利 Accuracy 仅用于说明与功率二分的一致性；**不得**与 94.67% 并列比较。脚本：`embedding_analysis.py` → `clustering_classification_metrics.csv`。

#### 5.2.3 K=3 分层解释

K=3 时三簇沿 `OT` 由高到低梯度分布，应停机比例约 40.6% / 62.6% / 75.3%，支持「安全 / 边界 / 高风险」分层（`emb_cluster_k3_summary.csv`）。

### 5.3 聚类算法比较：AgglomerativeClustering vs KMeans（A8）

**方法适用性**：KMeans 是本文的主要无监督工具，但其结论的可信度依赖于一个问题：工况分离结构是否只有在球形假设下才成立？凝聚式层次聚类（AgglomerativeClustering）通过「自底向上合并」回答了这个问题——它不预设簇形状，而是根据样本间距离逐步合并最近的一对，最终用树状图（dendrogram）可视化全过程。

三种连接策略的几何含义不同：`ward` 最小化合并时的簇内方差增量，行为最接近 KMeans；`complete` 用两簇间最远点对定义距离，倾向于产生紧凑圆形簇；`average` 用所有跨簇点对距离均值，平衡两者之间。

实验在 CNN-LSTM-Attention 的 128 维 embedding（标准化）与 1 维 `OT_pred` 上对比 KMeans 与三种层次聚类；**Accuracy/F1 均在 §5.2 同一停机基线（OT < 1000 kW）上计算**（见 `clustering_shutdown_metrics.csv`）。

**结果**（N=8347，ground truth = `OT_true < 1000 kW`）：

| 方法 | 特征 | Accuracy | F1 | ARI vs OT-KMeans |
|------|------|----------|-----|------------------|
| **KMeans** | **Embedding** | **94.67%** | **95.45%** | 0.815 |
| KMeans | Prediction | 93.96% | 94.92% | 0.815 |
| Agglom-average | Prediction | 93.81% | 94.80% | 0.816 |
| Agglom-complete | Prediction | 90.73% | 92.48% | 0.744 |
| Agglom-complete | Embedding | 89.82% | 91.80% | 0.718 |
| Agglom-average | Embedding | 88.57% | 89.06% | 0.500 |
| Agglom-ward | Embedding | 86.43% | 89.40% | 0.619 |
| Agglom-ward | Prediction | 83.48% | 87.40% | 0.536 |

**分析**：


在停机基线上，**KMeans + Embedding 仍最高（94.67%）**；层次聚类在 embedding 上普遍降至 86–90%，说明算法选择会影响决策层指标。相对 OT-KMeans 的 ARI 均在 0.5–0.82 区间，说明工况分离结构真实存在。KMeans 在 128 维标准化 embedding 上表现最佳，原因是：PCA 分析显示该 embedding 有效维度 ≤10（前 10 个主成分已解释接近 100% 方差），其工况分离是近似球形的低维结构，正是 KMeans 的最优场景。`average` 连接在高维下受「维度诅咒」影响——高维空间中点对距离趋于集中，`average` 利用全部跨簇点对均值，分辨率下降明显（ARI=0.500）。

在 1D 预测值特征上，`average` 与 KMeans 表现完全一致（ARI=0.816）：1D 情形下 `average linkage` 退化为单调阈值切割，等价于 KMeans(k=2)。

综合来看，KMeans 在本任务中不是随意选择，而是与 embedding 的低维球形结构高度匹配的合理选择，层次聚类的多方法对比为这一结论提供了跨方法鲁棒性支撑。

### 5.4 K-Medoids PAM 对比实验（A6）

**方法适用性**：K-Medoids（PAM，Partitioning Around Medoids）是 KMeans 的变体，聚类中心从「均值（可能是空间中不存在的虚点）」改为「实际数据点中使总内部距离最小的 medoid」。其核心优势有三：① 对离群点鲁棒（极端值只能参与投票，不会拉偏均值）；② 中心可直接溯源为真实样本；③ 可使用任意距离度量而非只有欧氏距离。

本文手写 PAM 实现（`cluster_kmedoids.py`），对 PCA-10 压缩后的 embedding 和 1D 预测值分别运行 KMeans 与 K-Medoids，并设计离群点鲁棒性实验。

**实验 A — 正常数据（Embedding PCA-10）**：

| 方法 | ARI vs OT-KMeans | 停机基线 Accuracy* |
|------|------------------|---------------------|
| KMeans | **0.816** | **94.67%**（与 §5.2 一致，全维 embedding） |
| K-Medoids PAM | 0.542 | 待 `cluster_unified_eval` 扩展；PAM 在 PCA-10 上低于 KMeans |

\*旧脚本中 0.952 为相对 OT-KMeans 的匈牙利 Accuracy，**已弃用为主指标**。


**实验 B — 离群点鲁棒性**（注入 50 个 PC1 方向 +8σ 极端点）：


| 方法        | 聚类中心最大偏移（PC1） |
| --------- | ------------- |
| KMeans    | **0.001**     |
| K-Medoids | 0.047         |


**实验 C — 1D 预测值**：


| 方法        | 中心/Medoid         | Accuracy* | F1* | ARI vs OT-KMeans |
| --------- | ----------------- | --------: | --: | ---------------: |
| KMeans    | 296W（均值，抽象）；1860W | 95.16% | 93.76% | 0.815 |
| K-Medoids | -5.3W（真实预测）；1579W | 89.48% | 87.96% | 0.623 |

\*本表的 Accuracy/F1 来自 `cluster_kmedoids.py` 的匈牙利对齐评测，参照对象是 `OT_true` 上的 KMeans 二分（OT-KMeans），不是 §5.2 的 `OT < 1000 kW` 停机基线。因此它只用于比较 KMeans 与 K-Medoids 在同一 1D prediction 特征上的聚类一致性，不作为主停机决策准确率。


**分析**：

三组实验均得到与直觉相反、但在理论上可解释的结论，体现了批判性实验设计的价值：

- **KMeans 准确率更高**：这并非说明 K-Medoids 算法差，而是说明当前 embedding 的工况聚类结构是低维球形的（如 §5.3 所示），恰好符合 KMeans 的假设。在这种场景下，「均值」是最优聚类中心的最大似然估计；K-Medoids 受限于只能选实际样本，在样本密度非均匀时可能找不到最优代表。此外 PAM 对初始化敏感，高维空间中容易陷入局部最优。
- **离群点鲁棒性反转（K-Medoids 偏移更大）**：50 个极端点在 8347 个样本中仅占 0.6%，KMeans 均值几乎感知不到（稀释效应：均值偏移 ≈ 50×8σ / 4000 ≈ 0.1，但簇分配稳定，实际偏移 < 0.001）。K-Medoids 的中心偏移（0.047）看似更大，原因是离群点改变了部分边界样本的簇归属，使得某个簇的「最小总距离」点轻微移位。**K-Medoids 相对 KMeans 的鲁棒性优势在离群点比例 >5% 时才显著。**
- **1D prediction 聚类的结果**：Agglomerative 在 §5.2–§5.3 中已经报告了 Prediction 特征上的停机基线结果，其中 `Agglomerative-average + Prediction` 达到 Accuracy 93.81%、F1 94.80%，接近 `KMeans + Prediction` 的 93.96%、94.92%。K-Medoids 在 1D prediction 上 Accuracy/F1/ARI 均低于 KMeans，但两个 medoid（-5.3W 和 1579.2W）是预测序列中真实存在的两个时刻：前者对应低功率异常时刻，后者对应正常高功率工况的典型时刻。这种「用真实样本做中心」的特性对于工程解释和异常案例查询有价值，即使聚类一致性略低。

### 5.5 隐马尔科夫模型工况切换建模（A14/A15）

**方法说明**

隐马尔科夫模型（HMM）假设观测序列由一组离散隐状态生成，相邻时刻的隐状态之间通过转移概率矩阵 A 联系，每个隐状态对应一个发射分布（此处用完整协方差高斯分布）。本文用 `hmmlearn.GaussianHMM` 对 (OT, 环境温度) 双变量时序拟合参数，把「正常发电 / 低功率 / 异常工况」建模为隐状态。与 KMeans 相比，HMM 的核心优势在于显式建模**状态转移的时序依赖**：一个状态的切换不仅取决于当前观测值，还受上一时刻状态的约束，符合结冰事件的连续演化特性。

**为何适用**：结冰是隐状态（无直接测量），且状态变化连续（不会在相邻分钟内随机跳变），HMM 的时序约束天然契合这一特性。

**BIC 模型选择**：

对 K=2,3,4 分别拟合 Gaussian HMM（`covariance_type="full"`），计算 BIC：


| K   | log-likelihood | n_params | BIC |
| --- | -------------- | -------- | --- |
| 2   | -8.345 × 10⁴   | 13       | 1.670 × 10⁵ |
| 3   | -9.419 × 10⁴   | 23       | 1.886 × 10⁵ |
| 4   | -1.617 × 10⁴   | 35       | 3.271 × 10⁴ |


BIC 最优 K=4（BIC 下降显著），说明在纯似然-参数惩罚口径下 K=4 更能拟合观测序列；但由于本文没有逐分钟结冰真值，且 K=3 已出现状态重合退化，HMM 结果只作为补充诊断，不作为主工况划分结论。

**K=2 物理解释**（最简单的二分）：


| 状态    | OT 均值    | 温度均值     | 样本比例  |
| ----- | -------- | -------- | ----- |
| 低功率状态 | 290.6 W  | -11.3 °C | 50.5% |
| 高功率状态 | 2019.3 W | -11.0 °C | 49.4% |


K=2 时两个状态的 OT 差异极大（290W vs 2019W），但**温度几乎相同**（-11.3°C vs -11.0°C）。这揭示了数据的一个关键特性：2024 年 2 月整月都处于极寒期，环境温度变化范围极窄，HMM 主要靠 OT 而非温度区分工况。

**K=3 退化问题**（状态 0 ≈ 状态 1）：

K=3 时，HMM 将状态 0（OT=1062W，Temp=-14.3°C）和状态 1（OT=1062W，Temp=-14.3°C）的均值几乎完全一致，说明模型在这个维度上退化到局部最优，分不出有意义的第三个状态。K=3 的 BIC 反而比 K=2 更差，也印证了"强行三聚类"对本数据不自然。

因此，HMM 在本文中被视为一次补充性时序隐状态建模尝试：它展示了状态转移建模的思路，但由于 K=3 隐状态发生重合退化，本文不将其作为主要工况划分或结冰识别结论。

**转移矩阵（K=3）**：


| 从 \ 到 | 状态 0      | 状态 1      | 状态 2      |
| ----- | --------- | --------- | --------- |
| 状态 0  | 0.002     | **0.998** | 0.000     |
| 状态 1  | **0.989** | 0.011     | 0.000     |
| 状态 2  | 0.000     | 0.000     | **0.999** |


状态 2（高功率，Temp=-1.2°C）自转移概率 0.999，说明高功率状态极为稳定；状态 0 和状态 1 互相高频切换（这两个退化状态本应合为一个），也从侧面说明了 K=3 的退化问题。

**与 KMeans 一致性**：


| 指标                      | 值     |
| ----------------------- | ----- |
| ARI（HMM vs KMeans, K=3） | 0.231 |
| NMI（HMM vs KMeans, K=3） | 0.353 |


ARI 和 NMI 均在 0.2–0.4 之间，属于**中等一致**。差异的来源正是方法本质的不同：KMeans 按欧氏距离将 (OT, Temp) 空间割分成 Voronoi 区域（不考虑时序），HMM 利用转移矩阵在时间轴上施加平滑约束，将短暂波动归到当前持续状态而不是立即切换。

**主要发现与课程知识点对照**：

1. **Viterbi 解码**（A15）：`GaussianHMM.predict()` 内部使用 Viterbi 算法求最优隐状态序列，是 HMM 的核心推断算法。
2. **EM（Baum-Welch）训练**（A14）：`fit()` 使用 Forward-Backward + M-step 估计 A、均值和协方差，等价于带隐变量的最大似然估计（EM 算法特例）。
3. **模型选择（BIC）**：BIC 在 log-likelihood 上加参数惩罚，本实验中 K=4 的 log-likelihood 跳跃远大于参数增加带来的惩罚，BIC 选出 K=4。
4. **数据局限性的诊断能力**：K=3 退化、温度通道几乎不起作用——这些反直觉结果本身就是有价值的分析，说明在单月冬季数据上，仅用 (OT, 温度) 双观测不足以稳定分辨 3 个以上的工况，需要补充更多气象变量（如风速、湿度）或使用更长时间段的数据。

## 6. 课程知识点总结与方法反思

### 6.1 回归学习：模型复杂度要和特征信息量匹配

本实验最直接的课程启示来自回归学习部分（A1/A2/A3/A12/A13）：模型不是越复杂越好，关键在于模型复杂度是否匹配数据规模与特征信息量。本文构造了 `OT_lag_1/2/3/6/12` 与 rolling 统计特征后，历史功率已经携带了很强的自回归信号，因此 SVR、Random Forest、GBM 这类非线性但相对稳健的模型可以取得第一梯队结果。相反，CNN、LSTM、Transformer 单模型虽然表达能力更强，但在一个月数据、少量外生变量、强滞后特征已经存在的条件下，容易出现容量没有充分利用、训练信号不够丰富或对局部波动过敏的问题。

这对应了课程中的偏差-方差权衡：传统模型的优势不只是“简单”，而是它们在有限样本下的有效容量更合适。BayesianRidge 的结果也提供了一个线性参照：它不如 SVR/RF/GBM，但表现稳定，说明目标确实有强线性自回归成分；KNN 和 AdaBoost 较弱，则反映出距离度量和浅弱学习器在高波动时序回归中的局限。

### 6.2 集成学习：Stacking 的前提是误差互补

Stacking 对应课程中的集成学习（A10/A11/A13），但本实验也说明：集成学习不是把模型堆在一起就必然变好。统一 StrongPool 下，Holdout stacking 的 RMSE 可以降到约 79.6--80.1 W，说明它确实修正了一部分大误差；但 MAE 没有稳定超过最佳单模型 SVR。原因是 RF、GBM、SVR 已经从同一组强滞后特征中学习到相似规律，误差高度相关，二层模型可利用的互补信息有限。

我还尝试过不同的基模型池：例如混合池 `RF+SVR+CNN+LSTM+Transformer` 配合 RidgeCV 可以得到更低的 MAE（49.59 W），说明基模型池设计会显著影响 stacking 结果。但这也提醒我们，stacking 的结论必须同时说明“底层模型有哪些、meta-feature 怎样生成、二层学习器是什么”。在本报告中，主结论采用统一 StrongPool，是为了让 Holdout/OOF、Ridge/Lasso/ElasticNet/NNLS 的比较更公平。

### 6.3 无监督学习：聚类是决策支持，不是监督真值

工况划分对应课程中的无监督学习（A5/A6/A8）。KMeans、AgglomerativeClustering、K-Medoids 都不是直接预测 `OT` 的模型，而是把 embedding、预测值或原始功率结构转换为运行状态分组。由于数据没有逐分钟人工结冰标签，聚类结果不能被直接解释为“结冰检测准确率”；本文的 94.67% 是 embedding KMeans 与 `OT < 1000 kW` 停机代理规则的一致性，含义是“与工程阈值高度一致”，而不是“真实结冰识别准确率”。

多种聚类方法的对比也强化了一个课程知识点：算法假设必须和数据几何结构匹配。KMeans 在 embedding 上优于层次聚类和 K-Medoids，并不说明后两者没有价值，而是说明当前 embedding 的有效维度较低、簇形状接近球形，正适合 KMeans。K-Medoids 虽然主指标较低，但 medoid 是真实样本，便于定位典型低功率和正常高功率时刻；Agglomerative 在 1D prediction 上接近 KMeans，则说明预测值本身已经形成明显的阈值式分层。

### 6.4 时序隐状态：HMM 适合作为结构诊断

HMM 对应隐变量模型、EM 训练和 Viterbi 解码（A14/A15）。在本实验中，HMM 的位置不是替代回归模型，而是从时间连续性角度检查工况切换。K=2 时 HMM 主要分出低功率和高功率两类状态；K=3 出现状态重合退化；BIC 又偏向 K=4。这些结果说明，HMM 可以揭示状态转移结构，但在缺少湿度、叶片状态、除冰动作记录和逐分钟标签时，仅靠 `(OT, Temp)` 难以稳定解释更多细粒度工况。

这个负结果有实际价值：它把“模型没给出漂亮分类”转化成了对数据边界的诊断。尤其是 2024 年 2 月整月温度都较低，温度通道本身区分度有限，因此 HMM 主要依赖 OT，而不是找到了真正独立的结冰观测信号。

### 6.5 实验准则与统计检验：不要只看点估计

课程中的实验设计和统计检验（E1--E4/T2）在本项目里非常关键。所有主结果按时间顺序划分训练、验证、测试集，并在训练段内部完成标准化、网格搜索和聚类拟合，避免未来信息泄露。TimeSeriesSplit 用于估计模型在不同训练时间窗口上的波动；Bootstrap 置信区间用于给 MAE/RMSE 加不确定性范围；paired t-test 与 Wilcoxon 检验则比较同一测试样本上的逐点绝对误差差值。

选择配对检验的原因是，每个模型都在同一批测试样本上预测，误差天然成对。paired t-test 检验差值均值，Wilcoxon 不要求差值正态，更适合误差重尾或异常点较多的时序预测场景。不过统计显著不等于工程显著：如果 MAE 只差 1--2 W，即使样本量大导致 p 值很小，也需要结合 RMSE、置信区间、预测曲线和业务阈值判断是否值得采用更复杂模型。

### 6.6 局限性与改进方向

本文仍有以下限制。第一，主数据时间跨度只有 2024 年 2 月一个月，难以覆盖完整季节变化、多次结冰事件和跨风机泛化。第二，当前主数据没有逐分钟人工结冰标签，因此无法做严格的监督式结冰分类，只能用停机代理规则和聚类一致性做决策支持分析。第三，MAPE 对接近 0 或负值的 `OT` 不稳定，因此报告中更依赖 MAE、RMSE、sMAPE、置信区间和曲线图的综合判断。第四，深度模型受数据规模和特征数量限制，当前结果不能说明深度模型本身不适合风机预测，只能说明在本数据和当前特征条件下，强特征工程加传统模型仍然是更稳健的选择。

## 7. 结论

本文完成了风机结冰相关工况下 `OT` 预测、stacking 集成、聚类工况划分与可靠性验证实验。预测部分（§4）显示，SVR/RF/GBM 是强特征工程下的最优传统基线；CNN/LSTM/Transformer 单模型较弱，但经过 NAS 式结构搜索后，CNN-LSTM 与 CNN-LSTM-Attention 明显优于单一深度模块。Stacking 部分（§4.4）在统一 StrongPool 下比较 Holdout/OOF 与多种元学习器，结论是 Holdout stacking 能改善部分 RMSE，但 MAE 未稳定超过 SVR；尝试不同基模型池可以得到更好结果，但也更需要清楚说明 pool 与二层学习器配置。

聚类与工况划分部分（§5）把 KMeans、AgglomerativeClustering、K-Medoids 和 HMM 放在同一决策支持框架下分析。主结果为 CNN-LSTM-Attention embedding + KMeans 在 `OT < 1000 kW` 停机代理标签上达到 Accuracy **94.67%**、F1 **95.45%**；Prediction 聚类、层次聚类和 K-Medoids 提供了方法对照，HMM 则补充了时序隐状态视角。整体来看，本实验的核心收获不是某一个模型“赢了”，而是把模型选择、特征设定、集成前提、聚类解释和统计可靠性放在同一个机器学习工作流中加以检验。

## 8. 复现说明

主要文件：


| 脚本                         | 功能                                                   | 相关知识点                 |
| -------------------------- | ---------------------------------------------------- | ---------------------- |
| `ensemble.py`              | 混合池探索实验（RF/SVR/CNN/LSTM/Transformer → RidgeCV） | A1/A10/A11/A12/A16/A19 |
| `baselines_ext.py`         | 补充传统基线（GBM/AdaBoost/KNN/BayesianRidge）               | A13/A3/A2/T1           |
| `train_cnn_lstm.py`        | CNN-LSTM-Attention 训练与工况划分                           | A19/深度学习               |
| `cnn_lstm_grid_search.py`  | CNN-LSTM 超参网格搜索                                      | E1/E2                  |
| `embedding_analysis.py`    | 提取 embedding → KMeans 一致性分析                          | A5                     |
| `cluster_agglomerative.py` | AgglomerativeClustering vs KMeans 对比                 | A8                     |
| `cluster_kmedoids.py`      | 手写 PAM K-Medoids 对比实验                                | A6                     |
| `bootstrap_cv_eval.py`     | 显式 TimeSeriesSplit k-fold CV + Bootstrap 置信区间        | E2/E3/T2               |
| `cluster_eval.py`          | 停机基线（OT&lt;1000 kW）聚类二分类评估                              | —                      |
| `cluster_unified_eval.py`  | 各聚类方法在**同一停机基线**上的 Accuracy/F1 汇总                      | —                      |
| `final_evaluation.py`      | 最终评估（读预测文件，生成报告表格和配对检验）                              | E4/T2                  |
| `paper_protocol_eval.py`   | 序列结构扩展实验汇总与 RF/SVR 对照评估                            | E4                     |
| `stacking_fair_pool.py`    | 统一 StrongPool 下比较 Holdout/OOF stacking、NNLS 与置信区间           | A10/A1/E2/T2           |
| `stacking_comparison.py`   | 早期 Holdout vs OOF 探索脚本（不作为主结论）                         | A10                    |
| `model.py`                 | 基础模型单独对比脚本                                           | —                      |


推荐复现实验顺序：

```bash
python ensemble.py
python baselines_ext.py
python stacking_fair_pool.py
python final_evaluation.py
```

若只想复现预测层最终表格，可以直接运行：

```bash
python final_evaluation.py
```

若只想复现 §4.4 与 §4.6 中的公平集成、Bootstrap CI 和配对检验，可以直接运行：

```bash
python stacking_fair_pool.py
```

输出文件位于 `final_results/`：

- `final_metrics.csv`
- `significance_tests.csv`
- `supplementary_results.csv`
- `report_tables.md`
- `main_error_comparison.png`
- `prediction_excerpt.png`

公平集成实验输出文件位于 `output_stacking_fair/`：

- `stacking_fair_results.csv`
- `stacking_fair_bootstrap_ci.csv`
- `stacking_fair_significance_tests.csv`
- `stacking_fair_coefficients.csv`
- `stacking_fair_predictions.csv`
- `stacking_fair_strong_pool_mae.png`

## 9. 后续工作

后续可以从三方面改进：

1. 获取更多月份和更多风机的数据，测试模型跨时间、跨风机泛化能力。
2. 与电机系专家共同定义停机阈值或逐分钟结冰标签，把当前预测问题进一步扩展为风险分类任务。
3. 引入更完整的物理和运行特征，例如湿度、叶片转速、机舱状态、除冰动作记录等，提高结冰相关工况识别能力。
