# DNaIR-LLM: LLM-Enhanced Reinforcement Learning for Recommendation

本文档旨在全面回顾、梳理并固化本项目从“理论推导”到“数据工程管道”，再到“核心算法底层重构”及“顶会级多维评价体系建立”的全链路工作。系统性地证明了**“大语言模型（LLM）+ 确定性强化学习（RL）”**架构如何从数学底层和交互闭环上，彻底击穿推荐系统的流行度偏差与冷启动死局。

*(注：Baseline模型确立不在本次总结范围内)*

---

## 📁 项目架构 (Project Structure)

```text
DNaIR-LLM/
├── context/                 # 核心文档、规划、总结 (Core documentation & plans)
├── data/                    # 原数据与处理后数据 (Raw and processed embeddings/API metadata)
├── dataset/                 # 评测推荐数据集 (Evaluation datasets like ml100k)
├── docs/                    # 阶段性报告 (Phase summaries)
├── figs/                    # 架构图示 (Architecture diagrams)
├── model/                   # 核心神经网络模块 (Core Deep Learning Models)
│   ├── dqn.py               # 双塔 DQN 与动作生成器
│   ├── environment.py       # 对偶奖励强化学习环境
│   └── semantic_bridge.py   # GPU 常驻语义桥接器
├── paper/                   # 参考文献与理论源头 (Reference papers)
├── scripts/
│   └── data_prep/           # 数据挖掘与 LLM 自动化 API 接入脚本
├── tests/                   # 单元测试与集成测试防线 (Unit & Integration tests)
├── util/                    # 多维评测指标、稀疏矩阵等工具库 (Utilities for metrics)
├── main.py                  # CLI 执行入口 (Execution entry point)
├── train.py                 # 主训练与多线程验证循环 (Main training & evaluation loop)
└── requirements.txt         # 环境依赖 (Dependencies)
```

---

## 💡 Phase 1: 理论重构与对偶公式推导 (Ideation & Mathematical Derivation)

### 1. 痛点洞察：DNaIR 的“冷启动死局”

原始 [DNaIR (2025)] 强化学习模型虽然引入了多样性生成器，但其极其依赖于“交互频次”的协同过滤基础。一旦面临历史交互数极少（甚至为 0）的冷门商品，系统不仅无法计算出准确的 CF 相似度，更会在“贝叶斯质量因子先验”公式下直接判定其质量平庸（统统默认赋值为 0.5）。这导致长尾商品即便再优质，也无法获得合理的强化学习奖励（Reward）。

### 2. 突破性方程：LLM 零样本质量评分的注入

我们从数学公式的底层进行了大刀阔斧的重构，正式将由大模型（LLM）进行无依赖阅读文本产生出的“内在语义质量分 $\hat{\psi}_a$” 注入环境，提出了**语义质量感知对偶奖励公式**：

$$
R_{new}(\tilde{s}_t, \tilde{a}) = \text{sim}(\tilde{s}_t, \tilde{a}) + \beta \cdot \hat{\psi}_a \cdot r_{nov}(a)
$$

此举将去偏的逻辑从“强行推劣质低频物”升华为“挖掘那些高品质但尚未被发现的遗珠”，从根本上保障了用户的个性化体验。

---

## 🛠️ Phase 2: 数据工程与语义处理管道搭建 (Data & Semantic Pipeline)

在大语言模型和深度 Q 网络 (DQN) 之间，我们建立了一套高度自动化、低延迟的数据处理生命线：

1. **结构化文本编织**：基于 MovieLens 100K 原始数据集，整合导演、演员、类型等离散标签，将其转化为结构化的“实体电影文本画像”。
2. **DeepSeek API 零样本打分（核心缺口堵漏）**：
   - 编写了高并发 `generate_semantic_quality.py` 脚本，对 1682 部电影进行 API 推理，让 LLM 剥离流行度偏见，“仅凭电影内容的结构文本”打出 $0.0-1.0$ 的独立质量分（$\hat{\psi}_a$）。
   - 证明了 DeepSeek 的打分极为客观：彻底颠覆原来 0.5 的死板均线，电影得分真实分布在 $[0.30, 0.90]$，且仅有极小比例低于 0.5。
3. **BGE 稠密向量离线灌注**：
   - 基于最高质量的开源中文语义嵌入模型 BGE-M3，为全库电影离线生成尺寸为 `[1682, 768]` 的高维向量矩阵。
   - 彻底摆脱了在线环境中对大模型进行高延迟实时推理的可能性，为毫秒级在线决策打下坚实基础。

---

## ⚙️ Phase 3: 核心算法重构与底层代码实现 (Core Architecture Refactoring)

这是本项目含金量最高的“浴火重生”阶段，我们对原有的 PyTorch 框架进行了四项致命缺陷的深度清洗与升维：

### 重构 1：双塔结构 (Dual-Tower DQN) 与 GPU常驻

- 将原来只能接受单一离散 IDs 的网络拆分为双路并行的 **CF（协同过滤）塔**与 **Semantic（文本语义）塔**。
- 将原来使用 CPU for 循环拼接查找向量的灾难代码，重整为 `nn.Embedding.from_pretrained(freeze=True)` 形式，让数百万级参数量的冷僻向量全部常驻显存（GPU / Apple MPS），实现了零内存拷贝消耗的急速前向传播。

### 重构 2：经验回放池的“指针化防爆” (Replay Buffer Pointer Architecture)

- 高维浮点语义向量一旦存入容量百万级的 Buffer 会导致物理内存爆满崩溃。我们将 Buffer 改制为只存 `int` 类型 ID（存储容量节约 99.8%）。DQN 在执行 `learn()` 从 Buffer 抽样批次数据时，利用上述 GPU 常驻接口“On-the-fly”解冻查表。

### 重构 3：InfoNCE 假负样本掩码 (False-Negative Mask)

- 如果 Batch 内两个用户分别观看了相同的优秀冷门电影（如《星际穿越》），原生的 InfoNCE 损失会误把它们当成负样本强制在向量空间内推远。
- 我们重构了对比损失 `compute_infonce_loss` 方法，动态加载相同 Action 的掩膜（Mask -> -inf），精准保护了极其宝贵的聚类空间结构防止维度崩塌。

### 重构 4：DSAG 多样性发生器的逻辑闭环

- 我们将大语言模型的 $\hat{\psi}_a$ 正式接管了 `environment.py` 里的 `step()` 奖励。
- 在前向选择动作时，修复了一个困扰 `FAISS` 和 MacOS OpenMP 在并发查询时的底层 C++ **段错误崩溃 (Segmentation Fault 139)**：将原来无脑向全库搜索的逻辑，升级为利用 DSAG（动态加权发生器）提供的精锐子集 `I_sim_list` 进行纯 PyTorch 原生矩阵乘法打分。

---

## 📊 Phase 4: 多维评价体系与学术延展设计 (Evaluation System Protocol)

为了能在顶级学术会议（如 WWW, SIGIR, KDD）上呈现无可挑剔的数据对抗，我们对标了业界最新的公平性质检报告（如 FairLRM 2026），搭建了史无前例苛刻的第四阶段评测矩阵：

### 1. 通用推荐质检 (Accuracy & Quality)

- **`NDCG@K` / `Precision@K`**：用于保障我们的模型在“破茧”之后，仍然没有脱离用户的真实兴趣区间底线。
- **`Recall@K`**：衡量正例相关产品的捕获能力。

### 2. 去偏与反茧房防线 (Debiasing & Diversity)

- **`LtC` (长尾覆盖率 / Long-tail Coverage)**：直观展现系统向外输出触达的“冷僻知识”容量占比。
- **`MRMC` (均值排名校准误差)**：通过严格的 KL 散度验证推荐列表中冷热门物品被扭曲的程度，值越低，即推荐机制越公平。
- **`Novelty@K` 及 `ILS`**：辅助验证推荐列表中物品的单体信息量以及内部异质性阻断“同质长尾套娃”。

### 3. 长线留存价值 (Long-term Value)

- 依靠强化学习机制，记录用户的自然消亡步长与累积连续回报，证明本模型在多轮决策闭环上拥有长期黏性的商业转化价值。

### 4. 极致严厉的学术延展实验 (Academic Rigor)

- 💣 **绝对冷启动子集（Extreme Cold-Start Subset）隔离评测**：专门切割出训练集交互次数 $\le 5$ 的无人区领地。让主程序单独汇报 `NDCG_Cold@K` 和 `Recall_Cold@K` 供我们证明大模型的“降维打击能力”。
- 📈 **帕累托前沿权衡图 (Pareto Frontier Trade-off)**：计划利用 Accuracy-Diversity 二维矩阵绘制模型上限推翻图。
- ⚖️ **统计学显著性证明 (Significance Test, p<0.01)**：模型架构内置了跑分多次平均与校验防线，彻底杜绝幸存者偏差带来的偶然性实验得分。
```