# AAAI-27 七页故事框架

## 论文标题（备选）

1. Rules as Interventions: Recovering Implicit Reward Functions Across Cooperative Structures in a Multi-Agent Card Game
2. Beyond Rule Changes: How Cooperative Structures Reshape Implicit Rewards in Multi-Agent Systems
3. The Same Agent, Different Rules: IRL Reveals How Game Rules Reweight Implicit Preferences

## 核心叙事线

> **一句话**：同一个RL策略，在三种规则（单人/静态团队/红A动态团队）下产生不同轨迹；IRL恢复出不同的隐式奖励权重；LLM能部分预测这种偏移。

> **Why this matters**：游戏规则改变的不只是"行为约束"，它改变的是智能体"看起来在追求什么"——这对AI safety中的reward misspecification和mechanism design有直接启示。

## 7页逐页分配

```
┌──────────┬─────────────────────────────────────────────────┐
│ Page 1   │ Introduction + Figure 1 (三种规则模式图解)        │
│          │ Hook: "规则改变行为是表象，改变奖励函数才是本质"     │
├──────────┼─────────────────────────────────────────────────┤
│ Page 2   │ Related Work (1/3页) + Background: 510K (2/3页) │
│          │ 规则介绍只讲3种模式差异，不讲15种牌型细节             │
│          │ Table 1: 特征定义表 (φ₁~φ₅)                      │
├──────────┼─────────────────────────────────────────────────┤
│ Page 3   │ Method (1.5页) + Figure 2 (实验流程图)           │
│          │ 3.1 训练 (1/3页) → 3.2 平移实验设计 (2/3页)       │
│          │ 3.3 MaxEnt IRL (2/3页) → 3.4 LLM设置 (1/3页)    │
├──────────┼─────────────────────────────────────────────────┤
│ Pages 4-5│ Experiments (2页) — 论文核心                     │
│          │ 4.1 训练收敛 (1/4页)                              │
│          │ 4.2 平移行为分析 (1/4页)                          │
│          │ 4.3 IRL权重对比 (1页) — Figure 3 (雷达图/柱状图)   │
│          │ 4.4 不同checkpoint的权重演化 (1/4页)               │
│          │ Table 2: 三种模式的权重对比表                      │
├──────────┼─────────────────────────────────────────────────┤
│ Page 6   │ LLM解释实验 (1页)                                 │
│          │ Figure 4: LLM预测vs实际权重对比图                  │
│          │ 分析：LLM在哪些维度预测准、哪些维度偏差大            │
├──────────┼─────────────────────────────────────────────────┤
│ Page 7   │ Discussion (1/2页) + Conclusion (1/2页)          │
│          │ AI safety启示 + 局限性 + 未来工作                  │
├──────────┼─────────────────────────────────────────────────┤
│ 参考     │ 不限页                                           │
│ 附录     │ 实验细节（不计入正文章节）                          │
└──────────┴─────────────────────────────────────────────────┘
```

## 三张核心图

### Figure 1: 三种规则模式（示意图，~1/4页）
- 左：4个独立玩家（SINGLE）
- 中：2v2固定组队（STATIC）
- 右：红A决定组队（DYNAMIC）
- 视觉上：SINGLE=4个独立圆圈 → STATIC=两组圆圈连线 → DYNAMIC=圆圈中标注红A

### Figure 2: 实验管线图（~1/3页）
```
Train π in SINGLE → Freeze π → 
  ├─ π in SINGLE → trajectories → IRL → w_single
  ├─ π in STATIC → trajectories → IRL → w_static
  └─ π in DYNAMIC→ trajectories → IRL → w_dynamic
                            ↓
                LLM predicts weight shifts
```

### Figure 3: 奖励权重对比（~1/2页，核心图）
- 多组雷达图或分组柱状图
- X轴：5个特征维度
- Y轴：归一化权重
- 三条线/三组柱：SINGLE / STATIC / DYNAMIC

## 五维特征定义 (φ₁~φ₅)

| φ | 特征名 | 定义 | 直觉 |
|---|--------|------|------|
| φ₁ | 吃分强度 | 当前局累计510K得分 | 越高表示越主动吃分 |
| φ₂ | 出牌消耗 | 已出牌数 / 总手牌数 | 代表出牌积极程度 |
| φ₃ | 压制倾向 | 出炸弹/510K/大牌的频率 | 进攻性指标 |
| φ₄ | 跟牌vs引牌比 | 非引牌时的主动出牌率 | 独立性vs跟随性 |
| φ₅ | 手牌健康度 | 剩余手牌中高分牌比例 | 反映对牌力的关注 |

## 预期结果（hypothesis）

1. **SINGLE模式**：φ₁（吃分）和φ₃（压制）权重最高 → 自利型
2. **STATIC模式**：φ₄（跟牌vs引牌）权重上升 → 开始出现配合信号
3. **DYNAMIC模式**：φ₂（出牌消耗）体现红A推断行为和团队配合 → 出现全新权重结构

## LLM实验设计

**Input to LLM:**
```
游戏中 5-10-K 有三种规则：
1. 单人竞争：各玩各的，先出完牌的获胜
2. 静态合作：固定2v2组队（0&2 vs 1&3），一队两人全出完即结束
3. 动态合作：持红A的玩家自动组队（隐藏信息），需通过出牌推断队友

我们训练了一个RL策略（单人模式），然后把它分别平移到三种规则下。
IRL恢复出了该策略在每种规则下的隐式奖励权重（5个维度的权重向量）。
请预测：从规则1→2→3，各维度的权重会如何变化？为什么？
```

**Compare**: LLM predicted weight shift vs actual IRL recovered shift.

## Figure 4: LLM预测结果对比
- 热力图或散点图
- X轴: 实际权重变化 Δw
- Y轴: LLM预测权重变化 Δw_pred
- 对角线为完美预测

## 消融实验

在论文中加入一个sub-analysis：使用不同训练阶段的checkpoint（early/mid/late），看平移后的IRL权重差异是否随策略成熟度增大。

## 论文竞争力评估

| 维度 | 水平 | 说明 |
|------|------|------|
| 问题新颖性 | ★★★★☆ | 规则干预+IRL+策略平移，尚未见类似工作 |
| 实验设计 | ★★★★☆ | 控制变量干净（策略固定，规则变化） |
| 结果显著性 | ★★★☆☆ | 取决于实际权重差异是否显著（待验证） |
| 方法深度 | ★★★☆☆ | MaxEnt IRL + 线性特征是基础方法 |
| LLM部分 | ★★★★☆ | 规则干预预测是比较新颖的应用 |
| 游戏选择 | ★★★☆☆ | 510K小众但新鲜 |
| **整体** | **★★★★☆** | **强AAMAS，弱NeurIPS，AAAI有希望** |

## 截稿倒计时（关键日期）

- 6/17: AAAI-27 OpenReview开放注册
- 6/24: 开放投稿
- 7/21: 摘要截稿（UTC-12）
- **7/28: 全文截稿（UTC-12）** ← 死线
