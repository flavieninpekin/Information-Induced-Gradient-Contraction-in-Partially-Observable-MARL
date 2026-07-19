# 致学术指导人：论文方向决策参考

> 项目：510K 纸牌游戏 × RL × IRL
> 当前状态：论文草稿已出，故事方向待定
> AAAI-27 截稿：2026年7月28日（剩余约2周）

---

## 一、项目现状速览

**做了什么**：
- 实现了 510K（四人类斗地主纸牌游戏）全部规则 + Gymnasium 接口
- 在四种规则模式下训练了 PPO self-play 策略（22 个独立种子）
- 用 7 维可解释特征提取行为特征，用路径积分（path integral）衡量训练过程的稳定性
- OBVIOUS 消融模式 + 跨算法验证（MAPPO）+ 特征消融

**当前论文草稿**：`paper.tex`（已编译为 `paper.pdf`，7页 + 参考文献）

---

## 二、故事演变脉络（来回变的原因）

### 阶段 1：IRL as Diagnostic（AAAI27_STORY.md）
> 标题：Rules as Interventions: Recovering Implicit Reward Functions Across Cooperative Structures
- 叙事：同一策略平移到不同规则下 → IRL 恢复隐式奖励 → 发现奖励权重偏移
- 加 LLM 预测规则→奖励映射
- **问题**：后来发现"环境动力学 vs 策略适应"的二分法是 trivial 的——无论隐式还是显式奖励，变化一定发生，没有信息量

### 阶段 2：Rule→Reward Prediction（new_direction.md + paths_comparison.md）
> 标题：Learning to Predict Implicit Rewards from Game Rules
- 叙事：从"分析已有差异"转向"预测规则变化后的奖励偏移"
- 需要额外训练 9 个新策略（~18h 计算），时间紧张
- 与阶段 1 并存为两条备选路径

### 阶段 3（当前）：Cooperation Constrains Chaos（paper.tex）
> 标题：Cooperation Constrains Chaos: Cooperative Structures Stabilize Self-Play Training Trajectories
- **核心发现**：合作约束越强 → 训练路径越短 → 训练越稳定（反直觉）
- 路径积分分析 + OBVIOUS 消融 + 统计检验
- **放弃 IRL，转向训练动态分析**
- 当前论文草稿的完整故事

---

## 三、当前论文核心故事

**一句话**：合作约束在 self-play 训练中起隐式正则化作用——共享奖励结构降低了梯度方差，产生更稳定的训练轨迹。

**数据支撑**：

| 模式 | n | 路径长度 | 曲率(中位) | 步长方差 |
|------|---|---------|-----------|---------|
| SINGLE | 5 | 0.456 ± 0.071 | 7.3x | 6.07e-4 |
| STATIC | 5 | 0.329 ± 0.086 | 5.9x | 3.05e-4 |
| OBVIOUS | 8 | 0.328 ± 0.062 | 5.0x | 2.29e-4 |
| DYNAMIC | 4 | 0.293 ± 0.049 | 7.7x | 2.79e-4 |

- Spearman r = -0.62, p = 0.019, Cohen's d (S vs D) = 2.67
- OBVIOUS ≈ STATIC（Δ=0.001）→ 已知队友信息是因果主因
- 7/7 特征删除条件下单调趋势不变

**机制解释**：梯度方差降低级联——合作约束越强 → 队友间梯度信号正相关 → 平均梯度方差降低 → 更稳定的训练轨迹。

---

## 四、尚未解决的问题

### 4.1 故事方向·根本选择

| 方向 | 内容 | 需要额外工作 | 目标会议 |
|------|------|------------|---------|
| **A. 当前方向**：合作稳定训练（反直觉实证） | 论文已写完，结构需修 | 2天修结构+扩充Related Work | AAAI / AAMAS |
| **B. 回到 IRL+LLM 预测** | 规则→奖励的LLM预测 | 额外3-5规则变体训练+LLM实验 | AAAI（更强故事） |
| **C. 混合**：在现有论文中加入 IRL 分析 | 既讲训练动态，也讲IRL奖励偏移 | 需补充IRL分析 | AAAI / AAMAS |

### 4.2 论文草稿的硬伤（review 指出）

1. **结构**：Results 和 Discussion 部分重叠，需重新组织
2. **Related Work**：太薄（仅覆盖10篇文献，需扩充至20+篇）
3. **跨算法验证**：MAPPO 初步结果（5种子）有噪声，方向反转
4. **机制解释**：梯度方差解释停留在定性层面，缺乏直接测量
5. **泛化性**：仅在 510K 单一游戏上验证

### 4.3 时间线

- AAAI-27：7月28日截稿（约2周）
- AAMAS-27：约10月截稿（有更多时间）
- CoG 2027：春末截稿（保底选择）

---

## 五、随附材料清单

| 文件 | 说明 |
|------|------|
| `paper.pdf` | 当前论文草稿（7页+参考文献） |
| `paper.tex` | LaTeX源码 |
| `RESULTS.md` | 完整实验结果汇总（含MAPPO验证和特征消融） |
| `PROJECT_SUMMARY.md` | 项目总览 + 五个备选问题方向 |
| `AAAI27_STORY.md` | 早期故事框架（IRL + LLM 方向） |
| `new_direction.md` | 对"trivial"问题的自我修正 |
| `paths_comparison.md` | 两条论文路径对比分析 |
| `review/*.txt` | 两轮审稿意见反馈 |
| `510k_rules.md` | 510K 游戏规则文档 |

---

## 六、核心问题
1. **当前"合作稳定训练"这个故事——有说服力吗？** 还是应该换回 IRL/LLM 方向？
2. 如果走当前方向，AAAI 够格还是应该降级投 AAMAS？
3. 您对"路径积分"这个方法论贡献怎么看？够作为方法贡献卖点吗？
4. 如果需要加实验——加什么最有说服力（更多种子？更多规则变体？核IRL？）
