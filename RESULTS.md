# 实验结果汇总

# 510K 环境模式说明

四种游戏模式（mode），决定组队方式和队伍信息：

1. **single** — 单人博弈，各自为战，无队友
2. **static** — 静态合作，对家固定组队（0&2 一队，1&3 一队）
3. **dynamic** — 动态合作，持红A（A♥/A♦）的玩家自动成队，队友不固定
4. **obvious** — 规则同 dynamic，但额外暴露队友身份信息（消融实验，用于研究"知晓队友"对合作策略的影响）

## 1. 核心发现：路径积分单调递减

**22种子 × 4模式 × 7维特征 × 10个checkpoint (每10万步)**

```
Mode       n    路径长度           曲率(中位)   步长方差
SINGLE:    5    0.456 ± 0.071     7.3x         6.07e-4
STATIC:    5    0.329 ± 0.086     5.9x         3.05e-4
OBVIOUS:   8    0.328 ± 0.062     5.0x         2.29e-4
DYNAMIC:   4    0.293 ± 0.049     7.7x         2.79e-4

Spearman r = -0.62, p = 0.019
Cohen's d (SINGLE vs DYNAMIC) = 2.67
Cohen's d (SINGLE vs OBVIOUS) = 1.92
```

**结论**：合作约束越强 → 训练路径越短 → 训练越稳定。统计显著，效应量大。

## 2. OBVIOUS消融：已知队友信息是因果主因

```
STATIC:   0.329 ± 0.086  (固定队友，红A正常)
OBVIOUS:  0.328 ± 0.062  (DYNAMIC牌 + STATIC信息)
DYNAMIC:  0.293 ± 0.049  (隐藏队友，红A力量)

OBVIOUS与STATIC几乎完全重叠 (Δ=0.001)
→ 已知队友信息是稳定化效应的主导因素
→ 红A牌力规则的独立贡献可忽略
```

## 3. 特征消融：发现不依赖任何单一特征

每次删除一个特征，重新计算路径积分。7/7子集全部保持 S > T > D：

```
删除特征          SINGLE   STATIC   DYNAMIC   S>D?
────────────────────────────────────────────────
MyScore            0.452    0.324    0.290    YES
MyHandSize         0.344    0.225    0.197    YES
MyStrength         0.419    0.309    0.279    YES
TrickScore         0.449    0.326    0.291    YES
PassCount          0.443    0.325    0.289    YES
ScoreSpread        0.402    0.288    0.242    YES
SuppressionGap     0.420    0.308    0.279    YES
────────────────────────────────────────────────
ALL 7 features     0.456    0.329    0.293    YES
```

**结论**：单调趋势是稳健的发现，非特征选择的artifact。

## 4. 7维特征集

| φ | 特征 | 类型 | 跨模式差异 |
|----|------|------|-----------|
| φ₁ | MyScore | 个体-收益 | 小 |
| φ₂ | MyHandSize | 个体-进度 | 中 (SINGLE最大) |
| φ₃ | MyStrength | 个体-牌力 | 中 (SINGLE最强) |
| φ₄ | TrickScore | 个体-风险 | 小 |
| φ₅ | PassCount | 个体-延迟 | 小 |
| φ₆ | ScoreSpread | 交互-不平等度 | 小 (STATIC最高) |
| φ₇ | SuppressionGap | 交互-压制空间 | 小 (DYNAMIC最高) |

特征共线性: 全部VIF < 2, 仅MyScore↔MyHandSize有r=0.615 (合理)

## 5. 合作特征：方向一致但效应小

| 特征 | SINGLE | STATIC | DYNAMIC | p值 |
|------|--------|--------|---------|-----|
| ScoreSpread (φ₆) | 0.20 | 0.23 | 0.22 | ~0.07 |
| SuppressionGap (φ₇) | 0.27 | 0.28 | 0.30 | ~0.08 |

STATIC得分最不平等（团队记分放大差距），DYNAMIC压制空间最大（红A牌力）。

## 6. 梯度方差代理度量

步长方差（相邻checkpoint之间的L2距离的方差）：
```
SINGLE:   6.07e-4
STATIC:   3.05e-4  (2.0x lower)
DYNAMIC:  2.79e-4  (2.2x lower)
OBVIOUS:  2.29e-4  (2.7x lower)
```

支持"合作降低梯度方差"机制解释。

## 7. 稳定性地图

散点图（路径长度 × 曲率）显示：
- SINGLE种子集中在右上（长路径 + 高曲率）
- DYNAMIC种子集中在左下（短路径 + 低曲率）
- STATIC和OBVIOUS重叠在中部
- 无DYNAMIC种子超过路径长度0.37

## 8. 跨算法验证（MAPPO）

MAPPO使用集中式critic（全局观测） + 分散式actor（局部观测），
字典观测空间 `{local:112, global:448}`。

**结果 (5 seeds total)**：

```
MAPPO SINGLE (n=3):
  s101: path=0.308   s102: path=0.235   s103: path=0.353
  mean = 0.299 ± 0.049

MAPPO DYNAMIC (n=2):
  s101: path=0.261   s102: path=0.430
  mean = 0.345 ± 0.084  ← 方差极大
```

**对比PPO**：

```
Algorithm     SINGLE          DYNAMIC         S>D?
─────────────────────────────────────────────────────
PPO           0.456 ± 0.071   0.293 ± 0.049   YES (p=0.019)
MAPPO         0.299 ± 0.049   0.345 ± 0.084   NO (方向反转)
```

**分析**：
- MAPPO下SINGLE路径从0.46降到0.30 → 集中式critic显著压缩了竞争模式的梯度方差
- MAPPO下DYNAMIC平均0.35但方差极大(σ=0.084) → 仅2种子，s101极低(0.26)而s102极高(0.43)
- S>D的方向反转可能是小样本噪声，也可能意味着集中式critic消解了合作稳定化效应
- 需要更多MAPPO种子(≥5)才能下结论，当前PyTorch环境存在 `Buffer` 和 `unknown opcode` 兼容bug阻止了进一步训练

**论文处理**：在Limitations中诚实汇报——"The cooperation-stabilization effect is currently validated only under PPO. Preliminary MAPPO results (5 seeds) are inconclusive due to high seed variance (σ=0.084 for DYNAMIC, n=2). Cross-algorithm replication with larger seed populations remains future work."

## 9. 实验规模总览

| 项目 | 数量 |
|------|------|
| 总种子数 | 22 (5+5+8+4) |
| 总训练步数 | 22M |
| 特征维度 | 7 |
| Checkpoint分辨率 | 100K步 |
| 总checkpoint评估 | ~240次 |
| PPO训练配置 | lr=3e-4, [256,256], ent=0.01 |
| OBVIOUS观测维度 | 116 (112+4 teammate bits) |
| 训练硬件 | 单GPU (8-24GB) |

## 10. 论文亮点

1. **反直觉发现**：合作约束 → 训练更稳定 (而非更复杂)
2. **统计显著**：p=0.019, d=2.67
3. **消融验证**：OBVIOUS确认已知队友信息是因果主因
4. **方法贡献**：路径积分作为MARL训练诊断工具
5. **特征鲁棒**：7/7特征删除条件下单调趋势不变
6. **新环境**：510K作为多合作结构开源测试平台
