# 下一步：业界环境 + 跨算法

## 一、业界主流MARL环境

### 1. Overcooked (最推荐)
- 两个厨师在狭小厨房里合作做菜（切菜→上菜→洗碗）
- 天然的"隐藏角色"：如果厨师A专长切菜、B专长上菜，但这个信息是隐藏的
- 已有标准benchmark: `Human_Aware_RL/overcooked_ai`
- **改造成本**: 中等。需要把角色信息设为可隐藏/可见的开关
- **论文叙述**: "我们进一步在标准MARL benchmark Overcooked上验证了路径积分+κ的诊断框架"

### 2. SMAC (StarCraft Multi-Agent Challenge)
- 星际争霸微操：多个友方单位合作打击敌方
- 天然的部分可观测：每个单位只能看到视野范围内的敌人
- 已有标准benchmark: `oxwhirl/smac` (已停更) / `oxwhirl/smacv2`
- **改造成本**: 高。环境重，训练慢，需要集成PyMARL或EPyMARL
- **论文叙述**: "在SMAC的部分可观测设定下，路径积分+κ区分了稳定推进和停滞不前"

### 3. Hanabi
- 合作卡牌游戏：你不能看自己的牌，只能看别人的
- 天然的部分可观测 + 关系推理（谁提示了谁的什么牌）
- 已有benchmark: `deepmind/hanabi-learning-environment`
- **改造成本**: 高。环境特殊(自博弈不直接适用)，需要特殊算法
- **论文叙述**: "在Hanabi中，路径积分+κ检测到信息缺失导致的梯度消失"

### 4. MPE (Multi-Particle Environment)
- 简单物理模拟：小球合作/竞争完成任务
- 天然的关系变量：追捕-逃跑、合作搬运
- 已有标准benchmark: `openai/multiagent-particle-envs`
- **改造成本**: 低。环境轻量，观察空间简单
- **劣势**: 太简单，没有"隐藏关系"的自然设定，需要人工构造

### 5. Melting Pot (DeepMind)
- 社会困境 + 多智能体博弈
- 天然的社会关系变量（合作or背叛是隐藏的）
- 已有标准benchmark: `google-deepmind/meltingpot`
- **改造成本**: 中等。环境重，但关系变量天然存在

### 6. Google Research Football
- 11v11足球模拟
- 天然的"位置/角色"隐藏（队友的战术角色不可见）
- 已有标准benchmark: `google-research/football`
- **改造成本**: 高。环境重，训练极慢

---

## 二、推荐优先级

| 优先级 | 环境 | 理由 |
|--------|------|------|
| ★★★ | Overcooked | 天然隐藏角色，改造中等，论文认可度高 |
| ★★☆ | MPE | 改造最简单，但需人工构造隐藏关系 |
| ★☆☆ | SMAC | 天然部分可观测，但训练慢且环境已停更 |
| ☆☆☆ | Hanabi/MeltingPot | 极好但改造成本太高 |

---

## 三、跨算法训练方案

### 方案A: DQN + 动作掩码 (最推荐)
- 这是和PPO最不同的算法家族（value-based vs policy-based）
- SB3提供DQN，需要添加动作掩码支持（设置非法动作Q值为-1e8）
- 实现量: ~100行wrapper代码
- 每种子训练时间: ~2h（和PPO相近）
- **优势**: 如果DQN下S>O≈T>D的单调性仍然成立 → 效应鲁棒性极强

### 方案B: A2C (最简单)
- SB3内置A2C，直接用，不需要额外wrapper
- PPO的超集（PPO = A2C + clip + importance sampling）
- 实现量: 改一行代码（PPO→A2C）
- **劣势**: A2C和PPO太接近，审稿人可能认为"这不算跨算法"

### 方案C: TRPO (最严格)
- 自然策略梯度，PPO的前身
- 没有标准SB3实现，需要自己写或找第三方库
- 实现量: ~500行
- **劣势**: 实现成本高，且TRPO近两年使用率低

### 方案D: Discrete SAC
- 熵正则化的off-policy算法
- SB3不直接支持discrete SAC，但可以改
- 实现量: ~300行
- **优势**: SAC的熵最大化和我们κ=0的发现天然相关

---

## 四、建议执行顺序

```
现在 → 整理文档发给朋友 (已完成)
     → 决定跨算法方案 (DQN or A2C)
     → 实现 + 训练SINGLE/DYNAMIC各3种子
     → 路径积分 + κ测量验证S>D
     → 决定是否加Overcooked验证
     → 整合进论文
```
