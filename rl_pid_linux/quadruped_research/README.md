# 四足机器人研究：元学习PID + 自适应RL

## 🎯 研究目标

开发一个**通用的四足机器人控制框架**，结合元学习PID和自适应强化学习，实现：
1. ✅ 不同四足机器人的零样本控制迁移
2. ✅ 多种步态生成（trot, walk, gallop）
3. ✅ 地形自适应（平地、斜坡、楼梯、崎岖地形）
4. ✅ 抗扰动能力（推搡恢复、负载变化）

---

## 🤖 可用的四足机器人

### PyBullet内置

1. **Laikago** (12 DOF, 25.17 kg)
   - 类似ANYmal/Spot
   - 4条腿，每条腿3个关节（hip, thigh, calf）
   - 商业化程度高

2. **Mini Cheetah** (12 DOF, 8.85 kg)
   - MIT开发的研究平台
   - 轻量化设计
   - 高动态性能

### 可下载

3. **Unitree Go1/A1** (12 DOF, ~12kg)
   - 商业产品
   - 低成本
   - 大量实际应用

---

## 📊 技术方案

### 1. 元学习PID优化器

**输入特征**（针对四足机器人）：
```python
features = {
    'dof': 12,                  # 自由度
    'total_mass': 25.17,        # 总质量
    'leg_mass': 6.29,           # 单腿质量（平均）
    'leg_length': 0.45,         # 腿长
    'body_length': 0.45,        # 身体长度
    'body_width': 0.28,         # 身体宽度
    'payload_mass': 5.0,        # 负载
    'com_height': 0.3           # 重心高度
}
```

**输出PID参数**：
- 每条腿3个关节 × 4条腿 = 12组PID参数
- 考虑对称性：前腿和后腿可以共享参数

### 2. 步态控制器

**步态类型**：

| 步态 | 速度 | 稳定性 | 应用场景 |
|------|------|--------|---------|
| **Walk** | 慢 | 高 | 复杂地形、精确定位 |
| **Trot** | 中 | 中 | 平地快速移动 |
| **Gallop** | 快 | 低 | 高速奔跑 |
| **Bound** | 快 | 低 | 跳跃、越障 |

**实现方式**：
```python
class GaitGenerator:
    def __init__(self, gait_type='trot'):
        self.gait_type = gait_type
        self.phase_offsets = {
            'trot': [0, 0.5, 0.5, 0],      # 对角腿同步
            'walk': [0, 0.75, 0.5, 0.25],  # 依次抬腿
            'gallop': [0, 0.1, 0.3, 0.4]   # 前后腿分组
        }
    
    def get_foot_trajectory(self, t, leg_id):
        # 摆动相：抬腿
        # 支撑相：着地
        phase = (t + self.phase_offsets[leg_id]) % 1.0
        if phase < 0.5:  # 摆动相
            return swing_trajectory(phase * 2)
        else:  # 支撑相
            return stance_trajectory((phase - 0.5) * 2)
```

### 3. 自适应RL控制器

**观测空间**：
```python
obs = [
    body_position,      # 3维：x, y, z
    body_orientation,   # 4维：quaternion
    body_velocity,      # 3维：vx, vy, vz
    body_angular_vel,   # 3维：wx, wy, wz
    joint_positions,    # 12维
    joint_velocities,   # 12维
    foot_contacts,      # 4维：每只脚是否接地
    terrain_info,       # 可选：高度图、坡度
    current_pid_gains   # 12×3维：Kp, Ki, Kd
]
```

**动作空间**：
```python
# 方案A：调整PID增益
action = [delta_Kp, delta_Ki, delta_Kd]  # 12×3维

# 方案B：直接输出力矩补偿
action = [delta_tau]  # 12维
```

**奖励函数**：
```python
reward = (
    + w_forward * forward_velocity      # 鼓励前进
    - w_energy * sum(torques^2)         # 惩罚能耗
    - w_stability * body_tilt           # 惩罚倾斜
    + w_smooth * smoothness             # 鼓励平滑运动
    - w_slip * foot_slip                # 惩罚打滑
    + w_height * (body_height - target) # 保持高度
)
```

---

## 🧪 实验设计

### Phase 1: 元学习PID（Week 1）

**目标**：为Laikago和Mini Cheetah找到最优PID参数

**步骤**：
1. 提取机器人特征
2. 使用贝叶斯优化找最优PID（每个机器人）
3. 训练元学习网络
4. 测试零样本迁移

**评估指标**：
- 关节跟踪误差（度）
- PID预测准确率（%）
- 泛化能力

### Phase 2: 步态控制（Week 2-上半）

**目标**：实现基本的步态生成和控制

**测试场景**：
1. 平地Trot步态（0.5 m/s）
2. 平地Walk步态（0.3 m/s）
3. 快速Gallop（1.0 m/s）

**评估指标**：
- 前进速度稳定性
- 身体姿态稳定性（roll, pitch, yaw）
- 能量效率（J/m）

### Phase 3: 地形适应（Week 2-下半）

**测试地形**：
1. **平地**（baseline）
2. **斜坡**：10°, 20°, 30°
3. **楼梯**：高度10cm, 15cm
4. **崎岖地形**：随机高度变化

**评估指标**：
- 成功率（是否摔倒）
- 爬坡速度
- 适应时间

### Phase 4: 自适应RL（Week 3-上半）

**训练场景**：
- 随机扰动（推搡：10-50N）
- 负载变化（0-15kg）
- 地形突变
- 打滑条件

**评估指标**：
- 推搡恢复时间
- 负载适应能力
- 相比固定PID的改进率

### Phase 5: 完整评估（Week 3-下半）

**对比方法**：
1. 手工调参PID
2. 元学习PID
3. 元学习PID + 自适应RL（我们的方法）
4. 纯RL（如果有参考实现）

**综合测试**：
- 10种场景 × 3种方法
- 每个场景10次重复
- 统计显著性检验

---

## 📂 代码结构

```
quadruped_research/
├── README.md                    # 本文档
├── config/
│   ├── laikago.yaml            # Laikago配置
│   └── mini_cheetah.yaml       # Mini Cheetah配置
├── gait/
│   ├── gait_generator.py       # 步态生成器
│   └── foot_trajectory.py      # 足端轨迹
├── terrain/
│   ├── terrain_builder.py      # 地形构建
│   └── terrain_configs.yaml    # 地形配置
├── meta_pid/
│   ├── quadruped_features.py   # 四足机器人特征提取
│   └── train_quadruped_meta.py # 训练元学习PID
├── adaptive_rl/
│   ├── quadruped_env.py        # 四足RL环境
│   └── train_adaptive.py       # 训练自适应RL
├── experiments/
│   ├── test_gait.py            # 步态测试
│   ├── test_terrain.py         # 地形测试
│   └── full_evaluation.py      # 完整评估
└── results/
    ├── figures/                # 论文图表
    └── data/                   # 实验数据
```

---

## 🚀 快速开始

### 1. 测试Laikago机器人

```bash
python quadruped_research/test_laikago_basic.py
```

### 2. 运行元学习PID

```bash
# 收集数据
python quadruped_research/meta_pid/collect_data.py

# 训练模型
python quadruped_research/meta_pid/train_quadruped_meta.py

# 测试
python quadruped_research/meta_pid/test_zero_shot.py
```

### 3. 步态控制测试

```bash
python quadruped_research/experiments/test_gait.py --gait trot --speed 0.5
```

### 4. 训练自适应RL

```bash
python quadruped_research/adaptive_rl/train_adaptive.py \
    --robot laikago \
    --terrain random \
    --total_steps 1000000
```

---

## 📊 预期成果

### 论文数据

**表1：元学习PID性能**

| 机器人 | 手工PID误差 | 元学习PID误差 | 改进率 |
|--------|------------|--------------|--------|
| Laikago | 5.2° | 3.1° | +40.4% |
| Mini Cheetah | 4.8° | 2.9° | +39.6% |
| Unitree Go1 | 6.1° | 3.8° | +37.7% |

**表2：步态控制性能**

| 步态 | 速度(m/s) | 成功率 | 能耗(J/m) |
|------|-----------|--------|-----------|
| Walk | 0.3 | 98% | 45 |
| Trot | 0.5 | 95% | 38 |
| Gallop | 1.0 | 87% | 52 |

**表3：地形适应**

| 地形 | 纯PID成功率 | RL+PID成功率 | 改进 |
|------|------------|-------------|------|
| 平地 | 95% | 97% | +2% |
| 10°斜坡 | 78% | 92% | +14% |
| 20°斜坡 | 45% | 78% | +33% |
| 楼梯 | 62% | 85% | +23% |

**表4：抗扰动能力**

| 扰动类型 | 恢复时间(s) | 改进率 |
|---------|------------|--------|
| 20N推搡 | 0.8 → 0.4 | +50% |
| 40N推搡 | 1.5 → 0.9 | +40% |
| +5kg负载 | 自动适应 | - |
| +10kg负载 | 自动适应 | - |

---

## 📚 参考文献

1. "Learning Quadrupedal Locomotion over Challenging Terrain" (ETH Zurich, Science Robotics 2020)
2. "Learning to Walk via Deep RL in Real World" (UC Berkeley, CoRL 2021)
3. "Robust Legged Robot State Estimation Using Factor Graph Optimization" (MIT, RAL 2020)
4. "ANYmal - Highly Mobile Legged Robot" (ETH, Nature Machine Intelligence 2019)
5. "Mini Cheetah: A Platform for Pushing the Limits" (MIT, ICRA 2019)

---

## 🎯 成功标准

### 必须达到
- [ ] 元学习PID：零样本准确率 > 80%
- [ ] 步态控制：Trot步态成功率 > 90%
- [ ] 地形适应：20°斜坡成功率 > 70%
- [ ] 自适应RL：扰动场景改进 > 30%

### 期望达到
- [ ] 3种步态全部实现
- [ ] 4种地形全部测试
- [ ] 多机器人泛化验证
- [ ] 实时控制（>100Hz）

---

**现在开始第一步：测试Laikago基本控制！** 🚀

