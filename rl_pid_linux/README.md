# RL+PID Linux实现

> **状态**: ✅ 环境已配置，代码已生成，可以开始训练
> **目标**: 使用PyBullet训练RL+PID策略，完成论文实验

---

## 🚀 快速开始

### 1. 激活环境
```bash
source ~/rl_robot_env/bin/activate
cd ~/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/rl_pid_linux
```

### 2. 运行完整系统测试（30秒）
```bash
python tests/test_full_system.py
```

**预期输出**：6个✅全部通过

### 3. 开始训练（阶段1：保守配置）
```bash
# 阶段1：delta_scale_max=2.0（2-4小时）
python training/train_ddpg.py --config configs/stage1_small.yaml --name rl_pid_stage1

# 监控训练（另开一个终端）
tensorboard --logdir=./logs/tensorboard/
```

---

## 📁 项目结构

```
rl_pid_linux/
├── controllers/          # 控制器
│   ├── pid_controller.py         # PID基线
│   └── rl_pid_hybrid.py          # RL+PID混合控制器（渐进式策略）
├── envs/                 # 仿真环境
│   ├── franka_env.py              # PyBullet环境（Gymnasium接口）
│   └── trajectory_gen.py          # 轨迹生成器
├── training/             # 训练脚本
│   └── train_ddpg.py              # DDPG训练主程序
├── configs/              # 配置文件
│   ├── stage1_small.yaml          # 阶段1：delta_scale_max=2.0
│   └── stage2_medium.yaml         # 阶段2：delta_scale_max=5.0
├── tests/                # 测试脚本
│   └── test_full_system.py        # 完整系统测试
├── logs/                 # 训练日志（自动创建）
└── models/               # 训练好的模型（自动创建）
```

---

## 🎯 训练策略（渐进式）

### 阶段1：小补偿（当前）⭐
```bash
python training/train_ddpg.py --config configs/stage1_small.yaml
```
- **Delta Scale Max**: 2.0
- **目标**: 验证系统稳定性
- **时间**: 2-4小时（500k steps）
- **成功标准**: 奖励上升，跟踪误差下降，不发散

### 阶段2：中等补偿
```bash
python training/train_ddpg.py --config configs/stage2_medium.yaml
```
- **Delta Scale Max**: 5.0
- **前提**: 阶段1训练稳定
- **时间**: 4-6小时（1M steps）
- **成功标准**: RMSE降低>30% vs 纯PID

---

## 📊 监控训练

### TensorBoard
```bash
tensorboard --logdir=./logs/tensorboard/
```
在浏览器打开: http://localhost:6006

**关键指标**：
- `rollout/ep_rew_mean`: 平均回合奖励（应该上升）
- `train/actor_loss`: Actor损失
- `train/critic_loss`: Critic损失

### 日志文件
- 训练日志: `logs/tensorboard/`
- 评估日志: `logs/eval/`
- 模型检查点: `logs/models/checkpoints/`
- 最佳模型: `logs/models/best/`

---

## 🔧 配置说明

### 关键参数（configs/stage1_small.yaml）

```yaml
# PID参数
pid_params:
  Kp: [50, 50, 50, 50, 20, 10, 10]  # 比例增益
  Ki: [0.5, 0.5, 0.5, 0.5, 0.2, 0.1, 0.1]  # 积分增益
  Kd: [5, 5, 5, 5, 2, 1, 1]  # 微分增益

# RL补偿参数
rl_params:
  delta_scale_max: 2.0  # ⭐ 最大补偿系数
  warmup_disable_steps: 100  # 前100步纯PID
  warmup_ramp_steps: 500  # 500步渐进增加
  
  # 奖励权重
  w_track: 20.0  # 跟踪奖励（平衡版，不是100！）
  w_vel: 0.001  # 速度惩罚
  w_action: 0.0001  # 动作惩罚
```

---

## ⚠️ 常见问题

### Q1: 训练很慢
**A**: 正常。阶段1需要2-4小时。可以降低`total_timesteps`到100k进行快速测试。

### Q2: 奖励一直是负数
**A**: 正常。跟踪误差惩罚导致。关注趋势（应该上升）而不是绝对值。

### Q3: 出现NaN或发散
**A**: 检查：
1. `delta_scale_max`是否太大（阶段1应该≤2.0）
2. 轨迹是否太激进（降低`speed`）
3. PID增益是否合理

### Q4: 如何恢复训练
**A**: 
```python
# 在train_ddpg.py中添加
model = DDPG.load("logs/models/checkpoints/rl_pid_stage1_500000_steps.zip")
model.set_env(train_env)
model.learn(total_timesteps=500000)  # 继续训练
```

---

## 📈 下一步

### 完成阶段1后：

1. **评估模型**
   ```bash
   python evaluation/evaluate_model.py --model logs/models/rl_pid_stage1_final.zip
   ```

2. **对比纯PID**
   ```bash
   python evaluation/compare_with_pid.py
   ```

3. **开始阶段2**（如果阶段1稳定）
   ```bash
   python training/train_ddpg.py --config configs/stage2_medium.yaml
   ```

4. **多场景测试**（阶段2完成后）
   - 25种场景
   - Monte Carlo 100次
   - 统计分析

---

## 📚 参考文档

- 详细计划: `../LINUX_IMPLEMENTATION_ROADMAP.md`
- 论文策略: `../PUBLICATION_STRATEGY.md`
- MATLAB参考: `../MATLAB_Implementation/PROJECT_STATUS_FOR_LINUX.md`

---

## ✅ 成功标准

### 阶段1（最小目标）
- ✅ 训练完成不崩溃
- ✅ 奖励曲线上升
- ✅ 跟踪误差下降
- ✅ 系统稳定（无发散）

### 阶段2（论文目标）
- ✅ RMSE降低>40% vs 纯PID
- ✅ 5个核心场景测试通过
- ✅ 统计显著性p<0.05

---

**准备好了吗？运行第一个测试：**
```bash
python tests/test_full_system.py
```

