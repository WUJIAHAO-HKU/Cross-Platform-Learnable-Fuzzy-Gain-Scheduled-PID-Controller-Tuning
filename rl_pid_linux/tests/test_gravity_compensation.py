"""
测试重力补偿效果
对比有无重力补偿的PID性能
"""

import yaml
import numpy as np
from envs.franka_env import FrankaRLPIDEnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("🧪 重力补偿效果测试")
print("=" * 70)

# 加载配置
with open('configs/stage1_optimized.yaml', 'r') as f:
    config = yaml.safe_load(f)

# ================================================================================
# 测试1：无重力补偿的PID
# ================================================================================
print("\n" + "=" * 70)
print("测试1：PID控制（无重力补偿）")
print("=" * 70)

config_no_grav = config.copy()
config_no_grav['pid_params'] = config['pid_params'].copy()
config_no_grav['pid_params']['enable_gravity_compensation'] = False

env_no_grav = FrankaRLPIDEnv(config_no_grav, gui=False)
obs, _ = env_no_grav.reset()

errors_no_grav = []
rewards_no_grav = []
total_reward_no_grav = 0

for step in range(1000):
    action = np.zeros(7, dtype=np.float32)
    obs, reward, terminated, truncated, info = env_no_grav.step(action)
    errors_no_grav.append(info['err_norm'])
    rewards_no_grav.append(reward)
    total_reward_no_grav += reward
    
    if step % 200 == 0:
        print(f"  Step {step:4d}: err={info['err_norm']:.4f}, reward={reward:6.2f}")

print(f"\n无重力补偿PID:")
print(f"  平均误差: {np.mean(errors_no_grav):.4f}弧度 ({np.mean(errors_no_grav)*57.3:.2f}度)")
print(f"  中位误差: {np.median(errors_no_grav):.4f}弧度")
print(f"  最大误差: {np.max(errors_no_grav):.4f}弧度")
print(f"  最小误差: {np.min(errors_no_grav):.4f}弧度")
print(f"  总奖励: {total_reward_no_grav:.1f}")

env_no_grav.close()

# ================================================================================
# 测试2：有重力补偿的PID
# ================================================================================
print("\n" + "=" * 70)
print("测试2：PID控制（有重力补偿）")
print("=" * 70)

config_with_grav = config.copy()
config_with_grav['pid_params'] = config['pid_params'].copy()
config_with_grav['pid_params']['enable_gravity_compensation'] = True

env_with_grav = FrankaRLPIDEnv(config_with_grav, gui=False)
obs, _ = env_with_grav.reset()

errors_with_grav = []
rewards_with_grav = []
total_reward_with_grav = 0

for step in range(1000):
    action = np.zeros(7, dtype=np.float32)
    obs, reward, terminated, truncated, info = env_with_grav.step(action)
    errors_with_grav.append(info['err_norm'])
    rewards_with_grav.append(reward)
    total_reward_with_grav += reward
    
    if step % 200 == 0:
        print(f"  Step {step:4d}: err={info['err_norm']:.4f}, reward={reward:6.2f}")

print(f"\n有重力补偿PID:")
print(f"  平均误差: {np.mean(errors_with_grav):.4f}弧度 ({np.mean(errors_with_grav)*57.3:.2f}度)")
print(f"  中位误差: {np.median(errors_with_grav):.4f}弧度")
print(f"  最大误差: {np.max(errors_with_grav):.4f}弧度")
print(f"  最小误差: {np.min(errors_with_grav):.4f}弧度")
print(f"  总奖励: {total_reward_with_grav:.1f}")

env_with_grav.close()

# ================================================================================
# 对比结果
# ================================================================================
print("\n" + "=" * 70)
print("📊 对比结果")
print("=" * 70)

error_reduction = np.mean(errors_no_grav) - np.mean(errors_with_grav)
error_reduction_pct = (error_reduction / np.mean(errors_no_grav)) * 100
reward_improvement = total_reward_with_grav - total_reward_no_grav

print(f"\n误差改善:")
print(f"  无重力补偿: {np.mean(errors_no_grav):.4f}弧度 ({np.mean(errors_no_grav)*57.3:.2f}度)")
print(f"  有重力补偿: {np.mean(errors_with_grav):.4f}弧度 ({np.mean(errors_with_grav)*57.3:.2f}度)")
print(f"  误差降低: {error_reduction:.4f}弧度 ({error_reduction*57.3:.2f}度)")
print(f"  改善率: {error_reduction_pct:.2f}%")

print(f"\n奖励改善:")
print(f"  无重力补偿: {total_reward_no_grav:.1f}")
print(f"  有重力补偿: {total_reward_with_grav:.1f}")
print(f"  改善: {reward_improvement:+.1f}")

if error_reduction > 0:
    print(f"\n✅ 重力补偿有效！误差降低 {error_reduction_pct:.2f}%")
else:
    print(f"\n⚠️  重力补偿效果不明显")

# ================================================================================
# 绘制对比图
# ================================================================================
print("\n" + "=" * 70)
print("📊 生成对比图")
print("=" * 70)

times = np.arange(1000) * 0.001

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 误差对比
axes[0].plot(times, errors_no_grav, 'b-', 
             label=f'无重力补偿 (平均={np.mean(errors_no_grav):.4f})', 
             alpha=0.7, linewidth=1.5)
axes[0].plot(times, errors_with_grav, 'r-', 
             label=f'有重力补偿 (平均={np.mean(errors_with_grav):.4f})', 
             alpha=0.7, linewidth=1.5)
axes[0].set_xlabel('时间 (s)', fontsize=12)
axes[0].set_ylabel('跟踪误差 (弧度)', fontsize=12)
axes[0].set_title(f'重力补偿效果对比 - 误差降低{error_reduction_pct:.1f}%', 
                  fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 奖励对比
axes[1].plot(times, np.cumsum(rewards_no_grav), 'b-', 
             label=f'无重力补偿 (总计={total_reward_no_grav:.1f})', 
             linewidth=2)
axes[1].plot(times, np.cumsum(rewards_with_grav), 'r-', 
             label=f'有重力补偿 (总计={total_reward_with_grav:.1f})', 
             linewidth=2)
axes[1].set_xlabel('时间 (s)', fontsize=12)
axes[1].set_ylabel('累积奖励', fontsize=12)
axes[1].set_title('累积奖励对比', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gravity_compensation_comparison.png', dpi=150, bbox_inches='tight')
print("✅ 对比图已保存至: gravity_compensation_comparison.png")

print("\n" + "=" * 70)
print("✅ 测试完成")
print("=" * 70)

