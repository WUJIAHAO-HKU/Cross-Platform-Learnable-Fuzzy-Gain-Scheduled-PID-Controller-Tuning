"""
评估训练好的RL+PID模型
"""

import yaml
import numpy as np
import argparse
from stable_baselines3 import PPO
from envs.franka_env import FrankaRLPIDEnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 解析命令行参数
parser = argparse.ArgumentParser(description='评估RL+PID模型')
parser.add_argument('--gui', action='store_true', help='启用PyBullet可视化')
parser.add_argument('--steps', type=int, default=10000, help='评估步数')
parser.add_argument('--slow', action='store_true', help='慢速播放（GUI模式）')
parser.add_argument('--model', type=str, default='./logs/rl_pid_ppo_final', help='模型路径（不含.zip）')
parser.add_argument('--config', type=str, default='configs/stage1_final.yaml', help='配置文件路径')
args = parser.parse_args()

# 加载配置
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# 加载训练好的模型
model_path = args.model.replace('.zip', '')  # 移除.zip后缀（如果有）
model = PPO.load(model_path)
print(f"✅ 模型加载成功: {model_path}")
print(f"✅ 配置加载成功: {args.config}")

if args.gui:
    print("\n🎬 启动可视化模式...")
    print("   可视化窗口将显示机械臂运动")
    if args.slow:
        print("   慢速播放模式已启用")
    print("   关闭窗口可停止评估\n")

# 创建环境
env = FrankaRLPIDEnv(config, gui=args.gui)

import time

print("\n" + "=" * 70)
print("测试1：纯PID基线（action=0）")
print("=" * 70)

if args.gui:
    print("🎬 正在可视化纯PID控制...")

obs, _ = env.reset()
total_reward_pid = 0
errors_pid = []
times = []

for step in range(args.steps):
    action = np.zeros(7, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward_pid += reward
    errors_pid.append(info['err_norm'])
    times.append(step * 0.001)
    
    # 慢速播放
    if args.gui and args.slow:
        time.sleep(0.01)  # 10倍慢速
    
    if step % 2000 == 0:
        print(f"Step {step:5d}: err={info['err_norm']:.4f}, reward={reward:6.2f}")

print(f"\n纯PID 总奖励: {total_reward_pid:.1f}")
print(f"纯PID 平均误差: {np.mean(errors_pid):.4f}弧度 ({np.mean(errors_pid)*57.3:.1f}度)")
print(f"纯PID 中位误差: {np.median(errors_pid):.4f}弧度")
print(f"纯PID 最大误差: {np.max(errors_pid):.4f}弧度")

print("\n" + "=" * 70)
print("测试2：RL+PID（使用训练的策略）")
print("=" * 70)

if args.gui:
    print("🎬 正在可视化RL+PID控制...")
    print("   观察机械臂如何通过RL补偿改善跟踪性能\n")

obs, _ = env.reset()
total_reward_rl = 0
errors_rl = []
actions_rl = []
delta_taus = []

for step in range(args.steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward_rl += reward
    errors_rl.append(info['err_norm'])
    actions_rl.append(action.copy())
    delta_taus.append(info.get('delta_tau', np.zeros(7)))
    
    # 慢速播放
    if args.gui and args.slow:
        time.sleep(0.01)  # 10倍慢速
    
    if step % 2000 == 0:
        print(f"Step {step:5d}: err={info['err_norm']:.4f}, reward={reward:6.2f}, action_norm={np.linalg.norm(action):.4f}")

print(f"\nRL+PID 总奖励: {total_reward_rl:.1f}")
print(f"RL+PID 平均误差: {np.mean(errors_rl):.4f}弧度 ({np.mean(errors_rl)*57.3:.1f}度)")
print(f"RL+PID 中位误差: {np.median(errors_rl):.4f}弧度")
print(f"RL+PID 最大误差: {np.max(errors_rl):.4f}弧度")

print("\n" + "=" * 70)
print("性能对比")
print("=" * 70)

reward_improvement = total_reward_rl - total_reward_pid
error_reduction = np.mean(errors_pid) - np.mean(errors_rl)
percent_improvement = (reward_improvement / abs(total_reward_pid)) * 100

print(f"奖励改善: {reward_improvement:+.1f} ({percent_improvement:+.2f}%)")
print(f"误差降低: {error_reduction:.4f}弧度 ({error_reduction*57.3:.2f}度)")
print(f"误差改善率: {(error_reduction / np.mean(errors_pid) * 100):.2f}%")

actions_rl = np.array(actions_rl)
delta_taus = np.array(delta_taus)
print(f"\nRL补偿统计:")
print(f"  平均action范数: {np.mean(np.linalg.norm(actions_rl, axis=1)):.4f}")
print(f"  平均delta_tau范数: {np.mean(np.linalg.norm(delta_taus, axis=1)):.4f} Nm")
print(f"  最大delta_tau范数: {np.max(np.linalg.norm(delta_taus, axis=1)):.4f} Nm")

env.close()

# 绘制对比图
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 误差对比
axes[0].plot(times, errors_pid, 'b-', label=f'纯PID (平均={np.mean(errors_pid):.4f})', alpha=0.7)
axes[0].plot(times, errors_rl, 'r-', label=f'RL+PID (平均={np.mean(errors_rl):.4f})', alpha=0.7)
axes[0].set_xlabel('时间 (s)', fontsize=12)
axes[0].set_ylabel('跟踪误差 (弧度)', fontsize=12)
axes[0].set_title('跟踪误差对比', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# RL补偿力矩
delta_tau_norms = np.linalg.norm(delta_taus, axis=1)
axes[1].plot(times, delta_tau_norms, 'g-', label=f'RL补偿力矩范数 (平均={np.mean(delta_tau_norms):.3f} Nm)')
axes[1].set_xlabel('时间 (s)', fontsize=12)
axes[1].set_ylabel('补偿力矩范数 (Nm)', fontsize=12)
axes[1].set_title('RL补偿力矩', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
print(f"\n✅ 对比图已保存至: evaluation_results.png")

print("\n" + "=" * 70)
print("评估完成")
print("=" * 70)

