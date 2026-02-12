"""
可视化机械臂运动 - 对比纯PID和RL+PID
"""

import yaml
import numpy as np
import argparse
import pybullet as p
from stable_baselines3 import PPO
from envs.franka_env import FrankaRLPIDEnv
import time

parser = argparse.ArgumentParser(description='可视化机械臂运动')
parser.add_argument('--mode', choices=['pid', 'rl', 'compare'], default='compare',
                    help='可视化模式: pid(纯PID), rl(RL+PID), compare(对比)')
parser.add_argument('--steps', type=int, default=3000, help='运行步数')
parser.add_argument('--speed', type=float, default=1.0, help='播放速度倍率（1.0=正常，0.5=慢速，2.0=快速）')
args = parser.parse_args()

# 加载配置
with open('configs/stage1_final.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("\n" + "=" * 70)
print("🎬 Franka Panda 机械臂运动可视化")
print("=" * 70)
print(f"\n模式: {args.mode}")
print(f"步数: {args.steps}")
print(f"速度: {args.speed}x")
print("\n操作提示:")
print("  - 鼠标左键拖动：旋转视角")
print("  - 鼠标右键拖动：平移视角")
print("  - 鼠标滚轮：缩放")
print("  - Ctrl+C：停止运行")
print("=" * 70 + "\n")

if args.mode == 'compare':
    print("📊 对比模式：将先运行纯PID，然后运行RL+PID")
    print("    观察两者的跟踪性能差异\n")

def run_controller(env, model=None, name="控制器", steps=3000, speed=1.0):
    """运行控制器并显示统计"""
    print(f"\n🎬 正在运行: {name}")
    print("-" * 70)
    
    obs, _ = env.reset()
    
    # 配置相机视角
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.4]
    )
    
    errors = []
    rewards = []
    actions_norm = []
    
    sleep_time = 0.001 / speed  # 根据速度倍率调整sleep时间
    
    for step in range(steps):
        if model is None:
            # 纯PID
            action = np.zeros(7, dtype=np.float32)
        else:
            # RL+PID
            action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        errors.append(info['err_norm'])
        rewards.append(reward)
        if model is not None:
            actions_norm.append(np.linalg.norm(action))
        
        # 控制播放速度
        time.sleep(sleep_time)
        
        # 每500步显示一次状态
        if (step + 1) % 500 == 0:
            avg_err = np.mean(errors[max(0, step-499):step+1])
            print(f"  Step {step+1:4d}/{steps}: 平均误差={avg_err:.4f}弧度, 即时奖励={reward:6.2f}")
    
    # 显示统计
    print("\n📊 统计结果:")
    print(f"  平均误差: {np.mean(errors):.4f}弧度 ({np.mean(errors)*57.3:.2f}度)")
    print(f"  中位误差: {np.median(errors):.4f}弧度")
    print(f"  最大误差: {np.max(errors):.4f}弧度")
    print(f"  最小误差: {np.min(errors):.4f}弧度")
    print(f"  总奖励: {sum(rewards):.1f}")
    
    if model is not None and len(actions_norm) > 0:
        print(f"\n  RL补偿:")
        print(f"    平均action范数: {np.mean(actions_norm):.4f}")
        print(f"    最大action范数: {np.max(actions_norm):.4f}")
    
    return errors, rewards

# 创建环境（GUI模式）
env = FrankaRLPIDEnv(config, gui=True)

try:
    if args.mode == 'pid':
        # 只运行纯PID
        run_controller(env, model=None, name="纯PID控制", 
                      steps=args.steps, speed=args.speed)
    
    elif args.mode == 'rl':
        # 只运行RL+PID
        model = PPO.load("./logs/rl_pid_ppo_final")
        print("✅ RL模型加载成功")
        run_controller(env, model=model, name="RL+PID控制", 
                      steps=args.steps, speed=args.speed)
    
    elif args.mode == 'compare':
        # 对比模式
        print("\n" + "=" * 70)
        print("第1部分：纯PID控制")
        print("=" * 70)
        errors_pid, rewards_pid = run_controller(
            env, model=None, name="纯PID控制", 
            steps=args.steps, speed=args.speed
        )
        
        print("\n" + "=" * 70)
        print("第2部分：RL+PID控制")
        print("=" * 70)
        model = PPO.load("./logs/rl_pid_ppo_final")
        print("✅ RL模型加载成功")
        errors_rl, rewards_rl = run_controller(
            env, model=model, name="RL+PID控制", 
            steps=args.steps, speed=args.speed
        )
        
        # 对比结果
        print("\n" + "=" * 70)
        print("📊 对比结果")
        print("=" * 70)
        
        error_reduction = np.mean(errors_pid) - np.mean(errors_rl)
        reward_improvement = sum(rewards_rl) - sum(rewards_pid)
        
        print(f"\n  纯PID平均误差:   {np.mean(errors_pid):.4f}弧度")
        print(f"  RL+PID平均误差:  {np.mean(errors_rl):.4f}弧度")
        print(f"  误差降低:        {error_reduction:.4f}弧度 ({error_reduction*57.3:.2f}度)")
        print(f"  误差改善率:      {(error_reduction / np.mean(errors_pid) * 100):.2f}%")
        
        print(f"\n  纯PID总奖励:     {sum(rewards_pid):.1f}")
        print(f"  RL+PID总奖励:    {sum(rewards_rl):.1f}")
        print(f"  奖励改善:        {reward_improvement:+.1f} ({(reward_improvement / abs(sum(rewards_pid)) * 100):+.2f}%)")
        
        if error_reduction > 0:
            print(f"\n  ✅ RL+PID相比纯PID性能提升 {(error_reduction / np.mean(errors_pid) * 100):.2f}%")
        else:
            print(f"\n  ⚠️  注意：RL+PID未显示性能提升")

except KeyboardInterrupt:
    print("\n\n⚠️  用户中断")

finally:
    env.close()
    print("\n✅ 可视化完成")
    print("=" * 70 + "\n")

