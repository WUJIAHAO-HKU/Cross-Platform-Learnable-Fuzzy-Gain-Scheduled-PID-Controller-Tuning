import yaml
import numpy as np
import matplotlib.pyplot as plt
from envs.franka_env import FrankaRLPIDEnv

# 加载配置
with open('configs/stage1_final.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建环境
env = FrankaRLPIDEnv(config, gui=False)
obs, _ = env.reset()

# 记录误差
errors = []
rewards = []
times = []

# 运行1000步
for step in range(1000):
    action = np.zeros(7, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    errors.append(info['err_norm'])
    rewards.append(reward)
    times.append(step * 0.001)  # dt=0.001
    
    if step % 100 == 0:
        print(f"Step {step:4d}: err={info['err_norm']:.4f}, reward={reward:.2f}")

env.close()

# 统计
print("\n" + "=" * 60)
print("误差统计")
print("=" * 60)
print(f"初始误差: {errors[0]:.4f}弧度 ({errors[0]*57.3:.1f}度)")
print(f"最大误差: {max(errors):.4f}弧度 ({max(errors)*57.3:.1f}度) at step {errors.index(max(errors))}")
print(f"平均误差: {np.mean(errors):.4f}弧度 ({np.mean(errors)*57.3:.1f}度)")
print(f"最终误差: {errors[-1]:.4f}弧度 ({errors[-1]*57.3:.1f}度)")
print(f"总奖励: {sum(rewards):.1f}")
print(f"平均奖励: {np.mean(rewards):.2f}")

# 分段统计
print("\n分段统计:")
for i, label in enumerate(['0-200步', '200-400步', '400-600步', '600-800步', '800-1000步']):
    start = i * 200
    end = (i + 1) * 200
    seg_err = np.mean(errors[start:end])
    print(f"{label}: 平均误差={seg_err:.4f}弧度 ({seg_err*57.3:.1f}度)")
