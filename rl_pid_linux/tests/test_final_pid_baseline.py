import yaml
import numpy as np
from envs.franka_env import FrankaRLPIDEnv

# 加载配置
with open('configs/stage1_final.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建环境
env = FrankaRLPIDEnv(config, gui=False)
obs, _ = env.reset()

# 记录误差
errors = []

# 运行1000步纯PID
for step in range(1000):
    action = np.zeros(7, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    errors.append(info['err_norm'])
    
    if step % 200 == 0:
        print(f"Step {step:4d}: err={info['err_norm']:.4f}, reward={reward:.2f}")

env.close()

print("\n" + "=" * 60)
print("PID基线性能（Kp=50, circle轨迹speed=0.1, amp=0.1）")
print("=" * 60)
print(f"平均误差: {np.mean(errors):.4f}弧度 ({np.mean(errors)*57.3:.1f}度)")
print(f"中位误差: {np.median(errors):.4f}弧度 ({np.median(errors)*57.3:.1f}度)")
print(f"最大误差: {np.max(errors):.4f}弧度 ({np.max(errors)*57.3:.1f}度)")

# 误差分布
bins = [0, 0.10, 0.15, 0.20, 0.25, 0.30, 1.0]
labels = ['<0.10', '0.10-0.15', '0.15-0.20', '0.20-0.25', '0.25-0.30', '>0.30']
hist, _ = np.histogram(errors, bins=bins)
print("\n误差分布:")
for label, count in zip(labels, hist):
    pct = count / len(errors) * 100
    print(f"  {label}弧度: {count:4d} ({pct:5.1f}%)")
