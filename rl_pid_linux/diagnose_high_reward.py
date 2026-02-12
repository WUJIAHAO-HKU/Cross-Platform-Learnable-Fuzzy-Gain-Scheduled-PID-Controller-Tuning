import yaml
import numpy as np
from envs.franka_env import FrankaRLPIDEnv

# 加载配置
with open('configs/stage1_final.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建环境
env = FrankaRLPIDEnv(config, gui=False)
obs, _ = env.reset()

# 记录详细信息
errors = []
rewards = []
sparse_rewards = []
track_rewards = []

for step in range(10000):
    action = np.zeros(7, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    errors.append(info['err_norm'])
    rewards.append(reward)
    sparse_rewards.append(info.get('r_sparse', 0))
    track_rewards.append(info.get('r_track', 0))
    
    # 打印前10步和每1000步
    if step < 10 or step % 1000 == 0:
        print(f"Step {step:5d}: err={info['err_norm']:.4f}, reward={reward:7.2f}, r_sparse={info.get('r_sparse', 0):5.1f}, r_track={info.get('r_track', 0):7.2f}")

env.close()

print("\n" + "=" * 70)
print("10000步完整Episode奖励分析")
print("=" * 70)
print(f"总奖励: {sum(rewards):.1f}")
print(f"平均单步奖励: {np.mean(rewards):.2f}")
print(f"\n误差统计:")
print(f"  平均: {np.mean(errors):.4f}弧度 ({np.mean(errors)*57.3:.1f}度)")
print(f"  中位: {np.median(errors):.4f}弧度")
print(f"  最大: {np.max(errors):.4f}弧度")
print(f"  最小: {np.min(errors):.4f}弧度")
print(f"\n稀疏奖励统计:")
print(f"  总计: {sum(sparse_rewards):.1f}")
print(f"  平均: {np.mean(sparse_rewards):.2f}/step")
print(f"  触发+100次数: {sum(1 for r in sparse_rewards if r == 100)}")
print(f"  触发+50次数: {sum(1 for r in sparse_rewards if r == 50)}")
print(f"  触发+20次数: {sum(1 for r in sparse_rewards if r == 20)}")
print(f"  触发+5次数: {sum(1 for r in sparse_rewards if r == 5)}")
print(f"\n密集奖励统计:")
print(f"  总track惩罚: {sum(track_rewards):.1f}")
print(f"  平均track惩罚: {np.mean(track_rewards):.2f}/step")

# 误差分布
bins = [0, 0.35, 0.45, 0.55, 0.65, 1.0]
labels = ['<0.35(+100)', '0.35-0.45(+50)', '0.45-0.55(+20)', '0.55-0.65(+5)', '>0.65(0)']
hist, _ = np.histogram(errors, bins=bins)
print("\n误差分布:")
for label, count in zip(labels, hist):
    pct = count / len(errors) * 100
    print(f"  {label}: {count:5d}步 ({pct:5.1f}%)")
