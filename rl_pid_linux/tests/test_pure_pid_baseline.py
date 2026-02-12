import yaml
import numpy as np
from envs.franka_env import FrankaRLPIDEnv

# 加载配置
with open('configs/stage1_final.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建环境
env = FrankaRLPIDEnv(config, gui=False)

# 测试5个episode，纯PID（action=0）
errors = []
rewards = []

for ep in range(5):
    obs, _ = env.reset()
    ep_reward = 0
    ep_errors = []
    
    for step in range(1000):
        # ⭐ 纯PID：action=0
        action = np.zeros(7, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_errors.append(info['err_norm'])
        
        if terminated or truncated:
            break
    
    avg_err = np.mean(ep_errors)
    max_err = np.max(ep_errors)
    errors.append(avg_err)
    rewards.append(ep_reward)
    print(f"Episode {ep+1}: 平均误差={avg_err:.4f}, 最大误差={max_err:.4f}, 总奖励={ep_reward:.1f}")

print(f"\n总体: 平均误差={np.mean(errors):.4f}弧度 ({np.mean(errors)*57.3:.1f}度)")
env.close()
