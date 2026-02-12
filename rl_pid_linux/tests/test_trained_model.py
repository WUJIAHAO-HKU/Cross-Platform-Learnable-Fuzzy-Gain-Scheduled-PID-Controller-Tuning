import yaml
import numpy as np
from stable_baselines3 import PPO
from envs.franka_env import FrankaRLPIDEnv

# 加载配置
with open('configs/stage1_final.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 加载训练好的模型
model = PPO.load("./logs/rl_pid_ppo_final")

# 创建环境
env = FrankaRLPIDEnv(config, gui=False)

# 测试5个episode
errors = []
rewards = []

for ep in range(5):
    obs, _ = env.reset()
    ep_reward = 0
    ep_errors = []
    
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_errors.append(info['err_norm'])
        
        if terminated or truncated:
            break
    
    avg_err = np.mean(ep_errors)
    errors.append(avg_err)
    rewards.append(ep_reward)
    print(f"Episode {ep+1}: 平均误差={avg_err:.4f}弧度, 总奖励={ep_reward:.1f}")

print(f"\n总体: 平均误差={np.mean(errors):.4f}弧度")
env.close()
