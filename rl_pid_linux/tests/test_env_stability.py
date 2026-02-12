import sys
import yaml
import numpy as np
from envs.franka_env import FrankaRLPIDEnv

# 加载配置
with open('configs/stage1_final.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 禁用RL
config['rl_params']['delta_scale_max'] = 0.0

print("测试RL训练环境中的纯PID稳定性...")
env = FrankaRLPIDEnv(config, gui=False)

for episode in range(5):
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(1000):
        action = np.zeros(7)  # 纯PID（零RL动作）
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step == 0:
            print(f"\nEpisode {episode+1}, Step 1:")
            print(f"  Tracking error: {info['tracking_error']:.4f}")
            print(f"  Reward: {reward:.2f}")
        
        if terminated or truncated:
            print(f"  ❌ Episode {episode+1} terminated at step {step+1}")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Final tracking error: {info['tracking_error']:.4f}")
            if 'q' in info:
                print(f"  Final q_max: {np.max(np.abs(info['q'])):.2f}")
            break
    else:
        print(f"  ✅ Episode {episode+1} completed 1000 steps")
        print(f"  Total reward: {total_reward:.2f}")

env.close()
