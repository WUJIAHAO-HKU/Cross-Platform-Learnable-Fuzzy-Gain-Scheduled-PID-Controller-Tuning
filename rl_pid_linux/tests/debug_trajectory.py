import yaml
import numpy as np
from envs.franka_env import FrankaRLPIDEnv

# 加载配置
with open('configs/stage1_final.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建环境
env = FrankaRLPIDEnv(config, gui=False)

# 重置环境
obs, _ = env.reset()

# 获取初始状态
q_init = env._get_robot_state()[0]
qref_init, qd_ref_init = env.traj_gen.get_reference(0)

print("=" * 60)
print("初始状态诊断")
print("=" * 60)
print(f"机器人初始位置 q_init:")
print(q_init)
print(f"\n参考轨迹初始位置 qref_init:")
print(qref_init)
print(f"\n初始误差 (qref - q):")
initial_error = qref_init - q_init
print(initial_error)
print(f"\n初始误差范数: {np.linalg.norm(initial_error):.4f} 弧度 ({np.linalg.norm(initial_error)*57.3:.1f}度)")

# 测试10步
print("\n" + "=" * 60)
print("前10步跟踪情况")
print("=" * 60)

for step in range(10):
    action = np.zeros(7, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: err_norm={info['err_norm']:.4f}, reward={reward:.2f}")

env.close()
