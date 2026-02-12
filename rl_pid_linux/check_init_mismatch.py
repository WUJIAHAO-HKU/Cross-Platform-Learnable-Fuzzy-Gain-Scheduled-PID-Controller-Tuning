import yaml
import numpy as np
import pybullet as p
import pybullet_data

# 加载配置
with open('configs/stage1_final.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 初始化PyBullet
physics_client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 加载Franka
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# ⭐ 获取默认初始位置（PyBullet重置后的位置）
default_joint_positions = []
for i in range(7):
    joint_info = p.getJointState(robot_id, i)
    default_joint_positions.append(joint_info[0])

default_q = np.array(default_joint_positions)

# ⭐ 轨迹的基准位置（来自trajectory_gen.py）
traj_base = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])

# ⭐ 检查初始轨迹位置（t=0）
t = 0
speed = config['trajectory']['speed']
amplitude = config['trajectory']['amplitude']
qref_t0 = traj_base.copy()
qref_t0[0] = traj_base[0] + amplitude * np.sin(speed * t)
qref_t0[1] = traj_base[1] + amplitude * np.cos(speed * t)

print("=" * 70)
print("初始化位置不匹配检查")
print("=" * 70)
print(f"\nPyBullet默认初始位置 (default_q):")
print(default_q)
print(f"\n轨迹基准位置 (traj_base):")
print(traj_base)
print(f"\n轨迹t=0位置 (qref_t0):")
print(qref_t0)

print(f"\n差异1 (default_q - traj_base):")
diff1 = default_q - traj_base
print(diff1)
print(f"范数: {np.linalg.norm(diff1):.4f}弧度 ({np.linalg.norm(diff1)*57.3:.1f}度)")

print(f"\n差异2 (default_q - qref_t0):")
diff2 = default_q - qref_t0
print(diff2)
print(f"范数: {np.linalg.norm(diff2):.4f}弧度 ({np.linalg.norm(diff2)*57.3:.1f}度)")

p.disconnect()
