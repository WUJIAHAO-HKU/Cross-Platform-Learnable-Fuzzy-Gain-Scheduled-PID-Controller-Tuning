"""
Laikago四足机器人基础测试

功能：
1. 加载Laikago机器人
2. 测试关节控制
3. 简单步态尝试
4. 为元学习PID提取特征
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

class LaikagoRobot:
    """Laikago机器人控制类"""
    
    def __init__(self, gui=True):
        """
        初始化Laikago机器人
        
        Args:
            gui: 是否显示GUI
        """
        # 连接PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(0.001, physicsClientId=self.client)
        
        # 加载地面
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        
        # 加载Laikago
        start_pos = [0, 0, 0.5]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(
            "laikago/laikago.urdf",
            start_pos,
            start_orientation,
            physicsClientId=self.client
        )
        
        print(f"✅ Laikago机器人已加载 (ID: {self.robot_id})")
        
        # 获取关节信息
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        print(f"   总关节数: {self.num_joints}")
        
        # 分析关节
        self.analyze_joints()
        
        # 设置初始姿态
        self.reset_to_default_pose()
    
    def analyze_joints(self):
        """分析关节结构"""
        print(f"\n📊 关节结构分析:")
        
        self.joint_info = {}
        self.controllable_joints = []
        
        # 四足机器人的腿：FR(前右), FL(前左), RR(后右), RL(后左)
        self.leg_joints = {
            'FR': [],  # Front Right
            'FL': [],  # Front Left  
            'RR': [],  # Rear Right
            'RL': []   # Rear Left
        }
        
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.controllable_joints.append(i)
                self.joint_info[i] = {
                    'name': joint_name,
                    'type': 'Revolute' if joint_type == p.JOINT_REVOLUTE else 'Prismatic',
                    'lower_limit': info[8],
                    'upper_limit': info[9],
                    'max_force': info[10],
                    'max_velocity': info[11]
                }
                
                # 根据名字分配到对应的腿
                for leg_name in ['FR', 'FL', 'RR', 'RL']:
                    if leg_name in joint_name:
                        self.leg_joints[leg_name].append(i)
                        break
        
        print(f"   可控关节数: {len(self.controllable_joints)} (DOF)")
        print(f"\n   各腿关节分布:")
        for leg_name, joints in self.leg_joints.items():
            joint_names = [self.joint_info[j]['name'] for j in joints]
            print(f"     {leg_name}: {len(joints)}个关节")
            for jn in joint_names:
                print(f"        - {jn}")
        
        # 验证对称性
        if len(self.leg_joints['FR']) == len(self.leg_joints['FL']) == \
           len(self.leg_joints['RR']) == len(self.leg_joints['RL']):
            print(f"\n   ✅ 对称性验证通过：每条腿 {len(self.leg_joints['FR'])} 个关节")
        else:
            print(f"\n   ⚠️  腿部关节数不对称")
    
    def reset_to_default_pose(self):
        """重置到默认站立姿态"""
        # Laikago的稳定站立姿态
        # 每条腿3个关节: hip(外展), thigh(大腿), calf(小腿)
        # 参考Laikago的实际站立姿态
        default_angles = {
            'FR': [0.0, 0.67, -1.3],   # 前右
            'FL': [0.0, 0.67, -1.3],   # 前左
            'RR': [0.0, 0.67, -1.3],   # 后右
            'RL': [0.0, 0.67, -1.3]    # 后左
        }
        
        for leg_name, joints in self.leg_joints.items():
            angles = default_angles[leg_name]
            for joint_id, angle in zip(joints, angles):
                p.resetJointState(
                    self.robot_id,
                    joint_id,
                    angle,
                    physicsClientId=self.client
                )
        
        # 启用力矩控制模式并设置高增益PD控制
        # 这样可以在重置后保持姿态
        for joint_id in self.controllable_joints:
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.VELOCITY_CONTROL,
                force=0,
                physicsClientId=self.client
            )
        
        print(f"\n   ✅ 已重置到默认站立姿态")
    
    def get_robot_state(self):
        """获取机器人状态"""
        # 基座状态
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client
        )
        base_vel, base_ang_vel = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.client
        )
        
        # 关节状态
        joint_states = p.getJointStates(
            self.robot_id,
            self.controllable_joints,
            physicsClientId=self.client
        )
        joint_positions = np.array([s[0] for s in joint_states])
        joint_velocities = np.array([s[1] for s in joint_states])
        
        return {
            'base_pos': np.array(base_pos),
            'base_orn': np.array(base_orn),
            'base_vel': np.array(base_vel),
            'base_ang_vel': np.array(base_ang_vel),
            'joint_pos': joint_positions,
            'joint_vel': joint_velocities
        }
    
    def set_joint_torques(self, torques):
        """
        设置关节力矩
        
        Args:
            torques: (12,) 每个关节的力矩
        """
        p.setJointMotorControlArray(
            self.robot_id,
            self.controllable_joints,
            p.TORQUE_CONTROL,
            forces=torques,
            physicsClientId=self.client
        )
    
    def set_joint_positions(self, positions, kp=100, kd=10):
        """
        使用PD控制器设置关节位置
        
        Args:
            positions: (12,) 目标位置
            kp: P增益
            kd: D增益
        """
        p.setJointMotorControlArray(
            self.robot_id,
            self.controllable_joints,
            p.POSITION_CONTROL,
            targetPositions=positions,
            positionGains=[kp] * len(self.controllable_joints),
            velocityGains=[kd] * len(self.controllable_joints),
            physicsClientId=self.client
        )
    
    def extract_features(self):
        """提取机器人特征（用于元学习PID）"""
        # 获取总质量
        total_mass = 0
        for i in range(-1, self.num_joints):
            dynamics = p.getDynamicsInfo(self.robot_id, i, physicsClientId=self.client)
            total_mass += dynamics[0]
        
        # 获取基座尺寸
        base_collision = p.getCollisionShapeData(self.robot_id, -1, physicsClientId=self.client)
        if base_collision:
            body_dimensions = base_collision[0][3]  # half extents
        else:
            body_dimensions = [0.3, 0.15, 0.1]  # 默认值
        
        # 估算腿长（从关节位置）
        leg_length = 0.0
        if self.leg_joints['FR']:
            for joint_id in self.leg_joints['FR']:
                link_state = p.getLinkState(self.robot_id, joint_id, physicsClientId=self.client)
                leg_length += np.linalg.norm(link_state[0])
            leg_length /= len(self.leg_joints['FR'])
        
        features = {
            'dof': len(self.controllable_joints),
            'total_mass': total_mass,
            'body_length': body_dimensions[0] * 2,
            'body_width': body_dimensions[1] * 2,
            'body_height': body_dimensions[2] * 2,
            'leg_length': leg_length,
            'num_legs': len([k for k in self.leg_joints if self.leg_joints[k]]),
            'joints_per_leg': len(self.leg_joints['FR']) if self.leg_joints['FR'] else 0
        }
        
        return features
    
    def close(self):
        """关闭连接"""
        p.disconnect(physicsClientId=self.client)


def test_basic_control():
    """测试基本控制"""
    print("=" * 80)
    print("Laikago四足机器人基础测试")
    print("=" * 80)
    
    # 创建机器人
    robot = LaikagoRobot(gui=True)
    
    # 提取特征
    print("\n📊 机器人特征:")
    features = robot.extract_features()
    for key, value in features.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # 测试1: 站立平衡
    print("\n" + "=" * 80)
    print("测试1: 站立平衡 (5秒)")
    print("=" * 80)
    
    # 使用PD控制保持站立姿态
    default_pos = np.array([0.0, 0.67, -1.3] * 4)
    
    for _ in range(5000):
        # 高增益PD控制保持姿态
        robot.set_joint_positions(default_pos, kp=500, kd=50)
        p.stepSimulation(physicsClientId=robot.client)
        time.sleep(0.001)
    
    state = robot.get_robot_state()
    print(f"   基座高度: {state['base_pos'][2]:.3f} m")
    print(f"   基座速度: {np.linalg.norm(state['base_vel']):.3f} m/s")
    
    if state['base_pos'][2] > 0.25:
        print(f"   ✅ 机器人保持站立")
    else:
        print(f"   ⚠️  机器人可能倒下了")
    
    # 测试2: 关节运动
    print("\n" + "=" * 80)
    print("测试2: 单腿关节运动 (前右腿)")
    print("=" * 80)
    
    # 前右腿做简单运动
    t = 0
    dt = 0.001
    duration = 3.0
    
    while t < duration:
        # 正弦波运动
        target_positions = np.zeros(12)
        # 保持其他腿不动（默认姿态）
        default_pos = [0.0, 0.9, -1.8] * 4
        target_positions[:] = default_pos
        
        # 前右腿的第2个关节（thigh）做正弦运动
        fr_joints = robot.leg_joints['FR']
        if len(fr_joints) > 1:
            thigh_idx = robot.controllable_joints.index(fr_joints[1])
            target_positions[thigh_idx] = 0.9 + 0.3 * np.sin(2 * np.pi * t / 2.0)
        
        robot.set_joint_positions(target_positions)
        p.stepSimulation(physicsClientId=robot.client)
        time.sleep(dt)
        t += dt
    
    print("   ✅ 单腿运动完成")
    
    # 测试3: 简单trot步态
    print("\n" + "=" * 80)
    print("测试3: 简单Trot步态尝试 (5秒)")
    print("=" * 80)
    print("   注意: 这只是关节位置的微小周期变化")
    print("   使用高增益PD控制保持稳定")
    
    t = 0
    while t < 5.0:
        # 简单的对角步态：FR+RL同步，FL+RR同步
        phase = (t % 1.0) / 1.0  # 0-1
        
        # 非常小的抬腿幅度，避免失控
        if phase < 0.5:
            # FR+RL抬起
            fr_rl_lift = 0.1 * np.sin(phase * 2 * np.pi)
            fl_rr_lift = 0.0
        else:
            # FL+RR抬起
            fr_rl_lift = 0.0
            fl_rr_lift = 0.1 * np.sin((phase - 0.5) * 2 * np.pi)
        
        # 构造目标位置（基于稳定站立姿态的微小变化）
        target_positions = np.array([
            # FR (前右): hip, thigh, calf
            0.0, 0.67 + fr_rl_lift, -1.3 - fr_rl_lift * 1.5,
            # FL (前左)
            0.0, 0.67 + fl_rr_lift, -1.3 - fl_rr_lift * 1.5,
            # RR (后右)
            0.0, 0.67 + fl_rr_lift, -1.3 - fl_rr_lift * 1.5,
            # RL (后左)
            0.0, 0.67 + fr_rl_lift, -1.3 - fr_rl_lift * 1.5
        ])
        
        # 使用高增益PD控制
        robot.set_joint_positions(target_positions, kp=500, kd=50)
        p.stepSimulation(physicsClientId=robot.client)
        time.sleep(0.001)
        t += 0.001
    
    final_state = robot.get_robot_state()
    print(f"   最终基座高度: {final_state['base_pos'][2]:.3f} m")
    print(f"   最终位移: x={final_state['base_pos'][0]:.3f}, y={final_state['base_pos'][1]:.3f} m")
    
    if abs(final_state['base_pos'][0]) < 1.0 and abs(final_state['base_pos'][1]) < 1.0:
        print(f"   ✅ 机器人保持稳定（位移<1m）")
    else:
        print(f"   ⚠️  机器人移动过大，可能需要调整参数")
    
    # 保持显示5秒
    print("\n   保持显示5秒...")
    for _ in range(5000):
        p.stepSimulation(physicsClientId=robot.client)
        time.sleep(0.001)
    
    robot.close()
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)
    print("\n下一步:")
    print("  1. 提取特征用于元学习PID")
    print("  2. 实现真正的步态规划器")
    print("  3. 集成PID控制器")


if __name__ == '__main__':
    test_basic_control()

