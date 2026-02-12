#!/usr/bin/env python3
"""
Laikago四足机器人 - 最终稳定版本
基于官方URDF和实际物理参数
"""

import pybullet as p
import pybullet_data
import time
import numpy as np


class LaikagoRobot:
    """Laikago四足机器人控制器"""
    
    # 通过实验验证的稳定站立姿态
    # 深蹲姿态：腿部弯曲较多，重心较低，稳定性最好
    INIT_MOTOR_ANGLES = np.array([
        0.0, 1.0, -2.0,   # FR (前右): abduction, hip, knee
        0.0, 1.0, -2.0,   # FL (前左)
        0.0, 1.0, -2.0,   # RR (后右)
        0.0, 1.0, -2.0    # RL (后左)
    ])
    
    # 关节顺序（根据URDF）
    JOINT_NAMES = [
        "FR_hip_motor_2_chassis_joint",
        "FR_upper_leg_2_hip_motor_joint",
        "FR_lower_leg_2_upper_leg_joint",
        "FL_hip_motor_2_chassis_joint",
        "FL_upper_leg_2_hip_motor_joint",
        "FL_lower_leg_2_upper_leg_joint",
        "RR_hip_motor_2_chassis_joint",
        "RR_upper_leg_2_hip_motor_joint",
        "RR_lower_leg_2_upper_leg_joint",
        "RL_hip_motor_2_chassis_joint",
        "RL_upper_leg_2_hip_motor_joint",
        "RL_lower_leg_2_upper_leg_joint",
    ]
    
    # 每条腿的关节索引
    LEG_INDICES = {
        'FR': [0, 1, 2],
        'FL': [3, 4, 5],
        'RR': [6, 7, 8],
        'RL': [9, 10, 11]
    }
    
    def __init__(self, gui=True, start_height=0.5):
        """
        初始化Laikago
        
        Args:
            gui: 是否显示GUI
            start_height: 初始高度
        """
        # 连接PyBullet
        if gui:
            self.client = p.connect(p.GUI)
            # 设置相机视角
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-20,
                cameraTargetPosition=[0, 0, 0.3],
                physicsClientId=self.client
            )
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(0.001, physicsClientId=self.client)  # 1ms
        
        # 加载地面
        self.plane_id = p.loadURDF(
            "plane.urdf",
            physicsClientId=self.client
        )
        
        # 加载Laikago
        start_pos = [0, 0, start_height]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(
            "laikago/laikago.urdf",
            start_pos,
            start_orn,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.client
        )
        
        # 获取关节信息
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        
        # 找到可控关节
        self.motor_id_list = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            if joint_info[2] == p.JOINT_REVOLUTE or joint_info[2] == p.JOINT_PRISMATIC:
                self.motor_id_list.append(i)
        
        # 验证关节数量
        assert len(self.motor_id_list) == 12, f"期望12个关节，实际{len(self.motor_id_list)}个"
        
        print(f"✅ Laikago加载成功")
        print(f"   可控关节数: {len(self.motor_id_list)}")
        print(f"   起始高度: {start_height}m")
    
    def reset(self, motor_angles=None):
        """
        重置机器人姿态
        
        Args:
            motor_angles: 关节角度 (12,)，默认使用INIT_MOTOR_ANGLES
        """
        if motor_angles is None:
            motor_angles = self.INIT_MOTOR_ANGLES
        
        # 重置关节位置（不重置基座，让它保持在初始位置）
        for i, motor_id in enumerate(self.motor_id_list):
            p.resetJointState(
                self.robot_id,
                motor_id,
                motor_angles[i],
                targetVelocity=0,
                physicsClientId=self.client
            )
        
        # 使用PD控制让机器人稳定下来
        # 注意: PyBullet的POSITION_CONTROL模式中，增益范围是0-1
        for step in range(3000):
            self.apply_action(motor_angles, motor_kp=0.5, motor_kd=0.1)
            p.stepSimulation(physicsClientId=self.client)
            
            # 检查是否稳定
            if step % 1000 == 999:
                state = self.get_state()
                height = state['base_pos'][2]
                speed = np.linalg.norm(state['base_vel'])
                if speed < 0.01 and 0.15 < height < 0.30:
                    print(f"✅ 机器人已稳定 (高度={height:.3f}m, 速度={speed:.4f}m/s)")
                    break
        
        print("✅ 机器人已重置")
    
    def apply_action(self, motor_commands, motor_kp=0.5, motor_kd=0.1):
        """
        应用关节控制指令
        
        Args:
            motor_commands: 目标关节角度 (12,)
            motor_kp: PD控制器的P增益 (推荐范围: 0.1-1.0)
            motor_kd: PD控制器的D增益 (推荐范围: 0.01-0.2)
        """
        # 使用POSITION_CONTROL模式
        # PyBullet会自动计算所需力矩
        for i, motor_id in enumerate(self.motor_id_list):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=motor_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=motor_commands[i],
                positionGain=motor_kp,
                velocityGain=motor_kd,
                force=100,  # 使用URDF中的最大力矩
                physicsClientId=self.client
            )
    
    def get_state(self):
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
            self.robot_id, self.motor_id_list, physicsClientId=self.client
        )
        motor_angles = np.array([state[0] for state in joint_states])
        motor_velocities = np.array([state[1] for state in joint_states])
        
        return {
            'base_pos': np.array(base_pos),
            'base_orn': np.array(base_orn),
            'base_vel': np.array(base_vel),
            'base_ang_vel': np.array(base_ang_vel),
            'motor_angles': motor_angles,
            'motor_velocities': motor_velocities,
        }
    
    def close(self):
        """断开连接"""
        p.disconnect(physicsClientId=self.client)


def test_standing_balance(duration=10.0):
    """测试1: 站立平衡"""
    print("\n" + "=" * 80)
    print(f"测试1: 站立平衡 ({duration}秒)")
    print("=" * 80)
    
    robot = LaikagoRobot(gui=True)
    robot.reset()
    
    # 保持站立
    steps = int(duration / 0.001)
    for i in range(steps):
        robot.apply_action(robot.INIT_MOTOR_ANGLES)
        p.stepSimulation(physicsClientId=robot.client)
        time.sleep(0.001)
        
        # 每秒打印状态
        if i % 1000 == 0:
            state = robot.get_state()
            height = state['base_pos'][2]
            vel = np.linalg.norm(state['base_vel'])
            print(f"   t={i/1000:.1f}s: 高度={height:.3f}m, 速度={vel:.3f}m/s")
    
    # 最终评估
    final_state = robot.get_state()
    height = final_state['base_pos'][2]
    pos_xy = final_state['base_pos'][:2]
    vel = np.linalg.norm(final_state['base_vel'])
    
    print(f"\n📊 最终状态:")
    print(f"   高度: {height:.3f}m")
    print(f"   XY位置: ({pos_xy[0]:.3f}, {pos_xy[1]:.3f})m")
    print(f"   速度: {vel:.3f}m/s")
    
    # 判断稳定性
    stable = True
    if 0.18 < height < 0.25:
        print("   ✅ 高度正常 (0.18-0.25m, 深蹲姿态)")
    else:
        print(f"   ❌ 高度异常 (应该0.18-0.25m)")
        stable = False
    
    if vel < 0.05:
        print("   ✅ 速度稳定 (<0.05m/s)")
    else:
        print(f"   ❌ 速度过大 (应该<0.05m/s)")
        stable = False
    
    if np.linalg.norm(pos_xy) < 1.0:
        print(f"   ✅ 位置可接受 (偏移={np.linalg.norm(pos_xy):.2f}m < 1m)")
    else:
        print(f"   ⚠️  位置偏移较大 (偏移={np.linalg.norm(pos_xy):.2f}m)")
        # 不影响稳定性评分
    
    robot.close()
    return stable


def test_simple_trot(duration=10.0, frequency=1.0):
    """测试2: 简单Trot步态"""
    print("\n" + "=" * 80)
    print(f"测试2: 简单Trot步态 ({duration}秒, {frequency}Hz)")
    print("=" * 80)
    
    robot = LaikagoRobot(gui=True)
    robot.reset()
    
    # Trot步态参数（基于深蹲姿态调整）
    stance_angle = 1.0   # 支撑相：保持深蹲角度
    swing_angle = 0.7    # 摆动相：抬腿（角度减小）
    
    t = 0
    dt = 0.001
    steps = int(duration / dt)
    
    for i in range(steps):
        # 计算步态相位 (0-1)
        phase = (t * frequency) % 1.0
        
        # 对角步态: FR+RL一组, FL+RR一组
        if phase < 0.5:
            # FR+RL在支撑相, FL+RR在摆动相
            fr_rl_hip = stance_angle
            fl_rr_hip = swing_angle
        else:
            # FR+RL在摆动相, FL+RR在支撑相
            fr_rl_hip = swing_angle
            fl_rr_hip = stance_angle
        
        # 构造目标角度
        target_angles = np.array([
            0.0, fr_rl_hip, -2.0,  # FR
            0.0, fl_rr_hip, -2.0,  # FL
            0.0, fl_rr_hip, -2.0,  # RR
            0.0, fr_rl_hip, -2.0   # RL
        ])
        
        robot.apply_action(target_angles, motor_kp=0.5, motor_kd=0.1)
        p.stepSimulation(physicsClientId=robot.client)
        time.sleep(dt)
        t += dt
        
        # 每2秒打印状态
        if i % 2000 == 0:
            state = robot.get_state()
            height = state['base_pos'][2]
            pos_x = state['base_pos'][0]
            print(f"   t={t:.1f}s: 高度={height:.3f}m, X位置={pos_x:.3f}m")
    
    # 最终评估
    final_state = robot.get_state()
    height = final_state['base_pos'][2]
    distance = final_state['base_pos'][0]
    lateral = abs(final_state['base_pos'][1])
    
    print(f"\n📊 最终状态:")
    print(f"   高度: {height:.3f}m")
    print(f"   前进距离: {distance:.3f}m")
    print(f"   横向偏移: {lateral:.3f}m")
    
    # 判断步态效果
    if distance > 0.5:
        print("   ✅ 成功前进 (>0.5m)")
    else:
        print("   ⚠️  前进不足")
    
    if 0.15 < height < 0.30:
        print("   ✅ 高度稳定 (0.15-0.30m)")
    else:
        print("   ⚠️  高度异常")
    
    if lateral < 0.5:
        print("   ✅ 横向稳定 (<0.5m)")
    else:
        print("   ⚠️  横向偏移过大")
    
    # 保持显示5秒
    print("\n   保持显示5秒...")
    for _ in range(5000):
        p.stepSimulation(physicsClientId=robot.client)
        time.sleep(0.001)
    
    robot.close()


if __name__ == '__main__':
    print("=" * 80)
    print("Laikago四足机器人 - 最终稳定测试")
    print("=" * 80)
    
    # 测试1: 站立平衡
    standing_ok = test_standing_balance(duration=10.0)
    
    if standing_ok:
        print("\n✅ 站立测试通过！继续步态测试...")
        # 测试2: Trot步态
        test_simple_trot(duration=10.0, frequency=1.0)
        
        print("\n" + "=" * 80)
        print("✅ 所有测试完成！")
        print("=" * 80)
        print("\n🎯 下一步:")
        print("  1. ✅ 基础控制稳定")
        print("  2. 🔄 集成元学习PID优化器")
        print("  3. 🔄 实现完整步态规划器")
        print("  4. 🔄 添加自适应RL控制")
        print("  5. 🔄 鲁棒性测试（扰动、地形）")
    else:
        print("\n" + "=" * 80)
        print("❌ 站立测试失败！")
        print("=" * 80)
        print("\n建议:")
        print("  1. 检查URDF文件是否正确加载")
        print("  2. 尝试调整motor_kp和motor_kd参数")
        print("  3. 检查初始高度设置")

