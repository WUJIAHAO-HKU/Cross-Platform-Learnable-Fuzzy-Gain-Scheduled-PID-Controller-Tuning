#!/usr/bin/env python3
"""
Laikago稳定控制测试 - 使用POSITION_CONTROL模式
这个版本更稳定，适合作为后续开发的基础
"""

import pybullet as p
import pybullet_data
import time
import numpy as np


class LaikagoStable:
    """稳定的Laikago控制器"""
    
    def __init__(self, gui=True):
        """初始化"""
        # 连接到PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.001)  # 1ms时间步
        
        # 加载环境
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        
        # 加载Laikago
        start_pos = [0, 0, 0.5]  # 从较高位置落下，确保接触
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
        self.joint_indices = []
        self.joint_names = []
        
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            if info[2] == p.JOINT_REVOLUTE:  # 只考虑旋转关节
                self.joint_indices.append(i)
                self.joint_names.append(info[1].decode('utf-8'))
        
        self.num_controllable_joints = len(self.joint_indices)
        
        # Laikago的稳定站立姿态（经过验证的）
        # 参考: https://github.com/google-research/motion_imitation
        self.default_pose = np.array([
            0.0, 0.9, -1.8,  # FR (前右)
            0.0, 0.9, -1.8,  # FL (前左)
            0.0, 0.9, -1.8,  # RR (后右)
            0.0, 0.9, -1.8   # RL (后左)
        ])
        
        print(f"✅ Laikago加载完成")
        print(f"   可控关节数: {self.num_controllable_joints}")
        print(f"   关节名称: {self.joint_names}")
    
    def reset_to_standing(self):
        """重置到站立姿态"""
        # 设置关节位置
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(
                self.robot_id,
                joint_idx,
                self.default_pose[i],
                physicsClientId=self.client
            )
        
        # 设置基座位置（稍高一点，让机器人落下）
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [0, 0, 0.45],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.client
        )
        
        # 等待稳定
        for _ in range(500):
            self.set_joint_positions(self.default_pose)
            p.stepSimulation(physicsClientId=self.client)
            time.sleep(0.001)
        
        print("✅ 已重置到站立姿态")
    
    def set_joint_positions(self, target_positions, max_force=25.0):
        """
        使用POSITION_CONTROL设置关节位置
        
        Args:
            target_positions: 目标关节位置 (12,)
            max_force: 最大力矩
        """
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_positions[i],
                force=max_force,
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
            self.robot_id, self.joint_indices, physicsClientId=self.client
        )
        joint_pos = np.array([s[0] for s in joint_states])
        joint_vel = np.array([s[1] for s in joint_states])
        
        return {
            'base_pos': np.array(base_pos),
            'base_orn': np.array(base_orn),
            'base_vel': np.array(base_vel),
            'base_ang_vel': np.array(base_ang_vel),
            'joint_pos': joint_pos,
            'joint_vel': joint_vel
        }
    
    def close(self):
        """关闭连接"""
        p.disconnect(physicsClientId=self.client)


def test_stable_standing():
    """测试1: 稳定站立"""
    print("\n" + "=" * 80)
    print("测试1: 稳定站立 (10秒)")
    print("=" * 80)
    
    robot = LaikagoStable(gui=True)
    robot.reset_to_standing()
    
    # 保持站立10秒
    for i in range(10000):
        robot.set_joint_positions(robot.default_pose)
        p.stepSimulation(physicsClientId=robot.client)
        time.sleep(0.001)
        
        if i % 1000 == 0:
            state = robot.get_state()
            print(f"   t={i/1000:.1f}s: 高度={state['base_pos'][2]:.3f}m, "
                  f"速度={np.linalg.norm(state['base_vel']):.3f}m/s")
    
    final_state = robot.get_state()
    print(f"\n   最终高度: {final_state['base_pos'][2]:.3f}m")
    print(f"   最终位置: x={final_state['base_pos'][0]:.3f}, "
          f"y={final_state['base_pos'][1]:.3f}m")
    
    # 评估
    if 0.25 < final_state['base_pos'][2] < 0.35:
        print("   ✅ 站立稳定！")
        stable = True
    else:
        print("   ⚠️  高度异常")
        stable = False
    
    if np.linalg.norm(final_state['base_vel']) < 0.1:
        print("   ✅ 速度稳定！")
    else:
        print("   ⚠️  速度过大")
        stable = False
    
    robot.close()
    return stable


def test_simple_trot():
    """测试2: 简单Trot步态"""
    print("\n" + "=" * 80)
    print("测试2: 简单Trot步态 (10秒)")
    print("=" * 80)
    
    robot = LaikagoStable(gui=True)
    robot.reset_to_standing()
    
    # Trot步态参数
    frequency = 1.0  # 1Hz
    stance_height = 0.9
    swing_height = 0.6  # 抬腿时大腿关节角度减小
    
    t = 0
    dt = 0.001
    
    for i in range(10000):
        # 计算步态相位
        phase = (t * frequency) % 1.0
        
        # 对角步态: FR+RL一组, FL+RR一组
        if phase < 0.5:
            # FR+RL在支撑相，FL+RR在摆动相
            fr_rl_thigh = stance_height
            fl_rr_thigh = swing_height
        else:
            # FR+RL在摆动相，FL+RR在支撑相
            fr_rl_thigh = swing_height
            fl_rr_thigh = stance_height
        
        # 构造目标位置
        target_pos = np.array([
            0.0, fr_rl_thigh, -1.8,  # FR
            0.0, fl_rr_thigh, -1.8,  # FL
            0.0, fl_rr_thigh, -1.8,  # RR
            0.0, fr_rl_thigh, -1.8   # RL
        ])
        
        robot.set_joint_positions(target_pos)
        p.stepSimulation(physicsClientId=robot.client)
        time.sleep(dt)
        t += dt
        
        if i % 2000 == 0:
            state = robot.get_state()
            print(f"   t={t:.1f}s: 高度={state['base_pos'][2]:.3f}m, "
                  f"x={state['base_pos'][0]:.3f}m")
    
    final_state = robot.get_state()
    print(f"\n   最终高度: {final_state['base_pos'][2]:.3f}m")
    print(f"   前进距离: {final_state['base_pos'][0]:.3f}m")
    print(f"   横向偏移: {abs(final_state['base_pos'][1]):.3f}m")
    
    # 评估
    if final_state['base_pos'][0] > 0.5:
        print("   ✅ 成功前进！")
    else:
        print("   ⚠️  前进不足")
    
    if abs(final_state['base_pos'][1]) < 0.5:
        print("   ✅ 横向稳定！")
    else:
        print("   ⚠️  横向偏移过大")
    
    robot.close()


if __name__ == '__main__':
    print("=" * 80)
    print("Laikago稳定控制测试")
    print("=" * 80)
    
    # 测试1: 站立
    standing_ok = test_stable_standing()
    
    if standing_ok:
        # 测试2: 步态
        test_simple_trot()
        
        print("\n" + "=" * 80)
        print("✅ 所有测试完成！")
        print("=" * 80)
        print("\n下一步:")
        print("  1. 集成元学习PID优化器")
        print("  2. 实现完整的步态规划器")
        print("  3. 添加RL增强控制")
    else:
        print("\n" + "=" * 80)
        print("⚠️  站立测试失败，需要先解决基础稳定性问题")
        print("=" * 80)

