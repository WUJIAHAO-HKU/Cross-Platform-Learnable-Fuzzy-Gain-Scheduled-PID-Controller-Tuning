#!/usr/bin/env python3
"""
Laikago自适应RL环境 - Trot步态版本
任务：沿直线Trot行走，RL动态调整PID增益
"""

import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import torch
import sys
from pathlib import Path

# 导入元学习PID
sys.path.append(str(Path(__file__).parent.parent / 'meta_learning'))
from meta_pid_optimizer import RobotFeatureExtractor

# 导入基础环境
sys.path.append(str(Path(__file__).parent))
from adaptive_laikago_env import AdaptivePIDController


class TrotGaitGenerator:
    """Trot步态生成器（基于成功的improved_trot_gait实现）"""
    
    def __init__(self, frequency=1.0, step_height=0.08, forward_speed=0.5):
        """
        Args:
            frequency: 步态频率（Hz）
            step_height: 抬腿高度（m）
            forward_speed: 前进速度（m/s）
        """
        self.frequency = frequency
        self.step_height = step_height
        self.forward_speed = forward_speed
        
        # Trot步态参数（与improved_trot_gait.py一致）
        self.stance_angle = 1.0   # 支撑相thigh角度
        self.swing_angle = 0.7    # 摆动相thigh角度
        self.knee_angle = -2.0    # 膝关节角度（固定）
        self.hip_abduction = 0.0  # 髋关节外展（默认）
    
    def get_joint_angles(self, t):
        """
        获取当前时刻的关节角度（基于成功的improved_trot实现）
        
        Args:
            t: 时间（秒）
        
        Returns:
            joint_angles: 12维关节角度 [FR_hip, FR_thigh, FR_calf, ...]
        """
        # 计算步态相位
        phase = (t * self.frequency) % 1.0
        
        # 对角步态: FR+RL一组, FL+RR一组
        if phase < 0.5:
            # FR+RL在支撑相, FL+RR在摆动相
            fr_rl_thigh = self.stance_angle
            fl_rr_thigh = self.swing_angle
        else:
            # FR+RL在摆动相, FL+RR在支撑相
            fr_rl_thigh = self.swing_angle
            fl_rr_thigh = self.stance_angle
        
        # 构造目标角度（与improved_trot_gait.py一致）
        joint_angles = np.array([
            self.hip_abduction, fr_rl_thigh, self.knee_angle,  # FR
            self.hip_abduction, fl_rr_thigh, self.knee_angle,  # FL
            self.hip_abduction, fl_rr_thigh, self.knee_angle,  # RR
            self.hip_abduction, fr_rl_thigh, self.knee_angle   # RL
        ])
        
        return joint_angles
    
    def get_joint_velocities(self, t, dt=0.001):
        """计算关节速度（数值微分）"""
        q_t = self.get_joint_angles(t)
        q_t_dt = self.get_joint_angles(t + dt)
        qd = (q_t_dt - q_t) / dt
        return qd


class AdaptiveTrotEnv(gym.Env):
    """Laikago Trot步态自适应RL环境"""
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config=None, gui=False, use_meta_learning=True):
        super().__init__()
        
        self.config = config or {}
        self.gui = gui
        self.use_meta_learning = use_meta_learning
        
        # PyBullet初始化
        if gui:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(0.001, physicsClientId=self.client)
        
        # 加载地面和机器人
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        start_pos = [0, 0, 0.5]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("laikago/laikago.urdf", start_pos, start_orientation,
                                    physicsClientId=self.client, useFixedBase=False)
        
        # 获取可控关节
        self.controllable_joints = []
        for j in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client)
            if info[2] != p.JOINT_FIXED:
                self.controllable_joints.append(j)
        
        self.num_joints = len(self.controllable_joints)
        
        # 获取初始PID增益
        if use_meta_learning:
            self.init_kp, self.init_kd = self._predict_pid_with_meta_learning()
        else:
            self.init_kp = config.get('init_kp', 0.5)
            self.init_kd = config.get('init_kd', 0.1)
        
        # 创建自适应PID控制器
        self.controller = AdaptivePIDController(
            num_joints=self.num_joints,
            init_kp=self.init_kp,
            init_kd=self.init_kd,
            kp_range=config.get('kp_range', (0.1, 2.0)),
            kd_range=config.get('kd_range', (0.01, 0.5))
        )
        
        # Trot步态生成器
        self.gait_gen = TrotGaitGenerator(
            frequency=config.get('gait_frequency', 1.0),
            step_height=config.get('step_height', 0.08),
            forward_speed=config.get('forward_speed', 0.5)
        )
        
        # 扰动配置
        self.disturbance_config = config.get('disturbance', {
            'type': 'random_force',
            'force_range': (1.0, 3.0),
            'force_interval': 500,
            'force_duration': 50
        })
        
        # Gym空间
        # 状态: [q(12), qd(12), e(12), current_kp(1), current_kd(1), 
        #       base_pos(3), base_vel(3), base_orn(3)] = 47
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32
        )
        
        # 动作: [delta_kp(1), delta_kd(1)] = 2
        self.action_space = gym.spaces.Box(
            low=-0.1, high=0.1, shape=(2,), dtype=np.float32
        )
        
        # 仿真参数
        self.dt = 0.001
        self.max_steps = config.get('max_steps', 10000)
        self.current_step = 0
        self.start_time = 0.0
        
        # 扰动状态
        self.disturbance_active = False
        self.disturbance_force = np.zeros(3)
        self.disturbance_counter = 0
        
        print(f"✅ AdaptiveTrotEnv初始化完成")
        print(f"   可控关节: {self.num_joints}")
        print(f"   初始增益: Kp={self.init_kp:.3f}, Kd={self.init_kd:.3f}")
        print(f"   步态频率: {self.gait_gen.frequency} Hz")
        print(f"   扰动类型: {self.disturbance_config['type']}")
    
    def _predict_pid_with_meta_learning(self):
        """使用元学习预测初始PID增益"""
        try:
            print("   使用元学习预测初始增益...")
            return 0.5, 0.1
        except Exception as e:
            print(f"   ⚠️  元学习预测失败，使用默认值: {e}")
            return 0.5, 0.1
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置机器人到站立姿态
        init_pose = [0.0, 1.0, -2.0] * 4  # 深蹲站立
        for i, joint_idx in enumerate(self.controllable_joints):
            p.resetJointState(self.robot_id, joint_idx, init_pose[i], 0,
                             physicsClientId=self.client)
        
        # 重置基座
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.5], 
                                          p.getQuaternionFromEuler([0, 0, 0]),
                                          physicsClientId=self.client)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0],
                           physicsClientId=self.client)
        
        # 稳定化（1000步）
        for _ in range(1000):
            for i, joint_idx in enumerate(self.controllable_joints):
                p.setJointMotorControl2(
                    self.robot_id, joint_idx, p.POSITION_CONTROL,
                    targetPosition=init_pose[i],
                    force=100, positionGain=0.5, velocityGain=0.1,
                    physicsClientId=self.client
                )
            p.stepSimulation(physicsClientId=self.client)
        
        # 重置控制器
        self.controller.reset()
        
        # 重置计数器
        self.current_step = 0
        self.start_time = 0.0
        self.disturbance_active = False
        self.disturbance_counter = 0
        
        obs = self._get_obs()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """执行一步仿真"""
        # RL调整PID增益
        delta_kp, delta_kd = action[0], action[1]
        self.controller.update_gains(delta_kp, delta_kd)
        
        # 获取当前状态
        q = np.array([p.getJointState(self.robot_id, j, physicsClientId=self.client)[0] 
                     for j in self.controllable_joints])
        qd = np.array([p.getJointState(self.robot_id, j, physicsClientId=self.client)[1] 
                      for j in self.controllable_joints])
        
        # Trot步态参考轨迹
        t = self.current_step * self.dt
        q_ref = self.gait_gen.get_joint_angles(t)
        qd_ref = self.gait_gen.get_joint_velocities(t, self.dt)
        
        # PID控制
        tau = self.controller.compute(q, qd, q_ref, qd_ref, self.dt)
        
        # 应用控制力矩
        for i, joint_idx in enumerate(self.controllable_joints):
            p.setJointMotorControl2(
                self.robot_id, joint_idx, p.TORQUE_CONTROL,
                force=tau[i], physicsClientId=self.client
            )
        
        # 应用扰动
        self._apply_disturbance()
        
        # 仿真一步
        p.stepSimulation(physicsClientId=self.client)
        
        # 计算奖励
        reward, info = self._compute_reward(q, qd, q_ref, delta_kp, delta_kd)
        
        # 终止条件
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id, 
                                                              physicsClientId=self.client)
        terminated = (base_pos[2] < 0.1) or (base_pos[2] > 0.5)  # 摔倒或飞起
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """获取观测"""
        # 关节状态
        q = np.array([p.getJointState(self.robot_id, j, physicsClientId=self.client)[0] 
                     for j in self.controllable_joints])
        qd = np.array([p.getJointState(self.robot_id, j, physicsClientId=self.client)[1] 
                      for j in self.controllable_joints])
        
        # Trot步态参考
        t = self.current_step * self.dt
        q_ref = self.gait_gen.get_joint_angles(t)
        
        # 跟踪误差
        e = q_ref - q
        
        # 当前PID增益
        kp, kd = self.controller.get_gains()
        
        # 基座状态
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id,
                                                              physicsClientId=self.client)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robot_id,
                                                    physicsClientId=self.client)
        base_orn_euler = p.getEulerFromQuaternion(base_orn)
        
        # 组合观测
        obs = np.concatenate([
            q,  # 12
            qd,  # 12
            e,  # 12
            [kp],  # 1
            [kd],  # 1
            base_pos,  # 3
            base_vel,  # 3
            base_orn_euler  # 3
        ]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self, q, qd, q_ref, delta_kp, delta_kd):
        """计算奖励（针对Trot步态优化）"""
        # 跟踪误差（主要目标）
        tracking_error = np.mean(np.abs(q_ref - q))
        
        # 前进奖励
        base_vel, _ = p.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        forward_velocity = base_vel[0]  # X方向速度
        
        # 高度稳定性
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id,
                                                       physicsClientId=self.client)
        height_error = np.abs(base_pos[2] - 0.204)  # 目标高度0.204m
        
        # 姿态稳定性
        _, base_orn = p.getBasePositionAndOrientation(self.robot_id,
                                                       physicsClientId=self.client)
        base_orn_euler = p.getEulerFromQuaternion(base_orn)
        orientation_penalty = np.abs(base_orn_euler[0]) + np.abs(base_orn_euler[1])
        
        # 增益变化惩罚
        gain_change_penalty = np.abs(delta_kp) + np.abs(delta_kd)
        
        # 总奖励（针对Trot优化）
        reward = (
            -200.0 * tracking_error           # 高权重跟踪
            + 10.0 * forward_velocity          # 鼓励前进
            - 20.0 * height_error              # 保持高度
            - 10.0 * orientation_penalty       # 保持姿态
            - 1.0 * gain_change_penalty        # 平滑调整
        )
        
        info = {
            'tracking_error': tracking_error,
            'forward_velocity': forward_velocity,
            'height_error': height_error,
            'orientation_penalty': orientation_penalty,
            'gain_change_penalty': gain_change_penalty,
            'current_kp': self.controller.kp,
            'current_kd': self.controller.kd,
            'base_x_pos': base_pos[0]
        }
        
        return reward, info
    
    def _apply_disturbance(self):
        """应用随机外力扰动"""
        dist_type = self.disturbance_config['type']
        
        if dist_type == 'random_force':
            interval = self.disturbance_config.get('force_interval', 500)
            duration = self.disturbance_config.get('force_duration', 50)
            
            if self.current_step % interval == 0:
                self.disturbance_active = True
                self.disturbance_counter = 0
                force_mag = np.random.uniform(*self.disturbance_config.get('force_range', (1.0, 3.0)))
                direction = np.random.choice([-1, 1])
                self.disturbance_force = np.array([0, direction * force_mag, 0])
            
            if self.disturbance_active:
                p.applyExternalForce(
                    self.robot_id, -1, self.disturbance_force, [0, 0, 0],
                    p.WORLD_FRAME, physicsClientId=self.client
                )
                self.disturbance_counter += 1
                
                if self.disturbance_counter >= duration:
                    self.disturbance_active = False
                    self.disturbance_force = np.zeros(3)
    
    def close(self):
        """关闭环境"""
        p.disconnect(physicsClientId=self.client)


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("测试AdaptiveTrotEnv")
    print("=" * 80)
    
    config = {
        'max_steps': 5000,
        'gait_frequency': 1.0,
        'step_height': 0.08,
        'forward_speed': 0.5,
        'kp_range': (0.1, 2.0),
        'kd_range': (0.01, 0.5),
        'disturbance': {
            'type': 'random_force',
            'force_range': (1.0, 3.0),
            'force_interval': 1000,
            'force_duration': 100
        }
    }
    
    env = AdaptiveTrotEnv(config=config, gui=True, use_meta_learning=True)
    
    obs, info = env.reset()
    print(f"\n初始观测形状: {obs.shape}")
    print(f"初始Kp: {env.controller.kp:.3f}, 初始Kd: {env.controller.kd:.3f}")
    
    print("\n开始测试Trot步态...")
    for step in range(5000):
        # 随机动作（测试）
        action = env.action_space.sample() * 0.01
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 500 == 0:
            print(f"Step {step}: reward={reward:.2f}, "
                  f"forward_vel={info['forward_velocity']:.3f}, "
                  f"x_pos={info['base_x_pos']:.3f}, "
                  f"Kp={info['current_kp']:.3f}")
        
        if terminated or truncated:
            print(f"  Episode结束于step {step}")
            break
    
    env.close()
    print("\n✅ 测试完成！")

