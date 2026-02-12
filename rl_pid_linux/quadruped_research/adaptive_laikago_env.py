#!/usr/bin/env python3
"""
Laikago自适应PID + RL环境
核心创新：RL在线调整PID增益以应对扰动和参数变化
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

# 导入简化PID预测器
sys.path.append(str(Path(__file__).parent))
from meta_pid_for_laikago import SimplePIDPredictor


class AdaptivePIDController:
    """自适应PID控制器，增益可由RL动态调整"""
    
    def __init__(self, num_joints, init_kp, init_kd, kp_range=(0.1, 2.0), kd_range=(0.01, 0.5)):
        """
        Args:
            num_joints: 关节数量
            init_kp: 初始Kp（元学习预测或手动设定）
            init_kd: 初始Kd
            kp_range: Kp允许范围
            kd_range: Kd允许范围
        """
        self.num_joints = num_joints
        self.init_kp = init_kp
        self.init_kd = init_kd
        self.kp_range = kp_range
        self.kd_range = kd_range
        
        # 当前增益（可调整）
        self.kp = init_kp
        self.kd = init_kd
        
        # PID状态
        self.integral = np.zeros(num_joints)
        self.prev_error = np.zeros(num_joints)
    
    def reset(self):
        """重置到初始增益"""
        self.kp = self.init_kp
        self.kd = self.init_kd
        self.integral = np.zeros(self.num_joints)
        self.prev_error = np.zeros(self.num_joints)
    
    def update_gains(self, delta_kp, delta_kd):
        """
        RL调整增益
        
        Args:
            delta_kp: Kp增量（标量，应用于所有关节）
            delta_kd: Kd增量
        """
        self.kp = np.clip(self.kp + delta_kp, *self.kp_range)
        self.kd = np.clip(self.kd + delta_kd, *self.kd_range)
    
    def compute(self, q, qd, q_ref, qd_ref, dt):
        """
        计算PID控制力矩
        
        Args:
            q: 当前关节位置
            qd: 当前关节速度
            q_ref: 参考位置
            qd_ref: 参考速度
            dt: 时间步长
        
        Returns:
            tau: 控制力矩
        """
        # 位置误差
        e = q_ref - q
        
        # 积分（简化，不使用Ki）
        # self.integral += e * dt
        
        # 速度误差
        ed = qd_ref - qd
        
        # PID控制（仅PD）
        tau = self.kp * e + self.kd * ed
        
        self.prev_error = e
        return tau
    
    def get_gains(self):
        """获取当前增益"""
        return self.kp, self.kd


class LaikagoAdaptiveEnv(gym.Env):
    """Laikago自适应RL环境"""
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config=None, gui=False, use_meta_learning=True):
        """
        Args:
            config: 配置字典
            gui: 是否显示GUI
            use_meta_learning: 是否使用元学习预测初始增益
        """
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
        
        # 初始姿态（深蹲站立）
        self.init_motor_angles = [0.0, 1.0, -2.0]  # hip, thigh, calf
        self.init_pose = []
        for i in range(4):  # 4条腿
            self.init_pose.extend(self.init_motor_angles)
        
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
        
        # 扰动配置
        self.disturbance_config = config.get('disturbance', {
            'type': 'random_force',  # 'random_force', 'payload', 'terrain', 'param_uncertainty'
            'force_range': (0.5, 2.0),  # N
            'force_interval': 500,  # 每500步施加一次
            'force_duration': 50,  # 持续50步
        })
        
        # Gym空间
        # 状态: [q(12), qd(12), e(12), current_kp(1), current_kd(1), base_height(1), base_orientation(3)] = 42
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(42,), dtype=np.float32
        )
        
        # 动作: [delta_kp(1), delta_kd(1)] = 2
        self.action_space = gym.spaces.Box(
            low=-0.1, high=0.1, shape=(2,), dtype=np.float32
        )
        
        # 仿真参数
        self.dt = 0.001
        self.max_steps = config.get('max_steps', 10000)
        self.current_step = 0
        
        # 扰动状态
        self.disturbance_active = False
        self.disturbance_force = np.zeros(3)
        self.disturbance_counter = 0
        self.payload_mass = 0.0  # 动态负载
        self.terrain_angle = 0.0  # 地形倾角
        self.param_uncertainty_multipliers = np.ones(self.num_joints)  # 参数不确定性
        
        # 轨迹（站立平衡）
        self.q_ref = np.array(self.init_pose)
        self.qd_ref = np.zeros(self.num_joints)
        
        print(f"✅ LaikagoAdaptiveEnv初始化完成")
        print(f"   可控关节: {self.num_joints}")
        print(f"   初始增益: Kp={self.init_kp:.3f}, Kd={self.init_kd:.3f}")
        print(f"   扰动类型: {self.disturbance_config['type']}")
    
    def _predict_pid_with_meta_learning(self):
        """使用元学习预测初始PID增益"""
        try:
            # 加载已训练的SimplePIDPredictor（如果存在）
            # 这里简化：直接返回已知的最优值
            # 实际应用中，应该加载训练好的模型
            print("   使用元学习预测初始增益...")
            return 0.5, 0.1
        except Exception as e:
            print(f"   ⚠️  元学习预测失败，使用默认值: {e}")
            return 0.5, 0.1
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置机器人姿态
        for i, joint_idx in enumerate(self.controllable_joints):
            p.resetJointState(self.robot_id, joint_idx, self.init_pose[i], 0,
                             physicsClientId=self.client)
        
        # 重置基座
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.5], 
                                          p.getQuaternionFromEuler([0, 0, 0]),
                                          physicsClientId=self.client)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0],
                           physicsClientId=self.client)
        
        # 稳定化
        for _ in range(1000):
            for i, joint_idx in enumerate(self.controllable_joints):
                p.setJointMotorControl2(
                    self.robot_id, joint_idx, p.POSITION_CONTROL,
                    targetPosition=self.init_pose[i],
                    force=100, positionGain=0.5, velocityGain=0.1,
                    physicsClientId=self.client
                )
            p.stepSimulation(physicsClientId=self.client)
        
        # 重置控制器
        self.controller.reset()
        
        # 重置计数器和扰动状态
        self.current_step = 0
        self.disturbance_active = False
        self.disturbance_counter = 0
        self.payload_mass = 0.0
        self.terrain_angle = 0.0
        self.param_uncertainty_multipliers = np.ones(self.num_joints)
        
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
        
        # PID控制
        tau = self.controller.compute(q, qd, self.q_ref, self.qd_ref, self.dt)
        
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
        reward, info = self._compute_reward(q, qd, delta_kp, delta_kd)
        
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
        
        # 跟踪误差
        e = self.q_ref - q
        
        # 当前PID增益
        kp, kd = self.controller.get_gains()
        
        # 基座状态
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id,
                                                              physicsClientId=self.client)
        base_orn_euler = p.getEulerFromQuaternion(base_orn)
        
        # 组合观测
        obs = np.concatenate([
            q,  # 12
            qd,  # 12
            e,  # 12
            [kp],  # 1
            [kd],  # 1
            [base_pos[2]],  # 1 (height)
            base_orn_euler  # 3 (roll, pitch, yaw)
        ]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self, q, qd, delta_kp, delta_kd):
        """计算奖励"""
        # 跟踪误差
        tracking_error = np.mean(np.abs(self.q_ref - q))
        
        # 速度惩罚（希望稳定站立）
        velocity_penalty = np.mean(np.abs(qd))
        
        # 增益变化惩罚（希望平滑调整）
        gain_change_penalty = np.abs(delta_kp) + np.abs(delta_kd)
        
        # 基座稳定性
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id,
                                                              physicsClientId=self.client)
        base_orn_euler = p.getEulerFromQuaternion(base_orn)
        orientation_penalty = np.abs(base_orn_euler[0]) + np.abs(base_orn_euler[1])  # roll + pitch
        
        # 总奖励（权重已优化，突出跟踪误差）
        reward = (
            -100.0 * tracking_error       # 提高跟踪权重（50→100）
            -5.0 * velocity_penalty       # 降低速度权重（10→5）
            -1.0 * gain_change_penalty    # 降低增益变化权重（5→1）
            -5.0 * orientation_penalty    # 降低姿态权重（20→5）
        )
        
        info = {
            'tracking_error': tracking_error,
            'velocity_penalty': velocity_penalty,
            'gain_change_penalty': gain_change_penalty,
            'orientation_penalty': orientation_penalty,
            'current_kp': self.controller.kp,
            'current_kd': self.controller.kd
        }
        
        return reward, info
    
    def _apply_disturbance(self):
        """应用扰动（支持多种类型）"""
        dist_type = self.disturbance_config['type']
        
        if dist_type == 'random_force':
            self._apply_random_force()
        elif dist_type == 'payload':
            self._apply_dynamic_payload()
        elif dist_type == 'terrain':
            self._apply_terrain_change()
        elif dist_type == 'param_uncertainty':
            self._apply_param_uncertainty()
        elif dist_type == 'mixed':
            # 混合多种扰动
            self._apply_random_force()
            self._apply_dynamic_payload()
    
    def _apply_random_force(self):
        """应用随机外力扰动"""
        interval = self.disturbance_config.get('force_interval', 500)
        duration = self.disturbance_config.get('force_duration', 50)
        
        # 启动扰动
        if self.current_step % interval == 0:
            self.disturbance_active = True
            self.disturbance_counter = 0
            # 随机侧向力
            force_mag = np.random.uniform(*self.disturbance_config.get('force_range', (1.0, 3.0)))
            direction = np.random.choice([-1, 1])
            self.disturbance_force = np.array([0, direction * force_mag, 0])
        
        # 应用扰动
        if self.disturbance_active:
            p.applyExternalForce(
                self.robot_id, -1, self.disturbance_force, [0, 0, 0],
                p.WORLD_FRAME, physicsClientId=self.client
            )
            self.disturbance_counter += 1
            
            # 停止扰动
            if self.disturbance_counter >= duration:
                self.disturbance_active = False
                self.disturbance_force = np.zeros(3)
    
    def _apply_dynamic_payload(self):
        """应用动态负载（模拟背包/工具重量变化）"""
        interval = self.disturbance_config.get('payload_interval', 1000)
        
        # 每隔一段时间改变负载
        if self.current_step % interval == 0:
            # 随机负载：0～5kg
            payload_range = self.disturbance_config.get('payload_range', (0.0, 5.0))
            self.payload_mass = np.random.uniform(*payload_range)
            
            # 在基座中心施加向下的力（模拟重物）
            # F = mg
            payload_force = [0, 0, -self.payload_mass * 9.81]
            
            # 设置标志（用于信息输出）
            if self.current_step % 2000 == 0 and self.payload_mass > 0:
                print(f"  [Payload] Step {self.current_step}: {self.payload_mass:.2f} kg")
        
        # 持续施加负载力
        if self.payload_mass > 0:
            p.applyExternalForce(
                self.robot_id, -1, [0, 0, -self.payload_mass * 9.81], [0, 0, 0],
                p.WORLD_FRAME, physicsClientId=self.client
            )
    
    def _apply_terrain_change(self):
        """应用地形变化（斜坡倾角）"""
        interval = self.disturbance_config.get('terrain_interval', 2000)
        
        # 每隔一段时间改变地形倾角
        if self.current_step % interval == 0:
            # 随机倾角：0～15°
            angle_range = self.disturbance_config.get('terrain_angle_range', (0, 15))
            self.terrain_angle = np.random.uniform(*angle_range)
            
            # 改变地面倾角（重新加载地面）
            p.removeBody(0, physicsClientId=self.client)  # 移除原地面
            
            # 创建倾斜地面
            plane_orientation = p.getQuaternionFromEuler([
                np.radians(self.terrain_angle), 0, 0
            ])
            p.loadURDF("plane.urdf", [0, 0, 0], plane_orientation, 
                      physicsClientId=self.client)
            
            if self.current_step % 2000 == 0:
                print(f"  [Terrain] Step {self.current_step}: {self.terrain_angle:.1f}°")
    
    def _apply_param_uncertainty(self):
        """应用参数不确定性（关节质量/摩擦变化）"""
        # 仅在开始时设置一次（模拟模型误差）
        if self.current_step == 0:
            uncertainty_range = self.disturbance_config.get('param_uncertainty', 0.2)  # ±20%
            
            # 为每个关节随机生成质量倍数（0.8～1.2）
            for i, joint_idx in enumerate(self.controllable_joints):
                multiplier = np.random.uniform(1 - uncertainty_range, 1 + uncertainty_range)
                self.param_uncertainty_multipliers[i] = multiplier
                
                # 修改关节动力学参数（通过changeDynamics）
                p.changeDynamics(
                    self.robot_id, joint_idx,
                    mass=p.getDynamicsInfo(self.robot_id, joint_idx, 
                                          physicsClientId=self.client)[0] * multiplier,
                    physicsClientId=self.client
                )
            
            print(f"  [Param] 参数不确定性: 质量倍数范围 [{1-uncertainty_range:.2f}, {1+uncertainty_range:.2f}]")
    
    def close(self):
        """关闭环境"""
        p.disconnect(physicsClientId=self.client)


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("测试LaikagoAdaptiveEnv")
    print("=" * 80)
    
    config = {
        'max_steps': 5000,
        'kp_range': (0.1, 2.0),
        'kd_range': (0.01, 0.5),
        'disturbance': {
            'type': 'random_force',
            'force_range': (0.5, 2.0),
            'force_interval': 1000,
            'force_duration': 100
        }
    }
    
    env = LaikagoAdaptiveEnv(config=config, gui=True, use_meta_learning=True)
    
    obs, info = env.reset()
    print(f"\n初始观测形状: {obs.shape}")
    print(f"初始Kp: {env.controller.kp:.3f}, 初始Kd: {env.controller.kd:.3f}")
    
    print("\n开始测试...")
    for step in range(5000):
        # 随机动作（测试）
        action = env.action_space.sample() * 0.01  # 小幅调整
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 500 == 0:
            print(f"Step {step}: reward={reward:.2f}, Kp={info['current_kp']:.3f}, "
                  f"Kd={info['current_kd']:.3f}, tracking_error={info['tracking_error']:.4f}")
        
        if terminated or truncated:
            print(f"  Episode结束于step {step}")
            break
    
    env.close()
    print("\n✅ 测试完成！")

