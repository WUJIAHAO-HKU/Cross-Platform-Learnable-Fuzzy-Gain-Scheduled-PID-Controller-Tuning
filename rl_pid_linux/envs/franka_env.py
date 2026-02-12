"""
Franka Panda + PyBullet环境
实现Gymnasium接口，用于RL训练
"""

import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
from controllers.rl_pid_hybrid import RLPIDHybrid, compute_reward, reset_reward_state
from .trajectory_gen import TrajectoryGenerator


class FrankaRLPIDEnv(gym.Env):
    """Franka Panda RL+PID训练环境"""
    
    def __init__(self, config, gui=False):
        super().__init__()
        self.config = config
        self.gui = gui
        
        # PyBullet初始化
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        
        # 加载机器人
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        self.num_joints = 7
        self.joint_indices = list(range(self.num_joints))
        
        # 禁用默认电机
        for i in self.joint_indices:
            p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL, force=0)
        
        # 控制器
        self.controller = RLPIDHybrid(config)
        
        # ⭐ 设置PID控制器的robot_id（用于重力补偿）
        self.controller.pid.set_robot(self.robot_id, self.client)
        
        # 轨迹生成器
        traj_config = config.get('trajectory', {'type': 'circle'})
        self.traj_gen = TrajectoryGenerator(traj_config['type'], traj_config)
        
        # Gym空间
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(7,), dtype=np.float32
        )
        
        # 参数
        self.dt = config.get('simulation', {}).get('time_step', 0.001)
        self.max_steps = config.get('max_steps', 10000)
        self.current_step = 0
        
        # 初始位置
        self.init_q = config['robot'].get('init_position', 
                                           [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 重置机器人
        for i, q in enumerate(self.init_q):
            p.resetJointState(self.robot_id, i, q, 0)
        
        # 重置控制器和轨迹
        self.controller.reset()
        self.traj_gen.reset()
        reset_reward_state()
        
        self.current_step = 0
        
        # 获取初始状态
        q, qd = self._get_robot_state()
        qref, qd_ref = self.traj_gen.get_reference(0)
        state = self.controller._construct_state(q, qd, qref)
        
        return state, {}
    
    def step(self, action):
        """执行一步"""
        # 获取当前状态
        q, qd = self._get_robot_state()
        t = self.current_step * self.dt
        qref, qd_ref = self.traj_gen.get_reference(t)
        
        # 计算控制力矩（PID + RL补偿）
        # 临时注入RL策略
        class TempPolicy:
            def __init__(self, action):
                self.action = action
            def predict(self, state, deterministic=True):
                return self.action, None
        
        self.controller.rl_policy = TempPolicy(action)
        tau_total, info = self.controller.compute_control(q, qd, qref, qd_ref, training=True)
        
        # 应用力矩
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.TORQUE_CONTROL,
            forces=tau_total
        )
        p.stepSimulation()
        
        # 新状态
        q_new, qd_new = self._get_robot_state()
        qref_new, qd_ref_new = self.traj_gen.get_reference(t + self.dt)
        next_state = self.controller._construct_state(q_new, qd_new, qref_new)
        
        # 计算奖励
        reward, reward_info = compute_reward(
            q_new, qd_new, qref_new, action, info['delta_tau'], 
            self.config.get('rl_params', {})
        )
        
        # 检查终止
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # 安全检查：发散检测
        if np.any(np.abs(q_new) > 3.0) or np.any(np.isnan(q_new)):
            reward -= 1000
            terminated = True
        
        # 合并info
        step_info = {
            **info,
            **reward_info,
            'tracking_error': np.linalg.norm(qref_new - q_new),
            'q': q_new,
            'qref': qref_new
        }
        
        return next_state, reward, terminated, truncated, step_info
    
    def _get_robot_state(self):
        """获取关节状态"""
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        q = np.array([s[0] for s in joint_states], dtype=np.float32)
        qd = np.array([s[1] for s in joint_states], dtype=np.float32)
        return q, qd
    
    def close(self):
        p.disconnect()

