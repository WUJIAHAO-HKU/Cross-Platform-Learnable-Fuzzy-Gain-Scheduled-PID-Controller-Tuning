"""
PID控制器
从MATLAB LowLevelPID.m移植
"""

import numpy as np


class PIDController:
    """7-DOF机器人PID控制器（带重力补偿）"""
    
    def __init__(self, Kp, Ki, Kd, dt=0.001, enable_gravity_compensation=True):
        self.Kp = self._to_array(Kp)
        self.Ki = self._to_array(Ki)
        self.Kd = self._to_array(Kd)
        self.dt = dt
        self.enable_gravity_compensation = enable_gravity_compensation
        
        self.integral_error = np.zeros(7)
        self.prev_error = np.zeros(7)
        self.integral_limit = 100.0
        
        # 用于重力补偿的PyBullet接口（将在compute时设置）
        self.robot_id = None
        self.client_id = None
        
    def _to_array(self, value):
        if np.isscalar(value):
            return np.ones(7) * value
        return np.array(value, dtype=np.float32)
    
    def set_robot(self, robot_id, client_id):
        """设置机器人ID和物理客户端（用于重力补偿）"""
        self.robot_id = robot_id
        self.client_id = client_id
    
    def compute_gravity_compensation(self, q):
        """
        计算重力补偿力矩
        使用简化的重力模型（基于Franka Panda的近似参数）
        """
        if not self.enable_gravity_compensation:
            return np.zeros(7, dtype=np.float32)
        
        # Franka Panda各连杆的近似参数（基于公开数据）
        # 质量 (kg)
        m = np.array([4.970, 0.646, 3.228, 3.587, 1.225, 1.666, 0.735])
        
        # 各连杆质心位置在本地坐标系中的Z偏移 (m)
        # 这些是从关节到质心的近似距离
        l_cz = np.array([0.0, 0.0, -0.06, 0.0, -0.026, 0.0, 0.01])
        
        # 重力加速度
        g = 9.81
        
        # 计算重力补偿力矩
        tau_g = np.zeros(7, dtype=np.float32)
        
        # 简化模型：每个关节受到后续所有连杆重力的影响
        # Joint 1 (base rotation) - 主要受侧向重力影响，简化处理
        tau_g[0] = 0.0  # 基座旋转关节受重力影响很小
        
        # Joint 2 (shoulder pitch)
        # 受所有后续连杆重力影响
        total_mass_2 = np.sum(m[1:])
        # 简化：假设等效力臂约0.3m
        tau_g[1] = total_mass_2 * g * 0.3 * np.cos(q[1])
        
        # Joint 3 (shoulder roll)
        tau_g[2] = 0.0  # 侧向关节受重力影响较小
        
        # Joint 4 (elbow pitch)
        total_mass_4 = np.sum(m[3:])
        # 简化：假设等效力臂约0.25m
        tau_g[3] = total_mass_4 * g * 0.25 * np.cos(q[1] + q[3])
        
        # Joint 5 (wrist 1)
        tau_g[4] = 0.0  # 侧向关节
        
        # Joint 6 (wrist 2 pitch)
        total_mass_6 = np.sum(m[5:])
        # 简化：假设等效力臂约0.1m
        tau_g[5] = total_mass_6 * g * 0.1 * np.cos(q[1] + q[3] + q[5])
        
        # Joint 7 (wrist 3)
        tau_g[6] = 0.0  # 旋转关节
        
        return tau_g
    
    def compute(self, q, qd, qref, qd_ref=None):
        """计算PID控制输出（含重力补偿）"""
        error = qref - q
        self.integral_error += error * self.dt
        self.integral_error = np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        
        if qd_ref is not None:
            error_derivative = qd_ref - qd
        else:
            error_derivative = -qd
        
        # PID反馈控制
        tau_pid = (self.Kp * error + self.Ki * self.integral_error + self.Kd * error_derivative)
        
        # 重力补偿（前馈）
        tau_gravity = self.compute_gravity_compensation(q)
        
        # 总控制力矩 = PID反馈 + 重力补偿
        tau_total = tau_pid + tau_gravity
        
        self.prev_error = error.copy()
        return tau_total
    
    def reset(self):
        self.integral_error = np.zeros(7)
        self.prev_error = np.zeros(7)


def get_default_pid_gains():
    """默认PID增益（参考MATLAB实验4）"""
    return {
        'Kp': np.array([100, 100, 100, 100, 40, 20, 20], dtype=np.float32),
        'Ki': np.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.1, 0.1], dtype=np.float32),
        'Kd': np.array([5, 5, 5, 5, 2, 1, 1], dtype=np.float32)
    }

