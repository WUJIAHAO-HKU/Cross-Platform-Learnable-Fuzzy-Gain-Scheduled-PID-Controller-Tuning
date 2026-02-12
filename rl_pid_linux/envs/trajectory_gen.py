"""
轨迹生成器
生成参考轨迹用于跟踪控制
"""

import numpy as np


class TrajectoryGenerator:
    """轨迹生成器"""
    
    def __init__(self, traj_type='circle', params=None):
        self.traj_type = traj_type
        self.params = params or {}
        self.time = 0.0
        
    def get_reference(self, t):
        """
        获取t时刻的参考关节位置和速度
        Returns: qref (7,), qd_ref (7,)
        """
        if self.traj_type == 'circle':
            return self._circle_trajectory(t)
        elif self.traj_type == 'sine':
            return self._sine_trajectory(t)
        elif self.traj_type == 'step':
            return self._step_trajectory(t)
        else:
            return self._static_trajectory()
    
    def _circle_trajectory(self, t):
        """圆形轨迹（关节空间）"""
        speed = self.params.get('speed', 0.3)
        amplitude = self.params.get('amplitude', 0.2)
        
        base = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
        
        # 位置：前2个关节做圆周运动
        qref = base.copy()
        qref[0] = base[0] + amplitude * np.sin(speed * t)
        qref[1] = base[1] + amplitude * np.cos(speed * t)
        
        # ⭐ 速度：对位置求导
        qd_ref = np.zeros(7, dtype=np.float32)
        qd_ref[0] = amplitude * speed * np.cos(speed * t)
        qd_ref[1] = -amplitude * speed * np.sin(speed * t)
        
        return qref.astype(np.float32), qd_ref
    
    def _sine_trajectory(self, t):
        """正弦波轨迹"""
        frequency = self.params.get('frequency', 0.5)
        amplitude = self.params.get('amplitude', 0.15)
        
        base = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
        phases = 2 * np.pi * frequency * t * np.arange(1, 8) / 7.0
        
        qref = base + amplitude * np.sin(phases)
        qd_ref = amplitude * 2 * np.pi * frequency * np.arange(1, 8) / 7.0 * np.cos(phases)
        
        return qref.astype(np.float32), qd_ref.astype(np.float32)
    
    def _step_trajectory(self, t):
        """阶跃轨迹"""
        step_time = self.params.get('step_time', 2.0)
        
        base = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
        target = np.array([0.3, -0.5, 0.2, -2.0, 0.1, 1.8, 0.6])
        qd_ref = np.zeros(7, dtype=np.float32)  # 阶跃轨迹速度为0
        
        if t < step_time:
            return base.astype(np.float32), qd_ref
        else:
            return target.astype(np.float32), qd_ref
    
    def _static_trajectory(self):
        """静态目标"""
        qref = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79], dtype=np.float32)
        qd_ref = np.zeros(7, dtype=np.float32)
        return qref, qd_ref
    
    def reset(self):
        self.time = 0.0

