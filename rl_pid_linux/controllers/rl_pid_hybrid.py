"""
RL+PID混合控制器
从MATLAB RLHighLevelBlock_Toolbox.m移植
核心：渐进式补偿策略
"""

import numpy as np
from .pid_controller import PIDController, get_default_pid_gains


class RLPIDHybrid:
    """RL+PID混合控制器，渐进式补偿"""
    
    def __init__(self, config):
        pid_gains = config.get('pid_params', get_default_pid_gains())
        self.pid = PIDController(**pid_gains)
        
        rl_params = config.get('rl_params', {})
        self.delta_scale_min = rl_params.get('delta_scale_min', 0.5)
        self.delta_scale_max = rl_params.get('delta_scale_max', 5.0)
        self.warmup_disable_steps = rl_params.get('warmup_disable_steps', 100)
        self.warmup_ramp_steps = rl_params.get('warmup_ramp_steps', 500)
        self.delta_tau_clip = rl_params.get('delta_tau_clip', 10.0)
        
        self.rl_policy = None
        self.step_count = 0
        self.current_delta_scale = 0.0
        
    def load_policy(self, policy):
        self.rl_policy = policy
        
    def compute_control(self, q, qd, qref, qd_ref=None, training=False):
        # 1. PID基线
        tau_pid = self.pid.compute(q, qd, qref, qd_ref)
        
        # 2. 渐进式补偿系数
        delta_scale = self._compute_delta_scale()
        
        # 3. RL补偿
        delta_tau = np.zeros(7, dtype=np.float32)
        if self.rl_policy is not None and delta_scale > 0:
            state = self._construct_state(q, qd, qref)
            raw_action = self._get_rl_action(state, deterministic=not training)
            delta_tau = delta_scale * raw_action
            delta_tau = np.clip(delta_tau, -self.delta_tau_clip, self.delta_tau_clip)
        
        # 4. 总控制
        tau_total = tau_pid + delta_tau
        self.step_count += 1
        self.current_delta_scale = delta_scale
        
        info = {
            'tau_pid': tau_pid,
            'delta_tau': delta_tau,
            'delta_scale': delta_scale,
            'step_count': self.step_count
        }
        return tau_total, info
    
    def _compute_delta_scale(self):
        """渐进式策略"""
        if self.step_count < self.warmup_disable_steps:
            return 0.0
        elif self.step_count < self.warmup_disable_steps + self.warmup_ramp_steps:
            progress = (self.step_count - self.warmup_disable_steps) / self.warmup_ramp_steps
            return self.delta_scale_min + progress * (self.delta_scale_max - self.delta_scale_min)
        else:
            return self.delta_scale_max
    
    def _construct_state(self, q, qd, qref):
        q_err = qref - q
        return np.concatenate([q_err, qd]).astype(np.float32)
    
    def _get_rl_action(self, state, deterministic=True):
        if hasattr(self.rl_policy, 'predict'):
            action, _ = self.rl_policy.predict(state, deterministic=deterministic)
        else:
            action = self.rl_policy(state)
        return action
    
    def reset(self):
        self.pid.reset()
        self.step_count = 0
        self.current_delta_scale = 0.0


def compute_reward(q, qd, qref, action, delta_tau, config):
    """奖励函数（带稀疏奖励）"""
    track_err = qref - q
    err_norm_sq = np.sum(track_err**2)
    err_norm = np.sqrt(err_norm_sq)
    
    w_track = config.get('w_track', 5.0)
    w_vel = config.get('w_vel', 0.001)
    w_action = config.get('w_action', 0.0001)
    w_delta = config.get('w_delta', 0.0001)
    
    # ⭐ 基础奖励：增大权重让误差降低有更大收益
    r_track = -w_track * err_norm_sq
    r_vel = -w_vel * np.sum(qd**2)
    r_action = -w_action * np.sum(action**2)
    r_delta = -w_delta * np.sum(delta_tau**2)
    
    # ⭐ 稀疏奖励：严格阈值，强迫RL真正改进PID基线
    r_sparse = 0.0
    if err_norm < 0.024:  # 误差<0.024弧度（卓越，~5.7度）
        r_sparse = 2.0   # 显著奖励
    elif err_norm < 0.028:  # 误差<0.028弧度（优秀，~8.6度）
        r_sparse = 1.0   # 中等奖励
    elif err_norm < 0.032:  # 误差<0.032弧度（良好，~11.5度）
        r_sparse = 0.5   # 小奖励
    
    total_reward = r_track + r_vel + r_action + r_delta + r_sparse
    
    return total_reward, {
        'r_track': r_track,
        'r_vel': r_vel,
        'r_action': r_action,
        'r_delta': r_delta,
        'r_sparse': r_sparse,
        'err_norm': err_norm
    }


def reset_reward_state():
    if hasattr(compute_reward, 'err_accum'):
        delattr(compute_reward, 'err_accum')

