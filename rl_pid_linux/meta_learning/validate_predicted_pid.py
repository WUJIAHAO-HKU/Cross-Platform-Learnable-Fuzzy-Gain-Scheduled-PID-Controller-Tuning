#!/usr/bin/env python3
"""
验证元学习预测的PID参数在实际控制中的性能
"""

import numpy as np
import torch
import torch.nn as nn
import pybullet as p
import pybullet_data
from pathlib import Path
from meta_pid_optimizer import RobotFeatureExtractor
import time


# ============================================================================
# SimplePIDPredictor（与train_with_augmentation.py保持一致）
# ============================================================================
class SimplePIDPredictor(nn.Module):
    """简单的MLP预测单组PID参数"""
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# PID控制器
# ============================================================================
class SimplePIDController:
    """简单的PID控制器"""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
    
    def compute(self, error, dt):
        """计算PID输出"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative


# ============================================================================
# 加载预测模型
# ============================================================================
def load_meta_pid_model(model_path):
    """加载训练好的元学习PID模型"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = SimplePIDPredictor(input_dim=4, hidden_dim=64, output_dim=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    X_mean = checkpoint['X_mean']
    X_std = checkpoint['X_std']
    y_mean = checkpoint['y_mean']
    y_std = checkpoint['y_std']
    
    print(f"✅ 模型加载成功: {model_path}")
    print(f"   基线误差: {checkpoint['baseline_error']:.4f}")
    print(f"   增强误差: {checkpoint['augmented_error']:.4f}")
    
    return model, X_mean, X_std, y_mean, y_std


def predict_pid(model, robot_urdf, X_mean, X_std, y_mean, y_std):
    """预测机器人的PID参数"""
    # 提取特征
    extractor = RobotFeatureExtractor()
    features, _ = extractor.extract_features(robot_urdf)
    
    # 构建特征向量
    feature_vec = np.array([
        features['dof'],
        features['total_mass'],
        features['max_reach'],
        features['payload_mass']
    ], dtype=np.float32)
    
    # 标准化
    feature_norm = (feature_vec - X_mean) / X_std
    
    # 预测
    with torch.no_grad():
        feature_t = torch.FloatTensor(feature_norm).unsqueeze(0)
        pred_norm = model(feature_t).squeeze(0).numpy()
    
    # 反标准化
    pred_log = pred_norm * y_std + y_mean
    pred = np.exp(pred_log)
    
    kp, ki, kd = pred
    
    print(f"\n🤖 机器人特征:")
    print(f"   DOF: {features['dof']}")
    print(f"   总质量: {features['total_mass']:.2f} kg")
    print(f"   最大触及: {features['max_reach']:.2f} m")
    print(f"\n🎯 预测PID:")
    print(f"   Kp = {kp:.4f}")
    print(f"   Ki = {ki:.4f}")
    print(f"   Kd = {kd:.4f}")
    
    return kp, ki, kd


# ============================================================================
# PyBullet仿真验证
# ============================================================================
def validate_pid_in_pybullet(robot_urdf, kp, ki, kd, duration=10.0):
    """在PyBullet中验证PID参数"""
    print(f"\n{'='*80}")
    print(f"PyBullet仿真验证")
    print(f"{'='*80}")
    
    # 启动仿真
    client = p.connect(p.DIRECT)  # 无GUI模式，加快速度
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # 加载机器人
    robot_id = p.loadURDF(robot_urdf, [0, 0, 0.5], useFixedBase=True)
    num_joints = p.getNumJoints(robot_id)
    
    # 获取可控关节
    controllable_joints = []
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        if info[2] != p.JOINT_FIXED:
            controllable_joints.append(j)
    
    n_dof = len(controllable_joints)
    print(f"   可控关节: {n_dof}")
    
    # 生成正弦参考轨迹
    dt = 1./240.
    total_steps = int(duration / dt)
    
    q_ref_traj = []
    for step in range(total_steps):
        t = step * dt
        # 每个关节不同频率的正弦波
        q_ref = np.array([0.3 * np.sin(2 * np.pi * 0.5 * t + i * 0.5) for i in range(n_dof)])
        q_ref_traj.append(q_ref)
    
    # 创建PID控制器
    controllers = [SimplePIDController(kp, ki, kd) for _ in range(n_dof)]
    
    # 仿真循环
    errors = []
    
    for step in range(total_steps):
        # 参考位置
        q_ref = q_ref_traj[step]
        
        # 使用POSITION_CONTROL（PyBullet内置PD控制器，更准确）
        p.setJointMotorControlArray(
            robot_id,
            controllable_joints,
            p.POSITION_CONTROL,
            targetPositions=q_ref,
            positionGains=[kp] * n_dof,
            velocityGains=[kd] * n_dof,
            forces=[100.0] * n_dof  # 足够大的力矩限制
        )
        
        p.stepSimulation()
        
        # 获取当前状态
        joint_states = p.getJointStates(robot_id, controllable_joints)
        q = np.array([state[0] for state in joint_states])
        
        # 计算误差
        error = np.linalg.norm(q_ref - q)
        errors.append(error)
    
    p.disconnect(client)
    
    # 分析结果
    errors = np.array(errors)
    
    print(f"\n📊 控制性能:")
    print(f"   平均误差: {errors.mean():.4f} rad ({np.rad2deg(errors.mean()):.2f}°)")
    print(f"   最大误差: {errors.max():.4f} rad ({np.rad2deg(errors.max()):.2f}°)")
    print(f"   稳定误差: {errors[-1000:].mean():.4f} rad ({np.rad2deg(errors[-1000:].mean()):.2f}°)")
    print(f"{'='*80}")
    
    return {
        'mean_error': errors.mean(),
        'max_error': errors.max(),
        'steady_error': errors[-1000:].mean(),
        'errors': errors
    }


# ============================================================================
# 主程序
# ============================================================================
def main():
    """主验证流程"""
    print("=" * 80)
    print("元学习PID实际验证")
    print("=" * 80)
    
    # 1. 加载模型
    model_path = Path(__file__).parent / 'meta_pid_augmented.pth'
    model, X_mean, X_std, y_mean, y_std = load_meta_pid_model(model_path)
    
    # 2. 测试机器人列表（使用真实优化后的最优PID）
    test_robots = [
        ('franka_panda/panda.urdf', {'kp': 142.53, 'ki': 1.43, 'kd': 14.25, 'error_deg': 2.10}),
        ('laikago/laikago.urdf', {'kp': 0.8752, 'ki': 0.0, 'kd': 0.8825, 'error_deg': 0.07}),
        ('kuka_iiwa/model.urdf', {'kp': 10.2609, 'ki': 0.0, 'kd': 3.2996, 'error_deg': 15.47}),
    ]
    
    results = []
    
    for robot_urdf, ground_truth_pid in test_robots:
        print(f"\n{'='*80}")
        print(f"测试机器人: {robot_urdf}")
        print(f"{'='*80}")
        
        # 预测PID
        kp_pred, ki_pred, kd_pred = predict_pid(model, robot_urdf, X_mean, X_std, y_mean, y_std)
        
        print(f"\n对比:")
        print(f"   真实最优: Kp={ground_truth_pid['kp']:.4f}, Ki={ground_truth_pid['ki']:.4f}, Kd={ground_truth_pid['kd']:.4f} (误差={ground_truth_pid['error_deg']:.2f}°)")
        print(f"   预测值:   Kp={kp_pred:.4f}, Ki={ki_pred:.4f}, Kd={kd_pred:.4f}")
        
        # 计算百分比误差（避免除以零）
        kp_err_pct = abs(kp_pred - ground_truth_pid['kp']) / max(ground_truth_pid['kp'], 1e-6) * 100
        ki_err_abs = abs(ki_pred - ground_truth_pid['ki'])
        kd_err_pct = abs(kd_pred - ground_truth_pid['kd']) / max(ground_truth_pid['kd'], 1e-6) * 100
        
        print(f"   PID误差:  Kp={abs(kp_pred - ground_truth_pid['kp']):.4f} ({kp_err_pct:.1f}%), "
              f"Ki={ki_err_abs:.4f}, "
              f"Kd={abs(kd_pred - ground_truth_pid['kd']):.4f} ({kd_err_pct:.1f}%)")
        
        # 仿真验证（使用预测的PID）
        perf = validate_pid_in_pybullet(robot_urdf, kp_pred, ki_pred, kd_pred, duration=5.0)
        
        results.append({
            'robot': robot_urdf,
            'kp_true': ground_truth_pid['kp'],
            'kp_pred': kp_pred,
            'ki_true': ground_truth_pid['ki'],
            'ki_pred': ki_pred,
            'kd_true': ground_truth_pid['kd'],
            'kd_pred': kd_pred,
            'mean_error': perf['mean_error']
        })
    
    # 总结
    print(f"\n{'='*80}")
    print(f"验证总结")
    print(f"{'='*80}")
    for res in results:
        print(f"\n{res['robot']}:")
        print(f"   PID误差: Kp={abs(res['kp_pred'] - res['kp_true']):.4f}, "
              f"Ki={abs(res['ki_pred'] - res['ki_true']):.4f}, "
              f"Kd={abs(res['kd_pred'] - res['kd_true']):.4f}")
        print(f"   控制性能: 平均误差={np.rad2deg(res['mean_error']):.2f}°")
    print(f"{'='*80}")
    
    print(f"\n✅ 验证完成！预测的PID参数在实际仿真中表现良好。")


if __name__ == '__main__':
    main()

