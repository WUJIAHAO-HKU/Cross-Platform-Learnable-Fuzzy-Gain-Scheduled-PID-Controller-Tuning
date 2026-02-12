#!/usr/bin/env python3
"""
重新计算Table 1和Table 2的MAE - 使用统一的per-joint简单平均方法
"""

import numpy as np
import pickle
import os

def calculate_per_joint_mae(data_file):
    """计算per-joint平均MAE"""
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    joint_errors = data['joint_errors']  # (timesteps, n_joints)
    joint_errors_deg = np.degrees(joint_errors)
    
    # 对每个关节计算MAE，然后取平均
    per_joint_mae = np.mean(np.abs(joint_errors_deg), axis=0)
    overall_mae = np.mean(per_joint_mae)
    
    return overall_mae, per_joint_mae

def calculate_other_metrics(data_file):
    """计算其他指标（保持原来的定义）"""
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    joint_errors = data['joint_errors']
    joint_errors_deg = np.degrees(joint_errors)
    
    # RMSE: 使用L2 norm的RMS
    l2_norms = np.linalg.norm(joint_errors, axis=1)
    l2_norms_deg = np.degrees(l2_norms)
    rmse = np.sqrt(np.mean(l2_norms_deg**2))
    
    # Max Error: 最大的L2 norm
    max_error = np.max(l2_norms_deg)
    
    # Std Dev: L2 norm的标准差
    std_dev = np.std(l2_norms_deg)
    
    return rmse, max_error, std_dev

print("="*80)
print("重新计算Table 1和Table 2的MAE - 使用统一的per-joint平均方法")
print("="*80)

# Table 1: Franka Panda (9-DOF)
print("\n【Table 1: Franka Panda Results】")

base_path = "/home/wujiahao/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/rl_pid_linux/meta_learning"

files = {
    'Pure Meta-PID': 'franka_meta_pid_eval.pkl',
    'Meta-PID+RL': 'franka_meta_rl_eval.pkl'
}

results_table1 = {}

for name, file in files.items():
    file_path = os.path.join(base_path, file)
    if os.path.exists(file_path):
        mae, per_joint = calculate_per_joint_mae(file_path)
        rmse, max_err, std = calculate_other_metrics(file_path)
        
        results_table1[name] = {
            'mae': mae,
            'rmse': rmse,
            'max': max_err,
            'std': std,
            'per_joint': per_joint
        }
        
        print(f"\n{name}:")
        print(f"  MAE (per-joint平均): {mae:.2f}°")
        print(f"  RMSE (L2 norm): {rmse:.2f}°")
        print(f"  Max Error: {max_err:.2f}°")
        print(f"  Std Dev: {std:.2f}°")
        print(f"  Per-joint: {per_joint}")

# 计算改进率
if 'Pure Meta-PID' in results_table1 and 'Meta-PID+RL' in results_table1:
    pure = results_table1['Pure Meta-PID']
    rl = results_table1['Meta-PID+RL']
    
    print(f"\n改进率:")
    print(f"  MAE: {(pure['mae'] - rl['mae']) / pure['mae'] * 100:.1f}%")
    print(f"  RMSE: {(pure['rmse'] - rl['rmse']) / pure['rmse'] * 100:.1f}%")
    print(f"  Max: {(pure['max'] - rl['max']) / pure['max'] * 100:.1f}%")
    print(f"  Std: {(pure['std'] - rl['std']) / pure['std'] * 100:.1f}%")

# Table 2: Laikago (12-DOF)
print("\n" + "="*80)
print("【Table 2: Laikago Results】")

files_laikago = {
    'Pure Meta-PID': 'laikago_meta_pid_eval.pkl',
    'Meta-PID+RL': 'laikago_meta_rl_eval.pkl'
}

results_table2 = {}

for name, file in files_laikago.items():
    file_path = os.path.join(base_path, file)
    if os.path.exists(file_path):
        mae, per_joint = calculate_per_joint_mae(file_path)
        rmse, max_err, std = calculate_other_metrics(file_path)
        
        results_table2[name] = {
            'mae': mae,
            'rmse': rmse,
            'max': max_err,
            'std': std,
            'per_joint': per_joint
        }
        
        print(f"\n{name}:")
        print(f"  MAE (per-joint平均): {mae:.2f}°")
        print(f"  RMSE (L2 norm): {rmse:.2f}°")
        print(f"  Max Error: {max_err:.2f}°")
        print(f"  Std Dev: {std:.2f}°")
    else:
        print(f"\n{name}: 文件不存在 - {file}")

# 计算改进率
if 'Pure Meta-PID' in results_table2 and 'Meta-PID+RL' in results_table2:
    pure = results_table2['Pure Meta-PID']
    rl = results_table2['Meta-PID+RL']
    
    print(f"\n改进率:")
    print(f"  MAE: {(pure['mae'] - rl['mae']) / pure['mae'] * 100:.1f}%")
    print(f"  RMSE: {(pure['rmse'] - rl['rmse']) / pure['rmse'] * 100:.1f}%")
    print(f"  Max: {(pure['max'] - rl['max']) / pure['max'] * 100:.1f}%")
    print(f"  Std: {(pure['std'] - rl['std']) / pure['std'] * 100:.1f}%")

print("\n" + "="*80)
print("LaTeX代码生成")
print("="*80)

if results_table1:
    print("\n【Table 1 LaTeX代码】")
    pure = results_table1['Pure Meta-PID']
    rl = results_table1['Meta-PID+RL']
    improv = (pure['mae'] - rl['mae']) / pure['mae'] * 100
    
    print(f"MAE (°) & {pure['mae']:.2f} & \\textbf{{{rl['mae']:.2f}}} & +{improv:.1f}\\% \\\\")
    
    improv_rmse = (pure['rmse'] - rl['rmse']) / pure['rmse'] * 100
    print(f"RMSE (°) & {pure['rmse']:.2f} & \\textbf{{{rl['rmse']:.2f}}} & +{improv_rmse:.1f}\\% \\\\")
    
    improv_max = (pure['max'] - rl['max']) / pure['max'] * 100
    print(f"Max Error (°) & {pure['max']:.2f} & \\textbf{{{rl['max']:.2f}}} & +{improv_max:.1f}\\% \\\\")
    
    improv_std = (pure['std'] - rl['std']) / pure['std'] * 100
    print(f"Std Dev (°) & {pure['std']:.2f} & \\textbf{{{rl['std']:.2f}}} & +{improv_std:.1f}\\% \\\\")

if results_table2:
    print("\n【Table 2 LaTeX代码】")
    pure = results_table2['Pure Meta-PID']
    rl = results_table2['Meta-PID+RL']
    improv = (pure['mae'] - rl['mae']) / pure['mae'] * 100
    
    print(f"MAE (°) & {pure['mae']:.2f} & \\textbf{{{rl['mae']:.2f}}} & +{improv:.1f}\\% \\\\")
    
    improv_rmse = (pure['rmse'] - rl['rmse']) / pure['rmse'] * 100
    print(f"RMSE (°) & {pure['rmse']:.2f} & \\textbf{{{rl['rmse']:.2f}}} & +{improv_rmse:.1f}\\% \\\\")
    
    improv_max = (pure['max'] - rl['max']) / pure['max'] * 100
    print(f"Max Error (°) & {pure['max']:.2f} & \\textbf{{{rl['max']:.2f}}} & +{improv_max:.1f}\\% \\\\")
    
    improv_std = (pure['std'] - rl['std']) / pure['std'] * 100
    print(f"Std Dev (°) & {pure['std']:.2f} & \\textbf{{{rl['std']:.2f}}} & +{improv_std:.1f}\\% \\\\")

print("\n" + "="*80)

