#!/usr/bin/env python3
"""
验证实际跟踪误差（弧度/角度）
对比纯Meta-PID和Meta-PID+RL的真实仿真性能
"""

import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from meta_rl_combined_env import MetaRLCombinedEnv
import matplotlib.pyplot as plt


def verify_tracking_error(robot_urdf, model_path=None, steps=10000, test_name=""):
    """
    验证实际跟踪误差
    
    Args:
        robot_urdf: 机器人URDF路径
        model_path: RL模型路径（None表示纯Meta-PID）
        steps: 测试步数
        test_name: 测试名称
    """
    print(f"\n{'='*80}")
    print(f"验证: {test_name}")
    print(f"{'='*80}")
    
    # 创建环境
    env = MetaRLCombinedEnv(robot_urdf=robot_urdf, gui=False)
    
    # 加载RL模型（如果有）
    model = None
    if model_path is not None:
        model = PPO.load(model_path)
        print(f"✅ RL模型加载成功")
    else:
        print(f"✅ 使用固定Meta-PID（无RL调整）")
    
    obs, _ = env.reset()
    
    # 记录数据
    actual_errors = []  # 实际误差 (弧度)
    actual_errors_deg = []  # 实际误差 (角度)
    joint_errors = []  # 每个关节的误差
    kp_values = []
    kd_values = []
    
    for step in range(steps):
        # 选择动作
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(2)  # 固定Meta-PID
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 获取实际关节误差
        joint_states = p.getJointStates(env.robot_id, env.controllable_joints)
        q_actual = np.array([s[0] for s in joint_states])
        q_ref = env._get_reference_trajectory()
        
        # 计算实际误差
        joint_error = np.abs(q_ref - q_actual)  # 每个关节的绝对误差
        actual_error_rad = np.linalg.norm(q_ref - q_actual)  # 总误差范数（弧度）
        actual_error_deg = np.degrees(actual_error_rad)  # 转换为角度
        
        actual_errors.append(actual_error_rad)
        actual_errors_deg.append(actual_error_deg)
        joint_errors.append(joint_error)
        kp_values.append(info['current_kp'])
        kd_values.append(info['current_kd'])
        
        if step % 2000 == 0:
            print(f"Step {step:5d}: "
                  f"误差={actual_error_deg:.2f}°, "
                  f"Kp={info['current_kp']:.2f}, "
                  f"Kd={info['current_kd']:.2f}")
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
    
    # 统计结果
    actual_errors = np.array(actual_errors)
    actual_errors_deg = np.array(actual_errors_deg)
    joint_errors = np.array(joint_errors)
    
    results = {
        'actual_errors_rad': actual_errors,
        'actual_errors_deg': actual_errors_deg,
        'joint_errors': joint_errors,
        'kp_values': np.array(kp_values),
        'kd_values': np.array(kd_values),
        'mean_error_rad': np.mean(actual_errors),
        'mean_error_deg': np.mean(actual_errors_deg),
        'median_error_deg': np.median(actual_errors_deg),
        'max_error_deg': np.max(actual_errors_deg),
        'std_error_deg': np.std(actual_errors_deg),
    }
    
    print(f"\n📊 {test_name} 实际跟踪性能:")
    print(f"   平均误差: {results['mean_error_deg']:.4f}° ({results['mean_error_rad']:.6f} rad)")
    print(f"   中位误差: {results['median_error_deg']:.4f}°")
    print(f"   最大误差: {results['max_error_deg']:.4f}°")
    print(f"   标准差:   {results['std_error_deg']:.4f}°")
    
    # 每个关节的平均误差
    mean_joint_errors = np.mean(joint_errors, axis=0)
    print(f"\n   各关节平均误差 (角度):")
    for i, err in enumerate(mean_joint_errors):
        print(f"      关节{i+1}: {np.degrees(err):.4f}°")
    
    return results


def plot_actual_comparison(pure_results, rl_results, save_path='actual_tracking_comparison.png'):
    """绘制实际误差对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 实际跟踪误差对比 (角度)
    ax = axes[0, 0]
    window = 100
    pure_smooth = np.convolve(pure_results['actual_errors_deg'], 
                               np.ones(window)/window, mode='valid')
    rl_smooth = np.convolve(rl_results['actual_errors_deg'], 
                             np.ones(window)/window, mode='valid')
    
    ax.plot(pure_smooth, label='Pure Meta-PID', alpha=0.8, linewidth=1.5)
    ax.plot(rl_smooth, label='Meta-PID + RL', alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Tracking Error (degrees)')
    ax.set_title('Actual Tracking Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 误差分布直方图
    ax = axes[0, 1]
    ax.hist(pure_results['actual_errors_deg'], bins=50, alpha=0.6, 
            label='Pure Meta-PID', density=True)
    ax.hist(rl_results['actual_errors_deg'], bins=50, alpha=0.6, 
            label='Meta-PID + RL', density=True)
    ax.set_xlabel('Tracking Error (degrees)')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 各关节误差对比
    ax = axes[1, 0]
    mean_joint_errors_pure = np.mean(pure_results['joint_errors'], axis=0)
    mean_joint_errors_rl = np.mean(rl_results['joint_errors'], axis=0)
    
    x = np.arange(len(mean_joint_errors_pure))
    width = 0.35
    ax.bar(x - width/2, np.degrees(mean_joint_errors_pure), width, 
           label='Pure Meta-PID', alpha=0.8)
    ax.bar(x + width/2, np.degrees(mean_joint_errors_rl), width, 
           label='Meta-PID + RL', alpha=0.8)
    ax.set_xlabel('Joint Index')
    ax.set_ylabel('Mean Error (degrees)')
    ax.set_title('Per-Joint Error Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'J{i+1}' for i in x])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 累积分布函数 (CDF)
    ax = axes[1, 1]
    pure_sorted = np.sort(pure_results['actual_errors_deg'])
    rl_sorted = np.sort(rl_results['actual_errors_deg'])
    pure_cdf = np.arange(1, len(pure_sorted)+1) / len(pure_sorted)
    rl_cdf = np.arange(1, len(rl_sorted)+1) / len(rl_sorted)
    
    ax.plot(pure_sorted, pure_cdf, label='Pure Meta-PID', linewidth=2)
    ax.plot(rl_sorted, rl_cdf, label='Meta-PID + RL', linewidth=2)
    ax.set_xlabel('Tracking Error (degrees)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 对比图已保存: {save_path}")


def main():
    robot_urdf = 'franka_panda/panda.urdf'
    model_path = 'logs/meta_rl_panda/best_model/best_model'
    steps = 10000
    
    print("="*80)
    print("验证实际跟踪误差 (弧度/角度)")
    print("="*80)
    print(f"机器人: {robot_urdf}")
    print(f"测试步数: {steps}")
    
    # 验证1: 纯Meta-PID
    pure_results = verify_tracking_error(
        robot_urdf, 
        model_path=None, 
        steps=steps, 
        test_name="纯Meta-PID（固定预测值）"
    )
    
    # 验证2: Meta-PID + RL
    rl_results = verify_tracking_error(
        robot_urdf, 
        model_path=model_path, 
        steps=steps, 
        test_name="Meta-PID + RL（动态调整）"
    )
    
    # 性能对比
    print("\n" + "="*80)
    print("实际性能对比总结")
    print("="*80)
    
    error_improvement = (pure_results['mean_error_deg'] - rl_results['mean_error_deg']) / pure_results['mean_error_deg'] * 100
    max_error_improvement = (pure_results['max_error_deg'] - rl_results['max_error_deg']) / pure_results['max_error_deg'] * 100
    
    print(f"\n✅ 平均误差改善: {pure_results['mean_error_deg']:.4f}° → {rl_results['mean_error_deg']:.4f}° "
          f"({error_improvement:+.2f}%)")
    print(f"✅ 最大误差改善: {pure_results['max_error_deg']:.4f}° → {rl_results['max_error_deg']:.4f}° "
          f"({max_error_improvement:+.2f}%)")
    print(f"✅ 中位误差改善: {pure_results['median_error_deg']:.4f}° → {rl_results['median_error_deg']:.4f}°")
    print(f"✅ 标准差改善:   {pure_results['std_error_deg']:.4f}° → {rl_results['std_error_deg']:.4f}°")
    
    # 绘制对比图
    plot_actual_comparison(pure_results, rl_results)
    
    print("\n" + "="*80)
    print("✅ 验证完成！")
    print("="*80)


if __name__ == '__main__':
    main()

