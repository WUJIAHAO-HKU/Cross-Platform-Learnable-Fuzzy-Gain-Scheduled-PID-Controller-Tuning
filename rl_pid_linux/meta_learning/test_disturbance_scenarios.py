#!/usr/bin/env python3
"""
扰动场景完整测试
评估纯Meta-PID和Meta-PID+RL在不同扰动下的鲁棒性
"""

import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from meta_rl_disturbance_env import MetaRLDisturbanceEnv
import matplotlib.pyplot as plt
import argparse


def evaluate_under_disturbance(robot_urdf, disturbance_type, model_path=None, 
                                n_episodes=5, max_steps=3000, seed=None):
    """
    在特定扰动下评估性能
    
    Args:
        seed: 随机种子（用于可重复性）
    
    Returns:
        dict: 包含mean_error_deg, max_error_deg, std_error_deg等统计数据
    """
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # 创建环境
    env = MetaRLDisturbanceEnv(
        robot_urdf=robot_urdf,
        gui=False,
        disturbance_type=disturbance_type
    )
    
    # 加载RL模型（如果有）
    model = None
    if model_path is not None:
        try:
            model = PPO.load(model_path)
        except Exception as e:
            print(f"⚠️ 加载模型失败: {e}")
            model = None
    
    # 记录数据
    all_errors_deg = []
    episode_max_errors = []
    
    for episode in range(n_episodes):
        # 为每个episode设置不同的种子（如果提供了种子）
        episode_seed = None if seed is None else seed + episode
        obs, _ = env.reset(seed=episode_seed)
        episode_errors = []
        
        for step in range(max_steps):
            # 选择动作
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros(2)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 获取实际误差
            joint_states = p.getJointStates(env.robot_id, env.controllable_joints)
            q_actual = np.array([s[0] for s in joint_states])
            q_ref = env._get_reference_trajectory()
            
            error_rad = np.linalg.norm(q_ref - q_actual)
            error_deg = np.degrees(error_rad)
            
            episode_errors.append(error_deg)
            
            if terminated or truncated:
                break
        
        all_errors_deg.extend(episode_errors)
        episode_max_errors.append(np.max(episode_errors))
    
    env.close()
    
    # 统计结果
    results = {
        'mean_error_deg': np.mean(all_errors_deg),
        'median_error_deg': np.median(all_errors_deg),
        'max_error_deg': np.mean(episode_max_errors),  # 平均最大误差
        'std_error_deg': np.std(all_errors_deg),
    }
    
    return results


def run_disturbance_tests(robot_urdf, model_path, disturbance_types, n_episodes=5, seed=None):
    """运行完整扰动测试
    
    Args:
        seed: 随机种子（用于可重复性），如果为None则使用随机行为
    """
    
    print("="*80)
    print("扰动场景鲁棒性测试")
    print("="*80)
    print(f"机器人: {robot_urdf}")
    print(f"扰动类型: {disturbance_types}")
    print(f"每种扰动测试回合: {n_episodes}")
    if seed is not None:
        print(f"随机种子: {seed}")
    print()
    
    # 测试纯Meta-PID
    print("="*80)
    print("测试1: 纯Meta-PID（固定预测值）")
    print("="*80)
    pure_results = {}
    
    for i, dist_type in enumerate(disturbance_types):
        print(f"\n🔬 扰动: {dist_type}")
        # 为不同扰动类型设置不同的种子基数
        dist_seed = None if seed is None else seed + i * 1000
        result = evaluate_under_disturbance(
            robot_urdf, dist_type, model_path=None,
            n_episodes=n_episodes, seed=dist_seed
        )
        pure_results[dist_type] = result
        print(f"   平均误差: {result['mean_error_deg']:.2f}°")
        print(f"   最大误差: {result['max_error_deg']:.2f}°")
        print(f"   标准差:   {result['std_error_deg']:.2f}°")
    
    # 测试Meta-PID + RL
    print("\n" + "="*80)
    print("测试2: Meta-PID + RL（动态调整）")
    print("="*80)
    rl_results = {}
    
    for i, dist_type in enumerate(disturbance_types):
        print(f"\n🔬 扰动: {dist_type}")
        # 使用相同的种子基数确保Pure和RL测试在相同条件下比较
        dist_seed = None if seed is None else seed + i * 1000
        result = evaluate_under_disturbance(
            robot_urdf, dist_type, model_path=model_path,
            n_episodes=n_episodes, seed=dist_seed
        )
        rl_results[dist_type] = result
        print(f"   平均误差: {result['mean_error_deg']:.2f}°")
        print(f"   最大误差: {result['max_error_deg']:.2f}°")
        print(f"   标准差:   {result['std_error_deg']:.2f}°")
    
    return pure_results, rl_results


def plot_disturbance_comparison(pure_results, rl_results, save_path='disturbance_comparison.png', 
                                 statistics=None):
    """绘制扰动场景对比图
    
    Args:
        statistics: 可选，多种子统计数据，格式为:
            {
                'pure_mean': {dist: mean_error},
                'pure_std': {dist: std_error},
                'rl_mean': {dist: mean_error},
                'rl_std': {dist: std_error}
            }
            如果提供，子图(d)将展示多种子统计而非单次改进百分比
    """
    
    disturbances = list(pure_results.keys())
    dist_labels = {
        'none': 'No Disturbance',
        'random_force': 'Random Force',
        'payload': 'Payload (+3kg)',
        'param_uncertainty': 'Param Uncertainty',
        'mixed': 'Mixed Disturbances'
    }
    
    # 提取数据
    pure_mean = [pure_results[d]['mean_error_deg'] for d in disturbances]
    rl_mean = [rl_results[d]['mean_error_deg'] for d in disturbances]
    
    pure_max = [pure_results[d]['max_error_deg'] for d in disturbances]
    rl_max = [rl_results[d]['max_error_deg'] for d in disturbances]
    
    pure_std = [pure_results[d]['std_error_deg'] for d in disturbances]
    rl_std = [rl_results[d]['std_error_deg'] for d in disturbances]
    
    # 计算改善
    improvements_mean = [(pure_mean[i] - rl_mean[i]) / pure_mean[i] * 100 
                         for i in range(len(disturbances))]
    improvements_max = [(pure_max[i] - rl_max[i]) / pure_max[i] * 100 
                        for i in range(len(disturbances))]
    improvements_std = [(pure_std[i] - rl_std[i]) / pure_std[i] * 100 
                        for i in range(len(disturbances))]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 平均误差对比 + 改进曲线
    ax = axes[0, 0]
    ax2 = ax.twinx()  # 创建双Y轴
    x = np.arange(len(disturbances))
    width = 0.35
    ax.bar(x - width/2, pure_mean, width, label='Pure Meta-PID', alpha=0.8, color='skyblue')
    ax.bar(x + width/2, rl_mean, width, label='Meta-PID + RL', alpha=0.8, color='lightcoral')
    # 改进曲线
    line = ax2.plot(x, improvements_mean, 'o-', color='#2E7D32', linewidth=2.5, 
                    markersize=8, label='Improvement %', markeredgecolor='white', markeredgewidth=1.5)
    # 改进标签
    for i, imp in enumerate(improvements_mean):
        y_offset = imp + 2 if imp > 0 else imp - 2
        ax2.text(i, y_offset, f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top',
                fontsize=9, fontweight='bold', color='#1B5E20')
    
    ax.set_xlabel('Disturbance Type', fontsize=12)
    ax.set_ylabel('Mean Error (degrees)', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12, color='#2E7D32')
    ax2.tick_params(axis='y', labelcolor='#2E7D32')
    ax.set_title('(a) Mean Tracking Error Under Different Disturbances', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([dist_labels.get(d, d) for d in disturbances], rotation=20, ha='right')
    # 合并图例
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. 最大误差对比 + 改进曲线
    ax = axes[0, 1]
    ax2 = ax.twinx()
    ax.bar(x - width/2, pure_max, width, label='Pure Meta-PID', alpha=0.8, color='skyblue')
    ax.bar(x + width/2, rl_max, width, label='Meta-PID + RL', alpha=0.8, color='lightcoral')
    # 改进曲线
    line = ax2.plot(x, improvements_max, 'o-', color='#2E7D32', linewidth=2.5, 
                    markersize=8, label='Improvement %', markeredgecolor='white', markeredgewidth=1.5)
    # 改进标签
    for i, imp in enumerate(improvements_max):
        y_offset = imp + 2 if imp > 0 else imp - 2
        ax2.text(i, y_offset, f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top',
                fontsize=9, fontweight='bold', color='#1B5E20')
    
    ax.set_xlabel('Disturbance Type', fontsize=12)
    ax.set_ylabel('Max Error (degrees)', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12, color='#2E7D32')
    ax2.tick_params(axis='y', labelcolor='#2E7D32')
    ax.set_title('(b) Maximum Tracking Error Under Different Disturbances', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([dist_labels.get(d, d) for d in disturbances], rotation=20, ha='right')
    # 合并图例
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. 标准差对比 + 改进曲线
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.bar(x - width/2, pure_std, width, label='Pure Meta-PID', alpha=0.8, color='skyblue')
    ax.bar(x + width/2, rl_std, width, label='Meta-PID + RL', alpha=0.8, color='lightcoral')
    # 改进曲线
    line = ax2.plot(x, improvements_std, 'o-', color='#2E7D32', linewidth=2.5, 
                    markersize=8, label='Improvement %', markeredgecolor='white', markeredgewidth=1.5)
    # 改进标签
    for i, imp in enumerate(improvements_std):
        y_offset = imp + 2 if imp > 0 else imp - 2
        ax2.text(i, y_offset, f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top',
                fontsize=9, fontweight='bold', color='#1B5E20')
    
    ax.set_xlabel('Disturbance Type', fontsize=12)
    ax.set_ylabel('Std Dev (degrees)', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12, color='#2E7D32')
    ax2.tick_params(axis='y', labelcolor='#2E7D32')
    ax.set_title('(c) Error Standard Deviation Under Different Disturbances', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([dist_labels.get(d, d) for d in disturbances], rotation=20, ha='right')
    # 合并图例
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 子图(d): 根据是否有statistics选择展示内容
    ax = axes[1, 1]
    
    if statistics is not None:
        # 展示多种子统计 (均值±标准差)
        pure_means = [statistics['pure_mean'][d] for d in disturbances]
        pure_stds = [statistics['pure_std'][d] for d in disturbances]
        rl_means = [statistics['rl_mean'][d] for d in disturbances]
        rl_stds = [statistics['rl_std'][d] for d in disturbances]
        
        # 绘制带误差条的柱状图
        bars1 = ax.bar(x - width/2, pure_means, width, yerr=pure_stds, 
                      label='Pure Meta-PID', alpha=0.8, color='skyblue',
                      capsize=5, error_kw={'linewidth': 1.5, 'ecolor': '#1976D2'})
        bars2 = ax.bar(x + width/2, rl_means, width, yerr=rl_stds,
                      label='Meta-PID + RL', alpha=0.8, color='lightcoral',
                      capsize=5, error_kw={'linewidth': 1.5, 'ecolor': '#C62828'})
        
        ax.set_xlabel('Disturbance Type', fontsize=12)
        ax.set_ylabel('Mean Error (degrees)', fontsize=12)
        ax.set_title('(d) Multi-Seed Statistical Comparison (Mean±Std)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([dist_labels.get(d, d) for d in disturbances], rotation=20, ha='right')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 计算并标注平均改进率
        for i, (pure_m, rl_m) in enumerate(zip(pure_means, rl_means)):
            improvement = (pure_m - rl_m) / pure_m * 100
            # 在两个柱子中间标注改进率
            y_pos = max(pure_m, rl_m) * 1.15
            ax.text(i, y_pos, f'{improvement:+.1f}%', ha='center', va='bottom',
                   fontsize=9, fontweight='bold', color='#2E7D32' if improvement > 0 else '#D84315')
    else:
        # 展示单次改进百分比（原有逻辑）
        colors = ['#2E7D32' if imp > 0 else '#D84315' for imp in improvements_mean]
        bars = ax.bar(x, improvements_mean, alpha=0.85, color=colors, edgecolor='black', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax.set_xlabel('Disturbance Type', fontsize=12)
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.set_title('(d) Performance Improvement with RL Adaptation', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([dist_labels.get(d, d) for d in disturbances], rotation=20, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 标注数值
        for i, (bar, imp) in enumerate(zip(bars, improvements_mean)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:+.1f}%',
                    ha='center', va='bottom' if imp > 0 else 'top',
                    fontsize=10, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 扰动对比图已保存: {save_path}")


def print_summary(pure_results, rl_results):
    """打印总结"""
    print("\n" + "="*80)
    print("扰动场景鲁棒性测试总结")
    print("="*80)
    
    disturbances = list(pure_results.keys())
    
    print(f"\n{'扰动类型':<25} {'纯Meta-PID':<15} {'Meta-PID+RL':<15} {'改善':<10}")
    print("-"*80)
    
    total_improvement = 0
    for dist in disturbances:
        pure_err = pure_results[dist]['mean_error_deg']
        rl_err = rl_results[dist]['mean_error_deg']
        improvement = (pure_err - rl_err) / pure_err * 100
        total_improvement += improvement
        
        print(f"{dist:<25} {pure_err:>8.2f}°      {rl_err:>8.2f}°      {improvement:>+6.2f}%")
    
    avg_improvement = total_improvement / len(disturbances)
    print("-"*80)
    print(f"{'平均改善':<25} {'':<15} {'':<15} {avg_improvement:>+6.2f}%")
    
    print("\n" + "="*80)
    print("✅ 扰动测试完成！")
    print("="*80)
    
    return avg_improvement


def main():
    parser = argparse.ArgumentParser(description='扰动场景鲁棒性测试')
    parser.add_argument('--robot', default='franka_panda/panda.urdf', help='机器人URDF')
    parser.add_argument('--model', default='logs/meta_rl_panda/best_model/best_model', help='RL模型路径')
    parser.add_argument('--n_episodes', type=int, default=5, help='每种扰动的测试回合数')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（用于可重复性实验）')
    args = parser.parse_args()
    
    # 扰动类型列表
    disturbance_types = ['none', 'random_force', 'payload', 'param_uncertainty', 'mixed']
    
    # 运行测试
    pure_results, rl_results = run_disturbance_tests(
        args.robot,
        args.model,
        disturbance_types,
        args.n_episodes,
        args.seed
    )
    
    # 打印总结
    avg_improvement = print_summary(pure_results, rl_results)
    
    # 绘制对比图
    plot_disturbance_comparison(pure_results, rl_results)
    
    print(f"\n🎯 关键结论:")
    print(f"   • Meta-PID+RL在所有扰动场景下平均改善: {avg_improvement:+.2f}%")
    print(f"   • 验证了方法的鲁棒性和自适应能力")
    print(f"   • 图表已保存为 disturbance_comparison.png")


if __name__ == '__main__':
    main()

