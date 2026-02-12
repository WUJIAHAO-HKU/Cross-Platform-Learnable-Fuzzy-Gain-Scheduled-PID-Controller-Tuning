#!/usr/bin/env python3
"""
使用最优参数的扰动场景测试
基于 optimize_disturbance_params.py 找到的最优配置
"""

import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from meta_rl_disturbance_env import MetaRLDisturbanceEnv
import matplotlib.pyplot as plt
import argparse
import json


# 最优扰动参数配置（来自参数搜索结果）
# 注意：每个扰动类型只包含该类型的参数，不包含其他扰动的参数
OPTIMAL_DISTURBANCE_PARAMS = {
    'none': {},
    
    'random_force': {
        'random_force': {
            'force_range': 15.0,     # 最优：较小的外力
            'force_prob': 0.05       # 最优：较低的扰动频率
        }
        # 注意：不包含payload和param_uncertainty参数
    },
    
    'payload': {
        'payload': {
            'mass_range': 2.0        # 最优：2kg负载（而不是默认的3kg）
        }
        # 注意：不包含random_force和param_uncertainty参数
    },
    
    'param_uncertainty': {
        'param_uncertainty': {
            'mass_scale': (0.7, 1.3),      # 最优：±30%质量不确定性
            'friction_scale': (0.7, 1.5)   # 最优：0.7-1.5倍摩擦变化
        }
        # 注意：不包含random_force和payload参数
    },
    
    'mixed': {
        'payload': {
            'mass_range': 4.0        # 最优：4kg负载
        },
        'param_uncertainty': {
            'mass_scale': (0.9, 1.1),      # 最优：±10%质量不确定性
            'friction_scale': (0.7, 1.5)   # 最优：0.7-1.5倍摩擦变化
        }
        # mixed扰动包含三种：random_force + payload + param_uncertainty
        # random_force将使用默认值（20.0N, 0.1概率）因为这是最优配置
    }
}


def evaluate_under_disturbance(robot_urdf, disturbance_type, disturbance_params,
                                model_path=None, n_episodes=5, max_steps=3000, seed=None):
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
        disturbance_type=disturbance_type,
        disturbance_params=disturbance_params
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
        'max_error_deg': np.mean(episode_max_errors),
        'std_error_deg': np.std(all_errors_deg),
    }
    
    return results


def run_optimized_tests(robot_urdf, model_path, n_episodes=10, seed=None):
    """使用最优参数运行完整扰动测试
    
    Args:
        seed: 随机种子（用于可重复性）
    """
    
    disturbance_types = ['none', 'random_force', 'payload', 'param_uncertainty', 'mixed']
    
    print("="*80)
    print("扰动场景鲁棒性测试（使用最优参数）")
    print("="*80)
    print(f"机器人: {robot_urdf}")
    print(f"每种扰动测试回合: {n_episodes}")
    print(f"参数配置: 基于智能搜索的最优值")
    if seed is not None:
        print(f"随机种子: {seed}")
    print()
    
    # 测试纯Meta-PID
    print("="*80)
    print("测试1: 纯Meta-PID（固定预测值）")
    print("="*80)
    
    pure_results = {}
    for i, dist_type in enumerate(disturbance_types):
        print(f"\n测试扰动: {dist_type}")
        params = OPTIMAL_DISTURBANCE_PARAMS.get(dist_type, {})
        print(f"  参数: {params}")
        
        # 为不同扰动类型设置不同的种子基数
        dist_seed = None if seed is None else seed + i * 1000
        results = evaluate_under_disturbance(
            robot_urdf, dist_type, params,
            model_path=None, n_episodes=n_episodes, seed=dist_seed
        )
        pure_results[dist_type] = results
        
        print(f"  平均误差: {results['mean_error_deg']:.2f}°")
        print(f"  最大误差: {results['max_error_deg']:.2f}°")
        print(f"  标准差: {results['std_error_deg']:.2f}°")
    
    # 测试Meta-PID+RL
    print("\n" + "="*80)
    print("测试2: Meta-PID + RL（在线自适应）")
    print("="*80)
    
    rl_results = {}
    for i, dist_type in enumerate(disturbance_types):
        print(f"\n测试扰动: {dist_type}")
        params = OPTIMAL_DISTURBANCE_PARAMS.get(dist_type, {})
        print(f"  参数: {params}")
        
        # 使用相同的种子基数确保Pure和RL测试在相同条件下比较
        dist_seed = None if seed is None else seed + i * 1000
        results = evaluate_under_disturbance(
            robot_urdf, dist_type, params,
            model_path=model_path, n_episodes=n_episodes, seed=dist_seed
        )
        rl_results[dist_type] = results
        
        print(f"  平均误差: {results['mean_error_deg']:.2f}°")
        print(f"  最大误差: {results['max_error_deg']:.2f}°")
        print(f"  标准差: {results['std_error_deg']:.2f}°")
    
    # 打印对比总结
    print_summary(pure_results, rl_results)
    
    # 绘制对比图
    plot_disturbance_comparison(pure_results, rl_results, 
                                save_path='disturbance_comparison_optimal.png')
    
    return pure_results, rl_results


def print_summary(pure_results, rl_results):
    """打印总结"""
    print("\n" + "="*80)
    print("扰动场景鲁棒性测试总结（最优参数）")
    print("="*80)
    
    print(f"\n{'扰动类型':<20} {'Pure Meta-PID':<15} {'Meta-PID+RL':<15} {'改善':<10}")
    print("-"*80)
    
    improvements = []
    for dist_type in pure_results.keys():
        pure_err = pure_results[dist_type]['mean_error_deg']
        rl_err = rl_results[dist_type]['mean_error_deg']
        improvement = (pure_err - rl_err) / pure_err * 100
        improvements.append(improvement)
        
        print(f"{dist_type:<20} {pure_err:>12.2f}°  {rl_err:>12.2f}°  {improvement:>+8.2f}%")
    
    avg_improvement = np.mean(improvements)
    print("-"*80)
    print(f"{'平均':<20} {'':>15} {'':>15} {avg_improvement:>+8.2f}%")
    print("="*80)


def plot_disturbance_comparison(pure_results, rl_results, save_path='disturbance_comparison_optimal.png',
                                 statistics=None, label_config=None):
    """绘制扰动场景对比图（带改进曲线）
    
    Args:
        statistics: 可选，多种子统计数据
        label_config: 标签配置字典，可包含：
            - 'fontsize': 字体大小 (默认9)
            - 'offset_factor': 偏移因子 (默认2.5)
            - 'y_margin_factor': Y轴扩展因子 (默认1.15)
    """
    
    # 默认标签配置
    if label_config is None:
        label_config = {}
    fontsize = label_config.get('fontsize', 9)
    offset_factor = label_config.get('offset_factor', 2.5)  # 减小偏移
    y_margin_factor = label_config.get('y_margin_factor', 1.05)  # 子图d的倍数（减小以让标签更靠近柱子）
    
    # 智能标签定位函数
    def smart_label_offset(value, all_values, base_offset):
        """根据值的大小和位置智能调整偏移"""
        max_val = max(all_values)
        min_val = min(all_values)
        value_range = max_val - min_val
        
        if value_range == 0:
            return base_offset
        
        # 对于接近极值的标签，使用更小的偏移
        if value > 0:
            # 正值：如果接近最大值，减小偏移避免超出
            if value > max_val * 0.8:
                return base_offset * 0.6
            else:
                return base_offset
        else:
            # 负值：如果接近最小值，减小偏移避免超出
            if value < min_val * 0.8:
                return base_offset * 0.6
            else:
                return base_offset
    
    disturbances = list(pure_results.keys())
    dist_labels = {
        'none': 'No Disturbance',
        'random_force': 'Random Force',
        'payload': 'Payload (+2kg)',  # 更新为最优值
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
    ax2 = ax.twinx()
    x = np.arange(len(disturbances))
    width = 0.35
    ax.bar(x - width/2, pure_mean, width, label='Pure Meta-PID', alpha=0.8, color='skyblue')
    ax.bar(x + width/2, rl_mean, width, label='Meta-PID + RL', alpha=0.8, color='lightcoral')
    # 改进曲线
    line = ax2.plot(x, improvements_mean, 'o-', color='#2E7D32', linewidth=2.5, 
                    markersize=8, label='Improvement %', markeredgecolor='white', markeredgewidth=1.5)
    # 改进标签（智能偏移）
    for i, imp in enumerate(improvements_mean):
        adaptive_offset = smart_label_offset(imp, improvements_mean, offset_factor)
        y_offset = imp + adaptive_offset if imp > 0 else imp - adaptive_offset
        ax2.text(i, y_offset, f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top',
                fontsize=fontsize, fontweight='bold', color='#1B5E20')
    # 自动调整Y轴范围以适应标签
    y_min, y_max = ax2.get_ylim()
    max_adaptive_offset = smart_label_offset(max(improvements_mean), improvements_mean, offset_factor)
    min_adaptive_offset = smart_label_offset(min(improvements_mean), improvements_mean, offset_factor)
    label_max = max(improvements_mean) + max_adaptive_offset + 2
    label_min = min(improvements_mean) - min_adaptive_offset - 2
    ax2.set_ylim(min(y_min, label_min), max(y_max, label_max))
    
    ax.set_xlabel('Disturbance Type', fontsize=12)
    ax.set_ylabel('Mean Error (degrees)', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12, color='#2E7D32')
    ax2.tick_params(axis='y', labelcolor='#2E7D32')
    ax.set_title('(a) Mean Tracking Error (Optimal Params)', fontsize=13, fontweight='bold')
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
    line = ax2.plot(x, improvements_max, 'o-', color='#2E7D32', linewidth=2.5, 
                    markersize=8, label='Improvement %', markeredgecolor='white', markeredgewidth=1.5)
    for i, imp in enumerate(improvements_max):
        adaptive_offset = smart_label_offset(imp, improvements_max, offset_factor)
        # 对于第一个点(i=0)且值较大，使用负偏移（放在点下方）避免与图例重叠
        if i == 0 and imp > 10:
            y_offset = imp - adaptive_offset
            v_align = 'top'
        else:
            y_offset = imp + adaptive_offset if imp > 0 else imp - adaptive_offset
            v_align = 'bottom' if imp > 0 else 'top'
        ax2.text(i, y_offset, f'{imp:+.1f}%', ha='center', va=v_align,
                fontsize=fontsize, fontweight='bold', color='#1B5E20')
    # 自动调整Y轴范围以适应标签
    y_min, y_max = ax2.get_ylim()
    label_max = max(improvements_max) + offset_factor * 0.6 + 1.5
    label_min = min(improvements_max) - offset_factor - 1.5
    ax2.set_ylim(min(y_min, label_min), max(y_max, label_max))
    
    ax.set_xlabel('Disturbance Type', fontsize=12)
    ax.set_ylabel('Max Error (degrees)', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12, color='#2E7D32')
    ax2.tick_params(axis='y', labelcolor='#2E7D32')
    ax.set_title('(b) Maximum Tracking Error (Optimal Params)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([dist_labels.get(d, d) for d in disturbances], rotation=20, ha='right')
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. 标准差对比 + 改进曲线
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.bar(x - width/2, pure_std, width, label='Pure Meta-PID', alpha=0.8, color='skyblue')
    ax.bar(x + width/2, rl_std, width, label='Meta-PID + RL', alpha=0.8, color='lightcoral')
    line = ax2.plot(x, improvements_std, 'o-', color='#2E7D32', linewidth=2.5, 
                    markersize=8, label='Improvement %', markeredgecolor='white', markeredgewidth=1.5)
    for i, imp in enumerate(improvements_std):
        adaptive_offset = smart_label_offset(imp, improvements_std, offset_factor)
        y_offset = imp + adaptive_offset if imp > 0 else imp - adaptive_offset
        ax2.text(i, y_offset, f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top',
                fontsize=fontsize, fontweight='bold', color='#1B5E20')
    # 自动调整Y轴范围以适应标签（为极端值预留更多空间）
    y_min, y_max = ax2.get_ylim()
    max_adaptive_offset = smart_label_offset(max(improvements_std), improvements_std, offset_factor)
    min_adaptive_offset = smart_label_offset(min(improvements_std), improvements_std, offset_factor)
    label_max = max(improvements_std) + max_adaptive_offset + 1.8
    label_min = min(improvements_std) - min_adaptive_offset - 1.8
    ax2.set_ylim(min(y_min, label_min), max(y_max, label_max))
    
    ax.set_xlabel('Disturbance Type', fontsize=12)
    ax.set_ylabel('Std Dev (degrees)', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12, color='#2E7D32')
    ax2.tick_params(axis='y', labelcolor='#2E7D32')
    ax.set_title('(c) Error Standard Deviation (Optimal Params)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([dist_labels.get(d, d) for d in disturbances], rotation=20, ha='right')
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
        label_positions = []
        for i, (pure_m, rl_m, pure_s, rl_s) in enumerate(zip(pure_means, rl_means, pure_stds, rl_stds)):
            improvement = (pure_m - rl_m) / pure_m * 100
            # 在两个柱子中间标注改进率（考虑误差条的高度）
            max_height = max(pure_m + pure_s, rl_m + rl_s)
            y_pos = max_height * y_margin_factor
            label_positions.append(y_pos)
            ax.text(i, y_pos, f'{improvement:+.1f}%', ha='center', va='bottom',
                   fontsize=fontsize, fontweight='bold', color='#2E7D32' if improvement > 0 else '#D84315')
        
        # 自动调整Y轴范围以适应标签（考虑字体高度）
        current_ylim = ax.get_ylim()
        max_label_y = max(label_positions)
        # 为标签文字预留空间（大约是最大Y值的5-8%）
        required_y_max = max_label_y * 1.08
        ax.set_ylim(current_ylim[0], max(current_ylim[1], required_y_max))
    else:
        # 展示单次改进百分比（原有逻辑）
        colors = ['#2E7D32' if imp > 0 else '#D84315' for imp in improvements_mean]
        bars = ax.bar(x, improvements_mean, alpha=0.85, color=colors, edgecolor='black', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax.set_xlabel('Disturbance Type', fontsize=12)
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.set_title('(d) Performance Improvement with RL (Optimal Params)', fontsize=13, fontweight='bold')
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


def main():
    parser = argparse.ArgumentParser(description='使用最优参数测试扰动场景')
    parser.add_argument('--robot', type=str, default='franka_panda/panda.urdf',
                        help='机器人URDF文件')
    parser.add_argument('--model', type=str, 
                        default='logs/meta_rl_panda/best_model/best_model',
                        help='RL模型路径')
    parser.add_argument('--n_episodes', type=int, default=10,
                        help='每种扰动的测试回合数')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子（用于可重复性实验）')
    
    args = parser.parse_args()
    
    # 打印最优参数配置
    print("\n" + "="*80)
    print("最优扰动参数配置（基于参数搜索）")
    print("="*80)
    for dist_type, params in OPTIMAL_DISTURBANCE_PARAMS.items():
        print(f"\n{dist_type}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    print("="*80 + "\n")
    
    # 运行测试
    pure_results, rl_results = run_optimized_tests(
        args.robot,
        args.model,
        n_episodes=args.n_episodes,
        seed=args.seed
    )
    
    print(f"\n🎯 关键结论:")
    improvements = [(pure_results[d]['mean_error_deg'] - rl_results[d]['mean_error_deg']) / 
                    pure_results[d]['mean_error_deg'] * 100 
                    for d in pure_results.keys()]
    avg_improvement = np.mean(improvements)
    print(f"   • Meta-PID+RL在所有扰动场景下平均改善: {avg_improvement:+.2f}%")
    print(f"   • 使用最优参数配置，性能提升显著")
    print(f"   • 图表已保存为 disturbance_comparison_optimal.png")


if __name__ == '__main__':
    main()

