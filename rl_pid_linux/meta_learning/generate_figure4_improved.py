#!/usr/bin/env python3
"""
生成改进的Figure 4：在子图(c)中添加RMSE/MAE误差曲线
"""

import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from meta_rl_combined_env import MetaRLCombinedEnv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle


def setup_publication_style():
    """设置出版级别的图表样式"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'patch.linewidth': 0.5,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'axes.grid': False,
        'grid.alpha': 0.3,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def verify_tracking_error(robot_urdf, model_path=None, steps=10000, test_name=""):
    """
    验证实际跟踪误差
    """
    print(f"\n{'='*80}")
    print(f"评估: {test_name}")
    print(f"{'='*80}")
    
    # 创建环境
    env = MetaRLCombinedEnv(robot_urdf=robot_urdf, gui=False)
    
    # 加载RL模型（如果有）
    model = None
    if model_path is not None:
        try:
            model = PPO.load(model_path)
            print(f"✅ RL模型加载成功")
        except Exception as e:
            print(f"⚠️  RL模型加载失败: {e}")
            print(f"   使用固定Meta-PID")
    else:
        print(f"✅ 使用固定Meta-PID（无RL调整）")
    
    obs, _ = env.reset()
    
    # 记录数据
    actual_errors_deg = []  # 总误差 (角度)
    joint_errors = []  # 每个关节的误差 (弧度)
    mae_history = []  # MAE历史
    rmse_history = []  # RMSE历史
    
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
        joint_error = np.abs(q_ref - q_actual)  # 每个关节的绝对误差（弧度）
        actual_error_rad = np.linalg.norm(q_ref - q_actual)  # 总误差范数（弧度）
        actual_error_deg = np.degrees(actual_error_rad)  # 转换为角度
        
        actual_errors_deg.append(actual_error_deg)
        joint_errors.append(joint_error)
        
        # 计算滑动窗口的MAE和RMSE（用于平滑显示）
        if len(actual_errors_deg) >= 100:
            recent_errors = actual_errors_deg[-100:]
            mae = np.mean(recent_errors)
            rmse = np.sqrt(np.mean(np.array(recent_errors)**2))
        else:
            mae = np.mean(actual_errors_deg)
            rmse = np.sqrt(np.mean(np.array(actual_errors_deg)**2))
        
        mae_history.append(mae)
        rmse_history.append(rmse)
        
        if step % 2000 == 0:
            print(f"Step {step:5d}: 误差={actual_error_deg:.2f}°, MAE={mae:.2f}°, RMSE={rmse:.2f}°")
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
    
    # 统计结果
    actual_errors_deg = np.array(actual_errors_deg)
    joint_errors = np.array(joint_errors)
    mae_history = np.array(mae_history)
    rmse_history = np.array(rmse_history)
    
    results = {
        'actual_errors_deg': actual_errors_deg,
        'joint_errors': joint_errors,
        'mae_history': mae_history,
        'rmse_history': rmse_history,
        'mean_error_deg': np.mean(actual_errors_deg),
        'median_error_deg': np.median(actual_errors_deg),
        'max_error_deg': np.max(actual_errors_deg),
        'std_error_deg': np.std(actual_errors_deg),
        'overall_mae': np.mean(actual_errors_deg),
        'overall_rmse': np.sqrt(np.mean(actual_errors_deg**2)),
    }
    
    print(f"\n📊 {test_name} 实际跟踪性能:")
    print(f"   MAE:    {results['overall_mae']:.2f}°")
    print(f"   RMSE:   {results['overall_rmse']:.2f}°")
    print(f"   Median: {results['median_error_deg']:.2f}°")
    print(f"   Max:    {results['max_error_deg']:.2f}°")
    print(f"   Std:    {results['std_error_deg']:.2f}°")
    
    return results


def plot_comprehensive_comparison(pure_results, rl_results, save_path='Figure4_comprehensive_tracking_performance.png'):
    """
    绘制综合跟踪性能对比图
    改进子图(c): 添加MAE/RMSE随时间变化的曲线
    """
    setup_publication_style()
    
    # 创建2x2子图布局
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 颜色方案
    color_pure = '#4A90E2'  # 蓝色
    color_rl = '#F5A623'    # 橙色
    
    # ========================================================================
    # 子图 (a): Actual Tracking Error Comparison
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 平滑处理
    window = 100
    pure_smooth = np.convolve(pure_results['actual_errors_deg'], 
                              np.ones(window)/window, mode='valid')
    rl_smooth = np.convolve(rl_results['actual_errors_deg'], 
                            np.ones(window)/window, mode='valid')
    
    ax1.plot(pure_smooth, label='Pure Meta-PID', color=color_pure, alpha=0.8, linewidth=1.5)
    ax1.plot(rl_smooth, label='Meta-PID + RL', color=color_rl, alpha=0.8, linewidth=1.5)
    ax1.set_xlabel('Time Step', fontweight='bold')
    ax1.set_ylabel('Tracking Error (degrees)', fontweight='bold')
    ax1.set_title('(a) Actual Tracking Error Comparison', fontweight='bold', loc='left')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 计算改善百分比
    improvement = (pure_results['overall_mae'] - rl_results['overall_mae']) / pure_results['overall_mae'] * 100
    ax1.text(0.98, 0.02, f'{improvement:.1f}% improvement with RL adaptation', 
             transform=ax1.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9, fontweight='bold')
    
    # ========================================================================
    # 子图 (b): Error Distribution
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.hist(pure_results['actual_errors_deg'], bins=50, alpha=0.6, 
            color=color_pure, label='Pure Meta-PID', density=True, edgecolor='black', linewidth=0.5)
    ax2.hist(rl_results['actual_errors_deg'], bins=50, alpha=0.6, 
            color=color_rl, label='Meta-PID + RL', density=True, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Tracking Error (degrees)', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title('(b) Error Distribution', fontweight='bold', loc='left')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 添加均值线
    ax2.axvline(pure_results['overall_mae'], color=color_pure, linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvline(rl_results['overall_mae'], color=color_rl, linestyle='--', linewidth=2, alpha=0.7)
    
    # ========================================================================
    # 子图 (c): Per-Joint Error Comparison with Improvement Curve (双Y轴)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # 计算各关节平均误差
    mean_joint_errors_pure = np.mean(pure_results['joint_errors'], axis=0)
    mean_joint_errors_rl = np.mean(rl_results['joint_errors'], axis=0)
    
    n_joints = len(mean_joint_errors_pure)
    x = np.arange(n_joints) + 1  # Joint indices starting from 1
    width = 0.35
    
    # 左Y轴：误差值柱状图
    bars1 = ax3.bar(x - width/2, np.degrees(mean_joint_errors_pure), width, 
                     label='Pure Meta-PID', color=color_pure, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax3.bar(x + width/2, np.degrees(mean_joint_errors_rl), width, 
                     label='Meta-PID + RL', color=color_rl, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax3.set_xlabel('Joint Index', fontweight='bold')
    ax3.set_ylabel('Mean Absolute Error (degrees)', fontweight='bold', color='black')
    ax3.set_title('(c) Per-Joint Error Comparison', fontweight='bold', loc='left')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'J{i}' for i in x])
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 创建右Y轴：改进百分比曲线
    ax3_twin = ax3.twinx()
    
    # 计算每个关节的改进百分比
    improvement_percentages = []
    for i in range(n_joints):
        pure_err = np.degrees(mean_joint_errors_pure[i])
        rl_err = np.degrees(mean_joint_errors_rl[i])
        if pure_err > 0:
            improvement_pct = (pure_err - rl_err) / pure_err * 100
        else:
            improvement_pct = 0
        improvement_percentages.append(improvement_pct)
    
    improvement_percentages = np.array(improvement_percentages)
    
    # 绘制改进百分比曲线（使用深绿色）
    color_improvement = '#2E7D32'  # 深绿色
    line = ax3_twin.plot(x, improvement_percentages, 
                         color=color_improvement, marker='o', markersize=6,
                         linewidth=2.5, label='Improvement (%)', 
                         linestyle='-', alpha=0.9, zorder=10)
    
    # 在数据点上标注改善百分比（J2放在上方，其他放在下方）
    for i, (xi, yi) in enumerate(zip(x, improvement_percentages)):
        if abs(yi) > 1:  # 只显示改善超过1%的
            color_text = 'green' if yi > 0 else 'red'
            
            # J2（i=1，因为索引从0开始）放在曲线上方，其他放在下方
            if i == 1:  # J2
                y_offset = yi + 2.5
                va = 'bottom'
            else:  # 其他关节
                y_offset = yi - 3.0
                va = 'top'
            
            ax3_twin.text(xi, y_offset, f'{yi:+.1f}%', 
                         ha='center', va=va, fontsize=7, 
                         color=color_text, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor=color_text, alpha=0.7, linewidth=1))
    
    ax3_twin.set_ylabel('Improvement (%)', fontweight='bold', color=color_improvement)
    ax3_twin.tick_params(axis='y', labelcolor=color_improvement)
    ax3_twin.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
    
    # 设置右Y轴范围（为下方标注留出更多空间）
    max_abs_improvement = max(abs(improvement_percentages.min()), abs(improvement_percentages.max()))
    ax3_twin.set_ylim(-max_abs_improvement * 0.5, max_abs_improvement * 1.3)
    
    # 合并图例（放在中间上方，横向排列，避免遮挡数据）
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper center',           # 位置：上方中间
              bbox_to_anchor=(0.5, 0.7),   # 精确位置：水平中心(0.5), 图表内部上方
              framealpha=0.95,              # 背景透明度
              fontsize=8,                   # 字体大小
              edgecolor='gray',             # 边框颜色
              fancybox=True)                # 圆角边框
    
    # 添加改善信息文本框
    joints_improved = np.sum(improvement_percentages > 0)
    avg_joint_improvement = np.mean(improvement_percentages[improvement_percentages > 0]) if joints_improved > 0 else 0
    max_improvement_joint = np.argmax(improvement_percentages) + 1
    max_improvement_value = improvement_percentages[np.argmax(improvement_percentages)]
    
    info_text = f'Joint {max_improvement_joint} benefits most: {max_improvement_value:.1f}% improvement\n{joints_improved}/{n_joints} joints improved, avg {avg_joint_improvement:.1f}%'
    ax3.text(0.98, 0.98, info_text,
             transform=ax3.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6, edgecolor='darkgreen'),
             fontsize=7, fontweight='bold')
    
    # ========================================================================
    # 子图 (d): Cumulative Distribution Function
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    pure_sorted = np.sort(pure_results['actual_errors_deg'])
    rl_sorted = np.sort(rl_results['actual_errors_deg'])
    pure_cdf = np.arange(1, len(pure_sorted)+1) / len(pure_sorted)
    rl_cdf = np.arange(1, len(rl_sorted)+1) / len(rl_sorted)
    
    ax4.plot(pure_sorted, pure_cdf, label='Pure Meta-PID', 
            color=color_pure, linewidth=2, alpha=0.8)
    ax4.plot(rl_sorted, rl_cdf, label='Meta-PID + RL', 
            color=color_rl, linewidth=2, alpha=0.8)
    ax4.set_xlabel('Tracking Error (degrees)', fontweight='bold')
    ax4.set_ylabel('Cumulative Probability', fontweight='bold')
    ax4.set_title('(d) Cumulative Distribution Function', fontweight='bold', loc='left')
    ax4.legend(loc='lower right', framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # 标注关键百分位数
    for percentile in [50, 90]:
        pure_val = np.percentile(pure_results['actual_errors_deg'], percentile)
        rl_val = np.percentile(rl_results['actual_errors_deg'], percentile)
        improvement_pct = (pure_val - rl_val) / pure_val * 100
        
        y_pos = percentile / 100
        ax4.axhline(y_pos, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax4.text(ax4.get_xlim()[1] * 0.98, y_pos, f'{percentile}th: {improvement_pct:+.1f}%', 
                ha='right', va='bottom', fontsize=7, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    # 不添加总标题和底部注释（在LaTeX中说明）
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存: {save_path}")
    
    # 同时保存PDF版本
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✅ PDF版本已保存: {pdf_path}")
    
    plt.close()


def main():
    """主函数"""
    robot_urdf = 'franka_panda/panda.urdf'
    model_path = 'logs/meta_rl_panda/best_model/best_model'
    steps = 10000
    
    print("="*80)
    print("生成 Figure 4: Comprehensive Tracking Performance Comparison")
    print("="*80)
    print(f"机器人: Franka Panda")
    print(f"测试步数: {steps}")
    print()
    
    # 评估纯Meta-PID
    print("\n" + "="*80)
    print("1/2: 评估 Pure Meta-PID")
    print("="*80)
    pure_results = verify_tracking_error(
        robot_urdf=robot_urdf,
        model_path=None,
        steps=steps,
        test_name="Pure Meta-PID"
    )
    
    # 评估Meta-PID + RL
    print("\n" + "="*80)
    print("2/2: 评估 Meta-PID + RL")
    print("="*80)
    rl_results = verify_tracking_error(
        robot_urdf=robot_urdf,
        model_path=model_path,
        steps=steps,
        test_name="Meta-PID + RL"
    )
    
    # 绘制对比图
    print("\n" + "="*80)
    print("生成综合对比图")
    print("="*80)
    plot_comprehensive_comparison(
        pure_results=pure_results,
        rl_results=rl_results,
        save_path='Figure4_comprehensive_tracking_performance.png'
    )
    
    # 打印总结
    print("\n" + "="*80)
    print("📊 性能对比总结")
    print("="*80)
    print(f"\n{'指标':<20} {'Pure Meta-PID':>15} {'Meta-PID + RL':>15} {'改善':>12}")
    print("-" * 70)
    
    mae_improvement = (pure_results['overall_mae'] - rl_results['overall_mae']) / pure_results['overall_mae'] * 100
    rmse_improvement = (pure_results['overall_rmse'] - rl_results['overall_rmse']) / pure_results['overall_rmse'] * 100
    max_improvement = (pure_results['max_error_deg'] - rl_results['max_error_deg']) / pure_results['max_error_deg'] * 100
    
    print(f"{'MAE (°)':<20} {pure_results['overall_mae']:>15.2f} {rl_results['overall_mae']:>15.2f} {mae_improvement:>11.1f}%")
    print(f"{'RMSE (°)':<20} {pure_results['overall_rmse']:>15.2f} {rl_results['overall_rmse']:>15.2f} {rmse_improvement:>11.1f}%")
    print(f"{'Median (°)':<20} {pure_results['median_error_deg']:>15.2f} {rl_results['median_error_deg']:>15.2f}")
    print(f"{'Max (°)':<20} {pure_results['max_error_deg']:>15.2f} {rl_results['max_error_deg']:>15.2f} {max_improvement:>11.1f}%")
    print(f"{'Std (°)':<20} {pure_results['std_error_deg']:>15.2f} {rl_results['std_error_deg']:>15.2f}")
    
    print("\n" + "="*80)
    print("✅ 完成！")
    print("="*80)
    print("\n生成的文件:")
    print("  - Figure4_comprehensive_tracking_performance.png")
    print("  - Figure4_comprehensive_tracking_performance.pdf")
    print("\n这些图表可以直接用于论文！")


if __name__ == '__main__':
    main()

