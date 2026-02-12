"""
多场景测试脚本：对比纯PID vs RL+PID在不同轨迹下的性能

测试场景：
1. 慢速圆形（已知RL改进1.71%）
2. 快速圆形
3. 正弦轨迹
4. 快速正弦
5. 阶跃轨迹
6. 8字形轨迹（复杂轨迹）
"""

import yaml
import numpy as np
import argparse
import json
from stable_baselines3 import PPO
from envs.franka_env import FrankaRLPIDEnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 定义测试场景
SCENARIOS = [
    {
        'name': '慢速圆形',
        'type': 'circle',
        'params': {'speed': 0.2, 'radius': 0.15},
        'difficulty': '简单',
        'description': '基准场景，PID应该表现良好'
    },
    {
        'name': '快速圆形',
        'type': 'circle',
        'params': {'speed': 0.5, 'radius': 0.15},
        'difficulty': '中等',
        'description': '高速运动，PID可能响应不够快'
    },
    {
        'name': '正弦轨迹',
        'type': 'sine',
        'params': {'frequency': 0.3, 'amplitude': 0.5},
        'difficulty': '中等',
        'description': '周期性变化，测试跟踪能力'
    },
    {
        'name': '快速正弦',
        'type': 'sine',
        'params': {'frequency': 0.8, 'amplitude': 0.5},
        'difficulty': '困难',
        'description': '高频信号，PID微分项可能引入噪声'
    },
    {
        'name': '阶跃轨迹',
        'type': 'step',
        'params': {'interval': 2.0, 'amplitude': 0.3},
        'difficulty': '困难',
        'description': '不连续轨迹，加速度突变'
    },
    {
        'name': '静态保持',
        'type': 'static',
        'params': {},
        'difficulty': '简单',
        'description': '静态保持，测试稳态性能'
    }
]


def run_single_test(env, model, scenario, num_steps=10000, use_rl=False):
    """
    运行单次测试
    
    Args:
        env: 环境
        model: RL模型（如果use_rl=True）
        scenario: 场景配置
        num_steps: 测试步数
        use_rl: 是否使用RL策略
    
    Returns:
        dict: 测试结果
    """
    obs, _ = env.reset()
    
    errors = []
    rewards = []
    actions = []
    delta_taus = []
    
    for step in range(num_steps):
        if use_rl:
            # 使用RL+PID
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)
        else:
            # 纯PID（action=0）
            action = np.zeros(env.action_space.shape)
        
        obs, reward, done, truncated, info = env.step(action)
        
        # 记录数据
        err_norm = np.linalg.norm(info['tracking_error'])
        errors.append(err_norm)
        rewards.append(reward)
        
        if use_rl and 'delta_tau' in info:
            delta_taus.append(np.linalg.norm(info['delta_tau']))
        
        if done or truncated:
            break
    
    # 计算统计量
    errors = np.array(errors)
    rewards = np.array(rewards)
    
    results = {
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'max_error': float(np.max(errors)),
        'std_error': float(np.std(errors)),
        'total_reward': float(np.sum(rewards)),
        'mean_reward': float(np.mean(rewards)),
        'error_history': errors.tolist()
    }
    
    if use_rl and delta_taus:
        delta_taus = np.array(delta_taus)
        results['mean_delta_tau'] = float(np.mean(delta_taus))
        results['max_delta_tau'] = float(np.max(delta_taus))
        results['mean_action_norm'] = float(np.mean([np.linalg.norm(a) for a in actions]))
    
    return results


def test_all_scenarios(model_path, config_path, num_repeats=3, num_steps=10000):
    """
    测试所有场景
    
    Args:
        model_path: RL模型路径
        config_path: 配置文件路径
        num_repeats: 每个场景重复次数
        num_steps: 每次测试步数
    
    Returns:
        dict: 所有测试结果
    """
    # 加载配置
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # 加载RL模型
    model = PPO.load(model_path.replace('.zip', ''))
    print(f"✅ 模型加载成功: {model_path}")
    
    all_results = {}
    
    for scenario in SCENARIOS:
        print("\n" + "=" * 70)
        print(f"测试场景: {scenario['name']} ({scenario['difficulty']})")
        print(f"描述: {scenario['description']}")
        print("=" * 70)
        
        # 修改配置
        test_config = base_config.copy()
        test_config['trajectory']['type'] = scenario['type']
        test_config['trajectory'].update(scenario['params'])
        
        scenario_results = {
            'name': scenario['name'],
            'type': scenario['type'],
            'difficulty': scenario['difficulty'],
            'description': scenario['description'],
            'params': scenario['params'],
            'pid_results': [],
            'rl_results': []
        }
        
        for repeat in range(num_repeats):
            print(f"\n  重复 {repeat+1}/{num_repeats}...")
            
            # 测试纯PID
            print("    [1/2] 纯PID测试中...")
            env = FrankaRLPIDEnv(test_config, gui=False)
            pid_result = run_single_test(env, None, scenario, num_steps, use_rl=False)
            env.close()
            scenario_results['pid_results'].append(pid_result)
            print(f"          平均误差: {pid_result['mean_error']:.4f}弧度 ({np.rad2deg(pid_result['mean_error']):.2f}度)")
            
            # 测试RL+PID
            print("    [2/2] RL+PID测试中...")
            env = FrankaRLPIDEnv(test_config, gui=False)
            rl_result = run_single_test(env, model, scenario, num_steps, use_rl=True)
            env.close()
            scenario_results['rl_results'].append(rl_result)
            print(f"          平均误差: {rl_result['mean_error']:.4f}弧度 ({np.rad2deg(rl_result['mean_error']):.2f}度)")
        
        # 计算平均性能
        pid_mean_error = np.mean([r['mean_error'] for r in scenario_results['pid_results']])
        rl_mean_error = np.mean([r['mean_error'] for r in scenario_results['rl_results']])
        improvement = (pid_mean_error - rl_mean_error) / pid_mean_error * 100
        
        scenario_results['summary'] = {
            'pid_mean_error': float(pid_mean_error),
            'pid_mean_error_deg': float(np.rad2deg(pid_mean_error)),
            'rl_mean_error': float(rl_mean_error),
            'rl_mean_error_deg': float(np.rad2deg(rl_mean_error)),
            'improvement_percent': float(improvement),
            'rl_mean_delta_tau': float(np.mean([r.get('mean_delta_tau', 0) for r in scenario_results['rl_results']]))
        }
        
        print(f"\n  📊 场景总结:")
        print(f"     纯PID:  {pid_mean_error:.4f}弧度 ({np.rad2deg(pid_mean_error):.2f}度)")
        print(f"     RL+PID: {rl_mean_error:.4f}弧度 ({np.rad2deg(rl_mean_error):.2f}度)")
        print(f"     改进率: {improvement:+.2f}%")
        
        all_results[scenario['name']] = scenario_results
    
    return all_results


def generate_comparison_plots(results, output_dir='results/multi_scenario'):
    """
    生成对比图表
    
    Args:
        results: 测试结果
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取数据
    scenarios = list(results.keys())
    pid_errors = [results[s]['summary']['pid_mean_error_deg'] for s in scenarios]
    rl_errors = [results[s]['summary']['rl_mean_error_deg'] for s in scenarios]
    improvements = [results[s]['summary']['improvement_percent'] for s in scenarios]
    difficulties = [results[s]['difficulty'] for s in scenarios]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 误差对比柱状图
    ax1 = axes[0, 0]
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, pid_errors, width, label='Pure PID', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, rl_errors, width, label='RL+PID', color='coral', alpha=0.8)
    
    ax1.set_xlabel('Trajectory Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Tracking Error (degrees)', fontsize=12, fontweight='bold')
    ax1.set_title('Tracking Error Comparison Across Scenarios', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=15, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}°', ha='center', va='bottom', fontsize=9)
    
    # 2. 改进率柱状图
    ax2 = axes[0, 1]
    colors = ['green' if imp > 10 else 'orange' if imp > 5 else 'lightcoral' for imp in improvements]
    bars = ax2.bar(scenarios, improvements, color=colors, alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Trajectory Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('RL Improvement Rate', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(scenarios, rotation=15, ha='right')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top', fontsize=10)
    
    # 3. 误差分布箱线图
    ax3 = axes[1, 0]
    pid_data = [results[s]['pid_results'][0]['error_history'] for s in scenarios]
    rl_data = [results[s]['rl_results'][0]['error_history'] for s in scenarios]
    
    # 转换为度数
    pid_data_deg = [np.rad2deg(d) for d in pid_data]
    rl_data_deg = [np.rad2deg(d) for d in rl_data]
    
    bp1 = ax3.boxplot(pid_data_deg, positions=np.arange(len(scenarios)) * 2 - 0.4,
                      widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor='steelblue', alpha=0.6),
                      medianprops=dict(color='darkblue', linewidth=2))
    bp2 = ax3.boxplot(rl_data_deg, positions=np.arange(len(scenarios)) * 2 + 0.4,
                      widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor='coral', alpha=0.6),
                      medianprops=dict(color='darkred', linewidth=2))
    
    ax3.set_xlabel('Trajectory Type', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Tracking Error Distribution (degrees)', fontsize=12, fontweight='bold')
    ax3.set_title('Error Distribution Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(np.arange(len(scenarios)) * 2)
    ax3.set_xticklabels(scenarios, rotation=15, ha='right')
    ax3.legend([bp1["boxes"][0], bp2["boxes"][0]], ['PID', 'RL+PID'], loc='upper right')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 难度vs改进率散点图
    ax4 = axes[1, 1]
    difficulty_map = {'简单': 1, '中等': 2, '困难': 3}
    difficulty_values = [difficulty_map[d] for d in difficulties]
    
    scatter = ax4.scatter(difficulty_values, improvements, c=improvements, 
                         cmap='RdYlGn', s=200, alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, txt in enumerate(scenarios):
        ax4.annotate(txt, (difficulty_values[i], improvements[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Scenario Difficulty', fontsize=12, fontweight='bold')
    ax4.set_ylabel('RL Improvement (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Difficulty vs Improvement Correlation', fontsize=14, fontweight='bold')
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(['Easy', 'Medium', 'Hard'])
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax4, label='Improvement (%)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/multi_scenario_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 对比图已保存: {output_dir}/multi_scenario_comparison.png")
    
    return fig


def generate_report(results, output_dir='results/multi_scenario'):
    """
    生成文本报告
    
    Args:
        results: 测试结果
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'{output_dir}/test_report_{timestamp}.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("多场景性能对比测试报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 总体总结
        f.write("【总体总结】\n")
        f.write("-" * 80 + "\n")
        
        all_improvements = [results[s]['summary']['improvement_percent'] for s in results]
        avg_improvement = np.mean(all_improvements)
        max_improvement_scenario = max(results.keys(), key=lambda s: results[s]['summary']['improvement_percent'])
        min_improvement_scenario = min(results.keys(), key=lambda s: results[s]['summary']['improvement_percent'])
        
        f.write(f"总测试场景数: {len(results)}\n")
        f.write(f"平均改进率: {avg_improvement:.2f}%\n")
        f.write(f"最佳改进场景: {max_improvement_scenario} ({results[max_improvement_scenario]['summary']['improvement_percent']:.2f}%)\n")
        f.write(f"最差改进场景: {min_improvement_scenario} ({results[min_improvement_scenario]['summary']['improvement_percent']:.2f}%)\n\n")
        
        # 各场景详细结果
        f.write("【各场景详细结果】\n")
        f.write("=" * 80 + "\n\n")
        
        for scenario_name, data in results.items():
            f.write(f"场景: {scenario_name}\n")
            f.write(f"难度: {data['difficulty']}\n")
            f.write(f"描述: {data['description']}\n")
            f.write(f"轨迹类型: {data['type']}\n")
            f.write(f"参数: {data['params']}\n")
            f.write("-" * 80 + "\n")
            
            summary = data['summary']
            f.write(f"纯PID平均误差:  {summary['pid_mean_error']:.4f}弧度 ({summary['pid_mean_error_deg']:.2f}度)\n")
            f.write(f"RL+PID平均误差: {summary['rl_mean_error']:.4f}弧度 ({summary['rl_mean_error_deg']:.2f}度)\n")
            f.write(f"改进率: {summary['improvement_percent']:+.2f}%\n")
            f.write(f"RL平均补偿力矩: {summary['rl_mean_delta_tau']:.3f} Nm\n")
            f.write("\n")
        
        # 结论与建议
        f.write("【结论与建议】\n")
        f.write("=" * 80 + "\n")
        
        # 分析哪些场景RL有显著优势
        good_scenarios = [s for s in results if results[s]['summary']['improvement_percent'] > 10]
        medium_scenarios = [s for s in results if 5 < results[s]['summary']['improvement_percent'] <= 10]
        poor_scenarios = [s for s in results if results[s]['summary']['improvement_percent'] <= 5]
        
        f.write(f"\nRL显著优势场景 (改进>10%): {len(good_scenarios)}个\n")
        for s in good_scenarios:
            f.write(f"  - {s}: {results[s]['summary']['improvement_percent']:.2f}%\n")
        
        f.write(f"\nRL中等优势场景 (5-10%): {len(medium_scenarios)}个\n")
        for s in medium_scenarios:
            f.write(f"  - {s}: {results[s]['summary']['improvement_percent']:.2f}%\n")
        
        f.write(f"\nRL轻微优势场景 (<5%): {len(poor_scenarios)}个\n")
        for s in poor_scenarios:
            f.write(f"  - {s}: {results[s]['summary']['improvement_percent']:.2f}%\n")
        
        f.write("\n建议:\n")
        if len(good_scenarios) >= 2:
            f.write("✅ RL在多个困难场景下表现出显著优势，值得实际应用\n")
        elif len(medium_scenarios) >= 3:
            f.write("⚠️  RL在大部分场景下有中等改进，可考虑在特定场景使用\n")
        else:
            f.write("❌ RL整体改进有限，可能需要调整训练策略或接受PID已足够好的结论\n")
    
    print(f"✅ 测试报告已保存: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='多场景性能对比测试')
    parser.add_argument('--model', type=str, default='logs/best_model/best_model.zip',
                       help='RL模型路径')
    parser.add_argument('--config', type=str, default='configs/stage1_optimized.yaml',
                       help='配置文件路径')
    parser.add_argument('--repeats', type=int, default=3,
                       help='每个场景重复次数')
    parser.add_argument('--steps', type=int, default=10000,
                       help='每次测试步数')
    parser.add_argument('--output', type=str, default='results/multi_scenario',
                       help='输出目录')
    args = parser.parse_args()
    
    print("=" * 80)
    print("多场景性能对比测试")
    print("=" * 80)
    print(f"RL模型: {args.model}")
    print(f"配置文件: {args.config}")
    print(f"重复次数: {args.repeats}")
    print(f"测试步数: {args.steps}")
    print(f"测试场景数: {len(SCENARIOS)}")
    print(f"总测试次数: {len(SCENARIOS) * args.repeats * 2} (PID + RL)")
    print("=" * 80)
    
    # 运行测试
    results = test_all_scenarios(args.model, args.config, args.repeats, args.steps)
    
    # 保存原始数据
    os.makedirs(args.output, exist_ok=True)
    json_path = f'{args.output}/raw_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 原始数据已保存: {json_path}")
    
    # 生成图表
    generate_comparison_plots(results, args.output)
    
    # 生成报告
    generate_report(results, args.output)
    
    print("\n" + "=" * 80)
    print("✅ 所有测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

