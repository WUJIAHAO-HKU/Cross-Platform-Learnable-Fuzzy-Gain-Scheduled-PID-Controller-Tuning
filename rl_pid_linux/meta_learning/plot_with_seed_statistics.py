#!/usr/bin/env python3
"""
使用种子搜索结果生成带有多种子统计的扰动对比图
子图(d)展示基于所有种子的统计信息（均值±标准差）
"""

import json
import argparse
from test_with_optimal_params import (
    evaluate_under_disturbance, 
    OPTIMAL_DISTURBANCE_PARAMS,
    plot_disturbance_comparison
)


def generate_plot_with_statistics(seed_results_file, best_seed, n_episodes=20, 
                                   save_path='disturbance_comparison_with_stats.png',
                                   label_config=None):
    """
    使用最佳种子进行测试，并使用多种子统计信息绘图
    
    Args:
        seed_results_file: 种子搜索结果JSON文件
        best_seed: 用于测试的最佳种子
        n_episodes: 测试的episode数
        save_path: 图表保存路径
        label_config: 标签配置字典，可包含：
            - 'fontsize': 字体大小 (默认9)
            - 'offset_factor': 偏移因子 (默认2.5)
            - 'y_margin_factor': Y轴扩展因子 (默认1.25)
    """
    # 加载种子搜索结果
    with open(seed_results_file, 'r') as f:
        seed_data = json.load(f)
    
    statistics = seed_data.get('statistics')
    if statistics is None:
        print("⚠️ 种子搜索结果中没有statistics字段，请使用新版find_best_seed.py重新搜索")
        return
    
    robot_urdf = seed_data['robot_urdf']
    model_path = seed_data['model_path']
    
    print("="*80)
    print("生成带有多种子统计的扰动对比图")
    print("="*80)
    print(f"种子搜索文件: {seed_results_file}")
    print(f"使用最佳种子: {best_seed}")
    print(f"测试episodes: {n_episodes}")
    print(f"总种子数: {seed_data['total_seeds_tested']}")
    print("="*80 + "\n")
    
    # 使用最佳种子测试每种扰动
    disturbance_types = ['none', 'random_force', 'payload', 'param_uncertainty', 'mixed']
    
    print("🔬 测试纯Meta-PID...")
    pure_results = {}
    for i, dist_type in enumerate(disturbance_types):
        params = OPTIMAL_DISTURBANCE_PARAMS.get(dist_type, {})
        dist_seed = best_seed + i * 1000
        
        result = evaluate_under_disturbance(
            robot_urdf, dist_type, params,
            model_path=None, n_episodes=n_episodes, seed=dist_seed
        )
        pure_results[dist_type] = result
        print(f"  {dist_type:<20}: {result['mean_error_deg']:.2f}°")
    
    print("\n🔬 测试Meta-PID+RL...")
    rl_results = {}
    for i, dist_type in enumerate(disturbance_types):
        params = OPTIMAL_DISTURBANCE_PARAMS.get(dist_type, {})
        dist_seed = best_seed + i * 1000
        
        result = evaluate_under_disturbance(
            robot_urdf, dist_type, params,
            model_path=model_path, n_episodes=n_episodes, seed=dist_seed
        )
        rl_results[dist_type] = result
        print(f"  {dist_type:<20}: {result['mean_error_deg']:.2f}°")
    
    # 生成图表（子图d使用多种子统计）
    print(f"\n📊 生成图表...")
    plot_disturbance_comparison(pure_results, rl_results, 
                                save_path=save_path,
                                statistics=statistics,
                                label_config=label_config)
    
    print(f"\n✅ 完成！图表已保存: {save_path}")
    print("="*80)
    print("📖 图表说明:")
    print("  子图(a): 平均误差 + 改进曲线（单次测试）")
    print("  子图(b): 最大误差 + 改进曲线（单次测试）")
    print("  子图(c): 误差标准差 + 改进曲线（单次测试）")
    print(f"  子图(d): 多种子统计对比 (基于{seed_data['total_seeds_tested']}个种子)")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='生成带有多种子统计的扰动对比图')
    parser.add_argument('--seed_results', type=str, default='seed_search_results.json',
                        help='种子搜索结果JSON文件')
    parser.add_argument('--best_seed', type=int, default=None,
                        help='最佳种子（如果不指定，自动从结果文件读取）')
    parser.add_argument('--n_episodes', type=int, default=20,
                        help='测试的episode数')
    parser.add_argument('--output', type=str, default='disturbance_comparison_with_stats.png',
                        help='输出图表路径')
    
    # 标签配置参数
    parser.add_argument('--fontsize', type=float, default=9,
                        help='改进标签字体大小 (默认9)')
    parser.add_argument('--offset_factor', type=float, default=2.5,
                        help='子图a/b/c改进标签偏移因子 (默认2.5)')
    parser.add_argument('--y_margin_factor', type=float, default=1.25,
                        help='子图d标签Y轴位置倍数 (默认1.25)')
    
    args = parser.parse_args()
    
    # 如果没有指定best_seed，从结果文件读取
    if args.best_seed is None:
        with open(args.seed_results, 'r') as f:
            seed_data = json.load(f)
        args.best_seed = seed_data['best_seed']
        print(f"✅ 自动使用最佳种子: {args.best_seed}")
    
    # 构建标签配置
    label_config = {
        'fontsize': args.fontsize,
        'offset_factor': args.offset_factor,
        'y_margin_factor': args.y_margin_factor
    }
    
    print(f"📊 标签配置: 字体{args.fontsize} | 偏移因子{args.offset_factor} | Y轴倍数{args.y_margin_factor}")
    
    generate_plot_with_statistics(
        args.seed_results,
        args.best_seed,
        args.n_episodes,
        args.output,
        label_config
    )


if __name__ == '__main__':
    main()

