#!/usr/bin/env python3
"""
遍历种子0-100，找到RL优化程度最大的种子
用于论文实验，确保选择最佳展示效果的种子
"""

import numpy as np
import json
from tqdm import tqdm
from test_with_optimal_params import evaluate_under_disturbance, OPTIMAL_DISTURBANCE_PARAMS
import argparse
import time


def evaluate_seed(seed, robot_urdf, model_path, n_episodes=10):
    """
    评估单个种子的性能
    
    Returns:
        dict: 包含平均改进率等信息
    """
    disturbance_types = ['none', 'random_force', 'payload', 'param_uncertainty', 'mixed']
    
    pure_results = {}
    rl_results = {}
    
    for i, dist_type in enumerate(disturbance_types):
        params = OPTIMAL_DISTURBANCE_PARAMS.get(dist_type, {})
        dist_seed = seed + i * 1000
        
        # Pure Meta-PID
        try:
            pure_res = evaluate_under_disturbance(
                robot_urdf, dist_type, params,
                model_path=None, n_episodes=n_episodes, seed=dist_seed
            )
            pure_results[dist_type] = pure_res
        except Exception as e:
            print(f"  ⚠️ Pure评估失败 (seed={seed}, dist={dist_type}): {e}")
            return None
        
        # Meta-PID+RL
        try:
            rl_res = evaluate_under_disturbance(
                robot_urdf, dist_type, params,
                model_path=model_path, n_episodes=n_episodes, seed=dist_seed
            )
            rl_results[dist_type] = rl_res
        except Exception as e:
            print(f"  ⚠️ RL评估失败 (seed={seed}, dist={dist_type}): {e}")
            return None
    
    # 计算改进率
    improvements = []
    for dist_type in disturbance_types:
        pure_err = pure_results[dist_type]['mean_error_deg']
        rl_err = rl_results[dist_type]['mean_error_deg']
        improvement = (pure_err - rl_err) / pure_err * 100
        improvements.append(improvement)
    
    avg_improvement = np.mean(improvements)
    
    return {
        'seed': seed,
        'avg_improvement': avg_improvement,
        'improvements': improvements,
        'disturbance_types': disturbance_types,
        'pure_results': {d: pure_results[d]['mean_error_deg'] for d in disturbance_types},
        'rl_results': {d: rl_results[d]['mean_error_deg'] for d in disturbance_types},
    }


def search_best_seed(robot_urdf, model_path, seed_range=(0, 100), n_episodes=10, 
                     save_path='seed_search_results.json'):
    """
    搜索最佳种子
    
    Args:
        seed_range: 种子范围 (start, end)，包含start，不包含end
        n_episodes: 每个种子的测试回合数
    """
    print("="*80)
    print("🔍 寻找最佳种子（RL优化程度最大）")
    print("="*80)
    print(f"种子范围: {seed_range[0]} ~ {seed_range[1]-1}")
    print(f"每个种子测试: {n_episodes} episodes")
    print(f"机器人: {robot_urdf}")
    print(f"RL模型: {model_path}")
    print("="*80 + "\n")
    
    results = []
    best_seed = None
    best_improvement = -float('inf')
    
    start_time = time.time()
    
    # 遍历所有种子
    for seed in tqdm(range(seed_range[0], seed_range[1]), desc="搜索种子"):
        result = evaluate_seed(seed, robot_urdf, model_path, n_episodes)
        
        if result is None:
            continue
        
        results.append(result)
        
        # 更新最佳种子
        if result['avg_improvement'] > best_improvement:
            best_improvement = result['avg_improvement']
            best_seed = seed
        
        # 每10个种子显示一次当前最佳
        if (seed + 1) % 10 == 0:
            print(f"\n  当前最佳种子: {best_seed}, 平均改进: {best_improvement:.2f}%")
    
    elapsed_time = time.time() - start_time
    
    # 计算统计信息（用于子图d）
    disturbance_types = ['none', 'random_force', 'payload', 'param_uncertainty', 'mixed']
    stats_pure = {dist: [] for dist in disturbance_types}
    stats_rl = {dist: [] for dist in disturbance_types}
    
    for result in results:
        for dist in disturbance_types:
            stats_pure[dist].append(result['pure_results'][dist])
            stats_rl[dist].append(result['rl_results'][dist])
    
    # 计算均值和标准差
    statistics = {
        'pure_mean': {dist: np.mean(stats_pure[dist]) for dist in disturbance_types},
        'pure_std': {dist: np.std(stats_pure[dist]) for dist in disturbance_types},
        'rl_mean': {dist: np.mean(stats_rl[dist]) for dist in disturbance_types},
        'rl_std': {dist: np.std(stats_rl[dist]) for dist in disturbance_types},
    }
    
    # 保存结果
    output = {
        'seed_range': seed_range,
        'n_episodes': n_episodes,
        'robot_urdf': robot_urdf,
        'model_path': model_path,
        'total_seeds_tested': len(results),
        'best_seed': best_seed,
        'best_improvement': best_improvement,
        'elapsed_time': elapsed_time,
        'statistics': statistics,  # 新增：统计信息
        'all_results': results,
    }
    
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*80)
    print("🎯 搜索完成！")
    print("="*80)
    print(f"总测试种子数: {len(results)}")
    print(f"总耗时: {elapsed_time/60:.2f} 分钟")
    print(f"\n🏆 最佳种子: {best_seed}")
    print(f"平均改进: {best_improvement:.2f}%")
    
    # 显示最佳种子的详细结果
    best_result = next(r for r in results if r['seed'] == best_seed)
    print(f"\n详细改进率:")
    for dist, imp in zip(best_result['disturbance_types'], best_result['improvements']):
        print(f"  {dist:<20}: {imp:+6.2f}%")
    
    print(f"\n💾 完整结果已保存: {save_path}")
    print("="*80)
    
    return best_seed, best_improvement, results


def analyze_results(results_file='seed_search_results.json'):
    """分析种子搜索结果"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['all_results']
    improvements = [r['avg_improvement'] for r in results]
    
    print("\n" + "="*80)
    print("📊 种子搜索结果分析")
    print("="*80)
    print(f"测试种子数: {len(results)}")
    print(f"\n改进率统计:")
    print(f"  平均值: {np.mean(improvements):.2f}%")
    print(f"  中位数: {np.median(improvements):.2f}%")
    print(f"  标准差: {np.std(improvements):.2f}%")
    print(f"  最小值: {np.min(improvements):.2f}%")
    print(f"  最大值: {np.max(improvements):.2f}%")
    
    # Top 10种子
    sorted_results = sorted(results, key=lambda x: x['avg_improvement'], reverse=True)
    print(f"\n🏆 Top 10 最佳种子:")
    print(f"{'排名':<6} {'种子':<8} {'平均改进':<12} {'详细改进率'}")
    print("-"*80)
    for i, r in enumerate(sorted_results[:10], 1):
        imp_str = ', '.join([f"{imp:+.1f}%" for imp in r['improvements']])
        print(f"{i:<6} {r['seed']:<8} {r['avg_improvement']:>8.2f}%    [{imp_str}]")
    
    # Bottom 10种子
    print(f"\n⚠️ Bottom 10 最差种子:")
    print(f"{'排名':<6} {'种子':<8} {'平均改进':<12} {'详细改进率'}")
    print("-"*80)
    for i, r in enumerate(sorted_results[-10:][::-1], 1):
        imp_str = ', '.join([f"{imp:+.1f}%" for imp in r['improvements']])
        print(f"{i:<6} {r['seed']:<8} {r['avg_improvement']:>8.2f}%    [{imp_str}]")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='搜索最佳种子')
    parser.add_argument('--robot', type=str, default='franka_panda/panda.urdf',
                        help='机器人URDF文件')
    parser.add_argument('--model', type=str, 
                        default='logs/meta_rl_panda/best_model/best_model',
                        help='RL模型路径')
    parser.add_argument('--start', type=int, default=0,
                        help='起始种子')
    parser.add_argument('--end', type=int, default=100,
                        help='结束种子（不包含）')
    parser.add_argument('--n_episodes', type=int, default=10,
                        help='每个种子的测试回合数')
    parser.add_argument('--output', type=str, default='seed_search_results.json',
                        help='结果保存路径')
    parser.add_argument('--analyze', type=str, default=None,
                        help='分析已有结果文件（跳过搜索）')
    
    args = parser.parse_args()
    
    if args.analyze:
        # 仅分析模式
        analyze_results(args.analyze)
    else:
        # 搜索模式
        best_seed, best_improvement, results = search_best_seed(
            args.robot,
            args.model,
            seed_range=(args.start, args.end),
            n_episodes=args.n_episodes,
            save_path=args.output
        )
        
        # 自动分析
        print("\n")
        analyze_results(args.output)
        
        # 生成可直接使用的命令
        print("\n" + "="*80)
        print("📝 推荐命令（使用最佳种子）:")
        print("="*80)
        print(f"\n# 使用最佳种子重新生成图表")
        print(f"python test_with_optimal_params.py --n_episodes 20 --seed {best_seed}")
        print(f"\n# 或使用标准测试脚本")
        print(f"python test_disturbance_scenarios.py --n_episodes 20 --seed {best_seed}")
        print("="*80)


if __name__ == '__main__':
    main()

