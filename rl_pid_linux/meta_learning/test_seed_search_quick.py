#!/usr/bin/env python3
"""
快速测试种子搜索功能
仅测试3个种子（0, 42, 99），每个3 episodes，用于验证脚本是否正常工作
"""

import sys
from find_best_seed import search_best_seed, analyze_results


def main():
    print("="*80)
    print("🧪 快速测试种子搜索功能")
    print("="*80)
    print("测试种子: 0, 42, 99")
    print("每个种子: 3 episodes")
    print("预计耗时: ~3分钟")
    print("="*80 + "\n")
    
    # 快速测试
    robot_urdf = 'franka_panda/panda.urdf'
    model_path = 'logs/meta_rl_panda/best_model/best_model'
    
    # 测试3个种子
    test_seeds = [0, 42, 99]
    
    print("手动测试模式：逐个测试种子...")
    from find_best_seed import evaluate_seed
    
    results = []
    for seed in test_seeds:
        print(f"\n测试种子 {seed}...")
        result = evaluate_seed(seed, robot_urdf, model_path, n_episodes=3)
        if result:
            results.append(result)
            print(f"  ✅ 平均改进: {result['avg_improvement']:.2f}%")
        else:
            print(f"  ❌ 测试失败")
    
    if len(results) == 0:
        print("\n❌ 所有测试都失败了！请检查环境和模型路径。")
        return
    
    # 找到最佳
    best_result = max(results, key=lambda x: x['avg_improvement'])
    
    print("\n" + "="*80)
    print("✅ 快速测试完成！")
    print("="*80)
    print(f"测试通过: {len(results)}/3")
    print(f"\n最佳种子: {best_result['seed']}")
    print(f"平均改进: {best_result['avg_improvement']:.2f}%")
    print(f"\n详细改进率:")
    for dist, imp in zip(best_result['disturbance_types'], best_result['improvements']):
        print(f"  {dist:<20}: {imp:+6.2f}%")
    
    print("\n" + "="*80)
    print("💡 下一步:")
    print("="*80)
    print("如果测试通过，可以运行完整搜索:")
    print(f"  python find_best_seed.py --n_episodes 10")
    print(f"\n或使用最佳测试种子生成图表:")
    print(f"  python test_with_optimal_params.py --n_episodes 20 --seed {best_result['seed']}")
    print("="*80)


if __name__ == '__main__':
    main()

