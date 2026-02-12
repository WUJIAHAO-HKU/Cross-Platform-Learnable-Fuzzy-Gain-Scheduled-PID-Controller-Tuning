#!/usr/bin/env python3
"""
数据过滤脚本：移除优化误差过大的不可控样本

用途：
- 过滤掉optimization_error > threshold的样本
- 保留高质量可控样本用于元学习训练
- 输出过滤统计信息
"""

import json
import argparse
import numpy as np
from pathlib import Path


def filter_samples(input_file, output_file, error_threshold=30.0, min_samples_per_type=30):
    """
    过滤优化误差过大的样本
    
    Args:
        input_file: 输入的优化后数据文件
        output_file: 输出的过滤后数据文件
        error_threshold: 误差阈值（度），默认30°
        min_samples_per_type: 每种机器人类型最少保留的样本数
    """
    
    print("="*80)
    print("数据样本过滤")
    print("="*80)
    
    # 加载数据
    print(f"\n📂 加载数据: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"   原始样本数: {len(data)}")
    
    # 按机器人类型分组（根据name字段推断类型）
    by_type = {}
    for sample in data:
        name = sample.get('name', 'unknown')
        # 推断机器人类型
        if 'panda' in name.lower():
            robot_type = 'Panda'
        elif 'laikago' in name.lower():
            robot_type = 'Laikago'
        elif 'kuka' in name.lower() or 'model' in name.lower():
            robot_type = 'KUKA'
        else:
            robot_type = name
        
        if robot_type not in by_type:
            by_type[robot_type] = []
        by_type[robot_type].append(sample)
    
    print(f"\n📊 原始样本分布:")
    for robot_type, samples in by_type.items():
        # 只统计有优化误差的样本（虚拟样本）
        errors = [s.get('optimization_error_deg', 0) for s in samples if 'optimization_error_deg' in s]
        if errors:
            print(f"   {robot_type}: {len(samples)}个样本, "
                  f"平均误差={np.mean(errors):.2f}°, "
                  f"中位误差={np.median(errors):.2f}°")
        else:
            print(f"   {robot_type}: {len(samples)}个样本 (真实机器人，无优化误差)")
    
    # 过滤逻辑
    print(f"\n🔍 过滤条件:")
    print(f"   误差阈值: {error_threshold}°")
    print(f"   每类最少保留: {min_samples_per_type}个")
    
    filtered_data = []
    filter_stats = {}
    
    for robot_type, samples in by_type.items():
        # 按优化误差排序（从小到大），真实样本（无误差字段）排在最前
        samples_sorted = sorted(samples, key=lambda x: x.get('optimization_error_deg', 0))
        
        # 应用阈值过滤（保留真实样本和误差小于阈值的虚拟样本）
        samples_passed = [s for s in samples_sorted 
                         if 'optimization_error_deg' not in s or s['optimization_error_deg'] <= error_threshold]
        
        # 确保至少保留min_samples_per_type个样本
        if len(samples_passed) < min_samples_per_type:
            print(f"   ⚠️  {robot_type}: 通过阈值的样本不足 ({len(samples_passed)}/{min_samples_per_type})")
            print(f"       强制保留误差最小的{min_samples_per_type}个样本")
            samples_kept = samples_sorted[:min_samples_per_type]
        else:
            samples_kept = samples_passed
        
        # 统计
        original_count = len(samples)
        kept_count = len(samples_kept)
        removed_count = original_count - kept_count
        keep_rate = kept_count / original_count * 100
        
        # 计算误差统计（只统计虚拟样本）
        errors_before = [s['optimization_error_deg'] for s in samples if 'optimization_error_deg' in s]
        errors_after = [s['optimization_error_deg'] for s in samples_kept if 'optimization_error_deg' in s]
        
        filter_stats[robot_type] = {
            'original': original_count,
            'kept': kept_count,
            'removed': removed_count,
            'keep_rate': keep_rate,
            'avg_error_before': np.mean(errors_before) if errors_before else 0,
            'avg_error_after': np.mean(errors_after) if errors_after else 0,
            'max_error_after': max(errors_after) if errors_after else 0
        }
        
        filtered_data.extend(samples_kept)
    
    # 打印过滤统计
    print(f"\n📊 过滤结果:")
    print(f"{'机器人类型':<15} {'原始':<8} {'保留':<8} {'移除':<8} {'保留率':<10} {'平均误差':<15}")
    print("-" * 80)
    
    for robot_type, stats in filter_stats.items():
        print(f"{robot_type:<15} "
              f"{stats['original']:<8} "
              f"{stats['kept']:<8} "
              f"{stats['removed']:<8} "
              f"{stats['keep_rate']:<10.1f}% "
              f"{stats['avg_error_before']:.2f}° → {stats['avg_error_after']:.2f}°")
    
    # 总体统计
    original_total = len(data)
    kept_total = len(filtered_data)
    removed_total = original_total - kept_total
    overall_keep_rate = kept_total / original_total * 100
    
    print("-" * 80)
    print(f"{'总计':<15} "
          f"{original_total:<8} "
          f"{kept_total:<8} "
          f"{removed_total:<8} "
          f"{overall_keep_rate:<10.1f}%")
    
    # 质量改善（只统计虚拟样本）
    all_errors_before = [s['optimization_error_deg'] for s in data if 'optimization_error_deg' in s]
    all_errors_after = [s['optimization_error_deg'] for s in filtered_data if 'optimization_error_deg' in s]
    
    print(f"\n📈 质量提升:")
    print(f"   平均误差: {np.mean(all_errors_before):.2f}° → {np.mean(all_errors_after):.2f}° "
          f"(改善 {(1 - np.mean(all_errors_after)/np.mean(all_errors_before))*100:.1f}%)")
    print(f"   中位误差: {np.median(all_errors_before):.2f}° → {np.median(all_errors_after):.2f}°")
    print(f"   最大误差: {max(all_errors_before):.2f}° → {max(all_errors_after):.2f}°")
    print(f"   标准差: {np.std(all_errors_before):.2f}° → {np.std(all_errors_after):.2f}°")
    
    # 保存过滤后的数据
    print(f"\n💾 保存过滤后的数据: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"\n✅ 过滤完成！")
    print(f"   原始样本: {original_total}")
    print(f"   保留样本: {kept_total}")
    print(f"   移除样本: {removed_total}")
    print(f"   保留率: {overall_keep_rate:.1f}%")
    print(f"   平均误差: {np.mean(all_errors_after):.2f}°")
    
    # 建议
    print(f"\n🎯 建议:")
    if overall_keep_rate < 70:
        print(f"   ⚠️  保留率较低 ({overall_keep_rate:.1f}%)")
        print(f"   建议: 检查数据增强的扰动范围是否过大")
    elif overall_keep_rate > 95:
        print(f"   💡 保留率很高 ({overall_keep_rate:.1f}%)")
        print(f"   建议: 可以适当降低阈值以进一步提升质量")
    else:
        print(f"   ✅ 保留率适中 ({overall_keep_rate:.1f}%)")
        print(f"   建议: 数据质量良好，可以继续训练")
    
    if np.mean(all_errors_after) > 20:
        print(f"   ⚠️  平均误差仍然较高 ({np.mean(all_errors_after):.2f}°)")
        print(f"   建议: 考虑进一步降低阈值或调整数据增强策略")
    
    print(f"\n📖 下一步:")
    print(f"   使用过滤后的数据训练元学习网络:")
    print(f"   python train_meta_learning.py --data {output_file}")
    
    return filtered_data, filter_stats


def analyze_removed_samples(input_file, output_file, error_threshold=30.0):
    """分析被移除的样本特征"""
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 分离保留和移除的样本（只考虑虚拟样本）
    kept_samples = [s for s in data 
                   if 'optimization_error_deg' not in s or s['optimization_error_deg'] <= error_threshold]
    removed_samples = [s for s in data 
                      if 'optimization_error_deg' in s and s['optimization_error_deg'] > error_threshold]
    
    if len(removed_samples) == 0:
        print("\n✅ 没有样本被移除")
        return
    
    print(f"\n🔬 被移除样本分析 (n={len(removed_samples)}):")
    
    # 按机器人类型统计
    removed_by_type = {}
    for sample in removed_samples:
        name = sample.get('name', 'unknown')
        # 推断机器人类型
        if 'panda' in name.lower():
            robot_type = 'Panda'
        elif 'laikago' in name.lower():
            robot_type = 'Laikago'
        elif 'kuka' in name.lower() or 'model' in name.lower():
            robot_type = 'KUKA'
        else:
            robot_type = name
        
        if robot_type not in removed_by_type:
            removed_by_type[robot_type] = []
        removed_by_type[robot_type].append(sample)
    
    for robot_type, samples in removed_by_type.items():
        errors = [s['optimization_error_deg'] for s in samples]
        print(f"\n   {robot_type} (移除{len(samples)}个):")
        print(f"     误差范围: {min(errors):.2f}° - {max(errors):.2f}°")
        print(f"     平均误差: {np.mean(errors):.2f}°")
        
        # 分析参数特征
        if 'augmentation_params' in samples[0]:
            mass_scales = [s['augmentation_params'].get('mass_scale', 1.0) for s in samples]
            inertia_scales = [s['augmentation_params'].get('inertia_scale', 1.0) for s in samples]
            
            print(f"     质量缩放: {np.mean(mass_scales):.2f} ± {np.std(mass_scales):.2f}")
            print(f"     惯性缩放: {np.mean(inertia_scales):.2f} ± {np.std(inertia_scales):.2f}")


def main():
    parser = argparse.ArgumentParser(description='过滤不可控的虚拟样本')
    parser.add_argument('--input', default='augmented_pid_data_optimized.json',
                        help='输入的优化后数据文件')
    parser.add_argument('--output', default='augmented_pid_data_filtered.json',
                        help='输出的过滤后数据文件')
    parser.add_argument('--error_threshold', type=float, default=30.0,
                        help='优化误差阈值（度），默认30°')
    parser.add_argument('--min_samples_per_type', type=int, default=30,
                        help='每种机器人类型最少保留的样本数，默认30')
    parser.add_argument('--analyze', action='store_true',
                        help='分析被移除样本的特征')
    
    args = parser.parse_args()
    
    # 执行过滤
    filtered_data, filter_stats = filter_samples(
        args.input,
        args.output,
        args.error_threshold,
        args.min_samples_per_type
    )
    
    # 可选：分析被移除的样本
    if args.analyze:
        analyze_removed_samples(args.input, args.output, args.error_threshold)


if __name__ == '__main__':
    main()

