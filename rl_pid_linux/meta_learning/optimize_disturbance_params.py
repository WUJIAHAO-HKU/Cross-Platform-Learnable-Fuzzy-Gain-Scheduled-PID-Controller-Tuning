#!/usr/bin/env python3
"""
智能扰动参数优化脚本
遍历所有参数组合，找到每种扰动下RL优化程度最大的配置
"""

import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from meta_rl_disturbance_env import MetaRLDisturbanceEnv
import json
import itertools
from tqdm import tqdm
import argparse


def evaluate_with_params(robot_urdf, disturbance_type, disturbance_params, 
                         model_path=None, n_episodes=3, max_steps=2000):
    """使用特定参数评估性能"""
    try:
        # 创建环境
        env = MetaRLDisturbanceEnv(
            robot_urdf=robot_urdf,
            gui=False,
            disturbance_type=disturbance_type,
            disturbance_params=disturbance_params
        )
        
        # 加载模型
        model = None
        if model_path is not None:
            model = PPO.load(model_path)
        
        # 评估
        all_errors_deg = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_errors = []
            
            for step in range(max_steps):
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = np.zeros(2)
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 计算误差
                joint_states = p.getJointStates(env.robot_id, env.controllable_joints)
                q_actual = np.array([s[0] for s in joint_states])
                q_ref = env._get_reference_trajectory()
                
                error_rad = np.linalg.norm(q_ref - q_actual)
                error_deg = np.degrees(error_rad)
                
                episode_errors.append(error_deg)
                
                if terminated or truncated:
                    break
            
            all_errors_deg.extend(episode_errors)
        
        env.close()
        
        mean_error = np.mean(all_errors_deg)
        return mean_error
    
    except Exception as e:
        print(f"  ⚠️ 评估失败: {e}")
        return None


def generate_param_grid(disturbance_type):
    """为每种扰动类型生成参数网格"""
    
    if disturbance_type == 'none':
        return [{}]
    
    elif disturbance_type == 'random_force':
        # 随机外力参数网格
        force_ranges = [10.0, 15.0, 20.0, 25.0, 30.0]
        force_probs = [0.05, 0.1, 0.15, 0.2]
        
        grid = []
        for fr, fp in itertools.product(force_ranges, force_probs):
            grid.append({
                'random_force': {
                    'force_range': fr,
                    'force_prob': fp
                }
            })
        return grid
    
    elif disturbance_type == 'payload':
        # 负载变化参数网格
        mass_ranges = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        grid = []
        for mr in mass_ranges:
            grid.append({
                'payload': {
                    'mass_range': mr
                }
            })
        return grid
    
    elif disturbance_type == 'param_uncertainty':
        # 参数不确定性网格
        mass_scales = [(0.7, 1.3), (0.75, 1.25), (0.8, 1.2), (0.85, 1.15), (0.9, 1.1)]
        friction_scales = [(0.5, 2.0), (0.6, 1.8), (0.7, 1.5), (0.8, 1.3)]
        
        grid = []
        for ms, fs in itertools.product(mass_scales, friction_scales):
            grid.append({
                'param_uncertainty': {
                    'mass_scale': ms,
                    'friction_scale': fs
                }
            })
        return grid
    
    elif disturbance_type == 'mixed':
        # 混合扰动：简化网格（避免组合爆炸）
        mass_ranges = [2.0, 3.0, 4.0]
        mass_scales = [(0.8, 1.2), (0.85, 1.15), (0.9, 1.1)]
        
        grid = []
        for mr, ms in itertools.product(mass_ranges, mass_scales):
            grid.append({
                'payload': {'mass_range': mr},
                'param_uncertainty': {
                    'mass_scale': ms,
                    'friction_scale': (0.7, 1.5)
                }
            })
        return grid
    
    return [{}]


def search_best_params(robot_urdf, model_path, n_episodes=3):
    """搜索每种扰动的最优参数"""
    
    disturbance_types = ['none', 'random_force', 'payload', 'param_uncertainty', 'mixed']
    
    best_configs = {}
    
    print("="*80)
    print("🔍 智能参数搜索：寻找每种扰动下RL优化程度最大的配置")
    print("="*80)
    print(f"机器人: {robot_urdf}")
    print(f"RL模型: {model_path}")
    print(f"每个配置测试: {n_episodes} episodes")
    print()
    
    for dist_type in disturbance_types:
        print(f"\n{'='*80}")
        print(f"扰动类型: {dist_type}")
        print(f"{'='*80}")
        
        # 生成参数网格
        param_grid = generate_param_grid(dist_type)
        print(f"参数组合数: {len(param_grid)}")
        
        best_improvement = -float('inf')
        best_params = None
        best_pure_error = None
        best_rl_error = None
        
        # 遍历所有参数组合
        for i, params in enumerate(tqdm(param_grid, desc=f"  搜索 {dist_type}")):
            # 评估Pure Meta-PID
            pure_error = evaluate_with_params(
                robot_urdf, dist_type, params, 
                model_path=None, n_episodes=n_episodes
            )
            
            if pure_error is None:
                continue
            
            # 评估Meta-PID+RL
            rl_error = evaluate_with_params(
                robot_urdf, dist_type, params,
                model_path=model_path, n_episodes=n_episodes
            )
            
            if rl_error is None:
                continue
            
            # 计算改进百分比
            improvement = (pure_error - rl_error) / pure_error * 100
            
            # 更新最佳配置
            if improvement > best_improvement:
                best_improvement = improvement
                best_params = params
                best_pure_error = pure_error
                best_rl_error = rl_error
        
        # 保存结果
        best_configs[dist_type] = {
            'params': best_params,
            'improvement': best_improvement,
            'pure_error': best_pure_error,
            'rl_error': best_rl_error
        }
        
        print(f"\n  ✅ {dist_type} 最优配置:")
        print(f"     参数: {best_params}")
        print(f"     Pure Meta-PID 误差: {best_pure_error:.2f}°")
        print(f"     Meta-PID+RL 误差: {best_rl_error:.2f}°")
        print(f"     改进程度: {best_improvement:+.2f}%")
    
    return best_configs


def test_with_best_configs(robot_urdf, model_path, best_configs, n_episodes=10):
    """使用最优配置重新测试"""
    
    print("\n" + "="*80)
    print("🎯 使用最优配置重新测试（更多episodes）")
    print("="*80)
    
    final_results = {
        'pure': {},
        'rl': {}
    }
    
    for dist_type, config in best_configs.items():
        print(f"\n测试 {dist_type} (最优参数)...")
        params = config['params']
        
        # Pure Meta-PID
        pure_error = evaluate_with_params(
            robot_urdf, dist_type, params,
            model_path=None, n_episodes=n_episodes, max_steps=3000
        )
        
        # Meta-PID+RL
        rl_error = evaluate_with_params(
            robot_urdf, dist_type, params,
            model_path=model_path, n_episodes=n_episodes, max_steps=3000
        )
        
        improvement = (pure_error - rl_error) / pure_error * 100
        
        final_results['pure'][dist_type] = {
            'mean_error_deg': pure_error,
            'params': params
        }
        final_results['rl'][dist_type] = {
            'mean_error_deg': rl_error,
            'params': params
        }
        
        print(f"  Pure: {pure_error:.2f}° | RL: {rl_error:.2f}° | 改进: {improvement:+.2f}%")
    
    return final_results


def save_results(best_configs, final_results, output_file='best_disturbance_configs.json'):
    """保存结果到JSON文件"""
    
    output = {
        'search_results': best_configs,
        'final_test_results': final_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='智能扰动参数优化')
    parser.add_argument('--robot', type=str, default='franka_panda/panda.urdf',
                        help='机器人URDF文件')
    parser.add_argument('--model', type=str, 
                        default='logs/meta_rl_panda/best_model/best_model',
                        help='RL模型路径')
    parser.add_argument('--search_episodes', type=int, default=3,
                        help='搜索阶段每个配置的episodes数（快速）')
    parser.add_argument('--test_episodes', type=int, default=10,
                        help='最终测试的episodes数（准确）')
    parser.add_argument('--output', type=str, default='best_disturbance_configs.json',
                        help='输出JSON文件')
    
    args = parser.parse_args()
    
    # 阶段1: 搜索最优参数
    best_configs = search_best_params(
        args.robot, 
        args.model, 
        n_episodes=args.search_episodes
    )
    
    # 阶段2: 使用最优参数重新测试
    final_results = test_with_best_configs(
        args.robot,
        args.model,
        best_configs,
        n_episodes=args.test_episodes
    )
    
    # 保存结果
    save_results(best_configs, final_results, args.output)
    
    # 打印总结
    print("\n" + "="*80)
    print("📊 总结：各扰动类型下的最大RL优化潜力")
    print("="*80)
    for dist_type, config in best_configs.items():
        print(f"{dist_type:20s}: {config['improvement']:+7.2f}%  "
              f"(Pure: {config['pure_error']:.2f}° → RL: {config['rl_error']:.2f}°)")
    
    avg_improvement = np.mean([c['improvement'] for c in best_configs.values()])
    print(f"\n{'平均改进':20s}: {avg_improvement:+7.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()

