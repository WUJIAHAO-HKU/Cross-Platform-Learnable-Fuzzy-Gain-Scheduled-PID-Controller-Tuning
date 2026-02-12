#!/usr/bin/env python3
"""
对比实验：固定PID vs 自适应RL
评估在不同扰动下的性能差异
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from stable_baselines3 import PPO

sys.path.append(str(Path(__file__).parent))
from adaptive_laikago_env import LaikagoAdaptiveEnv


def evaluate_fixed_pid(config, n_episodes=10, gui=False):
    """
    评估固定PID（无RL，delta_action=0）
    
    Args:
        config: 环境配置
        n_episodes: 评估轮数
        gui: 是否显示GUI
    
    Returns:
        results: 评估结果字典
    """
    print("\n" + "=" * 80)
    print("评估固定PID（无RL调整）")
    print("=" * 80)
    
    env = LaikagoAdaptiveEnv(config=config, gui=gui, use_meta_learning=True)
    
    all_rewards = []
    all_tracking_errors = []
    all_orientation_errors = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_errors = []
        episode_orientation = []
        
        for step in range(config['max_steps']):
            # 固定PID：不调整增益（action=0）
            action = np.zeros(2)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_errors.append(info['tracking_error'])
            episode_orientation.append(info['orientation_penalty'])
            
            if terminated or truncated:
                break
        
        all_rewards.append(episode_reward)
        all_tracking_errors.append(np.mean(episode_errors))
        all_orientation_errors.append(np.mean(episode_orientation))
        
        print(f"  Episode {episode+1}/{n_episodes}: "
              f"reward={episode_reward:.2f}, "
              f"tracking_error={np.mean(episode_errors):.6f}, "
              f"orientation_error={np.mean(episode_orientation):.6f}")
    
    env.close()
    
    results = {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_tracking_error': np.mean(all_tracking_errors),
        'std_tracking_error': np.std(all_tracking_errors),
        'mean_orientation_error': np.mean(all_orientation_errors),
        'std_orientation_error': np.std(all_orientation_errors)
    }
    
    print(f"\n固定PID结果:")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  跟踪误差: {results['mean_tracking_error']:.6f} ± {results['std_tracking_error']:.6f}")
    print(f"  姿态误差: {results['mean_orientation_error']:.6f} ± {results['std_orientation_error']:.6f}")
    
    return results


def evaluate_adaptive_rl(model_path, config, n_episodes=10, gui=False):
    """
    评估自适应RL
    
    Args:
        model_path: 训练好的模型路径
        config: 环境配置
        n_episodes: 评估轮数
        gui: 是否显示GUI
    
    Returns:
        results: 评估结果字典
    """
    print("\n" + "=" * 80)
    print("评估自适应RL")
    print("=" * 80)
    
    model = PPO.load(model_path)
    env = LaikagoAdaptiveEnv(config=config, gui=gui, use_meta_learning=True)
    
    all_rewards = []
    all_tracking_errors = []
    all_orientation_errors = []
    all_kp_adjustments = []
    all_kd_adjustments = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_errors = []
        episode_orientation = []
        episode_kp = []
        episode_kd = []
        
        for step in range(config['max_steps']):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_errors.append(info['tracking_error'])
            episode_orientation.append(info['orientation_penalty'])
            episode_kp.append(info['current_kp'])
            episode_kd.append(info['current_kd'])
            
            if terminated or truncated:
                break
        
        all_rewards.append(episode_reward)
        all_tracking_errors.append(np.mean(episode_errors))
        all_orientation_errors.append(np.mean(episode_orientation))
        all_kp_adjustments.append(episode_kp)
        all_kd_adjustments.append(episode_kd)
        
        print(f"  Episode {episode+1}/{n_episodes}: "
              f"reward={episode_reward:.2f}, "
              f"tracking_error={np.mean(episode_errors):.6f}, "
              f"final_Kp={episode_kp[-1]:.3f}, "
              f"final_Kd={episode_kd[-1]:.3f}")
    
    env.close()
    
    results = {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_tracking_error': np.mean(all_tracking_errors),
        'std_tracking_error': np.std(all_tracking_errors),
        'mean_orientation_error': np.mean(all_orientation_errors),
        'std_orientation_error': np.std(all_orientation_errors),
        'kp_adjustments': all_kp_adjustments,
        'kd_adjustments': all_kd_adjustments,
        'kp_range': (np.min([np.min(k) for k in all_kp_adjustments]),
                     np.max([np.max(k) for k in all_kp_adjustments])),
        'kd_range': (np.min([np.min(k) for k in all_kd_adjustments]),
                     np.max([np.max(k) for k in all_kd_adjustments]))
    }
    
    print(f"\n自适应RL结果:")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  跟踪误差: {results['mean_tracking_error']:.6f} ± {results['std_tracking_error']:.6f}")
    print(f"  姿态误差: {results['mean_orientation_error']:.6f} ± {results['std_orientation_error']:.6f}")
    print(f"  Kp范围: [{results['kp_range'][0]:.3f}, {results['kp_range'][1]:.3f}]")
    print(f"  Kd范围: [{results['kd_range'][0]:.3f}, {results['kd_range'][1]:.3f}]")
    
    return results


def compare_methods(scenario_name, model_path=None, n_episodes=10, gui=False):
    """
    对比固定PID vs 自适应RL
    
    Args:
        scenario_name: 扰动场景名称
        model_path: 自适应RL模型路径（如果None则只评估固定PID）
        n_episodes: 评估轮数
        gui: 是否显示GUI
    """
    from train_multi_disturbance import DISTURBANCE_SCENARIOS
    
    print("=" * 80)
    print(f"对比实验：{scenario_name}")
    print(f"描述：{DISTURBANCE_SCENARIOS[scenario_name]['description']}")
    print("=" * 80)
    
    # 配置
    config = {
        'max_steps': 5000,
        'init_kp': 0.5,
        'init_kd': 0.1,
        'kp_range': (0.1, 2.0),
        'kd_range': (0.01, 0.5),
        'disturbance': DISTURBANCE_SCENARIOS[scenario_name]
    }
    
    # 评估固定PID
    fixed_results = evaluate_fixed_pid(config, n_episodes, gui)
    
    # 评估自适应RL（如果提供了模型）
    if model_path:
        adaptive_results = evaluate_adaptive_rl(model_path, config, n_episodes, gui)
        
        # 计算改善率
        reward_improvement = ((adaptive_results['mean_reward'] - fixed_results['mean_reward']) 
                             / abs(fixed_results['mean_reward'])) * 100
        error_improvement = ((fixed_results['mean_tracking_error'] - adaptive_results['mean_tracking_error']) 
                            / fixed_results['mean_tracking_error']) * 100
        
        print("\n" + "=" * 80)
        print("性能对比")
        print("=" * 80)
        print(f"奖励改善: {reward_improvement:+.2f}%")
        print(f"跟踪误差改善: {error_improvement:+.2f}%")
        
        return fixed_results, adaptive_results, reward_improvement, error_improvement
    
    else:
        return fixed_results, None, None, None


def plot_comparison(scenario_name, fixed_results, adaptive_results):
    """生成对比图表"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 图1: 奖励对比
    methods = ['Fixed PID', 'Adaptive RL']
    rewards = [fixed_results['mean_reward'], adaptive_results['mean_reward']]
    reward_stds = [fixed_results['std_reward'], adaptive_results['std_reward']]
    
    axes[0].bar(methods, rewards, yerr=reward_stds, capsize=5, alpha=0.7)
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title(f'{scenario_name}: Reward Comparison')
    axes[0].grid(axis='y', alpha=0.3)
    
    # 图2: 跟踪误差对比
    errors = [fixed_results['mean_tracking_error'], adaptive_results['mean_tracking_error']]
    error_stds = [fixed_results['std_tracking_error'], adaptive_results['std_tracking_error']]
    
    axes[1].bar(methods, errors, yerr=error_stds, capsize=5, alpha=0.7, color=['orange', 'green'])
    axes[1].set_ylabel('Tracking Error (rad)')
    axes[1].set_title(f'{scenario_name}: Tracking Error Comparison')
    axes[1].grid(axis='y', alpha=0.3)
    
    # 图3: 增益调整示例（第一个episode）
    if adaptive_results and 'kp_adjustments' in adaptive_results:
        kp = adaptive_results['kp_adjustments'][0]
        kd = adaptive_results['kd_adjustments'][0]
        
        axes[2].plot(kp, label='Kp', alpha=0.7)
        axes[2].plot(kd, label='Kd', alpha=0.7)
        axes[2].axhline(0.5, color='r', linestyle='--', alpha=0.5, label='Initial Kp')
        axes[2].axhline(0.1, color='b', linestyle='--', alpha=0.5, label='Initial Kd')
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Gain Value')
        axes[2].set_title(f'{scenario_name}: Gain Adjustments')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'comparison_{scenario_name}.png', dpi=150)
    print(f"\n📊 图表已保存: comparison_{scenario_name}.png")
    plt.close()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='对比固定PID vs 自适应RL')
    parser.add_argument('--scenario', type=str, default='random_force',
                       choices=['random_force', 'payload', 'terrain', 'param_uncertainty', 'mixed'],
                       help='扰动场景')
    parser.add_argument('--model', type=str, default=None,
                       help='自适应RL模型路径')
    parser.add_argument('--n_episodes', type=int, default=10,
                       help='评估轮数')
    parser.add_argument('--gui', action='store_true',
                       help='显示GUI')
    parser.add_argument('--plot', action='store_true',
                       help='生成对比图表')
    
    args = parser.parse_args()
    
    # 运行对比实验
    fixed_results, adaptive_results, reward_imp, error_imp = compare_methods(
        scenario_name=args.scenario,
        model_path=args.model,
        n_episodes=args.n_episodes,
        gui=args.gui
    )
    
    # 生成图表
    if args.plot and adaptive_results:
        plot_comparison(args.scenario, fixed_results, adaptive_results)

