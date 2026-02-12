#!/usr/bin/env python3
"""
多场景扰动训练脚本
针对4种扰动类型训练独立的RL agent
"""

import os
import sys
from pathlib import Path
from train_adaptive_rl import train_adaptive_rl

# 扰动场景配置
DISTURBANCE_SCENARIOS = {
    'random_force': {
        'type': 'random_force',
        'force_range': (1.0, 3.0),
        'force_interval': 800,
        'force_duration': 100,
        'description': '随机外力（1～3N侧推）'
    },
    'payload': {
        'type': 'payload',
        'payload_range': (0.0, 5.0),
        'payload_interval': 1000,
        'description': '动态负载（0～5kg）'
    },
    'terrain': {
        'type': 'terrain',
        'terrain_angle_range': (0, 15),
        'terrain_interval': 2000,
        'description': '地形变化（0～15°斜坡）'
    },
    'param_uncertainty': {
        'type': 'param_uncertainty',
        'param_uncertainty': 0.2,
        'description': '参数不确定性（±20%质量）'
    },
    'mixed': {
        'type': 'mixed',
        'force_range': (1.0, 3.0),
        'force_interval': 800,
        'force_duration': 100,
        'payload_range': (0.0, 5.0),
        'payload_interval': 1000,
        'description': '混合扰动（外力+负载）'
    }
}


def train_all_scenarios(
    scenarios=['random_force', 'payload', 'terrain', 'param_uncertainty'],
    total_timesteps=500000,
    n_envs=4,
    use_gpu=True
):
    """
    训练所有扰动场景
    
    Args:
        scenarios: 要训练的场景列表
        total_timesteps: 每个场景的训练步数
        n_envs: 并行环境数量
        use_gpu: 是否使用GPU
    """
    print("=" * 80)
    print("多场景自适应RL训练")
    print("=" * 80)
    print(f"\n📋 计划训练场景: {len(scenarios)}个")
    for scenario in scenarios:
        print(f"   - {scenario}: {DISTURBANCE_SCENARIOS[scenario]['description']}")
    
    print(f"\n⏱️  预计总时间: {len(scenarios) * 2.5:.1f}小时（每场景约2.5小时@GPU）")
    print(f"💾 模型保存位置: ./logs/adaptive_rl/")
    
    input("\n按Enter开始训练，或Ctrl+C取消...")
    
    results = {}
    
    for i, scenario in enumerate(scenarios, 1):
        print("\n" + "=" * 80)
        print(f"训练场景 {i}/{len(scenarios)}: {scenario}")
        print(f"描述: {DISTURBANCE_SCENARIOS[scenario]['description']}")
        print("=" * 80)
        
        try:
            # 使用场景特定的配置训练
            model_path = train_adaptive_rl(
                total_timesteps=total_timesteps,
                n_envs=n_envs,
                disturbance_type=scenario,
                use_gpu=use_gpu
            )
            
            results[scenario] = {
                'status': 'success',
                'model_path': model_path
            }
            
            print(f"\n✅ {scenario} 训练完成！")
            print(f"   模型: {model_path}")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  {scenario} 训练被中断")
            results[scenario] = {'status': 'interrupted'}
            break
        except Exception as e:
            print(f"\n❌ {scenario} 训练失败: {e}")
            results[scenario] = {'status': 'failed', 'error': str(e)}
            continue
    
    # 总结
    print("\n" + "=" * 80)
    print("训练总结")
    print("=" * 80)
    
    for scenario, result in results.items():
        status_icon = {'success': '✅', 'interrupted': '⚠️', 'failed': '❌'}.get(result['status'], '❓')
        print(f"{status_icon} {scenario}: {result['status']}")
        if result['status'] == 'success':
            print(f"   模型: {result.get('model_path', 'N/A')}")
    
    return results


def quick_test_all_scenarios(steps=5000, gui=False):
    """
    快速测试所有扰动场景（验证环境）
    
    Args:
        steps: 每个场景测试步数
        gui: 是否显示GUI
    """
    import sys
    sys.path.append(str(Path(__file__).parent))
    from adaptive_laikago_env import LaikagoAdaptiveEnv
    
    print("=" * 80)
    print("快速测试所有扰动场景")
    print("=" * 80)
    
    for scenario_name, scenario_config in DISTURBANCE_SCENARIOS.items():
        print(f"\n🧪 测试场景: {scenario_name}")
        print(f"   描述: {scenario_config['description']}")
        
        config = {
            'max_steps': steps,
            'init_kp': 0.5,
            'init_kd': 0.1,
            'kp_range': (0.1, 2.0),
            'kd_range': (0.01, 0.5),
            'disturbance': scenario_config
        }
        
        try:
            env = LaikagoAdaptiveEnv(config=config, gui=gui, use_meta_learning=True)
            obs, _ = env.reset()
            
            total_reward = 0
            for step in range(steps):
                action = env.action_space.sample() * 0.01  # 小幅随机动作
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if step % 1000 == 0:
                    print(f"   Step {step}: reward={reward:.2f}, Kp={info['current_kp']:.3f}")
                
                if terminated or truncated:
                    print(f"   ⚠️  Episode结束于step {step}")
                    break
            
            env.close()
            print(f"   ✅ 测试完成！总奖励: {total_reward:.2f}")
            
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("✅ 所有场景测试完成！")
    print("=" * 80)


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='多场景扰动训练')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test'],
                       help='训练或测试模式')
    parser.add_argument('--scenarios', type=str, nargs='+',
                       default=['random_force', 'payload', 'terrain', 'param_uncertainty'],
                       help='要训练的场景')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='每个场景的训练步数')
    parser.add_argument('--n_envs', type=int, default=4,
                       help='并行环境数量')
    parser.add_argument('--gpu', action='store_true',
                       help='使用GPU训练')
    parser.add_argument('--gui', action='store_true',
                       help='测试时显示GUI')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        results = train_all_scenarios(
            scenarios=args.scenarios,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            use_gpu=args.gpu
        )
    
    elif args.mode == 'test':
        quick_test_all_scenarios(steps=5000, gui=args.gui)

