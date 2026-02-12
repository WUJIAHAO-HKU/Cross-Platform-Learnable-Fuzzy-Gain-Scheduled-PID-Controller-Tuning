#!/usr/bin/env python3
"""
训练自适应RL agent
目标：学习在线调整PID增益以应对扰动
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime

# 导入自适应环境
sys.path.append(str(Path(__file__).parent))
from adaptive_laikago_env import LaikagoAdaptiveEnv


def make_env(config, rank=0, gui=False):
    """创建环境"""
    def _init():
        env = LaikagoAdaptiveEnv(config=config, gui=gui, use_meta_learning=True)
        env = Monitor(env)
        return env
    return _init


def train_adaptive_rl(
    total_timesteps=500000,
    n_envs=4,
    learning_rate=3e-4,
    batch_size=256,
    n_epochs=10,
    disturbance_type='random_force',
    save_dir='./logs/adaptive_rl',
    use_gpu=True
):
    """
    训练自适应RL agent
    
    Args:
        total_timesteps: 总训练步数
        n_envs: 并行环境数量
        learning_rate: 学习率
        batch_size: 批大小
        n_epochs: 每次更新的轮数
        disturbance_type: 扰动类型
        save_dir: 保存目录
        use_gpu: 是否使用GPU
    """
    print("=" * 80)
    print("自适应RL训练开始")
    print("=" * 80)
    
    # 配置
    config = {
        'max_steps': 5000,
        'init_kp': 0.5,  # 元学习会覆盖这个值
        'init_kd': 0.1,
        'kp_range': (0.1, 2.0),
        'kd_range': (0.01, 0.5),
        'disturbance': {
            'type': disturbance_type,
            'force_range': (1.0, 3.0),  # 较强的扰动
            'force_interval': 800,
            'force_duration': 100
        }
    }
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(save_dir) / f"adaptive_{disturbance_type}_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 保存路径: {save_path}")
    print(f"🔧 配置:")
    print(f"   - 并行环境: {n_envs}")
    print(f"   - 扰动类型: {disturbance_type}")
    print(f"   - 扰动强度: {config['disturbance']['force_range']}")
    print(f"   - 学习率: {learning_rate}")
    print(f"   - GPU: {use_gpu}")
    
    # 创建环境
    print(f"\n🏗️  创建{n_envs}个并行环境...")
    if n_envs > 1:
        env = SubprocVecEnv([make_env(config, i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(config)])
    
    # 创建评估环境
    print("🏗️  创建评估环境...")
    eval_env = DummyVecEnv([make_env(config)])
    
    # 设置设备
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用设备: {device}")
    
    # 创建PPO模型
    print("\n🤖 创建PPO模型...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log=str(save_path / 'tensorboard')
    )
    
    print(f"   - 策略网络: MlpPolicy")
    print(f"   - 观测维度: {env.observation_space.shape}")
    print(f"   - 动作维度: {env.action_space.shape}")
    
    # 回调函数
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / 'best_model'),
        log_path=str(save_path / 'eval_logs'),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=str(save_path / 'checkpoints'),
        name_prefix='adaptive_rl'
    )
    
    # 开始训练
    print("\n" + "=" * 80)
    print("🚀 开始训练...")
    print("=" * 80)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            tb_log_name=f"adaptive_{disturbance_type}",
            progress_bar=False
        )
        
        # 保存最终模型
        final_model_path = save_path / 'final_model.zip'
        model.save(str(final_model_path))
        print(f"\n✅ 训练完成！最终模型保存至: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        model.save(str(save_path / 'interrupted_model.zip'))
        print(f"中断模型已保存")
    
    finally:
        env.close()
        eval_env.close()
        print("✅ 环境已关闭")
    
    return str(final_model_path)


def evaluate_adaptive_policy(model_path, n_episodes=10, disturbance_type='random_force', gui=False):
    """
    评估训练好的自适应策略
    
    Args:
        model_path: 模型路径
        n_episodes: 评估轮数
        disturbance_type: 扰动类型
        gui: 是否显示GUI
    """
    print("\n" + "=" * 80)
    print("评估自适应RL策略")
    print("=" * 80)
    
    # 配置（与训练时相同）
    config = {
        'max_steps': 5000,
        'init_kp': 0.5,
        'init_kd': 0.1,
        'kp_range': (0.1, 2.0),
        'kd_range': (0.01, 0.5),
        'disturbance': {
            'type': disturbance_type,
            'force_range': (1.0, 3.0),
            'force_interval': 800,
            'force_duration': 100
        }
    }
    
    # 加载模型
    print(f"📦 加载模型: {model_path}")
    model = PPO.load(model_path)
    
    # 创建环境
    env = LaikagoAdaptiveEnv(config=config, gui=gui, use_meta_learning=True)
    
    # 评估
    total_rewards = []
    tracking_errors = []
    kp_adjustments = []
    kd_adjustments = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_errors = []
        episode_kp = []
        episode_kd = []
        
        for step in range(config['max_steps']):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_errors.append(info['tracking_error'])
            episode_kp.append(info['current_kp'])
            episode_kd.append(info['current_kd'])
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        tracking_errors.append(np.mean(episode_errors))
        kp_adjustments.append(episode_kp)
        kd_adjustments.append(episode_kd)
        
        print(f"Episode {episode+1}/{n_episodes}: "
              f"reward={episode_reward:.2f}, "
              f"avg_error={np.mean(episode_errors):.6f}, "
              f"final_Kp={episode_kp[-1]:.3f}, "
              f"final_Kd={episode_kd[-1]:.3f}")
    
    env.close()
    
    # 统计
    print("\n" + "=" * 80)
    print("评估结果统计")
    print("=" * 80)
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"平均跟踪误差: {np.mean(tracking_errors):.6f} ± {np.std(tracking_errors):.6f}")
    print(f"Kp范围: [{np.min([np.min(k) for k in kp_adjustments]):.3f}, "
          f"{np.max([np.max(k) for k in kp_adjustments]):.3f}]")
    print(f"Kd范围: [{np.min([np.min(k) for k in kd_adjustments]):.3f}, "
          f"{np.max([np.max(k) for k in kd_adjustments]):.3f}]")
    
    return {
        'mean_reward': np.mean(total_rewards),
        'mean_error': np.mean(tracking_errors),
        'kp_range': (np.min([np.min(k) for k in kp_adjustments]), 
                     np.max([np.max(k) for k in kp_adjustments])),
        'kd_range': (np.min([np.min(k) for k in kd_adjustments]), 
                     np.max([np.max(k) for k in kd_adjustments]))
    }


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练自适应RL agent')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help='训练或评估模式')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='总训练步数')
    parser.add_argument('--n_envs', type=int, default=4,
                       help='并行环境数量')
    parser.add_argument('--disturbance', type=str, default='random_force',
                       choices=['random_force', 'payload', 'terrain'],
                       help='扰动类型')
    parser.add_argument('--model', type=str, default=None,
                       help='评估模式下的模型路径')
    parser.add_argument('--gui', action='store_true',
                       help='评估时显示GUI')
    parser.add_argument('--gpu', action='store_true',
                       help='使用GPU训练')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        model_path = train_adaptive_rl(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            disturbance_type=args.disturbance,
            use_gpu=args.gpu
        )
        print(f"\n🎉 训练完成！模型路径: {model_path}")
        
    elif args.mode == 'eval':
        if args.model is None:
            print("❌ 评估模式需要提供--model参数")
            sys.exit(1)
        
        evaluate_adaptive_policy(
            model_path=args.model,
            n_episodes=10,
            disturbance_type=args.disturbance,
            gui=args.gui
        )

