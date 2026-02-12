"""
DDPG训练脚本
使用Stable-Baselines3训练RL+PID策略
"""

import argparse
import yaml
import os
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from envs.franka_env import FrankaRLPIDEnv


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train(config_path, output_dir='./logs', model_name='rl_pid'):
    """训练RL+PID策略"""
    
    print("=" * 70)
    print("  RL+PID DDPG训练")
    print("=" * 70)
    
    # 检测GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  设备检测:")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   使用设备: {device.upper()}")
    if device == 'cuda':
        print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("   ✅ GPU加速已启用，训练速度将提升3-5倍！")
    else:
        print("   ⚠️  GPU不可用，使用CPU训练")
    
    # 加载配置
    config = load_config(config_path)
    print(f"\n✅ 配置加载完成: {config_path}")
    print(f"   Delta Scale Max: {config['rl_params']['delta_scale_max']}")
    print(f"   Total Timesteps: {config['training']['total_timesteps']}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/tensorboard", exist_ok=True)
    os.makedirs(f"{output_dir}/eval", exist_ok=True)
    
    # 创建环境
    print("\n✅ 创建训练环境...")
    
    # 并行环境数量（可根据CPU核心数调整）
    n_envs = config.get('n_envs', 1)  # 默认1，可在配置文件中设置
    
    if n_envs > 1:
        # 多环境并行训练
        print(f"   使用 {n_envs} 个并行环境")
        
        def make_env():
            def _init():
                return FrankaRLPIDEnv(config, gui=False)
            return _init
        
        train_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        eval_env = FrankaRLPIDEnv(config, gui=False)
        print(f"   ✅ {n_envs} 个并行训练环境创建成功")
    else:
        # 单环境训练（当前默认）
        train_env = FrankaRLPIDEnv(config, gui=False)
        eval_env = FrankaRLPIDEnv(config, gui=False)
        print("   ✅ 单环境训练模式")
    
    # 动作噪声（用于探索）
    if n_envs > 1:
        n_actions = train_env.action_space.shape[0]  # VecEnv的action_space
    else:
        n_actions = train_env.action_space.shape[0]
    
    noise_std = config['rl_params']['noise_std']
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=noise_std * np.ones(n_actions)
    )
    print(f"   探索噪声: {noise_std}")
    
    # 创建DDPG模型
    print("\n✅ 创建DDPG模型...")
    model = DDPG(
        "MlpPolicy",
        train_env,
        learning_rate=config['rl_params']['learning_rate_actor'],
        buffer_size=config['rl_params']['buffer_size'],
        batch_size=config['rl_params']['batch_size'],
        gamma=config['rl_params']['gamma'],
        tau=config['rl_params']['tau'],
        action_noise=action_noise,
        policy_kwargs={
            'net_arch': {
                'pi': config['rl_params']['actor_hidden'],
                'qf': config['rl_params']['critic_hidden']
            }
        },
        tensorboard_log=f"{output_dir}/tensorboard/",
        device=device,  # ⭐ GPU/CPU设备
        verbose=1
    )
    print("   模型创建成功")
    print(f"   Actor网络: {config['rl_params']['actor_hidden']}")
    print(f"   Critic网络: {config['rl_params']['critic_hidden']}")
    
    # 回调函数
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{output_dir}/models/best/",
        log_path=f"{output_dir}/eval/",
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['n_eval_episodes'],
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=f"{output_dir}/models/checkpoints/",
        name_prefix=model_name
    )
    
    callback = CallbackList([eval_callback, checkpoint_callback])
    
    # 开始训练
    print("\n" + "=" * 70)
    print("  开始训练")
    print("=" * 70)
    print(f"\n📊 监控训练进度：")
    print(f"   tensorboard --logdir={output_dir}/tensorboard/")
    print()
    
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callback,
            log_interval=config['training']['log_interval']
        )
        
        print("\n" + "=" * 70)
        print("  ✅ 训练完成！")
        print("=" * 70)
        
        # 保存最终模型
        final_model_path = f"{output_dir}/models/{model_name}_final"
        model.save(final_model_path)
        print(f"\n✅ 最终模型已保存: {final_model_path}.zip")
        
        # 保存配置
        config_save_path = f"{output_dir}/models/{model_name}_config.yaml"
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        print(f"✅ 配置已保存: {config_save_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被中断")
        save_path = f"{output_dir}/models/{model_name}_interrupted"
        model.save(save_path)
        print(f"✅ 中断模型已保存: {save_path}.zip")
    
    finally:
        train_env.close()
        eval_env.close()
        print("\n✅ 环境已关闭")


def main():
    parser = argparse.ArgumentParser(description='训练RL+PID DDPG模型')
    parser.add_argument('--config', type=str, default='configs/stage1_small.yaml',
                        help='配置文件路径')
    parser.add_argument('--output', type=str, default='./logs',
                        help='输出目录')
    parser.add_argument('--name', type=str, default='rl_pid_stage1',
                        help='模型名称')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)
    
    train(args.config, args.output, args.name)


if __name__ == "__main__":
    main()

