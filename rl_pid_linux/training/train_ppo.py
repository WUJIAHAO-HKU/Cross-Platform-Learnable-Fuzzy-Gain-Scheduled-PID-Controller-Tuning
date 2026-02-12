"""
PPO训练脚本（替代DDPG）
PPO对不稳定环境更鲁棒
包含完整的资源清理机制，防止GPU泄漏
"""

import os
import argparse
import yaml
import torch
import signal
import atexit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from envs.franka_env import FrankaRLPIDEnv

# 全局变量用于资源清理
train_env_global = None
eval_env_global = None
model_global = None


def cleanup_resources():
    """清理GPU和环境资源"""
    global train_env_global, eval_env_global, model_global
    
    print("\n🧹 正在清理资源...")
    
    try:
        # 关闭环境
        if train_env_global is not None:
            try:
                train_env_global.close()
                print("   ✅ 训练环境已关闭")
            except Exception as e:
                print(f"   ⚠️  训练环境关闭失败: {e}")
        
        if eval_env_global is not None:
            try:
                eval_env_global.close()
                print("   ✅ 评估环境已关闭")
            except Exception as e:
                print(f"   ⚠️  评估环境关闭失败: {e}")
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   ✅ GPU缓存已清空")
    
    except Exception as e:
        print(f"   ⚠️  清理过程出错: {e}")
    
    print("   ✅ 资源清理完成")


def signal_handler(sig, frame):
    """处理中断信号（Ctrl+C）"""
    print("\n⚠️  检测到中断信号 (Ctrl+C)")
    cleanup_resources()
    print("   程序已安全退出")
    sys.exit(0)


# 注册信号处理器和退出清理
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_resources)


def train(config_path, output_dir='./logs', model_name='rl_pid_ppo'):
    """训练PPO模型"""
    global train_env_global, eval_env_global, model_global
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # GPU检测
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n" + "="*70)
    print("  RL+PID PPO训练")
    print("="*70)
    print(f"\n🖥️  设备检测:")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   使用设备: {device.upper()}")
    if device == 'cuda':
        print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("   ✅ GPU加速已启用！")
    else:
        print("   ⚠️  GPU不可用，使用CPU训练")
    
    print(f"\n✅ 配置加载完成: {config_path}")
    print(f"   Delta Scale Max: {config['rl_params']['delta_scale_max']}")
    print(f"   Total Timesteps: {config['training']['total_timesteps']}")
    
    # 创建环境
    print(f"\n✅ 创建训练环境...")
    n_envs = config.get('n_envs', 4)  # PPO推荐多环境
    
    # ⭐ 使用真正的多进程并行（SubprocVecEnv）
    # 子进程运行环境仿真（CPU），主进程运行神经网络（GPU）
    if n_envs > 1:
        print(f"   使用 {n_envs} 个并行进程 (SubprocVecEnv)")
        if device == 'cuda':
            print(f"   ⚙️  环境仿真在子进程(CPU) + 神经网络在主进程(GPU)")
        else:
            print(f"   ⚙️  全部使用CPU模式")
        
        def make_env(rank):
            """
            创建环境的工厂函数
            每个子进程会独立调用这个函数
            """
            def _init():
                # 在子进程中创建环境（不使用GPU）
                env = FrankaRLPIDEnv(config, gui=False)
                return Monitor(env, info_keywords=())
            return _init
        
        # 创建多个独立的子进程，每个运行一个环境
        train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        print(f"   ✅ {n_envs} 个并行训练环境创建成功")
    else:
        print("   使用单环境训练模式")
        train_env = FrankaRLPIDEnv(config, gui=False)
        train_env = Monitor(train_env)
        print("   ✅ 单环境训练模式")
    
    # 评估环境
    eval_env = FrankaRLPIDEnv(config, gui=False)
    eval_env = Monitor(eval_env)
    
    # ⭐ 保存到全局变量以便清理
    train_env_global = train_env
    eval_env_global = eval_env
    
    # PPO超参数
    rl_params = config.get('rl_params', {})
    training_config = config.get('training', {})
    
    # ⭐ PPO特有参数
    n_steps = 2048  # 每次更新的步数（4个环境×2048=8192样本/更新）
    batch_size = 256  # mini-batch大小（8192/256=32个mini-batch）
    n_epochs = 10    # 每次更新的epoch数
    
    print(f"\n✅ 创建PPO模型...")
    print(f"   n_steps: {n_steps}")
    print(f"   batch_size: {batch_size}")
    print(f"   n_epochs: {n_epochs}")
    
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=rl_params.get('learning_rate_actor', 0.0003),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=rl_params.get('gamma', 0.99),
        gae_lambda=0.95,  # GAE参数
        clip_range=0.2,   # PPO裁剪参数
        ent_coef=0.01,    # 熵系数（鼓励探索）
        vf_coef=0.5,      # 价值函数系数
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=rl_params.get('actor_hidden', [256, 128, 64]),
                vf=rl_params.get('critic_hidden', [256, 256, 128])
            )
        ),
        tensorboard_log=f"{output_dir}/tensorboard/",
        device=device,
        verbose=1
    )
    
    print("   ✅ PPO模型创建成功")
    
    # ⭐ 保存模型到全局变量
    model_global = model
    
    # Callbacks
    os.makedirs(output_dir, exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{output_dir}/best_model/",
        log_path=f"{output_dir}/eval/",
        eval_freq=training_config.get('eval_freq', 10000),
        n_eval_episodes=training_config.get('n_eval_episodes', 5),
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config.get('save_freq', 50000),
        save_path=f"{output_dir}/checkpoints/",
        name_prefix=model_name
    )
    
    # 训练
    print("\n" + "="*70)
    print("  开始训练")
    print("="*70)
    print(f"\n📊 监控训练进度：")
    print(f"   tensorboard --logdir={output_dir}/tensorboard/\n")
    
    try:
        model.learn(
            total_timesteps=training_config.get('total_timesteps', 500000),
            callback=[eval_callback, checkpoint_callback],
            log_interval=training_config.get('log_interval', 10),
            progress_bar=False  # ⭐ 禁用进度条避免依赖问题
        )
        
        # 保存最终模型
        final_model_path = f"{output_dir}/{model_name}_final"
        model.save(final_model_path)
        print(f"\n✅ 训练完成！模型已保存至: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被中断 (Ctrl+C)")
        try:
            interrupted_path = f"{output_dir}/{model_name}_interrupted"
            model.save(interrupted_path)
            print(f"   ✅ 中断时的模型已保存至: {interrupted_path}")
        except Exception as e:
            print(f"   ⚠️  保存中断模型失败: {e}")
    
    except Exception as e:
        print(f"\n❌ 训练过程出错: {e}")
        print(f"   错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        try:
            error_path = f"{output_dir}/{model_name}_error"
            model.save(error_path)
            print(f"   ✅ 错误时的模型已保存至: {error_path}")
        except Exception as save_error:
            print(f"   ⚠️  保存错误模型失败: {save_error}")
    
    finally:
        # ⭐ 确保资源一定被清理
        print("\n🧹 清理训练资源...")
        try:
            train_env.close()
            print("   ✅ 训练环境已关闭")
        except Exception as e:
            print(f"   ⚠️  关闭训练环境失败: {e}")
        
        try:
            eval_env.close()
            print("   ✅ 评估环境已关闭")
        except Exception as e:
            print(f"   ⚠️  关闭评估环境失败: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   ✅ GPU缓存已清空")
        
        print("   ✅ 所有资源已清理完成")


def main():
    parser = argparse.ArgumentParser(description='PPO训练')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--output', type=str, default='./logs',
                        help='输出目录')
    parser.add_argument('--name', type=str, default='rl_pid_ppo',
                        help='模型名称')
    
    args = parser.parse_args()
    train(args.config, args.output, args.name)


if __name__ == '__main__':
    main()

