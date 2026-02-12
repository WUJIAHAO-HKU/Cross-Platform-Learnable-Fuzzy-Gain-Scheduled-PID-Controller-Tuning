#!/usr/bin/env python3
"""
快速测试自适应RL训练流程（小规模验证）
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from train_adaptive_rl import train_adaptive_rl

if __name__ == '__main__':
    print("=" * 80)
    print("🧪 快速测试：自适应RL训练流程")
    print("=" * 80)
    print("\n⚠️  这是一个小规模测试（20000步），用于验证流程是否正常")
    print("完整训练请使用: python train_adaptive_rl.py --timesteps 500000 --gpu\n")
    
    # 小规模训练
    model_path = train_adaptive_rl(
        total_timesteps=20000,  # 仅20k步，快速验证
        n_envs=2,  # 仅2个环境，降低资源占用
        learning_rate=3e-4,
        batch_size=128,
        n_epochs=5,
        disturbance_type='random_force',
        save_dir='./logs/adaptive_rl_test',
        use_gpu=False  # 测试时不用GPU
    )
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)
    print(f"模型保存在: {model_path}")
    print("\n如果测试通过，可以开始完整训练：")
    print("  python train_adaptive_rl.py --timesteps 500000 --n_envs 4 --gpu")

