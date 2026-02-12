"""
完整系统测试
验证所有组件能否正常工作
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

print("=" * 70)
print("  完整系统测试")
print("=" * 70)
print()

# 测试1：导入所有模块
print(">>> [1/6] 测试模块导入...")
try:
    from controllers.pid_controller import PIDController, get_default_pid_gains
    from controllers.rl_pid_hybrid import RLPIDHybrid, compute_reward
    from envs.trajectory_gen import TrajectoryGenerator
    from envs.franka_env import FrankaRLPIDEnv
    print("    ✅ 所有模块导入成功")
except Exception as e:
    print(f"    ❌ 模块导入失败: {e}")
    sys.exit(1)
print()

# 测试2：PID控制器
print(">>> [2/6] 测试PID控制器...")
try:
    gains = get_default_pid_gains()
    pid = PIDController(**gains)
    
    q = np.random.randn(7) * 0.1
    qd = np.random.randn(7) * 0.1
    qref = np.zeros(7)
    
    tau = pid.compute(q, qd, qref)
    assert tau.shape == (7,), "PID输出维度错误"
    assert not np.any(np.isnan(tau)), "PID输出包含NaN"
    
    print(f"    ✅ PID控制器工作正常")
    print(f"       输出力矩范围: [{tau.min():.2f}, {tau.max():.2f}] Nm")
except Exception as e:
    print(f"    ❌ PID测试失败: {e}")
    sys.exit(1)
print()

# 测试3：轨迹生成器
print(">>> [3/6] 测试轨迹生成器...")
try:
    traj_gen = TrajectoryGenerator('circle', {'speed': 0.3, 'amplitude': 0.2})
    
    qref_0, qd_ref_0 = traj_gen.get_reference(0.0)
    qref_1, qd_ref_1 = traj_gen.get_reference(1.0)
    
    assert qref_0.shape == (7,), "轨迹输出维度错误"
    assert qd_ref_0.shape == (7,), "速度输出维度错误"
    assert not np.allclose(qref_0, qref_1), "轨迹应该随时间变化"
    
    print("    ✅ 轨迹生成器工作正常")
    print(f"       t=0s: qref={qref_0[:3]}, qd_ref={qd_ref_0[:3]}")
    print(f"       t=1s: qref={qref_1[:3]}, qd_ref={qd_ref_1[:3]}")
except Exception as e:
    print(f"    ❌ 轨迹生成器测试失败: {e}")
    sys.exit(1)
print()

# 测试4：RL+PID控制器
print(">>> [4/6] 测试RL+PID混合控制器...")
try:
    config = {
        'pid_params': get_default_pid_gains(),
        'rl_params': {
            'delta_scale_min': 0.5,
            'delta_scale_max': 5.0,
            'warmup_disable_steps': 100,
            'warmup_ramp_steps': 500,
            'delta_tau_clip': 10.0
        }
    }
    
    controller = RLPIDHybrid(config)
    
    q = np.random.randn(7) * 0.1
    qd = np.random.randn(7) * 0.1
    qref = np.zeros(7)
    
    tau_total, info = controller.compute_control(q, qd, qref)
    
    assert tau_total.shape == (7,), "控制器输出维度错误"
    assert 'delta_scale' in info, "info应该包含delta_scale"
    assert info['delta_scale'] == 0.0, "前100步应该是纯PID（delta_scale=0）"
    
    # 测试渐进式策略
    controller.step_count = 300
    _, info = controller.compute_control(q, qd, qref)
    assert 0 < info['delta_scale'] < 5.0, "step=300时应该在渐进增加"
    
    controller.step_count = 700
    _, info = controller.compute_control(q, qd, qref)
    assert info['delta_scale'] == 5.0, "step=700时应该达到最大值"
    
    print("    ✅ RL+PID控制器工作正常")
    print("       渐进式策略验证通过:")
    print(f"         Step 0:   delta_scale = 0.0")
    print(f"         Step 300: delta_scale = {0.5 + (300-100)/500 * (5.0-0.5):.2f}")
    print(f"         Step 700: delta_scale = 5.0")
except Exception as e:
    print(f"    ❌ RL+PID测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# 测试5：加载配置文件
print(">>> [5/6] 测试配置文件加载...")
try:
    config_path = Path(__file__).parent.parent / 'configs' / 'stage1_small.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    assert 'robot' in config, "配置应该包含robot"
    assert 'rl_params' in config, "配置应该包含rl_params"
    assert config['rl_params']['delta_scale_max'] == 2.0, "阶段1应该是2.0"
    
    print(f"    ✅ 配置文件加载成功: {config_path.name}")
    print(f"       Delta Scale Max: {config['rl_params']['delta_scale_max']}")
    print(f"       Total Timesteps: {config['training']['total_timesteps']}")
except Exception as e:
    print(f"    ❌ 配置文件测试失败: {e}")
    sys.exit(1)
print()

# 测试6：创建环境（完整测试）
print(">>> [6/6] 测试PyBullet环境创建...")
try:
    env = FrankaRLPIDEnv(config, gui=False)
    
    # 测试reset
    obs, info = env.reset()
    assert obs.shape == (14,), f"观测维度错误: {obs.shape}"
    
    # 测试step
    action = np.zeros(7)  # 零动作
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs.shape == (14,), "step后观测维度错误"
    assert isinstance(reward, (int, float, np.number)), "奖励应该是数值"
    assert 'tracking_error' in info, "info应该包含tracking_error"
    
    env.close()
    
    print("    ✅ 环境创建和交互成功")
    print(f"       观测空间: {env.observation_space.shape}")
    print(f"       动作空间: {env.action_space.shape}")
    print(f"       初始跟踪误差: {info['tracking_error']:.4f} rad")
except Exception as e:
    print(f"    ❌ 环境测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# 总结
print("=" * 70)
print("  ✅ 所有测试通过！系统完全就绪！")
print("=" * 70)
print()
print("下一步：")
print("  cd ~/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/rl_pid_linux")
print("  python training/train_ddpg.py --config configs/stage1_small.yaml")
print()
print("=" * 70)

