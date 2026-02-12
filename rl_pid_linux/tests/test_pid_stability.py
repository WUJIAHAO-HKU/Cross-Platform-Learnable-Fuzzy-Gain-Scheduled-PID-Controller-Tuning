"""
测试纯PID控制器稳定性（不加RL）
用于诊断PID基线是否稳定
"""

import sys
sys.path.append('..')

import numpy as np
import pybullet as p
import pybullet_data
import yaml
import matplotlib.pyplot as plt
from controllers.pid_controller import PIDController
from envs.trajectory_gen import TrajectoryGenerator


def test_pid_only(config_path='../configs/pid_fix_static.yaml', duration=10.0):
    """测试纯PID跟踪"""
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化PyBullet（带GUI）
    print("=" * 60)
    print("  纯PID稳定性测试")
    print("=" * 60)
    print("\n⚠️  如果机器人快速发散，说明PID参数有问题！\n")
    
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    
    # 加载机器人
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    joint_indices = list(range(7))
    
    # 禁用默认电机
    for i in joint_indices:
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, force=0)
    
    # PID控制器
    pid_params = config['pid_params']
    pid = PIDController(**pid_params)
    
    # 轨迹生成器
    traj_config = config['trajectory']
    traj_gen = TrajectoryGenerator(traj_config['type'], traj_config)
    
    # 初始化
    init_q = config['robot']['init_position']
    for i, q in enumerate(init_q):
        p.resetJointState(robot_id, i, q, 0)
    
    # 仿真参数
    dt = config['simulation']['time_step']
    steps = int(duration / dt)
    
    # 记录数据
    time_log = []
    q_log = []
    qref_log = []
    error_log = []
    tau_log = []
    
    print("▶️  开始仿真...")
    print(f"   时长: {duration}秒")
    print(f"   步数: {steps}")
    print(f"   dt: {dt}秒\n")
    
    # 主循环
    for step in range(steps):
        t = step * dt
        
        # 获取状态
        joint_states = p.getJointStates(robot_id, joint_indices)
        q = np.array([s[0] for s in joint_states], dtype=np.float32)
        qd = np.array([s[1] for s in joint_states], dtype=np.float32)
        
        # 参考轨迹
        qref, qd_ref = traj_gen.get_reference(t)
        
        # ⭐ 纯PID控制（传入目标速度）
        tau = pid.compute(q, qd, qref, qd_ref)
        
        # 应用力矩
        p.setJointMotorControlArray(
            robot_id, joint_indices, p.TORQUE_CONTROL, forces=tau
        )
        p.stepSimulation()
        
        # 记录
        if step % 100 == 0:  # 每100步记录一次
            time_log.append(t)
            q_log.append(q.copy())
            qref_log.append(qref.copy())
            error_log.append(np.linalg.norm(qref - q))
            tau_log.append(tau.copy())
            
            # 实时打印
            if step % 1000 == 0:
                err = np.linalg.norm(qref - q)
                q_max = np.max(np.abs(q))
                print(f"   t={t:.2f}s: 误差={err:.4f}, q_max={q_max:.2f}, tau_max={np.max(np.abs(tau)):.2f}")
        
        # 检查发散
        if np.any(np.abs(q) > 3.5) or np.any(np.isnan(q)):
            print(f"\n❌ 发散检测！在 t={t:.2f}s")
            print(f"   关节角度: {q}")
            break
    
    p.disconnect()
    
    # 转换为numpy
    time_log = np.array(time_log)
    q_log = np.array(q_log)
    qref_log = np.array(qref_log)
    error_log = np.array(error_log)
    tau_log = np.array(tau_log)
    
    # 分析结果
    print("\n" + "=" * 60)
    print("  结果分析")
    print("=" * 60)
    
    final_error = error_log[-1]
    max_error = np.max(error_log)
    mean_error = np.mean(error_log)
    
    print(f"\n📊 跟踪性能:")
    print(f"   最终误差: {final_error:.4f} 弧度")
    print(f"   最大误差: {max_error:.4f} 弧度")
    print(f"   平均误差: {mean_error:.4f} 弧度")
    
    print(f"\n🔧 控制力矩:")
    print(f"   最大力矩: {np.max(np.abs(tau_log)):.2f} Nm")
    print(f"   平均力矩: {np.mean(np.abs(tau_log)):.2f} Nm")
    
    # 判断稳定性
    print(f"\n✅/❌ 稳定性判断:")
    if max_error > 0.5:
        print(f"   ❌ PID不稳定！最大误差{max_error:.2f} > 0.5弧度")
        print(f"   ⚠️  建议降低PID增益：Kp减半试试")
        stable = False
    elif mean_error > 0.1:
        print(f"   ⚠️  PID勉强稳定，但误差较大（均值{mean_error:.3f}）")
        print(f"   💡 可以微调PID参数")
        stable = True
    else:
        print(f"   ✅ PID稳定！误差在合理范围")
        stable = True
    
    # 绘图
    plot_results(time_log, q_log, qref_log, error_log, tau_log)
    
    return stable, {
        'final_error': final_error,
        'max_error': max_error,
        'mean_error': mean_error,
        'max_torque': np.max(np.abs(tau_log))
    }


def plot_results(time_log, q_log, qref_log, error_log, tau_log):
    """绘制结果"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. 关节角度跟踪
    ax = axes[0]
    for i in range(3):  # 只画前3个关节
        ax.plot(time_log, q_log[:, i], label=f'q{i+1}', alpha=0.7)
        ax.plot(time_log, qref_log[:, i], '--', label=f'qref{i+1}', alpha=0.5)
    ax.set_ylabel('关节角度 (rad)')
    ax.set_title('PID跟踪性能（前3关节）')
    ax.legend(ncol=6, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. 跟踪误差
    ax = axes[1]
    ax.plot(time_log, error_log, 'r-', linewidth=2, label='|qref-q|')
    ax.axhline(0.1, color='orange', linestyle='--', label='0.1 rad (目标)')
    ax.axhline(0.5, color='red', linestyle='--', label='0.5 rad (极限)')
    ax.set_ylabel('误差范数 (rad)')
    ax.set_title('跟踪误差')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(1.0, np.max(error_log) * 1.1))
    
    # 3. 控制力矩
    ax = axes[2]
    for i in range(3):
        ax.plot(time_log, tau_log[:, i], label=f'tau{i+1}', alpha=0.7)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('力矩 (Nm)')
    ax.set_title('控制力矩（前3关节）')
    ax.legend(ncol=3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pid_stability_test.png', dpi=150)
    print(f"\n📊 图表已保存: pid_stability_test.png")
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/stage1_small.yaml')
    parser.add_argument('--duration', type=float, default=10.0, help='仿真时长（秒）')
    args = parser.parse_args()
    
    stable, metrics = test_pid_only(args.config, args.duration)
    
    print("\n" + "=" * 60)
    if stable:
        print("  ✅ PID基线稳定，可以继续RL训练")
        print("  💡 下一步：运行RL训练")
    else:
        print("  ❌ PID基线不稳定，必须先修复PID！")
        print("  🔧 建议：降低Kp增益（减半）")
    print("=" * 60)

