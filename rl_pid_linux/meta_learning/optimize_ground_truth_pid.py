#!/usr/bin/env python3
"""
为Laikago和KUKA优化真实的最优PID参数
使用Bayesian优化（与Franka相同的方法）
"""

import numpy as np
import pybullet as p
import pybullet_data
from scipy.optimize import differential_evolution
import json
from pathlib import Path


def evaluate_pid(params, robot_urdf, duration=5.0, verbose=False):
    """
    评估PID参数的性能
    
    Args:
        params: [kp, kd] (简化，不使用Ki)
        robot_urdf: 机器人URDF路径
        duration: 仿真时长
    
    Returns:
        平均跟踪误差（越小越好）
    """
    kp, kd = params
    
    # 启动仿真
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)
    
    # 加载机器人
    robot_id = p.loadURDF(robot_urdf, [0, 0, 0.5], useFixedBase=True)
    num_joints = p.getNumJoints(robot_id)
    
    # 获取可控关节
    controllable_joints = []
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        if info[2] != p.JOINT_FIXED:
            controllable_joints.append(j)
    
    n_dof = len(controllable_joints)
    
    # 生成正弦参考轨迹
    dt = 1./240.
    total_steps = int(duration / dt)
    
    errors = []
    
    for step in range(total_steps):
        t = step * dt
        
        # 正弦参考轨迹
        q_ref = np.array([0.3 * np.sin(2 * np.pi * 0.5 * t + i * 0.5) for i in range(n_dof)])
        
        # 使用POSITION_CONTROL（内置PD控制器）
        p.setJointMotorControlArray(
            robot_id,
            controllable_joints,
            p.POSITION_CONTROL,
            targetPositions=q_ref,
            positionGains=[kp] * n_dof,
            velocityGains=[kd] * n_dof,
            forces=[100.0] * n_dof  # 足够大的力矩限制
        )
        
        p.stepSimulation()
        
        # 获取当前状态
        joint_states = p.getJointStates(robot_id, controllable_joints)
        q = np.array([state[0] for state in joint_states])
        
        # 计算误差
        error = np.linalg.norm(q_ref - q)
        errors.append(error)
    
    p.disconnect(client)
    
    mean_error = np.mean(errors)
    
    if verbose:
        print(f"   Kp={kp:.4f}, Kd={kd:.4f} -> 误差={mean_error:.4f} rad ({np.rad2deg(mean_error):.2f}°)")
    
    return mean_error


def optimize_pid_for_robot(robot_urdf, robot_name):
    """
    为单个机器人优化PID参数
    """
    print(f"\n{'='*80}")
    print(f"优化 {robot_name}")
    print(f"{'='*80}")
    
    # 定义搜索空间
    if 'laikago' in robot_name.lower():
        bounds = [(0.1, 50.0), (0.01, 10.0)]  # Laikago: 小机器人
    else:  # KUKA
        bounds = [(10.0, 200.0), (1.0, 30.0)]  # KUKA: 中等机器人
    
    print(f"搜索空间:")
    print(f"   Kp: [{bounds[0][0]}, {bounds[0][1]}]")
    print(f"   Kd: [{bounds[1][0]}, {bounds[1][1]}]")
    
    # 定义目标函数
    def objective(params):
        return evaluate_pid(params, robot_urdf, duration=5.0, verbose=False)
    
    # 使用差分进化算法优化
    print(f"\n🚀 开始优化（这可能需要几分钟）...")
    
    result = differential_evolution(
        objective,
        bounds,
        maxiter=30,        # 最大迭代次数
        popsize=10,        # 种群大小
        tol=0.001,         # 收敛容差
        seed=42,
        workers=1,         # PyBullet不支持多进程
        updating='immediate',
        disp=True
    )
    
    kp_opt, kd_opt = result.x
    error_opt = result.fun
    
    print(f"\n✅ 优化完成！")
    print(f"   最优 Kp = {kp_opt:.4f}")
    print(f"   最优 Kd = {kd_opt:.4f}")
    print(f"   最优 Ki = 0.0000 (固定)")
    print(f"   最小误差 = {error_opt:.4f} rad ({np.rad2deg(error_opt):.2f}°)")
    
    # 详细验证
    print(f"\n📊 详细验证（10秒仿真）:")
    final_error = evaluate_pid([kp_opt, kd_opt], robot_urdf, duration=10.0, verbose=True)
    
    return {
        'kp': float(kp_opt),
        'ki': 0.0,
        'kd': float(kd_opt),
        'error_deg': float(np.rad2deg(final_error))
    }


def main():
    """主优化流程"""
    print("=" * 80)
    print("优化真实PID参数（Ground Truth）")
    print("=" * 80)
    
    robots_to_optimize = [
        ('laikago/laikago.urdf', 'Laikago'),
        ('kuka_iiwa/model.urdf', 'KUKA iiwa'),
    ]
    
    results = {}
    
    for robot_urdf, robot_name in robots_to_optimize:
        optimal_pid = optimize_pid_for_robot(robot_urdf, robot_name)
        results[robot_name] = {
            'urdf': robot_urdf,
            'optimal_pid': optimal_pid
        }
    
    # 包含Franka的已知最优值
    results['Franka Panda'] = {
        'urdf': 'franka_panda/panda.urdf',
        'optimal_pid': {
            'kp': 142.53,
            'ki': 1.43,
            'kd': 14.25,
            'error_deg': 2.1  # 已知的最优误差
        }
    }
    
    # 保存结果
    output_path = Path(__file__).parent / 'optimized_ground_truth_pid.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"优化总结")
    print(f"{'='*80}")
    for robot_name, data in results.items():
        pid = data['optimal_pid']
        print(f"\n{robot_name}:")
        print(f"   Kp = {pid['kp']:.4f}")
        print(f"   Ki = {pid['ki']:.4f}")
        print(f"   Kd = {pid['kd']:.4f}")
        print(f"   误差 = {pid['error_deg']:.2f}°")
    
    print(f"\n💾 结果已保存: {output_path}")
    print(f"{'='*80}")
    
    print(f"\n🎯 下一步:")
    print(f"   1. 使用这些真实最优PID重新生成数据增强")
    print(f"   2. 重新训练元学习PID网络")
    print(f"   3. 重新评估性能提升")


if __name__ == '__main__':
    main()

