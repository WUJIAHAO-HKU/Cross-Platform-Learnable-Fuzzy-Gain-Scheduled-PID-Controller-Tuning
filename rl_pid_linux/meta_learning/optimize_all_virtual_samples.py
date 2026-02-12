#!/usr/bin/env python3
"""
为所有虚拟样本优化真实最优PID
使用多进程并行加速
"""

import numpy as np
import pybullet as p
import pybullet_data
import json
from pathlib import Path
from scipy.optimize import differential_evolution
from multiprocessing import Pool, cpu_count
import time
from tqdm import tqdm


# ============================================================================
# PID优化函数
# ============================================================================
def optimize_pid_for_virtual_robot(args):
    """
    为单个虚拟机器人优化PID
    
    Args:
        args: (robot_urdf, params, bounds, robot_id)
    
    Returns:
        dict: 优化结果
    """
    robot_urdf, params, bounds, robot_id = args
    
    # 创建独立的PyBullet客户端
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1./240., physicsClientId=client)
    
    # 加载机器人并应用参数修改
    robot_body_id = p.loadURDF(robot_urdf, [0, 0, 0.5], useFixedBase=True, physicsClientId=client)
    num_joints = p.getNumJoints(robot_body_id, physicsClientId=client)
    
    # 应用虚拟参数
    for j in range(num_joints):
        dyn_info = p.getDynamicsInfo(robot_body_id, j, physicsClientId=client)
        original_mass = dyn_info[0]
        
        p.changeDynamics(
            robot_body_id, j,
            mass=original_mass * params['mass_scale'],
            lateralFriction=params['friction'],
            linearDamping=dyn_info[6] * params['damping'],
            angularDamping=dyn_info[7] * params['damping'],
            physicsClientId=client
        )
    
    # 获取可控关节
    controllable_joints = []
    for j in range(num_joints):
        info = p.getJointInfo(robot_body_id, j, physicsClientId=client)
        if info[2] != p.JOINT_FIXED:
            controllable_joints.append(j)
    
    n_dof = len(controllable_joints)
    
    # 定义评估函数
    def evaluate_pid(pid_params):
        """评估PID性能"""
        kp, kd = pid_params
        
        # 重置机器人
        for j in controllable_joints:
            p.resetJointState(robot_body_id, j, 0.0, physicsClientId=client)
        
        # 仿真
        dt = 1./240.
        duration = 3.0  # 缩短仿真时间加速
        total_steps = int(duration / dt)
        
        errors = []
        for step in range(total_steps):
            t = step * dt
            # 正弦参考轨迹
            q_ref = np.array([0.3 * np.sin(2 * np.pi * 0.5 * t + i * 0.5) for i in range(n_dof)])
            
            # POSITION_CONTROL
            p.setJointMotorControlArray(
                robot_body_id,
                controllable_joints,
                p.POSITION_CONTROL,
                targetPositions=q_ref,
                positionGains=[kp] * n_dof,
                velocityGains=[kd] * n_dof,
                forces=[100.0] * n_dof,
                physicsClientId=client
            )
            
            p.stepSimulation(physicsClientId=client)
            
            # 获取状态
            joint_states = p.getJointStates(robot_body_id, controllable_joints, physicsClientId=client)
            q = np.array([state[0] for state in joint_states])
            
            # 计算误差
            error = np.linalg.norm(q_ref - q)
            errors.append(error)
        
        return np.mean(errors)
    
    # 差分进化优化（混合策略：粗搜索+精搜索）
    try:
        result = differential_evolution(
            evaluate_pid,
            bounds,
            maxiter=15,         # 减少迭代次数（粗搜索）
            popsize=8,          # 种群大小
            tol=0.01,
            seed=42 + robot_id,  # 每个进程不同的随机种子
            workers=1,
            updating='immediate',
            polish=True,        # 🔥 自动用L-BFGS-B局部优化（精搜索）
            disp=False
        )
        
        kp_opt, kd_opt = result.x
        error_opt = result.fun
        
        # 断开连接
        p.disconnect(client)
        
        return {
            'robot_id': robot_id,
            'kp': float(kp_opt),
            'ki': 0.0,
            'kd': float(kd_opt),
            'error_rad': float(error_opt),
            'error_deg': float(np.rad2deg(error_opt)),
            'success': True
        }
    
    except Exception as e:
        p.disconnect(client)
        return {
            'robot_id': robot_id,
            'error': str(e),
            'success': False
        }


# ============================================================================
# 主优化流程
# ============================================================================
def optimize_all_virtual_samples(n_workers=None):
    """
    为所有虚拟样本优化PID
    
    Args:
        n_workers: 并行进程数（默认：CPU核心数-1）
    """
    print("=" * 80)
    print("为所有虚拟样本优化真实最优PID")
    print("=" * 80)
    
    # 加载现有的增强数据（包含虚拟样本）
    data_path = Path(__file__).parent / 'augmented_pid_data.json'
    with open(data_path, 'r') as f:
        augmented_data = json.load(f)
    
    print(f"\n📦 加载数据: {len(augmented_data)}个样本")
    
    # 筛选需要优化的虚拟样本
    virtual_samples = [d for d in augmented_data if d['type'] == 'virtual']
    real_samples = [d for d in augmented_data if d['type'] == 'real']
    
    print(f"   真实样本: {len(real_samples)} (已有最优PID)")
    print(f"   虚拟样本: {len(virtual_samples)} (需要优化)")
    
    # 准备优化任务
    tasks = []
    for i, sample in enumerate(virtual_samples):
        # 确定搜索空间（根据基础机器人类型）
        if 'laikago' in sample['name']:
            base_urdf = 'laikago/laikago.urdf'
            bounds = [(0.1, 50.0), (0.01, 10.0)]
        elif 'kuka' in sample['name'] or 'model' in sample['name']:
            base_urdf = 'kuka_iiwa/model.urdf'
            bounds = [(1.0, 100.0), (0.5, 20.0)]
        else:  # franka
            base_urdf = 'franka_panda/panda.urdf'
            bounds = [(50.0, 300.0), (5.0, 30.0)]
        
        tasks.append((
            base_urdf,
            sample['augmentation_params'],
            bounds,
            i
        ))
    
    # 确定并行进程数
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    print(f"\n🚀 开始并行优化:")
    print(f"   总任务数: {len(tasks)}")
    print(f"   并行进程: {n_workers}")
    print(f"   预计耗时: {len(tasks) * 3 / n_workers / 60:.1f} 分钟")
    print(f"\n   (每个样本约3分钟，{n_workers}核并行)")
    
    start_time = time.time()
    
    # 并行优化
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(optimize_pid_for_virtual_robot, tasks),
            total=len(tasks),
            desc="优化进度",
            ncols=80
        ))
    
    elapsed_time = time.time() - start_time
    
    # 统计成功/失败
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n✅ 优化完成！")
    print(f"   总耗时: {elapsed_time/60:.1f} 分钟")
    print(f"   成功: {len(successful)}/{len(tasks)}")
    print(f"   失败: {len(failed)}/{len(tasks)}")
    
    if failed:
        print(f"\n⚠️  失败样本ID: {[r['robot_id'] for r in failed]}")
    
    # 更新虚拟样本的PID
    for i, result in enumerate(results):
        if result['success']:
            virtual_samples[i]['optimal_pid'] = {
                'kp': result['kp'],
                'ki': result['ki'],
                'kd': result['kd']
            }
            virtual_samples[i]['optimization_error_deg'] = result['error_deg']
            virtual_samples[i]['optimized'] = True
        else:
            virtual_samples[i]['optimized'] = False
    
    # 合并真实样本和优化后的虚拟样本
    optimized_data = real_samples + virtual_samples
    
    # 保存结果
    output_path = Path(__file__).parent / 'augmented_pid_data_optimized.json'
    with open(output_path, 'w') as f:
        json.dump(optimized_data, f, indent=2)
    
    print(f"\n💾 优化后的数据已保存: {output_path}")
    
    # 统计分析
    print(f"\n📊 优化质量统计:")
    errors = [r['error_deg'] for r in successful]
    print(f"   平均误差: {np.mean(errors):.2f}°")
    print(f"   中位误差: {np.median(errors):.2f}°")
    print(f"   最小误差: {np.min(errors):.2f}°")
    print(f"   最大误差: {np.max(errors):.2f}°")
    
    # 按类型分组统计
    print(f"\n   按机器人类型:")
    for robot_type in ['laikago', 'kuka', 'panda']:
        type_results = [r for r in successful if robot_type in virtual_samples[r['robot_id']]['name']]
        if type_results:
            type_errors = [r['error_deg'] for r in type_results]
            print(f"   {robot_type.capitalize():8s}: 平均误差={np.mean(type_errors):.2f}° (n={len(type_results)})")
    
    print(f"\n🎯 下一步:")
    print(f"   1. 使用 augmented_pid_data_optimized.json 重新训练")
    print(f"   2. 评估真实数据训练的性能提升")
    print(f"   3. 撰写论文！")
    
    return optimized_data, results


# ============================================================================
# 快速测试（优化10个样本）
# ============================================================================
def quick_test(n_samples=10):
    """快速测试：优化少量样本验证流程"""
    print("=" * 80)
    print(f"快速测试：优化{n_samples}个虚拟样本")
    print("=" * 80)
    
    # 加载数据
    data_path = Path(__file__).parent / 'augmented_pid_data.json'
    with open(data_path, 'r') as f:
        augmented_data = json.load(f)
    
    # 只取前n_samples个虚拟样本
    virtual_samples = [d for d in augmented_data if d['type'] == 'virtual'][:n_samples]
    
    print(f"\n测试样本: {n_samples}个")
    
    # 准备任务
    tasks = []
    for i, sample in enumerate(virtual_samples):
        if 'laikago' in sample['name']:
            base_urdf = 'laikago/laikago.urdf'
            bounds = [(0.1, 50.0), (0.01, 10.0)]
        elif 'kuka' in sample['name'] or 'model' in sample['name']:
            base_urdf = 'kuka_iiwa/model.urdf'
            bounds = [(1.0, 100.0), (0.5, 20.0)]
        else:
            base_urdf = 'franka_panda/panda.urdf'
            bounds = [(50.0, 300.0), (5.0, 30.0)]
        
        tasks.append((base_urdf, sample['augmentation_params'], bounds, i))
    
    # 并行优化
    n_workers = min(4, cpu_count())
    print(f"使用{n_workers}个进程...")
    
    start_time = time.time()
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(optimize_pid_for_virtual_robot, tasks), total=len(tasks)))
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ 测试完成！耗时: {elapsed:.1f}秒 ({elapsed/n_samples:.1f}秒/样本)")
    
    successful = [r for r in results if r['success']]
    print(f"   成功率: {len(successful)}/{n_samples}")
    
    if successful:
        errors = [r['error_deg'] for r in successful]
        print(f"   平均误差: {np.mean(errors):.2f}°")
    
    # 显示几个结果
    print(f"\n样本示例:")
    for i, r in enumerate(results[:3]):
        if r['success']:
            print(f"   样本{i}: Kp={r['kp']:.4f}, Kd={r['kd']:.4f}, 误差={r['error_deg']:.2f}°")


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # 快速测试模式
        n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        quick_test(n_test)
    else:
        # 完整优化
        n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else None
        optimize_all_virtual_samples(n_workers)

