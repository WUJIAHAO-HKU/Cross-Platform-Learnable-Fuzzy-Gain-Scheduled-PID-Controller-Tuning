#!/usr/bin/env python3
"""
计算末端执行器误差（笛卡尔空间）
这对实际应用更有意义
"""

import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from meta_rl_combined_env import MetaRLCombinedEnv


def get_endeffector_pose(robot_id, end_effector_link_id):
    """获取末端执行器位姿"""
    link_state = p.getLinkState(robot_id, end_effector_link_id)
    position = np.array(link_state[0])  # 位置 (x, y, z)
    orientation = np.array(link_state[1])  # 四元数 (x, y, z, w)
    return position, orientation


def quaternion_to_euler(q):
    """四元数转欧拉角"""
    euler = p.getEulerFromQuaternion(q)
    return np.array(euler)


def evaluate_endeffector_error(robot_urdf, model_path=None, steps=10000):
    """评估末端执行器误差"""
    
    test_name = "纯Meta-PID" if model_path is None else "Meta-PID + RL"
    
    print(f"\n{'='*80}")
    print(f"评估: {test_name}")
    print(f"{'='*80}")
    
    # 创建环境
    env = MetaRLCombinedEnv(robot_urdf=robot_urdf, gui=False)
    
    # 获取末端执行器链接ID（最后一个可控关节）
    end_effector_link = env.controllable_joints[-1]
    
    # 加载RL模型
    model = None
    if model_path is not None:
        model = PPO.load(model_path)
        print(f"✅ RL模型加载成功")
    else:
        print(f"✅ 使用固定Meta-PID")
    
    obs, _ = env.reset()
    
    # 记录数据
    position_errors = []  # 位置误差 (m)
    orientation_errors = []  # 姿态误差 (rad)
    joint_errors_list = []  # 关节误差列表
    
    for step in range(steps):
        # 选择动作
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(2)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 获取当前关节状态
        joint_states = p.getJointStates(env.robot_id, env.controllable_joints)
        q_actual = np.array([s[0] for s in joint_states])
        q_ref = env._get_reference_trajectory()
        
        # 计算关节误差
        joint_errors = np.abs(q_ref - q_actual)
        joint_errors_list.append(joint_errors)
        
        # 获取实际末端执行器位姿
        pos_actual, ori_actual = get_endeffector_pose(env.robot_id, end_effector_link)
        
        # 设置参考关节角度以获取参考末端位姿
        for i, joint_id in enumerate(env.controllable_joints):
            p.resetJointState(env.robot_id, joint_id, q_ref[i])
        
        # 获取参考末端执行器位姿
        pos_ref, ori_ref = get_endeffector_pose(env.robot_id, end_effector_link)
        
        # 恢复实际关节角度
        for i, joint_id in enumerate(env.controllable_joints):
            p.resetJointState(env.robot_id, joint_id, q_actual[i])
        
        # 计算位置误差（欧氏距离）
        position_error = np.linalg.norm(pos_ref - pos_actual)
        position_errors.append(position_error)
        
        # 计算姿态误差（四元数误差）
        euler_ref = quaternion_to_euler(ori_ref)
        euler_actual = quaternion_to_euler(ori_actual)
        orientation_error = np.linalg.norm(euler_ref - euler_actual)
        orientation_errors.append(orientation_error)
        
        if step % 2000 == 0:
            print(f"Step {step:5d}: "
                  f"pos_err={position_error*1000:.2f}mm, "
                  f"ori_err={np.degrees(orientation_error):.2f}°, "
                  f"Kp={info['current_kp']:.2f}")
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
    
    # 统计结果
    position_errors = np.array(position_errors) * 1000  # 转换为mm
    orientation_errors = np.array(orientation_errors)
    orientation_errors_deg = np.degrees(orientation_errors)
    joint_errors_array = np.array(joint_errors_list)
    joint_errors_deg = np.degrees(joint_errors_array)
    
    results = {
        # 末端执行器误差
        'mean_position_error_mm': np.mean(position_errors),
        'max_position_error_mm': np.max(position_errors),
        'std_position_error_mm': np.std(position_errors),
        
        'mean_orientation_error_deg': np.mean(orientation_errors_deg),
        'max_orientation_error_deg': np.max(orientation_errors_deg),
        'std_orientation_error_deg': np.std(orientation_errors_deg),
        
        # 关节空间误差
        'mean_joint_error_deg': np.mean(joint_errors_deg),
        'max_joint_error_deg': np.max(joint_errors_deg),
        'per_joint_mean_error_deg': np.mean(joint_errors_deg, axis=0),
        
        # L2范数误差（原始指标）
        'mean_l2_norm_error_deg': np.mean(np.linalg.norm(joint_errors_deg, axis=1)),
    }
    
    print(f"\n📊 {test_name} 性能评估:")
    print(f"\n【末端执行器误差】（实际应用关注）")
    print(f"   位置误差:")
    print(f"     平均: {results['mean_position_error_mm']:.2f} mm")
    print(f"     最大: {results['max_position_error_mm']:.2f} mm")
    print(f"     标准差: {results['std_position_error_mm']:.2f} mm")
    print(f"   姿态误差:")
    print(f"     平均: {results['mean_orientation_error_deg']:.2f}°")
    print(f"     最大: {results['max_orientation_error_deg']:.2f}°")
    print(f"     标准差: {results['std_orientation_error_deg']:.2f}°")
    
    print(f"\n【关节空间误差】（控制性能指标）")
    print(f"   平均关节误差（MAE）: {results['mean_joint_error_deg']:.2f}°")
    print(f"   最大关节误差: {results['max_joint_error_deg']:.2f}°")
    print(f"   L2范数误差: {results['mean_l2_norm_error_deg']:.2f}° (原始报告值)")
    
    print(f"\n   各关节平均误差:")
    for i, err in enumerate(results['per_joint_mean_error_deg']):
        print(f"      关节{i+1}: {err:.2f}°")
    
    return results


def main():
    robot_urdf = 'franka_panda/panda.urdf'
    model_path = 'logs/meta_rl_panda/best_model/best_model'
    
    print("="*80)
    print("末端执行器误差评估（笛卡尔空间）")
    print("="*80)
    print(f"机器人: {robot_urdf}")
    print(f"测试步数: 10000")
    print()
    
    # 评估1: 纯Meta-PID
    pure_results = evaluate_endeffector_error(
        robot_urdf, 
        model_path=None, 
        steps=10000
    )
    
    # 评估2: Meta-PID + RL
    rl_results = evaluate_endeffector_error(
        robot_urdf, 
        model_path=model_path, 
        steps=10000
    )
    
    # 性能对比
    print("\n" + "="*80)
    print("性能对比总结")
    print("="*80)
    
    # 末端执行器误差改善
    pos_improvement = (pure_results['mean_position_error_mm'] - rl_results['mean_position_error_mm']) / pure_results['mean_position_error_mm'] * 100
    ori_improvement = (pure_results['mean_orientation_error_deg'] - rl_results['mean_orientation_error_deg']) / pure_results['mean_orientation_error_deg'] * 100
    
    # 关节误差改善
    joint_improvement = (pure_results['mean_joint_error_deg'] - rl_results['mean_joint_error_deg']) / pure_results['mean_joint_error_deg'] * 100
    
    print(f"\n【末端执行器误差改善】⭐⭐⭐⭐⭐")
    print(f"  位置误差: {pure_results['mean_position_error_mm']:.2f}mm → {rl_results['mean_position_error_mm']:.2f}mm "
          f"({pos_improvement:+.2f}%)")
    print(f"  姿态误差: {pure_results['mean_orientation_error_deg']:.2f}° → {rl_results['mean_orientation_error_deg']:.2f}° "
          f"({ori_improvement:+.2f}%)")
    
    print(f"\n【关节空间误差改善】")
    print(f"  平均关节误差(MAE): {pure_results['mean_joint_error_deg']:.2f}° → {rl_results['mean_joint_error_deg']:.2f}° "
          f"({joint_improvement:+.2f}%)")
    print(f"  L2范数误差: {pure_results['mean_l2_norm_error_deg']:.2f}° → {rl_results['mean_l2_norm_error_deg']:.2f}°")
    
    print(f"\n💡 建议论文中报告:")
    print(f"   1. 末端执行器位置误差: {rl_results['mean_position_error_mm']:.2f}mm (最直观)")
    print(f"   2. 平均关节误差(MAE): {rl_results['mean_joint_error_deg']:.2f}° (更合理)")
    print(f"   3. L2范数误差: {rl_results['mean_l2_norm_error_deg']:.2f}° (作为补充)")
    
    print("\n" + "="*80)
    print("✅ 评估完成！")
    print("="*80)


if __name__ == '__main__':
    main()

