#!/usr/bin/env python3
"""
找到Laikago的稳定站立姿态
通过让机器人自然落下并调整关节，找到一个稳定的配置
"""

import pybullet as p
import pybullet_data
import time
import numpy as np


def find_stable_standing_pose():
    """
    方法：尝试多组关节角度，找到稳定的站立姿态
    """
    # 连接PyBullet
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(0.001)
    
    # 加载环境
    plane_id = p.loadURDF("plane.urdf")
    
    # 测试多组姿态
    test_poses = [
        # 姿态1: 腿稍微弯曲
        {
            'name': '轻微弯曲',
            'angles': np.array([
                0.0, 0.67, -1.3,  # FR
                0.0, 0.67, -1.3,  # FL
                0.0, 0.67, -1.3,  # RR
                0.0, 0.67, -1.3   # RL
            ])
        },
        # 姿态2: 腿更直
        {
            'name': '较直姿态',
            'angles': np.array([
                0.0, 0.5, -1.0,
                0.0, 0.5, -1.0,
                0.0, 0.5, -1.0,
                0.0, 0.5, -1.0
            ])
        },
        # 姿态3: 腿更弯
        {
            'name': '深蹲姿态',
            'angles': np.array([
                0.0, 1.0, -2.0,
                0.0, 1.0, -2.0,
                0.0, 1.0, -2.0,
                0.0, 1.0, -2.0
            ])
        },
        # 姿态4: 参考Unitree A1（类似Laikago）
        {
            'name': 'Unitree风格',
            'angles': np.array([
                0.0, 0.8, -1.6,
                0.0, 0.8, -1.6,
                0.0, 0.8, -1.6,
                0.0, 0.8, -1.6
            ])
        },
    ]
    
    results = []
    
    for pose_config in test_poses:
        print("\n" + "=" * 80)
        print(f"测试姿态: {pose_config['name']}")
        print(f"关节角度: {pose_config['angles'][:3]}")
        print("=" * 80)
        
        # 加载机器人
        robot_id = p.loadURDF(
            "laikago/laikago.urdf",
            [0, 0, 0.5],
            p.getQuaternionFromEuler([0, 0, 0]),
            flags=p.URDF_USE_SELF_COLLISION
        )
        
        # 获取可控关节
        num_joints = p.getNumJoints(robot_id)
        motor_ids = []
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                motor_ids.append(i)
        
        # 设置初始姿态
        for i, motor_id in enumerate(motor_ids):
            p.resetJointState(robot_id, motor_id, pose_config['angles'][i])
        
        # 让机器人稳定下来（使用强PD控制）
        for step in range(3000):
            for i, motor_id in enumerate(motor_ids):
                p.setJointMotorControl2(
                    robot_id,
                    motor_id,
                    p.POSITION_CONTROL,
                    targetPosition=pose_config['angles'][i],
                    force=100,
                    positionGain=0.5,  # 使用较高增益
                    velocityGain=0.1
                )
            p.stepSimulation()
            
            # 每秒检查一次
            if step % 1000 == 0 and step > 0:
                base_pos, _ = p.getBasePositionAndOrientation(robot_id)
                base_vel, _ = p.getBaseVelocity(robot_id)
                height = base_pos[2]
                speed = np.linalg.norm(base_vel)
                print(f"   t={step/1000:.1f}s: 高度={height:.3f}m, 速度={speed:.3f}m/s")
        
        # 最终评估
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        base_vel, _ = p.getBaseVelocity(robot_id)
        height = base_pos[2]
        speed = np.linalg.norm(base_vel)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)
        
        # 读取实际关节角度
        joint_states = p.getJointStates(robot_id, motor_ids)
        actual_angles = np.array([s[0] for s in joint_states])
        
        # 评分
        score = 0
        if 0.2 < height < 0.4:
            score += 50
        if speed < 0.01:
            score += 30
        if abs(roll) < 0.1 and abs(pitch) < 0.1:
            score += 20
        
        result = {
            'name': pose_config['name'],
            'target_angles': pose_config['angles'],
            'actual_angles': actual_angles,
            'height': height,
            'speed': speed,
            'roll': roll,
            'pitch': pitch,
            'score': score
        }
        results.append(result)
        
        print(f"\n📊 评估:")
        print(f"   最终高度: {height:.3f}m")
        print(f"   最终速度: {speed:.3f}m/s")
        print(f"   姿态(roll/pitch): {np.degrees(roll):.1f}° / {np.degrees(pitch):.1f}°")
        print(f"   评分: {score}/100")
        print(f"   实际关节角度: [{actual_angles[0]:.2f}, {actual_angles[1]:.2f}, {actual_angles[2]:.2f}]")
        
        # 删除机器人
        p.removeBody(robot_id)
        time.sleep(0.5)
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 所有姿态评估结果")
    print("=" * 80)
    
    # 按评分排序
    results.sort(key=lambda x: x['score'], reverse=True)
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['name']} (评分: {result['score']}/100)")
        print(f"   目标角度: [{result['target_angles'][0]:.2f}, {result['target_angles'][1]:.2f}, {result['target_angles'][2]:.2f}]")
        print(f"   实际角度: [{result['actual_angles'][0]:.2f}, {result['actual_angles'][1]:.2f}, {result['actual_angles'][2]:.2f}]")
        print(f"   高度: {result['height']:.3f}m, 速度: {result['speed']:.4f}m/s")
    
    # 推荐最佳姿态
    best = results[0]
    print("\n" + "=" * 80)
    print("✅ 推荐使用的站立姿态:")
    print("=" * 80)
    print(f"名称: {best['name']}")
    print(f"关节角度: {best['actual_angles'][:3]}")
    print(f"\nPython代码:")
    print(f"INIT_MOTOR_ANGLES = np.array([")
    for i in range(0, 12, 3):
        angles = best['actual_angles'][i:i+3]
        leg_name = ['FR', 'FL', 'RR', 'RL'][i//3]
        print(f"    {angles[0]:.4f}, {angles[1]:.4f}, {angles[2]:.4f},  # {leg_name}")
    print("])")
    
    p.disconnect()
    
    return best['actual_angles']


if __name__ == '__main__':
    print("=" * 80)
    print("Laikago稳定姿态搜索")
    print("=" * 80)
    
    best_pose = find_stable_standing_pose()
    
    print("\n\n🎯 下一步:")
    print("  1. 将上面的INIT_MOTOR_ANGLES复制到test_laikago_final.py")
    print("  2. 重新运行站立测试")
    print("  3. 如果仍不稳定，尝试调整motor_kp和motor_kd")

