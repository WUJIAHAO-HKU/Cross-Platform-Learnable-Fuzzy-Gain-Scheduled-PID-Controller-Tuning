#!/usr/bin/env python3
"""
从PyBullet仿真环境中捕获机器人截图
用于论文的实验平台展示
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os

def capture_franka_panda_screenshot(output_file='franka_panda_simulation.png'):
    """
    捕获Franka Panda机械臂在仿真环境中的截图
    """
    print(f"\n{'='*80}")
    print("捕获Franka Panda仿真截图")
    print('='*80)
    
    # 连接PyBullet (GUI模式)
    physicsClient = p.connect(p.DIRECT)  # 使用DIRECT模式不显示窗口
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 设置仿真环境
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5]
    )
    
    # 加载地面
    planeId = p.loadURDF("plane.urdf")
    
    # 加载Franka Panda机器人 (PyBullet自带)
    urdf_path = "franka_panda/panda.urdf"
    robotId = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
    print(f"✓ 加载机器人: {urdf_path}")
    
    # 设置机器人到一个好看的姿态
    num_joints = p.getNumJoints(robotId)
    print(f"✓ 机器人关节数: {num_joints}")
    
    # 设置关节位置（展示姿态）
    joint_positions = [0.0, -0.3, 0.0, -2.0, 0.0, 1.5, 0.785, 0.04, 0.04]
    for i in range(min(len(joint_positions), num_joints)):
        p.resetJointState(robotId, i, joint_positions[i])
    
    # 运行仿真几步使其稳定
    for _ in range(100):
        p.stepSimulation()
    
    # 设置相机参数
    width, height = 1440, 1080  # 减小尺寸（原来1920x1440）
    
    # 计算视图矩阵（画面更近，聚焦机械臂中心，增加俯视角度以减少地面白色背景）
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.5],  # 聚焦于机械臂中心
        distance=1.1,  # 距离更近，让机器人占据更多画面
        yaw=50,
        pitch=-35,  # 增加俯视角度，减少地面出现
        roll=0,
        upAxisIndex=2
    )
    
    # 计算投影矩阵
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width/height,
        nearVal=0.1,
        farVal=100.0
    )
    
    # 获取图像
    print("✓ 渲染图像...")
    img_arr = p.getCameraImage(
        width, height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    # 保存图像
    import matplotlib.pyplot as plt
    rgb_array = np.array(img_arr[2]).reshape(height, width, 4)
    rgb_array = rgb_array[:, :, :3]  # 只保留RGB通道
    
    plt.figure(figsize=(9, 6.75), dpi=150)  # 减小尺寸（原来12x9）
    plt.imshow(rgb_array)
    plt.axis('off')
    # 不添加标题，保持截图简洁
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none', pad_inches=0)
    plt.close()
    
    print(f"✓ 截图保存: {output_file}")
    
    # 断开连接
    p.disconnect()
    return True

def capture_laikago_quadruped_screenshot(output_file='laikago_quadruped_simulation.png'):
    """
    捕获Laikago四足机器人在仿真环境中的截图
    """
    print(f"\n{'='*80}")
    print("捕获Laikago四足机器人仿真截图")
    print('='*80)
    
    # 连接PyBullet
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 设置仿真环境
    p.setGravity(0, 0, -9.81)
    
    # 加载地面
    planeId = p.loadURDF("plane.urdf")
    
    # 加载Laikago机器人 (PyBullet自带)
    # 设置正确的朝向：水平放置，面向X轴正方向
    urdf_path = "laikago/laikago.urdf"
    start_pos = [0, 0, 0.48]  # 适当的初始高度
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])  # 水平放置
    
    robotId = p.loadURDF(urdf_path, start_pos, start_orientation, useFixedBase=False)
    print(f"✓ 加载机器人: {urdf_path}")
    
    num_joints = p.getNumJoints(robotId)
    print(f"✓ 机器人关节数: {num_joints}")
    
    # 设置机器人到站立姿态
    # Laikago的关节配置：每条腿3个关节（abduction, hip, knee）
    # 关节顺序：FR(0,1,2), FL(3,4,5), RR(6,7,8), RL(9,10,11)
    # 这是经过验证的稳定站立姿态！
    standing_pose = {
        # Front Right Leg
        0: 0.0,    # abduction (侧摆)
        1: 1.0,    # hip (髋关节) - 深蹲姿态
        2: -2.0,   # knee (膝关节)
        # Front Left Leg
        3: 0.0,
        4: 1.0,
        5: -2.0,
        # Rear Right Leg
        6: 0.0,
        7: 1.0,
        8: -2.0,
        # Rear Left Leg
        9: 0.0,
        10: 1.0,
        11: -2.0,
    }
    
    for joint_id, angle in standing_pose.items():
        if joint_id < num_joints:
            p.resetJointState(robotId, joint_id, angle)
    
    # 运行仿真使机器人稳定下来（使用PD控制）
    print("✓ 稳定机器人姿态...")
    for step in range(3000):
        # 对每个关节应用PD控制
        for joint_id in range(num_joints):
            p.setJointMotorControl2(
                robotId, joint_id,
                p.POSITION_CONTROL,
                targetPosition=standing_pose.get(joint_id, 0),
                force=500,  # 足够的力矩
                positionGain=0.5,  # Kp
                velocityGain=0.1   # Kd
            )
        p.stepSimulation()
        
        # 每1000步检查一次高度
        if step % 1000 == 0 and step > 0:
            base_pos, _ = p.getBasePositionAndOrientation(robotId)
            print(f"  步数 {step}: 高度 = {base_pos[2]:.3f}m")
    
    print("✓ 机器人已稳定")
    
    # 设置相机参数
    width, height = 1440, 1080  # 减小尺寸（原来1920x1440）
    
    # 获取机器人实际位置
    base_pos, base_orn = p.getBasePositionAndOrientation(robotId)
    actual_x, actual_y, actual_height = base_pos
    print(f"✓ 机器人位置: x={actual_x:.3f}, y={actual_y:.3f}, z={actual_height:.3f}m")
    
    # 计算视图矩阵（从侧前方观察，聚焦于机器人中心，增加俯视角度以减少地面白色背景）
    # 画面更近，并确保机器人居中
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[actual_x, actual_y, actual_height * 0.5],  # 聚焦于机器人实际中心
        distance=1.0,  # 距离更近，让机器人占据更多画面
        yaw=45,  # 从右前方45度观察
        pitch=-30,  # 增加俯视角度，减少地面出现
        roll=0,
        upAxisIndex=2
    )
    
    # 计算投影矩阵
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width/height,
        nearVal=0.1,
        farVal=100.0
    )
    
    # 获取图像
    print("✓ 渲染图像...")
    img_arr = p.getCameraImage(
        width, height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    # 保存图像
    import matplotlib.pyplot as plt
    rgb_array = np.array(img_arr[2]).reshape(height, width, 4)
    rgb_array = rgb_array[:, :, :3]
    
    plt.figure(figsize=(9, 6.75), dpi=150)  # 减小尺寸（原来12x9）
    plt.imshow(rgb_array)
    plt.axis('off')
    # 不添加标题，保持截图简洁
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none', pad_inches=0)
    plt.close()
    
    print(f"✓ 截图保存: {output_file}")
    
    # 断开连接
    p.disconnect()
    return True

def main():
    print("\n" + "="*80)
    print("PyBullet仿真环境机器人截图工具")
    print("="*80)
    
    # 捕获Franka Panda截图
    success1 = capture_franka_panda_screenshot('franka_panda_simulation.png')
    
    # 捕获Laikago截图
    success2 = capture_laikago_quadruped_screenshot('laikago_quadruped_simulation.png')
    
    print("\n" + "="*80)
    if success1 and success2:
        print("✅ 所有仿真截图生成完成！")
        print("="*80)
        print("\n生成的文件:")
        print("  • franka_panda_simulation.png     - Franka Panda机械臂仿真截图")
        print("  • laikago_quadruped_simulation.png - Laikago四足机器人仿真截图")
        print("\n图片规格:")
        print("  • 分辨率: 1920×1440 (高清)")
        print("  • DPI: 300 (印刷质量)")
        print("  • 格式: PNG (无损压缩)")
        print("\n用途:")
        print("  • 论文实验平台展示")
        print("  • 可插入Section 4 (Experimental Setup)")
        print("  • 展示真实仿真环境")
    else:
        print("⚠️ 部分截图生成失败")
        print("="*80)
        if not success1:
            print("  • Franka Panda截图失败 - 请检查URDF路径")
        if not success2:
            print("  • Laikago截图失败 - 请检查URDF路径")
    print()

if __name__ == '__main__':
    main()

