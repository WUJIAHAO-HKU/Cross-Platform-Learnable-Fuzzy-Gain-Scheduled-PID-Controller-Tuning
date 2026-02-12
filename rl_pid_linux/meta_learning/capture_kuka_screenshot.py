#!/usr/bin/env python3
"""
生成KUKA LBR iiwa机器人的PyBullet仿真截图
用于论文展示
"""

import pybullet as p
import pybullet_data
import os

def capture_kuka_screenshot():
    """捕获KUKA LBR iiwa机器人的仿真截图"""
    
    # 连接PyBullet (使用DIRECT模式，不显示GUI窗口)
    p.connect(p.DIRECT)
    
    # 设置额外的数据路径
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 设置重力
    p.setGravity(0, 0, -9.81)
    
    # 加载地面
    planeId = p.loadURDF("plane.urdf")
    
    # 加载KUKA LBR iiwa机器人
    urdf_path = "kuka_iiwa/model.urdf"
    start_pos = [0, 0, 0]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    
    robotId = p.loadURDF(urdf_path, start_pos, start_orientation, useFixedBase=True)
    
    print(f"✅ KUKA iiwa机器人加载成功！")
    print(f"机器人ID: {robotId}")
    
    # 获取关节信息
    num_joints = p.getNumJoints(robotId)
    print(f"总关节数: {num_joints}")
    
    # 设置一个美观的姿态（类似人类手臂伸展的姿态）
    # KUKA iiwa有7个关节，我们设置一个典型的工作姿态
    joint_positions = {
        0: 0.0,      # 基座旋转
        1: 0.5,      # 肩部俯仰
        2: 0.0,      # 肩部旋转
        3: -1.2,     # 肘部弯曲
        4: 0.0,      # 前臂旋转
        5: 1.0,      # 腕部俯仰
        6: 0.0       # 腕部旋转
    }
    
    # 设置关节位置
    for joint_id in range(min(7, num_joints)):
        if joint_id in joint_positions:
            p.resetJointState(robotId, joint_id, joint_positions[joint_id])
    
    # 运行一小段时间让机器人稳定
    for _ in range(100):
        p.stepSimulation()
    
    # 获取机器人的实际位置
    base_pos, base_orn = p.getBasePositionAndOrientation(robotId)
    actual_x, actual_y, actual_height = base_pos
    
    print(f"机器人基座位置: x={actual_x:.3f}, y={actual_y:.3f}, z={actual_height:.3f}")
    
    # 设置相机视角（从斜前方观察，确保机器人居中）
    # 增加俯视角度并拉近距离，减少白色背景
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[actual_x+0.3, actual_y, actual_height + 0.4],  # 提高聚焦点，使机器人更居中
        distance=1.3,      # 距离调整为1.5米
        yaw=50,            # 从右前方50度观察
        pitch=-35,         # 增加俯视角度，减少地面出现
        roll=0,
        upAxisIndex=2
    )
    
    # 设置投影矩阵（减小图片尺寸）
    width = 1440  # 减小尺寸
    height = 1080  # 减小尺寸
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width/height,  # 1.333...（与Franka和Laikago一致）
        nearVal=0.1,
        farVal=100.0
    )
    
    # 获取相机图像
    img = p.getCameraImage(
        width, height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    # 保存图像（使用matplotlib，与Franka和Laikago一致）
    import numpy as np
    import matplotlib.pyplot as plt
    
    rgb_array = np.array(img[2], dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (height, width, 4))[:, :, :3]
    
    plt.figure(figsize=(9, 6.75), dpi=150)  # 减小图片尺寸（原来12x9）
    plt.imshow(rgb_array)
    plt.axis('off')
    # 不添加标题，保持截图简洁（与Franka和Laikago一致）
    plt.tight_layout()
    
    output_path = "kuka_iiwa_simulation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none', pad_inches=0)
    plt.close()
    
    print(f"✅ 截图已保存: {output_path}")
    print(f"   尺寸: {width}x{height} 像素")
    print(f"   格式: PNG")
    
    # 断开连接
    p.disconnect()
    
    return output_path

if __name__ == "__main__":
    print("="*70)
    print("KUKA LBR iiwa 仿真截图生成")
    print("="*70)
    
    try:
        output_file = capture_kuka_screenshot()
        print("\n" + "="*70)
        print("✅ 截图生成成功！")
        print(f"输出文件: {output_file}")
        print("="*70)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

