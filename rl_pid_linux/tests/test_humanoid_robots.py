"""
测试PyBullet中可用的人形机器人
"""

import pybullet as p
import pybullet_data
import time
import numpy as np

# 可能的人形机器人URDF
humanoid_robots = [
    # Atlas系列
    'atlas/atlas_v4_with_multisense.urdf',
    'atlas/atlas.urdf',
    
    # NAO机器人
    'nao/nao.urdf',
    
    # Darwin-OP
    'darwin/darwin.urdf',
    
    # Cassie（双足机器人）
    'cassie/cassie.urdf',
    
    # Hubo
    'hubo/hubo_description/urdf/hubo.urdf',
    
    # Laikago（四足，但可参考）
    'laikago/laikago.urdf',
    
    # Humanoid（简单人形）
    'humanoid/humanoid.urdf',
    'humanoid.urdf',
    
    # MIT Cheetah
    'mini_cheetah/mini_cheetah.urdf'
]

print("=" * 80)
print("测试PyBullet中的人形机器人模型")
print("=" * 80)

available_robots = []

for robot_name in humanoid_robots:
    print(f"\n尝试加载: {robot_name}")
    
    # 连接PyBullet
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    
    try:
        # 尝试加载URDF
        robot_id = p.loadURDF(robot_name, [0, 0, 1.0], physicsClientId=client)
        
        # 获取关节信息
        num_joints = p.getNumJoints(robot_id, physicsClientId=client)
        
        # 统计可控关节
        controllable_joints = []
        joint_info = []
        
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i, physicsClientId=client)
            joint_type = info[2]
            joint_name = info[1].decode('utf-8')
            
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                controllable_joints.append(i)
                joint_info.append({
                    'id': i,
                    'name': joint_name,
                    'type': 'Revolute' if joint_type == p.JOINT_REVOLUTE else 'Prismatic'
                })
        
        # 获取机器人质量
        total_mass = 0
        for i in range(-1, num_joints):
            dynamics = p.getDynamicsInfo(robot_id, i, physicsClientId=client)
            total_mass += dynamics[0]
        
        print(f"  ✅ 成功加载！")
        print(f"     总关节数: {num_joints}")
        print(f"     可控关节数: {len(controllable_joints)}")
        print(f"     总质量: {total_mass:.2f} kg")
        print(f"     关节名称: {[j['name'] for j in joint_info[:5]]}...")  # 显示前5个
        
        available_robots.append({
            'name': robot_name,
            'num_joints': num_joints,
            'dof': len(controllable_joints),
            'mass': total_mass,
            'joints': joint_info
        })
        
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
    
    finally:
        p.disconnect(physicsClientId=client)

print("\n" + "=" * 80)
print(f"可用的人形机器人: {len(available_robots)}/{len(humanoid_robots)}")
print("=" * 80)

if available_robots:
    print("\n详细信息：\n")
    
    for i, robot in enumerate(available_robots, 1):
        print(f"{i}. {robot['name']}")
        print(f"   自由度: {robot['dof']}")
        print(f"   质量: {robot['mass']:.2f} kg")
        print(f"   关节数: {robot['num_joints']}")
        
        # 分类关节（简单分类）
        arm_joints = [j for j in robot['joints'] if any(kw in j['name'].lower() 
                     for kw in ['arm', 'shoulder', 'elbow', 'wrist', 'hand'])]
        leg_joints = [j for j in robot['joints'] if any(kw in j['name'].lower() 
                     for kw in ['leg', 'hip', 'knee', 'ankle', 'foot'])]
        
        print(f"   上肢关节: ~{len(arm_joints)}")
        print(f"   下肢关节: ~{len(leg_joints)}")
        print()
    
    # 保存结果
    import json
    with open('available_humanoid_robots.json', 'w') as f:
        json.dump(available_robots, f, indent=2)
    
    print(f"✅ 详细信息已保存: available_humanoid_robots.json")
    
    # 推荐
    print("\n" + "=" * 80)
    print("推荐用于研究的机器人：")
    print("=" * 80)
    
    # 按自由度排序
    sorted_robots = sorted(available_robots, key=lambda x: x['dof'], reverse=True)
    
    for robot in sorted_robots[:3]:
        print(f"\n📌 {robot['name']}")
        print(f"   优势: {robot['dof']}自由度，适合复杂控制研究")
        if robot['dof'] > 20:
            print(f"   建议: 可用于全身控制研究（步行+操作）")
        elif robot['dof'] > 10:
            print(f"   建议: 可用于上肢操作研究")
        else:
            print(f"   建议: 适合作为baseline对比")

else:
    print("\n❌ 未找到可用的人形机器人URDF")
    print("   建议：下载开源人形机器人模型")
    print("   资源：")
    print("   - https://github.com/robot-descriptions/robot_descriptions.py")
    print("   - https://github.com/unitreerobotics/unitree_mujoco")

print("\n" + "=" * 80)
print("下一步: 使用找到的机器人进行元学习PID测试")
print("=" * 80)

