"""
测试下载的人形/四足机器人模型
"""

import pybullet as p
import pybullet_data
import time
import os
from pathlib import Path
import json

def test_robot(urdf_path, robot_name):
    """测试单个机器人模型"""
    print(f"\n{'=' * 80}")
    print(f"测试: {robot_name}")
    print(f"路径: {urdf_path}")
    print('=' * 80)
    
    if not Path(urdf_path).exists():
        print(f"  ❌ URDF文件不存在")
        return None
    
    # 连接PyBullet
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    
    try:
        # 加载机器人
        robot_id = p.loadURDF(str(urdf_path), [0, 0, 1.0], physicsClientId=client)
        
        # 获取关节信息
        num_joints = p.getNumJoints(robot_id, physicsClientId=client)
        
        controllable_joints = []
        joint_names = []
        
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i, physicsClientId=client)
            joint_type = info[2]
            joint_name = info[1].decode('utf-8')
            
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                controllable_joints.append(i)
                joint_names.append(joint_name)
        
        # 获取总质量
        total_mass = 0
        for i in range(-1, num_joints):
            dynamics = p.getDynamicsInfo(robot_id, i, physicsClientId=client)
            total_mass += dynamics[0]
        
        # 分析关节分布
        arm_joints = [j for j in joint_names if any(kw in j.lower() 
                     for kw in ['arm', 'shoulder', 'elbow', 'wrist', 'hand', 'finger'])]
        leg_joints = [j for j in joint_names if any(kw in j.lower() 
                     for kw in ['leg', 'hip', 'knee', 'ankle', 'foot', 'toe'])]
        head_joints = [j for j in joint_names if any(kw in j.lower() 
                      for kw in ['head', 'neck'])]
        torso_joints = [j for j in joint_names if any(kw in j.lower() 
                       for kw in ['torso', 'waist', 'chest', 'spine'])]
        
        print(f"  ✅ 成功加载！")
        print(f"\n  📊 基本信息:")
        print(f"     总关节数: {num_joints}")
        print(f"     可控关节数: {len(controllable_joints)} (DOF)")
        print(f"     总质量: {total_mass:.2f} kg")
        
        print(f"\n  🦾 关节分布:")
        print(f"     上肢关节: {len(arm_joints)}")
        print(f"     下肢关节: {len(leg_joints)}")
        print(f"     头部关节: {len(head_joints)}")
        print(f"     躯干关节: {len(torso_joints)}")
        
        if arm_joints:
            print(f"\n  👋 上肢关节: {arm_joints[:5]}...")
        if leg_joints:
            print(f"  🦿 下肢关节: {leg_joints[:5]}...")
        
        # 判断机器人类型
        if len(arm_joints) > 4 and len(leg_joints) > 4:
            robot_type = "人形机器人（双臂双足）"
            suitability = "⭐⭐⭐⭐⭐ 非常适合全身控制研究"
        elif len(leg_joints) >= 8:
            robot_type = "四足机器人"
            suitability = "⭐⭐⭐⭐ 适合步态和地形适应研究"
        elif len(arm_joints) > 4:
            robot_type = "上肢机器人"
            suitability = "⭐⭐⭐ 适合操作任务研究"
        else:
            robot_type = "简化机器人"
            suitability = "⭐⭐ 适合作为baseline"
        
        print(f"\n  🤖 机器人类型: {robot_type}")
        print(f"  📝 研究适用性: {suitability}")
        
        result = {
            'name': robot_name,
            'path': str(urdf_path),
            'dof': len(controllable_joints),
            'total_joints': num_joints,
            'mass': total_mass,
            'type': robot_type,
            'arm_joints': len(arm_joints),
            'leg_joints': len(leg_joints),
            'joint_names': joint_names
        }
        
        return result
        
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return None
    
    finally:
        p.disconnect(physicsClientId=client)


def main():
    """主函数"""
    print("=" * 80)
    print("测试下载的人形/四足机器人模型")
    print("=" * 80)
    
    # 定义要测试的机器人
    robots_to_test = []
    
    # 1. Unitree机器人
    unitree_base = Path("robots/unitree_mujoco")
    if unitree_base.exists():
        # H1人形机器人
        h1_paths = list(unitree_base.glob("**/h1.urdf"))
        for path in h1_paths:
            robots_to_test.append((path, f"Unitree H1 ({path.parent.name})"))
        
        # G1人形机器人
        g1_paths = list(unitree_base.glob("**/g1.urdf"))
        for path in g1_paths:
            robots_to_test.append((path, f"Unitree G1 ({path.parent.name})"))
        
        # Go1四足机器人
        go1_paths = list(unitree_base.glob("**/go1.urdf"))
        for path in go1_paths:
            robots_to_test.append((path, f"Unitree Go1 ({path.parent.name})"))
    
    # 2. iCub人形机器人
    icub_base = Path("robots/icub-models")
    if icub_base.exists():
        icub_paths = list(icub_base.glob("**/model.urdf"))
        for path in icub_paths:
            robots_to_test.append((path, f"iCub ({path.parent.name})"))
    
    # 3. Robot Descriptions
    robot_desc_base = Path("robots/robot_descriptions.py")
    if robot_desc_base.exists():
        # 查找所有URDF
        urdf_files = list(robot_desc_base.glob("**/*.urdf"))
        for path in urdf_files:
            if any(kw in path.name.lower() for kw in ['humanoid', 'atlas', 'nao', 'talos']):
                robots_to_test.append((path, f"Robot Descriptions - {path.stem}"))
    
    if not robots_to_test:
        print("\n❌ 未找到下载的机器人模型")
        print("\n请先运行:")
        print("  chmod +x download_humanoid_models.sh")
        print("  ./download_humanoid_models.sh")
        return
    
    print(f"\n找到 {len(robots_to_test)} 个机器人模型\n")
    
    # 测试所有机器人
    successful_robots = []
    
    for urdf_path, robot_name in robots_to_test:
        result = test_robot(urdf_path, robot_name)
        if result:
            successful_robots.append(result)
    
    # 保存结果
    if successful_robots:
        output_file = 'downloaded_robots_info.json'
        with open(output_file, 'w') as f:
            json.dump(successful_robots, f, indent=2)
        
        print("\n" + "=" * 80)
        print(f"✅ 成功加载 {len(successful_robots)}/{len(robots_to_test)} 个机器人")
        print("=" * 80)
        
        # 按DOF排序并推荐
        sorted_robots = sorted(successful_robots, key=lambda x: x['dof'], reverse=True)
        
        print("\n🏆 推荐用于研究的机器人（按DOF排序）：\n")
        
        for i, robot in enumerate(sorted_robots[:5], 1):
            print(f"{i}. {robot['name']}")
            print(f"   DOF: {robot['dof']}, 质量: {robot['mass']:.2f} kg")
            print(f"   类型: {robot['type']}")
            print(f"   上肢: {robot['arm_joints']} | 下肢: {robot['leg_joints']}")
            
            # 研究建议
            if robot['dof'] > 25:
                print(f"   💡 建议: 全身控制研究（步行+操作）")
            elif robot['dof'] > 12:
                print(f"   💡 建议: 上肢操作或步态控制")
            else:
                print(f"   💡 建议: 基础控制研究")
            print()
        
        print(f"详细信息已保存: {output_file}")
        
        # 生成使用建议
        print("\n" + "=" * 80)
        print("下一步建议:")
        print("=" * 80)
        
        if any(r['dof'] > 20 for r in successful_robots):
            top_robot = sorted_robots[0]
            print(f"\n推荐使用: {top_robot['name']}")
            print(f"  - 高自由度（{top_robot['dof']} DOF）适合复杂控制研究")
            print(f"  - 可以测试元学习PID在人形机器人上的效果")
            print(f"\n快速开始:")
            print(f"  1. 修改 meta_learning/meta_pid_optimizer.py 中的测试URDF路径")
            print(f"  2. 运行: python meta_learning/meta_pid_optimizer.py")
        
    else:
        print("\n❌ 所有机器人加载都失败了")
        print("   可能原因: URDF格式问题或依赖缺失")


if __name__ == '__main__':
    main()

