#!/usr/bin/env python3
"""
元学习PID数据增强
通过物理参数扰动生成大量虚拟机器人样本
"""

import numpy as np
import pybullet as p
import pybullet_data
import json
from pathlib import Path
from meta_pid_optimizer import RobotFeatureExtractor


class PhysicsBasedAugmentation:
    """基于物理参数的数据增强"""
    
    def __init__(self, base_urdf_path, param_ranges=None):
        """
        Args:
            base_urdf_path: 基础机器人URDF路径
            param_ranges: 参数变化范围字典
        """
        self.base_urdf_path = base_urdf_path
        
        # 默认参数变化范围（保守版：确保物理合理性）
        self.param_ranges = param_ranges or {
            'mass_scale': (0.9, 1.1),          # 质量±10%
            'length_scale': (0.95, 1.05),      # 长度±5%
            'inertia_scale': (0.85, 1.15),     # 惯性±15%
            'friction': (0.8, 1.2),            # 摩擦系数±20%
            'damping': (0.7, 1.3)              # 阻尼±30%
        }
    
    def generate_virtual_robots(self, n_samples=100):
        """
        生成虚拟机器人样本
        
        Args:
            n_samples: 生成样本数量
        
        Returns:
            virtual_robots: 虚拟机器人列表，每个包含修改后的参数
        """
        virtual_robots = []
        
        for i in range(n_samples):
            # 随机采样参数
            params = {
                'mass_scale': np.random.uniform(*self.param_ranges['mass_scale']),
                'length_scale': np.random.uniform(*self.param_ranges['length_scale']),
                'inertia_scale': np.random.uniform(*self.param_ranges['inertia_scale']),
                'friction': np.random.uniform(*self.param_ranges['friction']),
                'damping': np.random.uniform(*self.param_ranges['damping'])
            }
            
            virtual_robots.append({
                'id': f'virtual_{i:04d}',
                'base_urdf': self.base_urdf_path,
                'params': params
            })
        
        return virtual_robots
    
    def apply_params_to_robot(self, robot_id, params, client_id):
        """
        将参数应用到PyBullet机器人
        
        Args:
            robot_id: 机器人ID
            params: 参数字典
            client_id: PyBullet客户端ID
        """
        num_joints = p.getNumJoints(robot_id, physicsClientId=client_id)
        
        for j in range(num_joints):
            # 获取原始动力学参数
            dyn_info = p.getDynamicsInfo(robot_id, j, physicsClientId=client_id)
            original_mass = dyn_info[0]
            
            # 应用修改
            p.changeDynamics(
                robot_id, j,
                mass=original_mass * params['mass_scale'],
                lateralFriction=params['friction'],
                linearDamping=dyn_info[6] * params['damping'],
                angularDamping=dyn_info[7] * params['damping'],
                physicsClientId=client_id
            )


def collect_augmented_data(base_robots, n_virtual_per_base=100, output_file='augmented_data.json'):
    """
    收集增强数据：真实机器人 + 虚拟机器人
    
    Args:
        base_robots: 基础机器人列表 [(urdf_path, optimal_pid)]
        n_virtual_per_base: 每个基础机器人生成的虚拟样本数
        output_file: 输出文件路径
    """
    print("=" * 80)
    print("元学习PID数据增强")
    print("=" * 80)
    
    all_data = []
    augmentor = PhysicsBasedAugmentation(None)
    extractor = RobotFeatureExtractor()
    
    for base_urdf, base_pid in base_robots:
        print(f"\n📦 处理基础机器人: {base_urdf}")
        
        # 1. 添加真实机器人数据
        features, _ = extractor.extract_features(base_urdf)
        all_data.append({
            'name': Path(base_urdf).stem,
            'type': 'real',
            'features': features,
            'optimal_pid': base_pid
        })
        print(f"   ✅ 真实机器人: {features}")
        
        # 2. 生成虚拟样本
        augmentor.base_urdf_path = base_urdf
        virtual_robots = augmentor.generate_virtual_robots(n_virtual_per_base)
        
        print(f"   🔄 生成{len(virtual_robots)}个虚拟样本...")
        
        for i, vr in enumerate(virtual_robots):
            # TODO: 为每个虚拟机器人运行PID优化
            # 这里使用启发式规则估计PID（简化版）
            mass_ratio = vr['params']['mass_scale']
            inertia_ratio = vr['params']['inertia_scale']
            
            # 启发式：Kp ∝ inertia, Kd ∝ sqrt(inertia*mass)
            estimated_kp = base_pid['kp'] * inertia_ratio
            estimated_kd = base_pid['kd'] * np.sqrt(inertia_ratio * mass_ratio)
            estimated_ki = base_pid.get('ki', 0.0)
            
            # 修改特征
            virtual_features = features.copy()
            virtual_features['total_mass'] *= mass_ratio
            virtual_features['total_inertia'] *= inertia_ratio
            
            all_data.append({
                'name': f"{Path(base_urdf).stem}_{vr['id']}",
                'type': 'virtual',
                'features': virtual_features,
                'optimal_pid': {
                    'kp': float(estimated_kp),
                    'ki': float(estimated_ki),
                    'kd': float(estimated_kd)
                },
                'augmentation_params': vr['params']
            })
            
            if (i + 1) % 20 == 0:
                print(f"      已生成 {i+1}/{len(virtual_robots)}")
        
        print(f"   ✅ 虚拟样本生成完成")
    
    # 保存数据
    output_path = Path(__file__).parent / output_file
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\n" + "=" * 80)
    print(f"✅ 数据增强完成！")
    print(f"   真实样本: {len(base_robots)}")
    print(f"   虚拟样本: {len(all_data) - len(base_robots)}")
    print(f"   总计: {len(all_data)}")
    print(f"   保存位置: {output_path}")
    print("=" * 80)
    
    return all_data


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    # 定义基础机器人（使用真实优化后的最优PID）
    base_robots = [
        ('franka_panda/panda.urdf', {'kp': 142.53, 'ki': 1.43, 'kd': 14.25}),  # 优化误差: 2.10°
        ('laikago/laikago.urdf', {'kp': 0.8752, 'ki': 0.0, 'kd': 0.8825}),     # 优化误差: 0.07°
        ('kuka_iiwa/model.urdf', {'kp': 10.2609, 'ki': 0.0, 'kd': 3.2996})     # 优化误差: 15.47°
    ]
    
    # 生成增强数据（每个基础机器人100个虚拟样本 = 300个虚拟样本）
    augmented_data = collect_augmented_data(
        base_robots,
        n_virtual_per_base=100,
        output_file='augmented_pid_data.json'
    )
    
    print(f"\n📊 样本统计:")
    print(f"   Franka系列: {1 + 100} = 101")
    print(f"   Laikago系列: {1 + 100} = 101")
    print(f"   KUKA系列: {1 + 100} = 101")
    print(f"   总计: 303样本")
    print(f"\n🎯 下一步: 使用这些数据训练元学习PID网络")

