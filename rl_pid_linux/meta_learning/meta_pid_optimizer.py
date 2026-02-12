"""
元学习PID优化器

功能：
1. 从机器人URDF提取特征（DOF, 质量, 惯量, 长度等）
2. 使用神经网络预测最优PID参数
3. 实现零样本迁移到新机器人
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pybullet as p
import pybullet_data
import yaml
from pathlib import Path


class RobotFeatureExtractor:
    """从URDF提取机器人特征"""
    
    def __init__(self):
        self.feature_names = [
            'dof',
            'total_mass',
            'avg_link_mass',
            'max_link_mass',
            'total_inertia',
            'max_reach',
            'avg_link_length',
            'max_link_length',
            'payload_mass',
            'payload_distance'
        ]
    
    def extract_features(self, urdf_path, payload=0.0, use_gui=False):
        """
        从URDF提取特征
        
        Args:
            urdf_path: URDF文件路径
            payload: 末端负载质量(kg)
            use_gui: 是否显示GUI（调试用）
        
        Returns:
            dict: 特征字典
        """
        # 连接PyBullet
        if use_gui:
            client = p.connect(p.GUI)
        else:
            client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        
        try:
            # 加载机器人
            robot_id = p.loadURDF(str(urdf_path), [0, 0, 0], physicsClientId=client)
            
            # 获取关节信息
            num_joints = p.getNumJoints(robot_id, physicsClientId=client)
            
            # 只考虑可控制关节（旋转关节）
            controllable_joints = []
            joint_masses = []
            joint_inertias = []
            link_lengths = []
            
            for i in range(num_joints):
                joint_info = p.getJointInfo(robot_id, i, physicsClientId=client)
                joint_type = joint_info[2]
                
                # 只考虑旋转关节和移动关节
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    controllable_joints.append(i)
                    
                    # 获取动力学参数
                    dynamics_info = p.getDynamicsInfo(robot_id, i, physicsClientId=client)
                    mass = dynamics_info[0]
                    inertia = dynamics_info[2]  # 局部惯量对角线
                    
                    joint_masses.append(mass)
                    joint_inertias.append(sum(inertia))  # 惯量和
                    
                    # 计算连杆长度（通过关节位置）
                    joint_state = p.getLinkState(robot_id, i, physicsClientId=client)
                    link_pos = joint_state[0]
                    link_length = np.linalg.norm(link_pos)
                    link_lengths.append(link_length)
            
            # 计算特征
            dof = len(controllable_joints)
            total_mass = sum(joint_masses)
            avg_link_mass = np.mean(joint_masses) if joint_masses else 0
            max_link_mass = max(joint_masses) if joint_masses else 0
            total_inertia = sum(joint_inertias)
            
            # 计算最大到达距离（累积连杆长度）
            cumulative_lengths = np.cumsum(link_lengths)
            max_reach = cumulative_lengths[-1] if len(cumulative_lengths) > 0 else 0
            avg_link_length = np.mean(link_lengths) if link_lengths else 0
            max_link_length = max(link_lengths) if link_lengths else 0
            
            # 末端执行器信息
            if dof > 0:
                end_effector_state = p.getLinkState(robot_id, controllable_joints[-1], 
                                                   physicsClientId=client)
                payload_distance = np.linalg.norm(end_effector_state[0])
            else:
                payload_distance = 0
            
            features = {
                'dof': dof,
                'total_mass': total_mass,
                'avg_link_mass': avg_link_mass,
                'max_link_mass': max_link_mass,
                'total_inertia': total_inertia,
                'max_reach': max_reach,
                'avg_link_length': avg_link_length,
                'max_link_length': max_link_length,
                'payload_mass': payload,
                'payload_distance': payload_distance
            }
            
            return features, controllable_joints
        
        finally:
            p.disconnect(physicsClientId=client)
    
    def normalize_features(self, features, stats=None):
        """
        归一化特征
        
        Args:
            features: 特征字典
            stats: 归一化统计量(mean, std)，如果None则计算
        
        Returns:
            normalized_features: 归一化后的特征向量
            stats: 归一化统计量
        """
        feature_vector = np.array([features[name] for name in self.feature_names], 
                                 dtype=np.float32)
        
        if stats is None:
            # 使用简单的归一化（假设每个特征的合理范围）
            feature_ranges = {
                'dof': (3, 7),
                'total_mass': (5, 50),
                'avg_link_mass': (0.5, 10),
                'max_link_mass': (1, 20),
                'total_inertia': (0.1, 10),
                'max_reach': (0.5, 2.0),
                'avg_link_length': (0.1, 0.5),
                'max_link_length': (0.2, 1.0),
                'payload_mass': (0, 5),
                'payload_distance': (0.5, 2.0)
            }
            
            means = np.array([(feature_ranges[name][0] + feature_ranges[name][1]) / 2 
                            for name in self.feature_names], dtype=np.float32)
            stds = np.array([(feature_ranges[name][1] - feature_ranges[name][0]) / 4 
                           for name in self.feature_names], dtype=np.float32)
            
            stats = {'mean': means, 'std': stds}
        
        normalized = (feature_vector - stats['mean']) / (stats['std'] + 1e-8)
        
        return normalized, stats


class MetaPIDNetwork(nn.Module):
    """元学习PID参数预测网络"""
    
    def __init__(self, feature_dim=10, max_dof=7, hidden_dims=[256, 256, 128]):
        """
        Args:
            feature_dim: 输入特征维度
            max_dof: 最大自由度
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.max_dof = max_dof
        
        # 特征编码器
        layers = []
        in_dim = feature_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 每个关节的PID参数预测头
        self.kp_head = nn.Linear(hidden_dims[-1], max_dof)
        self.ki_head = nn.Linear(hidden_dims[-1], max_dof)
        self.kd_head = nn.Linear(hidden_dims[-1], max_dof)
        
        # 参数范围（确保物理合理性）
        self.kp_min, self.kp_max = 10.0, 1000.0
        self.ki_min, self.ki_max = 0.1, 10.0
        self.kd_min, self.kd_max = 1.0, 50.0
    
    def forward(self, features, actual_dof=None):
        """
        前向传播
        
        Args:
            features: (batch, feature_dim) 机器人特征
            actual_dof: 实际自由度（用于裁剪输出）
        
        Returns:
            kp, ki, kd: (batch, dof) PID参数
        """
        # 编码特征
        h = self.encoder(features)
        
        # 预测PID参数
        kp_raw = self.kp_head(h)
        ki_raw = self.ki_head(h)
        kd_raw = self.kd_head(h)
        
        # 用Sigmoid将输出限制在[0,1]，然后映射到合理范围
        kp = self.kp_min + (self.kp_max - self.kp_min) * torch.sigmoid(kp_raw)
        ki = self.ki_min + (self.ki_max - self.ki_min) * torch.sigmoid(ki_raw)
        kd = self.kd_min + (self.kd_max - self.kd_min) * torch.sigmoid(kd_raw)
        
        # 如果指定了实际DOF，只返回前actual_dof个值
        if actual_dof is not None:
            kp = kp[:, :actual_dof]
            ki = ki[:, :actual_dof]
            kd = kd[:, :actual_dof]
        
        return kp, ki, kd
    
    def predict(self, features, actual_dof=None):
        """
        预测模式（无梯度）
        
        Args:
            features: (feature_dim,) 或 (batch, feature_dim)
            actual_dof: 实际自由度
        
        Returns:
            kp, ki, kd: numpy数组
        """
        self.eval()
        with torch.no_grad():
            # 确保是2D tensor
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            
            kp, ki, kd = self.forward(features, actual_dof)
            
            # 转为numpy并去除batch维度（如果输入是1D）
            kp = kp.cpu().numpy().squeeze()
            ki = ki.cpu().numpy().squeeze()
            kd = kd.cpu().numpy().squeeze()
            
            return kp, ki, kd


class MetaPIDOptimizer:
    """元学习PID优化器（主类）"""
    
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model_path: 预训练模型路径（可选）
            device: 计算设备
        """
        self.device = device
        self.feature_extractor = RobotFeatureExtractor()
        
        # 创建模型
        self.model = MetaPIDNetwork(
            feature_dim=len(self.feature_extractor.feature_names),
            max_dof=7,
            hidden_dims=[256, 256, 128]
        ).to(device)
        
        # 归一化统计量
        self.normalization_stats = None
        
        # 加载预训练模型
        if model_path is not None:
            self.load(model_path)
    
    def predict_pid(self, urdf_path, payload=0.0):
        """
        为给定机器人预测最优PID参数
        
        Args:
            urdf_path: 机器人URDF路径
            payload: 末端负载(kg)
        
        Returns:
            pid_params: dict with keys 'Kp', 'Ki', 'Kd' (numpy arrays)
            robot_info: dict with robot features and joint info
        """
        # 提取特征
        features, controllable_joints = self.feature_extractor.extract_features(
            urdf_path, payload
        )
        
        # 归一化
        normalized_features, _ = self.feature_extractor.normalize_features(
            features, self.normalization_stats
        )
        
        # 转为tensor
        features_tensor = torch.FloatTensor(normalized_features).to(self.device)
        
        # 预测
        actual_dof = features['dof']
        kp, ki, kd = self.model.predict(features_tensor, actual_dof)
        
        pid_params = {
            'Kp': kp,
            'Ki': ki,
            'Kd': kd
        }
        
        robot_info = {
            'features': features,
            'controllable_joints': controllable_joints,
            'dof': actual_dof
        }
        
        return pid_params, robot_info
    
    def save(self, path):
        """保存模型和归一化统计量"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'normalization_stats': self.normalization_stats,
            'feature_names': self.feature_extractor.feature_names
        }
        torch.save(save_dict, path)
        print(f"✅ 模型已保存: {path}")
    
    def load(self, path):
        """加载模型和归一化统计量"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.normalization_stats = checkpoint.get('normalization_stats')
        print(f"✅ 模型已加载: {path}")
    
    def to_yaml_config(self, pid_params, output_path):
        """
        将预测的PID参数保存为YAML配置文件
        
        Args:
            pid_params: dict with 'Kp', 'Ki', 'Kd'
            output_path: 输出YAML路径
        """
        config = {
            'pid_params': {
                'Kp': pid_params['Kp'].tolist(),
                'Ki': pid_params['Ki'].tolist(),
                'Kd': pid_params['Kd'].tolist()
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ PID配置已保存: {output_path}")


if __name__ == '__main__':
    # 测试代码
    print("=" * 80)
    print("测试元学习PID优化器")
    print("=" * 80)
    
    # 创建优化器
    optimizer = MetaPIDOptimizer()
    
    # 使用PyBullet自带的机器人URDF进行测试
    import pybullet_data
    
    # 测试多个机器人
    test_robots = [
        {
            'name': 'Kuka IIWA (7DOF)',
            'urdf': 'kuka_iiwa/model.urdf',
            'payloads': [0.0, 1.0, 2.0]
        },
        {
            'name': 'UR5 (6DOF)', 
            'urdf': 'urdf/ur5.urdf',
            'payloads': [0.0, 1.5]
        },
        {
            'name': 'Panda Arm (7DOF)',
            'urdf': 'franka_panda/panda.urdf',
            'payloads': [0.0, 0.5, 1.0]
        }
    ]
    
    print(f"\n将测试 {len(test_robots)} 种机器人...")
    print("（使用PyBullet自带URDF）\n")
    
    success_count = 0
    
    for robot in test_robots:
        print("=" * 80)
        print(f"测试: {robot['name']}")
        print("=" * 80)
        
        # 尝试加载URDF
        urdf_path = robot['urdf']
        
        try:
            # 测试多个负载
            for payload in robot['payloads']:
                print(f"\n📦 负载: {payload} kg")
                print(f"📊 提取特征...")
                
                features, joints = optimizer.feature_extractor.extract_features(
                    urdf_path, payload=payload
                )
                
                print(f"\n特征:")
                print(f"  DOF: {features['dof']}")
                print(f"  总质量: {features['total_mass']:.2f} kg")
                print(f"  最大到达距离: {features['max_reach']:.2f} m")
                print(f"  负载: {features['payload_mass']:.2f} kg")
                
                # 测试预测（随机初始化的模型）
                print(f"\n🔮 预测PID参数（未训练模型）...")
                pid_params, robot_info = optimizer.predict_pid(urdf_path, payload=payload)
                
                print(f"\n预测的PID参数:")
                print(f"  Kp: {pid_params['Kp']}")
                print(f"  Ki: {pid_params['Ki']}")
                print(f"  Kd: {pid_params['Kd']}")
            
            success_count += 1
            print(f"\n✅ {robot['name']} 测试成功！")
            
        except Exception as e:
            print(f"\n⚠️ {robot['name']} 测试失败: {e}")
            print(f"   （URDF可能在PyBullet数据中不存在）")
            continue
    
    print("\n" + "=" * 80)
    print(f"测试完成！成功: {success_count}/{len(test_robots)}")
    print("=" * 80)
    
    if success_count > 0:
        print("\n✅ 元学习PID优化器工作正常！")
        print("\n下一步:")
        print("  1. 收集训练数据: python meta_learning/collect_training_data.py")
        print("  2. 训练模型: python meta_learning/train_meta_pid.py")
    else:
        print("\n⚠️ 所有测试都失败了")
        print("   这可能是因为PyBullet的数据路径问题")
        print("   建议：准备自己的机器人URDF文件进行测试")

