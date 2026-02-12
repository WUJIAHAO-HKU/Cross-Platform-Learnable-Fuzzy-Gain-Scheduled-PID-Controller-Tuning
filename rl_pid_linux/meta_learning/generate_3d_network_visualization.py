#!/usr/bin/env python3
"""
生成顶刊级别的3D神经网络架构可视化图
包含Meta-PID Network和RL Policy Network的完整训练流程
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch
import matplotlib.gridspec as gridspec

# 设置高质量参数
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'

def draw_3d_layer(ax, x, y, z, width, height, depth, color, alpha=0.7, label=''):
    """绘制3D立方体层"""
    # 定义立方体的8个顶点
    vertices = [
        [x, y, z],
        [x + width, y, z],
        [x + width, y + height, z],
        [x, y + height, z],
        [x, y, z + depth],
        [x + width, y, z + depth],
        [x + width, y + height, z + depth],
        [x, y + height, z + depth]
    ]
    
    # 定义6个面
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 底面
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 顶面
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 前面
        [vertices[4], vertices[5], vertices[6], vertices[7]]   # 后面
    ]
    
    # 绘制面
    face_collection = Poly3DCollection(faces, alpha=alpha, 
                                       facecolors=color, 
                                       edgecolors='black', 
                                       linewidths=0.5)
    ax.add_collection3d(face_collection)
    
    # 添加标签
    if label:
        ax.text(x + width/2, y + height/2, z + depth + 0.3, label,
                fontsize=9, ha='center', va='bottom', weight='bold')
    
    return vertices

def draw_3d_arrow(ax, start, end, color='black', width=0.02):
    """绘制3D箭头"""
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d.proj3d import proj_transform
    
    class Arrow3D(FancyArrowPatch):
        def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._xyz = (x, y, z)
            self._dxdydz = (dx, dy, dz)

        def draw(self, renderer):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            super().draw(renderer)
            
        def do_3d_projection(self, renderer=None):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            
            return np.min(zs)
    
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    
    arrow = Arrow3D(start[0], start[1], start[2], 
                   dx, dy, dz,
                   mutation_scale=20, 
                   lw=2, 
                   arrowstyle='-|>', 
                   color=color)
    ax.add_artist(arrow)

def generate_meta_pid_network_3d():
    """生成Meta-PID Network的3D可视化"""
    print("📊 生成Meta-PID Network 3D架构图...")
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 配色方案（Nature风格）
    colors = {
        'input': '#3498db',      # 蓝色
        'encoder': '#e74c3c',    # 红色
        'hidden': '#f39c12',     # 橙色
        'output': '#27ae60',     # 绿色
        'activation': '#9b59b6', # 紫色
    }
    
    # ========== 输入层 ==========
    x_offset = 0
    input_vertices = draw_3d_layer(ax, x_offset, 0, 0, 0.5, 4, 0.5, 
                                   colors['input'], alpha=0.8, 
                                   label='Input\n(10D)')
    
    # 添加输入特征标签
    features = ['Mass', 'DOF', 'Link Lengths', 'Inertia', '...']
    for i, feat in enumerate(features):
        ax.text(x_offset - 1.5, 4 - i*0.8, 0.25, feat, 
                fontsize=8, ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.6))
    
    # ========== Encoder Layer 1 (256) ==========
    x_offset += 2
    enc1_vertices = draw_3d_layer(ax, x_offset, -0.5, 0, 0.8, 5, 0.8,
                                  colors['encoder'], alpha=0.8,
                                  label='Encoder 1\n(256)')
    
    # LayerNorm + ReLU
    ax.text(x_offset + 0.4, 5.5, 0.4, 'LayerNorm', 
            fontsize=7, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor=colors['activation'], alpha=0.5))
    ax.text(x_offset + 0.4, 5.8, 0.4, 'ReLU', 
            fontsize=7, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor=colors['activation'], alpha=0.5))
    
    # ========== Encoder Layer 2 (256) ==========
    x_offset += 2.5
    enc2_vertices = draw_3d_layer(ax, x_offset, -0.5, 0, 0.8, 5, 0.8,
                                  colors['encoder'], alpha=0.8,
                                  label='Encoder 2\n(256)')
    
    ax.text(x_offset + 0.4, 5.5, 0.4, 'LayerNorm', 
            fontsize=7, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor=colors['activation'], alpha=0.5))
    ax.text(x_offset + 0.4, 5.8, 0.4, 'ReLU', 
            fontsize=7, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor=colors['activation'], alpha=0.5))
    
    # ========== Hidden Layer (128) ==========
    x_offset += 2.5
    hidden_vertices = draw_3d_layer(ax, x_offset, 0.5, 0, 0.6, 3, 0.6,
                                    colors['hidden'], alpha=0.8,
                                    label='Hidden\n(128)')
    
    ax.text(x_offset + 0.3, 4.2, 0.3, 'Dropout(0.1)', 
            fontsize=7, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='gray', alpha=0.5))
    
    # ========== Output Heads (3个) ==========
    x_offset += 2.5
    
    # Kp head
    kp_vertices = draw_3d_layer(ax, x_offset, 3, 0, 0.4, 1.5, 0.4,
                                colors['output'], alpha=0.9,
                                label='K_p Head\n(7)')
    
    # Ki head
    ki_vertices = draw_3d_layer(ax, x_offset, 1.2, 0, 0.4, 1.5, 0.4,
                                colors['output'], alpha=0.9,
                                label='K_i Head\n(7)')
    
    # Kd head
    kd_vertices = draw_3d_layer(ax, x_offset, -0.6, 0, 0.4, 1.5, 0.4,
                                colors['output'], alpha=0.9,
                                label='K_d Head\n(7)')
    
    # ========== Sigmoid激活 ==========
    x_offset += 1.5
    
    for i, (name, y_pos) in enumerate([('K_p', 3.75), ('K_i', 1.95), ('K_d', 0.15)]):
        ax.text(x_offset, y_pos, 0.2, 'σ', 
                fontsize=16, ha='center', weight='bold',
                color=colors['activation'],
                bbox=dict(boxstyle='circle', facecolor='white', 
                         edgecolor=colors['activation'], linewidth=2))
        
        # 输出标签
        ax.text(x_offset + 1, y_pos, 0.2, f'${name} \\in [0,1]^7$', 
                fontsize=9, ha='left', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # ========== 绘制连接箭头 ==========
    # 输入 -> Encoder 1
    draw_3d_arrow(ax, [0.5, 2, 0.25], [2, 2, 0.4], color='gray')
    
    # Encoder 1 -> Encoder 2
    draw_3d_arrow(ax, [2.8, 2, 0.4], [4.5, 2, 0.4], color='gray')
    
    # Encoder 2 -> Hidden
    draw_3d_arrow(ax, [5.3, 2, 0.4], [7, 2, 0.3], color='gray')
    
    # Hidden -> 3 Heads
    for y_target in [3.75, 1.95, 0.15]:
        draw_3d_arrow(ax, [7.6, 2, 0.3], [9.5, y_target, 0.2], color='gray')
    
    # ========== 添加训练流程标注 ==========
    ax.text(4.5, -3, 2, 'Meta-Learning Training Phase', 
            fontsize=14, ha='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    ax.text(4.5, -3.8, 1.5, '303 Virtual Robots → Robot Features → Optimal PID', 
            fontsize=10, ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.5))
    
    # ========== 添加损失函数 ==========
    ax.text(4.5, -5, 1, 
            r'$\mathcal{L}_{meta} = \frac{1}{N}\sum_{v=1}^{N} \|\theta_v^* - \hat{\theta}_v\|_2^2$',
            fontsize=11, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='red', linewidth=2))
    
    # 设置视角和标签
    ax.set_xlabel('Network Depth', fontsize=11, weight='bold')
    ax.set_ylabel('Feature Dimension', fontsize=11, weight='bold')
    ax.set_zlabel('Layer Depth', fontsize=11, weight='bold')
    
    # 设置轴范围
    ax.set_xlim(-2, 12)
    ax.set_ylim(-6, 7)
    ax.set_zlim(-1, 3)
    
    # 设置视角
    ax.view_init(elev=20, azim=130)
    
    # 移除背景网格
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('white')
    
    plt.title('Meta-PID Network Architecture (Hierarchical Meta-Learning)', 
              fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('meta_pid_network_3d.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('meta_pid_network_3d.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✅ 已保存: meta_pid_network_3d.png/pdf")
    plt.close()

def generate_rl_policy_network_3d():
    """生成RL Policy Network的3D可视化"""
    print("📊 生成RL Policy Network 3D架构图...")
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 配色方案
    colors = {
        'observation': '#3498db',
        'policy': '#e74c3c',
        'value': '#f39c12',
        'action': '#27ae60',
    }
    
    # ========== Observation Input ==========
    x_offset = 0
    obs_vertices = draw_3d_layer(ax, x_offset, -1, 0, 0.6, 6, 0.6,
                                 colors['observation'], alpha=0.8,
                                 label='Observation\n(22D)')
    
    # 标注观测空间组成
    obs_components = [
        r'$e_q$ (7D)',
        r'$\dot{e}_q$ (7D)',
        r'$\ddot{e}_q$ (7D)',
        r'$t/T$ (1D)'
    ]
    for i, comp in enumerate(obs_components):
        ax.text(x_offset - 2, 4.5 - i*1.5, 0.3, comp,
                fontsize=8, ha='right', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    # ========== Policy Network Layers ==========
    x_offset += 2.5
    
    # Layer 1 (256)
    policy1 = draw_3d_layer(ax, x_offset, -1, 0, 0.8, 6, 0.8,
                            colors['policy'], alpha=0.8,
                            label='Policy Layer 1\n(256)')
    ax.text(x_offset + 0.4, 5.8, 0.4, 'Tanh', 
            fontsize=8, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='purple', alpha=0.5))
    
    x_offset += 2.5
    
    # Layer 2 (256)
    policy2 = draw_3d_layer(ax, x_offset, -1, 0, 0.8, 6, 0.8,
                            colors['policy'], alpha=0.8,
                            label='Policy Layer 2\n(256)')
    ax.text(x_offset + 0.4, 5.8, 0.4, 'Tanh', 
            fontsize=8, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='purple', alpha=0.5))
    
    # ========== Action Output ==========
    x_offset += 2.5
    action_vertices = draw_3d_layer(ax, x_offset, 1, 0, 0.5, 2, 0.5,
                                    colors['action'], alpha=0.9,
                                    label='Action\n(2D)')
    
    # 动作标注
    ax.text(x_offset + 1.5, 2.5, 0.25, r'$\Delta K_p$ ratio', 
            fontsize=9, ha='left',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(x_offset + 1.5, 1.5, 0.25, r'$\Delta K_d$ ratio', 
            fontsize=9, ha='left',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # ========== Value Network (并行分支) ==========
    x_offset_value = 5  # 从policy layer 2分叉
    
    value1 = draw_3d_layer(ax, x_offset_value, -5, 0, 0.6, 3, 0.6,
                           colors['value'], alpha=0.8,
                           label='Value Layer\n(256)')
    
    x_offset_value += 2
    value_out = draw_3d_layer(ax, x_offset_value, -4.5, 0, 0.4, 2, 0.4,
                              colors['value'], alpha=0.9,
                              label='Value\n(1D)')
    
    ax.text(x_offset_value + 1, -3.5, 0.2, r'$V(s)$',
            fontsize=10, ha='left', weight='bold',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    # ========== 绘制连接箭头 ==========
    draw_3d_arrow(ax, [0.6, 2, 0.3], [2.5, 2, 0.4], color='gray')
    draw_3d_arrow(ax, [3.3, 2, 0.4], [5, 2, 0.4], color='gray')
    draw_3d_arrow(ax, [5.8, 2, 0.4], [7.5, 2, 0.25], color='gray')
    
    # Value分支箭头
    draw_3d_arrow(ax, [5.4, 0, 0.4], [5, -2, 0.3], color='orange')
    draw_3d_arrow(ax, [5.6, -3.5, 0.3], [7, -3.5, 0.2], color='orange')
    
    # ========== PPO训练流程标注 ==========
    ax.text(4, -7, 2, 'PPO Online Training Phase', 
            fontsize=14, ha='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    ax.text(4, -7.8, 1.5, 
            'Observation → Policy → Action → Environment → Reward',
            fontsize=10, ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.5))
    
    # ========== PPO损失函数 ==========
    ax.text(4, -9, 1,
            r'$\mathcal{L}^{PPO} = \mathcal{L}^{CLIP} + c_1\mathcal{L}^{VF} - c_2 S[\pi]$',
            fontsize=11, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='red', linewidth=2))
    
    # 设置视角和标签
    ax.set_xlabel('Network Depth', fontsize=11, weight='bold')
    ax.set_ylabel('Feature Dimension', fontsize=11, weight='bold')
    ax.set_zlabel('Layer Depth', fontsize=11, weight='bold')
    
    ax.set_xlim(-3, 10)
    ax.set_ylim(-10, 7)
    ax.set_zlim(-1, 3)
    
    ax.view_init(elev=18, azim=125)
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('white')
    
    plt.title('RL Policy Network Architecture (PPO for Online Adaptation)',
              fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('rl_policy_network_3d.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('rl_policy_network_3d.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✅ 已保存: rl_policy_network_3d.png/pdf")
    plt.close()

def generate_combined_pipeline():
    """生成完整的训练流程图（2D高级版）"""
    print("📊 生成完整训练流程图...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # ========== Phase 1: Data Augmentation ==========
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 3)
    ax1.axis('off')
    ax1.set_title('Phase 1: Physics-Based Data Augmentation', 
                  fontsize=14, weight='bold', pad=10)
    
    # 3个base robots
    for i, (robot, dof, color) in enumerate([('Franka', '9-DOF', '#3498db'),
                                               ('KUKA', '7-DOF', '#e74c3c'),
                                               ('Laikago', '12-DOF', '#f39c12')]):
        x = 1 + i*2.5
        rect = FancyBboxPatch((x-0.3, 1.2), 0.6, 0.8, 
                              boxstyle='round,pad=0.05',
                              facecolor=color, edgecolor='black', 
                              linewidth=2, alpha=0.7)
        ax1.add_patch(rect)
        ax1.text(x, 1.6, robot, ha='center', va='center', 
                fontsize=10, weight='bold', color='white')
        ax1.text(x, 1.3, dof, ha='center', va='center',
                fontsize=8, color='white')
    
    # 箭头指向数据增强
    ax1.annotate('', xy=(8.5, 1.6), xytext=(7, 1.6),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax1.text(7.7, 2, 'Perturbation', ha='center', fontsize=9, style='italic')
    
    # 数据增强结果
    rect_aug = FancyBboxPatch((8.5, 0.8), 1.2, 1.6,
                              boxstyle='round,pad=0.1',
                              facecolor='#27ae60', edgecolor='black',
                              linewidth=2, alpha=0.7)
    ax1.add_patch(rect_aug)
    ax1.text(9.1, 1.9, '303', ha='center', fontsize=16, weight='bold', color='white')
    ax1.text(9.1, 1.5, 'Virtual', ha='center', fontsize=10, color='white')
    ax1.text(9.1, 1.1, 'Robots', ha='center', fontsize=10, color='white')
    
    # ========== Phase 2: Meta-Learning ==========
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5)
    ax2.axis('off')
    ax2.set_title('Phase 2: Meta-PID Network Training',
                  fontsize=14, weight='bold', pad=10)
    
    # 输入特征
    rect_feat = FancyBboxPatch((0.5, 2), 1.5, 1,
                               boxstyle='round,pad=0.1',
                               facecolor='#3498db', edgecolor='black',
                               linewidth=2, alpha=0.7)
    ax2.add_patch(rect_feat)
    ax2.text(1.25, 2.5, 'Robot\nFeatures', ha='center', va='center',
            fontsize=10, weight='bold', color='white')
    
    # 神经网络
    network_layers = [
        (3, 'Encoder\n256', '#e74c3c'),
        (4.5, 'Encoder\n256', '#e74c3c'),
        (6, 'Hidden\n128', '#f39c12')
    ]
    
    for x, label, color in network_layers:
        rect = FancyBboxPatch((x-0.4, 1.8), 0.8, 1.4,
                             boxstyle='round,pad=0.1',
                             facecolor=color, edgecolor='black',
                             linewidth=2, alpha=0.7)
        ax2.add_patch(rect)
        ax2.text(x, 2.5, label, ha='center', va='center',
                fontsize=9, weight='bold', color='white')
        
        # 连接箭头
        if x > 3:
            ax2.annotate('', xy=(x-0.5, 2.5), xytext=(x-1.3, 2.5),
                        arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    # 输出头
    for i, (name, y_pos, color) in enumerate([('K_p', 4, '#27ae60'),
                                                ('K_i', 2.5, '#27ae60'),
                                                ('K_d', 1, '#27ae60')]):
        rect = FancyBboxPatch((7.5, y_pos-0.3), 1, 0.6,
                             boxstyle='round,pad=0.05',
                             facecolor=color, edgecolor='black',
                             linewidth=2, alpha=0.8)
        ax2.add_patch(rect)
        ax2.text(8, y_pos, f'{name} Head', ha='center', va='center',
                fontsize=9, weight='bold', color='white')
        
        # 连接
        ax2.annotate('', xy=(7.4, y_pos), xytext=(6.5, 2.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # 损失函数
    ax2.text(5, 0.3, r'$\mathcal{L}_{meta} = \|\theta^* - \hat{\theta}\|_2^2$',
            ha='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    
    # ========== Phase 3: RL Fine-tuning ==========
    ax3 = fig.add_subplot(gs[1:, 2])
    ax3.set_xlim(0, 5)
    ax3.set_ylim(0, 8)
    ax3.axis('off')
    ax3.set_title('Phase 3: RL\nOnline Adaptation',
                  fontsize=13, weight='bold', pad=10)
    
    # PPO流程（垂直布局）
    stages = [
        (7, 'Observation', '#3498db'),
        (5.5, 'Policy π', '#e74c3c'),
        (4, 'Action Δθ', '#27ae60'),
        (2.5, 'Environment', '#9b59b6'),
        (1, 'Reward R', '#f39c12')
    ]
    
    for y, label, color in stages:
        rect = FancyBboxPatch((1, y-0.4), 3, 0.8,
                             boxstyle='round,pad=0.1',
                             facecolor=color, edgecolor='black',
                             linewidth=2, alpha=0.7)
        ax3.add_patch(rect)
        ax3.text(2.5, y, label, ha='center', va='center',
                fontsize=10, weight='bold', color='white')
        
        # 连接
        if y > 1:
            ax3.annotate('', xy=(2.5, y-0.5), xytext=(2.5, y-1.1),
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 反馈循环
    ax3.annotate('', xy=(3.8, 7.3), xytext=(3.8, 1.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='red',
                               linestyle='dashed', connectionstyle='arc3,rad=0.3'))
    
    # ========== 底部统计信息 ==========
    ax4 = fig.add_subplot(gs[2, :2])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 2)
    ax4.axis('off')
    
    stats = [
        ('Training Time', '20 min', '#3498db'),
        ('Training Samples', '200k', '#e74c3c'),
        ('Cross-Platform MAE', '5.37°', '#27ae60')
    ]
    
    for i, (label, value, color) in enumerate(stats):
        x = 1.5 + i*3
        rect = FancyBboxPatch((x-0.6, 0.5), 1.2, 1,
                             boxstyle='round,pad=0.1',
                             facecolor=color, edgecolor='black',
                             linewidth=2, alpha=0.7)
        ax4.add_patch(rect)
        ax4.text(x, 1.3, value, ha='center', va='center',
                fontsize=14, weight='bold', color='white')
        ax4.text(x, 0.8, label, ha='center', va='center',
                fontsize=8, color='white')
    
    plt.suptitle('Hierarchical Meta-Learning Framework for Cross-Platform PID Control',
                fontsize=16, weight='bold', y=0.98)
    
    plt.savefig('complete_training_pipeline.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('complete_training_pipeline.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✅ 已保存: complete_training_pipeline.png/pdf")
    plt.close()

def main():
    """主函数"""
    print("="*80)
    print("🎨 生成顶刊级别神经网络架构可视化")
    print("="*80)
    
    # 生成3个高质量可视化
    generate_meta_pid_network_3d()
    print()
    generate_rl_policy_network_3d()
    print()
    generate_combined_pipeline()
    
    print()
    print("="*80)
    print("✅ 所有可视化生成完成！")
    print("="*80)
    print()
    print("📁 生成的文件：")
    print("   1. meta_pid_network_3d.png/pdf - Meta-PID网络3D架构")
    print("   2. rl_policy_network_3d.png/pdf - RL策略网络3D架构")
    print("   3. complete_training_pipeline.png/pdf - 完整训练流程")
    print()
    print("🎯 特点：")
    print("   ✅ 3D立体效果，视觉冲击力强")
    print("   ✅ 丰富的标注和颜色编码")
    print("   ✅ 完整的数学公式")
    print("   ✅ 300 DPI高分辨率，适合顶刊")
    print("   ✅ 同时生成PNG和PDF格式")
    print()

if __name__ == '__main__':
    main()

