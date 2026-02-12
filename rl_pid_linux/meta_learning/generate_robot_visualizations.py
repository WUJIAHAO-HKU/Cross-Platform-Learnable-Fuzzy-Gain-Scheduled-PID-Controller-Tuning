#!/usr/bin/env python3
"""
生成机器人模型可视化图片
用于论文的系统架构图和方法说明
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrow
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

def create_franka_panda_visualization():
    """
    创建Franka Panda机械臂的可视化图
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 8), dpi=150)
    
    # 机械臂参数
    base_width = 0.3
    base_height = 0.15
    link_width = 0.08
    joint_radius = 0.12
    
    # 7-DOF机械臂的关节位置（示意）
    joint_positions = [
        (0, 0),           # 基座
        (0, 0.3),         # 关节1
        (0.25, 0.6),      # 关节2
        (0.35, 1.0),      # 关节3
        (0.3, 1.4),       # 关节4
        (0.45, 1.75),     # 关节5
        (0.5, 2.1),       # 关节6
        (0.55, 2.4),      # 关节7 + 末端执行器
    ]
    
    # 绘制基座
    base = FancyBboxPatch(
        (-base_width/2, -base_height), base_width, base_height,
        boxstyle="round,pad=0.02", 
        edgecolor='#2C3E50', facecolor='#34495E', linewidth=2
    )
    ax.add_patch(base)
    
    # 绘制连杆和关节
    colors_links = ['#3498DB', '#5DADE2', '#85C1E2', '#AED6F1', '#D6EAF8', '#EBF5FB']
    
    for i in range(len(joint_positions) - 1):
        x1, y1 = joint_positions[i]
        x2, y2 = joint_positions[i + 1]
        
        # 绘制连杆
        if i < len(colors_links):
            line = Line2D([x1, x2], [y1, y2], 
                         linewidth=link_width*100, 
                         color=colors_links[i],
                         solid_capstyle='round',
                         zorder=1)
            ax.add_line(line)
        
        # 绘制关节
        if i < 7:  # 前7个是旋转关节
            joint = Circle((x2, y2), joint_radius, 
                          edgecolor='#E67E22', 
                          facecolor='#F39C12',
                          linewidth=2.5,
                          zorder=2)
            ax.add_patch(joint)
            
            # 关节编号
            ax.text(x2, y2, f'J{i+1}', 
                   ha='center', va='center',
                   fontsize=9, fontweight='bold',
                   color='white', zorder=3)
    
    # 绘制末端执行器（夹爪）
    gripper_x, gripper_y = joint_positions[-1]
    
    # 左指
    left_finger = Rectangle((gripper_x - 0.15, gripper_y), 0.08, 0.25,
                            edgecolor='#27AE60', facecolor='#2ECC71',
                            linewidth=2)
    ax.add_patch(left_finger)
    
    # 右指
    right_finger = Rectangle((gripper_x + 0.07, gripper_y), 0.08, 0.25,
                             edgecolor='#27AE60', facecolor='#2ECC71',
                             linewidth=2)
    ax.add_patch(right_finger)
    
    # 添加坐标系（末端）
    arrow_length = 0.2
    ax.arrow(gripper_x, gripper_y + 0.3, arrow_length, 0, 
            head_width=0.06, head_length=0.05, fc='red', ec='red', linewidth=2)
    ax.arrow(gripper_x, gripper_y + 0.3, 0, arrow_length,
            head_width=0.06, head_length=0.05, fc='green', ec='green', linewidth=2)
    ax.text(gripper_x + arrow_length + 0.08, gripper_y + 0.3, 'X', 
           color='red', fontsize=11, fontweight='bold')
    ax.text(gripper_x, gripper_y + 0.3 + arrow_length + 0.08, 'Y',
           color='green', fontsize=11, fontweight='bold')
    
    # 添加标题和说明
    ax.text(0.3, -0.5, 'Franka Panda', 
           fontsize=16, fontweight='bold', ha='center',
           color='#2C3E50')
    ax.text(0.3, -0.7, '7-DOF Manipulator', 
           fontsize=12, ha='center', style='italic',
           color='#34495E')
    
    # 添加规格信息
    specs = [
        'DOF: 9 (7 arm + 2 gripper)',
        'Payload: 3 kg',
        'Reach: 855 mm',
        'Repeatability: ±0.1 mm'
    ]
    
    for idx, spec in enumerate(specs):
        ax.text(-0.5, -1.0 - idx*0.15, f'• {spec}',
               fontsize=9, color='#34495E',
               verticalalignment='top')
    
    # 设置坐标轴
    ax.set_xlim(-0.6, 1.2)
    ax.set_ylim(-1.8, 3.0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_laikago_quadruped_visualization():
    """
    创建Laikago四足机器人的可视化图
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    
    # 机身参数
    body_length = 1.0
    body_width = 0.4
    body_height = 0.15
    
    # 腿部参数
    hip_length = 0.15
    thigh_length = 0.4
    calf_length = 0.4
    
    # 机身中心位置
    body_x = 0
    body_y = 0.6
    
    # 绘制机身
    body = FancyBboxPatch(
        (body_x - body_length/2, body_y - body_height/2),
        body_length, body_height,
        boxstyle="round,pad=0.02",
        edgecolor='#2874A6', facecolor='#5DADE2',
        linewidth=3, zorder=5
    )
    ax.add_patch(body)
    
    # 添加机身文字
    ax.text(body_x, body_y, 'BODY', 
           ha='center', va='center',
           fontsize=11, fontweight='bold', color='white')
    
    # 四条腿的基座位置 (前左, 前右, 后左, 后右)
    leg_bases = [
        (body_x - body_length/2 + 0.1, body_y, 'FL'),  # Front Left
        (body_x + body_length/2 - 0.1, body_y, 'FR'),  # Front Right
        (body_x - body_length/2 + 0.1, body_y, 'RL'),  # Rear Left
        (body_x + body_length/2 - 0.1, body_y, 'RR'),  # Rear Right
    ]
    
    # 腿的角度配置 (用于站立姿态)
    leg_configs = [
        {'hip': -0.1, 'thigh': 0.8, 'calf': -1.5, 'side': -1},  # FL
        {'hip': 0.1, 'thigh': 0.8, 'calf': -1.5, 'side': 1},    # FR
        {'hip': -0.1, 'thigh': 0.8, 'calf': -1.5, 'side': -1},  # RL
        {'hip': 0.1, 'thigh': 0.8, 'calf': -1.5, 'side': 1},    # RR
    ]
    
    joint_num = 1
    for idx, ((base_x, base_y, leg_name), config) in enumerate(zip(leg_bases, leg_configs)):
        side = config['side']
        
        # Hip关节位置
        hip_x = base_x + side * hip_length
        hip_y = base_y
        
        # Hip连接线
        ax.plot([base_x, hip_x], [base_y, hip_y],
               'o-', color='#E67E22', linewidth=4, markersize=8, zorder=4)
        
        # Thigh（大腿）
        thigh_end_x = hip_x + thigh_length * np.sin(config['thigh'])
        thigh_end_y = hip_y - thigh_length * np.cos(config['thigh'])
        
        ax.plot([hip_x, thigh_end_x], [hip_y, thigh_end_y],
               'o-', color='#3498DB', linewidth=6, markersize=10, zorder=3)
        
        # Knee关节
        knee_x = thigh_end_x
        knee_y = thigh_end_y
        
        # Calf（小腿）
        calf_end_x = knee_x + calf_length * np.sin(config['calf'])
        calf_end_y = knee_y - calf_length * np.cos(config['calf'])
        
        ax.plot([knee_x, calf_end_x], [knee_y, calf_end_y],
               'o-', color='#2ECC71', linewidth=6, markersize=10, zorder=3)
        
        # 脚掌
        foot = Circle((calf_end_x, calf_end_y), 0.04,
                     edgecolor='#34495E', facecolor='#7F8C8D',
                     linewidth=2, zorder=2)
        ax.add_patch(foot)
        
        # 标注腿名称
        ax.text(base_x, base_y + 0.25, leg_name,
               ha='center', fontsize=9, fontweight='bold',
               color='#2C3E50')
        
        # 标注关节编号
        ax.text(hip_x + side*0.08, hip_y + 0.05, f'{joint_num}',
               fontsize=7, color='white', fontweight='bold',
               bbox=dict(boxstyle='circle', facecolor='#E67E22', edgecolor='none'))
        ax.text(knee_x + side*0.08, knee_y, f'{joint_num+1}',
               fontsize=7, color='white', fontweight='bold',
               bbox=dict(boxstyle='circle', facecolor='#3498DB', edgecolor='none'))
        ax.text(calf_end_x + side*0.08, calf_end_y, f'{joint_num+2}',
               fontsize=7, color='white', fontweight='bold',
               bbox=dict(boxstyle='circle', facecolor='#2ECC71', edgecolor='none'))
        
        joint_num += 3
    
    # 添加地面
    ground_y = -0.05
    ax.plot([-0.8, 0.8], [ground_y, ground_y],
           'k-', linewidth=3, zorder=1)
    # 地面阴影线
    for i in np.linspace(-0.8, 0.8, 20):
        ax.plot([i, i-0.05], [ground_y, ground_y-0.08],
               'k-', linewidth=1, alpha=0.5, zorder=1)
    
    # 添加标题
    ax.text(0, 1.1, 'Laikago Quadruped Robot',
           fontsize=16, fontweight='bold', ha='center',
           color='#2C3E50')
    ax.text(0, 1.0, '12-DOF (4 legs × 3 joints)',
           fontsize=12, ha='center', style='italic',
           color='#34495E')
    
    # 添加规格信息
    specs = [
        'DOF: 12 (3 per leg)',
        'Mass: 23 kg',
        'Max Speed: 3.5 m/s',
        'Gaits: Trot, Walk, Stand'
    ]
    
    for idx, spec in enumerate(specs):
        ax.text(-0.75, -0.35 - idx*0.12, f'• {spec}',
               fontsize=9, color='#34495E',
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        edgecolor='#BDC3C7',
                        alpha=0.8))
    
    # 添加图例
    legend_elements = [
        Line2D([0], [0], color='#E67E22', linewidth=4, marker='o', 
               markersize=8, label='Hip (Abduction)'),
        Line2D([0], [0], color='#3498DB', linewidth=4, marker='o',
               markersize=8, label='Thigh (Flexion)'),
        Line2D([0], [0], color='#2ECC71', linewidth=4, marker='o',
               markersize=8, label='Calf (Extension)')
    ]
    ax.legend(handles=legend_elements, loc='upper right',
             framealpha=0.9, fontsize=8)
    
    # 设置坐标轴
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.6, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    print("=" * 80)
    print("生成机器人模型可视化图片")
    print("=" * 80)
    
    # 生成Franka Panda可视化
    print("\n[1/2] 生成Franka Panda机械臂可视化...")
    fig1 = create_franka_panda_visualization()
    fig1.savefig('franka_panda_visualization.png', 
                 dpi=300, bbox_inches='tight', 
                 facecolor='white', edgecolor='none')
    fig1.savefig('franka_panda_visualization.pdf',
                 bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("   ✓ 保存: franka_panda_visualization.png (PNG, 300 DPI)")
    print("   ✓ 保存: franka_panda_visualization.pdf (PDF, 矢量图)")
    plt.close(fig1)
    
    # 生成Laikago可视化
    print("\n[2/2] 生成Laikago四足机器人可视化...")
    fig2 = create_laikago_quadruped_visualization()
    fig2.savefig('laikago_quadruped_visualization.png',
                 dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    fig2.savefig('laikago_quadruped_visualization.pdf',
                 bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("   ✓ 保存: laikago_quadruped_visualization.png (PNG, 300 DPI)")
    print("   ✓ 保存: laikago_quadruped_visualization.pdf (PDF, 矢量图)")
    plt.close(fig2)
    
    print("\n" + "=" * 80)
    print("✅ 机器人可视化图片生成完成！")
    print("=" * 80)
    print("\n生成的文件:")
    print("  • franka_panda_visualization.png/pdf     - 7-DOF机械臂")
    print("  • laikago_quadruped_visualization.png/pdf - 12-DOF四足机器人")
    print("\n用途:")
    print("  • 用于论文系统架构图")
    print("  • 用于方法说明和实验平台介绍")
    print("  • 可直接插入LaTeX文档")
    print("\n建议:")
    print("  • 使用PDF版本获得最佳质量（矢量图）")
    print("  • 在LaTeX中使用 \\includegraphics 插入")
    print("  • 可放置在实验设置章节或附录中")
    print()

if __name__ == '__main__':
    main()

