#!/usr/bin/env python3
"""
生成物理数据增强流程图 (Figure 2)
用于论文：Physics-Based Data Augmentation Flow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

def create_box(ax, x, y, width, height, text, color='lightblue', edgecolor='navy', style='round'):
    """创建带文字的方框"""
    if style == 'round':
        boxstyle = "round,pad=0.1"
    elif style == 'database':
        boxstyle = "round,pad=0.15"
    else:
        boxstyle = "square,pad=0.1"
    
    bbox = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle=boxstyle,
                          edgecolor=edgecolor, facecolor=color,
                          linewidth=2, zorder=2)
    ax.add_patch(bbox)
    ax.text(x, y, text, ha='center', va='center', fontsize=9,
            weight='bold', zorder=3, wrap=True)

def create_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    """创建箭头"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, color='black',
                           linewidth=2, zorder=1,
                           mutation_scale=20)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# 标题
ax.text(7, 9.5, 'Physics-Based Data Augmentation Flow', 
        fontsize=16, weight='bold', ha='center')

# ============ 第一层：输入 ============
y_level_1 = 8.5

# 基础机器人
create_box(ax, 3, y_level_1, 2.5, 0.8, 
           'Base Robots\n(K=3)', 
           color='#FFE5B4', edgecolor='#FF8C00')

# 扰动范围
create_box(ax, 7, y_level_1, 3, 0.8,
           'Perturbation Ranges\nΔmass: ±10%, Δinertia: ±15%\nΔfriction: ±20%, Δdamping: ±30%',
           color='#E6E6FA', edgecolor='#6A5ACD')

# 样本数
create_box(ax, 11, y_level_1, 2, 0.8,
           'M=100\nper robot',
           color='#F0E68C', edgecolor='#DAA520')

# ============ 箭头到第二层 ============
create_arrow(ax, 3, y_level_1 - 0.5, 7, 7.5)
create_arrow(ax, 7, y_level_1 - 0.5, 7, 7.5)
create_arrow(ax, 11, y_level_1 - 0.5, 7, 7.5)

# ============ 第二层：虚拟机器人生成 ============
y_level_2 = 7

create_box(ax, 7, y_level_2, 4, 1,
           'Virtual Robot Generation\nAlgorithm 1: Physics-Based Augmentation\n' +
           'Sample: α_mass ~ U(0.9, 1.1), α_inertia ~ U(0.85, 1.15)\n' +
           'Generate virtual URDF with perturbed parameters',
           color='#98FB98', edgecolor='#228B22', style='round')

# ============ 第三层：并行优化 ============
y_level_3 = 5.5

create_arrow(ax, 7, y_level_2 - 0.6, 7, y_level_3 + 0.6, label='300 virtual\nrobots')

# 多个并行优化框
positions = [2.5, 5, 7.5, 10, 11.5]
for i, x_pos in enumerate(positions):
    if i < len(positions) - 1:
        create_box(ax, x_pos, y_level_3, 2, 1.2,
                   f'Optimize PID\nfor Virtual {i+1}\n\n' +
                   'Algorithm 2:\nDE + Nelder-Mead\n' +
                   f'θ*_v, L_v(θ*)',
                   color='#FFB6C1', edgecolor='#DC143C', style='round')
    else:
        # 最后一个用省略号
        ax.text(x_pos, y_level_3, '...', fontsize=24, ha='center', va='center', weight='bold')

# ============ 第四层：质量过滤 ============
y_level_4 = 3.5

# 从所有优化框到过滤
for x_pos in positions[:-1]:
    create_arrow(ax, x_pos, y_level_3 - 0.7, 7, y_level_4 + 0.6)

create_box(ax, 7, y_level_4, 4.5, 0.9,
           'Quality Filtering\nRetain: L_v(θ*_v) < 30°\n' +
           'Filter out: High optimization error samples',
           color='#F0E68C', edgecolor='#DAA520')

# ============ 第五层：高质量数据集 ============
y_level_5 = 2

create_arrow(ax, 7, y_level_4 - 0.5, 7, y_level_5 + 0.5, label='303/350\nretained')

create_box(ax, 7, y_level_5, 5, 1,
           'High-Quality Augmented Dataset\n' +
           '203 samples (3 real + 200 virtual)\n' +
           'Each: {features, optimal_pid, optimization_error}',
           color='#87CEEB', edgecolor='#4682B4', style='database')

# ============ 第六层：元学习训练 ============
y_level_6 = 0.5

create_arrow(ax, 7, y_level_5 - 0.6, 7, y_level_6 + 0.5)

create_box(ax, 7, y_level_6, 4, 0.8,
           'Meta-Learning Training\nWeighted by optimization error\n→ PID Predictor Network',
           color='#DDA0DD', edgecolor='#9370DB')

# ============ 添加侧边标注 ============
# 左侧标注
ax.text(0.3, 8.5, 'Input', fontsize=10, weight='bold', 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
ax.text(0.3, 7, 'Generation', fontsize=10, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
ax.text(0.3, 5.5, 'Optimization', fontsize=10, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='pink', alpha=0.3))
ax.text(0.3, 3.5, 'Filtering', fontsize=10, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.3))
ax.text(0.3, 2, 'Dataset', fontsize=10, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
ax.text(0.3, 0.5, 'Training', fontsize=10, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='plum', alpha=0.3))

# ============ 添加关键统计信息框 ============
# 右下角统计框
stats_text = (
    'Key Statistics:\n'
    '━━━━━━━━━━━━━\n'
    '• Base robots: 3\n'
    '• Virtual per base: 100\n'
    '• Total generated: 300\n'
    '• Avg. opt. error: 19.74°\n'
    '• Samples retained: 203\n'
    '• Optimization time:\n'
    '  30-60s per robot'
)
ax.text(12.5, 1.5, stats_text, fontsize=8,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFACD', 
                 edgecolor='#DAA520', linewidth=2),
        verticalalignment='top', family='monospace')

plt.tight_layout()
plt.savefig('data_augmentation_flow.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('data_augmentation_flow.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ 物理数据增强流程图已生成:")
print("   • data_augmentation_flow.png (高分辨率)")
print("   • data_augmentation_flow.pdf (矢量图)")

plt.show()

