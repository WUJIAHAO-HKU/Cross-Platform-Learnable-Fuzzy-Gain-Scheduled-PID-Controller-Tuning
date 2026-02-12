#!/usr/bin/env python3
"""
测试J2标签位置的示意图
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置出版样式
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
})

# 模拟数据（类似Figure 4子图c的情况）
x = np.arange(1, 10)  # 9个关节
improvement_percentages = np.array([2.1, 72.6, 2.5, 2.5, 2.1, 2.1, 0, 0, 0])

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ============================================================================
# 左图：修改前（所有标注都在下方）
# ============================================================================
ax1.set_title('修改前：所有标注在下方（J2与曲线重叠❌）', fontsize=12, fontweight='bold')

# 绘制曲线
color_improvement = '#2E7D32'
ax1.plot(x, improvement_percentages, color=color_improvement, marker='o', 
         markersize=8, linewidth=2.5, linestyle='-', alpha=0.9, zorder=10)

# 所有标注都在下方
for i, (xi, yi) in enumerate(zip(x, improvement_percentages)):
    if abs(yi) > 1:
        color_text = 'green' if yi > 0 else 'red'
        # 统一在下方
        ax1.text(xi, yi - 5, f'{yi:+.1f}%', 
                ha='center', va='top', fontsize=9, 
                color=color_text, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor=color_text, alpha=0.8, linewidth=1.5))

ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Joint Index', fontweight='bold', fontsize=11)
ax1.set_ylabel('Improvement (%)', fontweight='bold', fontsize=11, color=color_improvement)
ax1.set_xticks(x)
ax1.set_xticklabels([f'J{i}' for i in x])
ax1.set_ylim(-15, 85)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(axis='y', labelcolor=color_improvement)

# 标注问题区域
ax1.annotate('重叠区域！', xy=(2, 72.6), xytext=(3.5, 60),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ============================================================================
# 右图：修改后（J2在上方，其他在下方）
# ============================================================================
ax2.set_title('修改后：J2在上方，其他在下方（避免重叠✅）', fontsize=12, fontweight='bold')

# 绘制曲线
ax2.plot(x, improvement_percentages, color=color_improvement, marker='o', 
         markersize=8, linewidth=2.5, linestyle='-', alpha=0.9, zorder=10)

# J2在上方，其他在下方
for i, (xi, yi) in enumerate(zip(x, improvement_percentages)):
    if abs(yi) > 1:
        color_text = 'green' if yi > 0 else 'red'
        
        # J2（i=1）在上方，其他在下方
        if i == 1:  # J2
            y_offset = yi + 5
            va = 'bottom'
        else:
            y_offset = yi - 5
            va = 'top'
        
        ax2.text(xi, y_offset, f'{yi:+.1f}%', 
                ha='center', va=va, fontsize=9, 
                color=color_text, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor=color_text, alpha=0.8, linewidth=1.5))

ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Joint Index', fontweight='bold', fontsize=11)
ax2.set_ylabel('Improvement (%)', fontweight='bold', fontsize=11, color=color_improvement)
ax2.set_xticks(x)
ax2.set_xticklabels([f'J{i}' for i in x])
ax2.set_ylim(-15, 85)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color_improvement)

# 标注改进
ax2.annotate('清晰可见！', xy=(2, 72.6), xytext=(3.5, 60),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('j2_label_position_comparison.png', dpi=300, bbox_inches='tight')
print("✅ J2标签位置对比图已保存: j2_label_position_comparison.png")

# 打印代码说明
print("\n" + "="*80)
print("📝 代码修改说明")
print("="*80)
print("""
修改内容：在标注循环中添加条件判断

修改前：
    for i, (xi, yi) in enumerate(zip(x, improvement_percentages)):
        if abs(yi) > 1:
            color_text = 'green' if yi > 0 else 'red'
            ax3_twin.text(xi, yi - 2.5, f'{yi:+.1f}%',    # 全部在下方
                         ha='center', va='top', ...)

修改后：
    for i, (xi, yi) in enumerate(zip(x, improvement_percentages)):
        if abs(yi) > 1:
            color_text = 'green' if yi > 0 else 'red'
            
            # J2（i=1，索引从0开始）特殊处理
            if i == 1:  # J2
                y_offset = yi + 2.5  # 在上方
                va = 'bottom'
            else:  # 其他关节
                y_offset = yi - 2.5  # 在下方
                va = 'top'
            
            ax3_twin.text(xi, y_offset, f'{yi:+.1f}%',
                         ha='center', va=va, ...)

关键点：
  • i == 1 对应 J2（因为Python索引从0开始）
  • y_offset = yi + 2.5  → 标注在数据点上方
  • y_offset = yi - 2.5  → 标注在数据点下方
  • va='bottom' → 文本框底部对齐到y_offset
  • va='top'    → 文本框顶部对齐到y_offset
""")
print("="*80)

plt.show()

