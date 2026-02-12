#!/usr/bin/env python3
"""
图例位置调整教程和实验脚本
演示不同的图例位置和样式配置
"""

import matplotlib.pyplot as plt
import numpy as np

# 创建示例数据
x = np.arange(1, 10)
y1 = np.random.rand(9) * 10 + 5
y2 = np.random.rand(9) * 10 + 3

# ============================================================================
# 常见图例位置示例
# ============================================================================

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('图例位置调整教程 - 不同loc参数效果', fontsize=16, fontweight='bold')

locations = [
    ('upper left', 0, 0),
    ('upper center', 0, 1), 
    ('upper right', 0, 2),
    ('center left', 1, 0),
    ('center', 1, 1),
    ('center right', 1, 2),
    ('lower left', 2, 0),
    ('lower center', 2, 1),
    ('lower right', 2, 2)
]

for loc_name, row, col in locations:
    ax = axes[row, col]
    ax.plot(x, y1, 'o-', label='Data 1', linewidth=2)
    ax.plot(x, y2, 's-', label='Data 2', linewidth=2)
    ax.set_title(f"loc='{loc_name}'", fontsize=12, fontweight='bold')
    ax.legend(loc=loc_name, framealpha=0.9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('legend_positions_basic.png', dpi=150, bbox_inches='tight')
print("✅ 基础位置示例已保存: legend_positions_basic.png")
plt.close()

# ============================================================================
# bbox_to_anchor 精确定位示例
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('图例精确定位 - bbox_to_anchor参数', fontsize=16, fontweight='bold')

anchor_configs = [
    ("外部右上", 'upper left', (1.02, 1.0), "放在图表右侧外部"),
    ("外部上方", 'lower left', (0.0, 1.02), "放在图表上方外部"),
    ("外部下方", 'upper left', (0.0, -0.15), "放在图表下方外部"),
    ("内部中上", 'upper center', (0.5, 0.98), "图表内部中间上方"),
    ("内部右上角", 'upper right', (0.98, 0.98), "图表内部右上角（带偏移）"),
    ("内部左下角", 'lower left', (0.02, 0.02), "图表内部左下角（带偏移）"),
]

for idx, (title, loc, anchor, desc) in enumerate(anchor_configs):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    
    ax.plot(x, y1, 'o-', label='Data 1', linewidth=2, markersize=6)
    ax.plot(x, y2, 's-', label='Data 2', linewidth=2, markersize=6)
    ax.set_title(f"{title}\n{desc}", fontsize=10, fontweight='bold')
    
    ax.legend(loc=loc, 
             bbox_to_anchor=anchor,
             framealpha=0.9,
             edgecolor='blue',
             fancybox=True)
    
    ax.grid(True, alpha=0.3)
    
    # 添加参数说明
    param_text = f"loc='{loc}'\nbbox_to_anchor={anchor}"
    ax.text(0.5, -0.25, param_text, 
           transform=ax.transAxes, 
           ha='center', va='top',
           fontsize=8, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('legend_positions_bbox.png', dpi=150, bbox_inches='tight')
print("✅ bbox_to_anchor示例已保存: legend_positions_bbox.png")
plt.close()

# ============================================================================
# 多列布局和样式示例
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('图例样式配置 - 多列布局和样式', fontsize=16, fontweight='bold')

style_configs = [
    ("单列默认", {'ncol': 1}),
    ("三列横向", {'ncol': 3, 'loc': 'upper center', 'bbox_to_anchor': (0.5, 1.08)}),
    ("小字体+圆角", {'ncol': 2, 'fontsize': 7, 'fancybox': True, 'shadow': True}),
    ("自定义边框", {'ncol': 2, 'edgecolor': 'red', 'linewidth': 2, 'framealpha': 0.8}),
]

for idx, (title, kwargs) in enumerate(style_configs):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    
    # 绘制多条线
    ax.plot(x, y1, 'o-', label='Series A', linewidth=2)
    ax.plot(x, y2, 's-', label='Series B', linewidth=2)
    ax.plot(x, y1 + 2, '^-', label='Series C', linewidth=2)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(**kwargs)
    ax.grid(True, alpha=0.3)
    
    # 显示参数
    param_text = '\n'.join([f'{k}={v}' for k, v in kwargs.items()])
    ax.text(0.02, 0.02, param_text,
           transform=ax.transAxes,
           fontsize=8,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig('legend_styles.png', dpi=150, bbox_inches='tight')
print("✅ 样式配置示例已保存: legend_styles.png")
plt.close()

# ============================================================================
# 推荐配置总结
# ============================================================================

print("\n" + "="*80)
print("📚 图例位置调整总结")
print("="*80)

print("""
1. 基础位置（loc参数）：
   - 'upper left', 'upper center', 'upper right'  (上方三个位置)
   - 'center left', 'center', 'center right'      (中间三个位置)
   - 'lower left', 'lower center', 'lower right'  (下方三个位置)
   - 'best'  (自动选择最佳位置，避免遮挡数据)

2. 精确定位（bbox_to_anchor）：
   格式：bbox_to_anchor=(x, y)
   - x: 水平位置 (0=左边界, 0.5=中间, 1=右边界)
   - y: 垂直位置 (0=底部, 0.5=中间, 1=顶部)
   
   常用组合：
   - (1.02, 1.0)  → 放在图表右侧外部
   - (0.5, 1.02)  → 放在图表上方中间
   - (0.5, -0.15) → 放在图表下方中间
   - (0.98, 0.98) → 放在图表内部右上角（略有偏移）

3. 多列布局（ncol参数）：
   - ncol=1  单列（默认）
   - ncol=2  两列
   - ncol=3  三列（适合横向排列）

4. 样式参数：
   - framealpha: 背景透明度 (0-1)
   - fontsize: 字体大小
   - edgecolor: 边框颜色
   - fancybox: True启用圆角边框
   - shadow: True添加阴影效果

5. Figure 4子图(c)的推荐配置：
   ax.legend(
       loc='upper center',           # 上方中间
       bbox_to_anchor=(0.5, 1.02),  # 略高于图表顶部
       ncol=3,                       # 横向3列
       framealpha=0.95,              # 高不透明度
       fontsize=8,
       edgecolor='gray',
       fancybox=True
   )

💡 调整建议：
   - 如果图例太高/太低：调整 bbox_to_anchor 的 y 值（如 1.02 → 1.05 或 0.98）
   - 如果图例太靠左/右：调整 bbox_to_anchor 的 x 值（如 0.5 → 0.4 或 0.6）
   - 如果图例太宽：减少 ncol 的值（如 3 → 2）
   - 如果图例太大：减小 fontsize（如 8 → 7）
""")

print("="*80)
print("✅ 教程完成！已生成3个示例图片供参考")
print("="*80)

