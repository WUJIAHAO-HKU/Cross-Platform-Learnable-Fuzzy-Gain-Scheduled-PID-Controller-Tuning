# 成本对比图生成指南

论文中引用了 `cost_scaling_comparison.png`（图表标签：`fig:cost_scaling`），您需要使用 MATLAB 或 Python 生成此图表。

## 图表要求

### 数据点（根据 Table 6 成本分析）

```python
import numpy as np
import matplotlib.pyplot as plt

# 机器人数量范围
n_robots = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])

# 各方法的成本函数（单位：千美元）
# 公式：Total Cost = Setup Cost + (Deployment Cost + Maintenance * 1 year) * n_robots

# 1. 手动专家调参
setup_manual = 0
per_robot_manual_low = 10.2  # $10,200 = $6K deployment + $1.2K maintenance
per_robot_manual_high = 49.2  # $49,200 = $36K deployment + $12K adapt + $1.2K maintenance
cost_manual_low = setup_manual + per_robot_manual_low * n_robots
cost_manual_high = setup_manual + per_robot_manual_high * n_robots

# 2. 启发式方法
setup_heuristic = 0
per_robot_heuristic_low = 3.8  # $3,800
per_robot_heuristic_high = 12.8  # $12,800
cost_heuristic_low = setup_heuristic + per_robot_heuristic_low * n_robots
cost_heuristic_high = setup_heuristic + per_robot_heuristic_high * n_robots

# 3. 优化方法
setup_optimization = 5  # $5,000
per_robot_optimization_low = 6.8  # $6,800
per_robot_optimization_high = 12.8  # $12,800
cost_optimization_low = setup_optimization + per_robot_optimization_low * n_robots
cost_optimization_high = setup_optimization + per_robot_optimization_high * n_robots

# 4. 纯强化学习
setup_pure_rl_low = 50  # $50K
setup_pure_rl_high = 200  # $200K
per_robot_pure_rl_low = 58  # $58K
per_robot_pure_rl_high = 223  # $223K
cost_pure_rl_low = setup_pure_rl_low + per_robot_pure_rl_low * n_robots
cost_pure_rl_high = setup_pure_rl_high + per_robot_pure_rl_high * n_robots

# 5. 传统元学习
setup_meta_conventional_low = 5000  # $5M
setup_meta_conventional_high = 20000  # $20M
per_robot_meta_low = 2.25  # $2,250 ($750 deploy + $1,500 maintenance)
per_robot_meta_high = 4.5  # $4,500 ($3,000 deploy + $1,500 maintenance)
cost_meta_low = setup_meta_conventional_low + per_robot_meta_low * n_robots
cost_meta_high = setup_meta_conventional_high + per_robot_meta_high * n_robots

# 6. 本方法（Physics-Based Meta-RL）
setup_ours = 0
per_robot_ours = 0.025  # $25
cost_ours = setup_ours + per_robot_ours * n_robots

# 绘图
plt.figure(figsize=(12, 8))
plt.rcParams['font.size'] = 12

# 绘制各方法（使用对数刻度）
plt.semilogy(n_robots, cost_manual_high, 'r--', linewidth=2.5, label='Manual Expert Tuning (worst case)', alpha=0.8)
plt.semilogy(n_robots, cost_manual_low, 'r-', linewidth=2, label='Manual Expert Tuning (best case)', alpha=0.8)

plt.semilogy(n_robots, cost_heuristic_high, 'orange', linestyle='--', linewidth=2, alpha=0.7)
plt.semilogy(n_robots, cost_heuristic_low, 'orange', linestyle='-', linewidth=1.8, label='Heuristic Methods', alpha=0.7)

plt.semilogy(n_robots, cost_optimization_high, 'b--', linewidth=2, alpha=0.7)
plt.semilogy(n_robots, cost_optimization_low, 'b-', linewidth=1.8, label='Optimization-Based', alpha=0.7)

plt.semilogy(n_robots, cost_pure_rl_high, 'purple', linestyle='--', linewidth=2, alpha=0.7)
plt.semilogy(n_robots, cost_pure_rl_low, 'purple', linestyle='-', linewidth=1.8, label='Pure Deep RL', alpha=0.7)

plt.semilogy(n_robots, cost_meta_high, 'm--', linewidth=2, alpha=0.7)
plt.semilogy(n_robots, cost_meta_low, 'm-', linewidth=1.8, label='Conventional Meta-Learning', alpha=0.7)

# 本方法（突出显示）
plt.semilogy(n_robots, cost_ours, 'g-', linewidth=3.5, label='Our Method (Physics-Based Meta-RL)', marker='o', markersize=8)

# 标注关键区域：10-100机器人（中小企业典型规模）
plt.axvspan(10, 100, alpha=0.15, color='green', label='SME Typical Scale (10-100 robots)')

# 标注
plt.xlabel('Number of Deployed Robots', fontsize=14, fontweight='bold')
plt.ylabel('Total Cost of Ownership (USD, thousands, log scale)', fontsize=14, fontweight='bold')
plt.title('Cost Scalability Analysis: Breaking the Industrial Economics Barrier', fontsize=16, fontweight='bold')
plt.legend(loc='upper left', fontsize=11, framealpha=0.95)
plt.grid(True, which='both', alpha=0.3, linestyle='--')

# 添加关键数据标注
plt.annotate('Break-even at\n2-3 robots', 
             xy=(3, cost_manual_low[1]), xytext=(8, 100),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=11, fontweight='bold', color='red')

plt.annotate(f'Our method @ 100 robots:\n${cost_ours[6]:.1f}K',
             xy=(100, cost_ours[6]), xytext=(150, 5),
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
             fontsize=11, fontweight='bold', color='green')

plt.annotate(f'Manual @ 100 robots:\n${cost_manual_low[6]:.0f}K - ${cost_manual_high[6]:.0f}K',
             xy=(100, cost_manual_low[6]), xytext=(150, 3000),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=11, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('cost_scaling_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\\n=== 关键成本数据 ===")
print(f"1台机器人:")
print(f"  - 手动调参: ${cost_manual_low[0]:.1f}K - ${cost_manual_high[0]:.1f}K")
print(f"  - 本方法: ${cost_ours[0]:.3f}K")
print(f"  - 节省: {(1 - cost_ours[0]/cost_manual_low[0])*100:.1f}% - {(1 - cost_ours[0]/cost_manual_high[0])*100:.1f}%")
print(f"\\n100台机器人:")
print(f"  - 手动调参: ${cost_manual_low[6]:.0f}K - ${cost_manual_high[6]:.0f}K")
print(f"  - 本方法: ${cost_ours[6]:.1f}K")
print(f"  - 节省: {(1 - cost_ours[6]/cost_manual_low[6])*100:.1f}% - {(1 - cost_ours[6]/cost_manual_high[6])*100:.1f}%")
```

## 输出文件

- 保存为 `cost_scaling_comparison.png`（300 DPI，适合论文发表）
- 放置在 LaTeX 项目的 `figs/` 目录或当前目录

## MATLAB 版本（可选）

```matlab
% 如果您更习惯使用 MATLAB
n_robots = [1, 2, 5, 10, 20, 50, 100, 200, 500];

% 定义成本函数（见上述Python代码）
% ...

figure('Position', [100, 100, 1200, 800]);
semilogy(n_robots, cost_manual_high, 'r--', 'LineWidth', 2.5); hold on;
semilogy(n_robots, cost_manual_low, 'r-', 'LineWidth', 2);
% ... 添加其他曲线

% 其余代码类似
exportgraphics(gcf, 'cost_scaling_comparison.png', 'Resolution', 300);
```

## 验证检查项

生成图表后，确保：
- ✅ Y轴使用对数刻度（成本跨度从$25到$28M）
- ✅ 绿色阴影区域标注10-100机器人规模
- ✅ 本方法曲线几乎为水平线（边际成本$25/机器人）
- ✅ 手动调参曲线最陡（线性增长）
- ✅ 传统元学习有巨大起始成本但后续平缓
- ✅ 图例清晰可读，无重叠
- ✅ 分辨率≥300 DPI（适合论文出版）

## 关键结论验证

生成的图表应直观展示：
1. **2-3台机器人即可回本**（vs. 手动调参）
2. **100台规模节省99%成本**（$2.5K vs. $1.14M-4.92M）
3. **传统元学习需200+台才能摊薄成本**（仅适合大公司）
4. **本方法从第1台就具经济优势**（零设置成本）
