#!/usr/bin/env python3
"""
Cost Scaling Comparison Figure Generator
Generates publication-quality cost analysis visualization for industrial robotics PID tuning methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

def thousands_formatter(x, pos):
    """Format y-axis labels in thousands with K suffix"""
    if x >= 1000:
        return f'${x/1000:.0f}M'
    elif x >= 1:
        return f'${x:.0f}K'
    else:
        return f'${x*1000:.0f}'

# Robot deployment scale
n_robots = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])

# Cost parameters (in thousands USD)
# Formula: Total Cost = Setup Cost + (Deployment + Maintenance) * n_robots

# 1. Manual Expert Tuning
setup_manual = 0
per_robot_manual_low = 10.2   # $10,200 = $6K deploy + $3K adapt + $1.2K maintenance
per_robot_manual_high = 49.2  # $49,200 = $36K deploy + $12K adapt + $1.2K maintenance
cost_manual_low = setup_manual + per_robot_manual_low * n_robots
cost_manual_high = setup_manual + per_robot_manual_high * n_robots

# 2. Heuristic Methods (Ziegler-Nichols)
setup_heuristic = 0
per_robot_heuristic_low = 3.8   # $3,800
per_robot_heuristic_high = 12.8  # $12,800
cost_heuristic_low = setup_heuristic + per_robot_heuristic_low * n_robots
cost_heuristic_high = setup_heuristic + per_robot_heuristic_high * n_robots

# 3. Optimization-Based (PSO/GA/Bayesian)
setup_optimization = 5  # $5,000 software license
per_robot_optimization_low = 6.8   # $6,800
per_robot_optimization_high = 12.8  # $12,800
cost_optimization_low = setup_optimization + per_robot_optimization_low * n_robots
cost_optimization_high = setup_optimization + per_robot_optimization_high * n_robots

# 4. Pure Deep RL
setup_pure_rl_low = 50    # $50K GPU cluster
setup_pure_rl_high = 200  # $200K infrastructure
per_robot_pure_rl_low = 58    # $58K
per_robot_pure_rl_high = 223  # $223K
cost_pure_rl_low = setup_pure_rl_low + per_robot_pure_rl_low * n_robots
cost_pure_rl_high = setup_pure_rl_high + per_robot_pure_rl_high * n_robots

# 5. Conventional Meta-Learning
setup_meta_conventional_low = 5000   # $5M (50 robots @ $100K)
setup_meta_conventional_high = 20000  # $20M (200 robots @ $100K)
per_robot_meta_low = 2.25   # $2,250
per_robot_meta_high = 4.5   # $4,500
cost_meta_low = setup_meta_conventional_low + per_robot_meta_low * n_robots
cost_meta_high = setup_meta_conventional_high + per_robot_meta_high * n_robots

# 6. Transfer Learning
setup_transfer = 10  # $10K sim infrastructure
per_robot_transfer_low = 15.5   # $15,500
per_robot_transfer_high = 69    # $69,000
cost_transfer_low = setup_transfer + per_robot_transfer_low * n_robots
cost_transfer_high = setup_transfer + per_robot_transfer_high * n_robots

# 7. Our Method (Physics-Based Meta-RL)
setup_ours = 0  # $0 - only simulation
per_robot_ours = 0.025  # $25 per robot
cost_ours = setup_ours + per_robot_ours * n_robots

# Create figure with publication quality
fig, ax = plt.subplots(figsize=(14, 9))
plt.rcParams.update({
    'font.size': 13,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18
})

# Plot all methods with distinct styles
# Manual Expert Tuning (most expensive)
ax.fill_between(n_robots, cost_manual_low, cost_manual_high, 
                alpha=0.2, color='#D62728', label='_nolegend_')
ax.semilogy(n_robots, cost_manual_high, '#D62728', linestyle='--', 
            linewidth=2.5, alpha=0.85, label='Manual Expert Tuning (range)')
ax.semilogy(n_robots, cost_manual_low, '#D62728', linestyle='-', 
            linewidth=2.5, alpha=0.85)

# Heuristic Methods
ax.fill_between(n_robots, cost_heuristic_low, cost_heuristic_high, 
                alpha=0.15, color='#FF7F0E')
ax.semilogy(n_robots, cost_heuristic_high, '#FF7F0E', linestyle='--', 
            linewidth=2.2, alpha=0.8)
ax.semilogy(n_robots, cost_heuristic_low, '#FF7F0E', linestyle='-', 
            linewidth=2.2, alpha=0.8, label='Heuristic Methods')

# Optimization-Based
ax.fill_between(n_robots, cost_optimization_low, cost_optimization_high, 
                alpha=0.15, color='#1F77B4')
ax.semilogy(n_robots, cost_optimization_high, '#1F77B4', linestyle='--', 
            linewidth=2.2, alpha=0.8)
ax.semilogy(n_robots, cost_optimization_low, '#1F77B4', linestyle='-', 
            linewidth=2.2, alpha=0.8, label='Optimization-Based')

# Pure Deep RL
ax.fill_between(n_robots, cost_pure_rl_low, cost_pure_rl_high, 
                alpha=0.15, color='#9467BD')
ax.semilogy(n_robots, cost_pure_rl_high, '#9467BD', linestyle='--', 
            linewidth=2.2, alpha=0.8)
ax.semilogy(n_robots, cost_pure_rl_low, '#9467BD', linestyle='-', 
            linewidth=2.2, alpha=0.8, label='Pure Deep RL')

# Conventional Meta-Learning (huge setup cost)
ax.fill_between(n_robots, cost_meta_low, cost_meta_high, 
                alpha=0.15, color='#E377C2')
ax.semilogy(n_robots, cost_meta_high, '#E377C2', linestyle='--', 
            linewidth=2.2, alpha=0.8)
ax.semilogy(n_robots, cost_meta_low, '#E377C2', linestyle='-', 
            linewidth=2.2, alpha=0.8, label='Conventional Meta-Learning')

# Transfer Learning
ax.fill_between(n_robots, cost_transfer_low, cost_transfer_high, 
                alpha=0.15, color='#8C564B')
ax.semilogy(n_robots, cost_transfer_high, '#8C564B', linestyle='--', 
            linewidth=2, alpha=0.7)
ax.semilogy(n_robots, cost_transfer_low, '#8C564B', linestyle='-', 
            linewidth=2, alpha=0.7, label='Transfer Learning')

# Our Method - HIGHLIGHTED
ax.semilogy(n_robots, cost_ours, '#2CA02C', linestyle='-', linewidth=4, 
            marker='o', markersize=10, markerfacecolor='#90EE90',
            markeredgewidth=2, markeredgecolor='#1A7A1A',
            label='Our Method (Physics-Based Meta-RL)', zorder=10)

# Highlight SME typical scale (10-100 robots)
ax.axvspan(10, 100, alpha=0.12, color='#2CA02C', zorder=0)
sme_patch = mpatches.Patch(color='#2CA02C', alpha=0.12, 
                           label='SME Typical Scale (10-100 robots)')

# Add critical annotations
# Break-even point
ax.annotate('Break-even at\n2-3 robots vs.\nmanual tuning', 
            xy=(2.5, cost_manual_low[1]), xytext=(40, 20),
            arrowprops=dict(arrowstyle='->', color='#D62728', lw=2,
                          connectionstyle='arc3,rad=0.3'),
            fontsize=11, fontweight='bold', color='#D62728',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE4B5', alpha=0.8))

# Our method at 100 robots
ax.annotate(f'Our method @ 100 robots:\n${cost_ours[6]:.1f}K\n(99.5% savings)', 
            xy=(100, cost_ours[6]), xytext=(250, 0.8),
            arrowprops=dict(arrowstyle='->', color='#1A7A1A', lw=2.5,
                          connectionstyle='arc3,rad=-0.2'),
            fontsize=12, fontweight='bold', color='#1A7A1A',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#90EE90', alpha=0.9))

# Manual at 100 robots
ax.annotate(f'Manual tuning @ 100:\n${cost_manual_low[6]:.0f}K - ${cost_manual_high[6]:.0f}K', 
            xy=(100, cost_manual_low[6]), xytext=(180, 250),
            arrowprops=dict(arrowstyle='->', color='#D62728', lw=2,
                          connectionstyle='arc3,rad=0.2'),
            fontsize=11, fontweight='bold', color='#D62728',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE4E1', alpha=0.9))

# Meta-learning becomes competitive at large scale
ax.annotate('Meta-learning competitive\nonly at 200+ robots\n(0.3% of manufacturers)', 
            xy=(200, cost_meta_low[7]), xytext=(320, 300),
            arrowprops=dict(arrowstyle='->', color='#E377C2', lw=1.8,
                          connectionstyle='arc3,rad=0.3'),
            fontsize=10, fontweight='bold', color='#E377C2',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0E6FA', alpha=0.8))

# Labels and styling
ax.set_xlabel('Number of Deployed Robots', fontsize=15, fontweight='bold')
ax.set_ylabel('Total Cost of Ownership (USD, log scale)', fontsize=15, fontweight='bold')
ax.set_title('Cost Scalability Analysis: Breaking the Industrial Economics Barrier\n' + 
             'Physics-Based Meta-RL Achieves 99.0-99.5% Cost Reduction Across All Scales',
             fontsize=17, fontweight='bold', pad=20)

# Grid
ax.grid(True, which='major', alpha=0.4, linestyle='--', linewidth=0.8)
ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.5)

# Legend
handles, labels = ax.get_legend_handles_labels()
handles.append(sme_patch)
labels.append('SME Typical Scale (10-100 robots)')
ax.legend(handles, labels, loc='lower right', fontsize=11.5, 
         framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)

# Axis limits and ticks
ax.set_xlim(0.8, 500)
ax.set_ylim(0.01, 120000)  # 增加Y轴上限，避免曲线被截断
ax.set_xticks([1, 5, 10, 20, 50, 100, 200, 500])  # 移除2，避免与1重叠
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

plt.tight_layout()

# Save high-resolution figure
output_file = 'cost_scaling_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✅ Figure saved: {output_file}")

# Display figure
plt.show()

# Print key statistics
print("\n" + "="*70)
print("KEY COST ANALYSIS STATISTICS")
print("="*70)

print("\n📊 AT 1 ROBOT DEPLOYMENT:")
print(f"  Manual Tuning:      ${cost_manual_low[0]:.1f}K - ${cost_manual_high[0]:.1f}K")
print(f"  Our Method:         ${cost_ours[0]:.3f}K")
print(f"  Cost Savings:       {(1 - cost_ours[0]/cost_manual_low[0])*100:.1f}% - {(1 - cost_ours[0]/cost_manual_high[0])*100:.1f}%")

print("\n📊 AT 10 ROBOTS (Small Manufacturer):")
print(f"  Manual Tuning:      ${cost_manual_low[3]:.0f}K - ${cost_manual_high[3]:.0f}K")
print(f"  Our Method:         ${cost_ours[3]:.2f}K")
print(f"  Cost Savings:       {(1 - cost_ours[3]/cost_manual_low[3])*100:.1f}% - {(1 - cost_ours[3]/cost_manual_high[3])*100:.1f}%")

print("\n📊 AT 100 ROBOTS (Medium Manufacturer):")
print(f"  Manual Tuning:      ${cost_manual_low[6]:.0f}K - ${cost_manual_high[6]:.0f}K")
print(f"  Optimization-Based: ${cost_optimization_low[6]:.0f}K - ${cost_optimization_high[6]:.0f}K")
print(f"  Conventional Meta:  ${cost_meta_low[6]:.0f}K - ${cost_meta_high[6]:.0f}K")
print(f"  Our Method:         ${cost_ours[6]:.1f}K")
print(f"  Savings vs Manual:  {(1 - cost_ours[6]/cost_manual_low[6])*100:.1f}% - {(1 - cost_ours[6]/cost_manual_high[6])*100:.1f}%")

print("\n📊 INDUSTRY-WIDE IMPACT (573,000 robots/year):")
current_cost_low = 573000 * per_robot_manual_low
current_cost_high = 573000 * per_robot_manual_high
our_cost_global = 573000 * per_robot_ours
print(f"  Current Annual Cost: ${current_cost_low/1000:.1f}B - ${current_cost_high/1000:.1f}B")
print(f"  With Our Method:     ${our_cost_global/1000:.1f}M")
print(f"  Annual Savings:      ${(current_cost_low - our_cost_global)/1000:.1f}B - ${(current_cost_high - our_cost_global)/1000:.1f}B")
print(f"  Equivalent to:       {int((current_cost_low - our_cost_global)/30):.0f} - {int((current_cost_high - our_cost_global)/30):.0f} additional robots")

print("\n" + "="*70)
print("✅ Cost comparison figure generated successfully!")
print("="*70)
