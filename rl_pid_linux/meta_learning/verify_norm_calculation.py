#!/usr/bin/env python3
"""
验证L2 norm与per-joint平均的关系
"""

import numpy as np

# Table 3的per-joint MAE数据
per_joint_mae_pure = np.array([2.57, 12.36, 4.10, 6.78, 5.41, 4.31, 11.45, 10.23, 10.36])
per_joint_mae_rl = np.array([2.26, 2.42, 3.87, 6.49, 5.32, 4.19, 11.26, 10.19, 10.33])

print("="*80)
print("L2 Norm vs 简单平均 验证")
print("="*80)

print("\n【Table 3的per-joint数据】")
print(f"Pure Meta-PID per-joint: {per_joint_mae_pure}")
print(f"Meta-PID+RL per-joint:   {per_joint_mae_rl}")

print("\n【方法1：简单算术平均】")
avg_pure = np.mean(per_joint_mae_pure)
avg_rl = np.mean(per_joint_mae_rl)
print(f"Pure: {avg_pure:.2f}°")
print(f"RL:   {avg_rl:.2f}°")

print("\n【方法2：L2 Norm（如果这是单个时刻的快照）】")
l2_pure = np.linalg.norm(per_joint_mae_pure)
l2_rl = np.linalg.norm(per_joint_mae_rl)
print(f"Pure: {l2_pure:.2f}°")
print(f"RL:   {l2_rl:.2f}°")

print("\n【方法3：RMS (Root Mean Square)】")
rms_pure = np.sqrt(np.mean(per_joint_mae_pure**2))
rms_rl = np.sqrt(np.mean(per_joint_mae_rl**2))
print(f"Pure: {rms_pure:.2f}°")
print(f"RL:   {rms_rl:.2f}°")

print("\n【Table 1的实际数据】")
table1_mae_pure = 28.67
table1_mae_rl = 24.88
table1_rmse_pure = 29.32
table1_rmse_rl = 25.45
print(f"MAE  - Pure: {table1_mae_pure:.2f}°, RL: {table1_mae_rl:.2f}°")
print(f"RMSE - Pure: {table1_rmse_pure:.2f}°, RL: {table1_rmse_rl:.2f}°")

print("\n【关键发现】")
print(f"✓ L2 norm (Pure) = {l2_pure:.2f}° 接近 Table 1 RMSE = {table1_rmse_pure:.2f}°")
print(f"✓ L2 norm (RL) = {l2_rl:.2f}° 接近 Table 1 RMSE = {table1_rmse_rl:.2f}°")
print(f"✓ RMS (Pure) = {rms_pure:.2f}° 也接近Table 1的值")
print()
print(f"⚠️ 但Table 1的MAE={table1_mae_pure:.2f}°既不等于简单平均({avg_pure:.2f}°)")
print(f"   也不等于L2 norm({l2_pure:.2f}°)或RMS({rms_pure:.2f}°)")

print("\n【推测】")
print("Table 1的MAE可能是：")
print("1. 对每个时刻t计算 ||e(t)||_2")
print("2. 然后对所有时刻求平均：MAE = mean_t(||e(t)||_2)")
print()
print("如果per-joint误差随时间有很大波动，那么：")
print("  mean_t(||e(t)||_2) ≠ ||mean_t(e(t))||_2")
print()
print("这样Table 1的MAE (28.67°)会介于：")
print(f"  - 简单平均 ({avg_pure:.2f}°)")
print(f"  - 和单时刻L2 norm ({l2_pure:.2f}°)之间")

print("\n" + "="*80)

