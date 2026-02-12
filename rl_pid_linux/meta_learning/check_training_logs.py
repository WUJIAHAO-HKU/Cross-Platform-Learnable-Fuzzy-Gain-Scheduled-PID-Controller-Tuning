#!/usr/bin/env python3
"""检查训练日志中的实际reward趋势"""
import numpy as np
import os

# 查看Franka Panda的评估日志
print('='*80)
print('Franka Panda 评估结果:')
print('='*80)
panda_eval = np.load('logs/meta_rl_panda/eval_logs/evaluations.npz', allow_pickle=True)
for key in panda_eval.files:
    data = panda_eval[key]
    print(f'{key}: shape={data.shape}')
    if 'results' in key and len(data) > 0:
        print(f'  最早的3个episodes rewards: {data[0][:3]}')
        print(f'  最后的3个episodes rewards: {data[-1][:3]}')
        print(f'  平均reward变化: {np.mean(data[0]):.2f} -> {np.mean(data[-1]):.2f}')
        print(f'  最小reward: {np.min(data):.2f}')
        print(f'  最大reward: {np.max(data):.2f}')

print()
print('='*80)
print('Laikago 评估结果:')
print('='*80)
# 查看Laikago的评估日志
laikago_eval = np.load('logs/meta_rl_laikago/eval_logs/evaluations.npz', allow_pickle=True)
for key in laikago_eval.files:
    data = laikago_eval[key]
    print(f'{key}: shape={data.shape}')
    if 'results' in key and len(data) > 0:
        print(f'  最早的3个episodes rewards: {data[0][:3]}')
        print(f'  最后的3个episodes rewards: {data[-1][:3]}')
        print(f'  平均reward变化: {np.mean(data[0]):.2f} -> {np.mean(data[-1]):.2f}')
        print(f'  最小reward: {np.min(data):.2f}')
        print(f'  最大reward: {np.max(data):.2f}')

