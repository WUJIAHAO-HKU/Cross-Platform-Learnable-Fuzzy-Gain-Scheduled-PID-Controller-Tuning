#!/bin/bash
# 数据准备流程自动化脚本
# 用途：生成虚拟样本 + 优化真实PID参数

set -e  # 遇到错误立即退出

export META_DIR="/home/wujiahao/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/rl_pid_linux/meta_learning"
cd $META_DIR
source ~/rl_robot_env/bin/activate

echo "=========================================="
echo "📦 数据准备流程"
echo "=========================================="
echo ""

echo "=========================================="
echo "步骤1: 生成虚拟机器人样本"
echo "预计时间: 5分钟"
echo "=========================================="
python data_augmentation.py

echo ""
echo "=========================================="
echo "步骤2: 优化真实最优PID参数"
echo "预计时间: 40分钟（8核并行）"
echo "=========================================="
python optimize_all_virtual_samples.py

echo ""
echo "=========================================="
echo "✅ 数据准备完成！"
echo "=========================================="
echo "输出文件: augmented_pid_data_optimized.json"
echo "样本数量: 303个"
echo ""
echo "下一步: 运行 ./train_full_pipeline.sh 开始训练"

