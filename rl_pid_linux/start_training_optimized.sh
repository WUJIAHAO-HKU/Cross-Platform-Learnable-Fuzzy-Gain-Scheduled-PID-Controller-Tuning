#!/bin/bash
# 启动优化PID的RL训练

cd /home/wujiahao/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/rl_pid_linux
source ~/rl_robot_env/bin/activate

echo "======================================================================="
echo "开始RL+PID训练（使用优化后的PID参数）"
echo "======================================================================="
echo ""
echo "配置："
echo "  - PID基线误差：2.08度"
echo "  - 预期RL+PID误差：1.5-1.8度"
echo "  - 训练步数：500,000步"
echo "  - 并行环境：4个"
echo "  - 预计时间：1-2小时（CPU）"
echo ""
echo "训练过程中可以："
echo "  1. 按 Ctrl+C 停止训练"
echo "  2. 打开新终端运行 'tensorboard --logdir logs' 查看进度"
echo ""
echo "======================================================================="
echo ""

python training/train_ppo.py \
    --config configs/stage1_optimized.yaml \
    --name ppo_optimized_pid \
    --output ./logs 2>&1 | tee training_optimized.log

echo ""
echo "======================================================================="
echo "训练完成！"
echo "======================================================================="
echo ""
echo "查看结果："
echo "  1. 模型保存在：./logs/ppo_optimized_pid/"
echo "  2. 训练日志：./training_optimized.log"
echo "  3. TensorBoard：tensorboard --logdir logs"
echo ""
echo "下一步："
echo "  python evaluate_trained_model.py --model logs/ppo_optimized_pid/best_model.zip --gui"
echo ""


