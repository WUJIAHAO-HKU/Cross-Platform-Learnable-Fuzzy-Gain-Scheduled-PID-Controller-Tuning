#!/bin/bash
# 🚀 开始RL+PID训练（优化后的PID参数）

echo "======================================================================"
echo "🚀 开始RL+PID训练"
echo "======================================================================"
echo ""
echo "📊 训练配置："
echo "  • PID基线误差: 2.08度（已优化62%）"
echo "  • 预期RL+PID误差: 1.5-1.8度"
echo "  • 训练步数: 500,000步"
echo "  • 并行环境: 4个"
echo "  • 预计时间: 1-2小时"
echo ""
echo "💡 提示："
echo "  • 按Ctrl+C可随时停止训练（会自动保存）"
echo "  • 可以新开终端查看进度："
echo "    cd rl_pid_linux && tensorboard --logdir logs"
echo ""
echo "======================================================================"
echo ""

cd /home/wujiahao/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/rl_pid_linux
source ~/rl_robot_env/bin/activate

python training/train_ppo.py \
    --config configs/stage1_optimized.yaml \
    --name ppo_optimized_pid \
    --output ./logs 2>&1 | tee training_optimized_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "======================================================================"
echo "✅ 训练完成！"
echo "======================================================================"
echo ""
echo "📂 输出文件："
echo "  • 模型: ./logs/ppo_optimized_pid/"
echo "  • 日志: ./training_optimized_*.log"
echo ""
echo "🎯 下一步："
echo "  python evaluate_trained_model.py --gui"
echo ""

