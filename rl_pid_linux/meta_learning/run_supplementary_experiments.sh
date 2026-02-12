#!/bin/bash
# 补充实验综合执行脚本
# 包括：训练曲线可视化、扰动场景测试、Laikago验证

set -e

export META_DIR="/home/wujiahao/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/rl_pid_linux/meta_learning"
cd $META_DIR
source ~/rl_robot_env/bin/activate

echo "════════════════════════════════════════════════════════════════════════════════"
echo "                     🔬 补充实验综合测试"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "本脚本将依次执行："
echo "  1️⃣  训练曲线可视化（约30秒）"
echo "  2️⃣  扰动场景鲁棒性测试（约30-60分钟）"
echo "  3️⃣  Laikago四足机器人验证（约10分钟）"
echo ""
read -p "是否继续？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "❌ 取消执行"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "实验1: 训练曲线可视化"
echo "预计时间: 30秒"
echo "════════════════════════════════════════════════════════════════════════════════"
python visualize_training_curves.py

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "实验2: 扰动场景鲁棒性测试（快速模式：每种扰动5回合）"
echo "预计时间: 约15分钟"
echo "════════════════════════════════════════════════════════════════════════════════"
python evaluate_robustness.py \
  --robot franka_panda/panda.urdf \
  --model logs/meta_rl_panda/best_model/best_model \
  --disturbances none random_force payload param_uncertainty \
  --n_episodes 5 \
  --max_steps 3000

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "实验3: Laikago四足机器人验证（可选，按Ctrl+C跳过）"
echo "预计时间: 约10分钟"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "⚠️  注意：此实验需要Laikago机器人的预训练模型"
echo "如果没有，请按Ctrl+C跳过，或先运行："
echo "  python train_meta_rl_combined.py laikago/laikago.urdf 200000 4"
echo ""
sleep 3

# 检查Laikago模型是否存在
if [ -f "logs/meta_rl_laikago/best_model/best_model.zip" ]; then
    echo "✅ 找到Laikago模型，开始验证..."
    python verify_actual_tracking_error.py --robot laikago/laikago.urdf \
      --model logs/meta_rl_laikago/best_model/best_model
else
    echo "⚠️  未找到Laikago预训练模型，跳过此实验"
    echo "💡 如需执行，请先运行："
    echo "   python train_meta_rl_combined.py laikago/laikago.urdf 200000 4"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ 补充实验完成！"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "生成的文件："
echo "  📊 training_curves.png              - 训练曲线可视化"
echo "  📊 robustness_comparison.png        - 扰动场景性能对比"
echo "  📊 actual_tracking_comparison.png   - 实际跟踪误差对比（已存在）"
echo ""
echo "这些图表可用于论文的实验章节！"

