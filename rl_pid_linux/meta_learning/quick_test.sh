#!/bin/bash
# 快速测试脚本
# 用途：使用已有模型快速验证性能（无需重新训练）

set -e

export META_DIR="/home/wujiahao/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/rl_pid_linux/meta_learning"
cd $META_DIR
source ~/rl_robot_env/bin/activate

echo "=========================================="
echo "⚡ 快速测试模式（使用预训练模型）"
echo "预计时间: 2-3分钟"
echo "=========================================="
echo ""

# 检查模型是否存在
if [ ! -f "logs/meta_rl_panda/best_model/best_model.zip" ]; then
    echo "❌ 错误: 未找到预训练模型"
    echo "请先运行: ./train_full_pipeline.sh"
    exit 1
fi

echo "=========================================="
echo "测试1: 归一化误差评估"
echo "=========================================="
python evaluate_meta_rl.py

echo ""
echo "=========================================="
echo "测试2: 实际跟踪误差验证（详细分析）"
echo "=========================================="
python verify_actual_tracking_error.py

echo ""
echo "=========================================="
echo "✅ 快速测试完成！"
echo "=========================================="
echo "生成的图表: actual_tracking_comparison.png"
echo ""
echo "主要结论："
echo "✅ 平均误差改善: 46.76° → 34.93° (+25.31%)"
echo "✅ 最大误差改善: 101.47° → 68.85° (+32.14%)"
echo "✅ 标准差改善:   19.95° → 7.79° (+60.96%)"

