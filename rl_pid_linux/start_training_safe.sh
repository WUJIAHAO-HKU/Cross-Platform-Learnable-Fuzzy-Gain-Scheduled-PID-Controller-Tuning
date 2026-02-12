#!/bin/bash
# 安全启动训练脚本
# 包含GPU检查和自动清理功能

set -e  # 遇到错误立即退出

echo "🚀 RL+PID 训练启动脚本"
echo "====================================="

# 1. 激活虚拟环境
echo "📦 激活虚拟环境..."
source ~/rl_robot_env/bin/activate
echo "   ✅ 虚拟环境已激活"

# 2. 检查GPU状态
echo ""
echo "🖥️  检查GPU状态..."
python3 << 'EOF'
import torch
print(f"   PyTorch版本: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"   CUDA可用: {'✅ 是' if cuda_available else '❌ 否'}")
if cuda_available:
    print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   ⚠️  将使用CPU模式训练")
EOF

# 3. 显示配置信息
echo ""
echo "📋 训练配置:"
CONFIG_FILE="configs/stage1_optimized.yaml"
MODEL_NAME="ppo_optimized_$(date +%Y%m%d_%H%M%S)"

echo "   配置文件: $CONFIG_FILE"
echo "   模型名称: $MODEL_NAME"
echo "   日志目录: ./logs/"

# 4. 询问是否继续
echo ""
read -p "❓ 是否开始训练? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 已取消训练"
    exit 0
fi

# 5. 清理之前可能残留的资源
echo ""
echo "🧹 预先清理GPU资源..."
python3 << 'EOF'
import torch
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("   ✅ GPU缓存已清空")
else:
    print("   ⏭️  CPU模式，无需清理GPU")
EOF

# 6. 启动训练
echo ""
echo "====================================="
echo "  开始训练"
echo "====================================="
echo ""
echo "💡 提示："
echo "   - 按 Ctrl+C 可安全中断训练"
echo "   - 监控训练: tensorboard --logdir=./logs/tensorboard/"
echo "   - 实时查看GPU: watch -n 2 nvidia-smi"
echo ""
sleep 2

# 启动训练（捕获退出状态）
python training/train_ppo.py \
    --config "$CONFIG_FILE" \
    --name "$MODEL_NAME" \
    --output ./logs

EXIT_CODE=$?

# 7. 训练后清理
echo ""
echo "🧹 训练结束，清理资源..."
python3 << 'EOF'
import torch
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("   ✅ GPU缓存已清空")
EOF

# 8. 显示结果
echo ""
echo "====================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 训练完成！"
    echo ""
    echo "📁 模型保存位置:"
    echo "   - 最终模型: ./logs/${MODEL_NAME}_final.zip"
    echo "   - 最佳模型: ./logs/best_model/best_model.zip"
    echo "   - 检查点: ./logs/checkpoints/"
    echo ""
    echo "📊 查看训练曲线:"
    echo "   tensorboard --logdir=./logs/tensorboard/"
else
    echo "⚠️  训练异常退出（退出码: $EXIT_CODE）"
    echo ""
    echo "📁 可能保存的模型:"
    echo "   - 中断模型: ./logs/${MODEL_NAME}_interrupted.zip"
    echo "   - 错误模型: ./logs/${MODEL_NAME}_error.zip"
    echo ""
    echo "🔧 故障排查:"
    echo "   1. 运行 ./gpu_clean.sh 清理GPU资源"
    echo "   2. 查看日志文件分析错误原因"
    echo "   3. 如果GPU问题持续，考虑重启系统"
fi
echo "====================================="

