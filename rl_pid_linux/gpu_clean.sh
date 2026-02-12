#!/bin/bash
# GPU资源清理脚本
# 在训练失败后运行此脚本来释放GPU资源，避免重启系统

echo "🧹 开始清理GPU资源..."

# 1. 查看当前GPU使用情况
echo ""
echo "📊 当前GPU状态:"
nvidia-smi

# 2. 查找所有Python训练进程
echo ""
echo "🔍 查找Python训练进程..."
PYTHON_PIDS=$(ps aux | grep -E 'train_ppo|train_ddpg|python.*rl_pid' | grep -v grep | awk '{print $2}')

if [ -n "$PYTHON_PIDS" ]; then
    echo "   发现训练进程: $PYTHON_PIDS"
    echo "   正在终止..."
    for pid in $PYTHON_PIDS; do
        sudo kill -9 $pid 2>/dev/null && echo "   ✅ 已终止进程 $pid"
    done
else
    echo "   ✅ 没有发现训练进程"
fi

# 3. 强制清理所有占用GPU的进程（谨慎使用）
echo ""
echo "⚠️  准备强制清理GPU设备文件..."
read -p "   这会终止所有GPU进程（包括桌面环境）。是否继续? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo fuser -k /dev/nvidia* 2>/dev/null
    echo "   ✅ GPU设备文件已清理"
else
    echo "   ⏭️  已跳过强制清理"
fi

# 4. 显示清理后的状态
echo ""
echo "📊 清理后GPU状态:"
nvidia-smi

echo ""
echo "✅ 清理完成！现在可以重新开始训练"

