#!/bin/bash
################################################################################
# 🚀 Meta-PID+RL 完整实验快速复现脚本
# 
# 用途：一键运行完整实验流程（从数据生成到结果可视化）
# 时长：约2.5小时
# 作者：吴嘉豪
# 日期：2025-10-31
################################################################################

set -e  # 遇到错误立即停止
set -u  # 使用未定义变量时报错

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 计时函数
start_time=$(date +%s)

print_time() {
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    log_info "⏱️  总耗时: ${hours}h ${minutes}m ${seconds}s"
}

# 错误处理
trap 'log_error "脚本在第 $LINENO 行出错！"; print_time; exit 1' ERR

################################################################################
# 配置参数
################################################################################

# 实验配置
SEED=42
N_EPISODES_TEST=10
N_EPISODES_DISTURBANCE=10

# 数据生成配置
N_FRANKA_VARIANTS=50
N_KUKA_VARIANTS=50
N_LAIKAGO_VARIANTS=50
OPTIMIZATION_WORKERS=4
ERROR_THRESHOLD=30.0

# 元学习配置
META_EPOCHS=1000
META_BATCH_SIZE=32
META_LR=0.001
META_HIDDEN_DIM=256

# RL训练配置
RL_TIMESTEPS=200000
RL_N_ENVS=4
RL_LR=3e-4

# 机器人URDF路径
FRANKA_URDF="franka_panda/panda.urdf"
LAIKAGO_URDF="laikago/laikago.urdf"

# 输出文件路径
DATA_AUGMENTED="augmented_pid_data.json"
DATA_OPTIMIZED="augmented_pid_data_optimized.json"
DATA_FILTERED="augmented_pid_data_filtered.json"
META_MODEL="meta_pid_network.pth"
FRANKA_RL_MODEL="logs/meta_rl_panda/best_model/best_model"
LAIKAGO_RL_MODEL="logs/meta_rl_laikago/best_model/best_model"

################################################################################
# 命令行参数解析
################################################################################

SKIP_DATA_GEN=false
SKIP_META_TRAIN=false
SKIP_RL_TRAIN=false
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data)
            SKIP_DATA_GEN=true
            shift
            ;;
        --skip-meta)
            SKIP_META_TRAIN=true
            shift
            ;;
        --skip-rl)
            SKIP_RL_TRAIN=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            N_EPISODES_TEST=3
            N_EPISODES_DISTURBANCE=3
            RL_TIMESTEPS=50000
            log_warning "快速模式：减少测试回合和训练步数（结果可能不准确）"
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --skip-data    跳过数据生成阶段（使用已有数据）"
            echo "  --skip-meta    跳过元学习训练（使用已有模型）"
            echo "  --skip-rl      跳过RL训练（使用已有模型）"
            echo "  --quick        快速模式（减少训练和测试次数，用于调试）"
            echo "  --help         显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                    # 完整运行所有阶段"
            echo "  $0 --skip-data        # 跳过数据生成，使用已有数据"
            echo "  $0 --quick            # 快速模式（用于测试脚本）"
            exit 0
            ;;
        *)
            log_error "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

################################################################################
# 环境检查
################################################################################

log_info "========================================="
log_info "阶段0: 环境检查与准备"
log_info "========================================="

# 检查Python
if ! command -v python &> /dev/null; then
    log_error "Python未安装或不在PATH中"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
log_success "Python版本: $PYTHON_VERSION"

# 检查必需的Python包
log_info "检查Python依赖..."
python -c "
import sys
required_packages = ['torch', 'pybullet', 'stable_baselines3', 'numpy', 'matplotlib', 'scipy']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print('缺少以下Python包:', ', '.join(missing))
    sys.exit(1)
print('✅ 所有依赖包已安装')
"

log_success "依赖检查通过"

# 创建必要的目录
mkdir -p logs/meta_rl_panda/best_model
mkdir -p logs/meta_rl_laikago/best_model
mkdir -p results
mkdir -p figures

log_success "目录结构已创建"

################################################################################
# 阶段1: 数据生成
################################################################################

if [ "$SKIP_DATA_GEN" = true ]; then
    log_warning "跳过数据生成阶段（使用已有数据）"
    
    # 检查数据文件是否存在
    if [ ! -f "$DATA_FILTERED" ]; then
        log_error "数据文件不存在: $DATA_FILTERED"
        log_error "请先运行完整流程或移除 --skip-data 选项"
        exit 1
    fi
    
    log_success "数据文件已存在: $DATA_FILTERED"
else
    log_info "========================================="
    log_info "阶段1: 数据生成（预计60分钟）"
    log_info "========================================="
    
    stage1_start=$(date +%s)
    
    # 1.1 生成基础虚拟样本
    log_info "步骤1.1: 生成虚拟样本（启发式估计）"
    python data_augmentation.py \
        --n_variants $N_FRANKA_VARIANTS $N_KUKA_VARIANTS $N_LAIKAGO_VARIANTS \
        --output $DATA_AUGMENTED \
        --seed $SEED
    
    log_success "虚拟样本生成完成: $DATA_AUGMENTED"
    
    # 1.2 混合优化
    log_info "步骤1.2: 混合优化获得最优PID（DE + Nelder-Mead）"
    python optimize_all_virtual_samples.py \
        --input $DATA_AUGMENTED \
        --output $DATA_OPTIMIZED \
        --method hybrid \
        --workers $OPTIMIZATION_WORKERS \
        --seed $SEED
    
    log_success "PID优化完成: $DATA_OPTIMIZED"
    
    # 1.3 数据过滤
    log_info "步骤1.3: 过滤不可控样本"
    python filter_samples.py \
        --input $DATA_OPTIMIZED \
        --output $DATA_FILTERED \
        --error_threshold $ERROR_THRESHOLD
    
    # 统计样本数量
    SAMPLE_COUNT=$(python -c "import json; print(len(json.load(open('$DATA_FILTERED'))))")
    log_success "数据过滤完成: $DATA_FILTERED (保留 $SAMPLE_COUNT 个样本)"
    
    stage1_end=$(date +%s)
    stage1_duration=$((stage1_end - stage1_start))
    log_success "阶段1完成，耗时: $((stage1_duration / 60))分钟"
fi

################################################################################
# 阶段2: 元学习训练
################################################################################

if [ "$SKIP_META_TRAIN" = true ]; then
    log_warning "跳过元学习训练（使用已有模型）"
    
    if [ ! -f "$META_MODEL" ]; then
        log_error "元学习模型不存在: $META_MODEL"
        exit 1
    fi
    
    log_success "元学习模型已存在: $META_MODEL"
else
    log_info "========================================="
    log_info "阶段2: 元学习训练（预计8分钟）"
    log_info "========================================="
    
    stage2_start=$(date +%s)
    
    python train_meta_learning.py \
        --data $DATA_FILTERED \
        --model_save_path $META_MODEL \
        --epochs $META_EPOCHS \
        --batch_size $META_BATCH_SIZE \
        --lr $META_LR \
        --hidden_dim $META_HIDDEN_DIM \
        --seed $SEED
    
    log_success "元学习模型训练完成: $META_MODEL"
    
    stage2_end=$(date +%s)
    stage2_duration=$((stage2_end - stage2_start))
    log_success "阶段2完成，耗时: $((stage2_duration / 60))分钟"
fi

################################################################################
# 阶段3: RL训练
################################################################################

if [ "$SKIP_RL_TRAIN" = true ]; then
    log_warning "跳过RL训练（使用已有模型）"
    
    if [ ! -f "${FRANKA_RL_MODEL}.zip" ]; then
        log_error "Franka RL模型不存在: ${FRANKA_RL_MODEL}.zip"
        exit 1
    fi
    
    log_success "RL模型已存在"
else
    log_info "========================================="
    log_info "阶段3: RL训练（预计40分钟）"
    log_info "========================================="
    
    stage3_start=$(date +%s)
    
    # 3.1 训练Franka Panda
    log_info "步骤3.1: 训练Franka Panda的RL策略"
    python train_ppo_with_meta_pid.py \
        --robot $FRANKA_URDF \
        --meta_model $META_MODEL \
        --total_timesteps $RL_TIMESTEPS \
        --n_envs $RL_N_ENVS \
        --learning_rate $RL_LR \
        --tensorboard_log logs/meta_rl_panda \
        --save_path $FRANKA_RL_MODEL \
        --seed $SEED
    
    log_success "Franka Panda RL训练完成"
    
    # 3.2 训练Laikago（可选）
    log_info "步骤3.2: 训练Laikago的RL策略"
    python train_ppo_with_meta_pid.py \
        --robot $LAIKAGO_URDF \
        --meta_model $META_MODEL \
        --total_timesteps $RL_TIMESTEPS \
        --n_envs $RL_N_ENVS \
        --learning_rate $RL_LR \
        --tensorboard_log logs/meta_rl_laikago \
        --save_path $LAIKAGO_RL_MODEL \
        --seed $SEED
    
    log_success "Laikago RL训练完成"
    
    stage3_end=$(date +%s)
    stage3_duration=$((stage3_end - stage3_start))
    log_success "阶段3完成，耗时: $((stage3_duration / 60))分钟"
fi

################################################################################
# 阶段4: 跨平台性能测试
################################################################################

log_info "========================================="
log_info "阶段4: 跨平台性能测试（预计10分钟）"
log_info "========================================="

stage4_start=$(date +%s)

python test_cross_platform.py \
    --robots $FRANKA_URDF $LAIKAGO_URDF \
    --meta_model $META_MODEL \
    --rl_models $FRANKA_RL_MODEL $LAIKAGO_RL_MODEL \
    --n_episodes $N_EPISODES_TEST \
    --max_steps 5000 \
    --seed $SEED

log_success "跨平台测试完成"

stage4_end=$(date +%s)
stage4_duration=$((stage4_end - stage4_start))
log_success "阶段4完成，耗时: $((stage4_duration / 60))分钟"

################################################################################
# 阶段5: 扰动场景鲁棒性测试
################################################################################

log_info "========================================="
log_info "阶段5: 扰动场景测试（预计30分钟）"
log_info "========================================="

stage5_start=$(date +%s)

python test_disturbance_scenarios.py \
    --robot $FRANKA_URDF \
    --model $FRANKA_RL_MODEL \
    --n_episodes $N_EPISODES_DISTURBANCE \
    --seed $SEED

log_success "扰动场景测试完成"

stage5_end=$(date +%s)
stage5_duration=$((stage5_end - stage5_start))
log_success "阶段5完成，耗时: $((stage5_duration / 60))分钟"

################################################################################
# 阶段6: 高级可视化
################################################################################

log_info "========================================="
log_info "阶段6: 生成高级可视化（预计5分钟）"
log_info "========================================="

stage6_start=$(date +%s)

# 6.1 RL训练仪表盘
log_info "生成RL训练动态仪表盘..."
python generate_advanced_visualizations.py \
    --type rl_dashboard \
    --tensorboard_log logs/meta_rl_panda \
    --output figures/rl_training_dashboard.png

# 6.2 逐关节误差对比
log_info "生成逐关节误差对比图..."
python generate_per_joint_error_comparison.py \
    --results cross_platform_results.json \
    --output figures/per_joint_error.png

# 6.3 特征相关性热图
log_info "生成特征相关性热图..."
python generate_feature_correlation_heatmap.py \
    --data $DATA_FILTERED \
    --output figures/feature_correlation_heatmap.png

log_success "所有可视化图表已生成"

stage6_end=$(date +%s)
stage6_duration=$((stage6_end - stage6_start))
log_success "阶段6完成，耗时: $((stage6_duration / 60))分钟"

################################################################################
# 实验总结
################################################################################

log_info "========================================="
log_info "📊 实验总结"
log_info "========================================="

# 打印总耗时
print_time

# 打印关键结果
log_info ""
log_info "📁 生成的文件:"
log_info "  数据文件: $DATA_FILTERED"
log_info "  元学习模型: $META_MODEL"
log_info "  Franka RL模型: ${FRANKA_RL_MODEL}.zip"
log_info "  Laikago RL模型: ${LAIKAGO_RL_MODEL}.zip"
log_info "  结果数据: cross_platform_results.json"
log_info "  扰动测试日志: disturbance_test.log"
log_info ""

log_info "📊 生成的图表:"
log_info "  训练曲线: training_curve_augmented.png"
log_info "  跟踪误差对比: actual_tracking_comparison.png"
log_info "  Meta vs RL对比: meta_rl_comparison.png"
log_info "  扰动场景对比: disturbance_comparison.png"
log_info "  RL训练仪表盘: figures/rl_training_dashboard.png"
log_info "  逐关节误差: figures/per_joint_error.png"
log_info "  特征热图: figures/feature_correlation_heatmap.png"
log_info ""

# 提取关键性能指标
log_info "🎯 关键性能指标:"
python -c "
import json

# 读取跨平台测试结果
try:
    with open('cross_platform_results.json') as f:
        results = json.load(f)
    
    print('  跨平台性能:')
    for robot, data in results.items():
        meta_mae = data['pure_meta_pid']['mean_error_deg']
        rl_mae = data['meta_pid_rl']['mean_error_deg']
        improvement = (meta_mae - rl_mae) / meta_mae * 100
        print(f'    {robot}: {improvement:+.2f}% 改善')
except:
    print('  跨平台结果文件未找到')

# 读取扰动测试结果
try:
    with open('disturbance_test.log') as f:
        lines = f.readlines()
        # 查找平均改善
        for line in lines:
            if '平均改善' in line:
                print(f'  扰动鲁棒性: {line.strip()}')
                break
except:
    print('  扰动测试日志未找到')
"

log_info ""
log_success "✅ 所有实验已完成！"
log_info ""
log_info "📖 查看完整结果："
log_info "  详细报告: 完整实验复现流程.md"
log_info "  快速查看: cat cross_platform_results.json | python -m json.tool"
log_info "  TensorBoard: tensorboard --logdir logs/"
log_info ""
log_info "🚀 下一步："
log_info "  1. 检查所有图表是否正确生成"
log_info "  2. 将图表上传到Overleaf论文项目"
log_info "  3. 更新论文中的数值结果"
log_info "  4. 备份所有模型和数据文件"
log_info ""

################################################################################
# 清理与备份（可选）
################################################################################

read -p "是否备份所有结果到results目录？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "备份结果..."
    
    BACKUP_DIR="results/experiment_$(date +%Y%m%d_%H%M%S)"
    mkdir -p $BACKUP_DIR
    
    # 复制关键文件
    cp $DATA_FILTERED $BACKUP_DIR/
    cp $META_MODEL $BACKUP_DIR/
    cp *.png $BACKUP_DIR/ 2>/dev/null || true
    cp *.json $BACKUP_DIR/ 2>/dev/null || true
    cp *.log $BACKUP_DIR/ 2>/dev/null || true
    
    # 复制RL模型（较大）
    cp -r logs/ $BACKUP_DIR/ 2>/dev/null || true
    
    log_success "结果已备份到: $BACKUP_DIR"
fi

log_success "🎉 实验流程全部完成！"

