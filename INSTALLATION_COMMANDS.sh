#!/bin/bash
# ============================================================================
# RL+PID Linux环境搭建脚本
# 用途：一键安装所有依赖
# 使用：bash INSTALLATION_COMMANDS.sh
# 预计时间：10-15分钟
# ============================================================================

set -e  # 遇到错误立即停止

echo "================================================================"
echo "  RL+PID Linux环境搭建"
echo "  预计时间：10-15分钟"
echo "================================================================"
echo ""

# 进入项目目录
cd ~/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/

# ============================================================================
# 第1步：创建Conda环境（2分钟）
# ============================================================================
echo ">>> [1/5] 创建Conda环境 'rl_robot'..."
if conda env list | grep -q "rl_robot"; then
    echo "    环境已存在，跳过创建"
else
    conda create -n rl_robot python=3.8 -y
    echo "    ✅ 环境创建成功"
fi

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rl_robot
echo "    ✅ 环境已激活"
echo ""

# ============================================================================
# 第2步：安装PyTorch（3-5分钟）
# ============================================================================
echo ">>> [2/5] 安装PyTorch..."
# 检查是否已安装
if python -c "import torch" 2>/dev/null; then
    echo "    PyTorch已安装，跳过"
else
    # CPU版本（如果有CUDA，可以改用GPU版本）
    pip install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo "    ✅ PyTorch安装成功"
fi
echo ""

# ============================================================================
# 第3步：安装强化学习库（2-3分钟）
# ============================================================================
echo ">>> [3/5] 安装强化学习库..."
pip install pybullet==3.2.5
pip install gym==0.21.0
pip install stable-baselines3==1.7.0
echo "    ✅ RL库安装成功"
echo ""

# ============================================================================
# 第4步：安装科学计算库（1-2分钟）
# ============================================================================
echo ">>> [4/5] 安装科学计算库..."
pip install numpy==1.23.5
pip install scipy==1.10.1
pip install matplotlib==3.7.1
pip install pandas==2.0.1
pip install seaborn==0.12.2
pip install scikit-learn==1.2.2
echo "    ✅ 科学计算库安装成功"
echo ""

# ============================================================================
# 第5步：安装工具库（1分钟）
# ============================================================================
echo ">>> [5/5] 安装工具库..."
pip install pyyaml==6.0
pip install tensorboard==2.13.0
pip install imageio==2.31.1
pip install opencv-python==4.7.0.72
pip install tqdm==4.65.0
echo "    ✅ 工具库安装成功"
echo ""

# ============================================================================
# 保存依赖列表
# ============================================================================
echo ">>> 保存依赖列表到 requirements.txt..."
pip freeze > requirements.txt
echo "    ✅ 已保存到 requirements.txt"
echo ""

# ============================================================================
# 验证安装
# ============================================================================
echo "================================================================"
echo "  验证安装"
echo "================================================================"

echo ">>> 检查关键库..."
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import pybullet as p; print('✅ PyBullet:', p.getVersionInfo())"
python -c "import gym; print('✅ Gym:', gym.__version__)"
python -c "from stable_baselines3 import DDPG; print('✅ Stable-Baselines3: OK')"
python -c "import numpy; print('✅ NumPy:', numpy.__version__)"
python -c "import matplotlib; print('✅ Matplotlib:', matplotlib.__version__)"
echo ""

# ============================================================================
# 创建项目结构
# ============================================================================
echo ">>> 创建项目目录结构..."
mkdir -p rl_pid_linux/{configs,envs,controllers,training,evaluation,visualization,models,logs,figures,tables,tests,data}

# 创建__init__.py文件
touch rl_pid_linux/envs/__init__.py
touch rl_pid_linux/controllers/__init__.py
touch rl_pid_linux/training/__init__.py
touch rl_pid_linux/evaluation/__init__.py
touch rl_pid_linux/visualization/__init__.py

echo "    ✅ 项目结构创建完成"
echo ""

# ============================================================================
# 记录安装信息
# ============================================================================
echo ">>> 记录安装信息..."
cat > rl_pid_linux/INSTALLATION_LOG.txt << EOF
=== RL+PID Linux环境安装记录 ===
安装时间: $(date)
Python版本: $(python --version)
Conda环境: rl_robot

已安装的关键库：
- PyTorch: $(python -c "import torch; print(torch.__version__)")
- PyBullet: $(python -c "import pybullet as p; print(p.getVersionInfo())")
- Gym: $(python -c "import gym; print(gym.__version__)")
- Stable-Baselines3: $(python -c "import stable_baselines3; print(stable_baselines3.__version__)")
- NumPy: $(python -c "import numpy; print(numpy.__version__)")

系统信息：
$(uname -a)

下一步：
1. 运行测试：cd rl_pid_linux && python tests/test_pybullet_franka.py
2. 开始训练：python training/train_ddpg.py --config configs/stage1_small.yaml
EOF

echo "    ✅ 安装信息已保存到 rl_pid_linux/INSTALLATION_LOG.txt"
echo ""

# ============================================================================
# 完成
# ============================================================================
echo "================================================================"
echo "  🎉 环境搭建完成！"
echo "================================================================"
echo ""
echo "接下来："
echo "  1. 查看安装日志："
echo "     cat rl_pid_linux/INSTALLATION_LOG.txt"
echo ""
echo "  2. 测试PyBullet（我会生成测试脚本）："
echo "     cd rl_pid_linux"
echo "     python tests/test_pybullet_franka.py"
echo ""
echo "  3. 开始算法移植："
echo "     # 我会生成所有核心代码文件"
echo ""
echo "激活环境的命令："
echo "  conda activate rl_robot"
echo ""
echo "================================================================"

