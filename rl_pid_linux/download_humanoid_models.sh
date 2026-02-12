#!/bin/bash
# 下载开源人形机器人模型

echo "========================================================================"
echo "下载开源人形/四足机器人模型"
echo "========================================================================"

# 创建模型目录
mkdir -p robots
cd robots

echo ""
echo "1. 下载Unitree机器人（人形H1 + 四足Go1）..."
if [ ! -d "unitree_mujoco" ]; then
    git clone https://github.com/unitreerobotics/unitree_mujoco.git
    echo "   ✅ Unitree模型已下载"
else
    echo "   ⚠️  目录已存在，跳过"
fi

echo ""
echo "2. 下载Robot Descriptions（包含多种机器人）..."
if [ ! -d "robot_descriptions.py" ]; then
    git clone https://github.com/robot-descriptions/robot_descriptions.py.git
    echo "   ✅ Robot Descriptions已下载"
else
    echo "   ⚠️  目录已存在，跳过"
fi

echo ""
echo "3. 下载iCub人形机器人..."
if [ ! -d "icub-models" ]; then
    git clone https://github.com/robotology/icub-models.git
    echo "   ✅ iCub模型已下载"
else
    echo "   ⚠️  目录已存在，跳过"
fi

echo ""
echo "========================================================================"
echo "下载完成！"
echo "========================================================================"
echo ""
echo "可用的人形/四足机器人："
echo "  - Unitree H1 (人形, 29 DOF): robots/unitree_mujoco/unitree_h1/"
echo "  - Unitree G1 (人形, 25 DOF): robots/unitree_mujoco/unitree_g1/"
echo "  - Unitree Go1 (四足, 12 DOF): robots/unitree_mujoco/unitree_go1/"
echo "  - iCub (人形, 53 DOF): robots/icub-models/"
echo ""
echo "下一步："
echo "  python test_downloaded_robots.py"

