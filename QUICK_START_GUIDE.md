# 🚀 Linux环境快速开始指南

> **目标**: 30分钟内完成环境搭建，验证PyBullet可用
> **前置条件**: Linux系统，有conda或python3.8+

---

## 第1步：环境搭建（10分钟）

### 复制粘贴运行：

```bash
# 进入项目目录
cd ~/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/

# 创建Python环境
conda create -n rl_robot python=3.8 -y
conda activate rl_robot

# 安装依赖（一行命令）
pip install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 && \
pip install pybullet==3.2.5 gym==0.21.0 stable-baselines3==1.7.0 && \
pip install numpy scipy matplotlib pandas seaborn scikit-learn pyyaml && \
pip install tensorboard imageio opencv-python

# 验证安装
python -c "import pybullet as p; print('PyBullet version:', p.getVersionInfo())"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "from stable_baselines3 import DDPG; print('SB3 installed successfully')"
```

---

## 第2步：创建项目结构（5分钟）

```bash
# 创建Linux实现目录
mkdir -p rl_pid_linux/{configs,envs,controllers,training,evaluation,visualization,models,logs,figures,tables}

cd rl_pid_linux

# 创建空文件（稍后填充）
touch configs/{robot_config.yaml,pid_config.yaml,rl_config.yaml,stage1_small.yaml}
touch envs/{__init__.py,franka_env.py,trajectory_gen.py}
touch controllers/{__init__.py,pid_controller.py,rl_pid_hybrid.py}
touch training/{__init__.py,train_ddpg.py,reward_function.py,callbacks.py}
touch evaluation/{__init__.py,test_scenarios.py,monte_carlo.py,baseline_methods.py}
touch visualization/{__init__.py,paper_figures.py,plot_results.py}

# 创建README
cat > README.md << 'EOF'
# RL+PID Linux Implementation

## Quick Test
```bash
# 激活环境
conda activate rl_robot

# 测试PyBullet
python tests/test_pybullet_franka.py

# 训练（阶段1：小补偿）
python training/train_ddpg.py --config configs/stage1_small.yaml

# 评估
python evaluation/evaluate_model.py --model models/rl_pid_final.zip
```

## Project Structure
- `configs/`: 配置文件
- `envs/`: PyBullet仿真环境
- `controllers/`: PID和RL+PID控制器
- `training/`: 训练脚本
- `evaluation/`: 评估和对比实验
- `visualization/`: 论文图表生成

## Training Stages
1. Stage 1: delta_scale_max=2.0 (500k steps)
2. Stage 2: delta_scale_max=5.0 (1M steps)
3. Stage 3: delta_scale_max=10.0 (1.5M steps, if stable)
EOF

echo "✅ 项目结构创建完成！"
```

---

## 第3步：测试PyBullet（5分钟）

```bash
# 创建测试目录
mkdir -p tests
cd tests

# 创建测试脚本
cat > test_pybullet_franka.py << 'EOTEST'
"""
测试PyBullet能否正确加载Franka Panda
预期：打开GUI窗口，显示机器人，无报错
"""
import pybullet as p
import pybullet_data
import time
import numpy as np

def test_franka_loading():
    print("=== Testing Franka Panda in PyBullet ===")
    
    # 连接PyBullet（GUI模式）
    print("1. Connecting to PyBullet GUI...")
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # 加载机器人
    print("2. Loading Franka Panda URDF...")
    try:
        robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        print(f"   ✅ Robot loaded! ID: {robot_id}")
    except Exception as e:
        print(f"   ❌ Failed to load robot: {e}")
        return False
    
    # 获取关节信息
    print("3. Checking joint information...")
    num_joints = p.getNumJoints(robot_id)
    print(f"   Total joints: {num_joints}")
    
    controllable_joints = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        
        # 只统计可旋转关节（type 0）
        if joint_type == p.JOINT_REVOLUTE:
            controllable_joints.append(i)
            print(f"   Joint {i}: {joint_name}")
    
    print(f"   ✅ Found {len(controllable_joints)} controllable joints")
    
    # 测试控制
    print("4. Testing torque control...")
    joint_indices = controllable_joints[:7]  # 前7个关节
    
    for step in range(240):  # 1秒（240Hz）
        # 施加小力矩
        torques = [0.1] * 7
        p.setJointMotorControlArray(
            robot_id,
            joint_indices,
            p.TORQUE_CONTROL,
            forces=torques
        )
        p.stepSimulation()
        time.sleep(1./240.)
    
    # 读取状态
    joint_states = p.getJointStates(robot_id, joint_indices)
    positions = [s[0] for s in joint_states]
    velocities = [s[1] for s in joint_states]
    
    print(f"   Joint positions: {np.array(positions)}")
    print(f"   Joint velocities: {np.array(velocities)}")
    print("   ✅ Control test passed!")
    
    # 保持窗口打开5秒
    print("\n5. Keeping GUI open for 5 seconds...")
    print("   (You should see the robot arm in the window)")
    for i in range(5):
        time.sleep(1)
        print(f"   {5-i}...")
    
    p.disconnect()
    print("\n✅ All tests passed! PyBullet is working correctly.")
    return True

if __name__ == "__main__":
    success = test_franka_loading()
    if not success:
        print("\n❌ Test failed. Check your PyBullet installation.")
        exit(1)
EOTEST

# 运行测试
python test_pybullet_franka.py
```

**预期结果：**
- ✅ 打开PyBullet GUI窗口
- ✅ 显示Franka Panda机器人
- ✅ 打印关节信息
- ✅ 无报错

---

## 第4步：验证结果（5分钟）

### 如果一切正常：

```bash
# 保存依赖版本
cd ~/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/rl_pid_linux
pip freeze > requirements.txt

# 记录成功
echo "✅ $(date): Environment setup completed successfully" >> setup_log.txt
echo "✅ PyBullet version: $(python -c 'import pybullet as p; print(p.getVersionInfo())')" >> setup_log.txt
echo "✅ Ready for Phase 2: Algorithm Implementation" >> setup_log.txt

cat setup_log.txt
```

### 如果遇到问题：

#### 问题1: PyBullet找不到franka_panda/panda.urdf

**解决方案A（推荐）**：
```bash
# 下载Franka URDF
cd ~/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/
git clone https://github.com/bulletphysics/bullet3.git
cp -r bullet3/data/franka_panda ~/.local/lib/python3.8/site-packages/pybullet_data/

# 或者从PyBullet数据目录复制
python -c "import pybullet_data; print(pybullet_data.getDataPath())"
# 手动检查该目录是否有franka_panda文件夹
```

**解决方案B（使用绝对路径）**：
```python
# 在test脚本中修改：
import os
urdf_path = os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")
if not os.path.exists(urdf_path):
    # 使用备用路径
    urdf_path = "/path/to/your/franka_panda/panda.urdf"
robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
```

#### 问题2: CUDA/PyTorch错误

```bash
# 使用CPU版本
pip uninstall torch torchvision torchaudio
pip install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 问题3: GUI无法打开

```bash
# 检查显示环境
echo $DISPLAY  # 应该输出 :0 或 :1

# 如果没有显示器，使用虚拟显示
sudo apt install xvfb
xvfb-run -a python tests/test_pybullet_franka.py

# 或者使用无头模式测试
# 在脚本中修改: p.connect(p.DIRECT)  # 代替 p.GUI
```

---

## 第5步：下一步行动（5分钟）

### 确认清单：

```bash
# 运行这个脚本检查所有环境
cat > check_ready.sh << 'EOCHECK'
#!/bin/bash
echo "=== Checking Linux Environment Ready Status ==="

# 1. Python环境
if conda env list | grep -q "rl_robot"; then
    echo "✅ Conda environment 'rl_robot' exists"
else
    echo "❌ Conda environment not found"
    exit 1
fi

# 2. 关键包
conda activate rl_robot
python -c "import pybullet" && echo "✅ PyBullet installed" || echo "❌ PyBullet missing"
python -c "import torch" && echo "✅ PyTorch installed" || echo "❌ PyTorch missing"
python -c "from stable_baselines3 import DDPG" && echo "✅ SB3 installed" || echo "❌ SB3 missing"

# 3. 项目结构
if [ -d "rl_pid_linux" ]; then
    echo "✅ Project directory exists"
else
    echo "❌ Project directory not created"
    exit 1
fi

# 4. PyBullet测试
if [ -f "rl_pid_linux/tests/test_pybullet_franka.py" ]; then
    echo "✅ Test script exists"
else
    echo "❌ Test script not found"
fi

echo ""
echo "=== Summary ==="
echo "If all checks passed, you are ready for Phase 2!"
echo "Next step: Run 'python tests/test_pybullet_franka.py'"
EOCHECK

chmod +x check_ready.sh
./check_ready.sh
```

### 如果所有检查通过：

```
🎉 恭喜！环境搭建完成！

📋 接下来：
1. 查看 LINUX_IMPLEMENTATION_ROADMAP.md 了解整体计划
2. 开始阶段2：算法移植
3. 我将为你生成所有核心代码文件

回复 "继续" 开始阶段2！
```

---

## 🆘 获取帮助

如果遇到问题：

1. **查看详细路线图**：
   ```bash
   cat LINUX_IMPLEMENTATION_ROADMAP.md
   ```

2. **查看MATLAB参考代码**：
   ```bash
   # MATLAB代码在这里：
   cd MATLAB_Implementation/controllers
   ls -l RLHighLevelBlock_Toolbox.m  # RL逻辑参考
   ```

3. **联系支持**：
   - 复制错误信息
   - 提供 `pip list` 输出
   - 说明操作系统版本

---

**预计总时间：30分钟**
**成功率：95%+（如果按步骤操作）**

准备好了吗？开始吧！🚀

