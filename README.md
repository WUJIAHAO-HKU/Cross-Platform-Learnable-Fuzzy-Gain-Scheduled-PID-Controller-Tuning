# Cross-Platform Learnable Fuzzy PID Controller via Meta-RL

**Status:** 🎉 **Published in Robotica (Cambridge University Press)** 🎉

[![Paper](https://img.shields.io/badge/Paper-Robotica-blue)](https://arxiv.org/pdf/2511.06500)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

## 📄 Publication Information

**Title:** Cross-Platform Learnable Fuzzy Gain-Scheduled Proportional-Integral-Derivative Controller Tuning via Physics-Constrained Meta-Learning and Reinforcement Learning Adaptation

**Journal:** Robotica (Cambridge University Press)  
**Submission Date:** January 14, 2026  
**Manuscript ID:** ROB-2026-0021  
**arXiv:** [2511.06500](https://arxiv.org/pdf/2511.06500)

**Authors:**
- Jiahao Wu (The University of Hong Kong) - Corresponding Author
- KaHo NG (The University of Hong Kong)
- Shengwen Yu (Guangzhou College of Commerce)

## 🚀 Overview

This repository contains the implementation of a novel hierarchical meta-reinforcement learning framework for automated PID controller tuning across heterogeneous robotic platforms. Our method achieves:

- **80.4% error reduction** on challenging high-load joints
- **19.2% improvement** under parameter uncertainty
- **Cross-platform generalization** (9-DOF manipulator + 12-DOF quadruped)
- **10-minute training** per platform on standard CPU

### Key Innovation: Physics-Constrained Data Augmentation

We generate 232 physically valid robot variants from only 3 base platforms through bounded parameter perturbations, enabling data-efficient meta-learning while maintaining physical plausibility.

## 📂 Repository Structure

```
rl_pid_linux/
├── meta_learning/              # Core meta-learning implementation
│   ├── train_meta_pid.py      # Meta-network training
│   ├── data_augmentation.py   # Physics-based augmentation
│   ├── meta_pid_optimizer.py  # Hybrid DE+Nelder-Mead optimizer
│   └── evaluate_meta_rl.py    # Cross-platform evaluation
├── controllers/               # Controller implementations  
│   ├── pid_controller.py      # Base PID controller
│   └── rl_pid_hybrid.py       # RL-PID hybrid controller
├── envs/                      # Simulation environments
├── training/                  # RL training scripts
│   ├── train_ppo.py          # PPO-based adaptation
│   └── train_ddpg.py         # DDPG baseline
└── tests/                     # Unit tests

submit_mateials/              # Camera-ready manuscript
└── meta_rl_pid_control_manuscript.tex

docs/                         # Documentation (archived)
```

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/WUJIAHAO-HKU/RL_Pid_Meta-Learning_Based-Data-Augmentation.git
cd RL_Pid_Meta-Learning_Based-Data-Augmentation

# Create conda environment
conda create -n meta_rl_pid python=3.8
conda activate meta_rl_pid

# Install dependencies
pip install -r requirements.txt

# Install PyBullet for physics simulation
pip install pybullet==3.2.5
```

## 🎯 Quick Start

### 1. Generate Augmented Training Data
```bash
cd rl_pid_linux/meta_learning
python data_augmentation.py --base_robots franka kuka laikago --samples_per_robot 100
```

### 2. Train Meta-Learning Network
```bash
python train_meta_pid.py --data augmented_pid_data_filtered.json --epochs 500
```

### 3. RL Adaptation (Optional)
```bash
cd ../training
python train_ppo.py --robot franka --meta_init ../meta_learning/meta_pid_augmented.pth --timesteps 1000000
```

### 4. Evaluate Cross-Platform Performance
```bash
cd ../meta_learning
python evaluate_meta_rl.py --robot franka --robot laikago --seeds 100
```

## 📊 Key Results

| Platform | Metric | Meta-PID | Meta-PID+RL | Improvement |
|----------|--------|----------|-------------|-------------|
| **Franka Panda** (9-DOF) | MAE | 7.51° | **6.26°** | **+16.6%** |
| | RMSE | 29.32° | **25.45°** | +13.2% |
| **Laikago** (12-DOF) | MAE | 5.91° | 5.91° | +0.0% |
| | RMSE | 29.70° | 29.29° | +1.4% |

### Robustness Under Disturbances (Franka Panda)
- **Parameter Uncertainty:** +19.2% improvement
- **No Disturbance:** +16.6% improvement  
- **Payload Variation:** +8.1% improvement
- **Average Across All Scenarios:** +10.0% improvement

## 🔬 Citation

If you use this code in your research, please cite:

```bibtex
@article{wu2026cross,
  title={Cross-Platform Learnable Fuzzy Gain-Scheduled Proportional-Integral-Derivative Controller Tuning via Physics-Constrained Meta-Learning and Reinforcement Learning Adaptation},
  author={Wu, Jiahao and NG, KaHo and Yu, Shengwen},
  journal={Robotica},
  year={2026},
  publisher={Cambridge University Press},
  note={Manuscript ID: ROB-2026-0021}
}
```

**arXiv Preprint:**
```bibtex
@misc{wu2024adaptive,
  title={Adaptive PID Control for Robotic Systems via Hierarchical Meta-Learning and Reinforcement Learning with Physics-Based Data Augmentation},
  author={Wu, Jiahao and Yu, Shengwen},
  year={2024},
  eprint={2511.06500},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyBullet physics simulation
- Stable-Baselines3 RL library
- Cambridge University Press for publication

## 📧 Contact

- **Jiahao Wu** - wuj277970@gmail.com
- **Project Link:** [https://github.com/WUJIAHAO-HKU/RL_Pid_Meta-Learning_Based-Data-Augmentation](https://github.com/WUJIAHAO-HKU/RL_Pid_Meta-Learning_Based-Data-Augmentation)

---
**Note:** This repository contains the official implementation of the method published in *Robotica*. For questions about the paper, please contact the corresponding author.
