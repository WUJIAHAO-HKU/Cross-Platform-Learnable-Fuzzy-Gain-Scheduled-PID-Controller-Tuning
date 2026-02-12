# Cross-Platform Meta-RL PID Controller Tuning

> **Under Review** at *Robotica* (Cambridge University Press) — Manuscript ID: ROB-2026-0021

**Paper:** Cross-Platform Learnable Fuzzy Gain-Scheduled PID Controller Tuning via Physics-Constrained Meta-Learning and Reinforcement Learning Adaptation  
**Authors:** Jiahao Wu · KaHo NG · Shengwen Yu  
**Preprint:** [arXiv:2511.06500](https://arxiv.org/abs/2511.06500)

---

## What This Does

A hierarchical framework that **automatically tunes PID controllers** for different robots without manual parameter engineering:

1. **Physics-based data augmentation** — perturb 3 base robots → 232 physically valid variants
2. **Meta-learning network** — learns a mapping from robot dynamics features to near-optimal PID gains
3. **RL fine-tuning (PPO)** — online adaptation that further reduces tracking error

Tested on **Franka Panda (9-DOF)** and **Laikago (12-DOF)** in PyBullet simulation.

## Project Structure

```
rl_pid_linux/
├── controllers/                 # PID & RL-PID hybrid controllers
│   ├── pid_controller.py
│   └── rl_pid_hybrid.py
├── envs/                        # Gym environments (PyBullet)
│   ├── franka_env.py
│   └── trajectory_gen.py
├── training/                    # RL training (PPO / DDPG)
│   ├── train_ppo.py
│   └── train_ddpg.py
├── meta_learning/               # Core: data augmentation + meta-network
│   ├── data_augmentation.py     # Physics-constrained sample generation
│   ├── meta_pid_optimizer.py    # Meta-network + hybrid DE/Nelder-Mead optimizer
│   ├── train_meta_pid.py        # Train the meta-network
│   ├── evaluate_meta_rl.py      # Evaluate Meta-PID vs Meta-PID+RL
│   ├── evaluate_robustness.py   # Disturbance robustness evaluation
│   ├── evaluate_laikago.py      # Laikago-specific evaluation
│   ├── meta_rl_combined_env.py  # Combined meta+RL environment
│   └── meta_rl_disturbance_env.py
├── quadruped_research/          # Laikago quadruped experiments
│   ├── adaptive_laikago_env.py
│   ├── meta_pid_for_laikago.py
│   ├── train_adaptive_rl.py
│   └── train_multi_disturbance.py
└── evaluate_trained_model.py
```

## Setup

```bash
conda create -n meta_rl_pid python=3.8
conda activate meta_rl_pid
pip install torch numpy pybullet pybullet_data \
    stable-baselines3 gymnasium scipy scikit-learn \
    matplotlib tqdm pyyaml
```

## Workflow

### Step 1 — Generate augmented training data
```bash
cd rl_pid_linux/meta_learning
python data_augmentation.py
# Output: augmented_pid_data.json → augmented_pid_data_filtered.json
```

### Step 2 — Train meta-learning network
```bash
python train_meta_pid.py
# Loads augmented_pid_data_filtered.json, trains MetaPIDNetwork
# Output: meta_pid_augmented.pth
```

### Step 3 — RL adaptation (PPO fine-tuning)
```bash
cd ../training
python train_ppo.py --robot franka --timesteps 1000000
```

### Step 4 — Evaluate
```bash
cd ../meta_learning

# Meta-PID vs Meta-PID+RL comparison
python evaluate_meta_rl.py

# Robustness under disturbances (payload, friction, parameter uncertainty)
python evaluate_robustness.py

# Laikago quadruped evaluation
python evaluate_laikago.py
```

## Citation

```bibtex
@misc{wu2024adaptive,
  title={Adaptive PID Control for Robotic Systems via Hierarchical 
         Meta-Learning and Reinforcement Learning with Physics-Based 
         Data Augmentation},
  author={Wu, Jiahao and Yu, Shengwen},
  year={2024},
  eprint={2511.06500},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

## Contact

Jiahao Wu — wuj277970@gmail.com
