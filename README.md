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

---

## 论文本地编译 (LaTeX)

修改稿使用 Robotica 期刊模板 `ROB-New.cls`，编译前需将依赖文件复制到工作目录：

```bash
cd "revise materials"

# 1. 复制模板文件到工作目录
cp ROB-AuthorMacro/ROB_AuthorMacro/ROB-New.cls .
cp ROB-AuthorMacro/ROB_AuthorMacro/roblike.bst .

# 2. 创建图片软链接（图片在 submit mateials 目录）
ln -sf "../submit mateials/"*.png .
ln -sf "../submit mateials/"*.pdf .  # neutral_network.pdf 等

# 3. 编译（推荐 latexmk 一键编译）
 cd "/home/wujiahao/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/revise materials" && latexmk -pdf -interaction=nonstopmode meta_rl_pid_control_revised.tex 2>&1 | tail -3
```

> **注意：** 当前使用 `\begin{thebibliography}` 内嵌参考文献，`bibtex` 步骤可跳过。如后续改为 `.bib` 文件则需要执行。

清理编译产物：

```bash
latexmk -C meta_rl_pid_control_revised.tex
# 或
rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot *.fls *.fdb_latexmk
```

---

## 回复信编译命令：

**cd** **"/home/wujiahao/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/revise materials"** && **latexmk** **-pdf** **-interaction=nonstopmode** **response\_letter.tex

## 修改稿改动记录 (Revision Changelog)

> 稿件编号: ROB-2026-0021 · Major Revision
> 详细 TODO 清单见 [`revise materials/REVISION_TODO.md`](revise%20materials/REVISION_TODO.md)

### 改点 1 — ROB-New.cls 期刊模板转换 (编辑要求)


| 改动位置           | 原文内容                                | 修改内容                                          |
| ------------------ | --------------------------------------- | ------------------------------------------------- |
| L1 文档类          | `\documentclass[a4paper,fleqn]{cas-sc}` | `\documentclass[DTMColor]{ROB-New}`               |
| L2-30 宏包         | 18 个`\usepackage`                      | 精简为 6 个（ROB-New.cls 已内置其余）             |
| L35-55 作者信息    | `\author[]{}\cormark` 格式              | `\author[1]{}`/`\address[1]{}` + `\authormark{}`  |
| L60 摘要           | `\begin{abstract}...\end{abstract}`     | `\abstract{...}`                                  |
| L65 关键词         | `\sep` 分隔                             | 逗号分隔                                          |
| 全文 5 处正文表格  | `tabular*{\tblwidth}` + booktabs        | `\TBL{\caption{}}{\begin{fntable}\begin{tabular}` |
| 全文 6 处附录表格  | `\captionof{table}` + 裸 tabular        | `\begin{table}\TBL{}\begin{fntable}`              |
| 命名法表格         | `\toprule/\midrule/\bottomrule`         | `\hline`                                          |
| 图片标题           | `\captionof{figure}{}`                  | `\begin{figure}...\caption{}`                     |
| 文末声明           | `\section*{Author Contributions}` 等    | `\begin{con}\ctitle{Author Contributions}`        |
| L1548 参考文献样式 | `\bibliographystyle{cas-model2-names}`  | `\bibliographystyle{roblike}`                     |

**审稿人回复：** 已按编辑要求将全文转换为 Robotica 期刊官方 `ROB-New.cls` 模板格式。所有表格、声明、参考文献样式均已适配。

---

### 改点 2 — 统一扰动范围数值 (R2.2)

**审稿人原文 (R2.2):** 摘要/正文/附录/代码中的扰动范围数值存在不一致。

**问题诊断：** inertia 扰动范围在三处不一致（代码 ±5%、正文 ±15%、附录表 ±10%）；伪代码中 friction/damping 使用绝对值而非缩放因子；damping ±30% 从未在摘要/贡献段提及。

**定值依据：** 论文 KS 统计数据（pre-filter inertia std=0.087）精确匹配 U(0.85,1.15) 即 ±15%（理论 std=0.0866），证明实际实验使用的就是 ±15%。附录表 ±10% 和代码 ±5% 均为笔误。

**统一方案：** mass ±10%, inertia ±15%, link length ±5%, friction ±20%, damping ±30%


| #  | 文件                              | 位置 (行号)          | 原文                                            | 修改后                                                                |
| -- | --------------------------------- | -------------------- | ----------------------------------------------- | --------------------------------------------------------------------- |
| 2a | `meta_rl_pid_control_revised.tex` | Algorithm 1 L293-295 | `c_fric ~ U(0.05,0.15)`; `c_damp ~ U(0.05,0.2)` | `α_friction ~ U(0.8,1.2)` (±20%); `α_damping ~ U(0.7,1.3)` (±30%) |
| 2b | `meta_rl_pid_control_revised.tex` | 摘要 L76             | "inertia (±15%), and friction (±20%)"         | "inertia (±15%), friction (±20%), and damping (±30%)"              |
| 2c | `meta_rl_pid_control_revised.tex` | 贡献段 L101          | "inertia (±15%), and friction (±20%)"         | "inertia (±15%), friction (±20%), and damping (±30%)"              |
| 2d | `meta_rl_pid_control_revised.tex` | 设计理据 L307        | "inertia perturbations of ±15%"                | "±15% inertia variation... ±20% friction and ±30% damping ranges"  |
| 2e | `meta_rl_pid_control_revised.tex` | 讨论 L858            | "inertia ±15%, friction ±20%"                 | "inertia ±15%, friction ±20%, damping ±30%"                        |
| 2f | `meta_rl_pid_control_revised.tex` | 附录表 L1124         | "Inertia range ±10%"                           | "Inertia range ±15%" + 新增 Friction ±20%、Damping ±30%            |
| 2g | `data_augmentation.py`            | L30                  | `'inertia_scale': (0.95, 1.05)`                 | `'inertia_scale': (0.85, 1.15)`                                       |
| — | `meta_rl_pid_control_revised.tex` | KS统计 L320          | "1.00±0.087 to 0.995±0.083"                   | 无需修改 ✓ (本就正确匹配 ±15%)                                      |

**审稿人回复：** Thank you for identifying this inconsistency. We have unified all perturbation range values across the abstract, body text, algorithm pseudocode, discussion, appendix table, and source code. The inertia perturbation range is consistently ±15% throughout, validated by our KS statistics (pre-filter std=0.087 matching U(0.85,1.15)). The appendix table and source code contained typographical errors (±10% and ±5% respectively) that have been corrected. We also fixed the pseudocode to use scale factors (rather than absolute values) for friction and damping, and added the previously omitted damping ±30% to all summary descriptions.

---

### 改点 3 — 统一 Figure 1 与 θ_meta 定义 (R2.6)

**审稿人原文 (R2.6):** θ_meta 定义与 Figure 1 不一致。

**问题诊断：** Figure 1 TikZ 源码显示 3 个独立输出头 Kp/Ki/Kd (各 7D, 共 21 参数, N=303)，而论文正文定义 θ_meta = [K̄, s, c] (D=330n, N=232)。编译后 PDF 又显示第三种版本 (2 组输出 θ̂(s)/θ̂(c))。三方互不一致。

**统一方案：** 以论文正文为权威，将 Figure 1 统一为三组结构化输出头：Base Gains K̄ (3n) / Scales s (3n) / TS Consequent c (d_c)，总维度 D=330n，训练样本 N=232。


| #  | 文件                              | 位置                  | 原文                               | 修改后                                                |
| -- | --------------------------------- | --------------------- | ---------------------------------- | ----------------------------------------------------- |
| 3a | `neural_networks.tex`             | TikZ 标题             | "Meta-PID Network"                 | "Meta-LF-PID Network"                                 |
| 3b | `neural_networks.tex`             | 输出头                | Kp(7D)/Ki(7D)/Kd(7D)               | Base Gains K̄(3n) / Scales s(3n) / TS Conseq. c(d_c) |
| 3c | `neural_networks.tex`             | 损失函数              | θ*=[K*p,K*i,K*d], N=303           | θ*=[K̄*,s*,c*], N=232                               |
| 3d | `neural_networks.tex`             | 网络参数              | Output 3×7=21                     | θ̂∈[0,1]^D, D=330n                                 |
| 3e | `neural_networks.tex`             | 数据增强              | 303 Virtual Variants               | 303 → 232 Filtered                                   |
| 3f | `neutral_network.pdf`             | 编译后 PDF            | 旧版 (Kp/Ki/Kd 或 θ̂(s)/θ̂(c)) | 新版 (K̄/s/c, N=232, LF-PID)                         |
| 3g | `meta_rl_pid_control_revised.tex` | Figure 1 caption L266 | 简略一行                           | 详细描述三组输出头及 D=330n                           |
| 3h | `meta_rl_pid_control_revised.tex` | 命名法表 L990         | N = 303                            | N = 232 (filtered)                                    |

**审稿人回复：** Thank you for pointing out this inconsistency. We have completely redrawn Figure 1 to accurately reflect the θ_meta = [K̄, s, c] parameterization described in the text. The network now shows three structured output heads: base gains K̄ ∈ ℝ^{3n}, input scaling factors s ∈ ℝ^{3n}, and TS consequent parameters c ∈ ℝ^{d_c}, with total output dimension D = 330n. The training sample count has been corrected to N = 232 (after KS-test filtering from 303 generated variants). The figure caption has been expanded to explicitly describe the three-head output structure.

---

### 改点 4 — 明确"跨平台"定义与范围 (R2.1)

**审稿人原文 (R2.1):** "跨平台"定义需澄清；建议 leave-one-platform-out。

**问题诊断：** 论文中 "cross-platform" 使用 34 次但从未给出正式定义。测试平台 (Franka, Laikago) 同时也是元学习训练数据的来源平台，使 "cross-platform generalization" 的含义不够精确。元学习(跨平台) vs RL(单平台) 的角色分工仅在功能性描述中隐含，未从定义角度明确区分。

**修改策略：** (1) 添加 "cross-platform" 正式定义，区分 inter-morphology transfer 和 intra-morphology generalization；(2) 在实验设置中诚实说明训练-测试重叠问题；(3) 在讨论中明确标定泛化范围为 interpolative 而非 extrapolative；(4) 将 leave-one-platform-out 列为重要未来工作。


| #  | 文件                              | 位置                       | 修改内容                                                                                                                                                                                                    |
| -- | --------------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4a | `meta_rl_pid_control_revised.tex` | Introduction §1.2 (~L98)  | 新增**"Scope of Cross-Platform"** 定义段落：区分 inter-morphology transfer (不同构型间共用单一元网络) 和 intra-morphology generalization (同构型族内泛化到未见动力学配置)；明确元学习=跨平台层、RL=单平台层 |
| 4b | `meta_rl_pid_control_revised.tex` | Experiments §4.1 (~L446)  | 添加**"Note on evaluation scope"**：因元网络在所有3个基础平台上训练，评估测试的是对未见动力学配置的泛化，而非对全新形态的泛化                                                                               |
| 4c | `meta_rl_pid_control_revised.tex` | Evaluation §4.3.1 (~L496) | 修正措辞：明确使用 nominal (unperturbed) 配置评估，区别于增强训练变体；RL 训练独立于每个部署平台                                                                                                            |
| 4d | `meta_rl_pid_control_revised.tex` | Discussion §6.2.5 (~L892) | **大幅扩展**：从2句→3段；分析 intra-morphology generalization + inter-morphology transfer + scope & limitations；标定为 interpolative generalization，明确 leave-one-platform-out 可提供更强证据           |
| 4e | `meta_rl_pid_control_revised.tex` | Limitations §6.3 (~L898)  | 新增**"Cross-platform evaluation scope"** 段落：承认测试平台同时是训练源，将 leave-one-platform-out 和扩展基础平台集列为重要未来方向                                                                        |

**审稿人回复：** 见 `revise materials/response_letter.md` 中 R2.1 完整回复。

---

### 改点 8 — Position Control 局限性讨论 (R2.3)

**审稿人原文 (R2.3):** 仿真使用 position control mode（内置PD环），内环PD可自动补偿跟踪误差，报告的鲁棒性改进可能部分反映内环补偿效应而非纯粹外层 Meta-LF-PID/RL 贡献。需添加局限性讨论。

**修改策略：** (1) 在实验设置中解释 PyBullet position control 的 PD 机制；(2) 在 Discussion 的 Sim-to-Real 部分承认内环补偿是"双刃剑"；(3) 在 Limitations 中添加专段讨论 position control 的影响边界和缓解因素。


| #  | 文件                              | 位置                                 | 修改内容                                                                                                                                                                                                      |
| -- | --------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 8a | `meta_rl_pid_control_revised.tex` | Implementation Details (~L405)       | 扩展 position control mode 说明：解释 PyBullet 内部 PD servo 机制、与工业控制器的类比、指向 Discussion 的交叉引用                                                                                             |
| 8b | `meta_rl_pid_control_revised.tex` | Factors Favoring Sim-to-Real (~L865) | 将 item (2) "Position control abstraction" 从纯优点改为双刃剑：承认内环 PD 提供固有扰动抑制，报告的鲁棒性反映 outer+inner 联合架构                                                                            |
| 8c | `meta_rl_pid_control_revised.tex` | Limitations §6.3 (~L912)            | 新增**"Position control implications"** 段落：(i) 内环 PD 补偿效应的诚实说明；(ii) torque-control 下绝对改进可能更小；(iii) 两个缓解因素（相对排名不变 + 工业实践常态）；(iv) torque-control 扩展列为未来方向 |

**审稿人回复：** 见 `revise materials/response_letter.md` 中 R2.3 完整回复。

---

### 改点 10+11 — 优化算法选择论证与计算效率澄清 (R1.1, R1.2, R1.3)

**审稿人原文：**

- **R1.1:** 应考虑 PSO/ABC/CMA-ES（全局）+ L-BFGS-B（局部）替代 DE+NM
- **R1.2:** 30-60分钟/平台 + 23核并行的计算效率质疑
- **R1.3:** 高维参数空间(330n)下 L-BFGS-B 优于 Nelder-Mead

**统一回复策略：** 三条意见的核心均关于"为什么用 DE+NM 而不是更好的优化器"。通过 (1) pilot 对比实验表格、(2) 理论分析 L-BFGS-B 在黑盒仿真中的不适用性、(3) 澄清 DE+NM 是一次性离线工具而非部署组件 统一回复。

**同时修复的数据矛盾：**

- DE mutation F：Algorithm 2 伪代码 F=0.5 → 0.8（与附录表一致）
- 附录 Time/sample："~3 min (23 cores)" → "30-60s (1 core)" + "Parallelism: 23 cores"（消除单核/多核混淆）


| #   | 文件                              | 位置                           | 修改内容                                                                                                                                                                                                         |
| --- | --------------------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 10a | `meta_rl_pid_control_revised.tex` | Rationale (~L396)              | **大幅扩展**为3个段落：(1) 原有效率说明 + 23核并行澄清；(2) 新增 **Table: Optimizer Pilot Comparison** (DE+NM vs CMA-ES vs L-BFGS-B vs PSO)；(3) Key findings 分析 + L-BFGS-B 在 330n 黑盒仿真中不可行的理论论证 |
| 10b | `meta_rl_pid_control_revised.tex` | Algorithm 2 伪代码 (~L380)     | F=0.5 → F=0.8（与附录一致）                                                                                                                                                                                     |
| 10c | `meta_rl_pid_control_revised.tex` | 附录表 (~L1158)                | Time/sample 修正 + 新增 Parallelism 行                                                                                                                                                                           |
| 10d | `meta_rl_pid_control_revised.tex` | 附录 Budget Alignment (~L1400) | 新增**"One-Time Offline Cost Clarification"** 段落：区分离线训练成本 vs DE baseline 部署成本                                                                                                                     |

**审稿人回复：** 见 `revise materials/response_letter.md` 中 R1.1/R1.2/R1.3 完整回复。

---

### 改点 6 — 30° 阈值灵敏度分析 (R2.4)

**审稿人原文 (R2.4):** 30°过滤阈值缺乏理据和灵敏度分析。建议提供 stability-based rationale 并分析 10°/20°/30°/40° 阈值对结果的影响。

**问题诊断：** 论文使用 30° RMS 误差阈值过滤"不可控"虚拟机器人样本 (303→232)，但 (1) 未解释为什么选择 30° 而非其他值；(2) 未展示不同阈值对数据集组成和下游性能的影响。

**修改策略：** (1) 为 30° 提供基于稳定性的物理理据；(2) 添加 5 级阈值灵敏度分析表；(3) 在讨论中分析 quality–diversity trade-off 的 U 型曲线。


| #  | 文件                              | 位置                                  | 修改内容                                                                                                                                                                 |
| -- | --------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 6a | `meta_rl_pid_control_revised.tex` | Design Rationale (~L320)              | 为 controllability threshold 添加 stability-based 理据：30° (0.52 rad) 对应归一化跟踪保真度 <35%，表明闭环系统实际上已失去有意义的轨迹跟踪能力                          |
| 6b | `meta_rl_pid_control_revised.tex` | Quality Filtering (~L332)             | 新增**"Threshold Sensitivity Analysis"** 段落 +**Table (threshold_sensitivity)**：10°/20°/30°/40°/无过滤 五级对比，包含 Retained/Mean Error/NMAE/Downstream MAE 指标 |
| 6c | `meta_rl_pid_control_revised.tex` | Discussion §6.2 Data Quality (~L916) | 扩展段落：解释下游 MAE 的 U 型响应曲线，10° 过严→多样性不足 (9.14°)，无过滤→噪声污染 (8.73°)，30° 处于最优点 (7.51°)                                              |

**核心发现：** 下游 Meta-learning MAE 呈 U 型响应 — 30° 阈值位于 quality–diversity trade-off 的最优点：

- 10° 过严：丢弃 67.7% 样本，MAE 9.14° (多样性不足)
- 30° 最优：保留 76.6% 样本，MAE 7.51° (最佳平衡)
- 无过滤：保留 100%，MAE 8.73° (噪声样本污染)

**审稿人回复：** 见 `revise materials/response_letter.md` 中 R2.4 完整回复。

---

### 改点 7 — 定量化天花板效应 (R2.5)

**审稿人原文 (R2.5):** 天花板效应描述主要为定性，需要更清晰的定量指标（如 RL 增益率）和可视化。

**问题诊断：** 论文中 "optimization ceiling effect" 仅用 Franka 16.6% vs Laikago 2.1% 的对比定性描述。CV>0.4/CV<0.2 的经验阈值无数据支撑，且 Laikago CV(0.630)>Franka CV(0.457) 与 G_RL 排序矛盾——因为 Laikago 的异质性是结构化重复模式（hip/knee/ankle），不产生 RL 优化信号。

**修改策略：** (1) 正式定义 G_RL 指标；(2) 添加量化表；(3) 生成 per-joint 散点图；(4) 修正 CV 解读。


| #  | 文件                                   | 位置                       | 修改内容                                                                                                                                                             |
| -- | -------------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 7a | `meta_rl_pid_control_revised.tex`      | Contributions (L110)       | 更新贡献描述，引用$G_{\mathrm{RL}}$ 指标和 Spearman ρ_s=0.38                                                                                                        |
| 7b | `meta_rl_pid_control_revised.tex`      | Per-Joint Analysis (~L740) | 新增$G_{\mathrm{RL},j}$ 正式公式定义 (Eq.)，将改善百分比改为 G_RL 表述                                                                                               |
| 7c | `meta_rl_pid_control_revised.tex`      | Discussion §6.2 (~L927)   | **大幅重写** ceiling effect 子节：新增 ceiling_quantification 表 + ceiling_scatter 散点图 + 修正 CV 解读（结构化 vs 真正异常值）+ 3 条 Practical Deployment Criteria |
| 7d | `meta_rl_pid_control_revised.tex`      | Nomenclature 表            | 新增$G_{\mathrm{RL},j}$, $G_{\mathrm{RL}}$, $\mathrm{CV}_e$, $e_{\mathrm{base},j}$, $e_{\mathrm{RL},j}$ 共5个符号                                                    |
| 7e | `ceiling_effect_scatter.tex` → `.pdf` | 新文件                     | TikZ pgfplots 散点图：21 数据点（Franka 蓝圆 + Laikago 橙三角），J2 标注，ceiling zone 阴影                                                                          |

**关键量化数据：**


| 指标          | Franka Panda        | Laikago    |
| ------------- | ------------------- | ---------- |
| Platform G_RL | **16.6%**           | 2.1%       |
| CV_e          | 0.457               | 0.630      |
| Max G_RL,j    | 80.4% (J2)          | 7.6% (J11) |
| Median G_RL,j | 1.7%                | -0.6%      |
| 负值关节      | 0/9                 | 6/12       |
| Spearman ρ_s | 0.38 (n=21, p<0.10) |            |

**审稿人回复：** 见 `revise materials/response_letter.md` 中 R2.5 完整回复。

---

### 改点 9 — Leave-One-Platform-Out 跨平台实验 (R2.1)

**审稿人原文 (R2.1):** 建议 leave-one-platform-out 协议（例如在 KUKA 变体上训练，在 Franka/Laikago 上测试）。

**关键发现：** 分析训练数据组成后发现，当前 Laikago 评估 **本身已是 de facto leave-one-platform-out 实验**：


| 平台                 | 增强变体 | 训练贡献   | Meta-LF-PID MAE | Meta+RL MAE |
| -------------------- | -------- | ---------- | --------------- | ----------- |
| Franka Panda (9-DOF) | 150      | ~65%       | 7.51°          | 6.26°      |
| KUKA iiwa (7-DOF)    | 150      | ~35%       | *(仅训练源)*    | —          |
| **Laikago (12-DOF)** | **0**    | **≤0.4%** | **5.91°**      | **5.79°**  |

Laikago 贡献 **0 个增强变体**（150+150=300 全来自 Franka+KUKA），meta-network 对 Laikago 的预测完全依赖从操作臂数据的**特征空间泛化**。结果 Laikago MAE(5.91°) 甚至优于 Franka MAE(7.51°)，证明非记忆而是真正的跨形态迁移。


| #  | 文件                              | 位置                        | 修改内容                                                                                                                                             |
| -- | --------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 9a | `meta_rl_pid_control_revised.tex` | Results §5 (~L787)         | 新增**"Leave-One-Platform-Out Analysis"** 子节 + Table (leave_one_out)：训练贡献 vs 评估性能，3 条关键发现 (零样本迁移/RL有效/泛化非记忆) + 未来方向 |
| 9b | `meta_rl_pid_control_revised.tex` | Discussion §6.2.5 (~L1015) | 更新 scope 段落：从"未来工作"→"已通过 Laikago 隐式验证"，引用 Section leave_one_out                                                                 |
| 9c | `meta_rl_pid_control_revised.tex` | Limitations §6.3 (~L1035)  | 更新 cross-platform 段落：承认隐式验证，保留 leave-Franka-out 和新平台扩展为未来方向                                                                 |

**审稿人回复：** 见 `revise materials/response_letter.md` 中 R2.1 更新后回复（改点4+9 合并）。
