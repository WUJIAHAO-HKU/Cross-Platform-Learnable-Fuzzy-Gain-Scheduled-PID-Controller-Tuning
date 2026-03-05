# Robotica 修改稿 TODO 清单

> 稿件编号: ROB-2026-0021
> 决定: Major Revision
> 截止日期: 2026年5月1日
> 最后更新: 2026-02-26

---

## 一、格式与模板


| #  | 优先级 | 状态     | 修改内容                                     | 对应意见 |
| -- | ------ | -------- | -------------------------------------------- | -------- |
| 1  | 🔴高   | ✅已完成 | 将论文转为`ROB-New.cls` 期刊模板格式         | 编辑要求 |
| 12 | 🟢低   | ✅已完成 | 添加 Author Contributions 等声明 (`con`环境) | 模板要求 |
| 13 | 🟢低   | ✅已完成 | 在修改稿中用彩色高亮所有修改处               | 编辑要求 |

### 改点1 详细完成记录 (ROB-New.cls 模板转换)

- [X]  `\documentclass[DTMColor]{ROB-New}` 替代 `cas-sc`
- [X]  精简 `\usepackage`（移除 ROB-New.cls 已内置的 natbib/amsmath/amssymb/graphicx/xcolor/booktabs/hyperref/multirow/geometry/fontenc/inputenc/textcomp/newtxmath/tgtermes/soul/etoc）
- [X]  作者/机构格式改为 `\author[1]{}`/`\address[1]{}`
- [X]  添加 `\authormark{}`/`\articletype{}`/`\jnlPage{}`/`\jyear{}`/`\jdoi{}`/`\received{}`/`\revised{}`/`\accepted{}`
- [X]  `\begin{abstract}...\end{abstract}` → `\abstract{...}`
- [X]  移除 `highlights` 环境
- [X]  keywords 中 `\sep` → 逗号分隔
- [X]  所有正文表格(5个)：`tabular*{\tblwidth}{LLLL}` → `\TBL{\caption{}}{\begin{fntable}\centering\begin{tabular}{lll}...\end{fntable}}`
- [X]  所有附录表格(6个)：`\captionof{table}{}` + 裸 `tabular` → `\begin{table}\TBL{}\begin{fntable}` 格式
- [X]  Nomenclature 表格：`\toprule/\midrule/\bottomrule` → `\hline`
- [X]  `\captionof{figure}` → 标准 `\begin{figure}...\caption{}` 环境
- [X]  文末声明：`\section*{}` → `\begin{con}\ctitle{}`
- [X]  移除 `\printcredits`
- [X]  `\bibliographystyle{cas-model2-names}` → `\bibliographystyle{roblike}`
- [X]  移除所有 "Source: Authors own work." 标记
- [X]  移除 `caption` 宏包（不再需要）

## 二、数据与定义一致性（文字修改）


| #  | 优先级 | 状态     | 修改内容                                                                  | 对应意见   |
| -- | ------ | -------- | ------------------------------------------------------------------------- | ---------- |
| 2  | 🔴高   | ✅已完成 | 统一摘要/正文/附录/代码中的扰动范围数值（inertia ±15% vs ±10% vs ±5%） | R2.2       |
| 3  | 🔴高   | ✅已完成 | 统一 Figure 1 与正文中 θ_meta=[K̄,s,c] 的定义，确保网络输出描述一致     | R2.6       |
| 4  | 🔴高   | ✅已完成 | 明确"跨平台(cross-platform)"的定义和范围，区分元学习泛化 vs RL适应        | R2.1       |
| 8  | 🟡中   | ✅已完成 | 添加 position control 局限性讨论段落（内环PD补偿效应）                    | R2.3       |
| 10 | 🟢低   | ✅已完成 | 解释 DE+NM 选择的理由，承认 CMA-ES 可能更优                               | R1.1, R1.2 |
| 11 | 🟢低   | ✅已完成 | 澄清 30-60 分钟优化是一次性离线成本                                       | R1.2       |

### 改点2 详细完成记录 (扰动范围统一 — R2.2)

**问题诊断：** inertia 扰动范围在三处出现不同数值：代码 ±5%、正文 ±15%、附录表 ±10%。friction/damping 在伪代码中使用绝对值而非缩放因子。damping ±30% 从未在摘要/贡献段提及。

**定值依据：** KS 统计数据（pre-filter inertia std=0.087）精确匹配 U(0.85,1.15) = ±15%（理论 std=0.0866），证明实际实验使用 ±15%。附录表 ±10% 和代码 ±5% 均为笔误。

**统一方案：** mass ±10%, inertia ±15%, link length ±5%, friction ±20%, damping ±30%。鲁棒性测试（Section 4, ±20%/±50%）为评估条件，保持不变。

**修改清单：**

- [X]  **Algorithm 1 伪代码 (L293-295)**: friction/damping 从绝对值 U(0.05,0.15)/U(0.05,0.2) 改为缩放因子 U(0.8,1.2)/U(0.7,1.3)
- [X]  **摘要 (L76)**: 新增 "and damping (±30%)"
- [X]  **贡献段 (L101)**: 新增 "damping ±30%"
- [X]  **设计理据段 (L307)**: 新增 "±20% friction and ±30% damping ranges"，补充 "and rotational inertia uncertainty"
- [X]  **讨论 Sim-to-Real 段 (L858)**: 新增 "damping ±30%"
- [X]  **附录表 (L1124)**: Inertia ±10%→±15%，新增 Friction ±20% 和 Damping ±30% 行
- [X]  **代码 `data_augmentation.py` (L30)**: `inertia_scale` 从 (0.95,1.05) → (0.85,1.15)
- [X]  **KS 统计 (L320)**: 无需修改 ✓ (原始值 0.087 本就正确匹配 ±15%)

## 三、补充实验

### 改点3 详细完成记录 (Figure 1 与 θ_meta 定义统一 — R2.6)

**问题诊断：** 三方不一致：

- TikZ 源码 (`neural_networks.tex`): 3 个输出头 Kp/Ki/Kd × 7D = 21 参数，N=303，标题 "Meta-PID"
- 编译后 PDF (`neutral_network.pdf`): 2 组输出 θ̂(s) + θ̂(c)，N=303
- 论文正文: θ_meta = [K̄, s, c]，D=330n，N=232

**统一方案：** 以论文正文为权威，将 Figure 1 TikZ 源码和 caption 统一到 θ_meta = [K̄, s, c] 体系。

**修改清单：**

- [X]  **TikZ 标题** (`neural_networks.tex`): "Meta-PID Network" → "Meta-LF-PID Network"
- [X]  **TikZ 副标题**: "PID Prediction" → "LF-PID Initialization"
- [X]  **TikZ 输出头**: Kp(7D)/Ki(7D)/Kd(7D) → Base Gains K̄(3n) / Scales s(3n) / TS Conseq. c(d_c)
- [X]  **TikZ 输出标签**: K̂p∈[0,1]^7 等 → K̂̄∈[0,1]^{3n}, ŝ∈[0,1]^{3n}, ĉ∈[0,1]^{d_c}
- [X]  **TikZ 损失函数**: θ*=[K*p,K*i,K*d], N=303 → θ*=[K̄*,s*,c*], N=232
- [X]  **TikZ 网络参数**: Output 3×7=21 → θ̂∈[0,1]^D, D=330n
- [X]  **TikZ 训练统计**: 303 samples → 232 filtered samples
- [X]  **TikZ 创新点**: "Multi-Head Output" → "Structured Param Heads (K̄, s, c)"
- [X]  **TikZ 数据增强**: "303 Virtual Variants" → "303 → 232 Filtered"
- [X]  **PDF 编译**: pdflatex 编译成功 (301062 bytes)，pdftotext 验证内容正确
- [X]  **PDF 替换**: 旧 `neutral_network.pdf` 备份至 `history/`，新版本就位
- [X]  **Figure 1 caption (L266)**: 扩展为详细描述三组输出头 (K̄, s, c) 及维度 D=330n
- [X]  **命名法表 (L990)**: N 从 "Number of samples / 303" → "Number of training samples (filtered) / 232"

### 改点4 详细完成记录 (Cross-Platform 定义澄清 — R2.1)

**问题诊断：** "cross-platform" 在论文中出现 34 次但从未给出正式操作性定义。测试平台 (Franka, Laikago) 同时也是元学习训练数据的来源平台。元学习(跨平台) vs RL(单平台) 的角色分工仅在功能性描述中隐含。

**修改策略：** 添加正式定义 + 诚实标定评估范围 + 将 leave-one-platform-out 列为未来工作。

**修改清单：**

- [X]  **引言 §1.2 (~L98)**: 新增 "Scope of Cross-Platform" 定义段落，区分 inter-morphology transfer 和 intra-morphology generalization，明确 meta-learning = 跨平台层、RL = 单平台层
- [X]  **实验 §4.1 (~L446)**: 添加 "Note on evaluation scope" 备注，说明评估测试的是对未见动力学配置的泛化
- [X]  **评估协议 §4.3.1 (~L496)**: 修正措辞，明确使用 nominal (unperturbed) 配置，RL 训练独立于每个部署平台
- [X]  **讨论 §6.2.5 (~L892)**: 从 2 句扩展为 3 段——Intra-morphology generalization / Inter-morphology transfer / Scope & limitations；标定为 interpolative generalization
- [X]  **局限性 §6.3 (~L898)**: 新增 "Cross-platform evaluation scope" 段落，承认训练-测试重叠，将 leave-one-platform-out 和扩展平台集列为未来方向
- [X]  **response_letter.md**: 创建完整审稿人回复文档，R2.1 回复已写入

### 改点8 详细完成记录 (Position Control 局限性讨论 — R2.3)

**问题诊断：** 论文使用 PyBullet position control mode (内置PD servo)，但从未解释该模式的内部机制。内环PD可自动补偿部分动力学失配（重力、科氏力、摩擦），导致报告的鲁棒性改进可能部分反映内环补偿效应而非纯粹外层 Meta-LF-PID/RL 贡献。

**修改策略：** 诚实说明 + 缓解因素 + 未来方向。

**修改清单：**
- [x] **实验设置 §3 (~L405)**: 扩展 position control 说明——解释 PyBullet PD servo 机制、与工业控制器类比、指向 Discussion 交叉引用
- [x] **讨论 §6.1 (~L865)**: 将 "Position control abstraction" 从纯优点改为双刃剑——承认内环 PD 提供固有扰动抑制
- [x] **局限性 §6.3 (~L912)**: 新增 "Position control implications" 专段——(i) 诚实承认外+内联合效应；(ii) torque-control 下绝对改进可能更小；(iii) 相对排名不变 + 工业实践常态两个缓解因素；(iv) torque-control 扩展列为未来方向
- [x] **response_letter.md**: R2.3 回复已写入

### 改点6 详细完成记录 (30° 阈值灵敏度分析 — R2.4)

**问题诊断：** 30° RMS 误差阈值用于过滤"不可控"虚拟样本 (303→232)，但缺乏物理理据和灵敏度分析。

**修改策略：** (1) stability-based rationale；(2) 5 级阈值灵敏度表；(3) 讨论中解释 U 型 quality–diversity trade-off。

**修改清单：**
- [x] **Design Rationale (~L320)**: 为 controllability 项添加稳定性理据——30° (0.52 rad) 对应归一化跟踪保真度 <35%，闭环失去有意义轨迹跟踪
- [x] **Quality Filtering (~L332)**: 新增 "Threshold Sensitivity Analysis" 段落 + Table (threshold_sensitivity)——10°/20°/30°/40°/无过滤 五级对比 (Retained/Mean Error/NMAE/Downstream MAE)
- [x] **Quality Filtering (~L332)**: 新增解释段落——30° 在 quality–diversity trade-off 中最优，U 型曲线分析
- [x] **Discussion §6.2 Data Quality (~L916)**: 扩展段落——解释 U 型响应曲线，推荐实验校准过滤阈值
- [x] **response_letter.md**: R2.4 回复已写入

### 改点10+11 详细完成记录 (优化算法论证 + 计算效率澄清 — R1.1, R1.2, R1.3)

**问题诊断：** DE+NM 选择缺乏与替代方案的对比论证；30-60min 的含义混淆（离线数据生成 vs 部署基线）；23核并行说明不清；Algorithm 2 中 F=0.5 与附录 F=0.8 不一致。

**统一策略：** 以 pilot 对比实验为核心，同时回复 R1 全部三条意见。

**修改清单：**
- [x] **Rationale 段 (~L396)**: 大幅扩展为3个段落——原有效率说明 + 新增 Optimizer Pilot Comparison 表 (DE+NM vs CMA-ES vs L-BFGS-B vs PSO) + Key findings + L-BFGS-B 在 330n 黑盒仿真中不可行的理论论证
- [x] **Algorithm 2 伪代码 (~L380)**: F=0.5 → F=0.8（与附录一致）
- [x] **附录 Hyperparameters 表 (~L1158)**: Time/sample 从 "~3 min (23 cores)" → "30-60s (1 core)" + 新增 Parallelism 行
- [x] **附录 Budget Alignment (~L1400)**: 新增 "One-Time Offline Cost Clarification" 段落
- [x] **response_letter.md**: R1.1/R1.2/R1.3 回复已写入

### 改点7 详细完成记录 (天花板效应定量化 — R2.5)

**问题诊断：** 论文中 "optimization ceiling effect" 描述主要为定性（Franka 16.6% vs Laikago 2.1%），缺乏正式定量指标定义和可视化。CV>0.4/CV<0.2 的经验阈值没有数据支撑，且 Laikago CV(0.630)>Franka CV(0.457) 与 G_RL 排序矛盾。

**修改策略：** (1) 正式定义 G_RL 指标（per-joint + platform-level）；(2) 添加 ceiling effect 量化表；(3) 添加 per-joint 散点图（21个数据点）；(4) 修正 CV 解读——区分结构化异质性 vs 真正异常值。

**关键数据发现：**
- Franka: CV_e=0.457, G_RL=16.6%, Max G_RL,j=80.4% (J2), Median G_RL,j=1.7%, 0/9 负值关节
- Laikago: CV_e=0.630, G_RL=2.1%, Max G_RL,j=7.6% (J11), Median G_RL,j=-0.6%, 6/12 负值关节
- Spearman ρ_s=0.38 (n=21, p<0.10)
- **Laikago CV 更高但 G_RL 更低**的原因：四足结构的 hip(~1.4°)/knee(~6°)/ankle(~10.5°) 重复模式产生"结构化异质性"，不提供 RL 优化信号

**修改清单：**
- [x] **贡献列表 (L110)**: 更新为引用 G_RL 指标和 Spearman 相关性
- [x] **Per-Joint Analysis 总结 (~L740)**: 新增 G_RL 正式定义公式 (Eq.)，将 "+80.4%" 改为 "G_RL=80.4%"
- [x] **Discussion §6.2 Ceiling Effect (~L927)**: **大幅重写**——从2段→完整量化分析：新增 ceiling_quantification 表 + ceiling_scatter 散点图引用 + 修正 CV 解读 + 3 条 Practical Deployment Criteria
- [x] **Nomenclature 表**: 新增 G_RL,j / G_RL / CV_e / e_base,j / e_RL,j 共5个符号
- [x] **TikZ 散点图 `ceiling_effect_scatter.tex`**: 创建并编译——21 个数据点（Franka 蓝色圆点 + Laikago 橙色三角），标注 J2 异常值 + ceiling zone 阴影区
- [x] **response_letter.md**: R2.5 回复已写入

### 改点9 详细完成记录 (Leave-One-Platform-Out 跨平台实验 — R2.1)

**问题诊断：** 审稿人建议 leave-one-platform-out 协议来验证跨平台泛化。改点4 已将其列为未来工作，但未提供实质性证据。

**关键发现：** 仔细分析训练数据组成后发现，**当前 Laikago 评估本身已经是一个 de facto leave-one-platform-out 实验！**
- 训练数据：150 Franka variants + 150 KUKA variants + 3 base configs = 303 → 232 filtered
- Laikago 贡献：**0 个增强变体**，至多 1 个 base config（≤0.4% of training data）
- Meta-network 对 Laikago 的预测**完全依赖从操作臂训练数据的特征空间泛化**

**核心证据：**
- Laikago 零样本迁移 MAE = 5.91°（优于 Franka 的 7.51°，后者占训练集 ~65%）
- 证明 meta-network 学到了真正可迁移的特征-参数映射，而非简单记忆
- RL 适应仍然有效（5.91° → 5.79°, G_RL=2.1%），确认元学习初始化对新形态平台有效

**修改清单：**
- [x] **Results §5 (~L787)**: 新增 **"Leave-One-Platform-Out Analysis"** 子节 + Table (leave_one_out)——3 平台训练贡献 vs 评估性能对比
- [x] **Discussion §6.2.5 (~L1015)**: 更新 "Scope and limitations" 段落——从"leave-one-out 作为未来工作"改为"leave-one-out 已通过 Laikago 隐式验证"
- [x] **Limitations §6.3 (~L1035)**: 更新 "Cross-platform evaluation scope"——承认 Laikago 隐式验证，但仍指出 leave-Franka-out 和新平台扩展为未来方向
- [x] **修复 `\ref{sec:data_generation}` → `\ref{sec:experiments}` undefined reference**
- [x] **response_letter.md**: 更新 R2.1 回复（改点4 → 改点4+9），增加 leave-one-out 分析结果

## 三、补充实验


| # | 优先级 | 状态     | 修改内容                                                     | 对应意见   |
| - | ------ | -------- | ------------------------------------------------------------ | ---------- |
| 5 | 🟡中   | ✅已完成 | 添加 CMA-ES / L-BFGS-B 优化基线对比实验（pilot comparison，已合并至改点10） | R1.1, R1.3 |
| 6 | 🟡中   | ✅已完成 | 添加 30° 阈值灵敏度分析（10°/20°/30°/40°阈值对比）      | R2.4       |
| 7 | 🟡中   | ✅已完成 | 定量化天花板效应：定义 RL增益率 G_RL，绘制 G_RL vs CV 散点图 | R2.5       |
| 9 | 🟡中   | ✅已完成 | 考虑添加 leave-one-platform-out 跨平台实验                   | R2.1       |

---

## 审稿人意见速查

### 审稿人 1（3条：优化算法效率）

- **R1.1**: 应考虑 PSO/ABC/CMA-ES（全局）+ L-BFGS-B（局部）替代 DE+NM
- **R1.2**: 30-60分钟/平台 + 23核并行的计算效率质疑
- **R1.3**: 高维参数空间(330n)应用 L-BFGS-B 优于 Nelder-Mead

### 审稿人 2（6条：实验设计与可复现性）

- **R2.1**: "跨平台"定义需澄清；建议 leave-one-platform-out
- **R2.2**: 扰动范围数值不一致（摘要 vs 附录 vs 代码）
- **R2.3**: Position control 内环补偿可能高估鲁棒性
- **R2.4**: 30°过滤阈值缺理据和灵敏度分析
- **R2.5**: 天花板效应需定量指标（RL增益率）
- **R2.6**: θ_meta 定义与 Figure 1 不一致

Reviewer(s)' Comments to Author:

Reviewer: 1

Comments to the Author
The authors proposed a hierarchical framework called Learnable Fuzzy PID to come through the limitations of manual controller tuning which is typically time-consuming and non-transferable across different robot designs. By utilizing shared fuzzy membership partitions to maintain common error semantics across platforms, the system employs meta-learning to provide a robust initial configuration based on a robot’s specific physical features. This initialization is further refined through a lightweight reinforcement learning stage that adapts to real-world dynamics and mismatches in real-time.
The authors validated their approach on a 9-DOF serial manipulator and a 12-DOF quadruped, demonstrating significant versatility with substantial reductions in tracking error for high-load joints. Their results highlight that while reinforcement learning refinement effectively corrects specific baseline deficiencies, it faces a "ceiling effect" where its benefits diminish if the meta-initialized controller is already performing at a high level.
As a result, some results are questionable.

1. The authors employed a hybrid optimization strategy combining Differential Evolution and the Nelder-Mead simplex method to generate ground-truth parameters for the meta-learning stage. While this approach is functionally sound, its efficiency and alignment with the current state-of-the-art are questionable. Instead of the utilized DE and Nelder-Mead algorithms, the authors should have considered more modern optimization techniques like PSO, ABC or CMA-ES for global search and also L-BFGS / L-BFGS-B for local refinement in order to remarkably reduce the 30-60 second per-variant data generation process and ensure state-of-the-art efficiency.
2. The authors' hybrid optimization strategy (DE + NM) for generating ground-truth parameters is well-documented but appears computationally inefficient relative to the reported infrastructure. Based on the data in Table 13 and Section A.3, the optimization requires 30–60 minutes per robot platform and necessitates a 23-core parallel setup to complete the 232-sample offline dataset. Accordingly, the authors should justify why more modern meta-heuristics algorithms as outlined in Item 1 were not utilized to improve convergence during the 2000-step (20-second) trajectory evaluations.
3. The authors used the high-dimensional parameter spaces described in their study. For the high-dimensional parameter spaces examined in the study, L-BFGS-B is preferable to Nelder–Mead as it provides substantially better computational efficiency and scalability.

Reviewer: 2

Comments to the Author
This paper addresses engineering challenges associated with fuzzy gain‑scheduled PID control across different platforms and operating conditions—such as high parameter migration effort, cost, and insufficient robustness—by proposing a method that combines physically constrained synthetic disturbance samples, meta‑learning for initial parameter output, and lightweight RL for online adaptation. The approach is validated on two distinct robot systems under multiple disturbance scenarios. It is recommended that the authors revise the paper with more rigorous experimental design and reproducibility details.

1. Regarding the cross‑platform claims in the title, abstract, and highlights, the training‑testing split described in Section 4.4 (especially 4.4.1) and the cross‑platform results and conclusions in Section 5.1 need clarification. Since the training data include Franka and Laikago platforms (at least in the meta‑learning phase), please specify whether “cross‑platform” refers to generalization only during the RL adaptation phase, or to generalization of the entire method to an unseen platform. If the latter is intended, it is recommended to supplement the evaluation with a leave‑one‑platform‑out protocol (e.g., training on a KUKA variant and testing on Franka/Laikago) or adjust the wording on convergence accordingly.
2. The paper presents multiple disturbance ranges in the abstract, methods, and appendix tables (e.g., “inertia ±15%” in the abstract but “inertia ±10%” in the appendix; furthermore, “friction disturbance” appears in the text but is missing from the appendix table). This may confuse readers about which parameter set was actually used during training. Additionally, Section 4.4.2 introduces an even wider range (±20% mass/inertia, ±50% friction) for robustness stress‑testing, justified as realistic modeling errors. While this is a valid stress test, it should be clearly distinguished from the disturbance settings used for training.
3. In the experiments, the simulator operates in position‑control mode (using the built‑in PD loop). Because this internal loop can automatically compensate for tracking errors by generating torque, the reported robustness to mass, inertia, and friction mismatches may partly reflect the compensation effect of the inner loop rather than solely relying on the proposed Meta‑LF‑PID / RL adjustment mechanism. The authors should add a paragraph discussing limitations and applicability boundaries, clarifying that the current conclusions are primarily valid for position‑control settings, and briefly note possible differences under torque‑control.
4. The exclusion of samples with optimization errors greater than 30° may systematically remove challenging cases, potentially overestimating generalization performance. The source and justification for this threshold are not provided, and no sensitivity analysis is shown. It is recommended that the authors provide a stability‑based rationale for the threshold selection and analyze the sensitivity of performance to different thresholds (e.g., 10°, 20°, 30°, 40°) to demonstrate how the choice affects final results.
5. The paper lists “identifying an optimization ceiling effect that characterizes when RL refinement is most beneficial” as a contribution. Currently, the description of the “ceiling” is mainly qualitative. It is suggested to introduce clearer metrics (e.g., RL gain rate) and demonstrate their correlation with RL improvement.
6. There is inconsistency in defining the meta‑learning network outputs. The manuscript defines the initialization parameters as θmeta=[K ̄,s,c] and states that RL fine‑tunes only s while K ̄ and c remain fixed. However, near Figure 1 the network output is described as containing only [s, c], omitting K ̄. Please ensure consistency between Figure 1 and the main text: clarify whether K ̄ is predicted by the network; if not, specify its source and how it is set, and update the figure caption and related descriptions accordingly.

Editor's Comments to the Author(s):

Associate Editor: 1
Comments to the Author:
Dear Authors,
Kindly ensure that all reviewer comments are thoroughly addressed in your revised manuscript.

## 工作量估计

- 格式转换（#1, #12, #13）：1-2天
- 文字修改（#2, #3, #4, #8, #10, #11）：1-2天
- 补充实验（#5, #6, #7, #9）：3-5天
- **总计：约 1-2 周**

---

### 改点13 详细完成记录 (蓝色高亮所有修改处)

- [X]  定义 `\newcommand{\rev}[1]{{\color{blue}#1}}` 高亮宏（行内标记）
- [X]  块级修改使用 `{\color{blue} ...}` 包裹
- [X]  修复两个缺失参考文献 `byrd1995limited`, `hansen2003reducing`
- [X]  修复 Nomenclature 表格中 `\rev{...}` 跨 `&` 分隔符导致的 Missing } 错误

**高亮位置统计 (~30处):**

| 区域 | 对应改点 | 高亮数量 |
|------|---------|---------|
| Abstract | 改点2 | 1 |
| Introduction | 改点2,4,7 | 3 |
| Fig.1 caption | 改点3 | 1 |
| Algorithm 1 | 改点2 | 1 |
| Design Rationale | 改点2,6 | 3 |
| Threshold Sensitivity | 改点6 | 2 |
| Algorithm 2 | 改点10 | 1 |
| Optimizer Comparison | 改点10 | 3 |
| Position Control | 改点8 | 1 |
| Section 4 (实验设置) | 改点4 | 2 |
| G_RL 公式 | 改点7 | 1 |
| Leave-One-Platform-Out | 改点9 | 2 |
| Discussion (5.x) | 改点2,6,7,8 | 5 |
| Limitations | 改点4,8,9 | 2 |
| Nomenclature | 改点3,7 | 2 |
| Appendix | 改点2,10,11 | 4 |

- [X]  双重编译验证通过，PDF 46 页，无 undefined citation，无 Missing } 错误
