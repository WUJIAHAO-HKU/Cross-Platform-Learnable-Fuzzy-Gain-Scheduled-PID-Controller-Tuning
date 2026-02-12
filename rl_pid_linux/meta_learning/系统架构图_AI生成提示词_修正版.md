# 系统架构图 AI生成提示词 - 完整修正版

**版本**: v2.0 修正版  
**日期**: 2025-10-29  
**说明**: 基于System structure.png发现的错误进行修正

---

## 🎯 完整AI生成提示词（直接复制使用）

```
Create a professional system architecture diagram for a robotics research paper with the following specifications:

LAYOUT: 
Horizontal flowchart, 16:12 aspect ratio, three main columns representing "Offline Meta-Learning" (left), "Online RL Adaptation" (middle), and "Control Execution" (right).

=================================================================
LEFT COLUMN - Offline Meta-Learning Stage:
=================================================================

Component 1.1 - Base Robots (Top):
- Box color: Light orange (#FFE5B4), border: #E67E22
- Size: 2.5 width × 1.2 height
- Content:
  """
  Base Robots (K=3)
  ─────────────────
  • Franka Panda (9-DOF)
  • KUKA iiwa (7-DOF)  
  • Laikago (12-DOF)
  """

↓ Arrow: "Physics-Based Augmentation"

Component 1.2 - Virtual Robots:
- Box color: Light green (#D5F4E6), border: #27AE60
- Content:
  """
  Virtual Robots
  ──────────────
  300 samples generated
  
  Perturbations:
  ±10% mass, ±15% inertia
  ±20% friction, ±30% damping
  """

↓ Arrow: "Hybrid Optimization (DE + Nelder-Mead)"

Component 1.3 - Optimal PID Dataset:
- Box color: Light blue (#AED6F1), border: #2874A6
- Content:
  """
  Optimal PID Dataset
  ───────────────────
  203 samples
  
  Each: {robot_features, 
         optimal_PID, 
         optimization_error}
  """
- ⚠️ IMPORTANT: Only "203 samples", NO "200 Dataset"!

↓ Arrow: "Weighted Training"

Component 1.4 - Meta-Learning Network (Bottom, Large):
- Box color: Purple (#D7BDE2), border: #7D3C98, thicker border (3px)
- Size: 2.8 width × 1.8 height
- Content:
  """
  Meta-Learning Network
  ═════════════════════
  Architecture: 3-layer MLP
  [5] → [64] → [64] → [3]
  
  Input Features:
  {DOF, total_mass, avg_inertia,
   max_reach, payload_mass}
  
  Output: θ_init = {K_p^init, K_i^init, K_d^init}
  """
- ⚠️ CRITICAL: Use K_p, K_i, K_d notation, NOT P, I, D!

=================================================================
MIDDLE COLUMN - Online RL Adaptation Stage:
=================================================================

Component 2.1 - RL Environment (Top):
- Box color: Light pink (#FFB6C1), border: #C0392B
- Size: 3.5 width × 2.0 height
- Content:
  """
  RL Environment (PyBullet)
  ═════════════════════════
  State s_t:
  [q_t, q̇_t, e_t, θ_t, q_ref, q̇_ref]
  
  Action a_t:
  [ΔK_p, ΔK_d] ∈ [-0.2, 0.2]²
  
  Reward r_t:
  -10·||e_t||/√n - 0.1·||q̇_t||/√n - 0.1·||a_t||
  """

↕ Bidirectional Arrows (labeled "State s_t" up, "Action a_t" down)

Component 2.2 - PPO Agent:
- Box color: Light yellow (#FFF8DC), border: #F39C12
- Size: 3.2 width × 1.5 height
- Content:
  """
  PPO Agent
  ═════════
  Policy Network: π(a|s; φ)
  Value Network: V(s; ψ)
  
  Training:
  • 200,000 timesteps
  • 4 parallel environments
  • Learning rate: 3×10⁻⁴
  • Discount factor: γ=0.99
  """

↓ Arrow: "Online Adaptation"

Component 2.3 - Adapted PID Controller (Bottom):
- Box color: Light cyan (#B2EBF2), border: #00ACC1
- Size: 3.0 width × 1.3 height
- Content:
  """
  Adapted PID Controller
  ══════════════════════
  θ_adapted = θ_init ⊙ (1 + a_t)
  
  Parameters:
  {K_p^ad, K_i^ad, K_d^ad}
  
  Adjusts gains online for:
  • Model uncertainties
  • External disturbances
  """
- ⚠️ IMPORTANT: Use K_p^ad, K_i^ad, K_d^ad, NOT "P_ad, D_e ̇e"!

→ Thick arrow from Meta-Learning Network (Component 1.4):
  Label: "PID Initialization θ_init"
  Style: Green, bold, curved

=================================================================
RIGHT COLUMN - Control Execution Stage:
=================================================================

Component 3.1 - PID Controller (Top):
- Box color: Light green (#C8E6C9), border: #388E3C
- Size: 3.0 width × 1.6 height
- Content:
  """
  PID Controller
  ══════════════
  Control Law:
  u_i(t) = K_p·e_i(t) + K_i·∫₀ᵗ e_i(τ)dτ + K_d·ė_i(t)
  
  where:
  • e_i = q_ref,i - q_i  (tracking error)
  • Gains: θ_adapted from RL
  """
- ⚠️ CRITICAL: Must include ALL THREE terms (P, I, D)!
- ⚠️ Use proper notation: K_p, K_i, K_d (not P, I, D alone)

↓ Arrow: "Control Commands u"

Component 3.2 - Robot Platform (Middle):
- Box color: Light gray (#ECEFF1), border: #546E7A
- Size: 3.2 width × 2.0 height
- Content:
  """
  Robot Platform
  ══════════════
  🤖 Physical Robot System
  
  • Joint actuators
  • Position sensors
  • Velocity sensors
  
  Dynamics:
  M(q)q̈ + C(q,q̇)q̇ + G(q) = τ
  """
- ⚠️ Include robot icon/illustration if possible
- 💡 OPTION: Can insert actual robot images:
  • franka_panda_visualization.png (for manipulator)
  • laikago_quadruped_visualization.png (for quadruped)

↓ Arrow: "Joint States (q, q̇)"

Component 3.3 - Performance Metrics (Bottom):
- Box color: Light purple (#E1BEE7), border: #8E24AA
- Size: 3.0 width × 1.3 height
- Content:
  """
  Performance Metrics
  ═══════════════════
  ✓ Tracking Error: 5.37° (MAE)
  ✓ Improvement: 24.1%
  ✓ Control Frequency: 240 Hz
  ✓ Training Time: 20 min
  """

⚠️ DO NOT add another "PID Controller" box here! 
⚠️ NO "PID Cotiller" or confused equations!

=================================================================
FEEDBACK LOOP:
=================================================================

Dashed arrow from Robot Platform (Component 3.2) back to RL Environment (Component 2.1):
- Style: Blue dashed line (- - - ->)
- Label: "State/Reward Feedback"
- Path: From right side of Robot Platform, curve up and left to RL Environment

=================================================================
OVERALL STYLE REQUIREMENTS:
=================================================================

Colors:
✓ Use the exact hex colors specified
✓ Soft, pastel palette for academic papers
✓ Consistent border thickness (2-3px)

Typography:
✓ Box titles: 12-14pt, bold
✓ Box content: 10-11pt, regular
✓ Arrow labels: 9-10pt, italic
✓ Mathematical symbols: proper LaTeX-style rendering

Layout:
✓ Rounded corners on all boxes (radius: 10-15px)
✓ Sufficient spacing between components (minimum 0.5 units)
✓ Three columns clearly separated
✓ Vertical alignment within each column

Arrows:
✓ Simple, clean arrow styles
✓ Curved arrows for cross-column connections
✓ Straight arrows for vertical flows
✓ All arrows must have labels
✓ Bidirectional arrows use ↕ or separate ↑↓

Background:
✓ Pure white (#FFFFFF)
✓ No grid or texture

Professional:
✓ Academic paper quality
✓ Clean, minimalist design
✓ No decorative elements
✓ Focus on information clarity

=================================================================
CRITICAL CORRECTIONS (Based on Previous Errors):
=================================================================

❌ WRONG:
- "200 Dataset" alongside "203 samples" (contradictory!)
- τ = P*e + I∫e dt (missing D term!)
- "PID Cotiller" (spelling error + confused equations)
- "P_ad, D_e ̇e" (unclear notation)
- Mixed notation: P, I, D vs K_p, K_i, K_d

✅ CORRECT:
- Only "203 samples" in Optimal PID Dataset
- Full PID equation: u_i = K_p·e_i + K_i·∫e_i dt + K_d·ė_i
- NO extra PID boxes at bottom right
- Consistent notation: K_p, K_i, K_d throughout
- Parameters clearly written: {K_p^ad, K_i^ad, K_d^ad}

=================================================================
VERIFICATION CHECKLIST:
=================================================================

Before finalizing, ensure:
□ All 9 main components are present
□ PID equation has THREE terms (P+I+D)
□ No spelling errors (especially "Controller")
□ Notation is consistent (K_p, K_i, K_d)
□ "203 samples" (not "200 Dataset")
□ Arrows are labeled and clear
□ Colors match specifications
□ No duplicate or confused boxes
□ Feedback loop is dashed and blue
□ Mathematical notation is readable

=================================================================
```

---

## 📝 简化版提示词（如果AI理解有困难）

```
Create a 3-column system architecture diagram for a robotics paper:

LEFT COLUMN (Offline Meta-Learning):
1. Base Robots (K=3): Franka, KUKA, Laikago [orange box]
   ↓ Physics-Based Augmentation
2. Virtual Robots: 300 samples [green box]
   ↓ Hybrid Optimization
3. Optimal PID Dataset: 203 samples [blue box]
   ↓ Weighted Training  
4. Meta-Learning Network: 3-layer MLP → outputs K_p, K_i, K_d [purple box]

MIDDLE COLUMN (Online RL):
5. RL Environment: State, Action, Reward [pink box]
   ↕ bidirectional arrows
6. PPO Agent: Policy + Value networks [yellow box]
   ↓ Online Adaptation
7. Adapted PID: K_p^ad, K_i^ad, K_d^ad [cyan box]

RIGHT COLUMN (Control):
8. PID Controller: u = K_p·e + K_i·∫e + K_d·ė [green box]
   ↓ Control commands
9. Robot Platform: Dynamics M(q)q̈ + C + G = τ [gray box]
   ↓ Joint states
10. Performance: 5.37° error, 24.1% improvement [purple box]

CONNECTIONS:
- Meta Network → Adapted PID (thick arrow, "θ_init")
- Robot → RL Environment (dashed feedback loop)

STYLE: Soft pastel colors, rounded boxes, professional academic look, white background.

CRITICAL: 
- PID equation MUST have all 3 terms (P+I+D)
- Use K_p, K_i, K_d notation consistently
- Only "203 samples" in dataset box
- NO extra confused boxes
```

---

## 🔧 如果AI生成后还有问题的调整指令

### 修正PID公式
```
"The PID Controller box equation is incomplete. Change it to:
u_i(t) = K_p·e_i(t) + K_i·∫e_i(τ)dτ + K_d·ė_i(t)

Make sure all three terms (proportional, integral, derivative) are visible."
```

### 修正符号一致性
```
"Replace all instances of 'P', 'I', 'D' with 'K_p', 'K_i', 'K_d' for consistency.
In the Adapted PID box, use: {K_p^ad, K_i^ad, K_d^ad}"
```

### 修正数据集标注
```
"In the 'Optimal PID Dataset' box, remove '200 Dataset'.
Only keep '203 samples' as the content."
```

### 删除多余框
```
"Remove any duplicate or confusing boxes at the bottom right. 
There should only be: PID Controller → Robot Platform → Performance Metrics"
```

---

## ✅ 关键改进点总结

| 问题 | 原错误 | 修正后 |
|------|--------|--------|
| **PID公式** | τ = P*e + I∫e dt | u_i = K_p·e_i + K_i·∫e_i dt + K_d·ė_i |
| **数据集** | "200 Dataset, 203 samples" | "203 samples" |
| **符号统一** | P, I, D 混用 | 全部使用 K_p, K_i, K_d |
| **Adapted PID** | "P_ad, D_e ̇e" | {K_p^ad, K_i^ad, K_d^ad} |
| **拼写错误** | "PID Cotiller" | 删除此框 |
| **混乱公式** | "M(q)^:2q + C(q)..." | 已删除 |
| **Output表示** | "P_meta, I_meta..." | θ_init = {K_p^init, K_i^init, K_d^init} |

---

## 🎯 推荐使用流程

1. **复制完整提示词**（上面的长版本）
2. **粘贴到ChatGPT或Claude**
3. **生成图像**
4. **检查关键点**：
   - ✅ PID公式有3项
   - ✅ 符号统一用K_p, K_i, K_d
   - ✅ 只有"203 samples"
   - ✅ 没有"Cotiller"或混乱公式
5. **如有问题，使用调整指令**
6. **保存为system_architecture.png**

---

**生成时间**: 2025-10-29  
**版本**: v2.0 完整修正版  
**状态**: ✅ 所有已知错误已修正

