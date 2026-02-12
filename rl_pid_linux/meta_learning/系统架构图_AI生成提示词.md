# 系统架构图 AI生成提示词 (Figure 1: Hierarchical Meta-RL System Architecture)

## 🎨 整体布局要求

**画布尺寸**: 宽16单位 × 高12单位，横向布局  
**风格**: 学术论文技术流程图，清晰专业，配色柔和  
**背景**: 纯白色  

---

## 📐 详细组件描述（从左到右，自上而下）

### 【第1列 - 离线阶段：左侧，占宽度30%】

#### 顶部标题
- 位置：x=2.5, y=11
- 文字：**"Offline Stage: Meta-Learning"**
- 样式：大号粗体，深蓝色，带浅蓝色背景框

---

#### 组件1.1: Base Robots (最左上)
- 位置：x=2.5, y=9.5
- 尺寸：宽2.5 × 高1.2
- 颜色：浅橙色 (#FFE5B4)，深橙色边框
- 内容：
  ```
  Base Robots (K=3)
  ─────────────
  • Franka Panda (9-DOF)
  • KUKA iiwa (7-DOF)
  • Laikago (12-DOF)
  ```

#### 向下箭头 ↓
- 标签：**"Physics-Based Augmentation"**

#### 组件1.2: Virtual Robots
- 位置：x=2.5, y=7.8
- 尺寸：宽2.5 × 高1.0
- 颜色：浅绿色 (#D5F4E6)，深绿色边框
- 内容：
  ```
  Virtual Robots
  ─────────────
  300 samples
  Perturbed parameters:
  ±10% mass, ±15% inertia
  ```

#### 向下箭头 ↓
- 标签：**"Hybrid Optimization\n(DE + Nelder-Mead)"**

#### 组件1.3: Optimal PID Dataset
- 位置：x=2.5, y=6.2
- 尺寸：宽2.5 × 高1.0
- 颜色：浅蓝色 (#AED6F1)，深蓝色边框
- 内容：
  ```
  Optimal PID Dataset
  ──────────────────
  203 samples
  {features, optimal_PID,
   optimization_error}
  ```

#### 向下箭头 ↓
- 标签：**"Weighted Training"**

#### 组件1.4: Meta-Learning Network (底部重要组件)
- 位置：x=2.5, y=4.2
- 尺寸：宽2.8 × 高1.5
- 颜色：深紫色背景 (#D7BDE2)，紫色粗边框
- 内容：
  ```
  Meta-Learning Network
  ═════════════════════
  Input: Robot Features
  • DOF, mass, inertia
  • reach, payload
  
  Network: 3-layer MLP
  [5] → [64] → [64] → [3]
  
  Output: θ_init = {K_p, K_d, K_i}
  ```

---

### 【第2列 - 在线阶段：中间，占宽度35%】

#### 顶部标题
- 位置：x=8, y=11
- 文字：**"Online Stage: Reinforcement Learning"**
- 样式：大号粗体，深红色，带浅红色背景框

---

#### 从左侧Meta-Learning到这里的粗箭头 →
- 起点：组件1.4右侧
- 终点：组件2.1左侧
- 标签：**"PID Initialization\nθ_init"**
- 样式：粗箭头，绿色

#### 组件2.1: RL Environment (中上)
- 位置：x=8, y=9
- 尺寸：宽3.5 × 高1.8
- 颜色：浅粉色 (#FFB6C1)，深红色边框
- 内容：
  ```
  RL Environment (PyBullet)
  ═════════════════════════
  Robot Simulation
  
  State s_t:
  [q_t, q̇_t, e_t, θ_t, q_ref, q̇_ref]
  
  Action a_t:
  [ΔK_p, ΔK_d] ∈ [-0.2, 0.2]
  
  Reward r_t:
  -10·||e_t|| - 0.1·||q̇_t|| - 0.1·||a_t||
  ```

#### 组件2.2: PPO Agent (中下)
- 位置：x=8, y=6.5
- 尺寸：宽3.2 × 高1.3
- 颜色：浅黄色 (#FFF8DC)，金色边框
- 内容：
  ```
  PPO Agent
  ══════════
  Policy π(a|s; φ)
  Value V(s; ψ)
  
  Training:
  • 200k timesteps
  • 4 parallel envs
  • lr = 3×10⁻⁴
  ```

#### 双向箭头 ↕ 连接组件2.1和2.2
- 上箭头标签：**"State s_t"**
- 下箭头标签：**"Action a_t"**

#### 向下箭头从组件2.2 ↓
- 标签：**"Online Adaptation"**

#### 组件2.3: Adapted PID (底部)
- 位置：x=8, y=4.2
- 尺寸：宽3.0 × 高1.2
- 颜色：浅青色 (#B2EBF2)，深青色边框
- 内容：
  ```
  Adapted PID Controller
  ══════════════════════
  θ_adapted = θ_init ⊙ (1 + a_t)
  
  Online adjustment for:
  • Model uncertainties
  • External disturbances
  ```

---

### 【第3列 - 控制执行：右侧，占宽度35%】

#### 顶部标题
- 位置：x=13.5, y=11
- 文字：**"Control Execution"**
- 样式：大号粗体，深灰色，带浅灰色背景框

---

#### 从组件2.3到这里的粗箭头 →
- 标签：**"PID Gains\nθ_adapted"**

#### 组件3.1: PID Controller (右上)
- 位置：x=13.5, y=9
- 尺寸：宽3.0 × 高1.5
- 颜色：浅绿色 (#C8E6C9)，深绿色边框
- 内容：
  ```
  PID Controller
  ══════════════
  u_i = K_p·e_i + K_i·∫e_i + K_d·ė_i
  
  Gains: θ_adapted
  Reference: q_ref(t)
  Feedback: q_actual(t)
  ```

#### 向下箭头 ↓
- 标签：**"Control\nCommands u"**

#### 组件3.2: Robot Platform (右中)
- 位置：x=13.5, y=6.8
- 尺寸：宽3.2 × 高1.8
- 颜色：浅灰色 (#ECEFF1)，深灰色边框
- 内容：
  ```
  Robot Platform
  ══════════════
  🤖 Physical Robot
  
  • Joint actuators
  • Position sensors
  • Velocity sensors
  
  Dynamics:
  M(q)q̈ + C(q,q̇) + G(q) = τ
  ```

#### 向下箭头 ↓
- 标签：**"Joint States\n(q, q̇)"**

#### 组件3.3: Performance Metrics (右下)
- 位置：x=13.5, y=4.2
- 尺寸：宽3.0 × 高1.2
- 颜色：浅紫色 (#E1BEE7)，紫色边框
- 内容：
  ```
  Performance Metrics
  ═══════════════════
  ✓ Tracking error: 5.37°
  ✓ Improvement: 24.1%
  ✓ Real-time: 240 Hz
  ```

---

### 【反馈回路】

#### 从组件3.2向上的箭头 → 组件2.1
- 路径：从Robot Platform右侧向上，然后向左连接到RL Environment
- 标签：**"Feedback\n(q_t, q̇_t)"**
- 样式：虚线箭头，蓝色

#### 从组件3.2向左的箭头 → 组件3.1
- 标签：**"Sensor\nFeedback"**
- 样式：实线箭头

---

### 【底部图例】

位置：x=8, y=1.5 (居中底部)

```
Process Flow Legend:
─────────────────────────────────────────────────────────
  Offline Meta-Learning  →  Online RL Adaptation  →  Robot Control
  
  Key Features:
  • Hierarchical: Two-stage learning (meta + RL)
  • Efficient: 203 samples → 24.1% improvement
  • Real-time: 20 min training, 240 Hz execution
```

---

### 【侧边标注框】(可选)

右下角 (x=1, y=1.5):
```
Innovation Highlights
─────────────────────
✓ Physics-based augmentation
✓ Hybrid optimization
✓ Meta-learning initialization
✓ Online RL adaptation
✓ Cross-platform generalization
```

---

## 🎨 配色方案总结

| 组件类型 | 背景色 | 边框色 | 说明 |
|---------|-------|--------|------|
| Base Robots | #FFE5B4 | #E67E22 | 浅橙 |
| Virtual Robots | #D5F4E6 | #27AE60 | 浅绿 |
| Dataset | #AED6F1 | #2874A6 | 浅蓝 |
| Meta Network | #D7BDE2 | #7D3C98 | 深紫 |
| RL Environment | #FFB6C1 | #C0392B | 浅粉 |
| PPO Agent | #FFF8DC | #F39C12 | 浅黄 |
| Adapted PID | #B2EBF2 | #00ACC1 | 浅青 |
| PID Controller | #C8E6C9 | #388E3C | 浅绿 |
| Robot | #ECEFF1 | #546E7A | 浅灰 |
| Metrics | #E1BEE7 | #8E24AA | 浅紫 |

---

## 🔤 字体规范

- **标题**: 16-18pt, 粗体
- **组件标题**: 12-14pt, 粗体
- **组件内容**: 10-11pt, 常规
- **箭头标签**: 9-10pt, 斜体
- **图例**: 10pt, 常规

---

## ⚠️ 重要注意事项

1. **层次分明**: 三列布局要清晰，每列有明显的垂直对齐
2. **间距充足**: 组件之间至少0.8单位间距，避免重叠
3. **箭头简洁**: 使用直箭头或简单曲线，避免复杂路径
4. **标签清晰**: 所有箭头都要有标签说明数据流
5. **边框统一**: 所有组件使用圆角矩形，线宽2-3px
6. **反馈回路**: 用虚线或不同颜色区分反馈路径
7. **专业美观**: 整体要有学术论文的专业感

---

## 📝 AI生成提示词（复制使用）

```
Create a professional system architecture diagram for a robotics research paper with the following specifications:

LAYOUT: Horizontal flowchart, 16:12 aspect ratio, three main columns representing "Offline Meta-Learning" (left), "Online RL Adaptation" (middle), and "Control Execution" (right).

LEFT COLUMN - Offline Stage:
- Top: "Base Robots (K=3)" box in light orange (#FFE5B4) listing Franka Panda, KUKA, Laikago
- Arrow down labeled "Physics-Based Augmentation"
- "Virtual Robots" box in light green (#D5F4E6) showing 300 samples
- Arrow down labeled "Hybrid Optimization (DE + Nelder-Mead)"
- "Optimal PID Dataset" box in light blue (#AED6F1) showing 203 samples
- Arrow down labeled "Weighted Training"
- Bottom: Large "Meta-Learning Network" box in purple (#D7BDE2) showing 3-layer MLP architecture with input features and output PID parameters

MIDDLE COLUMN - Online Stage:
- Top: "RL Environment" box in light pink (#FFB6C1) showing state space, action space, and reward function
- Bidirectional arrows connecting to "PPO Agent" box in light yellow (#FFF8DC) below it
- Bottom: "Adapted PID Controller" box in light cyan (#B2EBF2)
- Thick arrow from Meta-Learning Network labeled "PID Initialization θ_init"

RIGHT COLUMN - Control Execution:
- Top: "PID Controller" box in light green (#C8E6C9) showing control law
- Middle: "Robot Platform" box in light gray (#ECEFF1) with robot icon and dynamics equation
- Bottom: "Performance Metrics" box in light purple (#E1BEE7) showing tracking error and improvement

CONNECTIONS:
- Forward flow: left to right with labeled arrows
- Feedback loop: dashed blue arrow from Robot Platform back to RL Environment
- All boxes have rounded corners, 2-3px borders
- Arrow labels in 9-10pt italic font

STYLE: Clean academic paper style, soft colors, clear hierarchy, professional typography, white background.

TEXT: Use the exact text content specified in each box, maintain mathematical notation (subscripts, Greek letters), ensure all labels are readable.
```

---

## 🎯 简化版提示词（如果AI理解复杂提示词有困难）

```
Draw a 3-column system architecture flowchart:

Column 1 (Offline): Base Robots → Virtual Robots → Dataset → Meta-Learning Network
Column 2 (Online): RL Environment ↔ PPO Agent → Adapted PID
Column 3 (Execution): PID Controller → Robot → Metrics

Use soft pastel colors (orange, green, blue, purple, pink, yellow, cyan, gray).
Add arrows between components with labels.
Include a feedback loop from Robot back to RL Environment (dashed line).
Professional academic paper style, rounded rectangles, clear spacing.
```

---

生成后如需调整，可以要求AI修改:
- "增加组件间距"
- "调整某个组件的颜色"
- "加粗某些箭头"
- "调整字体大小"
- "简化/详化某个组件的内容"

