# 📊 论文图表规划与LaTeX编辑指南

**作者**: 吴家豪 (Jiahao Wu)  
**学校**: 香港大学 (The University of Hong Kong)  
**邮箱**: wuj277970@gmail.com

---

## 📋 目录

1. [图表完整规划](#图表完整规划)
2. [LaTeX文件编辑指南](#latex文件编辑指南)
3. [图表插入方法](#图表插入方法)
4. [具体插入位置](#具体插入位置)
5. [图表制作建议](#图表制作建议)

---

## 📊 图表完整规划

### **推荐图表清单（共8张）**

根据论文内容，建议插入以下图表：

| 图号 | 图表名称 | 类型 | 位置章节 | 优先级 | 状态 |
|------|---------|------|---------|--------|------|
| **Figure 1** | 系统架构图 | 示意图 | Section 3.2 | ⭐⭐⭐⭐⭐ | ⚠️ 待创建 |
| **Figure 2** | 物理数据增强流程图 | 示意图 | Section 3.3 | ⭐⭐⭐⭐⭐ | ⚠️ 待创建 |
| **Figure 3** | Meta-PID训练曲线 | 结果图 | Section 5.4 | ⭐⭐⭐⭐ | ⚠️ 待创建 |
| **Figure 4** | RL训练曲线 | 结果图 | Section 5.4 | ⭐⭐⭐⭐⭐ | ✅ 有数据 |
| **Figure 5** | Franka逐关节误差对比 | 结果图 | Section 5.1 | ⭐⭐⭐⭐⭐ | ⚠️ 待创建 |
| **Figure 6** | 实际跟踪误差对比 | 结果图 | Section 5.1 | ⭐⭐⭐⭐⭐ | ✅ 已有 |
| **Figure 7** | 扰动场景鲁棒性对比 | 结果图 | Section 5.2 | ⭐⭐⭐⭐⭐ | ✅ 已有 |
| **Figure 8** | 消融实验对比 | 结果图 | Section 5.5 | ⭐⭐⭐⭐ | ⚠️ 待创建 |

**说明**：
- ⭐⭐⭐⭐⭐ = 必须有（核心图表）
- ⭐⭐⭐⭐ = 强烈建议（提升质量）
- ⭐⭐⭐ = 可选（锦上添花）

---

## 📝 LaTeX文件编辑指南

### **1. 使用什么编辑器？**

#### **推荐选项**：

**选项A: Overleaf（最推荐）** ⭐⭐⭐⭐⭐
```
优点：
✅ 在线编辑，无需安装
✅ 实时预览
✅ 自动编译
✅ 易于协作
✅ 支持所有LaTeX包

访问: https://www.overleaf.com
```

**选项B: VS Code + LaTeX Workshop** ⭐⭐⭐⭐
```
优点：
✅ 功能强大
✅ 语法高亮
✅ 代码补全
✅ 本地编辑

安装：
1. 安装VS Code
2. 安装LaTeX Workshop插件
3. 安装TeXLive或MiKTeX
```

**选项C: TeXstudio** ⭐⭐⭐⭐
```
优点：
✅ 专业LaTeX编辑器
✅ 内置预览
✅ 跨平台

安装：
sudo apt-get install texstudio  # Ubuntu
```

---

### **2. 如何编辑LaTeX文件？**

#### **基本编辑流程**：

```bash
# 步骤1: 打开文件
cd /home/wujiahao/.../meta_learning/
code 论文_RAS_CAS格式.tex  # 或用其他编辑器

# 步骤2: 编辑内容
# （在编辑器中修改）

# 步骤3: 编译测试
./编译CAS论文.sh  # 使用一键脚本

# 步骤4: 查看PDF
xdg-open /home/wujiahao/.../els-cas-templates/论文_RAS_CAS格式.pdf
```

---

### **3. LaTeX基本语法速查**

#### **3.1 修改文本**
```latex
% 普通文本
这是普通文本

% 加粗
\textbf{加粗文本}

% 斜体
\textit{斜体文本}

% 引用
\citep{reference_key}
```

#### **3.2 添加章节**
```latex
\section{章节标题}
\subsection{子章节标题}
\subsubsection{子子章节标题}
```

#### **3.3 插入公式**
```latex
% 行内公式
$y = mx + c$

% 独立公式（带编号）
\begin{equation}
    E = mc^2
\end{equation}

% 多行公式
\begin{align}
    a &= b + c \\
    d &= e + f
\end{align}
```

#### **3.4 插入列表**
```latex
% 无序列表
\begin{itemize}
    \item 第一项
    \item 第二项
\end{itemize}

% 有序列表
\begin{enumerate}
    \item 第一项
    \item 第二项
\end{enumerate}
```

---

## 🖼️ 图表插入方法

### **方法1: 插入单张图片（推荐）**

```latex
\begin{figure}[htbp]  % h=here, t=top, b=bottom, p=page
  \centering
  \includegraphics[width=0.9\columnwidth]{图片文件名.png}
  \caption{图片标题描述}
  \label{fig:label_name}
\end{figure}
```

**说明**：
- `width=0.9\columnwidth`: 图片宽度为列宽的90%（双栏格式）
- `width=\textwidth`: 图片占满整个文本宽度（跨栏）
- `[htbp]`: 图片位置选项（h=此处，t=顶部，b=底部，p=单独页面）

---

### **方法2: 插入跨栏大图**

```latex
\begin{figure*}[t]  % * 表示跨双栏
  \centering
  \includegraphics[width=0.95\textwidth]{大图文件名.png}
  \caption{跨栏图片标题}
  \label{fig:wide_figure}
\end{figure*}
```

**适用场景**：
- 系统架构图（复杂）
- 多子图组合
- 需要更多空间展示细节的图

---

### **方法3: 插入多子图**

```latex
\begin{figure}[htbp]
  \centering
  \begin{subfigure}{0.45\columnwidth}
    \includegraphics[width=\textwidth]{子图1.png}
    \caption{子图1标题}
    \label{fig:sub1}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.45\columnwidth}
    \includegraphics[width=\textwidth]{子图2.png}
    \caption{子图2标题}
    \label{fig:sub2}
  \end{subfigure}
  \caption{总图标题}
  \label{fig:combined}
\end{figure}
```

**注意**：需要在开头添加：
```latex
\usepackage{subcaption}
```

---

### **方法4: 插入表格**

```latex
\begin{table}[htbp]
\centering
\caption{表格标题}
\label{tab:my_table}
\begin{tabular*}{\tblwidth}{@{}LLLL@{}}
\toprule
\textbf{列1} & \textbf{列2} & \textbf{列3} & \textbf{列4} \\
\midrule
数据1 & 数据2 & 数据3 & 数据4 \\
数据5 & 数据6 & 数据7 & 数据8 \\
\bottomrule
\end{tabular*}
\end{table}
```

---

### **方法5: 引用图表**

```latex
% 引用图
如Figure~\ref{fig:label_name}所示...

% 引用表
如Table~\ref{tab:my_table}所示...

% 引用公式
根据Equation~\ref{eq:my_equation}...
```

---

## 📍 具体插入位置

### **Figure 1: 系统架构图（Hierarchical Meta-RL Architecture）**

**插入位置**: Section 3.2之后

**在文件中搜索**: `\subsection{Hierarchical Meta-RL Architecture}`

**插入代码**（约第265行之后）：
```latex
\subsection{Hierarchical Meta-RL Architecture}

Our framework consists of two complementary components operating at different timescales:

% ========== 在这里插入Figure 1 ==========
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.95\textwidth]{system_architecture.png}
  \caption{Hierarchical Meta-RL architecture for adaptive PID control. The framework consists of two stages: (1) Meta-learning stage that predicts initial PID parameters from robot features, and (2) RL stage that adapts PID parameters online to handle disturbances and uncertainties. The green blocks represent the meta-learning network, the blue blocks represent the RL agent, and the yellow blocks represent the robot simulation environment.}
  \label{fig:architecture}
\end{figure*}
% ========================================

\subsubsection{Stage 1: Meta-Learning for PID Initialization}
```

**图片内容建议**：
- 左侧：Meta-learning阶段（Robot Features → Neural Network → Initial PID）
- 右侧：RL阶段（State → Policy Network → PID Adjustments → Robot → Next State）
- 箭头标明数据流向
- 用不同颜色区分两个阶段

---

### **Figure 2: 物理数据增强流程图（Physics-Based Data Augmentation）**

**插入位置**: Algorithm 1之后

**在文件中搜索**: `\end{algorithm}`（第一个，约第328行）

**插入代码**：
```latex
\end{algorithm}

% ========== 在这里插入Figure 2 ==========
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\columnwidth]{data_augmentation_flow.png}
  \caption{Physics-based data augmentation pipeline. Starting from 3 base robots, we generate 303 virtual variants by systematically perturbing physical parameters (mass, inertia, friction, damping) within physically realistic bounds. Each virtual robot is optimized to obtain ground-truth PID parameters, which are then used to train the meta-learning model.}
  \label{fig:augmentation}
\end{figure}
% ========================================

\textbf{Design Rationale:} The perturbation ranges are carefully chosen to:
```

**图片内容建议**：
```
Base Robot (3)
    ↓
Parameter Perturbation
(mass, inertia, friction, damping)
    ↓
Virtual Robots (300)
    ↓
PID Optimization (Differential Evolution)
    ↓
Optimal PID Database
    ↓
Meta-Learning Training
```

---

### **Figure 3: Meta-PID训练曲线（Meta-Learning Convergence）**

**插入位置**: Section 5.4.1

**在文件中搜索**: `\subsubsection{Meta-Learning Convergence}`（约第683行）

**插入代码**：
```latex
\subsubsection{Meta-Learning Convergence}

% ========== 在这里插入Figure 3 ==========
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\columnwidth]{meta_learning_training.png}
  \caption{Meta-learning training convergence. The training loss and validation loss both converge within 500 epochs ($\sim$5 minutes). The final meta-learning prediction error is 3.33\% on average across test robots.}
  \label{fig:meta_training}
\end{figure}
% ========================================

The meta-learning stage converges within 500 epochs ($\sim$5 minutes), with validation loss stabilizing around epoch 300.
```

**图片内容**：
- X轴：Epochs (0-500)
- Y轴：Loss (MSE)
- 两条线：Training Loss（蓝色）、Validation Loss（橙色）
- 标注：收敛点（epoch 300左右）

---

### **Figure 4: RL训练曲线（RL Training Dynamics）**

**插入位置**: Section 5.4.2

**在文件中搜索**: `\subsubsection{RL Training Dynamics}`（约第686行）

**插入代码**：
```latex
\subsubsection{RL Training Dynamics}

% ========== 在这里插入Figure 4 ==========
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\columnwidth]{training_curves.png}
  \caption{RL training curves for Franka Panda. (a) Mean episode reward improves from -67.45 to -38.92, representing a 42.3\% improvement. (b) Explained variance increases from 0.15 to 0.72, indicating effective value learning. Training converges at 200k timesteps with stable performance thereafter.}
  \label{fig:rl_training}
\end{figure}
% ========================================

The RL training curves for Franka Panda show:
```

**图片内容**：
- 上子图：Mean Reward vs Timesteps
- 下子图：Explained Variance vs Timesteps
- 标注关键点（初始值、最终值、收敛点）

**使用现有文件**: `training_curves.png`（已存在）

---

### **Figure 5: Franka逐关节误差对比（Per-Joint Error Breakdown）**

**插入位置**: Section 5.1.1的Per-Joint Analysis段落

**在文件中搜索**: `\textbf{Per-Joint Analysis:}`（约第585行）

**插入代码**：
```latex
\textbf{Per-Joint Analysis:} 

% ========== 在这里插入Figure 5 ==========
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\columnwidth]{per_joint_error.png}
  \caption{Per-joint error comparison for Franka Panda. Joints 2 and 7 (shoulder and wrist) exhibit the largest improvements (27.6\% and 24.4\%), as these joints experience higher loads and benefit most from adaptive control. The error bars represent standard deviation across 3 episodes.}
  \label{fig:joint_errors}
\end{figure}
% ========================================

Joints 2 and 7 (shoulder and wrist, respectively) exhibit the largest improvements...
```

**图片内容**：
- X轴：Joint Index (1-9)
- Y轴：MAE (degrees)
- 两组柱状图：Meta-PID（蓝色）vs Meta-PID+RL（绿色）
- 误差棒显示标准差

---

### **Figure 6: 实际跟踪误差对比（Actual Tracking Error Comparison）**

**插入位置**: Section 5.1.3（Cross-Platform Summary之后）

**在文件中搜索**: `\subsubsection{Cross-Platform Summary}`（约第610行）

**插入代码**：
```latex
These results demonstrate effective cross-platform generalization...

% ========== 在这里插入Figure 6 ==========
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.95\textwidth]{actual_tracking_comparison.png}
  \caption{Comprehensive tracking error analysis. (a) Mean absolute error comparison across both platforms. (b) Error distribution histograms showing tighter error distribution with RL adaptation. (c) Per-joint error breakdown for Franka Panda. (d) Cumulative distribution function (CDF) showing that Meta-PID+RL achieves lower errors with higher probability.}
  \label{fig:tracking_comparison}
\end{figure*}
% ========================================

\subsection{Robustness Under Disturbances}
```

**使用现有文件**: `actual_tracking_comparison.png`（已存在）

---

### **Figure 7: 扰动场景鲁棒性对比（Disturbance Robustness Comparison）**

**插入位置**: Section 5.2（Robustness Under Disturbances）的Key Observations之后

**在文件中搜索**: `\textbf{Key Observations:}`（约第645行）

**插入代码**：
```latex
\textbf{Key Observations:}

% ========== 在这里插入Figure 7 ==========
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\columnwidth]{disturbance_comparison.png}
  \caption{Robustness evaluation under various disturbance scenarios. The proposed Meta-PID+RL method shows significant improvements under parameter uncertainties (+23.1\%) and mixed disturbances (+21.1\%), validating its ability to adapt to model errors. The slight degradation under random force highlights a direction for future work.}
  \label{fig:robustness}
\end{figure}
% ========================================

\begin{enumerate}
    \item \textbf{Parameter Uncertainty:} The most substantial improvement...
```

**使用现有文件**: `disturbance_comparison.png`（已存在）

---

### **Figure 8: 消融实验对比（Ablation Study Results）**

**插入位置**: Section 5.5.3（Component Analysis之后）

**在文件中搜索**: `\subsubsection{Component Analysis}`（约第738行）

**插入代码**：
```latex
This demonstrates that all components are essential for optimal performance.

% ========== 在这里插入Figure 8 ==========
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\columnwidth]{ablation_study.png}
  \caption{Ablation study results showing the contribution of each component. Removing any component (meta-learning, data augmentation, or RL adaptation) leads to significant performance degradation. The full method achieves the best MAE of 5.37° on Franka Panda.}
  \label{fig:ablation}
\end{figure}
% ========================================

\section{Discussion}
```

**图片内容**：
```
柱状图：
- RL from scratch: 失败（无法收敛）
- w/o Data Augmentation: 31.2% error
- w/o RL Adaptation: 7.08° MAE
- Full Method: 5.37° MAE (最佳)
```

---

## 🎨 图表制作建议

### **推荐工具**：

#### **1. Python + Matplotlib（推荐）** ⭐⭐⭐⭐⭐
```python
import matplotlib.pyplot as plt
import numpy as np

# 设置学术风格
plt.style.use('seaborn-paper')
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Times New Roman'

# 绘图示例
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, label='Meta-PID', linewidth=2)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Mean Reward')
ax.legend()
ax.grid(True, alpha=0.3)

# 保存高分辨率
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

**优点**：
- 完全控制
- 高质量输出
- 可重现
- 数据已有（`.npz`, `.json`文件）

---

#### **2. PowerPoint/Keynote（系统架构图）** ⭐⭐⭐⭐
```
适用：Figure 1, Figure 2
优点：
- 易于绘制流程图
- 支持矢量图形
- 方便调整布局

导出：
- 保存为PNG（300 DPI）
- 或保存为PDF（更好）
```

---

#### **3. draw.io / Lucidchart（流程图）** ⭐⭐⭐⭐⭐
```
适用：Figure 1, Figure 2
优点：
- 在线工具，免费
- 丰富的模板
- 专业流程图

访问：https://app.diagrams.net/
```

---

### **图片格式要求**：

| 格式 | 用途 | 分辨率 | 推荐 |
|------|------|--------|------|
| PNG | 截图、图表 | 300 DPI | ✅ |
| PDF | 矢量图 | 矢量 | ⭐⭐⭐⭐⭐ |
| EPS | 矢量图 | 矢量 | ⭐⭐⭐⭐ |
| JPG | 照片 | 300 DPI | ⚠️ 避免 |

**推荐**：使用**PDF格式**（矢量），质量最好，文件最小。

---

### **配色建议（学术风格）**：

```python
# 推荐配色方案
COLORS = {
    'meta_pid': '#1f77b4',      # 蓝色
    'meta_rl': '#2ca02c',       # 绿色
    'baseline': '#d62728',      # 红色
    'reference': '#ff7f0e',     # 橙色
}

# 使用示例
plt.plot(x, y1, color=COLORS['meta_pid'], label='Meta-PID')
plt.plot(x, y2, color=COLORS['meta_rl'], label='Meta-PID+RL')
```

---

## 🔧 LaTeX图表位置控制

### **位置参数说明**：

```latex
\begin{figure}[htbp]
```

| 参数 | 含义 | 效果 |
|------|------|------|
| `h` | here | 尽量放在当前位置 |
| `t` | top | 页面顶部 |
| `b` | bottom | 页面底部 |
| `p` | page | 单独页面 |
| `!` | override | 强制忽略LaTeX限制 |

**推荐组合**：
- 小图：`[htbp]`（灵活放置）
- 大图：`[t]`（页面顶部）
- 跨栏：`[t]`（避免版式问题）

---

## 📝 快速操作步骤

### **步骤1: 准备图片文件**

```bash
# 将所有图片放到与LaTeX文件相同目录
cd /home/wujiahao/.../meta_learning/

# 检查现有图片
ls *.png

# 需要创建的图片（优先级高）：
# 1. system_architecture.png （Figure 1 - 必须）
# 2. data_augmentation_flow.png （Figure 2 - 必须）
# 3. per_joint_error.png （Figure 5 - 强烈建议）
# 4. ablation_study.png （Figure 8 - 强烈建议）
```

---

### **步骤2: 编辑LaTeX文件**

**使用Overleaf（推荐）**：
```
1. 访问 https://www.overleaf.com
2. 注册/登录
3. 上传 els-cas-templates/ 文件夹
4. 上传 论文_RAS_CAS格式.tex
5. 上传所有图片文件（*.png）
6. 设置主文档为 论文_RAS_CAS格式.tex
7. 点击 "Recompile"
8. 实时预览，边改边看
```

**或使用本地编辑**：
```bash
# 使用VS Code
cd /home/wujiahao/.../meta_learning/
code 论文_RAS_CAS格式.tex

# 按照上面的"具体插入位置"，在相应位置添加图表代码
# 保存文件

# 编译
./编译CAS论文.sh

# 查看结果
xdg-open /home/wujiahao/.../els-cas-templates/论文_RAS_CAS格式.pdf
```

---

### **步骤3: 插入图表代码**

**示例：插入Figure 1**

1. 打开 `论文_RAS_CAS格式.tex`
2. 搜索 `\subsection{Hierarchical Meta-RL Architecture}` (Ctrl+F)
3. 在该行下方插入：
```latex
\subsection{Hierarchical Meta-RL Architecture}

Our framework consists of two complementary components operating at different timescales:

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.95\textwidth]{system_architecture.png}
  \caption{Hierarchical Meta-RL architecture for adaptive PID control.}
  \label{fig:architecture}
\end{figure*}

\subsubsection{Stage 1: Meta-Learning for PID Initialization}
```
4. 保存文件
5. 重新编译

---

### **步骤4: 验证图表引用**

在论文中引用图表：
```latex
% 在需要引用的地方
As shown in Figure~\ref{fig:architecture}, our hierarchical architecture...

% LaTeX会自动生成图号
```

**检查**：
- 编译后PDF中图号是否正确
- 引用是否显示为 "Figure 1" 而不是 "??"
- 图片是否清晰

---

## ⚠️ 常见问题与解决

### **问题1: 图片找不到**

```
Error: File 'system_architecture.png' not found
```

**解决**：
```bash
# 确保图片文件在正确位置
# 对于Overleaf：上传图片到项目根目录
# 对于本地：将图片放在与.tex文件同一目录

# 检查文件名（区分大小写）
ls *.png
```

---

### **问题2: 图片位置不对**

**解决**：
```latex
% 方法1: 使用!强制放置
\begin{figure}[!htbp]

% 方法2: 使用float包
\usepackage{float}
\begin{figure}[H]  % H = 严格此处

% 方法3: 调整位置参数
\begin{figure}[t]  % 只放顶部
```

---

### **问题3: 图片太大/太小**

**解决**：
```latex
% 调整宽度
\includegraphics[width=0.9\columnwidth]{图片.png}  % 列宽90%
\includegraphics[width=0.8\textwidth]{图片.png}    % 文本宽度80%
\includegraphics[width=8cm]{图片.png}             % 固定8cm

% 调整高度
\includegraphics[height=6cm]{图片.png}

% 按比例缩放
\includegraphics[scale=0.8]{图片.png}
```

---

### **问题4: 图片不清晰**

**解决**：
1. 使用更高分辨率图片（≥300 DPI）
2. 使用矢量格式（PDF, EPS）
3. 重新生成图片时设置：
```python
plt.savefig('figure.pdf', dpi=300, bbox_inches='tight')  # 推荐
# 或
plt.savefig('figure.png', dpi=600, bbox_inches='tight')  # 超高清
```

---

### **问题5: 跨栏图表显示异常**

**解决**：
```latex
% 使用figure*环境（注意星号）
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.95\textwidth]{wide_figure.png}
  \caption{跨栏图}
  \label{fig:wide}
\end{figure*}
```

---

## 📚 参考资源

### **LaTeX学习资源**：

1. **Overleaf文档（推荐）**：
   - https://www.overleaf.com/learn
   - 中文教程：https://www.overleaf.com/learn/latex/Chinese

2. **LaTeX Wikibook**：
   - https://en.wikibooks.org/wiki/LaTeX

3. **CAS模板示例**：
   - 查看：`els-cas-templates/cas-dc-sample.pdf`
   - 源码：`els-cas-templates/cas-dc-sample.tex`

---

### **图表制作参考**：

1. **Matplotlib Gallery**：
   - https://matplotlib.org/stable/gallery/index.html

2. **Scientific Figure Design**：
   - 十大科学图表原则

3. **Nature图表指南**：
   - https://www.nature.com/nature/for-authors/preparing-your-submission/figures

---

## ✅ 检查清单

投稿前图表检查：

```
图表内容：
[ ] 所有图表编号正确（Figure 1-8）
[ ] 所有图表都有清晰的caption
[ ] 图表质量≥300 DPI
[ ] 图表在文中都有引用
[ ] 图表位置合理（靠近引用处）

图表质量：
[ ] 字体大小合适（≥8pt）
[ ] 坐标轴标签清晰
[ ] 图例完整
[ ] 颜色对比明显
[ ] 线条粗细适中

LaTeX编译：
[ ] 无编译错误
[ ] 无图片缺失警告
[ ] PDF正确生成
[ ] 图表引用显示为数字（非??）
```

---

## 🎯 总结

### **立即行动计划**：

**今天（1-2小时）**：
1. ✅ 熟悉LaTeX基本语法
2. ⚠️ 创建Figure 1（系统架构图）- 最重要
3. ⚠️ 创建Figure 2（数据增强流程图）

**本周（3-5小时）**：
4. ⚠️ 创建Figure 5（逐关节误差图）
5. ⚠️ 创建Figure 8（消融实验图）
6. ✅ 插入所有图表到LaTeX
7. ✅ 编译并检查

**投稿前（1小时）**：
8. ✅ 最终图表质量检查
9. ✅ 确认所有引用正确
10. ✅ 生成最终PDF

---

**您现在可以**：
1. 使用Overleaf打开论文（最简单）
2. 或使用本地编辑器（VS Code/TeXstudio）
3. 按照本文档的"具体插入位置"部分，逐个插入图表
4. 边改边编译，实时查看效果

**祝您论文顺利完成！** 🎉📊🚀

