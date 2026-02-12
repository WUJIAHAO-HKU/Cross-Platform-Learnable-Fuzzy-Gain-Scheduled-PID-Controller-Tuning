# MAE数据严重不一致问题报告

## 执行时间
2025-11-02

## 问题描述

用户发现**Table 1/2（Overall MAE）**和**Table 3（Per-joint MAE）**的数据存在巨大差异：

- **Table 1 Overall MAE**: 28.67° → 24.88°
- **Table 3 Per-joint平均**: 7.51° → 6.26°
- **差距**: **3.82倍！**

---

## 数据验证

### Table 1: Franka Panda Overall Performance

| 指标 | Pure Meta-PID | Meta-PID+RL | 改进 |
|------|---------------|-------------|------|
| MAE | 28.67° | 24.88° | +13.2% |
| RMSE | 29.32° | 25.45° | +13.2% |

### Table 3: Per-Joint Error Comparison

| 关节 | Pure Meta-PID | Meta-PID+RL | 改进 |
|------|---------------|-------------|------|
| J1 | 2.57° | 2.26° | +12.2% |
| J2 | 12.36° | 2.42° | +80.4% |
| J3 | 4.10° | 3.87° | +5.7% |
| J4 | 6.78° | 6.49° | +4.3% |
| J5 | 5.41° | 5.32° | +1.6% |
| J6 | 4.31° | 4.19° | +2.8% |
| J7 | 11.45° | 11.26° | +1.6% |
| J8 | 10.23° | 10.19° | +0.3% |
| J9 | 10.36° | 10.33° | +0.3% |
| **Average** | **67.57/9 = 7.51°** | **56.33/9 = 6.26°** | **+16.6%** |

### 差异分析

```
Table 1 Overall MAE: 28.67°
Table 3 Per-joint平均: 7.51°
比例: 28.67 / 7.51 = 3.82x
```

---

## 数学分析

### Overall MAE定义（论文Line 363）
```
MAE_overall = (1/T) Σ_t [(1/n) Σ_i |e_i(t)|]
```

### Per-joint MAE定义（论文Line 371）
```
MAE_joint_i = (1/T) Σ_t |e_i(t)|
```

### Per-joint MAE的平均
```
Average(MAE_joint) = (1/n) Σ_i [(1/T) Σ_t |e_i(t)|]
                   = (1/n) Σ_i [MAE_joint_i]
```

### 数学等价性验证
```
MAE_overall = (1/T) Σ_t [(1/n) Σ_i |e_i(t)|]
            = (1/nT) Σ_t,i |e_i(t)|

Average(MAE_joint) = (1/n) Σ_i [(1/T) Σ_t |e_i(t)|]
                   = (1/nT) Σ_t,i |e_i(t)|
```

**结论：数学上，这两个值应该完全相等！**

---

## 问题根源分析

### 可能原因1：来自不同的测试数据
- Table 1可能使用了不同的测试episode
- Table 3的per-joint数据来自`generate_per_joint_comparison.py`
- 需要确认数据来源

### 可能原因2：MAE计算方法不同
目前论文中的定义（Line 363）：
```latex
\item \textbf{Mean Absolute Error (MAE):} 
  $\frac{1}{T}\sum_{t=1}^{T}\left(\frac{1}{n}\sum_{i=1}^{n}|e_i(t)|\right)$ 
  - time-averaged mean per-joint error
```

但如果Table 1的MAE实际上是：
```
MAE_table1 = (1/T) Σ_t ||e(t)||_1  （L1 norm，不除以n）
```

那么：
```
MAE_table1 = n × MAE_overall
          ≈ 9 × 7.51 = 67.6°
```

但实际Table 1是28.67°，不是67.6°！

### 可能原因3：RMSE定义错误导致的误解

从Table 1的RMSE = 29.32°来看，这与MAE = 28.67°非常接近。

如果按照Line 364的定义：
```
RMSE = (1/T) Σ_t ||e(t)||_2
```

对于9个关节，如果每个关节平均误差约7.51°，那么：
```
||e||_2 ≈ √(9 × 7.51²) ≈ 22.5°
```

这比29.32°要小，说明per-joint误差的分布不均匀（J2特别大）。

---

## 改进率不一致

### Table 1改进率
```
(28.67 - 24.88) / 28.67 × 100% = 13.2%
```

### Table 3 Per-joint改进率
```
(7.51 - 6.26) / 7.51 × 100% = 16.6%
```

**改进率相差3.4个百分点！**

这进一步证明这两个数据来自**不同的测试**或使用了**不同的计算方法**。

---

## 问题严重性

### 高严重性 ⚠️⚠️⚠️

1. **数据一致性**：论文的核心性能数据自相矛盾
2. **可重复性**：读者无法验证哪个数据是正确的
3. **学术诚信**：可能被审稿人质疑数据的可靠性

---

## 建议解决方案

### 方案1：统一数据来源（推荐） ✅

1. **重新运行测试**：
   - 使用相同的测试脚本（推荐`generate_per_joint_comparison.py`）
   - 同时生成Table 1的overall metrics和Table 3的per-joint metrics
   - 确保两者来自完全相同的测试数据

2. **更新论文数据**：
   - 将Table 1的MAE更新为与Table 3一致的值
   - 如果overall MAE确实是7.51°，那么需要相应调整：
     - Abstract中的"24.88° MAE"
     - 所有引用28.67°和24.88°的地方

3. **验证RMSE的正确性**：
   - 检查RMSE的计算是否正确
   - 确保RMSE与MAE的关系合理

### 方案2：明确区分两种MAE

如果Table 1和Table 3确实使用了不同的MAE定义，需要：

1. **在论文中明确说明**：
   - Table 1的MAE是"joint space error的平均"（某种加权或norm）
   - Table 3的MAE是"per-joint平均误差的简单平均"

2. **更新所有引用**：
   - 在提到28.67°时，明确说明这是"overall joint space MAE"
   - 在提到per-joint数据时，说明这是"individual joint MAE"

3. **添加换算关系**：
   - 解释为什么overall MAE (28.67°) 是per-joint平均 (7.51°) 的3.82倍

但这种方案**不推荐**，因为会让读者困惑，且缺乏物理意义。

---

## 待办事项

- [ ] 确认Table 1和Table 3的数据来源
- [ ] 重新运行测试生成一致的数据
- [ ] 更新论文中所有相关数值
- [ ] 添加明确的MAE计算公式和说明
- [ ] 验证RMSE与MAE的数学关系

---

## 联系点

如需进一步讨论，请参考：
- `generate_per_joint_comparison.py` - Per-joint数据生成脚本
- `verify_mae_calculation.py` - MAE计算验证脚本
- `论文_RAS_CAS格式.tex` Line 359-375 - MAE定义部分

