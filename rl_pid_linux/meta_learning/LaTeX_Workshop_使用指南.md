# VS Code LaTeX Workshop 使用指南

## ❌ 问题诊断

您遇到的"预览PDF无内容"问题，通常是因为以下原因之一：

### **问题1: CAS模板文件缺失** ⭐⭐⭐⭐⭐（最可能）

**原因**: 
- 您的论文使用了 `cas-dc` 类（Elsevier CAS格式）
- 这需要以下文件在同一目录：
  - `cas-dc.cls` 
  - `cas-common.sty`
  - `cas-model2-names.bst`
- 这些文件在 `els-cas-templates/` 目录中，但不在 `meta_learning/` 目录

**症状**:
- 编译失败
- PDF无法生成
- 错误日志中有 "cas-dc.cls not found"


### **问题2: 编译未完成**

**原因**: 
- LaTeX需要多次编译（pdflatex → bibtex → pdflatex × 2）
- VS Code可能只编译了一次

**症状**:
- 看到PDF但内容不完整
- 图表编号显示为 "??"


### **问题3: 输出目录配置错误**

**原因**: 
- LaTeX Workshop可能将PDF输出到其他目录

**症状**:
- 编译成功但找不到PDF


---

## ✅ 解决方案

### **方案A: 复制CAS模板文件到当前目录（推荐）** ⭐⭐⭐⭐⭐

**立即执行**:
```bash
cd /home/wujiahao/.../meta_learning/

# 复制CAS模板文件
cp /home/wujiahao/.../els-cas-templates/cas-dc.cls .
cp /home/wujiahao/.../els-cas-templates/cas-common.sty .
cp /home/wujiahao/.../els-cas-templates/cas-model2-names.bst .

# 检查文件
ls cas-*
```

**然后在VS Code中**:
1. 打开 `论文_RAS_CAS格式.tex`
2. 按 `Ctrl+S` 保存（触发自动编译）
3. 等待编译完成（右下角会显示进度）
4. 点击右上角的 "View LaTeX PDF" 图标
5. 或按 `Ctrl+Alt+V` 预览PDF


---

### **方案B: 在els-cas-templates目录下编辑（更简单）** ⭐⭐⭐⭐⭐

**步骤**:
```bash
# 复制论文到模板目录
cp /home/wujiahao/.../meta_learning/论文_RAS_CAS格式.tex \
   /home/wujiahao/.../els-cas-templates/

# 在VS Code中打开模板目录
code /home/wujiahao/.../els-cas-templates/
```

**然后**:
1. 在VS Code中打开 `论文_RAS_CAS格式.tex`
2. 编辑并保存
3. 自动编译和预览


---

### **方案C: 使用命令行编译（最可靠）**

**如果VS Code仍然有问题**:
```bash
cd /home/wujiahao/.../meta_learning/
./编译CAS论文.sh

# 查看生成的PDF
xdg-open /home/wujiahao/.../els-cas-templates/论文_RAS_CAS格式.pdf
```


---

## 🔍 如何检查编译是否成功

### **方法1: 查看编译日志**

在VS Code中:
1. 打开 **Output** 面板（View → Output）
2. 选择 **LaTeX Workshop**
3. 查看编译日志

**成功的标志**:
```
Output written on 论文_RAS_CAS格式.pdf (XX pages, XXXXX bytes).
Transcript written on 论文_RAS_CAS格式.log.
```

**失败的标志**:
```
! LaTeX Error: File `cas-dc.cls' not found.
```


### **方法2: 检查生成的文件**

```bash
cd /home/wujiahao/.../meta_learning/
ls -lh 论文_RAS_CAS格式.*

# 应该看到:
# 论文_RAS_CAS格式.tex  (源文件)
# 论文_RAS_CAS格式.pdf  (输出文件)
# 论文_RAS_CAS格式.log  (日志文件)
# 论文_RAS_CAS格式.aux  (辅助文件)
```


---

## 🛠️ VS Code LaTeX Workshop 使用技巧

### **快捷键**:

| 操作 | 快捷键 |
|------|--------|
| 编译 | `Ctrl+Alt+B` |
| 预览PDF | `Ctrl+Alt+V` |
| 清理辅助文件 | `Ctrl+Alt+C` |
| 同步PDF位置 | `Ctrl+点击` |


### **自动编译设置**:

在 `.vscode/settings.json` 中（已配置）:
```json
{
    "latex-workshop.latex.autoBuild.run": "onSave"
}
```

**这意味着**: 每次保存 `.tex` 文件时自动编译


### **查看PDF的3种方式**:

1. **内置Tab预览**（推荐）:
   - 点击右上角 📄 图标
   - 或 `Ctrl+Alt+V`

2. **独立窗口预览**:
   - 设置: `"latex-workshop.view.pdf.viewer": "external"`
   - 使用系统默认PDF阅读器

3. **浏览器预览**:
   - 设置: `"latex-workshop.view.pdf.viewer": "browser"`


---

## ⚠️ 常见错误及解决

### **错误1: cas-dc.cls not found**

```
! LaTeX Error: File `cas-dc.cls' not found.
```

**解决**: 执行方案A（复制CAS文件）


---

### **错误2: 找不到图片文件**

```
! LaTeX Error: File `figure.png' not found.
```

**解决**: 
```bash
# 确保图片在正确位置
cd /home/wujiahao/.../meta_learning/
ls *.png

# 如果缺少图片，先生成
python3 生成论文图表.py
```


---

### **错误3: Undefined control sequence**

```
! Undefined control sequence.
l.123 \citep
```

**原因**: 缺少某个包或命令拼写错误

**解决**: 
- 检查 `\usepackage{}` 是否正确
- 检查命令拼写


---

### **错误4: PDF预览显示旧版本**

**原因**: PDF文件被缓存

**解决**:
1. 关闭PDF预览
2. 删除 `.pdf` 文件
3. 重新编译
4. 或重启VS Code


---

## 📋 完整诊断步骤

如果上述方案都不行，按以下步骤诊断：

### **步骤1: 检查LaTeX安装**

```bash
pdflatex --version

# 应该显示版本信息，如:
# pdfTeX 3.141592653-2.6-1.40.24 (TeX Live 2022)
```

**如果未安装**:
```bash
sudo apt-get update
sudo apt-get install texlive-full
```


---

### **步骤2: 手动编译测试**

```bash
cd /home/wujiahao/.../meta_learning/

# 复制CAS文件
cp /home/wujiahao/.../els-cas-templates/cas-*.* .

# 手动编译
pdflatex 论文_RAS_CAS格式.tex

# 查看是否生成PDF
ls -lh 论文_RAS_CAS格式.pdf
```


---

### **步骤3: 查看详细错误日志**

```bash
cd /home/wujiahao/.../meta_learning/
cat 论文_RAS_CAS格式.log | grep -i error
```


---

### **步骤4: 重置VS Code LaTeX Workshop**

在VS Code中:
1. 按 `Ctrl+Shift+P`
2. 输入 "LaTeX Workshop: Clean up auxiliary files"
3. 执行清理
4. 保存 `.tex` 文件重新编译


---

## 🎯 推荐工作流

### **方式1: 使用Overleaf（最简单）** ⭐⭐⭐⭐⭐

**优点**:
- 无需本地配置
- 自动编译和预览
- 不会有环境问题

**步骤**:
1. 访问 https://www.overleaf.com
2. 上传整个 `els-cas-templates/` 文件夹
3. 上传 `论文_RAS_CAS格式.tex`
4. 上传所有图片
5. 设置主文档并编译


---

### **方式2: 使用命令行编译 + PDF阅读器查看**

**步骤**:
```bash
# 编辑
code /home/wujiahao/.../meta_learning/论文_RAS_CAS格式.tex

# 编译
cd /home/wujiahao/.../meta_learning/
./编译CAS论文.sh

# 查看
xdg-open /home/wujiahao/.../els-cas-templates/论文_RAS_CAS格式.pdf
```

**优点**:
- 编译可靠
- 不依赖VS Code插件


---

### **方式3: VS Code + LaTeX Workshop（已配置好）**

**前提**: 已执行方案A或B

**步骤**:
1. 在VS Code中打开 `论文_RAS_CAS格式.tex`
2. 编辑内容
3. 按 `Ctrl+S` 保存（自动编译）
4. 按 `Ctrl+Alt+V` 预览PDF


---

## 📞 如果还有问题

### **检查清单**:

```
[ ] 已安装 pdflatex (texlive)
[ ] 已复制 cas-dc.cls 等文件到当前目录
[ ] 已在VS Code中安装 LaTeX Workshop
[ ] 已创建 .vscode/settings.json 配置
[ ] 保存 .tex 文件后看到编译进度
[ ] 编译日志中没有错误
[ ] 当前目录存在 .pdf 文件
```


### **获取帮助**:

1. **查看完整错误日志**:
   ```bash
   cat 论文_RAS_CAS格式.log | less
   ```

2. **查看LaTeX Workshop输出**:
   - VS Code: View → Output
   - 选择 "LaTeX Workshop"

3. **尝试最简单的测试**:
   ```bash
   cd /home/wujiahao/.../els-cas-templates/
   cp cas-dc-sample.tex test.tex
   pdflatex test.tex
   # 如果这个能编译成功，说明环境没问题
   ```


---

## ✅ 总结

**立即执行（2分钟）**:
```bash
cd /home/wujiahao/.../meta_learning/
cp /home/wujiahao/.../els-cas-templates/cas-*.* .
code 论文_RAS_CAS格式.tex
# 在VS Code中按 Ctrl+S 保存，然后 Ctrl+Alt+V 预览
```

**如果还不行**:
- 使用 `./编译CAS论文.sh` 命令行编译（100%可靠）
- 或使用 Overleaf 在线编辑（最简单）

