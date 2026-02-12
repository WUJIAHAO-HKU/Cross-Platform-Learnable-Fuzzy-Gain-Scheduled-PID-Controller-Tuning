#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                  📝 编译Elsevier CAS格式论文                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 定义路径
PAPER_FILE="论文_RAS_CAS格式.tex"
TEMPLATE_DIR="/home/wujiahao/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/els-cas-templates"
META_LEARNING_DIR="/home/wujiahao/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/rl_pid_linux/meta_learning"

echo "📂 步骤1: 复制论文到模板目录..."
cp "$META_LEARNING_DIR/$PAPER_FILE" "$TEMPLATE_DIR/"
echo "   ✅ 完成"
echo ""

echo "📂 步骤2: 进入模板目录..."
cd "$TEMPLATE_DIR"
echo "   当前目录: $(pwd)"
echo ""

echo "🔧 步骤3: 编译论文（第1次 - pdflatex）..."
pdflatex -interaction=nonstopmode "$PAPER_FILE" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ 第1次编译成功"
else
    echo "   ⚠️  第1次编译有警告（通常正常）"
fi
echo ""

echo "🔧 步骤4: 处理参考文献（bibtex）..."
bibtex "论文_RAS_CAS格式" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ BibTeX处理成功"
else
    echo "   ⚠️  BibTeX处理失败（如果没有.bib文件是正常的）"
fi
echo ""

echo "🔧 步骤5: 编译论文（第2次 - pdflatex）..."
pdflatex -interaction=nonstopmode "$PAPER_FILE" > /dev/null 2>&1
echo "   ✅ 第2次编译完成"
echo ""

echo "🔧 步骤6: 编译论文（第3次 - pdflatex，最终）..."
pdflatex -interaction=nonstopmode "$PAPER_FILE" > /dev/null 2>&1
echo "   ✅ 第3次编译完成"
echo ""

# 检查PDF是否生成
if [ -f "论文_RAS_CAS格式.pdf" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ 编译成功！"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📄 PDF文件位置:"
    echo "   $TEMPLATE_DIR/论文_RAS_CAS格式.pdf"
    echo ""
    
    # 获取PDF文件大小
    PDF_SIZE=$(du -h "论文_RAS_CAS格式.pdf" | cut -f1)
    echo "📊 PDF文件大小: $PDF_SIZE"
    echo ""
    
    echo "🖥️  要查看PDF，运行:"
    echo "   xdg-open $TEMPLATE_DIR/论文_RAS_CAS格式.pdf"
    echo ""
    
    echo "📋 生成的文件:"
    ls -lh 论文_RAS_CAS格式.* | awk '{print "   " $9 " (" $5 ")"}'
    echo ""
    
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "❌ 编译失败！"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "请检查错误日志:"
    echo "   $TEMPLATE_DIR/论文_RAS_CAS格式.log"
    echo ""
    echo "常见问题:"
    echo "   1. 检查LaTeX是否正确安装: pdflatex --version"
    echo "   2. 检查CAS模板文件是否存在"
    echo "   3. 查看日志文件获取详细错误信息"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

