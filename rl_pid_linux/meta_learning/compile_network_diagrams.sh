#!/bin/bash

# 编译神经网络架构图的脚本

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       编译神经网络架构图（PlotNeuralNet风格）                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo

cd "$(dirname "$0")"

# 检查依赖
echo "📦 检查依赖..."
if ! command -v pdflatex &> /dev/null; then
    echo "❌ pdflatex 未安装！"
    echo "   请安装: sudo apt-get install texlive-latex-base texlive-latex-extra"
    exit 1
fi

if ! command -v convert &> /dev/null; then
    echo "❌ ImageMagick convert 未安装！"
    echo "   请安装: sudo apt-get install imagemagick"
    exit 1
fi

echo "✅ 依赖检查通过"
echo

# 编译LaTeX
echo "🔨 编译 neural_network_architectures.tex ..."
pdflatex -interaction=nonstopmode neural_network_architectures.tex > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ PDF 生成成功: neural_network_architectures.pdf"
else
    echo "❌ PDF 编译失败！查看日志:"
    pdflatex neural_network_architectures.tex
    exit 1
fi

# 转换为PNG（高分辨率）
echo
echo "🖼️  转换为PNG图片（300 DPI）..."

# 提取第1页（Meta-Learning Network）
convert -density 300 -quality 100 \
    neural_network_architectures.pdf[0] \
    meta_learning_network.png

if [ $? -eq 0 ]; then
    echo "✅ Meta-Learning Network: meta_learning_network.png"
else
    echo "❌ PNG转换失败（第1页）"
fi

# 提取第2页（RL Network）
convert -density 300 -quality 100 \
    neural_network_architectures.pdf[1] \
    rl_adaptation_network.png

if [ $? -eq 0 ]; then
    echo "✅ RL Adaptation Network: rl_adaptation_network.png"
else
    echo "❌ PNG转换失败（第2页）"
fi

# 提取第3页（Complete System）
convert -density 300 -quality 100 \
    neural_network_architectures.pdf[2] \
    complete_system_architecture.png

if [ $? -eq 0 ]; then
    echo "✅ Complete System Architecture: complete_system_architecture.png"
else
    echo "❌ PNG转换失败（第3页）"
fi

# 清理临时文件
echo
echo "🧹 清理临时文件..."
rm -f neural_network_architectures.aux \
      neural_network_architectures.log \
      neural_network_architectures.out

echo "✅ 临时文件已清理"

# 显示结果
echo
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                 ✅ 编译完成                                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo
echo "生成的文件："
echo "  📄 neural_network_architectures.pdf        - 完整PDF（3页）"
echo "  🖼️  meta_learning_network.png              - 元学习网络架构图"
echo "  🖼️  rl_adaptation_network.png              - RL在线调整架构图"
echo "  🖼️  complete_system_architecture.png       - 完整系统架构图"
echo
echo "文件大小："
ls -lh meta_learning_network.png 2>/dev/null | awk '{print "  meta_learning_network.png:       " $5}'
ls -lh rl_adaptation_network.png 2>/dev/null | awk '{print "  rl_adaptation_network.png:       " $5}'
ls -lh complete_system_architecture.png 2>/dev/null | awk '{print "  complete_system_architecture.png:" $5}'
echo
echo "🎯 下一步: 将PNG图片插入到论文中"
echo

