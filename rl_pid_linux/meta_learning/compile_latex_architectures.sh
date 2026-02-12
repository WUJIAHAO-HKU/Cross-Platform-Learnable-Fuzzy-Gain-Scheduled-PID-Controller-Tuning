#!/bin/bash
# ==============================================================================
# 自动编译所有网络架构LaTeX图表
# 需要: pdflatex, imagemagick (可选，用于PNG转换)
# ==============================================================================

set -e  # 遇到错误立即停止

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=============================================================================="
echo -e "${BLUE}📊 编译所有网络架构LaTeX图表${NC}"
echo "=============================================================================="
echo ""

# 检查pdflatex是否安装
if ! command -v pdflatex &> /dev/null; then
    echo -e "${RED}❌ 错误: pdflatex 未安装！${NC}"
    echo ""
    echo "请安装LaTeX："
    echo "  Ubuntu/Debian: sudo apt-get install texlive-latex-extra"
    echo "  或使用Overleaf在线编译：https://www.overleaf.com"
    exit 1
fi

echo -e "${GREEN}✅ pdflatex 已安装${NC}"
echo ""

# 定义要编译的文件
FILES=(
    "meta_pid_network_architecture"
    "rl_policy_network_architecture"
    "complete_hierarchical_framework"
)

NAMES=(
    "Meta-PID Network Architecture"
    "RL Policy Network Architecture"
    "Complete Hierarchical Framework"
)

# 编译每个文件
SUCCESS_COUNT=0
TOTAL_COUNT=${#FILES[@]}

for i in "${!FILES[@]}"; do
    FILE="${FILES[$i]}"
    NAME="${NAMES[$i]}"
    
    echo -e "${YELLOW}📝 [$((i+1))/$TOTAL_COUNT] 编译: ${NAME}${NC}"
    echo "   文件: ${FILE}.tex"
    
    # 编译LaTeX（隐藏详细输出）
    if pdflatex -interaction=nonstopmode "${FILE}.tex" > /dev/null 2>&1; then
        echo -e "   ${GREEN}✅ PDF生成成功: ${FILE}.pdf${NC}"
        
        # 获取文件大小
        PDF_SIZE=$(du -h "${FILE}.pdf" | cut -f1)
        echo "   📏 文件大小: ${PDF_SIZE}"
        
        SUCCESS_COUNT=$((SUCCESS_COUNT+1))
        
        # 如果安装了ImageMagick，同时生成PNG
        if command -v convert &> /dev/null; then
            echo "   🔄 转换为PNG (300 DPI)..."
            if convert -density 300 "${FILE}.pdf" -quality 100 "${FILE}.png" 2>/dev/null; then
                PNG_SIZE=$(du -h "${FILE}.png" | cut -f1)
                echo -e "   ${GREEN}✅ PNG生成成功: ${FILE}.png (${PNG_SIZE})${NC}"
            else
                echo -e "   ${YELLOW}⚠️  PNG转换失败（PDF可用）${NC}"
            fi
        fi
    else
        echo -e "   ${RED}❌ 编译失败！${NC}"
        echo "   💡 提示: 查看 ${FILE}.log 文件了解详细错误"
        echo "   或尝试手动编译: pdflatex ${FILE}.tex"
    fi
    
    echo ""
done

# 清理临时文件
echo -e "${BLUE}🧹 清理临时文件...${NC}"
rm -f *.aux *.log *.out *.toc *.nav *.snm 2>/dev/null
echo -e "${GREEN}✅ 清理完成${NC}"
echo ""

# 输出总结
echo "=============================================================================="
if [ $SUCCESS_COUNT -eq $TOTAL_COUNT ]; then
    echo -e "${GREEN}✅ 所有图表编译成功！ ($SUCCESS_COUNT/$TOTAL_COUNT)${NC}"
else
    echo -e "${YELLOW}⚠️  部分图表编译失败 ($SUCCESS_COUNT/$TOTAL_COUNT)${NC}"
fi
echo "=============================================================================="
echo ""

# 列出生成的文件
echo -e "${BLUE}📁 生成的文件:${NC}"
echo ""

if ls *.pdf 1> /dev/null 2>&1; then
    echo "   PDF文件:"
    for pdf in *.pdf; do
        SIZE=$(du -h "$pdf" | cut -f1)
        echo "      • $pdf ($SIZE)"
    done
    echo ""
fi

if ls *.png 1> /dev/null 2>&1; then
    echo "   PNG文件:"
    for png in *.png; do
        # 跳过已存在的其他PNG文件
        if [[ "$png" == "meta_pid_network_architecture.png" ]] || \
           [[ "$png" == "rl_policy_network_architecture.png" ]] || \
           [[ "$png" == "complete_hierarchical_framework.png" ]]; then
            SIZE=$(du -h "$png" | cut -f1)
            echo "      • $png ($SIZE)"
        fi
    done
    echo ""
fi

# 下一步提示
echo "=============================================================================="
echo -e "${BLUE}📝 下一步操作:${NC}"
echo ""
echo "1. 查看生成的PDF文件："
echo "   evince meta_pid_network_architecture.pdf"
echo ""
echo "2. 插入到论文中（LaTeX）："
echo "   \includegraphics[width=0.95\textwidth]{meta_pid_network_architecture.pdf}"
echo ""
echo "3. 上传到Overleaf："
echo "   - 上传PDF文件到论文项目的figures/目录"
echo "   - 在论文中引用：\ref{fig:meta_pid_arch}"
echo ""
echo "4. 如需修改："
echo "   - 编辑对应的.tex文件"
echo "   - 重新运行此脚本编译"
echo ""
echo "=============================================================================="
echo ""

# 检查是否需要安装ImageMagick
if ! command -v convert &> /dev/null; then
    echo -e "${YELLOW}💡 提示: 安装ImageMagick可以自动生成PNG格式${NC}"
    echo "   sudo apt-get install imagemagick"
    echo ""
fi

exit 0

