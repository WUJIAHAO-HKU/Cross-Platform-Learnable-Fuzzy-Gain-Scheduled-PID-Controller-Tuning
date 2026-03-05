#!/usr/bin/env python3
"""
检查所有文献是否在正文中被引用
"""

import re

# 读取LaTeX文件
with open('meta_rl_pid_control_manuscript_with_highlight.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 提取所有\bibitem的标签
bibitem_pattern = r'\\bibitem\{([^}]+)\}'
all_bibitems = re.findall(bibitem_pattern, content)

print(f"文献总数: {len(all_bibitems)}")
print("=" * 80)

# 提取bibliography之前的正文部分
bib_start = content.find(r'\begin{thebibliography}')
main_text = content[:bib_start]

# 检查每个文献是否被引用
cited = []
uncited = []

for bibitem in all_bibitems:
    # 检查是否有\cite{bibitem}引用
    cite_pattern = r'\\cite\{[^}]*\b' + re.escape(bibitem) + r'\b[^}]*\}'
    if re.search(cite_pattern, main_text):
        cited.append(bibitem)
    else:
        uncited.append(bibitem)

print(f"\n已引用文献: {len(cited)}")
print(f"未引用文献: {len(uncited)}")
print("\n" + "=" * 80)

if uncited:
    print("\n未引用的文献列表:")
    print("-" * 80)
    for i, ref in enumerate(uncited, 1):
        print(f"{i:2d}. {ref}")
else:
    print("\n✓ 所有文献都已在正文中引用")

print("\n" + "=" * 80)
print("\n已引用的文献列表:")
print("-" * 80)
for i, ref in enumerate(cited, 1):
    print(f"{i:2d}. {ref}")
