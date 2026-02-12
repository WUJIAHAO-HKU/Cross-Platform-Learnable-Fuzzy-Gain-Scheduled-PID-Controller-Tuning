import re

# 读取文件
with open('meta_rl_pid_control_manuscript_with_highlight.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 提取参考文献列表(从\bibitem开始)
bib_section = content[content.find('\\begin{thebibliography}'):content.find('\\end{thebibliography}')]
bibitem_pattern = re.compile(r'\\bibitem\{([^}]+)\}')
bib_keys = bibitem_pattern.findall(bib_section)

print(f"参考文献总数: {len(bib_keys)}")
print(f"\n参考文献列表:")
for i, key in enumerate(bib_keys, 1):
    print(f"{i}. {key}")

# 提取正文部分(从\section{Introduction}到\bibliographystyle)
main_text = content[content.find('\\section{Introduction}'):content.find('\\bibliographystyle')]

# 提取正文中所有引用
cite_pattern = re.compile(r'\\cite\{([^}]+)\}')
all_cites = cite_pattern.findall(main_text)

# 处理多个引用的情况(如\cite{a,b,c})
cited_keys = []
for cite in all_cites:
    cited_keys.extend([k.strip() for k in cite.split(',')])

# 记录每个引用第一次出现的位置
first_cite_positions = {}
for cite in all_cites:
    for key in cite.split(','):
        key = key.strip()
        if key not in first_cite_positions:
            pos = main_text.find(f'\\cite{{{cite}}}')
            first_cite_positions[key] = pos

print(f"\n正文中引用总数: {len(set(cited_keys))}")

# 检查未被引用的文献
uncited = []
for key in bib_keys:
    if key not in cited_keys:
        uncited.append(key)

if uncited:
    print(f"\n⚠️  未被引用的文献 ({len(uncited)}个):")
    for key in uncited:
        print(f"  - {key}")
else:
    print("\n✅ 所有参考文献都被引用")

# 检查引用顺序
print("\n\n正文引用顺序:")
ordered_cites = sorted(first_cite_positions.items(), key=lambda x: x[1])
for i, (key, pos) in enumerate(ordered_cites, 1):
    print(f"{i}. {key}")

# 检查参考文献列表顺序是否与引用顺序一致
print("\n\n顺序检查:")
mismatches = []
for i, (cited_key, _) in enumerate(ordered_cites):
    if i < len(bib_keys):
        if bib_keys[i] != cited_key:
            mismatches.append((i+1, bib_keys[i], cited_key))

if mismatches:
    print(f"❌ 发现顺序不匹配 ({len(mismatches)}处):")
    for pos, bib_key, cite_key in mismatches:
        print(f"  位置{pos}: 参考文献列表为'{bib_key}', 应该是'{cite_key}'")
else:
    print("✅ 参考文献顺序与引用顺序一致")
