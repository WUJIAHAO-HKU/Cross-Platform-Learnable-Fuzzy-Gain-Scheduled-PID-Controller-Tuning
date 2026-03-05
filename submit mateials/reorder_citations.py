import re

# 读取文件
with open('meta_rl_pid_control_manuscript_with_highlight.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 提取参考文献部分
bib_start = content.find('\\begin{thebibliography}')
bib_end = content.find('\\end{thebibliography}')
bib_section = content[bib_start:bib_end]

# 提取所有\bibitem条目
bibitem_pattern = re.compile(r'\\bibitem\{([^}]+)\}.*?(?=\\bibitem\{|\Z)', re.DOTALL)
bibitems = {}
for match in bibitem_pattern.finditer(bib_section):
    key = match.group(1)
    full_text = match.group(0)
    bibitems[key] = full_text

# 提取正文引用顺序
main_text = content[content.find('\\section{Introduction}'):content.find('\\bibliographystyle')]
cite_pattern = re.compile(r'\\cite\{([^}]+)\}')
all_cites = cite_pattern.findall(main_text)

# 记录每个引用第一次出现的位置
first_cite_positions = {}
for cite in all_cites:
    for key in cite.split(','):
        key = key.strip()
        if key not in first_cite_positions:
            pos = main_text.find(f'\\cite{{{cite}}}')
            first_cite_positions[key] = pos

# 按照引用顺序排序
ordered_keys = sorted(first_cite_positions.keys(), key=lambda k: first_cite_positions[k])

# 生成重排序的参考文献
print("\\begin{thebibliography}{99}\n")
for key in ordered_keys:
    if key in bibitems:
        print(bibitems[key])
    else:
        print(f"% WARNING: Missing bibitem for {key}")

# 列出未被引用的文献
all_bib_keys = set(bibitems.keys())
cited_keys = set(ordered_keys)
uncited = all_bib_keys - cited_keys

if uncited:
    print("\n% ============================================================================")
    print(f"% WARNING: The following {len(uncited)} bibitems are NOT cited in the main text:")
    print("% These should be removed unless intentionally kept")
    print("% ============================================================================\n")
    for key in sorted(uncited):
        print(f"% UNCITED: {key}")
        # print(bibitems[key])

print("\n\\end{thebibliography}")
