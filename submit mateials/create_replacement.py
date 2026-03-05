# 读取原文件
with open('meta_rl_pid_control_manuscript_with_highlight.tex', 'r', encoding='utf-8') as f:
    original = f.read()

# 读取新的参考文献列表
with open('bibliography_clean.txt', 'r', encoding='utf-8') as f:
    new_bib = f.read()

# 找到旧的参考文献区域
bib_start = original.find('\\begin{thebibliography}{99}')
bib_end = original.find('\\end{thebibliography}') + len('\\end{thebibliography}')

if bib_start == -1 or bib_end == -1:
    print("ERROR: Could not find bibliography section")
    exit(1)

# 替换
new_content = original[:bib_start] + new_bib.strip() + '\n\n' + original[bib_end:]

# 写入新文件
with open('meta_rl_pid_control_manuscript_with_highlight_fixed.tex', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ Bibliography reordered and uncited references removed")
print(f"   Original file: {len(original)} characters")
print(f"   New file: {len(new_content)} characters")
print(f"   Difference: {len(original) - len(new_content)} characters removed")
