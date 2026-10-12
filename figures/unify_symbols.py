# -*- coding: utf-8 -*-
"""把全套图里的数学符号统一成斜体变量(ISO 80000-2)。

改之前全套是混着来的:
  * B 在轴标签里是正体("FNO-3D predicted B"),在 Fig. 5 的 B_s / B_D 里是斜体
  * q 在 Δq 里是斜体,在 "Corrected flux, q / B" 里是正体
  * |B - 1| 一处用 ASCII 连字符,一处用 U+2212 减号
  * n_x 写成 "n$_x$" —— n 正体、下标斜体,正好反了
  * R² 用 Unicode 上标,R 是正体

规则(ISO 80000-2,Nature 与 AGU 都遵循):
  单字母量符号斜体(B, q, D, f, n, R);描述性下标正体(measured, cor, obs);
  Δ 是算子,正体;单位正体。改用 mathtext 之后,减号统一由 mathtext 渲染成
  U+2212,ASCII/Unicode 混用的问题一并消失。

下标字号:mathtext 的下标按 0.7 倍排,母字号必须 ≥7.2 pt,否则下标跌破
figures/qa_check.py 的 5 pt 下限。全套标签都是 7.2 pt 或以上,安全。
"""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
V2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v2')

# 精确字符串替换。不做正则通配 —— "B" 单字母通配会打到 Borehole、By 这些词上。
RULES = [
    # ── B 作为量符号 ────────────────────────────────────────────────
    ("'FNO-3D predicted B'", "'FNO-3D predicted $B$'"),
    ("'Finite-element reference B'", "'Finite-element reference $B$'"),
    ("'Mean absolute error in B'", "'Mean absolute error in $B$'"),
    ("'Per-tile mean absolute error in B'", "'Per-tile mean absolute error in $B$'"),
    ("'Centre-pixel B (finite element)'", "'Centre-pixel $B$ (finite element)'"),
    ("'B (ridge < 1 < valley), scaled per row'",
     "'$B$ (ridge < 1 < valley), scaled per row'"),
    ("'B along centre row'", "'$B$ along centre row'"),
    ("'Finite-element B'", "'Finite-element $B$'"),
    ("'FNO-3D B'", "'FNO-3D $B$'"),
    ("'B = 1'", "'$B$ = 1'"),
    ("'Median |B - 1|'", "'Median $|B - 1|$'"),
    ("'Change in median |B - 1|\\nfrom 20 km (%)'",
     "'Change in median $|B - 1|$\\nfrom 20 km (%)'"),
    ("'Change in median |B - 1|\\nfrom adopted n$_x$ = 40 (%)'",
     "'Change in median $|B - 1|$\\nfrom adopted $n_x$ = 40 (%)'"),
    ("'Terrain sensitivity, |B − 1|'", "'Terrain sensitivity, $|B - 1|$'"),
    ("'|B - 1| band'", "'$|B - 1|$ band'"),
    ("'|B − 1| band'", "'$|B - 1|$ band'"),
    # 轴内那个单独的 B(Fig. 5 e 列)
    ("ax.text(0.02, 0.96, 'B', transform=ax.transAxes",
     "ax.text(0.02, 0.96, '$B$', transform=ax.transAxes"),
    ("ax.set_xlabel('B', labelpad=2)", "ax.set_xlabel('$B$', labelpad=2)"),
    # ── q 作为量符号 ────────────────────────────────────────────────
    ("'Corrected flux, q / B'", "'Corrected flux, $q\\\\,/\\\\,B$'"),
    # ── f(D)、D、n ──────────────────────────────────────────────────
    ("'Retained fraction f(D)'", "'Retained fraction $f(D)$'"),
    ("'Retained surface departure, f(D)'", "'Retained surface departure, $f(D)$'"),
    ("'Borehole interval depth, D (m)'", "'Borehole interval depth, $D$ (m)'"),
    ("'Elements per tile edge, n$_x$   (element 1,425 to 445 m)'",
     "'Elements per tile edge, $n_x$   (element 1,425 to 445 m)'"),
    ("'Vertical layers, n$_z$   (top layer %.0f to %.0f m)'",
     "'Vertical layers, $n_z$   (top layer %.0f to %.0f m)'"),
    ("'labels: change relative to finest (n$_z$ = %d)'",
     "'labels: change relative to finest ($n_z$ = %d)'"),
    # ── R² ──────────────────────────────────────────────────────────
    ("'Per-tile R²'", "'Per-tile $R^2$'"),
    ("'Pixel-wise R²'", "'Pixel-wise $R^2$'"),
    ("'Validation R²'", "'Validation $R^2$'"),
    ("R² = {_r2i:.3f}", "$R^2$ = {_r2i:.3f}"),
    ("R² = {r2_spatial:.3f}", "$R^2$ = {r2_spatial:.3f}"),
    ("'R² = {np.nanmax(r2):.4f}'", "'$R^2$ = {np.nanmax(r2):.4f}'"),
    # ── 图例里的 B ──────────────────────────────────────────────────
    ("f'Ridge, B < 1  ({100*up.mean():.1f}% of sites)'",
     "f'Ridge, $B$ < 1  ({100*up.mean():.1f}% of sites)'"),
    ("f'Valley, B > 1  ({100*dn.mean():.1f}% of sites)'",
     "f'Valley, $B$ > 1  ({100*dn.mean():.1f}% of sites)'"),
]

total = 0
for fn in sorted(os.listdir(V2)):
    if not fn.endswith('.py'):
        continue
    p = os.path.join(V2, fn)
    t = open(p, encoding='utf-8').read()
    orig = t
    hits = []
    for a, b in RULES:
        if a in t:
            t = t.replace(a, b)
            hits.append(a[:44])
    if t != orig:
        open(p, 'w', encoding='utf-8').write(t)
        print('{:<38} {} 处'.format(fn, len(hits)))
        total += len(hits)

print()
print('共替换 {} 处'.format(total))

print()
print('复查(以下若有输出,说明还有正体符号没改):')
pat = re.compile(r"'[^']*(?<![$\w])(?:reference B|predicted B|error in B|"
                 r"\|B [-−] 1\||R²|f\(D\)|n\$_[xz]\$)[^']*'")
left = 0
for fn in sorted(os.listdir(V2)):
    if not fn.endswith('.py'):
        continue
    for m in pat.finditer(open(os.path.join(V2, fn), encoding='utf-8').read()):
        print('  {}: {}'.format(fn, m.group(0)[:70]))
        left += 1
print('  (无)' if left == 0 else '  共 {} 处待处理'.format(left))
