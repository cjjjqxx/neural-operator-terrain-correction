# -*- coding: utf-8 -*-
"""
nature_style_v2.py -- 全套配图的统一样式层

设计约束(按 Nature 系期刊制版要求):
  * 单栏 89 mm / 1.5 栏 120 mm / 双栏 183 mm
  * 正文字号 7 pt,任何渲染字形不低于 5 pt
  * SVG 文本保持可编辑(svg.fonttype='none'),PDF 用 TrueType(fonttype=42)
  * 每张图只用一套配色:一个中性族 + 一个信号族 + 一个强调色

配色语义(全套统一,不允许逐图自创):
  HERO   深蓝   本文方法/校正后/主结果
  REF    中灰   参考解/校正前/基线
  ACCENT 砖红   需要读者注意的对比或异常
  WARN   琥珀   阈值、分位数标注
B 场一律用 CMAP_B(蓝-中性-红,中心=1),热流一律用 CMAP_Q,
不确定度一律用 CMAP_U —— 三者色相互不重叠,避免跨图误读。
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as _np0
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- 尺寸
MM = 1 / 25.4
W1 = 89 * MM       # 单栏
W15 = 120 * MM     # 1.5 栏
W2 = 183 * MM      # 双栏

# ---------------------------------------------------------------- 颜色
COL = {
    'hero':        '#1A5FA8',   # 本文方法 / 校正后
    'hero_light':  '#7FB0DC',
    'ref':         '#8A8A8A',   # 参考 / 校正前
    'ref_light':   '#C8C8C8',
    'accent':      '#B03A2E',   # 强调对比
    'accent_light':'#E4A79F',
    'warn':        '#D98C1F',   # 阈值/分位数
    'ink':         '#222222',   # 正文墨色
    'grid':        '#DDDDDD',
    'land':        '#ECECEC',
                                #  它与灰色数据点太近,点会融进底图)
    'ocean':       '#FFFFFF',
                                # #EFEFEF 只差一级灰,整幅图糊成浅灰一片,
                                # 数据的颜色浮不出来。海洋留白最干净。
    'coast':       '#FFFFFF',
}

SERIES = {
    'method':   ('#1A5FA8', '#7FB0DC'),   # 算子 / 我们的方法      —— 图 6
    'evidence': ('#1B6E6A', '#8FC4C0'),   # 观测中的事实           —— 图 9
    'bias':     ('#6B3FA0', '#B79BD6'),   # 采样偏倚 / 本文的发现  —— 图 10
}

try:
    from cmcrameri import cm as _cmc

    CMAP_B = _cmc.vik
    CMAP_RES = _cmc.bam
    CMAP_Q = LinearSegmentedColormap.from_list(
        'Q_seq', _cmc.roma_r(_np0.linspace(0.03, 0.97, 256)), N=256)
    # 顺序量(附图 S5 的观测支撑距离)。先后试过 oslo_r(单色蓝,八档分不开)与
    # lipari_r(中段的暗红->灰紫发浑),最后用 ColorBrewer 的 YlGnBu:
    # 浅黄 -> 浅绿 -> 青 -> 蓝 -> 深蓝,每档色相分明、明度单调递减,
    # 八个档次一眼可数。ColorBrewer 本就是为专题地图设计的配色,
    # 顺序型同样满足感知有序与色觉障碍友好。方向也对 —— 离观测越远越深。
    import matplotlib.cm as _mcmU
    CMAP_U = LinearSegmentedColormap.from_list(
        'U_seq', _mcmU.YlGnBu(_np0.linspace(0.04, 0.96, 256)), N=256)
    CRAMERI = True
except ImportError:      # 没装 cmcrameri 时退回手工色标,图仍能出,只是不够严谨
    CMAP_B = LinearSegmentedColormap.from_list(
        'B_div', ['#1A4F7A', '#5C93C0', '#B9CFE0', '#EFEDEA',
                  '#F0C3A8', '#D97C4E', '#8C3018'], N=256)
    CMAP_RES = LinearSegmentedColormap.from_list(
        'RES_div', ['#40004B', '#9970AB', '#D9C6E0', '#F5F2F0',
                    '#C7E6C0', '#5AAE61', '#00441B'], N=256)
    CMAP_Q = LinearSegmentedColormap.from_list(
        'Q_seq', ['#FFF6E0', '#FBD98A', '#F0A03C', '#D3611F',
                  '#9E2C10', '#5C1206'], N=256)
    CMAP_U = LinearSegmentedColormap.from_list(
        'U_seq', ['#DCE8F2', '#B3CDE5', '#7FA9D0', '#4A7FB5',
                  '#28598C', '#123657'], N=256)
    CRAMERI = False

# 高程:地学常用的地形色带(低地绿 -> 丘陵黄褐 -> 高山棕 -> 雪线灰白)。
# 灰阶虽然中性,但读者对 DEM 的既有阅读习惯是这套色系,识别高低更快。
CMAP_DEM = LinearSegmentedColormap.from_list(
    'DEM', ['#2E6B4F', '#6FA86B', '#B9C87A', '#E3D08A',
            '#C9A170', '#9E7757', '#8A6E63', '#C9C3BE', '#F2F0EE'], N=256)

import matplotlib as _mpl2
import numpy as _np2
RELIEF_RAMP = LinearSegmentedColormap.from_list(
    'relief', _mpl2.colormaps['viridis'](_np2.linspace(0.10, 0.80, 256)), N=256)


# 密度图专用色阶。两个毛病要一起治:
#   'Blues' 最浅一档接近纯白,低密度区(离群点所在)几乎不可见;
#   且它不是感知均匀的 —— 亮度不随数值线性推进,密度差看着就'平'。
# 取 cividis 的主段:感知均匀、为色觉障碍优化,色相停在蓝-金之间,
# 不与全套的蓝色主色打架(换 viridis 会引入黄绿,与 CMAP_Q 的暖色系冲突)。
# 取 viridis 主段:感知均匀、色觉障碍友好,是顶刊密度图的通行选择。
# 起点略微内缩(0.05),避开最暗的深紫 —— 白底上最低一档要能与背景分开。
import matplotlib as _mpl
import numpy as _np
DENSITY_RAMP = LinearSegmentedColormap.from_list(
    'density', _mpl.colormaps['viridis'](_np.linspace(0.05, 1.0, 256)), N=256)


def relief_colors(n=5):
    return [RELIEF_RAMP(i / (n - 1)) for i in range(n)]


# ------------------------------------------------------- 三级字号(全套统一)
# 制版下限是"缩放后 5 pt"。matplotlib 的 mathtext 上/下标固定为基础字号的 0.7 倍,
# 无法用 rcParams 调整,所以 6 pt 的 $R^2$ 会渲染成 4.2 pt 的上标而低于下限。
# 解决办法是全套禁用小字号里的 mathtext,改用 Unicode 上标字符(见 SUP),
# 它们按正常字号渲染,不受 0.7 倍缩放影响。
FS_TICK = 7.2    # 刻度、图内注记、图例
                 # (对数轴 10ⁿ 的 mathtext 上标 = 0.7x = 5.04 pt,刚好达标)
FS_LABEL = 7.5   # 坐标轴标题、面板小标题(mathtext 上标 = 0.7x = 5.25 pt,达标)
FS_PANEL = 8     # 面板字母(加粗)

# 常用单位与符号的 Unicode 写法,避免 mathtext 上下标掉到 5 pt 以下
# Arial 缺 U+207B(上标减号)的字形,直接用 Unicode 会印成方块;
# 该处必须走 mathtext(渲染用数学字体,一定有字形),并保证基础字号 >=7.2 pt
U = {
    'mW': 'mW m$^{-2}$',
    'R2': 'R²',
    'dq': 'Δq',
    'absB1': '|B − 1|',
}


def set_style(base=FS_TICK):
    """全局 rcParams。每个绘图脚本开头调用一次。"""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'svg.fonttype': 'none',     # SVG 文本保持可编辑
        'pdf.fonttype': 42,         # PDF 用 TrueType
        'font.size': base,
        'axes.labelsize': FS_LABEL,
        'axes.titlesize': FS_LABEL,
        'xtick.labelsize': FS_TICK,
        'ytick.labelsize': FS_TICK,
        'legend.fontsize': FS_TICK,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.linewidth': 0.6,
        'axes.labelcolor': COL['ink'],
        'axes.edgecolor': '#555555',
        'text.color': COL['ink'],
        'xtick.color': '#555555',
        'ytick.color': '#555555',
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.4,
        'ytick.major.size': 2.4,
        # 图例一律带不透明白底:防止数据线/参考线从图例文字上穿过
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.framealpha': 0.92,
        'legend.edgecolor': 'none',
        'legend.borderpad': 0.3,
        'legend.handlelength': 1.2,
        'legend.handletextpad': 0.45,
        'legend.labelspacing': 0.28,
        'lines.linewidth': 1.1,
        'lines.markersize': 3,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'savefig.bbox': None,
        'savefig.pad_inches': 0.0,
    })


def panel(ax, letter, x=-0.16, y=1.04, size=FS_PANEL):
    """左上角面板字母,统一位置与字重。"""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=size,
            fontweight='bold', color=COL['ink'], ha='left', va='bottom')


def panel_fig(fig, letter, x, y, size=FS_PANEL):
    """给没有独立 axes 的复合面板(例如整幅地图)打标。坐标为 figure 分数。"""
    fig.text(x, y, letter, fontsize=size, fontweight='bold',
             color=COL['ink'], ha='left', va='bottom')


def guide(ax, axis='y', value=None):
    """参考线:统一灰度与线宽,保证缩印后仍然可见(评审意见 H8)。"""
    kw = dict(color='#A6A6A6', lw=0.65, ls=(0, (2.5, 2)), zorder=1)
    if axis == 'y':
        ax.axhline(value, **kw)
    else:
        ax.axvline(value, **kw)


# 全球图的统一裁剪范围与宽高比。南极无测点不参与插值,最北的保留测点在
# 83.9°N;裁掉两端空白纬带后,等经纬投影的宽高比是 360/142 = 2.535。
# 数据实测 lat -54.94 .. 83.32。上下各留 3 度余量,否则北极群岛的轮廓与
# 最北那批散点直接贴住上边框(图 2 上看得很明显)。跨度 144 度,比例正好 2.5。
MAP_EXTENT = (-180.0, 180.0, -57.0, 87.0)
RECT_AR = (MAP_EXTENT[1] - MAP_EXTENT[0]) / (MAP_EXTENT[3] - MAP_EXTENT[2])


def map_ax(ax, land=True, graticule=True, labels=False, rect=True):
    """统一的地图底图样式。需要 cartopy。

    """
    import cartopy.feature as cfeature
    if rect:
        ax.set_extent(MAP_EXTENT, crs=__import__('cartopy.crs',
                                                 fromlist=['x']).PlateCarree())
    else:
        ax.set_global()
    if land:
        ax.add_feature(cfeature.LAND, facecolor=COL['land'],
                       edgecolor='none', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=COL['ocean'],
                   edgecolor='none', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.25,
                   edgecolor='#AAAAAA', zorder=3)
    # 经纬网:AGU 的图件规范把"地图须标注经纬度"写成明文条款
    # (Text & Graphics Requirements)。Nature 无对应条款,但全球图配经纬网
    # 是通行做法,且有了经纬网就不再需要指北针和比例尺。
    # 线画得比海岸线还淡 —— 它是参照系,不是数据。
    XT = [-120, -60, 0, 60, 120]
    YT = [-35, 0, 35, 70]
    if graticule:
        import matplotlib.ticker as _mt
        # 线宽 0.30:Nature 的下限是 0.25 pt,再细印刷时会消失。
        # zorder=1 让它压在底图之上、数据之下 —— 参照系不该盖住数据,
        # 因此陆地上有数据的地方看不到网格线,这是有意为之。
        gl = ax.gridlines(draw_labels=False, linewidth=0.30, color='#A0A0A0',
                          alpha=0.75, linestyle=(0, (2, 2)), zorder=1)
        gl.xlocator = _mt.FixedLocator(XT)
        gl.ylocator = _mt.FixedLocator(YT)
    if rect and labels:
        # 矩形投影才谈得上"把刻度标在轴上":经线是等距竖线、纬线是等距横线,
        # 读者可以像读普通坐标轴一样读出经纬度。Robinson 下标签不归属矩形边,
        # 只能靠 gridlines(draw_labels=...) 贴在弧线旁,读起来费劲得多。
        import cartopy.crs as _ccrs
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        ax.set_xticks(XT, crs=_ccrs.PlateCarree())
        ax.set_yticks(YT, crs=_ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelsize=FS_TICK, length=2.0, width=0.5, pad=1.5,
                       color='#999999', labelcolor=COL['ink'])
    for _s in ('geo', 'left', 'right', 'top', 'bottom'):
        if _s in ax.spines:
            ax.spines[_s].set_linewidth(0.5)
            ax.spines[_s].set_edgecolor('#666666' if rect else '#999999')
    return ax


def cbar(fig, mappable, rect, label, ticks=None, extend='neither',
         orientation='horizontal'):
    """统一的色标。rect 为 figure 分数坐标,由调用方负责放在坐标区/地图之外。"""
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation, extend=extend)
    cb.set_label(label, fontsize=FS_LABEL, labelpad=2)
    cb.ax.tick_params(labelsize=FS_TICK, length=1.8, width=0.5, pad=1.2)
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor('#888888')
    if ticks is not None:
        cb.set_ticks(ticks)
    return cb


def despine_all(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def check_inside_canvas(fig, tol_pt=1.0):
    """导出前检查有没有元素被画布边缘裁掉,返回越界项列表。

    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    bad = []
    for ax in fig.axes:
        items = [(ax.get_xaxis().get_label(), 'xlabel'),
                 (ax.get_yaxis().get_label(), 'ylabel'),
                 (ax.title, 'title')]
        items += [(t, 'text') for t in ax.texts]
        if not hasattr(ax, 'projection'):
            x0, x1 = sorted(ax.get_xlim())
            y0, y1 = sorted(ax.get_ylim())
            items += [(t, 'xtick') for t in ax.get_xticklabels()
                      if t.get_visible() and x0 <= t.get_position()[0] <= x1]
            items += [(t, 'ytick') for t in ax.get_yticklabels()
                      if t.get_visible() and y0 <= t.get_position()[1] <= y1]
        for obj, kind in items:
            txt = (obj.get_text() or '').strip()
            if not txt:
                continue
            try:
                bb = obj.get_window_extent(renderer=r)
            except Exception:
                continue
            if (bb.x0 < fb.x0 - tol_pt or bb.y0 < fb.y0 - tol_pt
                    or bb.x1 > fb.x1 + tol_pt or bb.y1 > fb.y1 + tol_pt):
                bad.append((kind, txt[:38]))
    for t in fig.texts:
        txt = (t.get_text() or '').strip()
        if not txt:
            continue
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        if (bb.x0 < fb.x0 - tol_pt or bb.y0 < fb.y0 - tol_pt
                or bb.x1 > fb.x1 + tol_pt or bb.y1 > fb.y1 + tol_pt):
            bad.append(('figtext', txt[:38]))
    return bad


def save(fig, path_stem, png=True, pdf=True, svg=False):
    """导出。SVG 供后期编辑,PDF 供投稿,PNG 供 Word 排版预览。"""
    off = check_inside_canvas(fig)
    if off:
        print('  ⚠ %d 个元素超出画布,导出后会被裁掉:' % len(off))
        for kind, txt in off[:6]:
            print('      %-8s %s' % (kind, txt))
    outs = []
    if pdf:
        fig.savefig(f'{path_stem}.pdf'); outs.append('.pdf')
    if svg:
        fig.savefig(f'{path_stem}.svg'); outs.append('.svg')
    if png:
        fig.savefig(f'{path_stem}.png', dpi=600); outs.append('.png')
    plt.close(fig)
    print(f'  -> {path_stem}{"+".join(outs)}')
