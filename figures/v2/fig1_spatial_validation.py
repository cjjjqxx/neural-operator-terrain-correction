# -*- coding: utf-8 -*-
"""
Fig. 1 | From three-dimensional finite-element simulations to a global correction.


d、e 的划分与训练脚本完全一致(k 均值 12 类,留出 0、4、8 三类)。

"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize, BoundaryNorm, ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import cartopy.crs as ccrs
from sklearn.cluster import KMeans
from cmcrameri import cm as _cmc
from nature_style_v2 import (set_style, COL, W2, CMAP_B, panel_fig, map_ax, save, cbar, FS_TICK, FS_LABEL)

set_style(base=7)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figs_v2')
os.makedirs(OUT, exist_ok=True)
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, ROOT)
from fem3d import Fem3D, Q_REF                      # noqa: E402
from tile_reader import cell_m_for, TILE_KM        # noqa: E402

# ════════════════════════════ 上半:三维有限元示例 ════════════════════════════
# 生产配置:水平 40 × 40、垂向 10 层,计算域深 max(3 km, 1.5 × 起伏),K = 2.5 W m⁻¹ K⁻¹。
# B 与 K、q0 的绝对值无关;温度场按 q0 = Q_REF 给出,只作示意。
ex = np.load(os.path.join(ROOT, 'andes_example_tile.npz'))
dem = ex['dem'].astype(float)
NX = NY = 40
NZ = 10
fem = Fem3D(nx=NX, ny=NY, nz=NZ)
fem.run(dem, cell_m_for())
P = fem.pts.reshape(3, NZ + 1, NY + 1, NX + 1) / 1000.0          # km
T = fem.T.reshape(NZ + 1, NY + 1, NX + 1)
Bf = fem.B_grid
relief = dem.max() - dem.min()
print('[fig1] 示例瓦片 %.2f°, %.2f°  起伏 %.0f m  域深 %.0f m  中心 B:有限元 %.3f,算子 %.3f'
      % (float(ex['lon']), float(ex['lat']), relief, 1000 * (P[2, NZ].min() - P[2, 0].min()) + 0,
         Bf[NY // 2, NX // 2], float(ex['B_center'])))

VE = 2.5                                   # 垂向放大
IY0 = NY // 2                              # 剖切于测点所在行(瓦片中线)
NS = 64                                    # 剖面上每列的垂向插值点数(只为画图,不改变解)
Lx = P[0, 0, 0, -1]
Ly = P[1, 0, -1, 0]
y0 = P[1, 0, IY0, 0]
zbot = P[2, 0].min()

from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import proj3d
from matplotlib import patheffects as _pe
from nature_style_v2 import CMAP_DEM
UP = 2                                              # 显示网格 = 像元网格加密一倍(双线性,不添信息,只让晕渲不显块面)
xh = np.linspace(0, Lx, UP * (dem.shape[1] - 1) + 1)
yh = np.linspace(0, Ly, UP * (dem.shape[0] - 1) + 1)
_dem_km = RegularGridInterpolator((np.linspace(0, Ly, dem.shape[0]), np.linspace(0, Lx, dem.shape[1])),
                                  dem / 1000.0)
ZF = _dem_km(tuple(np.meshgrid(yh, xh, indexing='ij')))   # 整幅瓦片(晕渲要用全幅梯度)
JY0 = int(np.argmin(np.abs(yh - y0)))                    # y0 恰是加密网格的第 127 行
assert abs(yh[JY0] - y0) < 1e-9
yh2 = yh[JY0:]
XH, YH = np.meshgrid(xh, yh2)
ZH = ZF[JY0:]
_SIG = np.linspace(0, 1, NZ + 1)


def section(u_nod, zs_nod, t_nod, u_hr, ztop):
    """把有限元温度场画到高分辨率剖线之下(只为画图,不改变解)。

    每列 NS 个点从域底均匀排到高分辨率地表 ztop;温度在 (σ, u) 上对节点双线性插值,
    σ 按有限元折面 zs_lin(u) 计,与网格的 σ 分层一致。
    高分辨率地表高出有限元折面的窄带(窄山脊)σ > 1,截到 σ = 1,即按 T = 0 填色:
    地表本就是 T = 0 的 Dirichlet 边界,近地表温度≈0,该带在本剖面最厚约 0.27 km,
    带内真实温度只有几度,在色标上看不出差别;高分辨率地表低于折面处(窄谷)直接截去。
    """
    s = np.linspace(0, 1, NS)[:, None]
    zz = zbot + s * (ztop[None, :] - zbot)
    uu = np.broadcast_to(u_hr[None, :], zz.shape).copy()
    zs_lin = np.interp(u_hr, u_nod, zs_nod)[None, :]
    sig = np.clip((zz - zbot) / (zs_lin - zbot), 0.0, 1.0)
    tt = RegularGridInterpolator((_SIG, u_nod), t_nod)((sig, uu))
    return uu, zz, tt


# 面 1:剖切面 y = y0(朝向读者);面 2:右侧面 x = Lx
xs_c, zs_c, ts_c = section(P[0, NZ, IY0, :], P[2, NZ, IY0, :], T[:, IY0, :], xh, ZH[0])
xr, zr, tr = section(P[1, NZ, IY0:, -1], P[2, NZ, IY0:, -1], T[:, IY0:, -1], yh2, ZH[:, -1])
_d1 = ZH[0] - np.interp(xh, P[0, NZ, IY0, :], P[2, NZ, IY0, :])
_d2 = ZH[:, -1] - np.interp(yh2, P[1, NZ, IY0:, -1], P[2, NZ, IY0:, -1])
print('[fig1a] 高分辨率剖线 − 有限元折面:剖切面 %.3f … %.3f km,右侧面 %.3f … %.3f km(σ > 1 处按 T = 0 填色)'
      % (_d1.min(), _d1.max(), _d2.min(), _d2.max()))

# 地表冷(浅)→ 深部热(深):取 lajolla 的反向一段,不用到最暗的近黑端
from matplotlib.colors import LinearSegmentedColormap
T_CM = LinearSegmentedColormap.from_list('temp', _cmc.lajolla(np.linspace(1.0, 0.20, 256)))
T_NORM = Normalize(0, np.ceil(T.max() / 20) * 20)
T_LEV = np.arange(20, T.max(), 20)

H_MM = 151.0
A3_X0, A3_TOP, A3_W, A3_H = 0.0, 0.0, 112.0, 75.0     # 三维坐标区(mm)
ELEV, AZIM, ZOOM = 26, -62, 1.04
LIGHT_AZ, LIGHT_ALT, SHADE_FRAC = 253, 40, 0.8        # 光从屏幕左上方来(见 LightSource 的方位约定)
ISO_COL = '#3A2A1A'
BLK_X0, BLK_TOP = 5.0, 12.0                          # 块体左缘、地表轮廓最高点(mm)
SITE_GAP = 2.0                                        # "Site"注记底边高出其上方地表轮廓(mm)
T0_TGT = (47.0, 53.0)                                 # "T = 0"引线所指的地表点(km)
T0_X = 52.0                                           # "T = 0"注记左缘(mm);底边与"Site"同一行
N_ARROW, ARROW_MM = 4, 4.0                            # 底部热流箭头个数、长度(mm)
CB_X, CB_H = 96.0, 28.0                               # 温度色标左缘、高度(mm)
fig = plt.figure(figsize=(W2, H_MM / 25.4))


def fx(mm):
    return mm / 183.0


def fy(top_mm, h_mm):
    return 1 - (top_mm + h_mm) / H_MM


ZTOP = ZH.max()
ax3 = fig.add_axes([fx(A3_X0), fy(A3_TOP, A3_H), fx(A3_W), A3_H / H_MM], projection='3d',
                   computed_zorder=False)
ax3.set_axis_off()
ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])   # 隐藏轴的刻度标签会被越界检查误报
ax3.set_proj_type('ortho')
ax3.view_init(elev=ELEV, azim=AZIM)
ax3.set_xlim(0, Lx)
ax3.set_ylim(y0, Ly)
ax3.set_zlim(zbot * VE, ZTOP * VE)
ax3.set_box_aspect((Lx, Ly - y0, (ZTOP - zbot) * VE), zoom=ZOOM)


def p2f(x, y, z):
    """三维点(z 已乘 VE)→ figure 分数坐标;视角、范围、box_aspect 都设好之后调用。"""
    u, v, _ = proj3d.proj_transform(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z), ax3.get_proj())
    return fig.transFigure.inverted().transform(ax3.transData.transform(np.column_stack([u, v])))


def isolines(xx, zz, tt):
    """在 x–z 平面上求等温线,返回线段列表。"""
    _f, _a = plt.subplots()
    cs = _a.contour(xx, zz, tt, levels=T_LEV)
    segs = cs.allsegs
    plt.close(_f)
    return segs


# 绘制次序(computed_zorder=False,按 zorder 画):地表 → 两个剖面 → 等温线 → 棱线 → 测点。
# 地表全部位于 y ≥ y0、x ≤ Lx,两个剖面是块体朝向读者的外表面:屏幕上凡与剖面重叠的地表
# 都在剖面之后,故剖面后画即得正确遮挡(山脊背后的谷地不会盖到剖面上)。
# ── 地表:CMAP_DEM 按本瓦片全幅高程拉伸(与图 3a 同一色带、同一拉伸方式)+ 柔和晕渲 ──
# 晕渲以中位亮度为界:迎光坡向白提亮、背光坡按比例压暗,平地保持色带原色(不像乘法晕渲那样整体发灰)。
DEM_NORM = Normalize(dem.min() / 1000.0, dem.max() / 1000.0)
_hs = LightSource(azdeg=LIGHT_AZ, altdeg=LIGHT_ALT).hillshade(ZF, vert_exag=VE, dx=xh[1] - xh[0],
                                                              dy=yh[1] - yh[0])
_k = SHADE_FRAC * (_hs - np.median(_hs))[..., None]
_base = CMAP_DEM(DEM_NORM(ZF))[..., :3]
rgb_t = np.clip(np.where(_k > 0, _base + _k * (1 - _base), _base * (1 + _k)), 0, 1)[JY0:]
# 远侧两条地表边(x = 0、y = Ly)先于地表画,只露出轮廓外侧的半条线宽;被前方山体挡住处自然隐去
_rim = dict(color='#7A7A7A', lw=0.7, zorder=0.5)
ax3.plot(np.zeros_like(yh2), yh2, ZH[:, 0] * VE, **_rim)
ax3.plot(xh, np.full_like(xh, Ly), ZH[-1] * VE, **_rim)
ax3.plot_surface(XH, YH, ZH * VE, facecolors=rgb_t, shade=False, linewidth=0, antialiased=False,
                 rstride=1, cstride=1, zorder=1, rasterized=True)
# ── 剖切面与右侧面:温度场 + 等温线 ──
ax3.plot_surface(xs_c, np.full_like(xs_c, y0), zs_c * VE, facecolors=T_CM(T_NORM(ts_c)), shade=False,
                 linewidth=0, antialiased=False, rstride=1, cstride=1, zorder=2, rasterized=True)
ax3.plot_surface(np.full_like(xr, Lx), xr, zr * VE, facecolors=T_CM(T_NORM(tr)), shade=False,
                 linewidth=0, antialiased=False, rstride=1, cstride=1, zorder=2, rasterized=True)
_iso = dict(color=ISO_COL, lw=0.4, alpha=0.6, zorder=3)
for lev in isolines(xs_c, zs_c, ts_c):
    for seg in lev:
        ax3.plot(seg[:, 0], np.full(len(seg), y0), seg[:, 1] * VE, **_iso)
for lev in isolines(xr, zr, tr):
    for seg in lev:
        ax3.plot(np.full(len(seg), Lx), seg[:, 0], seg[:, 1] * VE, **_iso)
# ── 块体棱线 ──
_e = dict(color='#4A4A4A', lw=0.5, zorder=4, solid_capstyle='round')
for x_, y_, z_ in ((0, y0, ZH[0, 0]), (Lx, y0, ZH[0, -1]), (Lx, Ly, ZH[-1, -1])):
    ax3.plot([x_, x_], [y_, y_], [zbot * VE, z_ * VE], **_e)
ax3.plot([0, Lx], [y0, y0], [zbot * VE] * 2, **_e)
ax3.plot([Lx, Lx], [y0, Ly], [zbot * VE] * 2, **_e)
ax3.plot(np.full_like(yh2, Lx), yh2, ZH[:, -1] * VE, color='#4A4A4A', lw=0.5, zorder=4)
ax3.plot(xh, np.full_like(xh, y0), ZH[0] * VE, color=COL['ink'], lw=0.75, zorder=5,
         solid_joinstyle='round')       # 剖切线 = 图 b 的剖面

# ── 块体定位:先画一次求投影,再平移三维坐标区,使块体左缘、地表轮廓顶落在给定位置(mm)──
# 平移不改变投影比例,块体大小只由坐标区尺寸与 ZOOM 决定。
_PTY = 1 / 72 / (H_MM / 25.4)              # 1 pt 对应的 figure 高度分数
_MMY, _MMX = 1 / H_MM, 1 / 183.0


def block_outline():
    """块体在画面上的外包:地表全部网格点 + 三个可见底角。"""
    q = np.vstack([p2f(XH.ravel(), YH.ravel(), ZH.ravel() * VE),
                   p2f(np.array([0, Lx, Lx]), np.array([y0, y0, Ly]), np.full(3, zbot * VE))])
    return q


fig.canvas.draw()
_q = block_outline()
_pos = ax3.get_position()
ax3.set_position([_pos.x0 + fx(BLK_X0) - _q[:, 0].min(), _pos.y0 + (1 - BLK_TOP * _MMY) - _q[:, 1].max(),
                  _pos.width, _pos.height])
fig.canvas.draw()
_q = block_outline()
BX0, BX1, BY0, BY1 = _q[:, 0].min(), _q[:, 0].max(), _q[:, 1].min(), _q[:, 1].max()
print('[fig1a] 块体外包 x %.1f–%.1f mm,y(自顶)%.1f–%.1f mm'
      % (BX0 * 183, BX1 * 183, (1 - BY1) * H_MM, (1 - BY0) * H_MM))

# ── 测点(瓦片中心):地表上的倒三角(尖端落在测点),细引线拉到块体上方空白处标注 ──
xc, zc = P[0, NZ, IY0, NX // 2], P[2, NZ, IY0, NX // 2]
S = p2f(xc, y0, zc * VE)[0]
_near = np.abs(_q[:, 0] - S[0]) < 1.0 * _MMX
SIL = _q[_near, 1].max()                     # 测点所在屏幕列上地表投影的最高点
LAB_Y = SIL + SITE_GAP * _MMY                # "Site"注记底边:紧贴地表轮廓之上
_halo = [_pe.withStroke(linewidth=1.6, foreground='white')]
_MS = 5.0
fig.add_artist(plt.Line2D([S[0], S[0]], [S[1] + _MS * _PTY, LAB_Y - 0.6 * _MMY], color=COL['ink'], lw=0.5,
                          transform=fig.transFigure, zorder=6, path_effects=_halo, solid_capstyle='butt'))
fig.add_artist(plt.Line2D([S[0]], [S[1] + 0.5 * _MS * _PTY], marker='v', ms=_MS, mfc=COL['ink'], mec='white',
                          mew=0.6, ls='none', transform=fig.transFigure, zorder=7))
fig.text(S[0], LAB_Y, 'Site', ha='center', va='bottom', fontsize=FS_TICK, color=COL['ink'])

# ── 边界条件注记 ──
# 地表 T = 0:注记在块体右上方的空白处,引线指向块体右后部的地表
_tgt = p2f(T0_TGT[0], T0_TGT[1], _dem_km((T0_TGT[1], T0_TGT[0])) * VE)[0]
_t0 = (fx(T0_X), LAB_Y)
fig.text(_t0[0], _t0[1], '$T$ = 0 at the land surface', ha='left', va='bottom', fontsize=FS_TICK,
         color=COL['ink'])
fig.add_artist(plt.Line2D([_t0[0] + 1.5 * _MMX, _tgt[0]], [_t0[1] - 0.6 * _MMY, _tgt[1]], color=COL['ink'],
                          lw=0.5, transform=fig.transFigure, zorder=6, path_effects=_halo))
fig.add_artist(plt.Line2D([_tgt[0]], [_tgt[1]], marker='o', ms=1.8, mfc=COL['ink'], mec='white', mew=0.4,
                          transform=fig.transFigure, zorder=7))
# 侧面绝热:紧贴右侧面
_rf_mid = p2f(Lx, 0.5 * (y0 + Ly), 0.5 * (zbot + 0.5 * (ZH[0, -1] + ZH[-1, -1])) * VE)[0]
fig.text(BX1 + 1.5 * _MMX, _rf_mid[1], 'Insulated' + chr(10) + 'sides', ha='left', va='center',
         fontsize=FS_TICK, color='#555555')
# 底部均匀热流:沿前缘底边等距的箭头(尖端离底边 0.8 mm)
for k in range(N_ARROW):
    _b = p2f(Lx * (k + 0.5) / N_ARROW, y0, zbot * VE)[0]
    fig.add_artist(FancyArrowPatch((_b[0], _b[1] - (0.8 + ARROW_MM) * _MMY), (_b[0], _b[1] - 0.8 * _MMY),
                                   transform=fig.transFigure, arrowstyle='-|>', mutation_scale=5.5,
                                   lw=0.8, color=COL['accent'], shrinkA=0, shrinkB=0, zorder=6))
_b0 = p2f(Lx * 0.5 / N_ARROW, y0, zbot * VE)[0]
fig.text(BX0, _b0[1] - (0.8 + ARROW_MM + 1.0) * _MMY, 'Uniform basal' + chr(10) + 'heat flow $q_0$',
         ha='left', va='top', fontsize=FS_TICK, color=COL['accent'], linespacing=1.15)
# 瓦片尺寸与垂向放大:次要信息,灰色小一号,作为 a 的副标题紧跟面板字母
fig.text(fx(6.0), 1 - 3.5 * _MMY, '%.0f km × %.0f km tile, vertical exaggeration ×%.1f' % (TILE_KM, TILE_KM, VE),
         ha='left', va='bottom', fontsize=FS_TICK - 0.6, color='#555555')

# 温度色标:竖排,中心与右侧面中部同高,放在"Insulated sides"右侧
cb_t = cbar(fig, plt.cm.ScalarMappable(norm=T_NORM, cmap=T_CM),
            [fx(CB_X), _rf_mid[1] - 0.5 * CB_H * _MMY, fx(2.0), CB_H * _MMY], 'Temperature (°C)',
            ticks=[0, 100, 200], orientation='vertical')
cb_t.ax.invert_yaxis()                      # 0 °C 在上:与剖面"上冷下热"同向

# ── b:沿剖切线的地表高程与 B(有限元) ──
xk = P[0, NZ, IY0, :]
zk = P[2, NZ, IY0, :]
bk = Bf[IY0, :]
TAPER = 0.12 * TILE_KM
axh = fig.add_axes([fx(125.0), fy(5.0, 17.0), fx(54.0), 17.0 / H_MM])
axB = fig.add_axes([fx(125.0), fy(24.5, 30.0), fx(54.0), 30.0 / H_MM], sharex=axh)
for a_ in (axh, axB):
    for x_lo, x_hi in ((0, TAPER), (Lx - TAPER, Lx)):
        a_.axvspan(x_lo, x_hi, color='#EFEFEF', lw=0, zorder=0)
    a_.axvline(xc, color=COL['ink'], lw=0.5, ls=(0, (2, 1.5)), zorder=1)
# 高程取 a 中同一条高分辨率剖线(在有限元节点处与有限元地表逐点一致),使 b 与 a 的剖面前缘形状相同;
# B 仍为有限元节点值
axh.fill_between(xh, zbot, ZH[0], color='#B9B9B9', lw=0, zorder=2)
axh.plot(xh, ZH[0], color=COL['ink'], lw=0.7, zorder=3)
axh.set_ylim(np.floor(ZH[0].min()) - 0.3, np.ceil(ZH[0].max()) + 0.2)
axh.set_ylabel('Elevation' + chr(10) + '(km)', labelpad=2)
axh.tick_params(labelbottom=False)
axh.text(TAPER / 2, axh.get_ylim()[1] - 0.15, 'taper', ha='center', va='top', fontsize=FS_TICK - 1.0, color='#777777')
axh.plot([xc], [zc + 0.35], marker='v', ms=4, color=COL['ink'], zorder=4)
axB.axhline(1.0, color='#8A8A8A', lw=0.5, zorder=1)
axB.fill_between(xk, 1.0, bk, where=bk >= 1, color=CMAP_B(0.80), lw=0, interpolate=True, zorder=2)
axB.fill_between(xk, 1.0, bk, where=bk < 1, color=CMAP_B(0.20), lw=0, interpolate=True, zorder=2)
axB.plot(xk, bk, color=COL['ink'], lw=0.6, zorder=3)
axB.plot([xc], [bk[NX // 2]], 'o', ms=3.5, mfc='white', mec=COL['ink'], mew=0.7, zorder=4)
axB.annotate('Site $B_s$ = %.2f' % bk[NX // 2], xy=(xc, bk[NX // 2]), xytext=(12.0, 1.24),
             fontsize=FS_TICK, color=COL['ink'], ha='left', va='center', zorder=5,
             arrowprops=dict(arrowstyle='-', color=COL['ink'], lw=0.5, shrinkA=1, shrinkB=2.5))
axB.set_ylabel('Surface $B = q_s/q_0$', labelpad=2)
axB.set_xlabel('Distance along the section (km)', labelpad=2)
axB.set_xlim(0, Lx)
axB.set_xticks([0, 14.25, 28.5, 42.75, 57]); axB.set_xticklabels(['0', '', '28.5', '', '57'])
axB.text(0.98, 0.96, 'valley: $B$ > 1', transform=axB.transAxes, ha='right', va='top', fontsize=FS_TICK - 0.5,
         color=CMAP_B(0.85))
axB.text(0.98, 0.04, 'ridge: $B$ < 1', transform=axB.transAxes, ha='right', va='bottom', fontsize=FS_TICK - 0.5,
         color=CMAP_B(0.15))
print('[fig1b] 剖面 B 范围 %.2f–%.2f,测点 B_s(有限元)%.3f' % (bk.min(), bk.max(), bk[NX // 2]))

# ── c:流程带 ──
axc = fig.add_axes([fx(4.0), fy(68.5, 12.0), fx(176.0), 12.0 / H_MM])
axc.set_axis_off(); axc.set_xlim(0, 176); axc.set_ylim(0, 12)
steps = [('11,627 real-terrain tiles', '57 km, outside Antarctica'),
         ('3D finite-element solutions', 'surface $B$ fields (a, b)'),
         ('Fourier neural operator', 'trained once (Figs 2, 3)'),
         ('36,392 IHFC land sites', '$B_s$ → $B_D$ → $q_{cor}$ ± $\\sigma_q$ (Figs 4–6)')]
bw, gap = 37.5, 7.0
for k, (t1, t2) in enumerate(steps):
    x0 = 1 + k * (bw + gap)
    axc.add_patch(FancyBboxPatch((x0, 1.0), bw, 10.0, boxstyle='round,pad=0,rounding_size=1.2',
                                 fc='#F4F4F4', ec='#9A9A9A', lw=0.5))
    axc.text(x0 + bw / 2, 7.4, t1, ha='center', va='center', fontsize=FS_TICK, color=COL['ink'])
    axc.text(x0 + bw / 2, 3.8, t2, ha='center', va='center', fontsize=FS_TICK, color='#555555')   # 下标须 ≥ 5 pt
    if k < len(steps) - 1:
        axc.annotate('', xy=(x0 + bw + gap - 0.8, 6.0), xytext=(x0 + bw + 0.8, 6.0),
                     arrowprops=dict(arrowstyle='-|>', color='#6A6A6A', lw=0.8, mutation_scale=7))

# ════════════════════════════ 下半:空间划分(原图 1) ════════════════════════════
d = np.load(os.path.join(ROOT, 'global_fem_dataset.npz'))
lons, lats = d['lon'], d['lat']
km = KMeans(n_clusters=12, random_state=42, n_init='auto')
labels = km.fit_predict(np.column_stack([lons, lats]))
is_test = np.isin(labels, [0, 4, 8])

TOP2 = 84.0                                     # 下半部分顶端(mm)
MAP_W = 0.680
map_h_mm = MAP_W * 183.0 / 2.535
ax = fig.add_axes([0.050, fy(TOP2 + 6.5, map_h_mm), MAP_W, map_h_mm / H_MM],
                  projection=ccrs.PlateCarree(central_longitude=0))
map_ax(ax, labels=True, graticule=False)
ax.scatter(lons[~is_test], lats[~is_test], s=0.9, c='#8C8C8C', transform=ccrs.PlateCarree(), zorder=2,
           edgecolors='none', rasterized=True, label=f'Training tiles (n = {(~is_test).sum():,})')
ax.scatter(lons[is_test], lats[is_test], s=0.9, c=COL['accent'], transform=ccrs.PlateCarree(), zorder=3,
           edgecolors='none', rasterized=True, label=f'Spatially held-out tiles (n = {is_test.sum():,})')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, fontsize=FS_TICK, markerscale=6,
          handletextpad=0.4, columnspacing=2.2, frameon=False, borderaxespad=0.1)

rel = d['relief']
axe = fig.add_axes([0.805, fy(TOP2 + 6.5, map_h_mm), 0.180, map_h_mm / H_MM])
_med = {}
for m, c, lab in ((~is_test, '#8C8C8C', 'Training'), (is_test, COL['accent'], 'Held-out')):
    v = np.sort(rel[m])
    axe.plot(v, np.arange(1, v.size + 1) / v.size, color=c, lw=1.4, solid_joinstyle='round', zorder=3, label=lab)
    _med[lab] = np.median(rel[m])
    axe.plot([_med[lab]] * 2, [0, 0.5], color=c, lw=0.7, ls=(0, (2, 1.5)), zorder=2)
axe.axhline(0.5, color=COL['ref'], lw=0.5, ls=':', zorder=1)
axe.set_xscale('log')
axe.set_xlim(30, 8000)
axe.set_ylim(0, 1)
axe.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axe.set_xticks([100, 1000])
axe.set_xticklabels(['100', '1,000'])
axe.set_xlabel('Tile relief (m)', labelpad=2)
axe.set_ylabel('Cumulative fraction of tiles', labelpad=2)
axe.legend(loc='upper left', frameon=False, fontsize=FS_TICK, handlelength=1.4)
for _k, (_lab, _c) in enumerate((('Held-out', COL['accent']), ('Training', '#6E6E6E'))):
    axe.text(0.985, 0.035 + 0.072 * _k, '%s %.0f m' % (_lab, _med[_lab]), transform=axe.transAxes,
             ha='right', va='bottom', fontsize=FS_TICK, color=_c)
axe.text(0.985, 0.035 + 0.072 * 2, 'Medians', transform=axe.transAxes,
         ha='right', va='bottom', fontsize=FS_TICK, color=COL['ink'])

panel_fig(fig, 'a', 0.012, 1 - 3.5 / H_MM)
panel_fig(fig, 'b', fx(113.0), 1 - 3.5 / H_MM)
panel_fig(fig, 'c', 0.012, 1 - 69.5 / H_MM)
panel_fig(fig, 'd', 0.012, 1 - (TOP2 + 3.5) / H_MM)
panel_fig(fig, 'e', 0.745, 1 - (TOP2 + 3.5) / H_MM)

save(fig, os.path.join(OUT, 'Fig1'))
import json as _json
_json.dump({'relief_m': float(relief), 'B_s_fem': float(Bf[NY // 2, NX // 2]), 'B_s_operator': float(ex['B_center']),
            'vertical_exaggeration': VE, 'isotherm_step_C': float(T_LEV[1] - T_LEV[0]),
            'q0_mW_m2': 1000.0 * Q_REF, 'K_W_mK': 2.5, 'section_B_min': float(bk.min()), 'section_B_max': float(bk.max())},
           open(os.path.join(OUT, 'fig1_example.json'), 'w', encoding='utf-8'), indent=2)
print('[fig1] 训练 %d / 留出 %d;起伏中位 %.0f / %.0f m' % ((~is_test).sum(), is_test.sum(),
                                                       _med['Training'], _med['Held-out']))
