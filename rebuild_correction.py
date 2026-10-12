#!/usr/bin/env python3
"""
"""
import sys, os, time, json; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
import torch, rasterio
import os

# ── IHFC 数据库路径 ──────────────────────────────────────────────────
# 不写死本地路径:优先取环境变量 IHFC_GHFDB,否则找本脚本同级的 data/。
# 数据本身从 GFZ Data Services 公开获取(doi:10.5880/fidgeo.2024.014)。
def _ihfc_path():
    p = os.environ.get('IHFC_GHFDB')
    if p and os.path.exists(p):
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, '..', 'IHFC_2024_full', 'IHFC_2024_GHFDB_v.2026.03.txt'),
                 os.path.join(here, 'data', 'IHFC_2024_GHFDB_v.2026.03.txt'),
                 os.path.join(here, 'IHFC_2024_GHFDB_v.2026.03.txt')):
        if os.path.exists(cand):
            return cand
    raise SystemExit(
        'IHFC database not found.\n'
        '  Set the environment variable IHFC_GHFDB to the file path, or place\n'
        '  IHFC_2024_GHFDB_v.2026.03.txt under ./data/ next to this script.\n'
        '  The dataset is openly available from GFZ Data Services, '
        'doi:10.5880/fidgeo.2024.014')



RES, FN, TILE_KM = 128, 40, 57  # FN须与训练数据集FN一致
DEM_PATH = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))

def taper_dem(dem):
    zm = dem.mean(); h, w = dem.shape
    k = max(2, int(h * 0.12))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    wy = np.ones(h); wx = np.ones(w)
    wy[:k] = ramp; wy[-k:] = ramp[::-1]; wx[:k] = ramp; wx[-k:] = ramp[::-1]
    return zm + (dem - zm) * np.outer(wy, wx)

def extract_dem(ds, lon, lat):
    """与训练阶段共用 tile_reader.read_tile,保证几何口径完全一致。

    reject_ocean 保持 False:测点已由 IHFC 的 environment 字段判定为陆地,而沿海
    测点的 57 km 瓦片本就可能过半是海面,在此拒绝会静默丢掉这些测点。
    """
    from tile_reader import read_tile
    try:
        dem = read_tile(ds, lon, lat)
    except Exception:
        return None
    return None if dem is None else dem.astype(np.float32)

KEEP_TOPO_CORRECTED = '--keep-topo-corrected' in sys.argv
topo_done_kept = set()
_OUT_POINTS = ('rebuild_points_with_topocorr.npz' if KEEP_TOPO_CORRECTED
               else 'rebuild_points.npz')
_OUT_GRID = ('rebuild_hf_05deg_with_topocorr.npz' if KEEP_TOPO_CORRECTED
             else 'rebuild_hf_05deg.npz')
if KEEP_TOPO_CORRECTED:
    print('*** 验证模式:保留已做地形校正的测点,输出 -> %s ***' % _OUT_POINTS)
    print('*** 该文件仅供 C15 有序检验使用,不得用作交付产品 ***')

print('=== Step 1: Parse IHFC (land-only, quality filtered) ===')
with open(_ihfc_path(), 'r', encoding='latin-1') as f:
    lines = f.readlines()

# Column mapping (from line 12):
# col 0=q, 3=lat, 4=lon, 6=environment, 64=Quality_Score_Parent
# 注意: Quality_Score_Parent 不是数值分数, 是复合字符串码 "U{score}.M{score}.PPPPPPP"
# (U-score数值不确定性 / M-score方法学质量 / P-flags场地扰动标识), 例如
# 'Ux.Mx.-------'、'U2.M3x.-------'。实测显示 U-score 83.4%为Ux(信息不足无法
# 计算,不等于数据差), M-score 53.5%为Mx——不能当成普通数值阈值过滤, 也不能只设
# 一个笼统质量阈值(IHFC_2024论文精读与写作指南.md 第7/14节明确警告过这一点)。
# 这里只把U段、M段原样拆出随点携带, 供后续敏感性检验使用, 不在此处做过滤。
all_pts = []
rejected_env = 0; rejected_q = 0; rejected_coord = 0; rejected_antarctica = 0
rejected_topo_done = 0

# ── Depth averaging of the transfer factor ────────────────────────────────────
# A borehole heat flow is the mean conductive flux over its logged interval, not
# the flux at the ground surface. The terrain perturbation decays with depth, so
# the surface value of B overstates the correction. depth_decay_characterize.py
# solves the finite-element problem with a refined vertical mesh at 40 real sites
# (10 in each |B-1| band) and integrates B(z) over [0, D] to obtain the retained
# fraction f(D) = (B_eff(D) - 1) / (B_surf - 1). The four bands agree closely
# (inter-quartile spread <= 0.07 out to 500 m), so a single curve is used.
_F_DEPTH_M = np.array([0.0, 100.0, 146.0, 250.0, 390.0, 500.0, 1000.0, 1450.0, 2000.0])
_F_DEPTH_V = np.array([1.000, 0.987, 0.970, 0.929, 0.873, 0.833, 0.677, 0.574, 0.476])
MEDIAN_DEPTH_M = 338.0   # median reported q_bottom over the retained sites

def f_depth(depth_m):
    """Retained fraction of the surface B departure for a borehole of depth D."""
    if depth_m is None or not np.isfinite(depth_m) or depth_m <= 0:
        depth_m = MEDIAN_DEPTH_M
    return float(np.interp(depth_m, _F_DEPTH_M, _F_DEPTH_V))


# ── 隐含深度平均:避免把同一段浅部扣两次 ────────────────────────────────────
# 有限元集合是用 nz=10 层、域深 max(3,000 m, 1.5 x relief) 求解的,因此顶层单元
# 厚 max(300 m, 0.15 x relief);而 B 取自最顶两层节点的有限差分,所以算子学到的
# "地表 B"本身已经是那一层上的平均通量,不是 z=0 的值。vertical_mesh_test.py 在
# 20 个 |B-1| >= 0.05 的测点上把 nz 从 10 加密到 80,证实生产解的 |B-1| 比收敛值
# 小 13.7%,对应的等效平均深度 405 m 与顶层单元厚中位 374 m 一致。
# 因此深度平均只应补上从隐含深度到实际测井深度的增量,而不是从 z=0 起算。
def center_above_min_m(dem):
    """测点所在中心列的地表高出瓦片最低点多少米(与 fem3d.build 同一口径:
    节点高程取 128→41 线性缩放后的中心节点,z_min 取原瓦片最小值)。"""
    from scipy.ndimage import zoom
    d = np.asarray(dem, dtype=np.float64)
    dc = zoom(d, (41 / d.shape[0], 41 / d.shape[1]), order=1)
    return float(dc[20, 20] - d.min())


def implicit_depth_m(relief_m, center_above_min=0.0):
    """生产有限元解在测点所在中心列隐含的平均深度 = 该列顶层单元厚度。

    """
    return (max(3000.0, 1.5 * float(relief_m)) + float(center_above_min)) / 10.0


def depth_factor(borehole_depth_m, relief_m, center_above_min=0.0):
    """f(D) / f(implicit) —— 只补增量,不重复扣减浅部。"""
    return f_depth(borehole_depth_m) / f_depth(implicit_depth_m(relief_m, center_above_min))

def _site_key(parent_id, lon, lat, q):
    """与下面父子去重所用的键一致:有 parent_id 用它,否则用坐标+通量。"""
    return parent_id if parent_id else ('__no_pid__', lon, lat, q)


def _site_coord(lon, lat):
    """地理位置键(约 11 m 精度)。同一位置在库中可能对应多个 parent_id,
    其中一个带 corr_TOPO 标记而另一个不带,仅按 parent_id 排除会漏掉这类测点。"""
    return (round(lon, 4), round(lat, 4))


topo_done_sites, topo_done_coords = set(), set()
for l in lines[13:]:
    parts = l.strip().split('\t')
    if len(parts) < 66: continue
    if parts[27].strip() != '[Present and corrected]': continue
    try:
        lat0, lon0, q0 = float(parts[3].strip()), float(parts[4].strip()), float(parts[0].strip())
    except (ValueError, IndexError): continue
    topo_done_sites.add(_site_key(parts[65].strip(), lon0, lat0, q0))
    topo_done_coords.add(_site_coord(lon0, lat0))

for l in lines[13:]:
    parts = l.strip().split('\t')
    if len(parts) < 66: continue
    try:
        lat = float(parts[3].strip())
        lon = float(parts[4].strip())
        q = float(parts[0].strip())
    except (ValueError, IndexError): continue
    qcode = parts[64].strip()
    qsegs = qcode.split('.')
    u_score = qsegs[0] if len(qsegs) > 0 else ''
    m_score = qsegs[1] if len(qsegs) > 1 else ''
    qual = f'{u_score}.{m_score}'
    parent_id = parts[65].strip() if len(parts) > 65 else ''
    # C15 corr_TOPO_flag (index 27) and C5 q_bottom (index 17); see the official
    # GHFDB data description shipped with the release.
    topo_flag = parts[27].strip() if len(parts) > 27 else ''
    try:
        q_bottom = float(parts[17].strip())
    except (ValueError, IndexError):
        q_bottom = float('nan')

    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        rejected_coord += 1; continue

    # Environment filter: only onshore continental + lakes/rivers
    env = parts[6].strip().lower()
    if 'onshore' not in env and 'lake' not in env and 'river' not in env:
        rejected_env += 1; continue

    # Quality range
    if q <= 5 or q > 500:
        rejected_q += 1; continue

    # Antarctica: never sampled in the training-tile domain (all six continental
    # regions below are explicitly "outside Antarctica"), and the ice-sheet surface
    # is a different terrain regime from the rock/soil topography the operator was
    # trained on, so applying it there is out-of-distribution. Excluded here to match
    # the scope stated throughout the manuscript.
    if lat <= -60:
        rejected_antarctica += 1; continue

    _is_topo_done = (_site_key(parent_id, lon, lat, q) in topo_done_sites
                     or _site_coord(lon, lat) in topo_done_coords)
    if _is_topo_done and not KEEP_TOPO_CORRECTED:
        rejected_topo_done += 1; continue
    if _is_topo_done:
        topo_done_kept.add(_site_coord(lon, lat))

    all_pts.append((lon, lat, q, qual, parent_id, q_bottom))

print(f'Onshore continental points (pre-dedup): {len(all_pts):,}')
print(f'  Rejected: env={rejected_env:,}, q_range={rejected_q:,}, coord={rejected_coord:,}, '
      f'antarctica={rejected_antarctica:,}, already-topo-corrected={rejected_topo_done:,}')

def _is_record_start(_l):
    _p = _l.strip().split('\t')
    if len(_p) >= 66:
        return True
    if not _p or not _p[0].strip():
        return False
    try:
        float(_p[0].strip())
        return True          # 断行记录的前半截
    except ValueError:
        return False         # 断行记录的后半截
_n_raw = sum(1 for _l in lines[13:] if _is_record_start(_l))
_c1 = _n_raw - rejected_env
_c2 = _c1 - rejected_q
_c3 = _c2 - rejected_antarctica
_c4 = _c3 - rejected_topo_done - rejected_coord   # = len(all_pts)
_filter_chain = [
    ('Database entries',      _n_raw),
    ('Onshore',               _c1),
    ('Plausible flux',        _c2),
    ('Outside Antarctica',    _c3),
    ('Not pre-corrected',     _c4),
]

# ═══ Step 1b: Parent/child record deduplication ═══
# IHFC 2024 uses a parent-record/child-record structure (see
# IHFC_2024论文精读与写作指南.md §6): a single physical site can appear as many
# rows (different depth intervals / correction assumptions), all carrying the same
# adopted parent q value. Counting every row as an independent site inflates n and
# double-weights heavily re-surveyed sites in the regional IQR filter and kriging.
# Keep exactly one row per parent_id.
seen_parent = {}
n_before_dedup = len(all_pts)
for lon, lat, q, qual, parent_id, q_bottom in all_pts:
    key = parent_id if parent_id else ('__no_pid__', lon, lat, q)
    if key not in seen_parent:
        seen_parent[key] = [lon, lat, q, qual, q_bottom]
    else:
        # Child rows of the same site cover different depth intervals; take the
        # deepest reported bottom as the interval the adopted flux represents.
        prev = seen_parent[key][4]
        if np.isfinite(q_bottom) and (not np.isfinite(prev) or q_bottom > prev):
            seen_parent[key][4] = q_bottom
all_pts = [tuple(v) for v in seen_parent.values()]
print(f'  Parent/child dedup: {n_before_dedup:,} rows -> {len(all_pts):,} unique sites '
      f'({n_before_dedup - len(all_pts):,} duplicate child rows removed)')
_filter_chain.append(('One row per site', len(all_pts)))

# ═══ Step 2: Outlier detection (IQR per broad region) ═══
print('\n=== Step 2: Outlier detection ===')
regions = {
    'N.America': (-170, -50, 15, 72),
    'S.America': (-80, -35, -55, 10),
    'Europe': (-10, 60, 35, 72),
    'Africa': (-20, 55, -35, 38),
    'Asia': (60, 150, 10, 55),
    'Oceania': (110, 155, -45, -10),
}

clean_pts = []
outlier_count = 0
# Two of the six boxes overlap slightly (Europe/Africa share lon[-10,55] x
# lat[35,38]); a point in that strip would otherwise be evaluated by both
# regional IQR passes and could be appended to clean_pts twice. Claim each
# point index to exactly one region (first match, in dict order) so every
# point is IQR-tested and counted only once.
claimed = set()
for rname, (lomin, lomax, lamin, lamax) in regions.items():
    region_idx = [i for i, (lon, lat, q, qual, dep) in enumerate(all_pts)
                  if lomin <= lon <= lomax and lamin <= lat <= lamax and i not in claimed]
    region_q = np.array([all_pts[i][2] for i in region_idx])
    if len(region_q) < 20: continue
    q1, q3 = np.percentile(region_q, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 3*iqr, q3 + 3*iqr
    n_out = int(((region_q < lo) | (region_q > hi)).sum())
    print(f'  {rname}: n={len(region_q):,}, IQR=[{q1:.0f},{q3:.0f}], range=[{lo:.0f},{hi:.0f}], outliers={n_out}')

    for i in region_idx:
        claimed.add(i)
        lon, lat, q, qual, dep = all_pts[i]
        if lo <= q <= hi:
            clean_pts.append((lon, lat, q, qual, dep))
        else:
            outlier_count += 1

# Also include points outside predefined regions (use global IQR)
remaining = [pt for i, pt in enumerate(all_pts) if i not in claimed]
if remaining:
    rq = np.array([q for _, _, q, _, _ in remaining])
    q1, q3 = np.percentile(rq, [25, 75])
    iqr = q3 - q1; lo, hi = q1 - 3*iqr, q3 + 3*iqr
    for lon, lat, q, qual, dep in remaining:
        if lo <= q <= hi: clean_pts.append((lon, lat, q, qual, dep))
        else: outlier_count += 1

print(f'\nClean points: {len(clean_pts):,} (removed {outlier_count:,} outliers)')

# ═══ Step 3: FNO correction ═══
print('\n=== Step 3: FNO correction ===')
from tile_reader import CELL_M as TR_CELL_M
from step3_train_large import FNO3D, D4Average
ck = torch.load('fno3d_large_model.pt', map_location='cuda', weights_only=False)
model = FNO3D(modes=24, width=96, H_out=FN+1, W_out=FN+1, n_layers=6).cuda().eval()
model.load_state_dict(ck['state'])
model = D4Average(model).eval()
print('推理:8 朝向平均(D4Average)')
print('深度换算:D_0 取测点所在中心列的顶层厚度(M5)')
X_m, X_s = ck['X_mean'], ck['X_std']; Y_m, Y_s = ck['Y_mean'], ck['Y_std']

# Build land mask using Natural Earth or a simple DEM-based approach
# Instead of B_global (which has sparse coverage), use the points themselves
RES05 = 0.5
lons_05 = np.arange(-179.75, 180, RES05)  # 720
lats_05 = np.arange(89.75, -90, -RES05)   # 360, north-to-south

# Build land mask from onshore point density
# Any 0.5° cell containing at least one of our clean onshore points → land
land_05 = np.zeros((len(lats_05), len(lons_05)), dtype=bool)
for lon, lat, _, _, _ in clean_pts:
    j = int((lon + 180) / RES05)
    i = int((90 - lat) / RES05)
    if 0 <= i < len(lats_05) and 0 <= j < len(lons_05):
        land_05[i, j] = True
# Expand land mask by 1 cell in all directions to capture coastal interpolation
from scipy.ndimage import binary_dilation
land_05 = binary_dilation(land_05, iterations=1)
print(f'Land mask from points (0.5°): {int(land_05.sum()):,} cells')

# All points need direct FNO (B_global is too sparse)
need_direct = list(clean_pts)
print(f'All {len(need_direct):,} points processed via direct FNO')

# Direct FNO for all points
# results 元组末尾携带 quality_score, 供后续"剔除低质量评分点是否显著改变结果"
# 的敏感性检验使用(附录用), 不在本脚本内做过滤
results = []
if need_direct:
    print('Running direct FNO (all points)...')
    ds = rasterio.open(DEM_PATH)
    batch_dems, batch_info = [], []
    t0 = time.time()
    for idx, (lon, lat, q, qual, dep) in enumerate(need_direct):
        dem = extract_dem(ds, lon, lat)
        if dem is not None:
            batch_dems.append(dem); batch_info.append((lon, lat, q, qual, dep))
        if len(batch_dems) >= 128 or idx == len(need_direct) - 1:
            if batch_dems:
                xb = (np.stack(batch_dems).reshape(-1, 1, 128, 128) - X_m) / X_s
                with torch.no_grad():
                    pred = model(torch.tensor(xb, dtype=torch.float32).cuda())
                c = FN // 2
                Bv = pred.cpu().numpy()[:, 0, c, c] * Y_s + Y_m
                for bi, (blon, blat, bq, bqual, bdep) in enumerate(batch_info):
                    b_surf = float(Bv[bi])
                    if 0.4 < b_surf < 2.5:
                        # Depth-averaged transfer factor: a borehole heat flow is the
                        # mean flux over its logged interval, and the terrain
                        # perturbation decays with depth, so the surface value of B
                        # overstates the correction (Methods, "Depth averaging").
                        # The operator output already carries the implicit averaging
                        # of the nz=10 top element, so only the increment is applied.
                        brel = float(batch_dems[bi].max() - batch_dems[bi].min())
                        babove = center_above_min_m(batch_dems[bi])
                        b_val = 1.0 + depth_factor(bdep, brel, babove) * (b_surf - 1.0)
                        results.append((blon, blat, bq, b_val, bq / b_val, bqual,
                                        b_surf, bdep))
            batch_dems, batch_info = [], []
        if (idx + 1) % 5000 == 0:
            e = time.time() - t0
            sys.stdout.write(f'\r  {idx+1}/{len(need_direct)} ({e:.0f}s)')
            sys.stdout.flush()
    ds.close()
    print(f'\n  Direct FNO done: {time.time()-t0:.0f}s')

print(f'Final corrected: {len(results):,} points')

_filter_chain.append(('Regional outliers out', len(results)))
if KEEP_TOPO_CORRECTED:
    # 验证模式的链条与交付产品不同,不可覆盖 Fig 1a 的数据源
    print('验证模式:跳过 filter_chain.json 写出,避免污染 Fig 1a')
else:
    with open('filter_chain.json', 'w', encoding='utf-8') as _fh:
        json.dump([{'label': k, 'n': int(v)} for k, v in _filter_chain],
                  _fh, ensure_ascii=False, indent=1)
    print('过滤级联已保存: filter_chain.json')
for _k, _v in _filter_chain:
    print(f'    {_k:<24} {_v:>8,}')

# ═══ Step 4: Bin + Interpolate ═══
print('\n=== Step 4: 0.5° interpolation ===')
lon_arr = np.array([r[0] for r in results])
lat_arr = np.array([r[1] for r in results])
q_meas = np.array([r[2] for r in results])
B_arr = np.array([r[3] for r in results])
q_corr = np.array([r[4] for r in results])
qual_arr = np.array([r[5] for r in results])
B_surf_arr = np.array([r[6] for r in results])
depth_arr = np.array([r[7] for r in results])

def bin_to_grid(lon_pts, lat_pts, vals):
    grid = np.full((len(lats_05), len(lons_05)), np.nan, dtype=np.float32)
    for i in range(len(lats_05)):
        for j in range(len(lons_05)):
            # lats_05[i] = top of cell (north side), lats_05[i+1] = bottom
            lat_hi = lats_05[i]
            lat_lo = lats_05[i+1] if i+1 < len(lats_05) else -90
            m = (lat_pts <= lat_hi) & (lat_pts > lat_lo) & \
                (lon_pts >= lons_05[j]) & (lon_pts < lons_05[j] + RES05)
            if m.sum() >= 1:
                grid[i, j] = np.median(vals[m])
    return grid

grid_qm_raw = bin_to_grid(lon_arr, lat_arr, q_meas)
grid_qc_raw = bin_to_grid(lon_arr, lat_arr, q_corr)
print(f'Cells with >=2 points: {int(np.isfinite(grid_qm_raw).sum()):,}')

# Gaussian fill + clip to land
def fill_and_clip(grid_raw, land_mask, sigma=5):
    smooth = gaussian_filter(np.nan_to_num(grid_raw, nan=0), sigma=sigma)
    wt = gaussian_filter(np.isfinite(grid_raw).astype(float), sigma=sigma)
    filled = np.where(np.isfinite(grid_raw), grid_raw,
                      np.where((wt > 0.003) & land_mask,
                               smooth / np.maximum(wt, 0.003), np.nan))
    # CLIP everything to land mask
    filled[~land_mask] = np.nan
    return filled.astype(np.float32)

grid_qm = fill_and_clip(grid_qm_raw, land_05)
grid_qc = fill_and_clip(grid_qc_raw, land_05)

# Clip corrections to physically reasonable range before computing dq
grid_dq = grid_qc - grid_qm

valid = int(np.isfinite(grid_qm).sum())
print(f'After fill+clip: {valid:,} cells ({100*valid/land_05.sum():.0f}% land)')
print(f'q_meas: [{np.nanmin(grid_qm):.0f}, {np.nanmax(grid_qm):.0f}], mean={np.nanmean(grid_qm):.1f}')
print(f'q_corr: [{np.nanmin(grid_qc):.0f}, {np.nanmax(grid_qc):.0f}], mean={np.nanmean(grid_qc):.1f}')
print(f'dq: [{np.nanmin(grid_dq):.0f}, {np.nanmax(grid_dq):.0f}], mean={np.nanmean(grid_dq):.1f}')

# Save
np.savez_compressed(_OUT_GRID,
                    grid_qm=grid_qm, grid_qc=grid_qc, grid_dq=grid_dq,
                    lons=lons_05.astype(np.float32), lats=lats_05.astype(np.float32),
                    land_mask=land_05)
# Also save raw points for reference
# quality_score = IHFC Quality_Score_Parent (col 64), NaN若原始数据缺失,
# 用于论文附录的"剔除低质量评分点后结果是否显著变化"敏感性检验
# 验证模式下额外标记哪些测点属于 C15="Present and corrected",
# 供 c15_ordinal_test.py 取出最强的那一档。
_extra = {}
if KEEP_TOPO_CORRECTED:
    _flag = np.array([_site_coord(float(a), float(b)) in topo_done_kept
                      for a, b in zip(lon_arr, lat_arr)], dtype=bool)
    _extra['is_topo_corrected'] = _flag
    print(f'其中已做地形校正的测点: {int(_flag.sum()):,}')
np.savez_compressed(_OUT_POINTS,
                    lon=lon_arr, lat=lat_arr,
                    q_measured=q_meas, B=B_arr, q_corrected=q_corr,
                    quality_score=qual_arr,
                    B_surface=B_surf_arr, borehole_depth=depth_arr, **_extra)
print(f'Saved: {_OUT_GRID} + {_OUT_POINTS}')
print('Done.')
