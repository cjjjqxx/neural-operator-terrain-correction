# -*- coding: utf-8 -*-
"""瓦片读取:训练与应用共用的唯一入口。



    59.4 km (南北) x 59.4 x cos(纬度) km (东西)

的地形当作正方形,东西方向被拉伸 1/cos(纬度) 倍,该方向坡度被系统性低估。实测
(latitude_scale_test.py)显示 |B-1| 因此偏低约 9%,并随纬度单调增大(赤道 1.4%,
60 度以上 18%)。

"""
import numpy as np
from scipy.ndimage import zoom

RES = 128                      # 重采样后的瓦片边长(像元)
TILE_KM = 57.0                 # 瓦片真实边长(km),南北与东西一致
CELL_M = TILE_KM * 1000 / RES  # 重采样后每像元米数(约 445 m,各向同性)
MAX_COL_HALF = 4000            # 经向半宽像元数上限,防止极高纬度读取过大


def taper_dem(dem):
    """边缘余弦渐变,减少零通量侧边界与截断坡面的冲突。宽度为瓦片边长的 12%。"""
    zm = dem.mean()
    h, w = dem.shape
    k = max(2, int(h * 0.12))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    wy, wx = np.ones(h), np.ones(w)
    wy[:k] = ramp; wy[-k:] = ramp[::-1]
    wx[:k] = ramp; wx[-k:] = ramp[::-1]
    return zm + (dem - zm) * np.outer(wy, wx)


def read_tile(ds, lon, lat, apply_taper=True, reject_ocean=False, tile_km=None, res=None):
    """读取以 (lon, lat) 为中心、真实边长 tile_km 见方的地形,返回 res x res 数组。

    ds 为已打开的 rasterio 数据集(EPSG:4326)。返回 None 表示该位置不可用
    (越界或无数据过多)。

    tile_km / res 默认取模块常量 TILE_KM / RES,即生产配置。窗口收敛性研究需要在
    多个窗口宽度下反复读取,传入 tile_km 即可;此时每像元米数为 tile_km*1000/res,
    调用方须把该值一并传给 Fem3D.build,否则窗口与网格尺度会脱节。

    reject_ocean 只供随机采样脚本使用:采样时需要排除落在海上的候选点。应用阶段
    必须保持 False —— 测点位置由 IHFC 的 environment 字段判定为陆地,而沿海测点的
    57 km 瓦片本就可能过半是海面,若在此拒绝会静默丢掉这些测点。海面在该 DEM 中编码
    为高程 0,作为近等温面参与求解,这一处理在方法部分说明。
    """
    tile_km = TILE_KM if tile_km is None else float(tile_km)
    res = RES if res is None else int(res)
    px_deg = abs(ds.transform.a)                       # 像元的经度跨度(度)
    py_deg = abs(ds.transform.e)                       # 像元的纬度跨度(度)
    half_lat_deg = tile_km / 2.0 / 111.320             # 半个瓦片对应的纬度跨度
    half_row = int(round(half_lat_deg / py_deg))
    coslat = max(np.cos(np.radians(lat)), 1e-3)
    half_col = int(round(half_lat_deg / px_deg / coslat))
    if half_col > MAX_COL_HALF:
        return None                                    # 过高纬度,经向窗口不合理

    r, c = ds.index(lon, lat)
    if r - half_row < 0 or r + half_row > ds.height:
        return None                                    # 纬向越界
    # 经向按整幅影像宽度取模,自然处理跨 180 度经线
    cols = (np.arange(c - half_col, c + half_col) % ds.width)
    rows = slice(r - half_row, r + half_row)
    if cols.max() - cols.min() == len(cols) - 1:       # 连续,直接窗口读取
        raw = ds.read(1, window=(( rows.start, rows.stop),
                                 (int(cols.min()), int(cols.max()) + 1))).astype(float)
    else:                                              # 跨接缝,分两段拼接
        split = int(np.argmax(np.diff(cols) < 0)) + 1
        a = ds.read(1, window=((rows.start, rows.stop),
                               (int(cols[0]), ds.width))).astype(float)
        b = ds.read(1, window=((rows.start, rows.stop),
                               (0, int(cols[-1]) + 1))).astype(float)
        raw = np.concatenate([a, b], axis=1)

    raw[raw < -1000] = np.nan
    if raw.size < 100 or np.isnan(raw).mean() > 0.5:
        return None
    if reject_ocean and (raw == 0).mean() > 0.5:       # 该 DEM 把海洋编码为高程 0
        return None
    m = ~np.isfinite(raw)
    if m.any():
        raw[m] = np.nanmean(raw)

    dem = zoom(raw, (res / raw.shape[0], res / raw.shape[1]), order=1)
    return taper_dem(dem) if apply_taper else dem


def cell_m_for(tile_km=None, res=None):
    """给定窗口宽度与分辨率,返回应传给 Fem3D.build 的每像元米数。"""
    return (TILE_KM if tile_km is None else float(tile_km)) * 1000 / (RES if res is None else int(res))


def tile_extent_km(ds, lat, tile_km=None):
    """返回该纬度处所读窗口的真实地面跨度(南北, 东西),单位 km。用于自检。"""
    px_deg, py_deg = abs(ds.transform.a), abs(ds.transform.e)
    half_lat_deg = (TILE_KM if tile_km is None else float(tile_km)) / 2.0 / 111.320
    half_row = int(round(half_lat_deg / py_deg))
    coslat = max(np.cos(np.radians(lat)), 1e-3)
    half_col = int(round(half_lat_deg / px_deg / coslat))
    ns = 2 * half_row * py_deg * 111.320
    ew = 2 * half_col * px_deg * 111.320 * np.cos(np.radians(lat))
    return ns, ew
