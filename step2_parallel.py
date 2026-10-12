#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_parallel.py — 全球DEM并行采样 + 3D FEM批量真值生成

用法:
  # 一次跑完
  python step2_parallel.py \
    --dem /data/SRTM15-450米全球一张图.tif \
    --out global_fem_dataset.npz \
    --target 8000 \
    --workers 8

  # 分批跑: 每次新增800个有效瓦片后自动暂停, 重复运行同一命令即可续跑
  python step2_parallel.py --dem <dem.tif> --target 8000 --max-per-bin 1600 --batch-limit 800

输出:
  global_fem_dataset.npz   — X(埋入)=[N,1,128,128], Y(B场)=[N,1,21,21]
  global_fem_checkpoint/   — 断点目录 (分片npz + state.json), 完成后自动清理
"""

import sys, io, os, time, json, shutil, argparse, traceback
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy.ndimage import zoom
from multiprocessing import Pool, Manager, cpu_count
import multiprocessing as mp

# ── 全局常量 ──
RES = 128        # DEM 重采样分辨率
FN = 40
TILE_KM = 57     # 瓦片边长 km;有限元单元尺寸取 tile_reader.CELL_M = 57 km / 128 ≈ 445 m
SEED = 42

CONTINENTS = {
    'Asia':      (60, 180, 5, 78),
    'Europe':    (-10, 60, 35, 82),
    'N_America': (-170, -10, 15, 84),
    'S_America': (-80, -35, -55, 10),
    'Africa':    (-20, 55, -35, 38),
    'Oceania':   (95, 180, -50, 5),
}

# ── 地形分层 ──
RELIEF_BINS = [30, 100, 300, 800, 2000, 1e9]
RELIEF_NAMES = ['flat', 'gentle', 'hilly', 'mountain', 'extreme']


def taper_dem(dem):
    """对DEM边缘做余弦衰减, 减少边界效应"""
    zm = dem.mean()
    h, w = dem.shape
    k = max(2, int(h * 0.12))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    wy, wx = np.ones(h), np.ones(w)
    wy[:k] = ramp; wy[-k:] = ramp[::-1]
    wx[:k] = ramp; wx[-k:] = ramp[::-1]
    return zm + (dem - zm) * np.outer(wy, wx)


def process_one_tile(args):
    """
    处理单个候选点 → 返回结果dict 或 None(跳过)
    独立进程内运行, 自备FEM和rasterio句柄
    """
    lon, lat, cont, dem_path, idx = args

    try:
        import rasterio
        from fem3d import Fem3D

        # 统一走 tile_reader:经向按 1/cos(纬度) 多读像元,使瓦片真实为 57 km 见方
        from tile_reader import read_tile
        with rasterio.open(dem_path) as ds:
            dem = read_tile(ds, lon, lat, reject_ocean=True)
        if dem is None:
            return None

        # 快速预过滤 (450m DEM 起伏阈值放宽至25m)
        if dem.max() - dem.min() < 25:
            return None

        relief = dem.max() - dem.min()
        if relief < 15:
            return None

        # 3D FEM
        f = Fem3D(nx=FN, ny=FN, nz=10)
        from tile_reader import CELL_M as TR_CELL_M
        f.build(dem, TR_CELL_M, depth=max(3000, relief * 1.5))
        f.solve_heat()
        Bv = f.compute_B()

        if Bv.min() < 0.01 or Bv.max() > 10 or not np.all(np.isfinite(Bv)):
            return None

        return {
            'dem': dem.astype(np.float32),
            'B': Bv.astype(np.float32),
            'lon': float(lon), 'lat': float(lat),
            'cont': cont, 'relief': float(relief),
        }

    except Exception as e:
        # 打印前几个异常帮助诊断, 避免静默失败
        import threading
        err_counter = getattr(process_one_tile, '_err_count', 0) + 1
        setattr(process_one_tile, '_err_count', err_counter)
        if err_counter <= 5:
            print(f'[WARN] 瓦片 #{idx} ({lon:.2f},{lat:.2f}) 失败: {type(e).__name__}: {e}',
                  file=sys.stderr, flush=True)
        return None


def stratified_sample(results, target=None):
    """报告地形自然分布, 不做人为截断。
    450m DEM + 57km窗口下, 地形类别分布由物理决定而非先验分桶 —
    论文中如实报告自然分布即可。
    """
    strat = {n: [] for n in RELIEF_NAMES}
    for r in results:
        for j, (tl, th) in enumerate(zip(RELIEF_BINS[:-1], RELIEF_BINS[1:])):
            if tl <= r['relief'] < th:
                strat[RELIEF_NAMES[j]].append(r)
                break

    print('  自然分布 (450m DEM, 57km 窗口):')
    for name in RELIEF_NAMES:
        pct = len(strat[name]) / max(1, len(results)) * 100
        print(f'    {name}: {len(strat[name])} ({pct:.1f}%)')

    all_tiles = [r for name in RELIEF_NAMES for r in strat[name]]
    np.random.shuffle(all_tiles)

    if target is not None and len(all_tiles) > target:
        return all_tiles[:target]
    return all_tiles


# ════════════════════════════════════════════════════════════
# 候选点流式生成器 (确定性无限流, 永不枯竭)
# ════════════════════════════════════════════════════════════

def _advance_rng(rng, n_candidates, continents):
    """快进 RandomState, 跳过 n_candidates 个候选点对应的随机数消耗。
    每个候选点消耗: 1次randint(选洲) + 2次uniform(经纬度)。"""
    skip = n_candidates
    while skip > 0:
        n = min(skip, 100000)
        for _ in range(n):
            rng.randint(0, len(continents))
            rng.uniform(0, 1)
            rng.uniform(0, 1)
        skip -= n


def _generate_chunk(rng, continents, count, start_idx):
    """从 RandomState 流式生成 count 个候选点 (会改变 rng 状态)。
    返回 [(lon, lat, continent, global_index), ...]"""
    pts = []
    for i in range(count):
        ci = rng.randint(0, len(continents))
        nm, (lomin, lomax, lamin, lamax) = continents[ci]
        lon = rng.uniform(lomin, lomax)
        lat = rng.uniform(lamin, lamax)
        pts.append((lon, lat, nm, start_idx + i))
    return pts


# ════════════════════════════════════════════════════════════
# 断点: 目录式分片存储
#   <ckpt>/state.json          — {'processed', 'target', 'seed'}
#   <ckpt>/shard_XXXXXXX.npz   — 每个chunk的新增有效瓦片 (以候选点偏移命名)
# 分片以chunk起始偏移命名 → 崩溃后重算同一chunk会幂等覆盖, 不会重复
# ════════════════════════════════════════════════════════════

def _atomic_write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def save_shard(ckpt_dir, offset, new_results):
    path = os.path.join(ckpt_dir, f'shard_{offset:09d}.npz')
    tmp = path + '.tmp.npz'
    np.savez_compressed(
        tmp,
        dems=np.stack([r['dem'] for r in new_results]),
        Bs=np.stack([r['B'].ravel() for r in new_results]),
        lons=np.array([r['lon'] for r in new_results]),
        lats=np.array([r['lat'] for r in new_results]),
        reliefs=np.array([r['relief'] for r in new_results]),
        conts=np.array([r['cont'] for r in new_results]),
    )
    os.replace(tmp, path)


def load_checkpoint(ckpt_dir, target):
    """返回 (results, processed); 无断点时 ([], 0)"""
    state_path = os.path.join(ckpt_dir, 'state.json')
    if not os.path.exists(state_path):
        return [], 0

    state = json.load(open(state_path, encoding='utf-8'))
    if state.get('target') != target:
        print(f'错误: 断点记录的 target={state.get("target")} 与本次 --target {target} 不一致!')
        print('候选点序列由 target 决定, 中途更改会导致采样重复/遗漏。')
        print(f'请沿用原 target 继续, 或删除断点目录后重新开始: {ckpt_dir}')
        sys.exit(1)

    processed = int(state.get('processed', 0))
    results = []
    for fn in sorted(os.listdir(ckpt_dir)):
        if not (fn.startswith('shard_') and fn.endswith('.npz')):
            continue
        offset = int(fn[len('shard_'):-len('.npz')])
        if offset >= processed:
            continue  # 上次崩溃留下的超前分片, 对应chunk会重算并覆盖
        d = np.load(os.path.join(ckpt_dir, fn))
        for i in range(len(d['lons'])):
            results.append({
                'dem': d['dems'][i],
                'B': d['Bs'][i],
                'lon': float(d['lons'][i]), 'lat': float(d['lats'][i]),
                'cont': str(d['conts'][i]), 'relief': float(d['reliefs'][i]),
            })
    return results, processed


def main():
    parser = argparse.ArgumentParser(description='全球DEM+FEM并行数据集生成')
    parser.add_argument('--dem', required=True, help='SRTM15 DEM .tif 路径')
    parser.add_argument('--out', default='global_fem_dataset.npz', help='输出 .npz')
    parser.add_argument('--target', type=int, default=8000, help='目标有效瓦片数')
    parser.add_argument('--workers', type=int, default=None, help='并行进程数 (默认CPU-1)')
    parser.add_argument('--ckpt', default='global_fem_checkpoint', help='断点目录')
    parser.add_argument('--chunk', type=int, default=200, help='每批提交数')
    parser.add_argument('--max-per-bin', type=int, default=1600, help='各地形类型上限')
    parser.add_argument('--batch-limit', type=int, default=0,
                        help='本次运行最多新增的有效瓦片数, 达到后保存断点并暂停 (0=不限, 一直跑到target)')
    args = parser.parse_args()

    n_workers = args.workers or max(1, cpu_count() - 1)
    print(f'配置: target={args.target}, workers={n_workers}, DEM={args.dem}')
    print(f'地形上限/类: {args.max_per_bin} (总计 ≤ {args.max_per_bin * 5})')
    if args.batch_limit > 0:
        print(f'分批模式: 本次最多新增 {args.batch_limit} 瓦片, 达到后自动暂停')

    # ── 断点恢复 ──
    results, processed = load_checkpoint(args.ckpt, args.target)
    if processed > 0:
        print(f'从断点恢复: {len(results)} 有效瓦片, 已处理 {processed} 候选点')
    else:
        print('无断点, 从头开始')

    batch_paused = False

    if len(results) >= args.target:
        print(f'已有 {len(results)} 瓦片, 达到目标!')
    else:
        # ── 流式候选点生成 (确定性无限流, 永不枯竭) ──
        cand_rng = np.random.RandomState(SEED)
        continents_list = list(CONTINENTS.items())
        # 快进到断点位置 (每个候选点消耗 randint+2×uniform)
        _advance_rng(cand_rng, processed, continents_list)

        global_offset = processed
        os.makedirs(args.ckpt, exist_ok=True)
        state_path = os.path.join(args.ckpt, 'state.json')

        new_this_run = 0
        t0 = time.time()
        pool = Pool(processes=n_workers)

        try:
            while len(results) < args.target:
                # 流式生成下一个chunk的候选点
                chunk_cands = _generate_chunk(cand_rng, continents_list, args.chunk, global_offset)
                tasks = [(lon, lat, cont, args.dem, idx)
                         for (lon, lat, cont, idx) in chunk_cands]

                chunk_results = pool.map(process_one_tile, tasks)

                new_results = [r for r in chunk_results if r is not None]
                results.extend(new_results)
                new_this_run += len(new_results)

                offset = global_offset
                global_offset += len(tasks)
                processed = global_offset

                # 断点保存: 先写分片, 再更新state (崩溃时最多重算一个chunk)
                if new_results:
                    save_shard(args.ckpt, offset, new_results)
                _atomic_write_json(state_path, {
                    'processed': processed,
                    'target': args.target,
                    'seed': SEED,
                    'n_results': len(results),
                })

                elapsed = time.time() - t0
                rate = len(results) / max(1, elapsed / 60)
                eta = (args.target - len(results)) / max(0.01, rate)
                msg = (f'  [{len(results)}/{args.target}] '
                       f'{rate:.1f}瓦片/min, '
                       f'ETA {eta:.1f}min, '
                       f'已处理 {processed} 候选')
                if args.batch_limit > 0:
                    msg += f' | 本批 {new_this_run}/{args.batch_limit}'
                print(msg)

                # 分批模式: 本批额度用完就暂停 (已达总目标则不暂停, 直接收尾)
                if (args.batch_limit > 0 and new_this_run >= args.batch_limit
                        and len(results) < args.target):
                    batch_paused = True
                    break
        finally:
            pool.close()
            pool.join()

        elapsed = time.time() - t0

        if batch_paused:
            print(f'\n本批完成: 新增 {new_this_run} 瓦片, 耗时 {elapsed/60:.1f}min')
            print(f'累计进度: {len(results)}/{args.target} 有效瓦片, 已处理 {processed} 候选点')
            print(f'断点已保存: {args.ckpt}/')
            print('下次运行相同命令即可自动续跑。')
            return

        print(f'\n采集完成: {len(results)} 瓦片, 耗时 {elapsed/60:.1f}min')

    # ── 分层抽样 (报告分布, 不做人为截断) ──
    print('\n地形分布:')
    final = stratified_sample(results, target=args.target)
    print(f'最终: {len(final)} 瓦片')

    # ── 保存 .npz ──
    N = len(final)
    X = np.zeros((N, 1, RES, RES), dtype=np.float32)
    Y = np.zeros((N, 1, FN + 1, FN + 1), dtype=np.float32)
    lons = np.zeros(N, dtype=np.float32)
    lats = np.zeros(N, dtype=np.float32)
    reliefs = np.zeros(N, dtype=np.float32)
    for i, r in enumerate(final):
        X[i, 0] = r['dem']
        Y[i, 0] = r['B'].reshape(FN + 1, FN + 1)
        lons[i] = r['lon']
        lats[i] = r['lat']
        reliefs[i] = r['relief']

    np.savez_compressed(args.out, X=X, Y=Y, lon=lons, lat=lats, relief=reliefs)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f'已保存: {args.out} ({size_mb:.1f} MB)')
    print(f'X shape: {X.shape}, Y shape: {Y.shape}')
    print(f'B range: [{Y.min():.4f}, {Y.max():.4f}]')

    # 清理断点
    if os.path.isdir(args.ckpt):
        shutil.rmtree(args.ckpt)
        print('已清理断点目录')

    print('完成!')


if __name__ == '__main__':
    mp.freeze_support()
    main()
