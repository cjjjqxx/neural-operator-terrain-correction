# -*- coding: utf-8 -*-
"""D9:量化两处过滤对所报告统计量的截断偏倚。



(2) 3xIQR 离群值过滤作用在**未校正**通量上,而本文的全部立论就是实测通量被地形
    系统性偏移(谷地偏高、山脊偏低),因此该过滤会优先切掉两端中被地形推出去的测点,
    直接影响随后报告的符号比例与平均位移。

做法
----
对两批被剔除的测点照常运行算子(但不施加校正),得到它们的 B,再报告:
  a. 这批测点的 |B-1| 分布,与保留测点对比;
  b. 若把它们计入表征性统计,中位数 / P90 / P99 / >10% 占比各会变成多少;
  c. 3xIQR 改用**校正后**通量重跑一次,比较两种口径下的保留测点数与符号比例。

"""
import sys, os, json, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get('IHFC_GHFDB', os.path.join('data', 'IHFC_2024_GHFDB_v.2026.03.txt'))
REGIONS = {'N.America': (-170, -50, 15, 72), 'S.America': (-80, -35, -55, 10),
           'Europe': (-10, 60, 35, 72), 'Africa': (-20, 55, -35, 38),
           'Asia': (60, 150, 10, 55), 'Oceania': (110, 155, -45, -10)}


def parse_db():
    """复现 rebuild_correction 的前置过滤,但**保留**被两处规则剔除的测点并打标。"""
    lines = open(DB, encoding='latin-1').read().split('\n')
    hdr = lines[12].split('\t')
    ix = {n: i for i, n in enumerate(hdr)}
    rows = []
    for l in lines[13:]:
        p = l.split('\t')
        if len(p) < 66:
            continue
        try:
            lat = float(p[3]); lon = float(p[4]); q = float(p[0])
        except (ValueError, IndexError):
            continue
        env = p[6].strip().lower()
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            continue
        if 'onshore' not in env and 'lake' not in env and 'river' not in env:
            continue
        if q <= 5 or q > 500 or lat <= -60:
            continue
        rows.append(dict(lon=lon, lat=lat, q=q,
                         topo=p[27].strip(), parent=p[65].strip()))
    return rows


def iqr_keep(lon, lat, q):
    """复现分区 3xIQR:返回布尔保留掩码。区外测点走全局 IQR。"""
    keep = np.zeros(len(q), bool)
    claimed = np.zeros(len(q), bool)
    for a, b, c, d in REGIONS.values():
        m = (~claimed) & (lon >= a) & (lon <= b) & (lat >= c) & (lat <= d)
        if m.sum() < 4:
            continue
        claimed |= m
        q1, q3 = np.percentile(q[m], [25, 75])
        lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
        keep[m] = (q[m] >= lo) & (q[m] <= hi)
    r = ~claimed
    if r.sum() >= 4:
        q1, q3 = np.percentile(q[r], [25, 75])
        lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
        keep[r] = (q[r] >= lo) & (q[r] <= hi)
    return keep


def stats(qm, B):
    rel = 100 * np.abs(qm / B - qm) / qm
    return dict(n=len(qm), median=float(np.median(rel)),
                p90=float(np.percentile(rel, 90)), p99=float(np.percentile(rel, 99)),
                frac_gt10=float(100 * (rel > 10).mean()),
                frac_B_gt1=float(100 * (B > 1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dem', required=True)
    ap.add_argument('--batch', type=int, default=256)
    a = ap.parse_args()

    import torch, rasterio
    from tile_reader import read_tile
    from step3_train_large import FNO3D, D4Average

    kept = np.load(os.path.join(HERE, 'rebuild_points.npz'))
    kept_key = set(zip(np.round(kept['lon'], 4), np.round(kept['lat'], 4)))

    rows = parse_db()
    # 测点层面去重(与主流水线一致):同一父记录只留一条
    seen, uniq = set(), []
    for r in rows:
        k = r['parent'] or f"{r['lon']:.4f}_{r['lat']:.4f}"
        if k in seen:
            continue
        seen.add(k); uniq.append(r)
    # 父记录任一子记录带 [Present and corrected] 即整点标记
    flagged_parents = {r['parent'] for r in rows if r['topo'] == '[Present and corrected]'}
    for r in uniq:
        r['was_topo_corrected'] = r['parent'] in flagged_parents

    lon = np.array([r['lon'] for r in uniq]); lat = np.array([r['lat'] for r in uniq])
    q = np.array([r['q'] for r in uniq])
    topo_flag = np.array([r['was_topo_corrected'] for r in uniq])
    in_kept = np.array([(round(x, 4), round(y, 4)) in kept_key for x, y in zip(lon, lat)])
    iqr_ok = iqr_keep(lon, lat, q)

    print(f'去重后测点 {len(uniq):,}')
    print(f'  已被原作者地形校正(整点剔除): {topo_flag.sum():,}')
    print(f'  被 3xIQR 剔除:              {(~iqr_ok).sum():,}')
    print(f'  最终保留(应≈36,393):        {in_kept.sum():,}\n')

    # ── 对所有测点跑算子 ──
    ck = torch.load(os.path.join(HERE, 'fno3d_large_model.pt'),
                    map_location='cuda', weights_only=False)
    net = FNO3D(modes=24, width=96, H_out=41, W_out=41, n_layers=6).cuda().eval()
    net.load_state_dict(ck['state'])
    model = D4Average(net).eval()
    Xm, Xs, Ym, Ys = ck['X_mean'], ck['X_std'], ck['Y_mean'], ck['Y_std']

    B = np.full(len(uniq), np.nan)
    with rasterio.open(a.dem) as ds:
        buf, idx = [], []
        for i in range(len(uniq)):
            d = read_tile(ds, lon[i], lat[i])
            if d is None:
                continue
            buf.append(d); idx.append(i)
            if len(buf) == a.batch or i == len(uniq) - 1:
                x = (np.stack(buf).reshape(-1, 1, 128, 128).astype(np.float32) - Xm) / Xs
                with torch.no_grad():
                    pr = model(torch.tensor(x).cuda()).cpu().numpy()[:, 0] * Ys + Ym
                B[idx] = pr[:, 20, 20]
                buf, idx = [], []
            if i % 5000 == 0:
                print(f'  推理 {i:,}/{len(uniq):,}', flush=True)

    ok = np.isfinite(B)
    out = {}
    print('\n=== a. 被剔除测点的地形效应有多大 ===')
    for lab, m in [('最终保留', in_kept & ok),
                   ('因已校正而剔除', topo_flag & ok),
                   ('因 3xIQR 剔除', (~iqr_ok) & ok)]:
        if m.sum() < 5:
            continue
        ab = np.abs(B[m] - 1)
        out[lab] = dict(n=int(m.sum()), median_absB1=float(np.median(ab)),
                        p90_absB1=float(np.percentile(ab, 90)),
                        frac_gt005=float(100 * (ab > 0.05).mean()))
        print(f'  {lab:<16} n={m.sum():>6,}  |B-1|中位={np.median(ab):.4f}  '
              f'P90={np.percentile(ab,90):.4f}  >5%占比={100*(ab>0.05).mean():.1f}%')

    print('\n=== b. 把被剔除测点计入表征性统计的影响 ===')
    base = stats(q[in_kept & ok], B[in_kept & ok])
    incl = in_kept | topo_flag | (~iqr_ok)
    full = stats(q[incl & ok], B[incl & ok])
    print(f'{"":22}{"中位%":>9}{"P90%":>9}{"P99%":>9}{">10%":>9}{"B>1占比":>10}')
    for lab, d in [('本文口径(保留集)', base), ('全部测点(含被剔除)', full)]:
        print(f'  {lab:<20}{d["median"]:>9.3f}{d["p90"]:>9.2f}{d["p99"]:>9.2f}'
              f'{d["frac_gt10"]:>9.2f}{d["frac_B_gt1"]:>10.2f}')
    out['representative'] = dict(kept=base, all_sites=full)

    print('\n=== c. 3xIQR 改用校正后通量 ===')
    qc = q / np.where(ok, B, 1.0)
    iqr_c = iqr_keep(lon, lat, qc)
    print(f'  基于未校正通量保留 {iqr_ok.sum():,};基于校正后通量保留 {iqr_c.sum():,}')
    print(f'  两者不一致的测点 {int((iqr_ok != iqr_c).sum()):,}')
    for lab, m in [('未校正口径', iqr_ok & ok), ('校正后口径', iqr_c & ok)]:
        print(f'  {lab}: B>1 占比 {100*(B[m]>1).mean():.2f}%')
    # 校正后口径下保留集的表征性统计(与本文口径同为地表 B_s,可直接比较)
    kc = iqr_c & ok & ~topo_flag
    ku = iqr_ok & ok & ~topo_flag
    out['representative']['iqr_uncorrected_kept'] = stats(q[ku], B[ku])
    out['representative']['iqr_corrected_kept'] = stats(q[kc], B[kc])
    out['representative']['no_iqr'] = stats(q[ok & ~topo_flag], B[ok & ~topo_flag])
    for lab in ('iqr_uncorrected_kept', 'iqr_corrected_kept', 'no_iqr'):
        d = out['representative'][lab]
        print(f'  {lab:<22} n={d["n"]:,} 中位 {d["median"]:.3f}% P90 {d["p90"]:.2f}% P99 {d["p99"]:.2f}% >10% {d["frac_gt10"]:.2f}% B>1 {d["frac_B_gt1"]:.2f}%')
    out['iqr'] = dict(keep_uncorrected=int(iqr_ok.sum()), keep_corrected=int(iqr_c.sum()),
                      n_differ=int((iqr_ok != iqr_c).sum()),
                      frac_B_gt1_uncorrected=float(100 * (B[iqr_ok & ok] > 1).mean()),
                      frac_B_gt1_corrected=float(100 * (B[iqr_c & ok] > 1).mean()))

    json.dump(out, open(os.path.join(HERE, 'truncation_bias_test.json'), 'w',
                        encoding='utf-8'), indent=2, ensure_ascii=False)
    print('\n已保存 truncation_bias_test.json')


if __name__ == '__main__':
    main()
