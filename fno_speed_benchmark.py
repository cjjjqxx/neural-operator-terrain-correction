# -*- coding: utf-8 -*-
"""算子推理耗时基准 —— 图 5a 的另一半。


更要紧的是**口径**。FEM 那个数是单瓦片串行实测:建一次网格、解一次方程。
7.67 ms 若是批量推理的摊薄值,两个数就不在同一口径上 —— 批量摊薄掉的
主机到显存传输、核函数启动、同步等待,恰恰是单独算一个点时躲不掉的开销。
把两者并排写成"×269",等于用算子最有利的口径对 FEM 最不利的口径。

因此这里分两种口径实测,让正文可以各取所需:
  · 单次调用(batch=1)  —— 使用者为一个测点算一次校正的真实耗时
  · 批量摊薄(batch=128)—— 本文算全球场时的实际单瓦片成本

GPU 是异步的,不同步就只能测到"把任务丢进队列"的时间。每次计时都显式
torch.cuda.synchronize(),并先跑若干次热身,排除首次调用的 cuDNN 选算法开销。
"""
import io
import json
import os
import sys
import time

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from step3_train_large import FNO3D                      # noqa: E402

FN = 40
CKPT = os.path.join(HERE, 'fno3d_large_model.pt')
# GPU 基准的批间波动不小(同一配置两次跑相差过 24%),多半来自频率调节与
# 其他进程抢占。热身与重复次数都调大,并在下面同时打印均值与中位,
# 好判断某一次结果是不是被噪声带偏了。
N_WARM = 30
N_REP = 200


def bench(model, x, sync):
    """返回每次调用的耗时(秒)列表。sync=False 时不同步 —— 只为演示不同步会测出什么。"""
    ts = []
    with torch.no_grad():
        for _ in range(N_WARM):
            model(x)
        if sync:
            torch.cuda.synchronize()
        for _ in range(N_REP):
            t0 = time.perf_counter()
            model(x)
            if sync:
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return np.array(ts)


dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ck = torch.load(CKPT, map_location=dev, weights_only=False)
model = FNO3D(modes=24, width=96, H_out=FN + 1, W_out=FN + 1, n_layers=6).to(dev).eval()
model.load_state_dict(ck['state'])
print('设备 %s | 参数 %.1fM'
      % (dev, sum(p.numel() for p in model.parameters()) / 1e6))

rng = np.random.RandomState(20260909)
res = dict(device=dev, n_rep=N_REP, n_warm=N_WARM)
print()
print('%8s %12s %12s %12s' % ('批大小', '每次调用', '每瓦片', 'P90/瓦片'))
for bs in (1, 8, 32, 128):
    x = torch.tensor(rng.randn(bs, 1, 128, 128).astype(np.float32)).to(dev)
    ts = bench(model, x, sync=True)
    per_call_ms = 1000 * float(np.mean(ts))
    per_tile_ms = per_call_ms / bs
    p90 = 1000 * float(np.percentile(ts, 90)) / bs
    res['batch%d_call_ms' % bs] = round(per_call_ms, 3)
    res['batch%d_per_tile_ms' % bs] = round(per_tile_ms, 4)
    print('%8d %9.2f ms %9.3f ms %9.3f ms' % (bs, per_call_ms, per_tile_ms, p90))

# 单点的端到端:使用者拿脚本算一个测点时,除了前向还要把数据搬上显存。
x1 = rng.randn(1, 1, 128, 128).astype(np.float32)
ts = []
with torch.no_grad():
    for _ in range(N_WARM):
        model(torch.tensor(x1).to(dev))
    torch.cuda.synchronize() if dev == 'cuda' else None
    for _ in range(N_REP):
        t0 = time.perf_counter()
        model(torch.tensor(x1).to(dev))
        if dev == 'cuda':
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
res['single_with_transfer_ms'] = round(1000 * float(np.mean(ts)), 3)

# CPU 口径。发布脚本的使用者多半没有 GPU,而"任何人都能跑"是本文的卖点之一 ——
# 那就得知道没有 GPU 时一个测点要等多久。
if dev == 'cuda':
    mc = FNO3D(modes=24, width=96, H_out=FN + 1, W_out=FN + 1, n_layers=6).eval()
    mc.load_state_dict(ck['state'])
    for bs in (1, 32):
        xc = torch.tensor(rng.randn(bs, 1, 128, 128).astype(np.float32))
        with torch.no_grad():
            for _ in range(2):
                mc(xc)
            tc = []
            for _ in range(6):
                t0 = time.perf_counter()
                mc(xc)
                tc.append(time.perf_counter() - t0)
        res['cpu_batch%d_per_tile_ms' % bs] = round(1000 * float(np.mean(tc)) / bs, 1)
    res['cpu_threads'] = torch.get_num_threads()
    print()
    print('CPU(%d 线程)  单次 %.0f ms  批量摊薄 %.0f ms/瓦片'
          % (res['cpu_threads'], res['cpu_batch1_per_tile_ms'],
             res['cpu_batch32_per_tile_ms']))

res['fno_single_ms'] = res['batch1_per_tile_ms']
res['fno_batch_ms'] = res['batch128_per_tile_ms']
json.dump(res, open(os.path.join(HERE, 'fno_speed_benchmark.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)

print()
print('单次调用(含主机→显存传输)  %.2f ms' % res['single_with_transfer_ms'])
print('批量摊薄(batch=128)        %.3f ms/瓦片' % res['fno_batch_ms'])
print('两者相差 %.0f 倍 —— 摊薄掉的是传输、核函数启动与同步等待'
      % (res['fno_single_ms'] / res['fno_batch_ms']))
print('已保存 fno_speed_benchmark.json')
