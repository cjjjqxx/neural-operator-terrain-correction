import os
import sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np, torch, rasterio
from scipy.ndimage import zoom
from step3_train_large import FNO3D

RES, FN, TILE_KM = 128, 40, 57
DEM_PATH = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))
N_MC = 20

def taper_dem(dem):
    zm = dem.mean(); h, w = dem.shape
    k = max(2, int(h * 0.12))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    wy = np.ones(h); wx = np.ones(w)
    wy[:k] = ramp; wy[-k:] = ramp[::-1]; wx[:k] = ramp; wx[-k:] = ramp[::-1]
    return zm + (dem - zm) * np.outer(wy, wx)

def extract_dem(ds, lon, lat):
    # 与应用阶段(rebuild_correction)完全同一读取入口,否则 dropout 离散度
    # 是在另一种几何上估的,与所报告的 B 对不上。
    from tile_reader import read_tile
    dem = read_tile(ds, lon, lat)
    return None if dem is None else dem.astype(np.float32)

t0 = time.time()
d = np.load('rebuild_points.npz')
lon, lat, q_meas, B_orig = d['lon'], d['lat'], d['q_measured'], d['B']
n_total = len(lon)
print(f'[{time.time()-t0:.0f}s] loaded {n_total} points', flush=True)

ck = torch.load('fno3d_large_model.pt', map_location='cuda', weights_only=False)
model = FNO3D(modes=24, width=96, H_out=FN+1, W_out=FN+1, n_layers=6).cuda()
model.load_state_dict(ck['state'])
X_m, X_s = ck['X_mean'], ck['X_std']; Y_m, Y_s = ck['Y_mean'], ck['Y_std']
c = FN // 2
print(f'[{time.time()-t0:.0f}s] model loaded', flush=True)

# warmup
dummy = torch.zeros(128,1,RES,RES, dtype=torch.float32).cuda()
model.eval()
with torch.no_grad():
    for _ in range(5): _ = model(dummy)
model.train()
with torch.no_grad():
    for _ in range(5): _ = model(dummy)
torch.cuda.synchronize()
print(f'[{time.time()-t0:.0f}s] GPU warmed up (eval+train paths)', flush=True)

# extract all DEMs once, cache in memory
print(f'[{time.time()-t0:.0f}s] extracting {n_total} DEM tiles...', flush=True)
ds = rasterio.open(DEM_PATH)
dems = np.zeros((n_total, RES, RES), dtype=np.float32)
valid = np.zeros(n_total, dtype=bool)
for i in range(n_total):
    dem = extract_dem(ds, lon[i], lat[i])
    if dem is not None:
        dems[i] = dem; valid[i] = True
    if (i+1) % 10000 == 0:
        print(f'  [{time.time()-t0:.0f}s] {i+1}/{n_total} extracted', flush=True)
ds.close()
print(f'[{time.time()-t0:.0f}s] DEM extraction done, valid={valid.sum()}/{n_total}', flush=True)

Xall = (dems.reshape(-1,1,RES,RES) - X_m) / X_s

# MC-dropout: N_MC passes, batch=128
model.train()  # dropout active; GroupNorm unaffected by train/eval
batch_size = 128
preds_sum = np.zeros(n_total, dtype=np.float64)
preds_sumsq = np.zeros(n_total, dtype=np.float64)
print(f'[{time.time()-t0:.0f}s] starting {N_MC} MC-dropout passes over {n_total} points...', flush=True)
# 优化:输入一次性驻留 GPU,避免每轮每批重新建 tensor 并 H2D 传输
# (36,392 点 / 128 = 285 批 x 20 轮 = 5,700 次传输)。累加也留在 GPU 上。
# 36,392 x 128 x 128 x 4B = 2.4 GB,显存放得下则整体上卡;放不下退回分批 H2D
try:
    Xgpu = torch.from_numpy(np.ascontiguousarray(Xall, dtype=np.float32)).cuda()
    ON_GPU = True
except RuntimeError:
    Xgpu = torch.from_numpy(np.ascontiguousarray(Xall, dtype=np.float32)).pin_memory()
    ON_GPU = False
print(f'  输入张量 {"驻留显存" if ON_GPU else "锁页内存分批传输"}', flush=True)
sum_g = torch.zeros(n_total, dtype=torch.float64, device='cuda')
sumsq_g = torch.zeros(n_total, dtype=torch.float64, device='cuda')
for k in range(N_MC):
    tk = time.time()
    with torch.no_grad():
        for s in range(0, n_total, batch_size):
            e = min(s+batch_size, n_total)
            xb = Xgpu[s:e] if ON_GPU else Xgpu[s:e].cuda(non_blocking=True)
            Bv = model(xb)[:, 0, c, c].double() * Y_s + Y_m
            sum_g[s:e] += Bv
            sumsq_g[s:e] += Bv * Bv
    preds_sum = sum_g.cpu().numpy(); preds_sumsq = sumsq_g.cpu().numpy()
    print(f'[{time.time()-t0:.0f}s] MC pass {k+1}/{N_MC} done ({time.time()-tk:.1f}s)', flush=True)

mean_B = preds_sum / N_MC
var_B = preds_sumsq/N_MC - mean_B**2
var_B = np.maximum(var_B, 0)
std_B = np.sqrt(var_B)

np.savez_compressed('mc_dropout_uncertainty.npz',
                     lon=lon, lat=lat, q_measured=q_meas, B_original=B_orig,
                     B_mc_mean=mean_B, B_mc_std=std_B, valid=valid, N_MC=N_MC)
print(f'[{time.time()-t0:.0f}s] SAVED mc_dropout_uncertainty.npz', flush=True)

print(f'\n=== Summary (valid points only, n={valid.sum()}) ===')
sv = valid
print(f'B_mc_mean vs original B: corr = {np.corrcoef(mean_B[sv], B_orig[sv])[0,1]:.4f}')
print(f'mean std_B = {std_B[sv].mean():.5f}, median = {np.median(std_B[sv]):.5f}')
print(f'std_B as %% of |B-1|: median relief-scaled...')
relief_proxy = np.abs(B_orig[sv]-1)
# correlation between uncertainty and departure from unity (relief proxy)
print(f'corr(std_B, |B-1|) = {np.corrcoef(std_B[sv], relief_proxy)[0,1]:.4f}')
for lo,hi,label in [(0,0.02,'|B-1|<0.02 (near-flat)'),(0.02,0.1,'0.02<=|B-1|<0.1'),(0.1,2,'|B-1|>=0.1 (high-relief)')]:
    m = sv & (relief_proxy>=lo) & (relief_proxy<hi)
    if m.sum()>0:
        print(f'  {label}: n={m.sum()}, mean std_B={std_B[m].mean():.5f}, median std_B={np.median(std_B[m]):.5f}')
print('DONE', flush=True)
