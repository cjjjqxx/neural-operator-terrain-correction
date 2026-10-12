import sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np, torch
from step3_train_large import FNO3D

RES, FN = 128, 40
ck = torch.load('fno3d_large_model.pt', map_location='cuda', weights_only=False)
model = FNO3D(modes=24, width=96, H_out=FN+1, W_out=FN+1, n_layers=6).cuda().eval()
model.load_state_dict(ck['state'])
X_m, X_s = ck['X_mean'], ck['X_std']

np.random.seed(0)
xb = (np.random.randn(128,1,RES,RES).astype(np.float32) - X_m)/X_s
xb_t = torch.tensor(xb, dtype=torch.float32).cuda()

print('warmup (not timed)...', flush=True)
with torch.no_grad():
    for _ in range(5):
        _ = model(xb_t)
torch.cuda.synchronize()
print('warmup done', flush=True)

print('timing 20 batches of 128 after warmup...', flush=True)
times=[]
with torch.no_grad():
    for i in range(20):
        torch.cuda.synchronize(); t0=time.time()
        pred = model(xb_t)
        torch.cuda.synchronize()
        dt = time.time()-t0
        times.append(dt)
        print(f'  batch {i}: {dt*1000:.1f}ms total, {dt*1000/128:.3f}ms/tile', flush=True)

times=np.array(times)
print(f'\nAfter warmup: mean={times.mean()*1000:.1f}ms/batch ({times.mean()*1000/128:.3f}ms/tile), median={np.median(times)*1000:.1f}ms/batch')
print(f'For 52272 points at this batched rate: {times.mean()*52272/128:.1f}s total')
