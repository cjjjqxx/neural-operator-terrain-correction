"""图3f 所用安第斯示例测点的瓦片 B 场(算子,8 朝向平均)。

"""
import os
import numpy as np
import rasterio
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
from tile_reader import read_tile
from step3_train_large import FNO3D, D4Average

DEM_PATH = os.environ.get('SRTM15_DEM', os.path.join('data', 'SRTM15_450m.tif'))
FN = 40                                   # 与 rebuild_correction.py 一致
LON0, LAT0 = -69.93, -32.68               # 4.2 节与图3f 的安第斯测点

pts = np.load('rebuild_points.npz')
i = int(np.argmin((pts['lon'] - LON0) ** 2 + (pts['lat'] - LAT0) ** 2))
lon, lat, b_ref = float(pts['lon'][i]), float(pts['lat'][i]), float(pts['B_surface'][i])

with rasterio.open(DEM_PATH) as ds:
    dem = read_tile(ds, lon, lat).astype(np.float32)

ck = torch.load('fno3d_large_model.pt', map_location='cuda', weights_only=False)
model = FNO3D(modes=24, width=96, H_out=FN + 1, W_out=FN + 1, n_layers=6).cuda().eval()
model.load_state_dict(ck['state'])
model = D4Average(model).eval()
xb = (dem.reshape(1, 1, 128, 128) - ck['X_mean']) / ck['X_std']
with torch.no_grad():
    B = model(torch.tensor(xb, dtype=torch.float32).cuda()).cpu().numpy()[0, 0] * ck['Y_std'] + ck['Y_mean']

c = FN // 2
assert abs(B[c, c] - b_ref) < 1e-3, (B[c, c], b_ref)
np.savez('andes_example_tile.npz', B=B.astype(np.float32), dem=dem, lon=lon, lat=lat,
         B_center=float(B[c, c]), B_surface_product=b_ref)
print('安第斯示例瓦片: (%.3f, %.3f) 中心 B = %.4f, 产物 B_surface = %.4f, 场范围 %.3f–%.3f'
      % (lon, lat, B[c, c], b_ref, B.min(), B.max()))
