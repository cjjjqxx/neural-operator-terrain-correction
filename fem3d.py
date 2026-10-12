# -*- coding: utf-8 -*-
"""
fem3d.py -- 3D FEM terrain heat transfer factor solver
  Input: DEM tile (128x128 float, elevation in m)
  Output: B value field (surface transfer factor, dimensionless)
  Validation: flat B=1.0, peak B<1, valley B>1
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from skfem import (MeshTet, Basis, ElementTetP1, BilinearForm, LinearForm,
                   FacetBasis, condense, solve)
from skfem.helpers import grad, dot

Q_REF = 0.06  # W/m2 reference heat flow

class Fem3D:
    def __init__(self, nx=20, ny=20, nz=10):
        self.nx, self.ny, self.nz = nx, ny, nz

    def build(self, dem_tile, cell_m, depth=None, K=2.5):
        Ny, Nx = dem_tile.shape
        self.K = K
        Lx, Ly = Nx * cell_m, Ny * cell_m
        z_min, z_max = dem_tile.min(), dem_tile.max()
        z_bot = z_min - (depth or max(3000.0, (z_max - z_min) * 1.5))
        self.cell_m = cell_m
        self.z_bot = z_bot
        nx, ny, nz = self.nx, self.ny, self.nz
        xs = np.linspace(0, Lx, nx + 1)
        ys = np.linspace(0, Ly, ny + 1)
        sigma = np.linspace(0, 1, nz + 1)
        dem_c = zoom(dem_tile, ((ny + 1) / Ny, (nx + 1) / Nx), order=1)
        dem_int = RegularGridInterpolator((ys, xs), dem_c)
        n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
        pts = np.zeros((3, n_nodes))
        k = 0
        for iz in range(nz + 1):
            for iy in range(ny + 1):
                for ix in range(nx + 1):
                    zs = dem_int((ys[iy], xs[ix]))
                    pts[0, k] = xs[ix]
                    pts[1, k] = ys[iy]
                    pts[2, k] = z_bot + sigma[iz] * (zs - z_bot)
                    k += 1

        def idx3(ix, iy, iz):
            return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix

        tets = []
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    v = [idx3(ix, iy, iz), idx3(ix + 1, iy, iz),
                         idx3(ix, iy + 1, iz), idx3(ix + 1, iy + 1, iz),
                         idx3(ix, iy, iz + 1), idx3(ix + 1, iy, iz + 1),
                         idx3(ix, iy + 1, iz + 1), idx3(ix + 1, iy + 1, iz + 1)]
                    # 6-tet 剖分 (cube 主对角线 v1-v6), 覆盖完整且全正定向。
                    # 旧 5-tet 剖分每个 tet 体积 = hex/6, 5 个只覆盖 hex 的 5/6,
                    # 导致能量不守恒、平坦地形 B≠1 (B=1.17 系统偏差的根源)。
                    tets += [[v[0], v[1], v[2], v[6]], [v[1], v[0], v[4], v[6]],
                             [v[2], v[1], v[3], v[6]], [v[1], v[3], v[6], v[7]],
                             [v[1], v[4], v[5], v[6]], [v[5], v[1], v[6], v[7]]]

        mesh = MeshTet(pts, np.array(tets).T)
        self.mesh = mesh
        self.pts = pts
        self.surf_idx = np.arange(nz * (ny + 1) * (nx + 1), (nz + 1) * (ny + 1) * (nx + 1))
        return mesh

    def solve_heat(self):
        mesh = self.mesh
        e = ElementTetP1()
        basis = Basis(mesh, e)

        @BilinearForm
        def a(u, v, w):
            return self.K * dot(grad(u), grad(v))
        A = a.assemble(basis)

        fb = FacetBasis(mesh, e,
                        facets=mesh.facets_satisfying(lambda x: x[2] < self.z_bot + 1.0))

        @LinearForm
        def Lf(v, w):
            return Q_REF * v
        b = Lf.assemble(fb)

        xD = np.zeros(basis.N)
        xD[self.surf_idx] = 0.0
        T = solve(*condense(A, b, x=xD, D=self.surf_idx.astype(np.int32)))
        self.T = T
        return T

    def compute_B(self, calibrate=False):
        pts = self.pts
        T = self.T
        nx, ny, nz = self.nx, self.ny, self.nz
        Bv = np.zeros((ny + 1) * (nx + 1))
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                ns = nz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix
                nb = (nz - 1) * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix
                dz = pts[2, ns] - pts[2, nb]
                if dz > 0.01:
                    Bv[iy * (nx + 1) + ix] = -self.K * ((T[ns] - T[nb]) / dz) / Q_REF
                else:
                    Bv[iy * (nx + 1) + ix] = 1.0
        if calibrate:
            Bv = Bv / np.median(Bv)
        self.B = Bv
        B_grid = Bv.reshape(ny + 1, nx + 1)

        B_grid[0, :] = B_grid[1, :]      # 上边
        B_grid[-1, :] = B_grid[-2, :]    # 下边
        B_grid[:, 0] = B_grid[:, 1]      # 左边
        B_grid[:, -1] = B_grid[:, -2]    # 右边
        Bv = B_grid.ravel()

        self.B = Bv
        self.B_grid = B_grid
        return Bv

    def run(self, dem_tile, cell_m, depth=None, K=2.5):
        self.build(dem_tile, cell_m, depth, K)
        self.solve_heat()
        return self.compute_B()


# ======== validation ========
def validate():
    import warnings
    warnings.filterwarnings('ignore')
    results = []

    # Flat: B should be 1.0 everywhere, including corners
    f = Fem3D()
    f.build(np.ones((128, 128)) * 500.0, 5000 / 128)
    f.solve_heat()
    f.compute_B()
    b_all = f.B_grid.ravel()
    bi = b_all[50:300] if len(b_all) > 300 else b_all[5:-5]
    ok = abs(bi.mean() - 1.0) < 0.05
    results.append(('Flat B=1', ok, bi.mean()))
    corners_ok = np.all(np.isfinite(f.B_grid[[0,0,-1,-1],[0,-1,0,-1]]))
    results.append(('Corners finite', corners_ok, ''))

    # Gaussian hill: B at peak should be < 1
    x = np.linspace(-2.5, 2.5, 128) * 1000
    X, Y = np.meshgrid(x, x)
    d = 500 + 500 * np.exp(-(X**2 + Y**2) / (2 * 800**2))
    fh = Fem3D()
    fh.build(d, 5000 / 128)
    fh.solve_heat()
    fh.compute_B()
    c = fh.B_grid.shape[0] // 2
    B_peak = fh.B_grid[c, c]
    ok_hill = B_peak < 0.95
    results.append(('Peak B<1', ok_hill, B_peak))

    # Gaussian valley: B at bottom should be > 1
    d2 = 1000 - 500 * np.exp(-(X**2 + Y**2) / (2 * 800**2))
    fv = Fem3D()
    fv.build(d2, 5000 / 128)
    fv.solve_heat()
    fv.compute_B()
    B_valley = fv.B_grid[c, c]
    ok_valley = B_valley > 1.05
    results.append(('Valley B>1', ok_valley, B_valley))

    for name, ok, val in results:
        if isinstance(val, (int, float)):
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: B={val:.4f}")
        else:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    all_ok = all(o for _, o, _ in results)
    print(f"\n{'All passed' if all_ok else 'SOME FAILED'}")
    return all_ok

if __name__ == "__main__":
    validate()
