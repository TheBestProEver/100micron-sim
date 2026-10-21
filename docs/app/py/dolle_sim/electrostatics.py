"""Long-range electrostatics: Ewald-split particle-mesh Poisson solver for a grounded box.

Contact charging against the walls charges the whole cloud with one sign, and
at ~1 Pa nothing screens it. Then the space-charge field comes almost entirely
from distant grains, which a pair cutoff would throw away. The Coulomb
interaction is therefore split (as in Ewald / P3M):

  * short range (r < r_c ~ 3.5 sigma): exact pairwise sum with the complementary kernel
        F_sr(r) = k q_i q_j [ erfc(r / (sqrt2 sigma)) / r^2 + sqrt(2/pi) exp(-r^2 / 2 sigma^2) / (sigma r) ]
  * long range: the charge density smoothed by a Gaussian of width sigma is solved
    on a grid with Dirichlet (phi = 0) walls, i.e. a grounded conducting chamber,
    using a sine transform (exact eigenfunctions of the box Laplacian). The field
    is interpolated back to the grains with the same cloud-in-cell weights used to
    deposit the charge.

The two parts add up to the full Coulomb interaction plus the long-range image
charges of grounded walls.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.fft import dstn, idstn
from scipy.special import erfc

EPS0 = 8.8541878128e-12
K_E = 8.9875517923e9


def short_range_kernel(r, sigma):
    """Radial factor g(r) such that F_sr = k q_i q_j g(r) r_hat (complement of the smoothed long-range force)."""
    x = r / (math.sqrt(2.0) * sigma)
    return erfc(x) / r ** 2 + math.sqrt(2.0 / math.pi) * np.exp(-x * x) / (sigma * r)


class SpaceChargeSolver:
    MAX_CELLS = 2_000_000

    def __init__(self, size, sigma, spacing=None):
        self.size = np.asarray(size, float)
        self.sigma = float(sigma)
        # linear (CIC) interpolation of a field smoothed over sigma has error ~(h/sigma)^2/8 -> ~3%
        h = spacing or self.sigma / 2.0
        # very large chambers: coarsen the mesh and widen the Ewald split to match
        cells = np.prod(np.ceil(self.size / h))
        if cells > self.MAX_CELLS:
            h *= (cells / self.MAX_CELLS) ** (1 / 3)
            self.sigma = max(self.sigma, 2.0 * h)
        self.n = np.maximum(8, np.ceil(self.size / h).astype(int))      # cells per axis
        self.h = self.size / self.n
        nx, ny, nz = self.n
        # continuous Laplacian eigenvalues of sin(pi k x / L) modes, with Gaussian smoothing
        kx = math.pi * np.arange(1, nx) / self.size[0]
        ky = math.pi * np.arange(1, ny) / self.size[1]
        kz = math.pi * np.arange(1, nz) / self.size[2]
        K2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
        # deconvolve the cloud-in-cell window of deposition + interpolation, sinc^2 each (P3M)
        def win(k, h):
            return np.sinc(k * h / (2 * math.pi)) ** 2
        W = (win(kx, self.h[0])[:, None, None] * win(ky, self.h[1])[None, :, None]
             * win(kz, self.h[2])[None, None, :])
        self.green = np.exp(-0.5 * self.sigma ** 2 * K2) / (EPS0 * K2) / np.maximum(W, 1e-3) ** 2
        self.E = None

    def _cic(self, pos):
        f = pos / self.h                       # node coordinates (nodes at 0..n)
        i0 = np.clip(np.floor(f).astype(int), 0, self.n - 1)
        w1 = np.clip(f - i0, 0.0, 1.0)
        return i0, w1

    def solve(self, pos, q):
        """Compute the long-range field on the node grid from grain charges."""
        nx, ny, nz = self.n
        rho = np.zeros((nx + 1, ny + 1, nz + 1))
        i0, w1 = self._cic(pos)
        w0 = 1.0 - w1
        for dx in (0, 1):
            wx = w1[:, 0] if dx else w0[:, 0]
            for dy in (0, 1):
                wy = w1[:, 1] if dy else w0[:, 1]
                for dz in (0, 1):
                    wz = w1[:, 2] if dz else w0[:, 2]
                    np.add.at(rho, (i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz), q * wx * wy * wz)
        rho /= np.prod(self.h)
        inner = rho[1:-1, 1:-1, 1:-1]           # boundary nodes are grounded (charge there is absorbed)
        phi_in = idstn(dstn(inner, type=1) * self.green, type=1)
        phi = np.zeros_like(rho)
        phi[1:-1, 1:-1, 1:-1] = phi_in
        gx, gy, gz = np.gradient(phi, *self.h, edge_order=2)
        self.E = np.stack((-gx, -gy, -gz), axis=-1)
        return self.E

    def field_at(self, pos):
        """Cloud-in-cell interpolation of the long-range field at grain positions."""
        if self.E is None:
            return np.zeros_like(pos)
        i0, w1 = self._cic(pos)
        w0 = 1.0 - w1
        out = np.zeros_like(pos)
        for dx in (0, 1):
            wx = w1[:, 0] if dx else w0[:, 0]
            for dy in (0, 1):
                wy = w1[:, 1] if dy else w0[:, 1]
                for dz in (0, 1):
                    wz = w1[:, 2] if dz else w0[:, 2]
                    out += (wx * wy * wz)[:, None] * self.E[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        return out
