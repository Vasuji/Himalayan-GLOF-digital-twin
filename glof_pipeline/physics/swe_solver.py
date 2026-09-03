r"""Two-dimensional shallow-water solver: the reference model and the FNO's teacher.

Scheme
------
First-order Godunov finite volume on the raster grid with

* an HLL approximate Riemann solver including the dry-bed wave speeds of Toro,
* hydrostatic reconstruction (Audusse et al., *SIAM J. Sci. Comput.* 25, 2004) for
  the bed-slope source term, which makes the scheme well balanced -- a lake at
  rest over arbitrary topography stays exactly at rest,
* wet/dry front handling with a depth threshold, and
* a semi-implicit Manning friction update, which is unconditionally stable as the
  depth goes to zero.

Governing equations, with :math:`h` depth, :math:`\mathbf{u}` depth-averaged
velocity, :math:`z` bed elevation:

.. math::

    \partial_t h + \nabla\cdot(h\mathbf{u}) = q, \qquad
    \partial_t (h\mathbf{u}) + \nabla\cdot(h\mathbf{u}\otimes\mathbf{u})
      + g h \nabla (h + z) = -\mathbf{\tau}_f .

Arrays are ``(ny, nx)`` with row index increasing downvalley.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SWEResult:
    """Time series of the routed flood wave."""

    time_s: np.ndarray            # (nt,)
    depth: np.ndarray             # (nt, ny, nx)
    momentum_x: np.ndarray        # (nt, ny, nx)
    momentum_y: np.ndarray        # (nt, ny, nx)
    volume_m3: np.ndarray         # (nt,) water volume in the domain
    injected_m3: np.ndarray       # (nt,) cumulative volume added by the source
    outflow_m3: np.ndarray        # (nt,) cumulative volume lost at open boundaries
    steps: int
    wall_time_s: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def max_depth(self) -> np.ndarray:
        """Envelope of maximum depth over the simulation."""
        return self.depth.max(axis=0)

    def mass_conservation_error(self) -> float:
        """Relative closure error of the domain water budget."""
        expected = self.volume_m3[0] + self.injected_m3 - self.outflow_m3
        scale = max(float(np.max(np.abs(expected))), 1.0)
        return float(np.max(np.abs(self.volume_m3 - expected)) / scale)

    def arrival_times(self, rows: list[tuple[str, int]], threshold_m: float) -> dict[str, float]:
        """First time [s] each receptor row exceeds ``threshold_m`` anywhere across it."""
        out: dict[str, float] = {}
        for name, row in rows:
            exceeded = self.depth[:, row, :].max(axis=1) >= threshold_m
            index = int(np.argmax(exceeded)) if exceeded.any() else -1
            out[name] = float(self.time_s[index]) if index >= 0 else float("nan")
        return out


def ritter_dam_break(
    x: np.ndarray, t: float, upstream_depth: float, gravity: float = 9.81
) -> np.ndarray:
    """Ritter's analytic dam-break solution on a flat, frictionless, dry bed.

    The dam sits at ``x = 0``, water of depth ``h0`` occupies ``x < 0``. Used to
    verify the solver's shock/rarefaction structure and wet/dry front speed.
    """
    x = np.asarray(x, dtype=float)
    if t <= 0.0:
        return np.where(x < 0.0, upstream_depth, 0.0)
    celerity = np.sqrt(gravity * upstream_depth)
    depth = np.zeros_like(x)
    depth[x <= -celerity * t] = upstream_depth
    middle = (x > -celerity * t) & (x < 2.0 * celerity * t)
    depth[middle] = (2.0 * celerity - x[middle] / t) ** 2 / (9.0 * gravity)
    return depth


class ShallowWaterSolver:
    """Well-balanced HLL finite-volume solver on a fixed raster bed."""

    def __init__(
        self,
        bed_elevation: np.ndarray,
        dx: float,
        gravity: float = 9.81,
        manning_n: float = 0.045,
        dry_depth: float = 1.0e-3,
        cfl: float = 0.45,
        boundary: str = "outflow",
    ):
        if bed_elevation.ndim != 2:
            raise ValueError("bed_elevation must be a 2-D array.")
        if boundary not in ("outflow", "wall"):
            raise ValueError("boundary must be 'outflow' or 'wall'.")
        self.z = np.asarray(bed_elevation, dtype=np.float64)
        self.dx = float(dx)
        self.g = float(gravity)
        self.n = float(manning_n)
        self.dry = float(dry_depth)
        self.cfl = float(cfl)
        self.boundary = boundary
        self.cell_area = self.dx * self.dx

    # -- helpers ------------------------------------------------------------
    def _velocity(self, h: np.ndarray, hu: np.ndarray) -> np.ndarray:
        return np.where(h > self.dry, hu / np.where(h > self.dry, h, 1.0), 0.0)

    def _pad(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray):
        """One ghost cell per side, filled according to the boundary condition."""
        hp = np.pad(h, 1, mode="edge")
        hup = np.pad(hu, 1, mode="edge")
        hvp = np.pad(hv, 1, mode="edge")
        zp = np.pad(self.z, 1, mode="edge")
        if self.boundary == "wall":
            hup[:, 0] = -hup[:, 1]
            hup[:, -1] = -hup[:, -2]
            hvp[0, :] = -hvp[1, :]
            hvp[-1, :] = -hvp[-2, :]
        return hp, hup, hvp, zp

    def _hll(self, hL, huL, hvL, hR, huR, hvR, axis: str):
        """HLL flux for the interface states, with dry-bed wave speeds."""
        g = self.g
        uL = self._velocity(hL, huL)
        uR = self._velocity(hR, huR)
        vL = self._velocity(hL, hvL)
        vR = self._velocity(hR, hvR)
        if axis == "y":  # normal velocity is v
            uL, vL, uR, vR = vL, uL, vR, uR
            huL, hvL, huR, hvR = hvL, huL, hvR, huR

        cL = np.sqrt(g * np.clip(hL, 0.0, None))
        cR = np.sqrt(g * np.clip(hR, 0.0, None))
        wetL = hL > self.dry
        wetR = hR > self.dry

        sL = np.where(wetR, uR - 2.0 * cR, uL - cL)
        sL = np.where(wetL, np.minimum(uL - cL, uR - cR), sL)
        sR = np.where(wetL, uL + 2.0 * cL, uR + cR)
        sR = np.where(wetR, np.maximum(uL + cL, uR + cR), sR)

        # Physical fluxes in the interface-normal direction.
        FL = np.stack([huL, huL * uL + 0.5 * g * hL**2, huL * vL])
        FR = np.stack([huR, huR * uR + 0.5 * g * hR**2, huR * vR])
        UL = np.stack([hL, huL, hvL])
        UR = np.stack([hR, huR, hvR])

        denom = np.where(np.abs(sR - sL) > 1e-12, sR - sL, 1.0)
        F_star = (sR * FL - sL * FR + sL * sR * (UR - UL)) / denom
        flux = np.where(sL >= 0.0, FL, np.where(sR <= 0.0, FR, F_star))
        dryboth = (~wetL) & (~wetR)
        flux = np.where(dryboth, 0.0, flux)
        return flux, np.maximum(np.abs(sL), np.abs(sR))

    def time_step(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray) -> float:
        """CFL-limited step for the unsplit two-dimensional update."""
        wet = h > self.dry
        if not wet.any():
            return float("inf")
        c = np.sqrt(self.g * np.where(wet, h, 0.0))
        u = np.abs(self._velocity(h, hu))
        v = np.abs(self._velocity(h, hv))
        rate = (u + c) / self.dx + (v + c) / self.dx
        peak = float(np.max(rate))
        return self.cfl / peak if peak > 0.0 else float("inf")

    def step(
        self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Advance one step; returns the new state and the volume lost at boundaries."""
        g, dx = self.g, self.dx
        hp, hup, hvp, zp = self._pad(h, hu, hv)
        surface = hp + zp

        # --- x-direction interfaces (between padded columns j and j+1) ------
        zL, zR = zp[:, :-1], zp[:, 1:]
        z_star = np.maximum(zL, zR)
        hL_star = np.clip(surface[:, :-1] - z_star, 0.0, None)
        hR_star = np.clip(surface[:, 1:] - z_star, 0.0, None)
        uL = self._velocity(hp[:, :-1], hup[:, :-1])
        uR = self._velocity(hp[:, 1:], hup[:, 1:])
        vL = self._velocity(hp[:, :-1], hvp[:, :-1])
        vR = self._velocity(hp[:, 1:], hvp[:, 1:])
        Fx, _ = self._hll(
            hL_star, hL_star * uL, hL_star * vL, hR_star, hR_star * uR, hR_star * vR, axis="x"
        )
        # Hydrostatic-reconstruction corrections, evaluated with each cell's own depth.
        corr_left = 0.5 * g * (hp[:, :-1] ** 2 - hL_star**2)   # for the cell left of the interface
        corr_right = 0.5 * g * (hp[:, 1:] ** 2 - hR_star**2)   # for the cell right of the interface
        Fx_right_of_cell = Fx[:, 1:-1, 1:].copy()
        Fx_left_of_cell = Fx[:, 1:-1, :-1].copy()
        Fx_right_of_cell[1] += corr_left[1:-1, 1:]
        Fx_left_of_cell[1] += corr_right[1:-1, :-1]

        # --- y-direction interfaces (between padded rows i and i+1) ---------
        zB, zT = zp[:-1, :], zp[1:, :]
        z_star_y = np.maximum(zB, zT)
        hB_star = np.clip(surface[:-1, :] - z_star_y, 0.0, None)
        hT_star = np.clip(surface[1:, :] - z_star_y, 0.0, None)
        uB = self._velocity(hp[:-1, :], hup[:-1, :])
        uT = self._velocity(hp[1:, :], hup[1:, :])
        vB = self._velocity(hp[:-1, :], hvp[:-1, :])
        vT = self._velocity(hp[1:, :], hvp[1:, :])
        Fy, _ = self._hll(
            hB_star, hB_star * uB, hB_star * vB, hT_star, hT_star * uT, hT_star * vT, axis="y"
        )
        # _hll returns [mass, normal momentum, tangential momentum]; for the y sweep
        # the normal component is v, so reorder into (mass, hu, hv).
        Fy = np.stack([Fy[0], Fy[2], Fy[1]])
        corr_bottom = 0.5 * g * (hp[:-1, :] ** 2 - hB_star**2)
        corr_top = 0.5 * g * (hp[1:, :] ** 2 - hT_star**2)
        Fy_top_of_cell = Fy[:, 1:, 1:-1].copy()
        Fy_bottom_of_cell = Fy[:, :-1, 1:-1].copy()
        Fy_top_of_cell[2] += corr_bottom[1:, 1:-1]
        Fy_bottom_of_cell[2] += corr_top[:-1, 1:-1]

        factor = dt / dx
        h_new = h - factor * (Fx_right_of_cell[0] - Fx_left_of_cell[0]) \
                  - factor * (Fy_top_of_cell[0] - Fy_bottom_of_cell[0])
        hu_new = hu - factor * (Fx_right_of_cell[1] - Fx_left_of_cell[1]) \
                   - factor * (Fy_top_of_cell[1] - Fy_bottom_of_cell[1])
        hv_new = hv - factor * (Fx_right_of_cell[2] - Fx_left_of_cell[2]) \
                   - factor * (Fy_top_of_cell[2] - Fy_bottom_of_cell[2])

        # Boundary volume accounting (mass flux leaving the domain edges).
        outflow = 0.0
        if self.boundary == "outflow":
            edge_flux = (
                -Fx[0, 1:-1, 0].sum()
                + Fx[0, 1:-1, -1].sum()
                - Fy[0, 0, 1:-1].sum()
                + Fy[0, -1, 1:-1].sum()
            )
            outflow = float(edge_flux * dt * dx)

        # Clip negative depths created at wet/dry fronts and drop their momentum.
        h_new = np.clip(h_new, 0.0, None)
        dry = h_new <= self.dry
        hu_new = np.where(dry, 0.0, hu_new)
        hv_new = np.where(dry, 0.0, hv_new)

        # Semi-implicit Manning friction.
        if self.n > 0.0:
            depth = np.where(dry, 1.0, h_new)
            u = hu_new / depth
            v = hv_new / depth
            speed = np.hypot(u, v)
            damping = 1.0 + dt * self.g * self.n**2 * speed / depth ** (4.0 / 3.0)
            hu_new = np.where(dry, 0.0, hu_new / damping)
            hv_new = np.where(dry, 0.0, hv_new / damping)

        return h_new, hu_new, hv_new, outflow

    # -- driver -------------------------------------------------------------
    def run(
        self,
        depth0: np.ndarray,
        duration_s: float,
        output_interval_s: float,
        momentum_x0: np.ndarray | None = None,
        momentum_y0: np.ndarray | None = None,
        source: Callable[[float, float], np.ndarray] | None = None,
        max_steps: int = 400_000,
        progress: Callable[[float], None] | None = None,
    ) -> SWEResult:
        """Integrate to ``duration_s``, storing a frame every ``output_interval_s``.

        ``source(t, dt)`` returns a depth increment [m] to add over the step, which
        is how the breach hydrograph is injected at the dam.
        """
        import time as _time

        h = np.array(depth0, dtype=np.float64, copy=True)
        hu = np.zeros_like(h) if momentum_x0 is None else np.array(momentum_x0, float, copy=True)
        hv = np.zeros_like(h) if momentum_y0 is None else np.array(momentum_y0, float, copy=True)

        n_frames = int(round(duration_s / output_interval_s)) + 1
        times = np.arange(n_frames) * output_interval_s
        depth_out = np.zeros((n_frames, *h.shape))
        hu_out = np.zeros_like(depth_out)
        hv_out = np.zeros_like(depth_out)
        volume = np.zeros(n_frames)
        injected = np.zeros(n_frames)
        outflow = np.zeros(n_frames)

        depth_out[0], hu_out[0], hv_out[0] = h, hu, hv
        volume[0] = h.sum() * self.cell_area

        t = 0.0
        steps = 0
        frame = 1
        cumulative_injected = 0.0
        cumulative_outflow = 0.0
        start = _time.perf_counter()

        while frame < n_frames and steps < max_steps:
            target = times[frame]
            while t < target - 1e-12 and steps < max_steps:
                dt = self.time_step(h, hu, hv)
                if not np.isfinite(dt):
                    dt = target - t
                dt = min(dt, target - t)
                if dt <= 0.0:
                    break
                if source is not None:
                    increment = source(t, dt)
                    h = h + increment
                    cumulative_injected += float(increment.sum() * self.cell_area)
                h, hu, hv, lost = self.step(h, hu, hv, dt)
                cumulative_outflow += lost
                t += dt
                steps += 1
            depth_out[frame], hu_out[frame], hv_out[frame] = h, hu, hv
            volume[frame] = h.sum() * self.cell_area
            injected[frame] = cumulative_injected
            outflow[frame] = cumulative_outflow
            if progress is not None:
                progress(t / max(duration_s, 1e-9))
            frame += 1

        wall_time = _time.perf_counter() - start
        return SWEResult(
            time_s=times[:frame],
            depth=depth_out[:frame],
            momentum_x=hu_out[:frame],
            momentum_y=hv_out[:frame],
            volume_m3=volume[:frame],
            injected_m3=injected[:frame],
            outflow_m3=outflow[:frame],
            steps=steps,
            wall_time_s=wall_time,
            meta={
                "dx": self.dx,
                "manning_n": self.n,
                "cfl": self.cfl,
                "boundary": self.boundary,
                "dry_depth": self.dry,
            },
        )


def hydrograph_source(
    shape: tuple[int, int],
    cells: np.ndarray,
    time_s: np.ndarray,
    discharge_m3_per_s: np.ndarray,
    cell_area: float,
) -> Callable[[float, float], np.ndarray]:
    """Build a source callable that injects a hydrograph over ``cells``.

    ``cells`` is an ``(n, 2)`` array of row/col indices, typically the breach
    opening in the moraine crest.
    """
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError("cells must be an (n, 2) array of row/col indices.")
    n_cells = cells.shape[0]
    rows, cols = cells[:, 0], cells[:, 1]

    def source(t: float, dt: float) -> np.ndarray:
        increment = np.zeros(shape)
        q = float(np.interp(t, time_s, discharge_m3_per_s, left=0.0, right=0.0))
        if q > 0.0:
            increment[rows, cols] = q * dt / (n_cells * cell_area)
        return increment

    return source
