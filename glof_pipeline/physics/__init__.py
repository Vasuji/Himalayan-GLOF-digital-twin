"""Reference physics: mass balance, moraine geotechnics, breach, shallow water.

These are the models the neural surrogates are trained against and benchmarked
on. They are deliberately conventional -- degree-day melt, effective-stress
limit equilibrium, Froehlich breach geometry, and a well-balanced HLL
finite-volume shallow-water scheme -- so that a reviewer can check every step
against the standard literature.
"""

from .breach import BreachResult, froehlich_geometry, froehlich_peak_discharge, simulate_breach
from .mass_balance import CatchmentForcing, degree_day_melt, integrate_catchment
from .moraine_stability import StabilityField, evaluate_stability, infinite_slope_fos
from .swe_solver import ShallowWaterSolver, SWEResult, ritter_dam_break

__all__ = [
    "BreachResult",
    "CatchmentForcing",
    "SWEResult",
    "ShallowWaterSolver",
    "StabilityField",
    "degree_day_melt",
    "evaluate_stability",
    "froehlich_geometry",
    "froehlich_peak_discharge",
    "infinite_slope_fos",
    "integrate_catchment",
    "ritter_dam_break",
    "simulate_breach",
]
