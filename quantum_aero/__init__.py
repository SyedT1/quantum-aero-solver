"""Validated Taylor-Green vortex experiments for the Airbus challenge."""

from .classical import LBMConfig, run_lbm, tgv_exact

__all__ = ["LBMConfig", "run_lbm", "tgv_exact"]
