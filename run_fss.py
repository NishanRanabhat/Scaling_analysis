#!/usr/bin/env python
"""
Unified FSS data collapse runner.

Usage: edit the parameters below, uncomment one experiment block, and run:
    python run_fss.py
"""
import numpy as np
from experiment import ExperimentRunner

if __name__ == "__main__":

    # ============================================================
    #  DATASET SELECTION
    # ============================================================
    # Known initial params (from previous converged results):
    #   dt=0.001, a=0.80, h=0.00: (0.4748, 0.053, 0.405)
    #   dt=0.001, a=0.80, h=0.30: (0.4776, 0.053, 0.404)
    #   dt=0.002, a=1.80, h=0.00: (0.7357, 0.090, 0.427)
    #   dt=0.002, a=1.80, h=0.30: (0.8007, 0.111, 0.425)

    # --- Fixed nu=2 (b=0.5), optimize only beta_c and a=kappa/nu ---
    runner = ExperimentRunner(
        dt="0.001", a="0.80", h="0.00",
        initial_params=(0.4748, 0.053, 0.50),
        sizes=[250, 300, 350, 400],
        poly_order=10,
        fixed_params={2: 0.5},  # fix b=1/nu=0.5, i.e. nu=2
    )

    # ============================================================
    #  EXPERIMENT 1: Single run
    # ============================================================
    runner.single_run(w=0.8)

    # ============================================================
    #  EXPERIMENT 2: Window scan
    # ============================================================
    runner.window_scan(np.arange(0.3, 2.1, 0.1), chain_initial=False)

    # ============================================================
    #  EXPERIMENT 3: Polynomial order scan
    # ============================================================
    #runner.poly_order_scan(range(6, 16), w=0.6)

    # ============================================================
    #  EXPERIMENT 4: Jackknife error estimation
    # ============================================================
    #runner.jackknife(w=0.6)

    # ============================================================
    #  EXPERIMENT 5: Window scan for each poly_order
    # ============================================================
    #runner.grid_scan(
    #    "poly_order", [8, 10, 12],
    #    "window_scan", w_values=np.arange(0.3, 2.0, 0.1)
    #)

    # ============================================================
    #  EXPERIMENT 6: Custom loop (e.g., jackknife at multiple windows)
    # ============================================================
    #for w in [0.6, 0.8, 1.0]:
    #    print(f"\n--- w = {w} ---")
    #    runner.jackknife(w=w)
