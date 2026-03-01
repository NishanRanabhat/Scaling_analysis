# Finite-Size Scaling (FSS) Data Collapse — User Guide

## Table of Contents

1. [Overview](#1-overview)
2. [Directory Structure](#2-directory-structure)
3. [Physics Background](#3-physics-background)
4. [Data Files](#4-data-files)
5. [The FSS Library](#5-the-fss-library)
6. [The Experiment Layer](#6-the-experiment-layer)
7. [Quick Start](#7-quick-start)
8. [Experiment Methods Reference](#8-experiment-methods-reference)
9. [Tunable Parameters and Their Effects](#9-tunable-parameters-and-their-effects)
10. [Recipes for Common Tasks](#10-recipes-for-common-tasks)
11. [Adding New Data](#11-adding-new-data)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Overview

This codebase performs **finite-size scaling (FSS) data collapse** to extract
critical exponents and the critical point from numerical data near a quantum
phase transition. The pipeline works in three layers:

```
run_fss.py          ← you edit this (choose dataset, choose experiment)
    │
experiment.py       ← loads data, runs experiments (window scan, jackknife, etc.)
    │
FSS/                ← general-purpose FSS optimization library
```

You interact primarily with `run_fss.py`. The other files rarely need editing.

---

## 2. Directory Structure

```
UPDATED_DATA/
├── run_fss.py              Entry point — configure and run experiments here
├── experiment.py           Data loading + ExperimentRunner class
├── data_vs_beta_*.dat      Input data files (20 files)
├── DATA_COLLAPSE.ipynb     Jupyter notebook for visualization
├── *.pdf                   Saved collapse plots
└── FSS/                    General-purpose FSS library (do not modify)
    ├── data_set.py         DataSet container class
    ├── finitesizescaling.py    FSS optimizer (core algorithm)
    └── utilities.py        Rescaling helper functions
```

---

## 3. Physics Background

### The scaling ansatz

Near a critical point beta_c, an observable O(beta, L) for a system of size L
obeys the finite-size scaling form:

```
O(beta, L) * L^(kappa/nu) = F( (beta - beta_c) * L^(1/nu) )
```

where F is a **universal scaling function** and:

- **beta_c** — critical point (inverse temperature)
- **nu** — correlation length exponent
- **kappa** — observable scaling exponent

If the correct exponents are chosen, data from **all system sizes collapses
onto a single curve** F(X).

### Optimization parameters

The optimizer works with three parameters `(beta_c, a, b)` where:

- `a = kappa / nu` — rescales the Y-axis: `Y = O * L^a`
- `b = 1 / nu` — rescales the X-axis: `X = (beta - beta_c) * L^b`

The physical exponents are derived as:
- `nu = 1 / b`
- `kappa = a * nu = a / b`

### The algorithm

1. For trial parameters (beta_c, a, b), rescale all data:
   - X = (beta - beta_c) * L^b
   - Y = observable * L^a
2. Combine rescaled (X, Y) pairs from all system sizes
3. Fit a polynomial of degree `poly_order` to the combined data
4. The **sum of squared residuals** measures collapse quality
5. Minimize the residuals over (beta_c, a, b) using scipy

Good parameters → data collapses → small residuals.
Bad parameters → data scattered → large residuals.

---

## 4. Data Files

### Naming convention

All data files follow this pattern:

```
data_vs_beta_dt={dt}_N={N}_chi_max=256_a={a}_h={h}_.dat
```

### Available datasets

| dt    | a    | h    | N (system sizes)        | Rows | Columns |
|-------|------|------|-------------------------|------|---------|
| 0.001 | 0.80 | 0.00 | 200, 250, 300, 350, 400 | 1000 | 3       |
| 0.001 | 0.80 | 0.30 | 200, 250, 300, 350, 400 | 1000 | 3       |
| 0.002 | 1.80 | 0.00 | 200, 250, 300, 350, 400 | 540  | 4       |
| 0.002 | 1.80 | 0.30 | 200, 250, 300, 350, 400 | 540  | 4       |

### Column layout

- Column 0: control parameter (beta or T)
- Column 1: primary observable (Schmidt gap) ← **default**
- Columns 2+: additional observables (if present)

### Known results (from previous analysis)

| dt    | a    | h    | beta_c | nu   | kappa |
|-------|------|------|--------|------|-------|
| 0.001 | 0.80 | 0.00 | 0.4748 | 2.47 | 0.130 |
| 0.001 | 0.80 | 0.30 | 0.4776 | 2.47 | 0.131 |
| 0.002 | 1.80 | 0.00 | 0.7357 | 2.34 | 0.210 |
| 0.002 | 1.80 | 0.30 | 0.8009 | 2.35 | 0.261 |

---

## 5. The FSS Library

The `FSS/` directory is a **general-purpose, reusable** finite-size scaling
library. It has no knowledge of your specific data files or naming conventions.

### FSS/data_set.py — DataSet

A simple container holding three arrays:

```python
class DataSet:
    system_size_list   # 1D array of L values, e.g. [200, 250, 300, 350, 400]
    domain_list        # 1D array of beta values (shared), or list of per-size arrays
    range_list         # 2D array of shape (num_sizes, num_beta_points)
```

### FSS/finitesizescaling.py — FSS

The core optimizer class:

```python
fss = FSS(
    dataset,                          # DataSet object
    poly_order=10,                    # degree of polynomial fit
    initial_params=(beta_c, a, b),    # starting guess
    scaling_window=(-0.8, 0.8),       # window in rescaled X
    optimization_routine="Nelder-Mead",
    min_points_per_size=60,           # minimum data points per L
    param_bounds=None,                # required for L-BFGS-B / dual-annealing
)

params, residual = fss.optimization()
# params = array([beta_c, a, b])
# residual = sum of squared polynomial fit residuals
```

**Key internal methods:**

- `rescaled_combined_data(params)` — applies the scaling transformation to all
  sizes, applies the window mask, and returns combined (X, Y) arrays
- `objective_function(params)` — calls `rescaled_combined_data`, fits a
  polynomial, returns residuals. Returns `np.inf` if the fit is degenerate
  (too few points).
- `optimization()` — calls `scipy.optimize.minimize` to find the best
  (beta_c, a, b)

**Scaling window and minimum points:**

The `_mask_scaled_window_with_min()` helper ensures that:
1. Only data points with rescaled X inside the window are used
2. If the window is too narrow (fewer than `min_points_per_size` points for
   a given L), the window is automatically expanded to include the nearest
   points around X=0

**Supported optimizers:**

| Routine          | Needs bounds? | Notes                              |
|------------------|---------------|------------------------------------|
| `Nelder-Mead`    | No            | Default. Derivative-free simplex.  |
| `L-BFGS-B`       | Yes           | Gradient-based, fast but local.    |
| `dual-annealing`  | Yes           | Global optimizer, slower.          |

### FSS/utilities.py — Helper functions

- `Y_rescaled(y, L, a)` → `y * L^a`
- `X_rescaled(x, L, xc, b)` → `(x - xc) * L^b`
- `slice_limits(data, lo, hi)` → boolean mask for `[lo, hi]`
- `closest_index(data, val)` → index of nearest element

---

## 6. The Experiment Layer

`experiment.py` sits between your run script and the FSS library. It handles
data loading and provides composable experiment methods.

### load_dataset()

```python
dataset = load_dataset(
    dt="0.002",     # timestep
    a="1.80",       # coupling parameter
    h="0.30",       # field strength
    sizes=[200, 250, 300, 350, 400],  # which system sizes to load
    obs_col=1,      # which column is the observable (default: 1)
)
```

- Constructs filenames from the parameters
- Reads data files with `np.loadtxt`
- Returns a `DataSet` object ready for the FSS class
- Uses `DATASET_REGISTRY` to handle per-dataset differences

### DATASET_REGISTRY

A dictionary at the top of `experiment.py` that stores metadata for each
known (dt, a, h) combination:

```python
DATASET_REGISTRY = {
    ("0.001", "0.80", "0.00"): {"obs_col": 1, "per_size_beta": False},
    ("0.001", "0.80", "0.30"): {"obs_col": 1, "per_size_beta": False},
    ("0.002", "1.80", "0.00"): {"obs_col": 1, "per_size_beta": False},
    ("0.002", "1.80", "0.30"): {"obs_col": 1, "per_size_beta": False},
}
```

When you add new data, add a new entry here (see Section 11).

### ExperimentRunner

```python
runner = ExperimentRunner(
    dt="0.002", a="1.80", h="0.30",    # dataset selection
    initial_params=(0.80, 0.11, 0.43),  # (beta_c, a=kappa/nu, b=1/nu)
    sizes=[200, 250, 300, 350, 400],    # system sizes
    poly_order=10,                      # polynomial degree
    scaling_window=(-0.8, 0.8),         # default window
    optimization_routine="Nelder-Mead", # optimizer
    min_points_per_size=60,             # min points guarantee
    obs_col=None,                       # None = use registry default
    param_bounds=None,                  # needed for L-BFGS-B
)
```

Constructor stores defaults. Every experiment method accepts keyword overrides
for any of these parameters, so you set common values once and vary only
what changes per experiment.

---

## 7. Quick Start

Edit `run_fss.py`:

```python
from experiment import ExperimentRunner
import numpy as np

runner = ExperimentRunner(
    dt="0.001", a="0.80", h="0.00",
    initial_params=(0.48, 0.06, 0.50),
    sizes=[250, 300, 350, 400],   # drop N=200
)

# Pick ONE experiment (uncomment):
runner.single_run(w=0.8)
```

Run:

```bash
python run_fss.py
```

Output:

```
beta_c  = 0.474800
nu      = 2.4700  (b = 1/nu = 0.404858)
kappa   = 0.1300  (a = kappa/nu = 0.052632)
residual = 1.234567e-04
```

---

## 8. Experiment Methods Reference

### single_run(**kwargs)

One FSS optimization. Prints beta_c, nu, kappa and the residual.

```python
result = runner.single_run(w=0.8)
# result = {"beta_c": ..., "nu": ..., "kappa": ..., "a": ..., "b": ..., "residual": ...}
```

Any keyword argument overrides the runner's defaults for this call:
```python
runner.single_run(w=1.0, poly_order=8, sizes=[300, 350, 400])
```

---

### window_scan(w_values, chain_initial=True, **kwargs)

Scan over scaling window half-widths. Use this to look for a **plateau** in
the exponents — the region of w where exponents are stable is the most
trustworthy.

```python
results = runner.window_scan(np.arange(0.3, 2.1, 0.1))
```

Output:

```
    w      beta_c        nu     kappa           b           a      residual
------------------------------------------------------------------------
 0.30    0.478452    2.4137    0.1584    0.414319    0.065652  8.098e-05
 0.40    0.479030    2.3679    0.1578    0.422314    0.066611  1.466e-04
 ...
```

**chain_initial**: when `True` (default), each w's result is used as the
initial guess for the next w. This helps the optimizer track a smooth path
across the landscape and is recommended for scans.

Set `chain_initial=False` to use the same initial guess for every w (useful
for checking sensitivity to initial conditions).

---

### poly_order_scan(orders, **kwargs)

Scan over the polynomial fitting degree. Helps assess whether your results
depend on the polynomial order.

```python
results = runner.poly_order_scan(range(6, 16), w=0.8)
```

If exponents are stable across poly_order = 8–14, the result is robust.
If they vary significantly, the scaling function may not be well-approximated
by a polynomial in the current window.

---

### jackknife(**kwargs)

Leave-one-out jackknife over system sizes. Drops each size in turn,
re-optimizes, and computes standard errors.

```python
results = runner.jackknife(w=0.8)
```

Output:

```
Full dataset: beta_c=0.800720, nu=2.3512, kappa=0.2609

Jackknife (dropping one size at a time):
 dropped      beta_c        nu     kappa
------------------------------------------
  N= 200    0.800533    2.3456    0.2598
  N= 250    0.800891    2.3578    0.2621
  ...

Results:
  beta_c = 0.800720 +/- 0.000200
  nu     = 2.3512 +/- 0.0300
  kappa  = 0.2609 +/- 0.0030
```

The error formula is the delete-1 jackknife:
`se = sqrt( sum_i (x_i - x_mean)^2 / n )`

---

### grid_scan(scan_over, values, inner_method, **inner_kwargs)

Compose experiments: run any method for each value of a scanned parameter.

**Example: window scan at each poly_order**

```python
runner.grid_scan(
    "poly_order", [8, 10, 12],
    "window_scan", w_values=np.arange(0.3, 2.0, 0.1)
)
```

This runs three full window scans (one per poly_order) and prints results
for each.

**Example: jackknife at each window**

```python
runner.grid_scan(
    "w", [0.6, 0.8, 1.0],
    "jackknife"
)
```

---

## 9. Tunable Parameters and Their Effects

### scaling_window (w)

Controls how much data around the critical point is used.

| w too small (< 0.3) | w too large (> 2.0) |
|----------------------|---------------------|
| Too few data points  | Includes non-scaling data |
| Polynomial overfits  | Corrections to scaling contaminate |
| Noisy exponents      | Exponents drift |

**Best practice:** Run `window_scan()` and look for a plateau in the
exponents. The plateau region is the trustworthy window range.

### poly_order

Degree of the polynomial approximating the universal scaling function F(X).

| Too low (< 6) | Too high (> 14) |
|----------------|-----------------|
| Underfits F(X) | Overfits noise  |
| Large residuals | Artificially small residuals |

**Default: 10.** Run `poly_order_scan()` to verify stability.

### initial_params

Starting guess for (beta_c, a, b). Nelder-Mead is a local optimizer, so
the initial guess matters.

**Tips:**
- Use the known results table (Section 4) as starting points
- If the optimizer converges to unphysical values, try a different initial guess
- The `chain_initial` feature in `window_scan` mitigates this by seeding
  from the previous result

### sizes / system sizes to drop

Dropping the smallest system sizes reduces **corrections to scaling** — the
finite-size effects that cause effective exponents to deviate from their
true asymptotic values.

**Trade-off:** Fewer sizes = less correction-to-scaling contamination, but
also less data to constrain the fit. With only 2 sizes, the fit is
underconstrained.

**Recommended:** Start with all 5 sizes, then try dropping N=200, then
dropping N=200 and N=250. If exponents shift systematically toward expected
values, corrections to scaling are significant.

### min_points_per_size

Minimum number of data points guaranteed per system size, even if the
scaling window would normally include fewer. Default: 60.

If the window is too narrow, the code automatically expands to include the
nearest `min_points_per_size` points around X=0. This prevents degenerate
fits with too few points.

### optimization_routine

| Routine | When to use |
|---------|-------------|
| `Nelder-Mead` (default) | Most cases. No derivatives needed. |
| `L-BFGS-B` | When you have good bounds and want speed. Requires `param_bounds`. |
| `dual-annealing` | When you suspect multiple local minima. Slow but global. Requires `param_bounds`. |

---

## 10. Recipes for Common Tasks

### Switch to a different dataset

Change the constructor parameters:

```python
runner = ExperimentRunner(
    dt="0.001", a="0.80", h="0.00",         # ← change these
    initial_params=(0.48, 0.05, 0.40),       # ← use appropriate guess
    sizes=[200, 250, 300, 350, 400],
)
```

### Drop system sizes to reduce corrections to scaling

```python
runner = ExperimentRunner(
    ...,
    sizes=[250, 300, 350, 400],   # dropped N=200
)
```

Or override per-call:
```python
runner.single_run(w=0.8, sizes=[300, 350, 400])
```

### Use a different observable column

```python
runner = ExperimentRunner(..., obs_col=2)
```

### Use a bounded optimizer

```python
runner = ExperimentRunner(
    ...,
    optimization_routine="L-BFGS-B",
    param_bounds=[(0.45, 0.50), (0.04, 0.08), (0.35, 0.55)],
)
```

### Full analysis pipeline for a dataset

```python
runner = ExperimentRunner(
    dt="0.001", a="0.80", h="0.00",
    initial_params=(0.48, 0.05, 0.40),
)

# Step 1: Quick single run to verify setup
runner.single_run(w=0.8)

# Step 2: Window scan to find stable region
runner.window_scan(np.arange(0.3, 2.1, 0.1))

# Step 3: Poly order scan at best window
runner.poly_order_scan(range(6, 16), w=1.0)

# Step 4: Jackknife for error bars at chosen window
runner.jackknife(w=1.0)
```

### Compare all four datasets

```python
for dt, a_val, h in [("0.001", "0.80", "0.00"),
                      ("0.001", "0.80", "0.30"),
                      ("0.002", "1.80", "0.00"),
                      ("0.002", "1.80", "0.30")]:
    print(f"\n=== dt={dt}, a={a_val}, h={h} ===")
    runner = ExperimentRunner(
        dt=dt, a=a_val, h=h,
        initial_params=(0.50, 0.08, 0.43),  # rough guess
    )
    runner.single_run(w=0.8)
```

---

## 11. Adding New Data

When you generate data for a new (dt, a, h) combination:

**Step 1:** Name your files following the convention:
```
data_vs_beta_dt={dt}_N={N}_chi_max=256_a={a}_h={h}_.dat
```

**Step 2:** Add an entry to `DATASET_REGISTRY` in `experiment.py`:
```python
DATASET_REGISTRY = {
    ...
    ("0.003", "2.00", "0.50"): {"obs_col": 1, "per_size_beta": False},
}
```

Set `per_size_beta: True` if different system sizes have different beta
grids (different values in column 0). Otherwise keep `False`.

**Step 3:** Use it:
```python
runner = ExperimentRunner(
    dt="0.003", a="2.00", h="0.50",
    initial_params=(0.50, 0.08, 0.40),
)
```

If your system sizes are different from [200, 250, 300, 350, 400],
pass them explicitly:
```python
runner = ExperimentRunner(..., sizes=[100, 200, 300, 400, 500])
```

---

## 12. Troubleshooting

### Optimizer returns unphysical values

- Try a different `initial_params` closer to the expected result
- Try `dual-annealing` with bounds to find the global minimum
- Reduce the window to focus near the critical region

### Residuals are zero or very small

If `objective_function` returns 0 or near-0, the polynomial may be
overfitting (too few data points relative to poly_order). The code returns
`np.inf` when residuals are empty (degenerate fit), but near-degenerate
cases can still slip through. Try reducing `poly_order` or widening the
window.

### No plateau in window scan

This indicates:
- System sizes may not be large enough for asymptotic scaling
- Corrections to scaling are significant → try dropping small sizes
- The polynomial order may be too high (absorbing non-universal features)
- The exponents a and b are strongly correlated

### ValueError: Unknown dataset

The (dt, a, h) combination is not in `DATASET_REGISTRY`. Add it
(see Section 11).

### Data shape mismatch

If you get "Length mismatch for L=..." errors, check that all data files
for a given (dt, a, h) have the same number of rows and the same beta
grid in column 0. If beta grids differ per size, set `per_size_beta: True`
in the registry.
