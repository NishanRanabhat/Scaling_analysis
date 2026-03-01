# Roadmap: Finite-Size Scaling Collapse Toolkit

This document outlines the vision, planned features, and development pipeline for turning the current FSS analysis codebase into a general-purpose, user-friendly scaling collapse software.

---

## Vision

A modern, pip-installable Python library for finite-size scaling data collapse that is:

- **General** — works with any observable, any control parameter, any scaling form
- **Modular** — users can swap fitting backends and define custom scaling ansätze
- **Accessible** — clean API, good documentation, and example notebooks
- **Robust** — multiple optimizers, error estimation, and quality-of-collapse metrics

### Existing landscape

| Tool | Limitation |
|------|-----------|
| `pyfssa` | Poorly maintained, limited fitting options |
| `autoScale` (Houdayer & Hartmann) | Fortran/C-based, hard to integrate into Python workflows |

There is a clear gap for a **Python-native, well-documented, extensible** FSS tool.

---

## Current State

The codebase currently supports:

- Loading data from structured text files
- Standard scaling ansatz: X = (β − β_c) · L^b, Y = Observable · L^a
- Polynomial fitting to measure collapse quality
- Nelder-Mead, L-BFGS-B, and dual annealing optimizers
- Window scans, polynomial order scans, and jackknife error bars
- Visualization notebook for raw and collapsed data

---

## Planned Features

### Phase 1: Pluggable Fitting Backends

**Goal:** Allow the user to choose how the master curve is fitted.

Currently the collapse quality is measured by the residual of a single polynomial fit. This works well for smooth, single-regime master curves but breaks down when the data has multiple regimes or sharp features.

**Planned backends:**

| Backend | Use case |
|---------|----------|
| `PolynomialFit(order=N)` | Smooth, single-regime curves (current default) |
| `SplineFit(knots=K)` | Multi-regime behavior, non-smooth master curves |
| `UserDefinedFit(func)` | User provides an explicit function, e.g. `a*tanh(b*x) + c` |
| `PadeFit(m, n)` | Rational approximation for curves with poles or asymptotes |

Each backend should implement a common interface:

```python
class FitBackend:
    def fit(self, X, Y) -> residual: ...
    def predict(self, X) -> Y: ...
```

### Phase 2: Custom Scaling Ansätze

**Goal:** Let users define their own rescaling beyond the standard power-law form.

**Built-in options:**

- **Standard:** X = (x − x_c) · L^b, Y = y · L^a
- **Logarithmic corrections:** Y = y · L^a · (log L)^c
- **Irrelevant scaling variable:** includes L^{−ω} correction terms
- **BKT-type:** exponential scaling near Berezinskii-Kosterlitz-Thouless transitions

**User-defined option:**

```python
collapse.set_ansatz(
    x_scale=lambda x, L, params: (x - params['xc']) * L ** params['b'],
    y_scale=lambda y, L, params: y * L ** params['a'] * np.log(L) ** params['c'],
    param_names=['xc', 'a', 'b', 'c'],
)
```

### Phase 3: Generalized Data Input

**Goal:** Accept data in any common format, not tied to a specific file naming convention.

**Supported formats:**

- NumPy arrays: `(L_list, x_array, y_array)`
- Text/CSV files with user-specified column mapping
- Pandas DataFrames with `(L, x, y)` columns
- HDF5 files (for large datasets)

```python
from fss import ScalingData

# From arrays
data = ScalingData.from_arrays(L=[16, 32, 64], x=x_arrays, y=y_arrays)

# From files
data = ScalingData.from_csv("data.csv", L_col="size", x_col="temperature", y_col="magnetization")

# From DataFrame
data = ScalingData.from_dataframe(df, L_col="L", x_col="T", y_col="m")
```

### Phase 4: Result Objects and Plotting

**Goal:** Return structured results with built-in visualization.

```python
result = collapse.optimize(initial_params=...)

# Structured access
result.params          # {'xc': 0.474, 'a': 0.053, 'b': 0.405}
result.exponents       # {'nu': 2.47, 'kappa': 0.131}
result.residual        # 1.23e-06
result.quality         # collapse quality metric (normalized)

# Built-in plots
result.plot_collapse()          # rescaled data
result.plot_raw()               # raw data
result.plot_residual_landscape() # 2D residual map over parameter pairs

# Error estimation
result.jackknife()     # returns errors on all params
result.bootstrap(n=1000)
```

### Phase 5: Packaging and Distribution

**Goal:** Make it pip-installable with proper documentation.

**Package structure:**

```
fss/
├── __init__.py
├── core/
│   ├── data.py              # ScalingData class
│   ├── ansatz.py             # Built-in and custom scaling ansätze
│   ├── collapse.py           # ScalingCollapse optimizer
│   └── result.py             # CollapseResult with plotting
├── fit/
│   ├── base.py               # FitBackend interface
│   ├── polynomial.py
│   ├── spline.py
│   └── user_defined.py
├── error/
│   ├── jackknife.py
│   └── bootstrap.py
├── io/
│   ├── loaders.py            # CSV, text, HDF5
│   └── exporters.py          # save results
└── examples/
    ├── ising_2d.py
    ├── percolation.py
    └── custom_ansatz.py
```

**Distribution:**

- `pyproject.toml` with dependencies (numpy, scipy, matplotlib)
- Publish on PyPI
- Documentation with Sphinx or MkDocs
- Example notebooks in `examples/`

---

## Envisioned User Experience

A typical workflow with the finished library:

```python
from fss import ScalingData, ScalingCollapse, SplineFit

# 1. Load data
data = ScalingData.from_csv("my_data.csv",
                             L_col="system_size",
                             x_col="temperature",
                             y_col="order_parameter")

# 2. Set up collapse
collapse = ScalingCollapse(data, fit_method=SplineFit(knots=12))

# 3. (Optional) custom ansatz
collapse.set_ansatz(
    x_scale=lambda x, L, p: (x - p['Tc']) * L ** (1/p['nu']),
    y_scale=lambda y, L, p: y * L ** (p['beta']/p['nu']),
    param_names=['Tc', 'nu', 'beta'],
)

# 4. Optimize
result = collapse.optimize(
    initial_params={'Tc': 2.27, 'nu': 1.0, 'beta': 0.125},
    method='Nelder-Mead',
)

# 5. Results
print(result.params)
print(result.quality)

# 6. Visualize
result.plot_collapse()

# 7. Error bars
errors = result.jackknife()
print(errors)
```

---

## Development Pipeline

| Priority | Task | Status |
|----------|------|--------|
| 1 | Restructure into Python package layout | Not started |
| 2 | Define `FitBackend` interface and implement polynomial + spline | Not started |
| 3 | Define `Ansatz` interface with built-in standard forms | Not started |
| 4 | Generalize data input (arrays, CSV, DataFrame) | Not started |
| 5 | Implement `CollapseResult` with plotting methods | Not started |
| 6 | Add bootstrap error estimation alongside jackknife | Not started |
| 7 | Write example notebooks (Ising, percolation, custom) | Not started |
| 8 | Add `pyproject.toml`, publish to PyPI | Not started |
| 9 | Write documentation (API reference + tutorials) | Not started |

---

## Contributing

If you'd like to contribute, pick a task from the pipeline above and open a PR. Key principles:

- Keep the API simple — one class per concept
- Every fitting backend must implement the `FitBackend` interface
- Every ansatz must be a callable or follow the `Ansatz` interface
- Add a test and an example for every new feature
