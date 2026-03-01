# Finite-Size Scaling Analysis

A Python toolkit for performing finite-size scaling (FSS) data collapse to extract critical exponents near phase transitions.

## Overview

Given an observable measured at different system sizes as a function of a control parameter (inverse temperature β), this code finds the optimal critical point β_c and scaling exponents by minimizing the residual of a polynomial fit to the rescaled data:

- **X** = (β − β_c) · L^b
- **Y** = Observable · L^a

The extracted critical exponents are **ν = 1/b** and **κ = a·ν**.

## Project Structure

```
├── FSS/                        # Core library
│   ├── data_set.py             # DataSet container class
│   ├── finitesizescaling.py    # FSS optimization engine
│   └── utilities.py            # Rescaling and helper functions
├── data/                       # Input data files (.dat)
├── experiment.py               # ExperimentRunner high-level API
├── run_fss.py                  # Entry point script
└── DATA_COLLAPSE.ipynb         # Jupyter notebook for visualization
```

## Usage

Edit `run_fss.py` to select your dataset and experiment type, then run:

```bash
python run_fss.py
```

### Available experiments

| Method | Description |
|--------|-------------|
| `single_run(w=...)` | Single FSS optimization at a given window half-width |
| `window_scan(w_values)` | Scan over scaling window half-widths |
| `poly_order_scan(orders)` | Scan over polynomial fitting orders |
| `jackknife(w=...)` | Leave-one-out jackknife error estimation |
| `grid_scan(...)` | Combine any two scans (e.g., window scan per poly order) |

### Example

```python
from experiment import ExperimentRunner

runner = ExperimentRunner(
    dt="0.002", a="1.80", h="0.30",
    initial_params=(0.80, 0.11, 0.43),
    sizes=[200, 250, 300, 350, 400],
    poly_order=10,
)

runner.single_run(w=0.8)
```

## Requirements

- Python 3
- NumPy
- SciPy
