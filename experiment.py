import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "FSS"))

from data_set import DataSet
from finitesizescaling import FSS

ALL_SIZES = [200, 250, 300, 350, 400]

# Registry of known dataset configurations.
# Key: (dt, a, h) tuple.
# Value: dict with obs_col and whether beta grids differ per size.
DATASET_REGISTRY = {
    ("0.001", "0.80", "0.00"): {"obs_col": 1, "per_size_beta": False},
    ("0.001", "0.80", "0.30"): {"obs_col": 1, "per_size_beta": False},
    ("0.002", "1.80", "0.00"): {"obs_col": 1, "per_size_beta": False},
    ("0.002", "1.80", "0.30"): {"obs_col": 1, "per_size_beta": False},
}


def load_dataset(dt, a, h, sizes=None, obs_col=None, data_dir=None):
    """
    Load data files and return a DataSet ready for FSS.

    Parameters
    ----------
    dt : str       e.g. "0.002"
    a  : str       e.g. "1.80"
    h  : str       e.g. "0.30"
    sizes : list of int, optional
        System sizes to include. Default: ALL_SIZES.
    obs_col : int, optional
        Column index for the observable. Default from registry.
    data_dir : str, optional
        Path to directory containing data files. Default: parent of FSS/.

    Returns
    -------
    DataSet
    """
    if sizes is None:
        sizes = list(ALL_SIZES)
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    key = (dt, a, h)
    info = DATASET_REGISTRY.get(key)
    if info is None:
        raise ValueError(
            f"Unknown dataset: dt={dt}, a={a}, h={h}. "
            f"Known: {list(DATASET_REGISTRY.keys())}"
        )

    col = obs_col if obs_col is not None else info["obs_col"]
    per_size_beta = info["per_size_beta"]

    raw = {}
    for N in sizes:
        fname = f"data_vs_beta_dt={dt}_N={N}_chi_max=256_a={a}_h={h}_.dat"
        raw[N] = np.loadtxt(os.path.join(data_dir, fname))

    L_list = np.array(sizes)

    if per_size_beta:
        beta_list = [raw[N][:, 0] for N in sizes]
        SG_list = [raw[N][:, col] for N in sizes]
    else:
        nrows = raw[sizes[0]].shape[0]
        beta_list = raw[sizes[0]][:, 0]
        SG_list = np.empty((len(sizes), nrows))
        for i, N in enumerate(sizes):
            SG_list[i, :] = raw[N][:, col]

    return DataSet(L_list, beta_list, SG_list)


class ExperimentRunner:
    """
    Composable FSS experiment runner.

    Usage:
        runner = ExperimentRunner(dt="0.002", a="1.80", h="0.30",
                                  initial_params=(0.80, 0.11, 0.43))
        runner.single_run(w=0.8)
        runner.window_scan(np.arange(0.3, 2.1, 0.1))
        runner.jackknife(w=0.8)
    """

    def __init__(self, dt, a, h, initial_params,
                 sizes=None, poly_order=10,
                 scaling_window=(-0.8, 0.8),
                 optimization_routine="Nelder-Mead",
                 param_bounds=None, obs_col=None,
                 min_points_per_size=60, data_dir=None,
                 fixed_params=None):

        self.dt = dt
        self.a = a
        self.h = h
        self.sizes = sizes or list(ALL_SIZES)
        self.initial_params = initial_params
        self.poly_order = poly_order
        self.scaling_window = scaling_window
        self.optimization_routine = optimization_routine
        self.param_bounds = param_bounds
        self.obs_col = obs_col
        self.min_points_per_size = min_points_per_size
        self.data_dir = data_dir
        self.fixed_params = fixed_params

    def _run_once(self, sizes=None, w=None, poly_order=None,
                  initial_params=None, optimization_routine=None,
                  param_bounds=None, min_points_per_size=None,
                  fixed_params=None):
        """Run a single FSS optimization. Returns (params_array, residual)."""
        sizes = sizes or self.sizes
        w = w if w is not None else self.scaling_window[1]
        poly_order = poly_order if poly_order is not None else self.poly_order
        initial_params = initial_params or self.initial_params
        optimization_routine = optimization_routine or self.optimization_routine
        param_bounds = param_bounds if param_bounds is not None else self.param_bounds
        min_points = (min_points_per_size if min_points_per_size is not None
                      else self.min_points_per_size)
        fixed_params = fixed_params if fixed_params is not None else self.fixed_params

        dataset = load_dataset(self.dt, self.a, self.h,
                               sizes=sizes, obs_col=self.obs_col,
                               data_dir=self.data_dir)

        fss = FSS(dataset,
                  poly_order=poly_order,
                  initial_params=initial_params,
                  param_bounds=param_bounds,
                  scaling_window=(-w, w),
                  optimization_routine=optimization_routine,
                  min_points_per_size=min_points,
                  fixed_params=fixed_params)

        params, residual = fss.optimization()
        return params, residual

    def single_run(self, **kwargs):
        """Single FSS optimization. Prints and returns result."""
        params, residual = self._run_once(**kwargs)
        beta_c, a_exp, b_exp = params
        nu = 1.0 / b_exp
        kappa = a_exp * nu

        print(f"beta_c  = {beta_c:.6f}")
        print(f"nu      = {nu:.4f}  (b = 1/nu = {b_exp:.6f})")
        print(f"kappa   = {kappa:.4f}  (a = kappa/nu = {a_exp:.6f})")
        print(f"residual = {residual:.6e}")

        return {"beta_c": beta_c, "nu": nu, "kappa": kappa,
                "a": a_exp, "b": b_exp, "residual": residual}

    def window_scan(self, w_values, chain_initial=True, **kwargs):
        """
        Scan over scaling window half-widths.

        Parameters
        ----------
        w_values : array-like
            Sequence of window half-widths to try.
        chain_initial : bool
            If True, use previous result as initial params for next w.
        """
        results = []
        print(f"{'w':>5s}  {'beta_c':>10s}  {'nu':>8s}  {'kappa':>8s}  "
              f"{'b':>10s}  {'a':>10s}  {'residual':>12s}")
        print("-" * 72)

        current_params = kwargs.pop("initial_params", None) or self.initial_params

        for w in w_values:
            params, residual = self._run_once(
                w=w, initial_params=current_params, **kwargs)
            beta_c, a_exp, b_exp = params
            nu = 1.0 / b_exp
            kappa = a_exp * nu

            print(f"{w:5.2f}  {beta_c:10.6f}  {nu:8.4f}  {kappa:8.4f}  "
                  f"{b_exp:10.6f}  {a_exp:10.6f}  {residual:12.6e}")

            results.append({"w": w, "beta_c": beta_c, "nu": nu, "kappa": kappa,
                            "a": a_exp, "b": b_exp, "residual": residual})

            if chain_initial:
                current_params = tuple(params)

        return results

    def poly_order_scan(self, orders, **kwargs):
        """Scan over polynomial fitting orders."""
        results = []
        print(f"{'order':>5s}  {'beta_c':>10s}  {'nu':>8s}  {'kappa':>8s}  "
              f"{'b':>10s}  {'a':>10s}  {'residual':>12s}")
        print("-" * 72)

        for p in orders:
            params, residual = self._run_once(poly_order=p, **kwargs)
            beta_c, a_exp, b_exp = params
            nu = 1.0 / b_exp
            kappa = a_exp * nu

            print(f"{p:5d}  {beta_c:10.6f}  {nu:8.4f}  {kappa:8.4f}  "
                  f"{b_exp:10.6f}  {a_exp:10.6f}  {residual:12.6e}")

            results.append({"poly_order": p, "beta_c": beta_c, "nu": nu,
                            "kappa": kappa, "a": a_exp, "b": b_exp,
                            "residual": residual})

        return results

    def jackknife(self, **kwargs):
        """
        Leave-one-out jackknife over system sizes.
        Returns dict with main result and jackknife standard errors.
        """
        all_sizes = kwargs.pop("sizes", None) or self.sizes

        # Full-data run
        main_params, main_res = self._run_once(sizes=all_sizes, **kwargs)
        beta_c_main = main_params[0]
        nu_main = 1.0 / main_params[2]
        kappa_main = main_params[1] * nu_main

        print(f"Full dataset: beta_c={beta_c_main:.6f}, "
              f"nu={nu_main:.4f}, kappa={kappa_main:.4f}")
        print(f"\nJackknife (dropping one size at a time):")
        print(f"{'dropped':>8s}  {'beta_c':>10s}  {'nu':>8s}  {'kappa':>8s}")
        print("-" * 42)

        beta_c_jk, nu_jk, kappa_jk = [], [], []

        for dropped in all_sizes:
            subset = [s for s in all_sizes if s != dropped]
            params, _ = self._run_once(sizes=subset, **kwargs)
            bc = params[0]
            nu = 1.0 / params[2]
            k = params[1] * nu

            print(f"  N={dropped:>4d}  {bc:10.6f}  {nu:8.4f}  {k:8.4f}")

            beta_c_jk.append(bc)
            nu_jk.append(nu)
            kappa_jk.append(k)

        def jk_se(vals):
            vals = np.array(vals)
            n = len(vals)
            mean = vals.mean()
            se = np.sqrt(np.sum((vals - mean) ** 2) / n)
            return se, mean

        se_bc, mean_bc = jk_se(beta_c_jk)
        se_nu, mean_nu = jk_se(nu_jk)
        se_k, mean_k = jk_se(kappa_jk)

        print(f"\nResults:")
        print(f"  beta_c = {beta_c_main:.6f} +/- {se_bc:.6f}")
        print(f"  nu     = {nu_main:.4f} +/- {se_nu:.4f}")
        print(f"  kappa  = {kappa_main:.4f} +/- {se_k:.4f}")

        return {
            "main": {"beta_c": beta_c_main, "nu": nu_main, "kappa": kappa_main},
            "jackknife_se": {"beta_c": se_bc, "nu": se_nu, "kappa": se_k},
            "jackknife_mean": {"beta_c": mean_bc, "nu": mean_nu, "kappa": mean_k},
        }

    def grid_scan(self, scan_over, values, inner_method, **inner_kwargs):
        """
        Run an inner experiment method for each value of a scanned parameter.

        Example
        -------
        runner.grid_scan("poly_order", [8, 10, 12],
                         "window_scan", w_values=np.arange(0.3, 2.0, 0.1))
        """
        all_results = {}
        method = getattr(self, inner_method)

        for val in values:
            print(f"\n{'=' * 60}")
            print(f"  {scan_over} = {val}")
            print(f"{'=' * 60}")

            inner_kwargs[scan_over] = val
            result = method(**inner_kwargs)
            all_results[val] = result

        return all_results
