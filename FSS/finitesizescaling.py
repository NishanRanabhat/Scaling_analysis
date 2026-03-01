import numpy as np
from scipy.optimize import curve_fit, minimize, dual_annealing, basinhopping 
from scipy import interpolate
from scipy.interpolate import interp2d, interp1d
from utilities import Y_rescaled,X_rescaled,slice_limits,closest_index

def _mask_scaled_window_with_min(beta, L, beta_c, b, window, min_points):
    """
    Build a mask for |X| inside 'window' where X=(beta-beta_c)L^b.
    If fewer than 'min_points' are inside, expand to the k smallest |X|.
    Returns: (mask [bool 1-D], X [float 1-D])
    """
    # Force 1-D arrays to avoid 0-D (scalar) indexing issues
    beta = np.atleast_1d(beta).astype(float)
    X = (beta - beta_c) * (L ** b)
    X = np.asarray(X).ravel()  # 1-D
    lo, hi = window

    # Inclusive mask -> monotonic counts when widening
    mask = ((X >= lo) & (X <= hi)).ravel()  # 1-D
    have = int(mask.sum())

    if min_points:
        k = min(int(min_points), X.size)
        if have < k and X.size > 0:
            # Select k points closest to X=0 and UNION with window mask
            order = np.argsort(np.abs(X))
            nearest = np.zeros(X.shape[0], dtype=bool)
            nearest[order[:k]] = True
            mask = mask | nearest

    return mask, X

class FSS:
    def __init__(self, dataset,
                 poly_order=8,
                 initial_params=None,
                 param_bounds=None,
                 scaling_window=(-2.0, 2.0),     # NEW default
                 optimization_routine=None,
                 min_points_per_size=60):        # NEW default
        self.dataset = dataset
        self.poly_order = poly_order
        self.initial_params = initial_params
        self.param_bounds = param_bounds
        self.scaling_window = scaling_window
        self.optimization_routine = optimization_routine
        self.min_points_per_size = int(min_points_per_size)

    def rescaled_combined_data(self, params):
        # params ordering: [beta_c, a, b] (adjust if yours differs)
        beta_c, a, b = params

        X_all, Y_all = [], []

        # Detect whether domain_list is a single common 1D grid or a list/array of per-size grids
        domain = self.dataset.domain_list

        for L_index, L in enumerate(self.dataset.system_size_list):

            # --- pick the correct beta grid for this size ---
            if isinstance(domain, np.ndarray) and domain.ndim == 1:
                # One common beta grid for all sizes
                beta = domain
            else:
                # Per-size beta grids; support either a list of arrays or a 2D array
                beta = domain[L_index]
                beta = np.asarray(beta)


            y = self.dataset.range_list[L_index]
            y = np.asarray(y)

            # --- sanity: lengths must match ---
            if beta.shape[0] != y.shape[0]:
                raise ValueError(
                    f"Length mismatch for L={L}: len(beta)={beta.shape[0]} vs len(y)={y.shape[0]}. "
                    "If domain_list is a single common grid, keep it 1D; "
                    "if per-size, ensure each beta[L_index] matches range_list[L_index]."
                )

            # --- build mask with min-points and rescale ---
            mask, X = _mask_scaled_window_with_min(
                beta=np.asarray(beta),
                L=float(L),
                beta_c=float(beta_c),
                b=float(b),
                window=self.scaling_window,
                min_points=self.min_points_per_size
            )

            X_res = X[mask]
            Y_res = (y[mask]) * (L ** a)

            #print(L,X_res.shape)
            # Optional: normalize per size to avoid one size dominating
            # Y_res = Y_res / (np.max(np.abs(Y_res)) + 1e-12)
            if X_res.size:
                X_all.append(X_res)
                Y_all.append(Y_res)

        # Concatenate and drop non-finite pairs
        X_fin = np.concatenate(X_all) if X_all else np.array([], dtype=float)
        Y_fin = np.concatenate(Y_all) if Y_all else np.array([], dtype=float)
        if X_fin.size:
            m = np.isfinite(X_fin) & np.isfinite(Y_fin)
            X_fin, Y_fin = X_fin[m], Y_fin[m]

        return X_fin, Y_fin

    def objective_function(self, params):
        X_fin, Y_fin = self.rescaled_combined_data(params)
        z = np.polyfit(X_fin, Y_fin, self.poly_order, full=True)
        residuals = z[1]
        return float(residuals[0]) if residuals.size else np.inf

    def optimization(self,):

        if self.optimization_routine == "L-BFGS-B":
            if self.param_bounds != None:
                result = minimize(self.objective_function, x0=self.initial_params, method=self.optimization_routine, bounds=self.param_bounds, options= {'ftol': 1e-12,'gtol': 1e-12,'maxiter': 50})
            else:
                raise ValueError("L-BFGS-B need valid param-bounds")
        elif self.optimization_routine == "Nelder-Mead":
            result = minimize(self.objective_function, x0=self.initial_params, method=self.optimization_routine, options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000})
        elif self.optimization_routine == "dual-annealing":
            if self.param_bounds != None:
                result = dual_annealing(self.objective_function,bounds=self.param_bounds)
            else:
                raise ValueError("dual-annealing need valid param-bounds")

        return result.x, result.fun






    

