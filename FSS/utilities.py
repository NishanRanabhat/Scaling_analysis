import numpy as np

def Y_rescaled(y,L,a):
    """
    Rescales an observable (y) using a finite-size scaling factor.

    This function applies a scaling transformation to the input value `y` 
    by multiplying it with the system size `L` raised to the power `a`.

    Parameters
    ----------
    y : float or array-like
        The observable value(s) to be rescaled.
    L : float or int
        The characteristic system size.
    a : float
        The scaling exponent for the observable.

    Returns
    -------
    float or numpy.ndarray
        The rescaled observable, computed as y * L**a.

    Examples
    --------
    >>> YL(2.0, 16, 0.125)
    2.0 * 16**0.125
    """
    return y * L**a

def X_rescaled(x, L, xc, b):
    """
    Rescales a control variable (x) relative to a critical value (xc) using a finite-size scaling factor.

    This function transforms the input value `x` by first subtracting 
    a critical value `xc` and then multiplying the result by `L` raised to the power `b`.

    Parameters
    ----------
    x : float or array-like
        The original value(s) of the control parameter.
    L : float or int
        The system size used for scaling.
    xc : float
        The critical value of the control parameter.
    b : float
        The scaling exponent for the control parameter.

    Returns
    -------
    float or numpy.ndarray
        The rescaled control parameter, computed as (x - xc) * L**b.

    Examples
    --------
    >>> XL(2.5, 16, 2.3, 1.0)
    (2.5 - 2.3) * 16**1.0
    """
    return (x - xc) * L**b

def slice_limits(data_list, lower_lim, upper_lim):
    """
    Return a boolean mask selecting values in [lower_lim, upper_lim] (inclusive).
    Using an inclusive mask makes counts monotonic when you widen the window.
    """
    data = np.asarray(data_list)
    return (data >= lower_lim) & (data <= upper_lim)

def closest_index(data_list, input_val):
    """
    Finds the index of the element in a monotonic array that is closest to a given value.

    The function assumes that `data_list` is a monotonic (increasing or decreasing) array.
    It computes the absolute difference between each element and the given `input_val` 
    and returns the index of the element with the minimum difference.

    Parameters
    ----------
    data_list : array-like
        A monotonic array (e.g., NumPy array or list) of numeric values.
    input_val : float
        The value for which the closest index in data_list is sought.

    Returns
    -------
    int
        The index of the element in data_list that is closest to input_val.

    Examples
    --------
    >>> data = np.array([0, 1, 2, 3, 4, 5])
    >>> closest_index(data, 2.7)
    3  # Since 3 is the closest to 2.7.
    """
    
    return np.where(np.abs(data_list-input_val)== min(np.abs(data_list-input_val)))[0][0]
