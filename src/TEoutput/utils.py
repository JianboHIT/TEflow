#!/usr/bin/env python3
 
import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.interpolate import interp1d


def interp(x, y, x2, method='linear', merge=False):
    '''
    A convenient method to implement scipy.interpolate.interp1d and Polynomial.fit

    Parameters
    ----------
    x : (N,) array_like
        A 1-D array of real values.
    y : (…,N) array_like
        A N-D array of real values. 
    x2 : array_like
        The sampling points to evaluate the interpolated values.
    method : str, optional
        'linear'(default), 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 
        'previous', 'next', 'poly1', 'poly2','poly3','poly4','poly5'.
    merge : bool, optional
        Whether to merge the sampling points into the interpolation result, by default False

    Returns
    -------
    ndarray
        The interpolated values, same shape as x.
    '''
    
    if method.lower().startswith('poly'):
        # polyfit
        order = int(method[-1])
        y2 = [x2,] if merge else []
        for yi in y:
            pfit = Poly.fit(x, yi, deg=order)
            y2.append(pfit(x2))
        datas = np.vstack(y2)
    else:
        # piecewise interpolate
        fx = interp1d(x, y, kind=method, fill_value='extrapolate')
        y2 = fx(x2)
        datas = np.vstack([x2,y2]) if merge else y2
    return datas