#   Copyright 2023-2024 Jianbo ZHU
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.special import expit, logit
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.interpolate import Akima1DInterpolator, make_interp_spline
from scipy.integrate import quad


def mixing(datas, weight=None, scale=None):
    '''
    Mix several datas with the same shape.

    Parameters
    ----------
    datas : sequence of array_like
        datas with the same shape
    weight : sequence of float, optional
        weights of mixing, with the same shape of datas, by default ones_like(datas) array
    scale : float, optional
        scale factor, by default 1/sum(weight)

    Returns
    -------
    ndarray
        mixing result, with the same shape of data
    '''
    
    num = len(datas)
    
    if weight is None:
        weight = np.ones(num)
    else:
        weight = np.array(weight)
        if len(weight) != num:
            raise ValueError("Parameter 'weight' must has the same length like datas")
        
    if scale is None:
        scale = 1/np.sum(weight)
    
    data = np.zeros_like(datas[0])
    shape = data.shape
    for i, (d, w) in enumerate(zip(datas, weight)):
        d = np.array(d)
        if d.shape != shape:
            raise ValueError('All datas must have the same array-shape! '
                             f'The shape of data #{i} is abnormal')
        data += scale * w * d
    return data

def fermidirac(x, inverse=False):
    '''
    Fermi-Dirac function:
      1/(1+e^x)
      
    The inverse function:
      ln(1/y-1)

    Parameters
    ----------
    x : array_like
        Argument of the Fermi-Dirac function
    inverse : bool, optional
        Calculate the value of inverse function, by default False

    Returns
    -------
    ndarray
        An array of the same shape as x
    '''
    if inverse:
        return (-1)*logit(np.clip(x, 0, 1))
    else:
        return expit((-1)*np.asarray(x))

def smoothstep(x, inverse=False, shift=True): 
    '''
    Smoothstep function:

      0, x < 0

      3x^2-2x^3, x <= 0 <= 1

      1, x > 1

    The inverse function:

      1/2 - sin(arcsin(1-2*y)/3)
    
    It might be more convenient when the function is shifted x' = (1-x)/2,
    which drops from 1 to 0 where -1 <= 0 <= 1.
    
    Parameters
    ----------
    x : array_like
        Argument of the smoothstep function (https://en.wikipedia.org/wiki/Smoothstep) 
    inverse : bool, optional
        Calculate the value of inverse function, by default False
    shift : bool, optional
        shift the origin smoothstep function with x' = (1-x)/2, by default True

    Returns
    -------
    ndarray
        An array of the same shape as x
    '''
    if shift:
        if inverse:
            x = np.asarray(x)
            return 2*np.sin(np.arcsin(1-2*x)/3)
        else:
            x = np.clip(x, -1, 1)
            return (x-1)*(x-1)*(x+2)/4
    else:
        if inverse:
            x = np.asarray(x)
            return 1/2 - np.sin(np.arcsin(1-2*x)/3)
        else:
            x = np.clip(x, 0, 1)
            return x*x*(3-2*x)

def _interpx_extra(x, y, xp, left=None, right=None):
    # this function does NOT check input, assuming numpy.ndarray inputs.
    yp = np.interp(xp, x, y, left=left, right=right)
    idx_left = (xp < x[0])
    idx_right = (xp > x[-1])
    if (left is None) and np.any(idx_left):
        y_left = y[0] + (y[1]-y[0])/(x[1]-x[0]) * (xp[idx_left]-x[0])
        yp[idx_left] = y_left
    if (right is None) and np.any(idx_right):
        y_right = y[-1] + (y[-1]-y[-2])/(x[-1]-x[-2]) * (xp[idx_right]-x[-1])
        yp[idx_right] = y_right
    return yp

def interpx(x, y, xp, left=None, right=None):
    '''
    Extends :func:`numpy.interp` to perform linear extrapolation for xp values
    outside the x range, differing from the constant extrapolation with end
    values or specified fill values used by the original function.

    Parameters
    ----------
    x : array_like
        A 1-dimensional array of x-coordinates of the data points. It must be
        strictly monotonically increasing and contain at least two elements.
    y : array_like
        A 1-dimensional array of y-coordinates corresponding to `x`. The arrays
        `x` and `y` must have the same length.
    xp : array_like
        The x-coordinates at which to interpolate/extrapolate values.
    left : float, optional
        Value to return for x < xp[0], default linear extrapolation.
    right : float, optional
        Value to return for x > xp[-1], default linear extrapolation.

    Returns
    -------
    ndarray
        Interpolated/extrapolated values at `xp`, with the output shape
        matching that of `xp`.
    '''
    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()
    xp = np.asarray(xp)
    if (x.ndim != 1) or (y.ndim != 1):
        raise ValueError('Input x and y must both be 1-dimensional.')
    if x.size < 2:
        raise ValueError('Input x must contain at least two elements.')
    return _interpx_extra(x, y, xp, left=left, right=right)

def _interp_vect(vy, vx, xp, method='linear', **kwargs):
    '''
    Interpolate points from vx and vy vectors.
    '''
    #   linear: _interpx_extra
    #     line: numpy.interp
    #  poly<N>: numpy.polynomial.polynomial.Polynomial.fit
    #    cubic: scipy.interpolate.CubicSpline
    #    pchip: scipy.interpolate.PchipInterpolator
    #    Akima: scipy.interpolate.Akima1DInterpolator
    #   spline: scipy.interpolate.make_interp_spline

    if method == 'linear':
        return _interpx_extra(vx, vy, xp, **kwargs)
    elif method == 'line':
        return np.interp(xp, vx, vy, **kwargs)
    elif method.lower().startswith('poly'):
        return Poly.fit(vx, vy, deg=int(method[-1]), **kwargs)(xp)
    elif method == 'cubic':
        return CubicSpline(vx, vy, **kwargs)(xp)
    elif method == 'pchip':
        return PchipInterpolator(vx, vy, **kwargs)(xp)
    elif method.lower() == 'akima':
        return Akima1DInterpolator(vx, vy, **kwargs)(xp)
    elif method == 'spline':
        return make_interp_spline(vx, vy, **kwargs)(xp)
    else:
        raise ValueError(f"Unsupported interpolation method '{method}'")

def vinterp(x, y, xp, method='linear', reorder=True, **kwargs):
    '''
    Vectorized interpolation of values.

    Parameters
    ----------
    x : (N,) array_like
        The x-coordinates of the data points.
    y : (…,N) array_like
        The y-coordinates corresponding to x. The length of the last dimension
        must be the same as x.
    xp : array_like
        The x-coordinates at which to evaluate the interpolated values.
    method : str, optional
        Specifies the interpolation method to be used. Supported methods include:

        - 'linear' (default): Uses :func:`interpx` for linear interpolation.
        - 'line': Uses raw :func:`numpy.interp` for linear interpolation.
        - 'poly<N>': Uses :func:`numpy.polynomial.polynomial.Polynomial.fit`
          for polynomial interpolation of degree N. Replace <N> with the
          degree of the polynomial, e.g., 'poly2' for quadratic.
        - 'cubic': Uses :func:`scipy.interpolate.CubicSpline` for cubic
          spline interpolation.
        - 'pchip': Uses :func:`scipy.interpolate.PchipInterpolator` for
          Piecewise Cubic Hermite Interpolating Polynomial.
        - 'akima': Uses :func:`scipy.interpolate.Akima1DInterpolator` for
          Akima interpolation.
        - 'spline': Uses :func:`scipy.interpolate.make_interp_spline` to
          customize a B-spline representation of the data.
    reorder : bool, optional
        If True (default), automatically reorders `x` and `y` if `x` is not
        monotonically increasing. If False, an error is raised if `x` is not
        monotonically increasing.
    **kwargs : any, optional
        Additional keyword arguments to be passed to the corresponding
        interpolation method.

    Returns
    -------
    ndarray
        The interpolated values.
    '''
    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()
    xp = np.asarray(xp)

    if x.ndim != 1:
        raise ValueError('Input x must be 1-dimensional after squeezing.')

    if y.shape[-1] != x.size:
        raise ValueError('The last dimension of y must match the size of x.')

    for arg in (x, y, xp):
        if not np.all(np.isfinite(arg)):
            raise ValueError('Inputs must not contain NaN or infinite values.')

    if np.any(np.diff(x) < 0):
        if reorder:
            # sort by 'x'
            sort_idx = np.argsort(x)
            x = x[sort_idx]
            y = y[..., sort_idx]
        else:
            raise ValueError('Input x must be strictly monotonically '\
                'increasing. Set reorder=True to enable automatic reordering.')

    return np.apply_along_axis(_interp_vect, axis=-1, arr=y,
                               vx=x, xp=xp, method=method,
                               **kwargs)

def vquad(func, a, b, args=(), *, where=True, fill_value=0, **kwargs):
    '''
    Extend `scipy.integrate.quad` by adding support for broadcasting over
    `a`, `b`, `args`, `where`, and `fill_value`. The `where` and `fill_value`
    parameters are introduced for enhanced flexibility in controlling
    integration.

    Note that this function is not a true ufunc in the numpy sense. It
    leverages broadcasting and Python loops to offer a convenient interface
    for integrating a function across varying ranges and with diverse
    parameters. It prioritizes ease of use over optimal performance.

    Parameters
    ----------
    func : callable
        A Python function or method to integrate.
    a : array_like
        Lower limit of integration.
    b : array_like
        Upper limit of integration.
    args : tuple of array_like, optional
        Extra arguments to pass to `func`. Each element of `args` will be
        broadcasted to match the shape of `a` and `b`. Default is an empty
        tuple.
    where : array_like, optional
        Boolean mask to specify where to perform the integration. Default
        is `True`, which means that the integration is performed everywhere.
    fill_value : array_like, optional
        The value to use for masked positions. Default is `0`.
    **kwargs
        Additional keyword arguments passed to `scipy.integrate.quad`.

    Returns
    -------
    ndarray
        Array of computed integral values with the same shape as the
        broadcasted shape of `a`, `b`, and each element of `args`.
    ndarray
        Array of estimated integration errors with the same shape as
        the first returned ndarray.

    Examples
    --------
    Consider a function `func(x, m, n)` defined as:

    >>> def func(x, m, n):
    ...     return m*x**2 if x<0 else n*x**3

    We can integrate `func` over the intervals `[-1, 1]` and `[-2, 2]` with
    `args` as `(3, [[4], [8], [12]])` . Here, `3` corresponds to 
    the parameter `m` in `func`, and the column vector `[[4], [8], [12]]` 
    corresponds to the parameter `n` in `func`. The broadcasting results in 
    an output shape of `(3, 2)`:

    >>> vquad(func, [-1, -2], [1, 2], args=(3, [[4], [8], [12]]))[0]
    array([[ 2., 24.],
           [ 3., 40.],
           [ 4., 56.]])
    '''
    broadcasted = np.broadcast(a, b, where, fill_value, *args)
    bshape = broadcasted.shape
    itg = np.empty(bshape)
    res = np.empty(bshape)
    indexed = np.ndindex(bshape)

    for idx, (ia, ib, unmask, default, *iargs) in zip(indexed, broadcasted):
        if unmask:
            itg[idx], res[idx] = quad(func, ia, ib, args=tuple(iargs), **kwargs)
        else:
            itg[idx], res[idx] = default, 0
    return itg, res
