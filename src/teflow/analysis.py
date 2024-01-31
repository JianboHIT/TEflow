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
from scipy.interpolate import interp1d
from scipy.integrate import quad


def interp(x, y, x2, method='linear', axis=-1, bounds_error=None, fill_value='extrapolate', merge=False):
    '''
    A convenient method to implement scipy.interpolate.interp1d and Polynomial.fit.

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
    axis : int, optional
        Specifies the axis of y along which to interpolate, by default -1.
    bounds_error : bool, optional
        If True, a ValueError is raised any time interpolation is attempted on a value outside 
        of the range of x (where extrapolation is necessary). If False, out of bounds values are 
        assigned fill_value. By default, an error is raised unless fill_value="extrapolate".
    fill_value : array-like or (array-like, array_like) or “extrapolate”, optional
        A ndarray (or float) or a two-element tuple is supported. If “extrapolate”, then points 
        outside the data range will be extrapolated.
    merge : bool, optional
        Whether to merge the sampling points into the interpolation result, by default False.

    Returns
    -------
    datas: ndarray
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
        fx = interp1d(x, y, 
                      kind=method, 
                      axis=axis, 
                      bounds_error=bounds_error, 
                      fill_value=fill_value)
        y2 = fx(x2)
        datas = np.vstack([x2,y2]) if merge else y2
    return datas

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

def boltzmann(x, inverse=False):
    '''
    Boltzmann function:
      1/(1+e^x)
      
    The inverse function:
      ln(1/y-1)

    Parameters
    ----------
    x : array_like
        Argument of the Boltzmann function
    inverse : bool, optional
        Calculate the value of inverse function, by default False

    Returns
    -------
    ndarray
        An array of the same shape as x
    '''
    if inverse:
        return np.log(1/x-1)
    else:
        return 1/2*(1-np.tanh(x/2)) # 1/(1+exp(u))

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
            return 2*np.sin(np.arcsin(1-2*x)/3)
        else:
            x = np.minimum(np.maximum(x, -1), 1)
            return (x-1)*(x-1)*(x+2)/4
    else:
        if inverse:
            return 1/2 - np.sin(np.arcsin(1-2*x)/3)
        else:
            x = np.minimum(np.maximum(x, 0), 1)
            return x*x*(3-2*x)

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
