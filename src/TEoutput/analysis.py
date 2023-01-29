import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.interpolate import interp1d


def interp(x, y, x2, method='linear', axis=-1, bounds_error=None, fill_value='extrapolate', merge=False):
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
    mix several data with the same shape

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
    array_like
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