import logging 
import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.interpolate import interp1d


def get_pkg_name():
    '''
    Get package name
    '''
    return __name__.split('.')[0]

def get_root_logger(stdout=True):
    '''
    Get root logger with package name
    '''
    logger = logging.getLogger(get_pkg_name())
    if stdout:
        console = get_logger_handler()
        logger.addHandler(console)
    return logger

def get_script_logger(name=None, level=None, stdout=True):
    '''
    Get a logger for scripts using
    '''
    if name is None:
        name = '{}_script'.format(get_pkg_name())
    logger = logging.getLogger(name)
    
    if level is not None:
        logger.setLevel(level)
    if stdout:
        console = get_logger_handler()
        logger.addHandler(console)
    return logger

def get_logger_handler(kind='CONSOLE',*args,**kwargs):
    '''
    Get kinds of handler of logging
    '''
    kind = kind.lower()
    if kind in {'cons', 'console', 'stream'}:
        formatter = logging.Formatter(
            fmt='[%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        handler = logging.StreamHandler()
    elif kind in {'file', 'logfile'}:
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)s @ %(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        handler = logging.FileHandler(*args, **kwargs)
    else:
        raise ValueError('The kind of handler is invaild.')
    handler.setFormatter(formatter)
    return handler

class AttrDict(dict):
    def __getattr__(self, attr):
        return self[attr]

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

def cutrange(datas_TXX, a, b):
    T, *props = datas_TXX
    v_a = np.array([np.interp(a,T,prop) for prop in props]).reshape((-1,1))
    v_b = np.array([np.interp(b,T,prop) for prop in props]).reshape((-1,1))
    
    idx = (T > a) & (T < b)
    v_props = np.hstack([v_a, np.array(props)[:, idx], v_b])
    v_T = np.hstack([a, T[idx], b])
    return np.vstack([v_T, v_props])
