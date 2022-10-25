import logging
import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.interpolate import interp1d


def get_pkg_name():
    '''
    Get package name
    '''
    return __name__.split('.')[0]

def get_root_logger(stdout=True, filename=None, mode='a', level=None):
    '''
    Get root logger object
    '''
    logger = logging.getLogger(get_pkg_name())
    if stdout:
        console = get_logger_handler()
        logger.addHandler(console)
    if filename is not None:
        fh = get_logger_handler(kind='file',
                                filename=filename,
                                mode=mode)
        logger.addHandler(fh)
    if level is not None:
        logger.setLevel(level)
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
        # define dot access mode like d.key
        return self[attr]
    
    def append(self, obj):
        # define append operation
        if isinstance(obj, dict):
            for key in self.keys():
                self[key].append(obj[key])
        else:
            raise ValueError('Only Dict() object can be appended')
    
    def extend(self, obj):
        # define extend operation
        if isinstance(obj, dict):
            for key in self.keys():
                self[key].extend(obj[key])
        else:
            raise ValueError('Only Dict() object can be extended')
        
    @classmethod
    def merge(cls, obj, obj2):
        # define merge operation: c = AttrDict.merge(a,b)
        if isinstance(obj, dict) and isinstance(obj2, dict):
            if obj.keys() == obj2.keys():
                out = cls()
                for key in obj.keys():
                    out[key] = [obj[key], obj2[key]]
                return out
            else:
                raise KeyError('Two object must have same keys')
        else:
            raise ValueError('Two Dict() object are required to merge')

class Metric():
    kinds = {'MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE'}
    def __init__(self, kind='MSE'):
        kind = kind.upper()
        if kind in self.kinds:
            self._kind = kind
            self._func = getattr(self, kind)
        else:
            raise ValueError('Invaild kind of metric')
    
    def __call__(self, y, y2, axis=-1):
        return self._func(y, y2, axis)
    
    @staticmethod
    def MSE(y, y2, axis=-1):
        '''
        Mean Square Error
        '''
        diff = np.array(y2) - np.array(y)
        v2 = np.mean(np.square(diff), axis=axis)
        return v2
    
    @staticmethod
    def RMSE(y, y2, axis=-1):
        diff = np.array(y2) - np.array(y)
        v2 = np.mean(np.square(diff), axis=axis)
        return np.sqrt(v2)
    
    @staticmethod
    def MAE(y, y2, axis=-1):
        '''
        Mean Absolute Error
        '''
        diff = np.array(y2) - np.array(y)
        v = np.mean(np.absolute(diff), axis=axis)
        return v
    
    @staticmethod
    def MAPE(y, y2, axis=-1):
        '''
        Mean Absolute Percentage Error
        '''
        rdiff = np.array(y2)/np.array(y)-1
        v = np.mean(np.absolute(rdiff), axis=axis)
        return v
    
    @staticmethod
    def SMAPE(y, y2, axis=-1):
        y = np.array(y)
        y2 = np.array(y2)
        diff = np.absolute(y2-y)
        sum_ = np.absolute(y2)+np.absolute(y)
        v = 2.0 * np.mean(diff/sum_, axis=axis)
        return v
    
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
        fx = interp1d(x, y, kind=method, fill_value='extrapolate')
        y2 = fx(x2)
        datas = np.vstack([x2,y2]) if merge else y2
    return datas
