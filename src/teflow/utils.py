#   Copyright 2023 Jianbo ZHU
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

import logging
from pathlib import PurePath

import numpy as np


def get_pkg_name():
    '''
    Get package name
    '''
    return __name__.split('.')[0]

def get_root_logger(stdout=True, filename=None, mode='a', level=None, 
                    fmt='[%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    file_fmt='%(asctime)s [%(levelname)s @ %(name)s] %(message)s',
                    file_datafmt='%Y-%m-%d %H:%M:%S'):
    '''
    Get root logger object
    '''
    logger = logging.getLogger(get_pkg_name())
    if stdout:
        console = get_logger_handler(kind='console', 
                                     fmt=fmt, 
                                     datefmt=datefmt)
        logger.addHandler(console)
    if filename is not None:
        fh = get_logger_handler(kind='file',
                                fmt=file_fmt,
                                datefmt=file_datafmt,
                                filename=filename,
                                mode=mode)
        logger.addHandler(fh)
    if level is not None:
        logger.setLevel(level)
    return logger

def get_logger_handler(kind='CONSOLE', fmt=None, datefmt=None, filename='log.txt', mode='a'):
    '''
    Get kinds of handler of logging
    '''
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    kind = kind.lower()
    if kind in {'cons', 'console', 'stream'}:
        handler = logging.StreamHandler()
    elif kind in {'file', 'logfile'}:
        handler = logging.FileHandler(filename, mode)
    else:
        raise ValueError('The kind of handler is invaild.')
    handler.setFormatter(formatter)
    return handler

class AttrDict(dict):
    '''
    AttrDict is an extension of Python's standard dictionary (dict)
    that provides additional functionalities. It supports dot access
    mode for direct attribute access to dictionary keys, as well as
    other operations such as :meth:`append`, :meth:`extend`,
    :meth:`merge`, and :meth:`sum`.

    With AttrDict, you can directly access dictionary values using
    dot notation.

    Examples
    --------
    >>> d = AttrDict({'key1': 'value1'})
    >>> d.key1
    'value1'

    Notes
    -----
    - Dot access mode only supports accessing existing keys.
      Setting values using dot notation is not supported.
    - This class may not handle nested dictionaries in dot access mode.
    '''
    def __getattr__(self, attr):
        # define dot access mode like d.key
        return self[attr]
    
    def append(self, obj):
        '''
        Append values from another dictionary to the current AttrDict.

        Parameters
        ----------
        obj : dict
            Dictionary from which values are to be appended.

        Raises
        ------
        ValueError
            If the object to append is not a dictionary.
        '''
        if isinstance(obj, dict):
            for key in self.keys():
                self[key].append(obj[key])
        else:
            raise ValueError('Only Dict() object can be appended')
    
    def extend(self, obj):
        '''
        Extend values from another dictionary to the current AttrDict.

        Parameters
        ----------
        obj : dict
            Dictionary from which values are to be extended.

        Raises
        ------
        ValueError
            If the object to extend is not a dictionary.
        '''
        if isinstance(obj, dict):
            for key in self.keys():
                self[key].extend(obj[key])
        else:
            raise ValueError('Only Dict() object can be extended')
        
    @classmethod
    def merge(cls, objs, keys=None, toArray=False):
        '''
        Merge a list of dictionaries.

        Parameters
        ----------
        objs : list of dicts
            List of dictionaries to merge.
        keys : list of str, optional
            Keys to consider for merging. If not provided, uses keys
            from the first dictionary.
        toArray : bool, optional
            If True, converts the merged result into a numpy.ndarray.

        Returns
        -------
        AttrDict
            Merged dictionary.

        Raises
        ------
        ValueError
            If any object in objs is not a dictionary.

        Examples
        --------
        >>> d1 = AttrDict({'a': 1, 'b': 2})
        >>> d2 = AttrDict({'a': 3, 'b': 4})
        >>> merged = AttrDict.merge([d1, d2])
        >>> print(merged)
        {'a': [1, 3], 'b': [2, 4]}
        '''
        if not all(isinstance(obj, dict) for obj in objs):
            raise ValueError('Dict() objects are required')
        if keys is None:
            keys = objs[0].keys()
        if toArray:
            rst = {key: np.array([obj[key] for obj in objs]) for key in keys}
        else:
            rst = {key: [obj[key] for obj in objs] for key in keys}
        return cls(rst)
    
    @classmethod
    def sum(cls, objs, start=0, keys=None):
        '''
        Sum values from a list of dictionaries.

        Parameters
        ----------
        objs : list of dicts
            List of dictionaries to sum.
        start : int or float, optional
            Starting value for sum. Default is 0.
        keys : list of str, optional
            Keys to consider for summing. If not provided, uses keys
            from the first dictionary.

        Returns
        -------
        AttrDict
            Dictionary with summed values.

        Raises
        ------
        ValueError
            If any object in objs is not a dictionary.

        Examples
        --------
        >>> d1 = AttrDict({'a': 1, 'b': 2})
        >>> d2 = AttrDict({'a': 3, 'b': 4})
        >>> summed = AttrDict.sum([d1, d2])
        >>> print(summed)
        {'a': 4, 'b': 6}
        '''
        if not all(isinstance(obj, dict) for obj in objs):
            raise ValueError('Dict() objects are required')
        if keys is None:
            keys = objs[0].keys()
        rst = {key: sum((obj[key] for obj in objs), start) for key in keys}
        return cls(rst)

class Metric():
    '''
    Metric class provides functionalities for computing various error
    metrics. This class can be initialized with a specific kind of
    metric (such as 'MSE', 'RMSE') and later be called directly to
    compute the error based on the initialized metric kind. Additionally,
    each metric kind can also be used directly as a static method.

    Examples
    --------
    >>> metric = Metric('MSE')
    >>> y = [1, 2, 3]
    >>> y2 = [2, 4, 5]
    >>> metric(y, y2)
    3.0
    >>> Metric.MSE(y, y2)
    3.0
    '''
    kinds = {'MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE'}  #: :meta private:
    def __init__(self, kind='MSE'):
        '''
        Parameters
        ----------
        kind : str, optional
            Kind of metric to be used. Default is 'MSE'.
        '''
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
        Mean-Square Error:
        
        .. math::
        
            \\frac{1}{n} \\sum_{i=1}^{n} (y_2[i] - y[i])^2
        '''
        diff = np.array(y2) - np.array(y)
        v2 = np.mean(np.square(diff), axis=axis)
        return v2
    
    @staticmethod
    def RMSE(y, y2, axis=-1):
        '''
        Root-Mean-Square error:
        
        .. math:: 
        
            \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (y_2[i] - y[i])^2}
        '''
        diff = np.array(y2) - np.array(y)
        v2 = np.mean(np.square(diff), axis=axis)
        return np.sqrt(v2)
    
    @staticmethod
    def MAE(y, y2, axis=-1):
        '''
        Mean Absolute Error:
        
        .. math:: 
            
            \\frac{1}{n} \\sum_{i=1}^{n} |y_2[i] - y[i]|
        '''
        diff = np.array(y2) - np.array(y)
        v = np.mean(np.absolute(diff), axis=axis)
        return v
    
    @staticmethod
    def MAPE(y, y2, axis=-1):
        '''
        Mean Absolute Percentage Error:
        
        .. math:: 
        
            \\frac{1}{n} \\sum_{i=1}^{n} 
                \\left| \\frac{y_2[i] - y[i]}{y[i]} \\right|
        '''
        rdiff = np.array(y2)/np.array(y)-1
        v = np.mean(np.absolute(rdiff), axis=axis)
        return v
    
    @staticmethod
    def SMAPE(y, y2, axis=-1):
        '''
        Symmetric Mean Absolute Percentage Error:
        
        .. math:: 
        
            \\frac{2}{n} \\sum_{i=1}^{n}
                \\frac{|y_2[i] - y[i]|}{|y_2[i]| + |y[i]|}
        '''
        y = np.array(y)
        y2 = np.array(y2)
        diff = np.absolute(y2-y)
        sum_ = np.absolute(y2)+np.absolute(y)
        v = 2.0 * np.mean(diff/sum_, axis=axis)
        return v

def suffixed(outputname, inputname, suffix, withparent=False):
    '''
    Append suffix to inputname if outputname is absent, otherwise return itself. 
    '''
    if outputname:
        return outputname
    else:
        p = PurePath(inputname)
        if p.suffix:
            out = f'{p.stem}_{suffix}{p.suffix}'
        else:
            out = f'{p.stem}_{suffix}'
        if withparent:
            return str(p.with_name(out))
        else:
            return out

def purify(fp, chars=None, usecols=None, sep=None):
    '''
    Remove #-type comments and strip line, then return a built-in
    `filter`/`map` object.
    '''
    if usecols:
        _fetch = lambda line: line.split('#', 1)[0].strip(chars).split(sep)
        _pick = lambda items: ' '.join(items[i] for i in usecols)
        return map(_pick, filter(None, map(_fetch, fp)))
    else:
        _fetch = lambda line: line.split('#', 1)[0].strip(chars)
        return filter(None, map(_fetch, fp))
