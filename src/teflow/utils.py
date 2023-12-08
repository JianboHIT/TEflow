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
from collections import OrderedDict
from collections.abc import Iterable, Sequence

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

class AttrDict(OrderedDict):
    '''
    AttrDict is an extension of Python's collections.OrderedDict()
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
    
    def retain(self, keys, match_order=False):
        '''
        Retain specified keys in the dictionary, optionally matching order.

        Parameters
        ----------
        keys : Iterable or Sequence
            The keys to be retained. When 'match_order' is False,
            'keys' can be any Iterable (like set, list, or string).
            When 'match_order' is True, 'keys' must be a Sequence
            (like list or tuple) to ensure the order of elements.
        match_order : bool, optional
            If True, the order of keys in the resulting dictionary will match 
            the order of keys in the 'keys' parameter. Default is False.

        Returns
        -------
        dict
            A dictionary of the keys that were removed and their values.

        Raises
        ------
        TypeError
            If 'keys' is not an Iterable or not a Sequence when 'match_order'
            is True.
        '''
        if not isinstance(keys, Iterable):
            raise TypeError("'keys' must be an Iterable.")

        popped = {key: self.pop(key) for key in set(self) - set(keys)}

        if match_order:
            if not isinstance(keys, Sequence):
                raise TypeError("When 'match_order' is True, "
                                "'keys' must be a Sequence.")
            indices = {key: index for index, key in enumerate(keys)}
            number = [(indices[key], key) for key in self]

            to_move = []
            sorted_number = sorted(number)
            num_ordered = 0
            for num, key in number:
                if num == sorted_number[num_ordered][0]:
                    num_ordered += 1
                else:
                    to_move.append((num, key))

            for _, key in sorted(to_move):
                self.move_to_end(key)

        return popped
    
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


class ExecWrapper:
    '''
    A utility class for dynamically managing the arguments of a callable or
    a class constructor, allowing for dynamic argument updates and execution.

    Attributes
    ----------
    UNSET
        The unset flag for an argument.

    Parameters
    ----------
    obj : Any
        A callable object or class constructor to manage.
    args : list, optional
        The names of required arguments for the managed object.
    opts : list, optional
        The names of optional arguments for the managed object.
    '''

    UNSET = object()    #: :meta private:

    def __init__(self, obj, args=(), opts=()):
        self.obj = obj
        self.args = {key: self.UNSET for key in args}
        self.opts = {key: self.UNSET for key in opts}

    def update(self, **kwargs):
        '''
        Updates the values of both required and optional arguments.

        Parameters
        ----------
        **kwargs : any, optional
            Keyword arguments to update the values of arguments. Note that
            any invalid arguments passed will be ignored without warning.
        '''
        for key, val in kwargs.items():
            if key in self.args:
                self.args[key] = val
            elif key in self.opts:
                self.opts[key] = val

    def execute(self, **kwargs):
        '''
        Executes the callable or instantiates the class. The arguments provided
        will override the corresponding stored values temporarily, but this
        applies only to the current execution. To make permanent changes to
        the arguments, use the :meth:`update` method.

        Parameters
        ----------
        **kwargs : any, optional
            Keyword arguments to be passed to the callable or constructor.

        Returns
        -------
        Any
            The result of the callable execution or class instantiation.

        Raises
        ------
        ValueError
            If any required argument is missing.
        RuntimeError
            If the execution of the callable or the class instantiation fails.
        '''
        args_tmp = {key:val for key, val in kwargs.items() if key in self.args}
        arguments = {**self.args, **args_tmp}
        unset_args = [key for key, val in arguments.items() if val is self.UNSET]
        if unset_args:
            text = ', '.join(unset_args)
            raise ValueError(f'Argument(s) {text} is necessary but not given')
        arguments.update((k, v) for k, v in self.opts.items() if v is not self.UNSET)
        arguments.update((k, v) for k, v in kwargs.items() if k in self.opts)
        try:
            return self.obj(**arguments)
        except Exception as e:
            raise RuntimeError(f'Failed to execute the object: {e}')


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
