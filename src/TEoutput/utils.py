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
    def merge(cls, objs, keys=None, toArray=False):
        # merge operation: rst = AttrDict.merge([a,b,c,...], [keys])
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
        # sum operation: rst = AttrDict.sum([a,b,c,...], [start])
        if not all(isinstance(obj, dict) for obj in objs):
            raise ValueError('Dict() objects are required')
        if keys is None:
            keys = objs[0].keys()
        rst = {key: sum((obj[key] for obj in objs), start) for key in keys}
        return cls(rst)

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
