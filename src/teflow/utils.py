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

import re
import logging
from io import StringIO
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from configparser import ConfigParser, ExtendedInterpolation, NoSectionError

import numpy as np


_handlers = dict()
def get_root_logger(stdout=True, filename=None, mode='a', level=None, 
                    fmt='[%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    file_fmt='%(asctime)s [%(levelname)s @ %(name)s] %(message)s',
                    file_datafmt='%Y-%m-%d %H:%M:%S',
                    *, pkgname=None):
    '''
    Get root logger object
    '''
    pkgname = pkgname or __package__
    logger = logging.getLogger(pkgname)
    if stdout:
        token = (pkgname, 'console')
        if token in _handlers:
            logger.removeHandler(_handlers[token])
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        _handlers[token] = console
    if filename is not None:
        token = (pkgname, filename)
        if token in _handlers:
            logger.removeHandler(_handlers[token])
        formatter = logging.Formatter(fmt=file_fmt, datefmt=file_datafmt)
        logfile = logging.FileHandler(filename, mode)
        logfile.setFormatter(formatter)
        logger.addHandler(logfile)
        _handlers[token] = logfile
    if level is not None:
        logger.setLevel(level)
    return logger


class AttrDict(OrderedDict):
    '''
    A lightweight extended dictionary class, designed to maintain key order
    and enable attribute-style access via dot notation for a more intuitive
    interaction.

    Examples
    --------
    >>> d = AttrDict({'key1': 'value1'})
    >>> d['key1']
    'value1'
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

        Examples
        --------
        >>> d = AttrDict(b=2, g=3, note='somewhat', c=2)
        >>> popped = d.retain('abcdefg', match_order=True)
        >>> # Equivalent to the above:
        >>> # popped = d.retain(['a','b','c','d','e','f','g'], match_order=True)
        >>> print(d)
        AttrDict([('b', 2), ('c', 2), ('g', 3)])
        >>> print(popped)
        {'note': 'somewhat'}
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
    

class ListDict(AttrDict):
    '''
    Extends :class:`AttrDict` to include methods for manipulating list
    values within the dictionary, such as :meth:`append`, :meth:`extend`,
    :meth:`merge`, and :meth:`sum`.
    '''
    def append(self, obj):
        '''
        Append values from another dictionary to the current instance.

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
            raise ValueError('Only dictionaries object can be appended')
    
    def extend(self, obj):
        '''
        Extend values from another dictionary to the current instance.

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
            raise ValueError('Only dictionaries object can be extended')
        
    @classmethod
    def merge(cls, objs, keys=None, toarray=False):
        '''
        Merge a list of dictionaries to a new instance.

        Parameters
        ----------
        objs : list of dicts
            List of dictionaries to merge.
        keys : list of str, optional
            Keys to consider for merging. If not provided, uses keys
            from the first dictionary.
        toarray : bool, optional
            If True, converts the merged result into a numpy.ndarray.

        Returns
        -------
        ListDict
            Merged dictionary.

        Raises
        ------
        ValueError
            If any object in objs is not a dictionary.

        Examples
        --------
        >>> d1 = {'a': 1, 'b': 2}
        >>> d2 = {'a': 3, 'b': 4}
        >>> merged = ListDict.merge([d1, d2])
        >>> print(merged)
        ListDict([('a', [1, 3]), ('b', [2, 4])])
        '''
        if not all(isinstance(obj, dict) for obj in objs):
            raise ValueError('Dict() objects are required')
        if keys is None:
            keys = objs[0].keys()
        if toarray:
            rst = [(key, np.array([obj[key] for obj in objs])) for key in keys]
        else:
            rst = [(key, [obj[key] for obj in objs]) for key in keys]
        return cls(rst)
    
    @classmethod
    def sum(cls, objs, start=0, keys=None):
        '''
        Sum values from a list of dictionaries to a new instance.

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
        ListDict
            Dictionary with summed values.

        Raises
        ------
        ValueError
            If any object in objs is not a dictionary.

        Examples
        --------
        >>> d1 = {'a': 1, 'b': 2}
        >>> d2 = {'a': 3, 'b': 4}
        >>> summed = ListDict.sum([d1, d2])
        >>> print(summed)
        ListDict([('a', 4), ('b', 6)])
        '''
        if not all(isinstance(obj, dict) for obj in objs):
            raise ValueError('Dict() objects are required')
        if keys is None:
            keys = objs[0].keys()
        rst = [(key, sum((obj[key] for obj in objs), start)) for key in keys]
        return cls(rst)


class CfgParser(ConfigParser):
    '''
    A custom configuration parser class derived from ConfigParser,
    with enhanced features:

    - Allows for case sensitivity in configuration options.
    - Allows whitespace in section names.
    - Predefined converters for `array`, and `list[_XXX]` types.
    - Delimiters: Only the '=' character is accepted for separating keys
      and values in the configuration file, ensuring a consistent format.
    - Comment Prefixes: The '#' character is used to indicate comments,
      allowing for clear and concise configuration files.
    - Inline Comment Prefixes: Inline comments are also indicated with the
      '#' character.
    - Interpolation: Uses `ExtendedInterpolation()` to provide advanced
      string formatting within the configuration file, allowing values
      to reference other values or variables.
    - Prefix Match: Introduce :meth:`pmatch` method to identify
      [SectionName.SectionType] type sections.

    All these settings are initialized to ensure a consistent and clear
    configuration file format, while still allowing for advanced features
    and flexibility. Users can override these settings as per their
    requirements.
    '''
    def __init__(self,
                 allow_section_whitespace=True,
                 case_sensitive=True,
                 **kwargs):
        '''
        Initialize the configuration parser with custom settings.

        Parameters
        ----------
        case_sensitive : bool, optional
            Whether the parsing should be case sensitive. Default is True.
        allow_section_whitespace : bool, optional
            Whether to allow whitespace in section names. Default is True.
        **kwargs
            Additional keyword arguments to pass to the ConfigParser.
        '''
        setting = {
            'delimiters': ('=',),
            'comment_prefixes': ('#',),
            'inline_comment_prefixes': ('#',),
            'interpolation': ExtendedInterpolation(),
            **kwargs,
            'converters': {
                'array': self._parse_array,
                'list': self._parse_list,
                'list_float': self._parse_list_float,
                'list_int': self._parse_list_int,
                'seq': self._parse_seq,
                **kwargs.get('converters', {}),
            },
        }
        super().__init__(**setting)
        if allow_section_whitespace:
            self.SECTCRE = re.compile(r"\[ *(?P<header>[^]]+?) *\]")
        if case_sensitive:
            self.optionxform = lambda x: x

    def pmatch(self, section):
        '''
        Matches sections by prefix SectionName and retrieves content from
        [SectionName.SectionType] or [SectionName] formatted sections.
        '''
        for sect, content in self.items():
            if sect.startswith(section+'.'):
                _, otype = sect.split('.', 1)
                return content, otype
            elif sect == section:
                return content, sect
        else:
            raise NoSectionError(f'{section}.<SectionType>')

    @staticmethod
    def _parse_array(text:str):
        text = text.strip()
        if text.startswith('file:'):
            sdata = text[5:].strip()            # path like
        elif text.startswith('array:'):
            sdata = StringIO(text[6:].strip())  # array like (with prefix)
        else:
            sdata = StringIO(text)              # array like (without prefix)
        return np.loadtxt(sdata, unpack=True, ndmin=2)

    @staticmethod
    def _parse_list(text:str):
        return [item for item in re.split(r'[\s,]+', text) if item]

    @staticmethod
    def _parse_list_float(text:str):
        return [float(item) for item in re.split(r'[\s,]+', text) if item]

    @staticmethod
    def _parse_list_int(text:str):
        return [int(item) for item in re.split(r'[\s,]+', text) if item]

    @staticmethod
    def _parse_seq(text:str):
        result = []
        for part in filter(None, re.split(r'(?<!:)[\s,]+(?!:)', text)):
            seq_parts = list(filter(None, re.split(r'[:\s]+', part)))
            if len(seq_parts) == 1:
                if part:
                    result.append(float(part))
            elif len(seq_parts) == 2:
                start, end = map(float, seq_parts)
                end += 1E-4
                while start < end:
                    result.append(start)
                    start += 1
            elif len(seq_parts) == 3:
                start, step, end = map(float, seq_parts)
                num = (end-start)/step
                if num < 0:
                    continue
                for _ in range(int(num+1E-4)+1):
                    result.append(start)
                    start += step
            else:
                raise ValueError(f'Invalid sequence format: {text}')
        return result


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
