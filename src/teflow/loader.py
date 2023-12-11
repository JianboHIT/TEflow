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
from collections.abc import Mapping, Sequence

import numpy as np


class TEdataset(Mapping):
    '''
    A read-only, dict-like interface class for managing thermoelectric data
    (in fact, it is inherited from `collections.abc.Mapping`).
    It primarily facilitates the conversion between various electrical
    conductivity-related physical quantities, such as electrical conductivity,
    resistivity, carrier concentration, and mobility, based on the quantitative
    conversion relationships: C = 10^4 / R = 1.602176634 * N * U.
    Here, C is the conductivity in S/cm, R is the resistivity in uOhm.m,
    N is the carrier concentration in 10^19 cm^(-3), and U is the mobility
    in cm^2/(V.s). The identifiers for these properties can be customized
    (refer to Notes for details). After initialization, properties can be accessed
    directly using '[]' or the :meth:`get` method, similar to a built-in `dict` object.
    The class also offers :meth:`gget` method for batch retrieval of multiple properties.
    Note that :meth:`get` method may compute and return derived properties,
    in contrast to '[]' which strictly returns original data.
    A conventional set of symbols are:

        ====== ========================== ===============
        Prop.  Description                Unit
        ====== ========================== ===============
        T      Temperature                K
        S      Seebeck coefficient        uV/K
        C      Electrical conductivity    S/cm
        R      Electrical resistivity     uOhm.m
        K      Thermal conductivity       W/(m.K)
        N      Carrier concentration      1E19 cm^(-3)
        U      Mobility                   cm^2/(V.s)
        L      Lorenz number              1E-8 W.Ohm/K^2
        X      <Placeholder>              --
        ====== ========================== ===============

    Additionally, a significant feature of this class is handling the
    temperature dependency of material properties, determined by the
    `independent` parameter at initialization. By default, set to True,
    material properties are considered entirely independent, and only
    the property itself is returned when accessed. If set to False,
    the class pairs each material property with a corresponding temperature,
    returning both temperature and property when accessed.

    Parameters
    ----------
    data : list
        A list of thermoelectric data, where each item represents a property and
        is converted to a numpy.ndarray object during initialization.
        It can also be a numpy.ndarray, which will be treated as a regular iterable.
        For instance, in the case of a two-dimensional array, each row represents
        a distinct property.
    group : list or str
        Specifies the identifiers for each item in `data`. This can be a list or
        a string (strings are automatically parsed using the :meth:`parse_group`
        method). The pairing of `group` and `data` behaves similarly to the
        built-in `zip` function, ceasing iteration when the shorter of the two
        is fully iterated. The default :meth:`parse_group` method utilizes the
        regular expression `r'[A-Z][a-z0-9_]?'` (i.e. :attr:`PATTERN` attribute)
        to match each identifier. This method can be overridden to change
        the parsing behavior.
    independent : bool, optional
        Controls the association of material properties with temperature.
        By default, it is set to True, where all properties, including
        temperature, are treated as distinct and independent. If set to False,
        all properties are associated with temperature data.

    Notes
    -----
    The default identifiers for electrical conductivity, resistivity, carrier
    concentration, and mobility are C, R, N, and U, respectively. To alter this
    default setting, override the :attr:`CONDALIAS` attribute before initialization.
    It is a 4-element tuple containing the identifiers for the aforementioned
    properties in the specified order. During initialization, these identifiers
    are bound to four corresponding methods (_getC, _getR, _getN, _getU). To
    modify the specific conversion strategies, you can override these methods
    according to your requirements.
    '''
    _UNSET = object()
    TEMPSYMBOL = 'T'
    IGNORED = {'X',}
    CONDALIAS = ('C', 'R', 'N', 'U')
    PATTERN = r'[A-Z][a-z0-9_]?'
    def __init__(self, data, group, independent=True):
        group = self.parse_group(group)
        if independent:
            self._independent = True
            self._data = {k: np.asarray(v) \
                for k,v in zip(group, data) if k not in self.IGNORED}
            self._temp = {k: None for k in self._data.keys()}
            self._tidx = {k: -1 for k in self._data.keys()}
        else:
            self._independent = False
            self._data = dict()
            self._temp = dict()
            self._tidx = dict()
            idx = 0
            for i, (g, d) in enumerate(zip(group, data)):
                if g == self.TEMPSYMBOL:
                    idx = i
                elif i > idx:
                    if g not in self.IGNORED:
                        self._data[g] = np.asarray(d)
                        self._temp[g] = np.asarray(data[idx])
                        self._tidx[g] = idx
                else:
                    raise ValueError(f"No matched T data was found for '{g}'")
        self._calc_cond = {
            self.CONDALIAS[0]: self._getC,
            self.CONDALIAS[1]: self._getR,
            self.CONDALIAS[2]: self._getN,
            self.CONDALIAS[3]: self._getU,
        }

    def __str__(self):
        name = self.__class__.__name__
        data = self._data
        if self._independent:
            props = [f'{k}#{len(v)}' for k, v in data.items()]
        else:
            TSYM = self.TEMPSYMBOL
            tidx = self._tidx
            props = [f'{k}_{TSYM}{tidx[k]}#{len(v)}' for k, v in data.items()]
        return '{}: {}'.format(name, ', '.join(props))

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        if self._independent:
            return self._data[key]
        else:
            return self._temp[key], self._data[key]

    def get(self, key, default=_UNSET):
        '''
        Fetches a material property, also including temperature
        if not independent. Raises ValueError for any property
        that cannot be fetched if `default` is not set.
        '''
        if key in self._data:
            temp, val = self._temp[key], self._data[key]
        elif key in self._calc_cond:
            temp, val = self._calc_cond[key](default=default)
        else:
            temp, val = default, default
        if val is self._UNSET:
            raise ValueError(f'Failed to fetch {key}')
        return val if self._independent else (temp, val)

    def gget(self, group, default=_UNSET):
        '''
        Batch fetch specified properties from `group` using :meth:`get`,
        and return a list.
        '''
        group = self.parse_group(group)
        return [self.get(g, default=default) for g in group]

    def _getC(self, default=None):
        data = self._data
        temp = self._temp
        tidx = self._tidx
        C, R, N, U = self.CONDALIAS
        if C in data:
            return temp[C], data[C]
        elif R in data:
            return temp[R], 1E4 / data[R]
        elif (N in data) and (U in data) and (tidx[N] == tidx[U]):
            return temp[N], 1.602176634 * data[N] * data[U]
        else:
            return default, default

    def _getR(self, default=None):
        data = self._data
        temp = self._temp
        tidx = self._tidx
        C, R, N, U = self.CONDALIAS
        if R in data:
            return temp[R], data[R]
        elif C in data:
            return temp[C], 1E4 / data[C]
        elif (N in data) and (U in data) and (tidx[N] == tidx[U]):
            return temp[N], 1E4 /(1.602176634 * data[N] * data[U])
        else:
            return default, default

    def _getN(self, default=None):
        data = self._data
        temp = self._temp
        tidx = self._tidx
        C, R, N, U = self.CONDALIAS
        if N in data:
            return temp[N], data[N]
        elif (U in data) and (C in data) and (tidx[U] == tidx[C]):
            return temp[U], data[C] / (1.602176634 * data[U])
        elif (U in data) and (R in data) and (tidx[U] == tidx[R]):
            return temp[U], 1E4 / data[R] / (1.602176634 * data[U])
        else:
            return default, default

    def _getU(self, default=None):
        data = self._data
        temp = self._temp
        tidx = self._tidx
        C, R, N, U = self.CONDALIAS
        if U in data:
            return temp[U], data[U]
        elif (N in data) and (C in data) and (tidx[N] == tidx[C]):
            return temp[N], data[C] / (1.602176634 * data[N])
        elif (N in data) and (R in data) and (tidx[N] == tidx[R]):
            return temp[N], 1E4 / data[R] / (1.602176634 * data[N])
        else:
            return default, default

    @classmethod
    def parse_group(cls, group):
        '''
        Parse `group` into identifiers based on :attr:`PATTERN` attribute.
        '''
        if isinstance(group, str):
            return re.findall(cls.PATTERN, group)
        elif isinstance(group, Sequence):
            return list(group)
        else:
            gtype = type(group).__name__
            raise ValueError(f"Expected a string or a sequence, got {gtype}")

    @classmethod
    def from_file(cls, filename, group, independent=True, delimiter=None):
        '''
        Construct the object or parse data directly from a file.

        Parameters
        ----------
        filename : str
            The name of the file containing the data.
        group : list or str
            Identifiers for each data item, passed directly to the initializer.
            Refer to initializer documentation for more details.
        independent : bool, optional
            Determines the treatment of material properties regarding temperature,
            passed directly to the initializer. See initializer for details.
        delimiter : str, optional
            The delimiter used in the file. Default is None, any consecutive
            whitespaces act as delimiter.

        Returns
        -------
        Any
            Determined by `output` argument.
        '''
        data = np.loadtxt(filename, delimiter=delimiter, unpack=True, ndmin=2)
        return cls(data=data, group=group, independent=independent)
