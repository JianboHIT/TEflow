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
from functools import wraps
from itertools import zip_longest
from collections import namedtuple
from collections.abc import Mapping, Sequence

import numpy as np

from .utils import AttrDict, CfgParser

logger = logging.getLogger(__name__)

_UNSET = object()

PeriodicTable = namedtuple('PeriodicTable', ['XX',
    'H' , 'He',
    'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar',
    'K' , 'Ca', 'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I' , 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt',
    'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
    'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
])  #: :meta private:

AtomicWeight = PeriodicTable(
    _UNSET,         # 0     XX  PlaceHolder
    1.00794,        # 1     H_  Hydrogen
    4.002602,       # 2     He  Helium
    6.941,          # 3     Li  Lithium
    9.012182,       # 4     Be  Beryllium
    10.811,         # 5     B_  Boron
    12.0107,        # 6     C_  Carbon
    14.0067,        # 7     N_  Nitrogen
    15.9994,        # 8     O_  Oxygen
    18.9984032,     # 9     F_  Fluorine
    20.1797,        # 10    Ne  Neon
    22.98976928,    # 11    Na  Sodium
    24.305,         # 12    Mg  Magnesium
    26.9815386,     # 13    Al  Aluminum
    28.0855,        # 14    Si  Silicon
    30.973762,      # 15    P_  Phosphorus
    32.065,         # 16    S_  Sulfur
    35.453,         # 17    Cl  Chlorine
    39.948,         # 18    Ar  Argon
    39.0983,        # 19    K_  Potassium
    40.078,         # 20    Ca  Calcium
    44.955912,      # 21    Sc  Scandium
    47.867,         # 22    Ti  Titanium
    50.9415,        # 23    V_  Vanadium
    51.9961,        # 24    Cr  Chromium
    54.938045,      # 25    Mn  Manganese
    55.845,         # 26    Fe  Iron
    58.933195,      # 27    Co  Cobalt
    58.6934,        # 28    Ni  Nickel
    63.546,         # 29    Cu  Copper
    65.409,         # 30    Zn  Zinc
    69.723,         # 31    Ga  Gallium
    72.64,          # 32    Ge  Germanium
    74.9216,        # 33    As  Arsenic
    78.96,          # 34    Se  Selenium
    79.904,         # 35    Br  Bromine
    83.798,         # 36    Kr  Krypton
    85.4678,        # 37    Rb  Rubidium
    87.62,          # 38    Sr  Strontium
    88.90585,       # 39    Y_  Yttrium
    91.224,         # 40    Zr  Zirconium
    92.90638,       # 41    Nb  Niobium
    95.94,          # 42    Mo  Molybdenum
    98.0,           # 43    Tc  Technetium
    101.07,         # 44    Ru  Ruthenium
    102.9055,       # 45    Rh  Rhodium
    106.42,         # 46    Pd  Palladium
    107.8682,       # 47    Ag  Silver
    112.411,        # 48    Cd  Cadmium
    114.818,        # 49    In  Indium
    118.71,         # 50    Sn  Tin
    121.76,         # 51    Sb  Antimony
    127.6,          # 52    Te  Tellurium
    126.90447,      # 53    I_  Iodine
    131.293,        # 54    Xe  Xenon
    132.9054519,    # 55    Cs  Cesium
    137.327,        # 56    Ba  Barium
    138.90547,      # 57    La  Lanthanum
    140.116,        # 58    Ce  Cerium
    140.90765,      # 59    Pr  Praseodymium
    144.242,        # 60    Nd  Neodymium
    145.0,          # 61    Pm  Promethium
    150.36,         # 62    Sm  Samarium
    151.964,        # 63    Eu  Europium
    157.25,         # 64    Gd  Gadolinium
    158.92535,      # 65    Tb  Terbium
    162.5,          # 66    Dy  Dysprosium
    164.93032,      # 67    Ho  Holmium
    167.259,        # 68    Er  Erbium
    168.93421,      # 69    Tm  Thulium
    173.04,         # 70    Yb  Ytterbium
    174.967,        # 71    Lu  Lutetium
    178.49,         # 72    Hf  Hafnium
    180.94788,      # 73    Ta  Tantalum
    183.84,         # 74    W_  Tungsten
    186.207,        # 75    Re  Rhenium
    190.23,         # 76    Os  Osmium
    192.217,        # 77    Ir  Iridium
    195.084,        # 78    Pt  Platinum
    196.966569,     # 79    Au  Gold
    200.59,         # 80    Hg  Mercury
    204.3833,       # 81    Tl  Thallium
    207.2,          # 82    Pb  Lead
    208.9804,       # 83    Bi  Bismuth
    210.0,          # 84    Po  Polonium
    210.0,          # 85    At  Astatine
    220.0,          # 86    Rn  Radon
    223.0,          # 87    Fr  Francium
    226.0,          # 88    Ra  Radium
    227.0,          # 89    Ac  Actinium
    232.03806,      # 90    Th  Thorium
    231.03588,      # 91    Pa  Protactinium
    238.02891,      # 92    U_  Uranium
    237.0,          # 93    Np  Neptunium
    244.0,          # 94    Pu  Plutonium
    243.0,          # 95    Am  Americium
    247.0,          # 96    Cm  Curium
    247.0,          # 97    Bk  Berkelium
    251.0,          # 98    Cf  Californium
    252.0,          # 99    Es  Einsteinium
    257.0,          # 100   Fm  Fermium
    258.0,          # 101   Md  Mendelevium
    259.0,          # 102   No  Nobelium
    262.0,          # 103   Lr  Lawrencium
    267.0,          # 104   Rf  Rutherfordium
    268.0,          # 105   Db  Dubnium
    269.0,          # 106   Sg  Seaborgium
    270.0,          # 107   Bh  Bohrium
    270.0,          # 108   Hs  Hassium
    278.0,          # 109   Mt  Meitnerium
    281.0,          # 110   Ds  Darmstadtium
    282.0,          # 111   Rg  Roentgenium
    285.0,          # 112   Cn  Copernicium
    286.0,          # 113   Nh  Nihonium
    289.0,          # 114   Fl  Flerovium
    290.0,          # 115   Mc  Moscovium
    293.0,          # 116   Lv  Livermorium
    294.0,          # 117   Ts  Tennessine
    294.0,          # 118   Og  Oganesson
)
'''
A constant namedtuple storing atomic weights in AMU (from pymatgen).
This namedtuple can be utilized wherever regular tuples are employed,
and it adds the capability to access fields by element name in addition
to position index. Furthermore, a placeholder is stored at index 0 to
align tuple indices with atomic numbers.

:meta hide-value:

Examples
--------
>>> print(AtomicWeight.Ne)
20.1797
>>> print(AtomicWeight[10])
20.1797
>>> print(getattr(AtomicWeight, 'Ne'))
20.1797
'''

AtomicRadius = PeriodicTable(
    _UNSET,         # 0     XX  PlaceHolder
    0.53,           # 1     H_  Hydrogen
    0.31,           # 2     He  Helium
    1.67,           # 3     Li  Lithium
    1.12,           # 4     Be  Beryllium
    0.87,           # 5     B_  Boron
    0.67,           # 6     C_  Carbon
    0.56,           # 7     N_  Nitrogen
    0.48,           # 8     O_  Oxygen
    0.42,           # 9     F_  Fluorine
    0.38,           # 10    Ne  Neon
    1.90,           # 11    Na  Sodium
    1.45,           # 12    Mg  Magnesium
    1.18,           # 13    Al  Aluminum
    1.11,           # 14    Si  Silicon
    0.98,           # 15    P_  Phosphorus
    0.88,           # 16    S_  Sulfur
    0.79,           # 17    Cl  Chlorine
    0.71,           # 18    Ar  Argon
    2.43,           # 19    K_  Potassium
    1.94,           # 20    Ca  Calcium
    1.84,           # 21    Sc  Scandium
    1.76,           # 22    Ti  Titanium
    1.71,           # 23    V_  Vanadium
    1.66,           # 24    Cr  Chromium
    1.61,           # 25    Mn  Manganese
    1.56,           # 26    Fe  Iron
    1.52,           # 27    Co  Cobalt
    1.49,           # 28    Ni  Nickel
    1.45,           # 29    Cu  Copper
    1.42,           # 30    Zn  Zinc
    1.36,           # 31    Ga  Gallium
    1.25,           # 32    Ge  Germanium
    1.14,           # 33    As  Arsenic
    1.03,           # 34    Se  Selenium
    0.94,           # 35    Br  Bromine
    0.88,           # 36    Kr  Krypton
    2.65,           # 37    Rb  Rubidium
    2.19,           # 38    Sr  Strontium
    2.12,           # 39    Y_  Yttrium
    2.06,           # 40    Zr  Zirconium
    1.98,           # 41    Nb  Niobium
    1.90,           # 42    Mo  Molybdenum
    1.83,           # 43    Tc  Technetium
    1.78,           # 44    Ru  Ruthenium
    1.73,           # 45    Rh  Rhodium
    1.69,           # 46    Pd  Palladium
    1.65,           # 47    Ag  Silver
    1.61,           # 48    Cd  Cadmium
    1.56,           # 49    In  Indium
    1.45,           # 50    Sn  Tin
    1.33,           # 51    Sb  Antimony
    1.23,           # 52    Te  Tellurium
    1.15,           # 53    I_  Iodine
    1.08,           # 54    Xe  Xenon
    2.98,           # 55    Cs  Cesium
    2.53,           # 56    Ba  Barium
    2.26,           # 57    La  Lanthanum
    2.10,           # 58    Ce  Cerium
    2.47,           # 59    Pr  Praseodymium
    2.06,           # 60    Nd  Neodymium
    2.05,           # 61    Pm  Promethium
    2.38,           # 62    Sm  Samarium
    2.31,           # 63    Eu  Europium
    2.33,           # 64    Gd  Gadolinium
    2.25,           # 65    Tb  Terbium
    2.28,           # 66    Dy  Dysprosium
    2.26,           # 67    Ho  Holmium
    2.26,           # 68    Er  Erbium
    2.22,           # 69    Tm  Thulium
    2.22,           # 70    Yb  Ytterbium
    2.17,           # 71    Lu  Lutetium
    2.08,           # 72    Hf  Hafnium
    2.00,           # 73    Ta  Tantalum
    1.93,           # 74    W_  Tungsten
    1.88,           # 75    Re  Rhenium
    1.85,           # 76    Os  Osmium
    1.80,           # 77    Ir  Iridium
    1.77,           # 78    Pt  Platinum
    1.74,           # 79    Au  Gold
    1.71,           # 80    Hg  Mercury
    1.56,           # 81    Tl  Thallium
    1.54,           # 82    Pb  Lead
    1.43,           # 83    Bi  Bismuth
    1.35,           # 84    Po  Polonium
    1.27,           # 85    At  Astatine
    1.2,            # 86    Rn  Radon
    _UNSET,         # 87    Fr  Francium
    _UNSET,         # 88    Ra  Radium
    _UNSET,         # 89    Ac  Actinium
    _UNSET,         # 90    Th  Thorium
    _UNSET,         # 91    Pa  Protactinium
    _UNSET,         # 92    U_  Uranium
    _UNSET,         # 93    Np  Neptunium
    _UNSET,         # 94    Pu  Plutonium
    _UNSET,         # 95    Am  Americium
    _UNSET,         # 96    Cm  Curium
    _UNSET,         # 97    Bk  Berkelium
    _UNSET,         # 98    Cf  Californium
    _UNSET,         # 99    Es  Einsteinium
    _UNSET,         # 100   Fm  Fermium
    _UNSET,         # 101   Md  Mendelevium
    _UNSET,         # 102   No  Nobelium
    _UNSET,         # 103   Lr  Lawrencium
    _UNSET,         # 104   Rf  Rutherfordium
    _UNSET,         # 105   Db  Dubnium
    _UNSET,         # 106   Sg  Seaborgium
    _UNSET,         # 107   Bh  Bohrium
    _UNSET,         # 108   Hs  Hassium
    _UNSET,         # 109   Mt  Meitnerium
    _UNSET,         # 110   Ds  Darmstadtium
    _UNSET,         # 111   Rg  Roentgenium
    _UNSET,         # 112   Cn  Copernicium
    _UNSET,         # 113   Nh  Nihonium
    _UNSET,         # 114   Fl  Flerovium
    _UNSET,         # 115   Mc  Moscovium
    _UNSET,         # 116   Lv  Livermorium
    _UNSET,         # 117   Ts  Tennessine
    _UNSET,         # 118   Og  Oganesson
)
'''
A constant namedtuple storing atomic radii in Angstroms for elements No.1-86.
Refer to :attr:`AtomicWeight` for usage.

References:

[1] Clementi, E., & Raimondi, D. L. (1963). Atomic screening constants
from SCF functions. The Journal of Chemical Physics, 38(11), 2686-2689.

[2] Clementi, E., Raimondi, D. L., & Reinhardt, W. P. (1967). Atomic
screening constants from SCF functions. II. Atoms with 37 to 86 electrons.
Journal of Chemical Physics, 47(4), 1300-1307.

:meta hide-value:
'''


class Compound(AttrDict):
    '''
    Represents the chemical composition of a compound using an ordered
    dictionary.
    '''
    @property
    def natom(self):
        '''
        int or float: The total number of atoms in the compound. Returns a
        rounded integer if it is within a 1E-8 tolerance of an integer,
        otherwise the original exact float.
        '''
        n = sum(self.values())
        n_round = round(n)
        return n_round if abs(n-n_round) < 1E-8 else n

    @property
    def weights(self):
        '''
        AttrDict: A dictionary mapping elements to atomic weights.
        '''
        out = AttrDict()
        vaild_keys = set(AtomicWeight._fields)
        for key in self:
            if key not in vaild_keys:
                raise KeyError(f"Unknown element '{key}' in {str(self)}.")
            out[key] = getattr(AtomicWeight, key)
        return out

    @property
    def weight_ave(self):
        '''
        float: Average atomic weight of the compound.
        '''
        wtot = sum(n*w for n, w in zip(self.values(), self.weights.values()))
        return wtot / self.natom

    def replace(self, old_atom, new_atom, match_order=False):
        '''
        Replaces an existing atom with a new atom in the compound.
        '''
        if old_atom not in self:
            raise KeyError(f"Old atom '{old_atom}' not found.")
        if new_atom in self:
            raise KeyError(f"New atom '{new_atom}' already exists.")
        if match_order:
            poped = []
            while True:
                key, val = self.popitem()
                if key == old_atom:
                    poped.append([new_atom, val])
                    break
                else:
                    poped.append([key, val])
            self.update(reversed(poped))
        else:
            self[new_atom] = self.pop(old_atom)

    def __str__(self):
        return self.to_string()

    def to_string(self, fmt='%.9g', style='join', omit_one=True):
        '''
        Generates a string representation of the compound, with support for
        'join' (default), 'split', and 'originlab' styles at present.
        '''
        dsps = []

        if style in {'origin', 'originlab', 'originpro',}:
            for name, num in self.items():
                num_ = fmt % num
                if omit_one and num_ == '1':
                    dsps.append(name)
                else:
                    dsps.append(f'{name}\\-({num_})')
            return ''.join(dsps)

        for name, num in self.items():
            num_ = fmt % num
            if omit_one and num_ == '1':
                dsps.append(name)
            else:
                dsps.append(f'{name}{num_}')
        if style == 'join':
            return ''.join(dsps)
        else:
            return ' '.join(dsps)

    @classmethod
    def from_string(cls, formula:str, skip_unknown=False, raise_unknown=False):
        '''Instance construction from a chemical formula.'''
        comp = cls()
        vaild_keys = set(AtomicWeight._fields)
        dcp = re.compile(r'(?P<name>[A-Z][a-z]?)[ _\-\(\\]*(?P<num>\d*\.?\d*)')
        for m in dcp.finditer(formula):
            name = m.group('name')
            if name not in vaild_keys:
                if raise_unknown:
                    raise ValueError(f'Unknown element {name} in {formula}.')
                elif skip_unknown:
                    continue
            num_ = m.group('num') or '1'
            comp[name] = float(num_) if '.' in num_ else int(num_)
        return comp


class TEdatasetBase(Mapping):
    Q_CHARGE = 1.602176634
    TEMP_SYMBOL = 'T'
    IGNORED = {'X',}
    CONDALIAS = ('C', 'R', 'N', 'U')
    PROP_PATTERN = r'[A-Z][a-z0-9_]?'
    def __init__(self, data, group):
        '''
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
            regular expression `r'[A-Z][a-z0-9_]?'` (i.e. :attr:`PROP_PATTERN` attribute)
            to match each identifier. This method can be overridden to change
            the parsing behavior.
        '''
        group = self.parse_group(group)
        self._initialize_data(data, group)
        self._calc_cond = {
            self.CONDALIAS[0]: self._getC,
            self.CONDALIAS[1]: self._getR,
            self.CONDALIAS[2]: self._getN,
            self.CONDALIAS[3]: self._getU,
        }

    def _initialize_data(self, data, group):
        self._data = ...
        self._temp = ...
        self._tidx = ...
        raise NotImplementedError

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        raise NotImplementedError

    def get(self, key, default=_UNSET):
        '''
        Fetches a material property, also including temperature
        if not independent. Raises ValueError for missing properties
        if `default` is not specified.
        '''
        raise NotImplementedError

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
            return temp[N], self.Q_CHARGE * data[N] * data[U]
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
            return temp[N], 1E4 /(self.Q_CHARGE * data[N] * data[U])
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
            return temp[U], data[C] / (self.Q_CHARGE * data[U])
        elif (U in data) and (R in data) and (tidx[U] == tidx[R]):
            return temp[U], 1E4 / data[R] / (self.Q_CHARGE * data[U])
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
            return temp[N], data[C] / (self.Q_CHARGE * data[N])
        elif (N in data) and (R in data) and (tidx[N] == tidx[R]):
            return temp[N], 1E4 / data[R] / (self.Q_CHARGE * data[N])
        else:
            return default, default

    @classmethod
    def parse_group(cls, group):
        '''
        Parse `group` into identifiers based on :attr:`PROP_PATTERN` attribute.
        '''
        if isinstance(group, str):
            return re.findall(cls.PROP_PATTERN, group)
        elif isinstance(group, Sequence):
            return list(group)
        else:
            gtype = type(group).__name__
            raise ValueError(f"Expected a string or a sequence, got {gtype}")

    @staticmethod
    def parse_datafile(filename, delimiter=None):
        '''
        Parse a block-structured data file. By default, any consecutive
        whitespaces act as delimiter.
        '''
        data = []
        dcp = re.compile(r'^ *(?P<rowline>[^#]+?) *(?=#|$)')
        dcv = re.compile(delimiter or r'\s+')
        with open(filename, 'r') as f:
            for line in f:
                m = dcp.match(line.rstrip())
                if not m:
                    continue
                row = []
                for item in dcv.split(m.group('rowline')):
                    try:
                        val = float(item)
                    except ValueError:
                        val = np.nan
                    finally:
                        row.append(val)
                data.append(row)
        return list(zip_longest(*data, fillvalue=np.nan))

    @classmethod
    def from_file(cls, filename, group, delimiter=None):
        '''
        Construct the object or parse data directly from a file.

        Parameters
        ----------
        filename : str
            The name of the file containing the data.
        group : list or str
            Identifiers for each data item, passed directly to the initializer.
            Refer to initializer documentation for more details.
        delimiter : str, optional
            The delimiter used in the file. Default is None, any consecutive
            whitespaces act as delimiter.

        Returns
        -------
        object
            An instance of this class.
        '''
        raise NotImplementedError


class TEdataset(TEdatasetBase):
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
    def _initialize_data(self, data, group):
        self._data = {k: np.asarray(v) \
            for k,v in zip(group, data) if k not in self.IGNORED}
        self._temp = {k: None for k in self._data.keys()}
        self._tidx = {k: -1 for k in self._data.keys()}

    def __str__(self):
        data = self._data
        props = [f'{k}#{len(v)}' for k, v in data.items()]
        return '{}: {}'.format(self.__class__.__name__, ', '.join(props))

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=_UNSET):
        if key in self._data:
            val = self._data[key]
        elif key in self._calc_cond:
            _, val = self._calc_cond[key](default=default)
        else:
            val = default
        if val is _UNSET:
            raise ValueError(f'Failed to fetch {key}')
        return val

    @classmethod
    def from_file(cls, filename, group, delimiter=None):
        data = cls.parse_datafile(filename, delimiter)
        nan_filter = lambda x: x[np.isfinite(x)]
        data = [nan_filter(np.array(line)) for line in data]
        return cls(data=data, group=group)


class TEdataset2(TEdatasetBase):
    '''
    Similar to :class:`TEdataset`, this class incorporates temperature
    dependencies. It automatically links material properties to their
    respective temperature data, allowing users to access both the property
    value and its associated temperature seamlessly.
    '''
    def _initialize_data(self, data, group):
        self._data = dict()
        self._temp = dict()
        self._tidx = dict()
        idx = 0
        for i, (g, d) in enumerate(zip(group, data)):
            if g == self.TEMP_SYMBOL:
                idx = i
            elif i > idx:
                if g not in self.IGNORED:
                    self._data[g] = np.asarray(d)
                    self._temp[g] = np.asarray(data[idx])
                    self._tidx[g] = idx
            else:
                raise ValueError(f"No matched T data was found for '{g}'")

    def __str__(self):
        data = self._data
        TSYM = self.TEMP_SYMBOL
        tidx = self._tidx
        props = [f'{k}_{TSYM}{tidx[k]}#{len(v)}' for k, v in data.items()]
        return '{}: {}'.format(self.__class__.__name__, ', '.join(props))

    def __getitem__(self, key):
        return self._temp[key], self._data[key]

    def get(self, key, default=_UNSET):
        if key in self._data:
            temp, val = self._temp[key], self._data[key]
        elif key in self._calc_cond:
            temp, val = self._calc_cond[key](default=default)
        else:
            temp, val = default, default
        if val is _UNSET:
            raise ValueError(f'Failed to fetch {key}')
        return temp, val

    @classmethod
    def from_file(cls, filename, group, delimiter=None):
        if delimiter is None:
            delimiter = r'[\t;,]'
        data = cls.parse_datafile(filename, delimiter)
        TSYM = cls.TEMP_SYMBOL
        groups = cls.parse_group(group)
        nan_filter = lambda x: x[:, np.all(np.isfinite(x), axis=0)]
        itemp = [i for i, s in enumerate(groups) if s == TSYM]
        iprop = itemp[1:] + [len(groups),]
        if not itemp:
            raise ValueError(f"Failed to find identifier '{TSYM}' in group")
        if len(itemp) == 1:
            data[itemp[0]:] = nan_filter(np.vstack(data[itemp[0]:]))
        else:
            for p, q in zip(itemp, iprop):
                data[p:q] = nan_filter(np.vstack(data[p:q]))
        return cls(data=data, group=group)


INSTRMETA = AttrDict()

def _registerInstr(name:str, *keys:str):
    def _decorator(func):
        @wraps(func)
        def _wrapedParser(text:str):
            text = text.strip()
            if text.startswith('file:'):
                with open(text[5:].strip(), 'r', errors='ignore') as f:
                    text = f.read()
            return func(text)
        INSTRMETA[name] = (_wrapedParser, keys)
        return _wrapedParser
    return _decorator

def _parse_CTA_ZEM(name:str, text:str, identifiers):
    rawlines = text.split('\n')
    for idx, line in enumerate(rawlines):
        if all(i in line for i in identifiers):
            logger.debug('Find identifiers at line #%d of %s', idx+1, name)
            break
    else:
        logger.debug('Identifiers: (%s)', ', '.join(identifiers))
        raise IOError(f'Failed to locate identifiers when parsing {name} file.')

    dataT, dataR, dataS = [], [], []
    for line in rawlines[idx+1:]:
        values = line.strip().split('\t')
        if len(values) < 5:
            continue
        logger.debug('  %s', '  '.join(values[i] for i in [0, 1, 4]))
        dataT.append(float(values[0]))
        dataR.append(float(values[1]))
        dataS.append(float(values[4]))
    return AttrDict(T=np.array(dataT), R=np.array(dataR), S=np.array(dataS))

@_registerInstr('CTA', 'T', 'R', 'S')
def parseCTA(text:str):
    identifiers = ('Temperature', 'Resistivity', 'Seebeck coefficient')
    datax = _parse_CTA_ZEM('CTA', text, identifiers)
    datax['T'] += 273   # degC  --> Kelvin
    return datax

@_registerInstr('ZEM', 'T', 'R', 'S')
def parseZEM(text:str):
    identifiers = ('Measurement temp', 'Resistivity', 'Seebeck coeff')
    datax = _parse_CTA_ZEM('ZEM', text, identifiers)
    datax['T'] += 273   # degC  --> Kelvin
    datax['R'] *= 1E6   # Ohm.m --> uOhm.m
    datax['S'] *= 1E6   # V/K   --> uV/K
    return datax

@_registerInstr('LFA', 'T', 'A')
def parseLFA(text:str):
    rawlines = text.split('\n')
    identifier = '##Results'
    for idx, line in enumerate(rawlines):
        if line.startswith(identifier):
            logger.debug('Find identifier at line #%d of %s', idx+1, 'LFA457')
            idx += 1    # skip title line
            break
    else:
        logger.debug('Identifiers: ', identifier)
        raise IOError(f'Failed to locate identifier when parsing LFA457 file.')

    dataT, dataA = [], []
    for line in rawlines[idx+1:]:
        line =line.strip()
        if line.startswith('#Mean'):
            values = [x for x in line.split(',') if x]
            if len(values) < 3:
                continue
            logger.debug(f'  {values[1]}, {values[2]}')
            dataT.append(float(values[1])+273)  # degC  --> Kelvin
            dataA.append(float(values[2]))      # Diffusivity, in mm^2/s
    if dataT and dataA:
        return AttrDict(T=np.array(dataT), A=np.array(dataA))
    else:
        logger.debug('Failed to locate #Mean lines, try to parse each line')

    try:
        for line in rawlines[idx+1:]:
            line =line.strip()
            values = line.split(',')
            if len(values) < 3:
                continue
            logger.debug(f'  {values[1]}, {values[2]}')
            dataT.append(float(values[1])+273)  # degC  --> Kelvin
            dataA.append(float(values[2]))      # Diffusivity, in mm^2/s
    except Exception as e:
        raise IOError(f'Failed to parse LFA file: {e}')
    else:
        return AttrDict(T=np.array(dataT), A=np.array(dataA))

def parse_CpT(weight, *args, model='Dulong-Petit'):
    '''Produce a temperature-dependent heat capacity function Cp(T)'''
    DP = 3 * 8.31446261815324   # 3R
    if model in ('Dulong-Petit', 'DP', '3R'):
        return lambda T: DP / weight * np.ones_like(T)
    elif model in ('Mg3Sb2',):
        raise NotImplementedError
        # weight, *_ = args
        # return lambda T: DP / weight * np.ones_like(T)
    else:
        raise ValueError(f"Invalid model: {model}")

def parse_TEfile(filename, specify=None):
    '''Parse thermoelectric data from a config file'''
    config = CfgParser()
    with open(filename, 'r') as f:
        config.read_file(f)
        logger.info(f'Read configuration from {filename}')

    entry = config['entry']
    logger.debug('Found entry section')

    if specify is not None:
        entry.update(specify)
        logger.debug('Update specify setting to entry:\n  %s' % specify)

    dsp = "Parameter '{}' is required in entry section!"
    data = []
    group = []

    # parse composition (comp)
    if 'sample' in entry:
        comp = Compound.from_string(entry.get('sample'), skip_unknown=True)
        logger.info(f'Compostion: {comp.to_string(style="split")}')
        weight = comp.weight_ave
        logger.info(f'Average Atomic Weight: {weight:.7g} au')
    else:
        raise ValueError("Parameter 'sample' is required for evaluating "
                         "atomic weight in entry section!")

    # parse electrical transport properties
    Found = []
    if 'ZEM' in entry:
        Found.append('ZEM')
    if 'CTA' in entry:
        Found.append('CTA')

    NumFound = len(Found)
    if NumFound == 0:
        raise ValueError(dsp.format("CTA' or 'ZEM"))
    elif NumFound > 1:
        raise ValueError(f"Overlapped parameters: {', '.join(Found)}")

    instr = Found[0]
    instr_file = entry.get(instr)
    logger.debug(f'{instr} file: {instr_file}')
    datax = INSTRMETA[instr][0](instr_file)
    logger.info(f'> Parsing {instr} data: {",".join(datax.keys())}')

    for key, val in datax.items():
        data.append(val)
        group.append(key)

    # parse thermal transport properties
    if 'LFA' in entry:
        lfa_file = entry.get('LFA')
        logger.debug(f'LFA file: {lfa_file}')
        datax = parseLFA(lfa_file)
        logger.info(f'> Parsing LFA data: {",".join(datax.keys())}')
    else:
        raise ValueError(dsp.format('LFA'))

    # parse density
    if 'density' in entry:
        density = eval(entry.get('density'), {"__builtins__": None}, {})
        logger.info(f'Density: {density:.4g} g/cm^3')
    else:
        raise ValueError(dsp.format('density'))

    # parse heat capacity (Cp)
    cpx = entry.get('Cp', '@Dulong-Petit').strip()
    if '@' in cpx:
        argx, model = cpx.rsplit('@', maxsplit=1)
        model = model.strip()
        args = [float(x) for x in re.split(r'[\s,]+', argx) if x]
        Cp_T = parse_CpT(weight, *args, model=model)
        logger.info(f'Cp: {model} model')
        Cp = Cp_T(datax['T'])
        logger.debug(f'{Cp}')
    else:
        Cp = np.asarray(entry.getlist_float('Cp'))
        if len(Cp) != len(datax['T']):
            raise ValueError(f"Cp values should have the same length as T")
        logger.info(f'Cp: {Cp}')

    # calculate thermal conductivity (kappa)
    kappa = datax['A'] * density * Cp
    logger.debug(f'Kappa: {kappa}')
    data.extend([datax['T'], datax['A'], kappa])
    group.extend(['T', 'A', 'K'])
    return TEdataset2(data=data, group=group)
