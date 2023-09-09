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

from functools import wraps, partial
import numpy as np


kB_eV = 8.617333262145179e-05   # eV/K
m_e = 9.1093837015e-31          # kg
hbar = 1.054571817e-34          # J.s
e0 = 1.602176634e-19            # C

UNIT = {
    'T': 'K',
    'E': 'eV',
    'N': '1E19 cm^(-3)',
    'U': 'cm^2/(V.s)',
    'C': 'S/cm',
    'S': 'uV/K',
    'K': 'W/(m.K)',
    'L': '1E-8 W.Ohm/K^2',
    'PF': 'uW/(cm.K^2)',
    'ZT': '1',
    'RH': 'cm^3/C',
    'DOS': '1E19 state/(eV.cm^3)',
    'TRS': 'S/cm',
    'HALL': 'S.cm/(V.s)',
}

