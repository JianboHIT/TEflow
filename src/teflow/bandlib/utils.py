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

