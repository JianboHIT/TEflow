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

import logging
from abc import ABC, abstractmethod
from scipy.integrate import romb
from scipy.optimize import root_scalar
import numpy as np

from .mathext import vquad, fermidirac
from .utils import AttrDict, ExecWrapper, CfgParser

logger = logging.getLogger(__name__)

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
    'RH': 'cm^3/C',
    'DOS': '1E19 state/(eV.cm^3)',
    'TRS': 'S/cm',
    'HALL': 'S.cm/(V.s)',
}
'''
====== ===================== =====================================
Key    Value                 Notes
====== ===================== =====================================
T      K                     Temperature
E      eV                    Energy
N      1E19 cm^(-3)          Carrier concentration
U      cm^2/(V.s)            Mobility
C      S/cm                  Conductivity
S      uV/K                  Seebeck coefficient
K      W/(m.K)               Thermal conductivity
L      1E-8 W.Ohm/K^2        Lorenz number
PF     uW/(cm.K^2)           Power factor
RH     cm^3/C                Hall coefficient
DOS    1E19 state/(eV.cm^3)  Density of states
TRS    S/cm                  Transport distribution function
HALL   S.cm/(V.s)            Hall transport distribution function
====== ===================== =====================================

:meta hide-value:
'''

kB_eV = 8.617333262145179e-05   #: Unit:: eV/K
m_e = 9.1093837015e-31          #: Unit:: kg
hbar = 1.054571817e-34          #: Unit:: J.s
q = 1.602176634e-19             #: Unit:: C


def romb_dfx(func, EF, T, k=0, ndiv=6, eps=1E-10):
    '''
    Calucate semi-infinity Fermi-Dirac integrals with dfx-type weight.

    Parameters
    ----------
    func : callable
        A callable object with form func(E, T), where E is energy in eV,
        and T is temperature in Kelvin. It is assumed to it support
        broadcasting properties.
    EF : ndarray
        Fermi level in eV.
    T : ndarray
        Temperature in Kelvin.
    k : int
        Exponent of the power term in the Fermi integral, and default is 0.
    ndiv : int
        Order of extrapolation for Romberg method. Default is 6.
    eps : float
        A tolerance used to prevent the integrand from becoming undefined
        where E=0. Default is 1E-10.

    Returns
    -------
    ndarray
        Integral values.

    :meta private:
    '''

    # func(E, T)
    EF, T = np.asarray(EF), np.asarray(T)

    km = 3       # maybe the best choice for Fermi-Dirac integrals
    kT = kB_eV * T
    Y = EF/kT    # auto boardcast
    width = 1/(1+np.exp(-Y/km))
    nbins = np.round(np.power(2, ndiv))
    tp = np.linspace(0, width, nbins+1)  # add new axis at first
    
    xp = np.zeros_like(tp) - Y + eps     # (E-EF)/(kB_eV*T)
    fp = np.zeros_like(tp)
    
    xp[1:-1] = km * np.log(1/tp[1:-1]-1)
    fp = func(kT*xp+EF, T)
    
    wt = np.power(xp, k) \
         * km * np.power(tp*(1-tp), km-1) \
         / np.power(np.power(tp, km) + np.power(1-tp, km), 2)
    return romb(fp * wt, axis=0) * width/nbins


class BaseBand(ABC):
    '''
    An abstract class about the band model. It offers the basic calculation
    for thermoelectric transport properties, such as the Seebeck coefficient,
    electrical conductivity, and electronic thermal conductivity. It generally
    supports numpy-style broadcasting operations.
    
    To subclass, three methods need to be implemented:
    
    1. `dos(E)`: Density of states, in 1E19 states/(eV.cm^3). In physics, it
    is usually denoted as :math:`g(E)` and defined as:
    
    .. math ::

        g(E) = 2 \\int_{\\substack{\\text{BZ}}}
               \\delta(E-E(k)) \\frac{d^3k}{8 \\pi ^3}
    
    2. `trs(E, T)`: Transport distribution function, or spectral conductivity,
    in S/cm. In physics, it is usually denoted as :math:`\\sigma_s(E, T)`, and
    is defined as:
    
    .. math ::
    
        \\sigma_s(E, T) = q^2 \\tau v^2 g(E)
    
    3. `hall(E, T)`: Hall transport distribution function, in S.cm/(V.s)
    (i.e., the product of S/cm and cm^2/(V.s)). Here, it is denoted as
    :math:`\\sigma_H(E, T)`, and is expressed as:
    
    .. math ::
    
        \\sigma_H(E, T) = q \\frac{\\tau}{m_b^{\\ast}} \\cdot \\sigma_s(E, T)
    
    Notably, energy (`E`) and Fermi level (`EF`) are in (eV), and temperatures
    (`T`) are in Kelvin (K), unless otherwise specified.
    '''
    
    _S0 = 86.17333262145179     # 1E6 * kB_eV
    _L0 = 0.7425843255087367    # 1E8 * kB_eV * kB_eV
    _q_sign = 1
    _caching = None
    cacheable = {'EF', 'T',
                 'N', 'K_0', 'K_1', 'K_2', 'CCRH',
                 'C', 'CS', 'S', 'PF',
                 'L', 'CL', 'Ke',
                 'U', 'RH', 'UH', 'NH',
                 }
    use_idos = False
    
    @abstractmethod
    def dos(self, E):
        '''Density of states, in 1E19 state/(eV.cm^3).'''
        pass
    
    @abstractmethod
    def trs(self, E, T):
        '''Transport distribution function, in S/cm.'''
        pass
    
    @abstractmethod
    def hall(self, E, T):
        '''Hall transport distribution function, in S.cm/(V.s), i.e.
        [S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)].'''
        pass
    
    def _N(self, EF, T):
        '''Carrier concentration, in 1E19 cm^(-3).'''
        # x = E/(kB_eV*T),  E = x*kB_eV*T
        if self.use_idos and hasattr(self, 'idos'):
            return romb_dfx(self.idos, EF, T)
        kernel = lambda E, _EF, _T: \
            self.dos(E) * fermidirac((E-_EF)/(kB_eV*_T))
        sep_point = np.maximum(0, EF)
        left = vquad(kernel, 0, sep_point, args=(EF, T))[0]
        right = vquad(kernel, sep_point, np.inf, args=(EF, T))[0]
        return left + right
    
    def _K_n(self, __n, EF, T):
        '''Integration of transport distribution function, in S/cm.'''
        return romb_dfx(self.trs, EF, T, k=round(__n))
    
    def _CCRH(self, EF, T):
        '''Integration of Hall transport distribution function,
        in S.cm/(V.s).'''
        return self._q_sign * romb_dfx(self.hall, EF, T)
   
    def compile(self, EF, T, max_level=2):
        '''Compile the object under specified Fermi energies (EF) and
        temperatures (T) to avoid redundant integration computations.
        The `max_level` parameter of integer type specifies the highest
        exponent for caching data, with a default value of 2.'''

        self._caching = {
            '_EF': EF,
            '_T': T,
            '_N': self._N(EF, T),
            '_K_n': [self._K_n(i, EF, T) for i in range(max_level+1)],
            '_CCRH': self._CCRH(EF, T),
        }
    
    def clear(self):
        '''Clear the cached data.'''
        self._caching = None
    
    def fetch(self, _prop, args=(), index=None, default=None):
        '''
        A proxy method to retrieve cached data (if compiled) or compute
        it directly (if not compiled). Typically, cached properties are
        named in the underscored form, i.e., as _XXX.

        Parameters
        ----------
        _prop : str
            Key of cached property.
        args : tuple, optional
            Parameters passed to the proxy method, by default ().
        index : int, optional
            The index of list or the order, by default None.
        default : any, optional
            The default value if failed to retrieve, by default None.

        Returns
        -------
        any
            Results.
        '''

        if self._caching:
            if _prop not in self._caching:
                raise KeyError(f'Failed to read uncompiled {_prop}')
            # if any(arg is not None for arg in args):
            #     raise ValueError(f'Conflicting arguments for compiled class')
            if index is None:
                return self._caching[_prop]
            else:
                return self._caching[_prop][index]
        elif _prop in {'_N', '_K_n', '_CCRH'}:
            if index is None:
                return getattr(self, _prop)(*args)
            else:
                return getattr(self, _prop)(index, *args)
        else:
            return default
    
    def __getitem__(self, key):
        if not self._caching:
            raise RuntimeError('Uncompiled class')
        elif key in {'EF', 'T'}:
            return self._caching[f'_{key}']
        elif key in self.cacheable:
            return getattr(self, key)()
        else:
            raise KeyError(f'Uncacheable property {key}')
    
    def N(self, EF=None, T=None):
        '''Carrier concentration, in 1E19 cm^(-3).'''
        return self.fetch('_N', args=(EF, T))
    
    def K_0(self, EF=None, T=None):
        '''Integration of transport distribution function, in S/cm.'''
        return self.fetch('_K_n', args=(EF, T), index=0)
    
    def K_1(self, EF=None, T=None):
        '''Integration of transport distribution function, in S/cm.'''
        return self.fetch('_K_n', args=(EF, T), index=1)
    
    def K_2(self, EF=None, T=None):
        '''Integration of transport distribution function, in S/cm.'''
        return self.fetch('_K_n', args=(EF, T), index=2)
    
    def CCRH(self, EF=None, T=None):
        '''Integration of Hall transport distribution function,
        in S.cm/(V.s).'''
        return self.fetch('_CCRH', args=(EF, T))
    
    def C(self, EF=None, T=None):
        '''Electrical conductivity, in S/cm.'''
        p0 = self.K_0(EF, T)
        return p0
    
    def CS(self, EF=None, T=None):
        '''The product of electrical conductivity and Seebeck
        coefficient, in [S/cm]*[uV/K].'''
        p1 = self.K_1(EF, T)
        return self._q_sign * 1E6 * kB_eV * p1
    
    def S(self, EF=None, T=None):
        '''Seebeck coefficient, in uV/K.'''
        p0 = self.K_0(EF, T)
        p1 = self.K_1(EF, T)
        return self._q_sign * self._S0 * p1/p0
    
    def PF(self, EF=None, T=None):
        '''Power factor, in uW/(cm.K^2).'''
        p0 = self.K_0(EF, T)
        p1 = self.K_1(EF, T)
        pr = np.power(p1, 2) / p0
        return 1E6 * kB_eV * kB_eV * pr
    
    def CL(self, EF=None, T=None):
        '''The product of electrical conductivity and Lorenz
        number (or electronic thermal conductivity divided by
        absolute temperature), in [W/(m.K)] / [K].'''
        p0 = self.K_0(EF, T)
        p1 = self.K_1(EF, T)
        p2 = self.K_2(EF, T)
        pr = p2 - np.power(p1, 2)/p0
        return 1E2 * kB_eV * kB_eV * pr
    
    def L(self, EF=None, T=None):
        '''Lorenz number, in 1E-8 W.Ohm/K^2.'''
        p0 = self.K_0(EF, T)
        p1 = self.K_1(EF, T)
        p2 = self.K_2(EF, T)
        pr = p2/p0 - np.power(p1/p0, 2)
        return self._L0 * pr
    
    def Ke(self, EF=None, T=None):
        '''Electronic thermal conductivity, in W/(m.K).'''
        p0 = self.K_0(EF, T)
        p1 = self.K_1(EF, T)
        p2 = self.K_2(EF, T)
        pr = p2 - np.power(p1, 2)/p0
        pT = self.fetch('_T', default=T)
        return 1E2 * kB_eV * kB_eV * pr * pT
    
    def U(self, EF=None, T=None):
        '''Carrier drift mobility, in cm^2/(V.s).'''
        pC = self.K_0(EF, T)     # S/cm
        pN = self.N(EF, T)          # 1E19 cm^-3
        pQ = self._q_sign * q
        return pC/(pQ*pN*1E19)
    
    def RH(self, EF=None, T=None):
        '''Hall coefficient, in cm^3/C.'''
        return self.CCRH(EF, T)/np.power(self.K_0(EF, T), 2)
    
    def UH(self, EF=None, T=None):
        '''Carrier Hall mobility, in cm^2/(V.s).'''
        return self.CCRH(EF, T)/self.K_0(EF, T)
    
    def NH(self, EF=None, T=None):
        '''Hall carrier concentration, in 1E19 cm^-3.'''
        pQ = self._q_sign * q
        return 1E-19*np.power(self.K_0(EF, T), 2)/self.CCRH(EF, T)/pQ

    def solve_EF(self, prop, value, T, near=0, between=None, **kwargs):
        '''
        A wrapper for the scipy.optimize.root_scalar method used to solve
        for the Fermi energy from specified thermoelectric transport
        properties and temperatures.

        Parameters
        ----------
        prop : str
            The thermoelectric transport property, such as 'S', 'C'.
        value : ndarray
            The target value of the specified property.
        T : ndarray
            The absolute temperature.
        near : float, optional
            Initial guess for the reduced Fermi energy value. Default is 0.
        between : tuple like (float, float), optional
            Guess range for the reduced Fermi energy. Default is None.
            Recommended for monotonic properties, use 'near'.
        **kwargs : any, optional
            Additional parameters to pass to scipy.optimize.root_scalar.

        Returns
        -------
        ndarray
            Fermi levels in eV.
        '''
        para = {'x0': near,
                'x1': None if near is None else near+1,
                'bracket': between}
        para.update(kwargs)
        def _solve(iVal, iT):
            residual = lambda x: getattr(self, prop)(x*kB_eV*iT, iT) - iVal
            out = root_scalar(residual, **para)
            return out.root*kB_eV*iT if out.converged else np.nan
        return np.vectorize(_solve)(value, T)


class MultiBands(BaseBand):
    '''
    A class for modeling multiple energy bands. Please note that its
    property calculations are now derived from sub-bands, rather than
    directly from the 'dos', 'trs', and 'hall' methods.
    '''
    cacheable = BaseBand.cacheable | {'Kbip'}

    def __init__(self, bands, deltas, btypes=None):
        '''
        Initialize an instance of MultiBands.

        Parameters
        ----------
        bands : tuple of BaseBand
            A collection of instances of :class:`BaseBand`.
        deltas : tuple of float
            Energy offsets for each band.
        btypes : tuple or str, optional
            Types of bands for each entry in `bands`. This can be a sequence
            matching the length of `bands`, or a single value applicable to
            all bands. 'C' signifies conduction bands, while 'V' signifies
            valence bands. If not specified (`None`), band types will be
            inferred using the :meth:`guess_btypes()` method based on `deltas`.
        '''

        dsp = 'Length of {} is not the same as the number of {}'
        if len(bands) != len(deltas):
            raise ValueError(dsp.format('bands', 'deltas'))

        if btypes is None:
            btypes = self.guess_btypes(deltas)
        elif len(btypes) == 1:
            btypes = [btypes[0],] * len(bands)

        for band, btype in zip(bands, btypes):
            if not isinstance(band, BaseBand):
                raise ValueError('Only subclasses of bandlib.BaseBand '
                                 'are supported.')
            if btype.lower()[0] == 'v':
                # Valence Band
                band._q_sign = +1
            elif btype.lower()[0] == 'c':
                # Conduction Band
                band._q_sign = -1
            else:
                raise ValueError(f'Failed to identify band type: {btype}')

        self.bands = bands
        self.deltas = np.asarray(deltas)

    def __str__(self):
        pstr = f'{self.__class__.__name__}:'
        for band, delta in zip(self.bands, self.deltas):
            btype = 'V' if band._q_sign > 0 else 'C'
            pstr += f'\n  {str(band)} @ {delta:.6g} #{btype}'
            # pstr += f'\n  {btype} | {str(band)} @ {delta:.6g}'
            # pstr += f'\n [{delta:.6g} # {btype}] {str(band)}'
        return pstr

    def dos(self, E):
        '''Density of states, in 1E19 state/(eV.cm^3).'''
        E = np.asarray(E)    # for "E-delta" operation
        dos_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            Er = np.maximum(-1*band._q_sign*(E-delta), 0)
            dos_tot += band.dos(Er)
        return dos_tot
    
    def trs(self, E, T):
        '''Transport distribution function, in S/cm.'''
        E = np.asarray(E)    # for "E-delta" operation
        trs_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            Er = np.maximum(-1*band._q_sign*(E-delta), 0)
            trs_tot += band.trs(Er, T)
        return trs_tot
    
    def hall(self, E, T):
        '''Hall transport distribution function, in S.cm/(V.s), i.e.
        [S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)].'''
        E = np.asarray(E)    # for "E-delta" operation
        hall_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            Er = np.maximum(-1*band._q_sign*(E-delta), 0)
            hall_tot += band.hall(Er, T)
        return hall_tot
    
    def _N(self, EF, T):
        '''Carrier concentration, in 1E19 cm^(-3).'''
        EF = np.asarray(EF)    # for "EF-delta" operation
        N_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            EFr = -1*band._q_sign*(EF-delta)
            N_tot += band.fetch('_N', args=(EFr, T))
        return N_tot
    
    def _K_n(self, __n, EF, T):
        '''Integration of transport distribution function, in S/cm.'''
        EF = np.asarray(EF)    # for "EF-delta" operation
        K_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            EFr = -1*band._q_sign*(EF-delta)
            K_tot += np.power(band._q_sign, __n) \
                     * band.fetch('_K_n', args=(EFr, T), index=__n)
        return K_tot
    
    def _CCRH(self, EF, T):
        '''Integration of Hall transport distribution function,
        in S.cm/(V.s).'''
        EF = np.asarray(EF)    # for "EF-delta" operation
        H_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            EFr = -1*band._q_sign*(EF-delta)
            H_tot += band.fetch('_CCRH', args=(EFr, T))
        return H_tot
    
    def compile(self, EF, T, max_level=2):
        '''
        Compile the object under specified Fermi energies (EF) and
        temperatures (T) to avoid redundant integration computations.
        The `max_level` parameter of integer type specifies the highest
        exponent for caching data, with a default value of 2.

        :meta private:
        '''

        EF = np.asarray(EF)    # for "EF-delta" operation
        for band, delta in zip(self.bands, self.deltas):
            EFr = -1*band._q_sign*(EF-delta)
            band.compile(EFr, T, max_level)
        return super().compile(EF, T, max_level)
    
    def clear(self):
        '''
        Clear the cached data.

        :meta private:
        '''
        for band in self.bands:
            band.clear()
        return super().clear()
    
    def Kbip(self, EF=None, T=None):
        '''Bipolar thermal conductivity, in W/(m.K).'''
        if EF is not None: EF = np.asarray(EF)    # for "EF-delta" operation
        Nc, Nv = 0, 0
        p0_c, p0_v, p1_c, p1_v = 0, 0, 0, 0
        for band, delta in zip(self.bands, self.deltas):
            if band._q_sign < 0:
                Nc += 1
                EFr = None if EF is None else EF-delta
                p0_c += band.K_0(EFr, T)
                p1_c -= band.K_1(EFr, T)
            else:
                Nv += 1
                EFr = None if EF is None else delta-EF
                p0_v += band.K_0(EFr, T)
                p1_v += band.K_1(EFr, T)

        if Nc == 0:
            if Nv == 0:
                return 0
            else:
                return np.zeros_like(p0_v)
        else:
            if Nv == 0:
                return np.zeros_like(p0_c)
            else:
                pr = (np.power(p1_c, 2)/p0_c + np.power(p1_v, 2)/p0_v) \
                     - np.power(p1_c+p1_v, 2)/(p0_c+p0_v)
                pT = self.fetch('_T', default=T)
                return 1E2 * kB_eV * kB_eV * pr * pT

    @staticmethod
    def guess_btypes(deltas: list):
        '''
        Guess the band types (btypes) based on the given energy offsets
        (`deltas`). This method classifies each band as a conduction band
        ('C') for positive delta or a valence band ('V') for negative delta.
        Deltas close to zero (1E-6) are considered ambiguous and result in
        a `ValueError`. However, if non-zero values in `deltas` are
        consistently positive or negative (excluding near-zero values), all
        bands are classified as either conduction or valence bands accordingly.

        Examples: deltas --> btypes

        - [  0,  0.1,  0.2] --> ['C', 'C', 'C']
        - [0.1,  0.1, -0.2] --> ['C', 'C', 'V']
        - [  0, -0.1, -0.2] --> ['V', 'V', 'V']
        - [  0,  0.1, -0.2] --> ValueError (Failed to classify the first band)

        Parameters
        ----------
        deltas : list of float
            A list of floats representing the energy offsets for each band.

        Returns
        -------
        list of str
            A list of strings ('C' for conduction bands, 'V' for valence bands)
            representing the band types. The length of the list is equal to
            the length of `deltas`.
        '''
        btypes = []
        for delta in deltas:
            if abs(delta) < 1E-6:
                btypes.append('X')
            elif delta > 0:
                btypes.append('C')
            else:
                btypes.append('V')
        if 'X' not in btypes:
            return btypes
        if 'V' not in btypes:
            return ['C',] * len(deltas)
        if 'C' not in btypes:
            return ['V',] * len(deltas)
        raise ValueError(f'Failed to guess btypes (deltas={deltas})')


class APSSPB(BaseBand):
    '''
    A class for describing single parabolic band (SPB) model when the
    acoustic phonon scattering (APS) mechanism predominates. In this
    model, there are three key parameters determine thermoelectric
    properties:
    
    (a) The effective mass of the density of states :math:`m_d^{\\ast}`
    
    (b) The intrinsic electrical conductivity :math:`\\sigma_0`
    
    (c) The ratio of longitudinal to transverse effective masses
        :math:`K^{\\ast}`
    
    These parameters correspond to class attributes m_d, sigma0, and
    Kmass, respectively. The core of constructing the class is
    obtaining the values of these parameters.

    Attributes
    ----------
    m_d : float, optional
        Effective mass of the density of states in :math:`m_e`,
        primarily influencing carrier concentration calculations.
        It should be a positive float, by default 1.
    sigma0 : float, optional
        Intrinsic electrical conductivity in `S/cm`, the core
        parameter influencing thermoelectric transport properties.
        It should be a positive float, by default 1.
    Kmass : float, optional
        The ratio of longitudinal to transverse effective mass,
        affecting calculations related to Hall coefficients. It
        should be a positive float, by default 1.
    '''

    use_idos = True # Enable acceleration algorithms
    _Yita_opt = 0.66812
    _sigma0_to_PFmax = 0.03015137550508442  # [S/cm] --> [uW/(cm.K^2)]
    _UWT_to_PFmax = 0.12122425768565884     # [cm^2/(V.s)] --> [uW/(cm.K^2)]
    _UWT_to_sigma0 = 4.020521639724753      # [cm^2/(V.s)] --> [S/cm]
    # _UWT_to_sigma0 = np.sqrt(np.pi)/2 * q \
    #       * np.power(2*m_e*kB_eV*q*300, 3/2) \
    #       / (2*np.pi*np.pi*np.power(hbar, 3)) * 1E-4 /100  # S/cm
    def __init__(self, m_d=1, sigma0=1, Kmass=1):
        self.m_d = m_d
        self.sigma0 = sigma0
        self.Kmass = Kmass

    def __str__(self):
        props = ['m_d', 'sigma0', 'Kmass']
        pstr = ', '.join(f'{p}={getattr(self, p):.6g}' for p in props)
        return f'{self.__class__.__name__}({pstr})'

    def dos(self, E):
        '''Density of states, in 1E19 state/(eV.cm^3).'''
        factor = 1E-25      # state/(eV.m^3) --> 1E19 state/(eV.cm^3)
        g0 = np.power(2*self.m_d*m_e*q, 3/2)/(2*np.pi*np.pi* np.power(hbar, 3))
        return factor * g0 * np.sqrt(E)

    def idos(self, E, T=None):
        '''Integral of density-of-states, in 1E19 state/(cm^3).'''
        factor = 1E-25      # state/(eV.m^3) --> 1E19 state/(eV.cm^3)
        gE = np.power(2*self.m_d*m_e*q*E, 3/2)/(3*np.pi*np.pi* np.power(hbar, 3))
        return factor * gE
    
    def trs(self, E, T):
        '''Transport distribution function, in S/cm.'''
        E, T = np.asarray(E), np.asarray(T)
        return self.sigma0 * E/(kB_eV*T)
    
    def hall(self, E, T):
        '''Hall transport distribution function, in S.cm/(V.s), i.e.
        [S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)].'''
        E, T = np.asarray(E), np.asarray(T)
        N0 = np.power(2*self.m_d*m_e*kB_eV*q*T, 3/2)/(2*np.pi*np.pi* np.power(hbar, 3))
        facotr = np.power(self.sigma0, 2) / (2/3 * 1E-6*N0 * q)  # N0: m^-3 --> cm^-3
        return self.Kstar * facotr * np.sqrt(E/(kB_eV*T))
    
    @property
    def Kstar(self):
        '''Anisotropy factor of effective mass for Hall effect.'''
        K = self.Kmass
        return 3*K*(K+2)/np.power(2*K+1, 2)
    
    @property
    def UWT(self):
        '''Temperature-independent weighted mobility, in cm^2/(V.s).'''
        return self.sigma0 / self._UWT_to_sigma0
    
    @property
    def PFmax(self):
        '''The maximum power factor, in uW/(cm.K^2).'''
        return self.sigma0 * self._sigma0_to_PFmax
    
    def EFopt(self, T):
        '''Optimal Fermi level for the maximum power factor, in eV'''
        return self._Yita_opt * kB_eV * np.asarray(T)
    
    @classmethod
    def from_DP(cls, m1=1, m2=None, Nv=1, Cii=1, Ed=1):
        '''
        Construct the class based on the Deformation Potential
        (DP) theory.

        Parameters
        ----------
        m1 : float, optional
            The ratio of the longitudinal effective mass to the static
            electron mass, by default 1.
        m2 : float or None, optional
            The ratio of the transverse effective mass to the static
            electron mass. If None (default), it is set equal to m1.
        Nv : int, optional
            Valley degeneracy, by default 1.
        Cii : float, optional
            Elastic constant in GPa, by default 1.
        Ed : float, optional
            Deformation potential in eV, by default 1.
        '''

        m2 = (m2 or m1)
        m_c = 3/(1/m1+2/m2)
        m_d = np.cbrt(Nv*Nv * m1*m2*m2)
        # factor = (2*q*q*hbar*1E9) / (3*m_e*np.pi*q*q) /100    # S/cm
        factor = 245.66655370009886     # S/cm
        sigma0 = factor * (Nv*Cii)/(m_c*Ed*Ed)
        return cls(m_d=m_d, sigma0=sigma0, Kmass=m1/m2)
    
    @classmethod
    def from_UWT(cls, m_d=1, UWT=1, Kmass=1):
        '''
        Construct the class based on temperature-independent weighted
        mobility.

        Parameters
        ----------
        m_d : float, optional
            Effective mass of the density of states in static electron mass,
            by default 1.
        UWT : float, optional
            Temperature-independent weighted mobility in cm^2/(V.s), by
            default 1.
        Kmass : float, optional
            The ratio of longitudinal to transverse effective mass, by
            default 1.
        '''
        sigma0 = cls._UWT_to_sigma0 * UWT
        return cls(m_d=m_d, sigma0=sigma0, Kmass=Kmass)

    @classmethod
    def valuate(cls, dataS, dataT=None, dataC=None, dataN=None,
                hall=False, Kmass=1):
        '''
        A class method for quickly evaluating the carriar transport
        properties (such as Lorenz number `L` and the temperature-independent
        weighted mobility `UWT`) based on experimental data.

        Parameters
        ----------
        dataS : ndarray
            Experimental data for Seebeck coefficient in uV/K.
        dataT : ndarray, optional
            Experimental data for temperature in Kelvin.
            Defaults to None.
        dataC : ndarray, optional
            Experimental data for electrical conductivity in S/cm.
            Defaults to None.
        dataN : ndarray, optional
            Experimental data for carrier concentration in 1E19 cm^-3.
            Defaults to None.
        hall : boolean, optional
            Whether to consider the Hall effect. Defaults to False.
        Kmass : float, optional
            The ratio of longitudinal to transverse effective mass, by
            default 1.

        Returns
        -------
        AttrDict
            An attribute dictionary containing:

            * `L`: Lorenz number in 1E-8 W.Ohm/K^2`.
            * `Ke`: Electronic thermal conductivity in W/(m.K),
              only if both `dataT` and `dataC` are provided.
            * `sigma0`: Intrinsic electrical conductivity in S/cm,
              only if both `dataT` and `dataC` are provided.
            * `UWT`: Temperature-independent weighted mobility in cm^2/(V.s),
              only if both `dataT` and `dataC` are provided.
            * `PFmax`: The maximum power factor in uW/(cm.K^2),
              only if both `dataT` and `dataC` are provided.
            * `m_d`: The ratio of effective mass to the electron mass,
              only if both `dataT` and `dataN` are provided.
            * `Nopt`: The optimal carrier concentration in 1E19 cm^(-3),
              only if both `dataT` and `dataN` are provided.
        '''
        spb = cls(Kmass=Kmass)

        if dataT is not None:
            dataEF = spb.solve_EF('S', dataS, dataT, maxiter=300)
            out = AttrDict(L=spb.L(dataEF, dataT))
        else:
            TEMP = 1/kB_eV
            yita = spb.solve_EF('S', dataS, TEMP)
            return AttrDict(L=spb.L(yita, TEMP))

        if dataC is not None:
            out['Ke'] = 1E-6 * out['L']*np.multiply(dataC, dataT)

            sigma0 = out['sigma0'] = np.divide(dataC, spb.C(dataEF, dataT))
            out['UWT'] = sigma0/cls._UWT_to_sigma0
            out['PFmax'] = sigma0 * cls._sigma0_to_PFmax

        if dataN is not None:
            EF_opt = np.multiply(cls._Yita_opt * kB_eV, dataT)
            if hall:
                N_ref = spb.NH(dataEF, dataT)
                Nopt_ref = spb.NH(EF_opt, dataT)
                logger.debug('Enable Hall effect.')
            else:
                N_ref = spb.N(dataEF, dataT)
                Nopt_ref = spb.N(EF_opt, dataT)
                logger.debug('Disable Hall effect.')

            N_ratio = np.divide(dataN, N_ref)
            out['m_d'] = np.power(N_ratio, 2/3)
            out['Nopt'] = N_ratio * Nopt_ref

        logger.debug(f'Calculated: {list(out.keys())}')
        return out


class APSSKB(BaseBand):
    '''
    A class for describing single Kane band (SKB) model when the acoustic
    phonon scattering (APS) mechanism predominates. In contrast to the
    classical single parabolic band (SPB) model (see :class:`APSSPB`), an
    additional parameter describing the energy band shape, namely the bandgap
    (Eg), is introduced.

    Attributes
    ----------
    m_d : float, optional
        Effective mass of the density of states in :math:`m_e`,
        primarily influencing carrier concentration calculations.
        It should be a positive float, by default 1.
    sigma0 : float, optional
        Intrinsic electrical conductivity in S/cm, the core
        parameter influencing thermoelectric transport properties.
        It should be a positive float, by default 1.
    Eg : float, optional
        Parameter bandgap in eV, which significantly influences
        various transport properties. It should be a positive
        float, by default 1.
    Kmass : float, optional
        The ratio of longitudinal to transverse effective mass,
        affecting calculations related to Hall coefficients. It
        should be a positive float, by default 1.
    '''

    use_idos = True # Enable acceleration algorithms
    _UWT_to_sigma0 = 4.020521639724753  # [S/cm] / [cm^2/(V.s)]
    def __init__(self, m_d=1, sigma0=1, Eg=1, Kmass=1):
        self.m_d = m_d
        self.sigma0 = sigma0
        if Eg > 0:
            self.Eg = Eg
        else:
            raise ValueError('Eg should be greater than 0')
        self.Kmass = Kmass

    def __str__(self):
        props = ['m_d', 'sigma0', 'Eg', 'Kmass']
        pstr = ', '.join(f'{p}={getattr(self, p):.6g}' for p in props)
        return f'{self.__class__.__name__}({pstr})'
        
    def dos(self, E):
        '''Density of states, in 1E19 state/(eV.cm^3).'''
        E = np.asarray(E)
        factor = 1E-25      # state/(eV.m^3) --> 1E19 state/(eV.cm^3)
        g0 = np.power(2*self.m_d*m_e*q, 3/2)/(2*np.pi*np.pi* np.power(hbar, 3))
        kane = np.sqrt(1+E/self.Eg) * (1+2*E/self.Eg)
        return factor * g0 * np.sqrt(E) * kane

    def idos(self, E, T=None):
        '''Integral of density-of-states, in 1E19 state/(cm^3).'''
        E = np.asarray(E)
        # factor = 1E-25      # state/(eV.m^3) --> 1E19 state/(eV.cm^3)
        gE = 1E-25 * np.power(2*self.m_d*m_e*q*E*(1+E/self.Eg), 3/2)\
             / (3*np.pi*np.pi* np.power(hbar, 3))
        return gE
    
    def trs(self, E, T):
        '''Transport distribution function, in S/cm.'''
        E, T = np.asarray(E), np.asarray(T)
        kane = 3*(1+E/self.Eg)/(np.power(1+2*E/self.Eg, 2)+2)
        return self.sigma0 * E/(kB_eV*T) * kane
    
    def hall(self, E, T):
        '''Hall transport distribution function, in S.cm/(V.s), i.e.
        [S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)].'''
        E, T = np.asarray(E), np.asarray(T)
        N0 = np.power(2*self.m_d*m_e*kB_eV*q*T, 3/2)/(2*np.pi*np.pi* np.power(hbar, 3))
        facotr = np.power(self.sigma0, 2) / (2/3 * 1E-6*N0 * q)  # N0: m^-3 --> cm^-3
        kane = 9*np.sqrt(1+E/self.Eg)/np.power(np.power(1+2*E/self.Eg, 2)+2, 2)
        return self.Kstar * facotr * np.sqrt(E/(kB_eV*T)) * kane
    
    @property
    def Kstar(self):
        '''Anisotropy factor of effective mass for Hall effect.'''
        K = self.Kmass
        return 3*K*(K+2)/np.power(2*K+1, 2)
    
    @property
    def UWT(self):
        '''Temperature-independent weighted mobility, in cm^2/(V.s).'''
        return self.sigma0 / self._UWT_to_sigma0
    
    @classmethod
    def from_DP(cls, m1=1, m2=None, Nv=1, Cii=1, Ed=1, Eg=1):
        '''
        Construct the class based on the Deformation Potential
        (DP) theory.

        Parameters
        ----------
        m1 : float, optional
            The ratio of the longitudinal effective mass to the static
            electron mass, by default 1.
        m2 : float or None, optional
            The ratio of the transverse effective mass to the static
            electron mass. If None (default), it is set equal to m1.
        Nv : int, optional
            Valley degeneracy, by default 1.
        Cii : float, optional
            Elastic constant in GPa, by default 1.
        Ed : float, optional
            Deformation potential in eV, by default 1.
        Eg : float, optional
            Bandgap in eV, by default 1.
        '''
        
        m2 = (m2 or m1)
        m_c = 3/(1/m1+2/m2)
        m_d = np.cbrt(Nv*Nv * m1*m2*m2)
        # factor = (2*q*q*hbar*1E9) / (3*m_e*np.pi*q*q) /100    # S/cm
        factor = 245.66655370009886     # S/cm
        sigma0 = factor * (Nv*Cii)/(m_c*Ed*Ed)
        return cls(m_d=m_d, sigma0=sigma0, Eg=Eg, Kmass=m1/m2)
    
    @classmethod
    def from_UWT(cls, m_d=1, UWT=1, Eg=1, Kmass=1):
        '''
        Construct the class based on temperature-independent weighted
        mobility.

        Parameters
        ----------
        m_d : float, optional
            Effective mass of the density of states in static electron mass,
            by default 1.
        UWT : float, optional
            Temperature-independent weighted mobility in cm^2/(V.s), by
            default 1.
        Eg : float, optional
            Bandgap in eV, by default 1.
        Kmass : float, optional
            The ratio of longitudinal to transverse effective mass, by
            default 1.
        '''
        sigma0 = cls._UWT_to_sigma0 * UWT
        return cls(m_d=m_d, sigma0=sigma0, Eg=Eg, Kmass=Kmass)

    @classmethod
    def valuate(cls, dataS, dataT, dataC=None, dataN=None,
                Eg=1, hall=False, Kmass=1):
        '''
        A class method for quickly evaluating the carriar transport
        properties (such as Lorenz number `L` and the temperature-independent
        weighted mobility `UWT`) based on experimental data.

        Parameters
        ----------
        dataS : ndarray
            Experimental data for Seebeck coefficient in uV/K.
        dataT : ndarray
            Experimental data for temperature in Kelvin.
        dataC : ndarray, optional
            Experimental data for electrical conductivity in S/cm.
            Defaults to None.
        dataN : ndarray, optional
            Experimental data for carrier concentration in 1E19 cm^-3.
            Defaults to None.
        Eg : float, optional
            Bandgap in eV, by default 1.
        hall : boolean, optional
            Whether to consider the Hall effect. Defaults to False.
        Kmass : float, optional
            The ratio of longitudinal to transverse effective mass, by
            default 1.

        Returns
        -------
        AttrDict
            An attribute dictionary containing:

            * `L`: Lorenz number in 1E-8 W.Ohm/K^2`.
            * `Ke`: Electronic thermal conductivity in W/(m.K),
              only if `dataC` is provided.
            * `sigma0`: Intrinsic electrical conductivity in S/cm,
              only if `dataC` is provided.
            * `UWT`: Temperature-independent weighted mobility in cm^2/(V.s),
              only if `dataC` is provided.
            * `m_d`: The ratio of effective mass to the electron mass,
              only if `dataN` is provided.
        '''
        skb =  cls(Kmass=Kmass, Eg=Eg)
        dataEF = skb.solve_EF('S', dataS, dataT)
        out = AttrDict(L=skb.L(dataEF, dataT))

        if dataC is not None:
            out['Ke'] = 1E-6 * out['L']*np.multiply(dataC, dataT)

            sigma0 = out['sigma0'] = np.divide(dataC, skb.C(dataEF, dataT))
            out['UWT'] = sigma0/cls._UWT_to_sigma0

        if dataN is not None:
            if hall:
                N_ref = skb.NH(dataEF, dataT)
                logger.debug('Enable Hall effect.')
            else:
                N_ref = skb.N(dataEF, dataT)
                logger.debug('Disable Hall effect.')
            N_ratio = np.divide(dataN, N_ref)
            out['m_d'] = np.power(N_ratio, 2/3)

        logger.debug(f'Calculated: {list(out.keys())}')
        return out


class LinearBand(BaseBand):
    '''
    A class for describing linear band model:

    .. math ::

        E &= v_F \\cdot \\hbar |k| \\\\
        g(E) &= \\frac{E^2}{\\pi^2 \\hbar^3 v_F^3} \\\\
        \\sigma_s(E, T) &= \\sigma_0 \\text{ when } E > 0

    Attributes
    ----------
    sigma0 : float, optional
        Intrinsic electrical conductivity in S/cm, the core
        parameter influencing thermoelectric transport properties.
        It should be a positive float, by default 1.
    vF : float, optional
        Fermi velocity in Angstrom/fs, i.e. 10^5 m/s. It should be
        a positive float, by default 1.
    '''
    vF = 1      #: :meta private: A/fs, i.e. 10^5 m/s
    sigma0 = 1  #: :meta private: S/cm
    def __init__(self, vF=1, sigma0=1):
        self.vF = np.absolute(vF)
        self.sigma0 = sigma0

    def __str__(self):
        props = ['vF', 'sigma0']
        pstr = ', '.join(f'{p}={getattr(self, p):.6g}' for p in props)
        return f'{self.__class__.__name__}({pstr})'

    def dos(self, E):
        '''Density of states, in 1E19 state/(eV.cm^3).'''
        E = np.asarray(E)
        factor = 1E-25      # state/(eV.m^3) --> 1E19 state/(eV.cm^3)
        g0 = np.power(1/np.pi, 2) * np.power(q/(self.vF*hbar), 3)
        return factor * g0 * np.power(E, 2)

    def trs(self, E, T):
        '''Transport distribution function, in S/cm.'''
        broadcasted = np.broadcast(E, T)
        return self.sigma0 * np.ones(broadcasted.shape)

    def hall(self, E, T):
        '''Hall transport distribution function. Not implemented now!'''
        raise NotImplementedError

    def _CCRH(self, EF, T):
        '''To ensure compatibility with the compile() method.'''
        return None


class RSPB:
    '''
    A class for modeling the Restructured Single Parabolic Band
    (RSPB) model in thermoelectric materials. 
    This class contains three types of attributes:
    
    * **Ending with 'r'**: Represents reduced properties, usually
      requiring a reduced carrier concentration (`Nr`) as input. 
      An optional parameter `factor` can also be passed in,
      which defaults to 1. This indicates the return of
      reduced material property values. 
      If a factor corresponding to the material property
      (indicated by attributes ending with '0') is passed in,
      the material property values in common units will be returned.
    * **Ending with '0'**: Represents factors for material properties
      in common units.
    * **Starting with 'i'**: Inverse functions of the reduced properties,
      used to solve the reduced carrier concentration from
      reduced material property values (or property values in
      common units, determined by the optional parameter `factor`).

    '''

    N0 = 2.5094122298407914     #: Unit:: 1E19 cm^-3
    S0 = 86.17333262145179      #: Unit:: uV/K
    L0 = 0.7425843255087367     #: Unit:: 1E-8 W.Ohm/K^2
    C0 = 4.020521639724753      #: Unit:: S/cm
    PF0 = 0.029855763500282857  #: Unit:: uW/(cm.K^2)
    _Nr_opt = 1.255
    _PFr_max = 4.017850558247082
    _UWT_to_PFmax = 0.11995599604650432 # [cm^2/(V.s)] --> [uW/(cm.K^2)]
    _UWT_to_sigma0 = 4.020521639724753  # [cm^2/(V.s)] --> [S/cm]
    
    @staticmethod
    def Nmr(Nr, factor=1, m_d=1, T=300):
        Nr, factor, m_d, T = map(np.asarray, [Nr, factor, m_d, T])
        return factor * np.power(m_d*T/300, 3/2) * Nr
    
    @staticmethod
    def Sr(Nr, factor=1, delta: float = 0.075):
        Nr, factor = map(np.asarray, [Nr, factor])
        return factor * np.log(1+delta+np.exp(2)/Nr)
    
    @staticmethod
    def iSr(Sr, factor=1, delta: float = 0.075):
        Sr, factor = map(np.asarray, [Sr, factor])
        return np.exp(2)/(np.exp(Sr/factor)-1-delta)
    
    @staticmethod
    def Ur(Nr, factor=1):
        Nr, factor = map(np.asarray, [Nr, factor])
        return factor * np.power(1+Nr/2, -1/3)
    
    @staticmethod
    def iUr(Ur, factor=1):
        Ur, factor = map(np.asarray, [Ur, factor])
        return 2*(np.power(factor/Ur, 3)-1)
    
    @staticmethod
    def Lr(Nr, factor=1):
        Nr, factor = map(np.asarray, [Nr, factor])
        scale = np.power(1+np.power(Nr/np.pi/2, -3/2), 2/3)
        return factor * (2+(np.pi*np.pi/3-2)/scale)
    
    @staticmethod
    def iLr(Lr, factor=1):
        Lr, factor = map(np.asarray, [Lr, factor])
        scale = (np.pi*np.pi/3-2)/(Lr/factor-2)
        return 2*np.pi*np.power(np.power(scale, 3/2)-1, -2/3)
    
    @classmethod
    def Cr(cls, Nr, factor=1, UWT=1):
        # N0 * mt32 * Nr * q * U0 * Ur
        #    = N0 * q * (mt32*U0) * Nr*Ur
        #    = C0 * UWT * Nr * Ur
        Nr, factor, UWT = map(np.asarray, [Nr, factor, UWT])
        return factor * UWT * Nr * cls.Ur(Nr)
    
    @classmethod
    def PFr(cls, Nr, factor=1, UWT=1, delta: float =0.075):
        # (C0 * UWT * Nr * Ur) * (S0 * Sr)^2
        #    = C0*S0^2 * (UWT*Nr*Ur)*Sr^2
        Nr, factor, UWT = map(np.asarray, [Nr, factor, UWT])
        return factor * cls.Cr(Nr, UWT=UWT) \
               * np.power(cls.Sr(Nr, delta=delta), 2)
    
    @classmethod
    def valuate(cls, dataS, dataC=None, dataT=None, dataN=None, delta=0.075):
        '''
        A class method for quickly evaluating the carriar transport
        properties (such as Lorenz number `L` and the temperature-independent
        weighted mobility `UWT`) based on experimental data.

        Parameters
        ----------
        dataS : ndarray
            Experimental data for Seebeck coefficient in uV/K.
        dataC : ndarray, optional
            Experimental data for electrical conductivity in S/cm.
            Defaults to None.
        dataT : ndarray, optional
            Experimental data for temperature in Kelvin.
            Used in conjunction with `dataN` to calculate `m_eff`.
            Defaults to None.
        dataN : ndarray, optional
            Experimental data for carrier concentration in 1E19 cm^-3.
            Used in conjunction with `dataT` to calculate `m_eff`.
            Defaults to None.
        delta : float, optional
            A parameter related to the Seebeck coefficient, defaults to 0.075.

        Returns
        -------
        AttrDict
            An attribute dictionary containing:
            
            * `L`: Lorenz number in 1E-8 W.Ohm/K^2`.
            * `UWT`: Temperature-independent weighted mobility in cm^2/(V.s).
            * `PFmax`: The maximum power factor in uW/(cm.K^2).
            * `Ke`: Electronic thermal conductivity in W/(m.K),
              only if both `dataT` and `dataN` are provided.
            * `m_eff`: The ratio of effective mass to the electron mass,
              only if both `dataT` and `dataN` are provided.
            * `Nopt`: The optimal carrier concentration in 1E19 cm^(-3),
              only if both `dataT` and `dataN` are provided.
        '''
        Nr = cls.iSr(dataS, factor=cls.S0, delta=delta)
        L = cls.Lr(Nr, factor=cls.L0)
        out = AttrDict(L=L)

        if dataC is not None:
            dataC = np.asarray(dataC)
            UWT = dataC/cls.Cr(Nr, factor=cls.C0)
            out['UWT'] = UWT
            out['PFmax'] = UWT * cls._UWT_to_PFmax
            out['sigma0'] = UWT * cls._UWT_to_sigma0

        if all(d is not None for d in (dataC, dataT, dataN)):
            dataT, dataN = np.asarray(dataT), np.asarray(dataN)
            out['Ke'] = 1E-6 * L*dataC*dataT
            out['m_eff'] = np.power(dataN/Nr, 2/3) * 300/dataT
            out['Nopt'] = cls._Nr_opt * cls.N0 * dataN/Nr

        logger.debug(f'Calculated: {list(out.keys())}')
        return out


def bandline(k, m_b=1, k0=0, E0=0, Eg=None):
    '''
    Model an idealized electronic dispersion relation based on
    the given parameters.

    Parameters
    ----------
    k : ndarray
        Wave vector in rad/Ang (radians per angstrom), i.e.,
        containing the :math:`2 \\pi` factor.
    m_b : ndarray, optional
        Ratio of the band effective mass to the electron mass.
        Defaults to 1.
    k0 : ndarray, optional
        Wave vector offset in rad/Ang. Defaults to 0.
    E0 : ndarray, optional
        Energy offset in eV. Defaults to 0.
    Eg : ndarray, optional
        Bandgap in eV. If set to None (default), it represents a parabolic band.

    Returns
    -------
    ndarray
        Energy values in eV.
    '''
    k, m_b, k0, E0 = map(np.asarray, [k, m_b, k0, E0])
    COEF_BAND = 7.619964222971923
    parabolic = COEF_BAND * np.power(k - k0, 2) / m_b / 2
    if Eg is None:
        return parabolic + E0
    else:
        Eg = np.asarray(Eg)
        kane = np.sqrt(0.25 + np.abs(parabolic)/Eg) + 0.5
        return parabolic / kane + E0


def dosline(E, m_d=1, E0=0, Vcell=1, Eg=None):
    '''
    Model an idealized density-of-states (DOS) based on the given parameters.

    Parameters
    ----------
    E : ndarray
        Energy values in eV.
    m_d : ndarray, optional
        Ratio of the DOS effective mass to the electron mass.
        Defaults to 1.
    E0 : ndarray, optional
        Energy offset in eV. Defaults to 0.
    Vcell : ndarray, optional
        Volume of the unit cell in cubic angstrom (Ang^3). Defaults to 1.
    Eg : ndarray, optional
        Bandgap in eV. If set to None (default), it represents a parabolic band.

    Returns
    -------
    ndarray
        Density-of-states values in states/(eV.cell).
        When `Vcell` is set to its default value of 1,
        the unit is also equivalent to states/(eV.Ang^3)
        or 1E24 states/(eV.cm^3).
    '''
    E, m_d, E0, Vcell = map(np.asarray, [E, m_d, E0, Vcell])
    COEF_DOS = 146.7959743657925
    energy = np.maximum(np.sign(m_d)*(E-E0), 0)
    coef = Vcell/COEF_DOS * np.power(np.abs(m_d), 3/2)
    if Eg is None:
        return coef * np.sqrt(energy)
    else:
        Eg = np.asarray(Eg)
        return coef * np.sqrt(E*(1+E/Eg))*(1+2*E/Eg)


EXECMETA = {
    'APSSPB': ExecWrapper(APSSPB,
        opts=['m_d', 'sigma0', 'Kmass'],
    ),
    'APSSPB_UWT': ExecWrapper(APSSPB.from_UWT,
        opts=['m_d', 'UWT', 'Kmass'],
    ),
    'APSSPB_DP': ExecWrapper(APSSPB.from_DP,
        opts=['m1', 'm2', 'Nv', 'Cii', 'Ed'],
    ),
    'APSSKB': ExecWrapper(APSSKB,
        opts=['m_d', 'sigma0', 'Eg', 'Kmass'],
    ),
    'APSSKB_UWT': ExecWrapper(APSSKB.from_UWT,
        opts=['m_d', 'UWT', 'Eg', 'Kmass'],
    ),
    'APSSKB_DP': ExecWrapper(APSSKB.from_DP,
        opts=['m1', 'm2', 'Nv', 'Cii', 'Ed', 'Eg'],
    ),
    'LinearBand': ExecWrapper(LinearBand,
        opts=['vF', 'sigma0'],
    ),
    'valuate.APSSPB': ExecWrapper(APSSPB.valuate,
        args=['dataS',],
        opts=['dataT', 'dataC', 'dataN', 'hall', 'Kmass'],
    ),
    'valuate.APSSKB': ExecWrapper(APSSKB.valuate,
        args=['dataS', 'dataT',],
        opts=['dataC', 'dataN', 'Eg', 'hall', 'Kmass'],
    ),
    'valuate.RSPB': ExecWrapper(RSPB.valuate,
        args=['dataS',],
        opts=['dataC', 'dataT', 'dataN', 'delta',],
    ),
}


def parse_Bands(filename, specify=None):
    '''Parse bands model, and label list of props from a config file'''
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

    # parse bands
    bands_ = entry.getlist('bands')
    if bands_ is None:
        raise ValueError(dsp.format('bands'))
    logger.info("Parse all bands:")

    bands = []
    for band_ in bands_:
        content, xtype = config.pmatch(band_)
        kwargs = {k:float(v) for k,v in content.items()}
        band = EXECMETA[xtype].execute(**kwargs)
        bands.append(band)
        logger.info(f'  {str(band)}')

    # parse deltas, btypes
    deltas = entry.getlist_float('deltas', [0, ]*len(bands))
    if deltas is None:
        raise ValueError(dsp.format('deltas'))
    logger.info('deltas: [%s]' % ', '.join(f'{i:.3g}' for i in deltas))

    btypes = entry.getlist('btypes')
    logger.info(f'btypes: {btypes or f"<Guess: {MultiBands.guess_btypes(deltas)}>"}')

    # initialze MultiBands
    model = MultiBands(bands, deltas, btypes)
    logger.info('Finish building MultiBands() instance')

    # parse T
    T = entry.getseq('T')
    if T is None:
        logger.info(dsp.format('T'))
        return model, None

    # parse EF
    if 'EF' not in entry:
        logger.info(dsp.format('EF'))
        return model, None

    EF_ = entry.get('EF')
    if '@' in EF_:
        solver = EF_.split('@')[-1].strip()
        refdata_ = EF_.split('@')[0].strip()
        logger.debug(f"Extract EF value <{EF_}> to '{refdata_}' & '{solver}' ")
        entry['EF'] = refdata_
        refdata = entry.getseq('EF')

        try:
            refdata, T = np.broadcast_arrays(refdata, T)
        except ValueError:
            raise ValueError(f'Mismatch is between given data and T')

        try:
            initial = entry.getfloat('initial', 0)
            EF = model.solve_EF(solver, refdata, T, near=initial)
            logger.info(f'Solve Fermi energies from {solver}, '
                        f'where the initial value is set to {initial:.3g}.')
        except Exception as e:
            raise RuntimeError(f'Failed to solve EF from {solver}: {e}')
    else:
        EF = entry.getseq('EF')

    try:
        EF, T = np.broadcast_arrays(EF, T)      # broadcasted (EF, T)
    except ValueError:
        logger.debug(f'Current EF: {EF}')
        logger.debug(f'Current T:  {T}')
        raise ValueError(f'Mismatch is between EF and T')

    # display logging
    logger.info('Get Fermi energies (EF) & temperatures (T) successfully')
    logger.debug('EF: [%s]' % ', '.join(f'{i:.3g}' for i in EF))
    logger.debug('T:  [%s]' % ', '.join(f'{i:.3g}' for i in T))

    # compile model and parse props
    model.compile(EF, T)
    props_default = 'T EF N C S PF L Ke'.split()
    props = entry.getlist('properties', props_default)
    logger.debug('Compile model (props: %s)' % ', '.join(props))
    return model, props
