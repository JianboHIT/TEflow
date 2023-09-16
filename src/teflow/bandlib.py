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

from abc import ABC, abstractmethod
from scipy.integrate import quad, romb
from scipy.optimize import root_scalar
import numpy as np


kB_eV = 8.617333262145179e-05   # eV/K
m_e = 9.1093837015e-31          # kg
hbar = 1.054571817e-34          # J.s
q = 1.602176634e-19             # C

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


def romb_dfx(func, EF, T, k=0, ndiv=8, eps=1E-10):
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
        Order of extrapolation for Romberg method. Default is 8.
    eps : float
        A tolerance used to prevent the integrand from becoming undefined
        where E=0. Default is 1E-10.

    Returns
    -------
    ndarray
        Integral values.
    '''

    # func(E, T)
    km = 2       # maybe the best choice for Fermi-Dirac integrals
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

        g(E) = \\int \\delta(E-E(k)) \\frac{d^3k}{8 \\pi ^3}
    
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
    
    _q_sign = 1
    _caching = None
    cacheable = {'EF', 'T',
                 'N', 'K_0', 'K_1', 'K_2', 'CCRH',
                 'C', 'CS', 'S', 'PF',
                 'L', 'CL', 'Ke',
                 'U', 'RH', 'UH', 'NH',
                 }
    
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
        kernel = lambda E, _EF, _T: self.dos(E) \
                                    * self.fx((E-_EF)/(kB_eV*_T))
        itg = lambda _EF, _T: quad(kernel, 0, np.inf, args=(_EF, _T))[0]
        return np.vectorize(itg)(EF, T)
    
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
            if any(arg is not None for arg in args):
                raise ValueError(f'Conflicting arguments for compiled class')
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
        return self._q_sign * 1E6 * kB_eV * p1/p0
    
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
        return 1E8 * kB_eV * kB_eV * pr
    
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

    def slove_EF(self, prop, value, T, near=0, between=None, **kwargs):
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
            Initial guess for the Fermi energy value. Default is 0.
        between : tuple like (float, float), optional
            Guess range for the Fermi energy. Default is None.Recommended
            for monotonic properties, use 'near'.
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
        def _slove(iVal, iT):
            residual = lambda x: getattr(self, prop)(x*kB_eV*iT, iT) - iVal
            out = root_scalar(residual, **para)
            return out.root*kB_eV*iT if out.converged else np.nan
        return np.vectorize(_slove)(value, T)
    
    @staticmethod
    def fx(x):
        '''Reduced Fermi Dirac distribution.'''
        p = np.tanh(x/2)
        return 1/2*(1-p)
    
    @staticmethod
    def dfx(x, k=0):
        '''The derivative of reduced Fermi-Dirac distribution multiplied
        by the k-th power function.'''
        k = round(k)
        p = np.tanh(x/2)
        if k == 0:
            return 1/4*(1-p*p)
        else:
            return 1/4*(1-p*p) * np.power(x, k)

    @staticmethod
    def ddfx(x, k=0):
        '''The derivative of dfx.'''
        k = round(k)
        p = np.tanh(x/2)
        if k == 0:
            return 1/4*(1-p*p) * (-p)
        elif k == 1:
            return 1/4*(1-p*p) * (1-x*p)
        else:
            return 1/4*(1-p*p) * (k-x*p)*np.power(x, k-1)


class MultiBand(BaseBand):
    '''
    A class for modeling multiple energy bands. Please note that its
    property calculations are now derived from sub-bands, rather than
    directly from the 'dos', 'trs', and 'hall' methods.
    '''
    cacheable = BaseBand.cacheable | {'Kbip'}

    def __init__(self, cbands=(), cdeltas=(), 
                       vbands=(), vdeltas=(),):
        '''
        Initialize an instance of MultiBand.

        Parameters
        ----------
        cbands : tuple of BandBase, optional
            Conduction bands, by default ().
        cdeltas : tuple of float, optional
            Energy offsets of conduction bands, by default ().
        vbands : tuple of BandBase, optional
            Valence bands, by default ().
        vdeltas : tuple of float, optional
            Energy offsets of valence bands, by default ()
        '''

        dsp = 'Length of {} is not the same as the number of {}'
        if len(cbands) != len(cdeltas):
            raise ValueError(dsp.format('cbands', 'cdeltas'))
        if len(vbands) != len(vdeltas):
            raise ValueError(dsp.format('vbands', 'vdeltas'))

        for band in cbands:
            band._q_sign = -1
        for band in vbands:
            band._q_sign = 1

        self.bands = cbands+vbands
        self.deltas = cdeltas+vdeltas
    
    def dos(self, E):
        '''Density of states, in 1E19 state/(eV.cm^3).'''
        dos_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            Er = np.maximum(-1*band._q_sign*(E-delta), 0)
            dos_tot += band.dos(Er)
        return dos_tot
    
    def trs(self, E, T):
        '''Transport distribution function, in S/cm.'''
        trs_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            Er = np.maximum(-1*band._q_sign*(E-delta), 0)
            trs_tot += band.trs(Er, T)
        return trs_tot
    
    def hall(self, E, T):
        '''Hall transport distribution function, in S.cm/(V.s), i.e.
        [S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)].'''
        hall_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            Er = np.maximum(-1*band._q_sign*(E-delta), 0)
            hall_tot += band.hall(Er, T)
        return hall_tot
    
    def _N(self, EF, T):
        '''Carrier concentration, in 1E19 cm^(-3).'''
        N_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            EFr = -1*band._q_sign*(EF-delta)
            N_tot += band.fetch('_N', args=(EFr, T))
        return N_tot
    
    def _K_n(self, __n, EF, T):
        '''Integration of transport distribution function, in S/cm.'''
        K_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            EFr = -1*band._q_sign*(EF-delta)
            K_tot += np.power(band._q_sign, __n) \
                     * band.fetch('_K_n', args=(EFr, T), index=__n)
        return K_tot
    
    def _CCRH(self, EF, T):
        '''Integration of Hall transport distribution function,
        in S.cm/(V.s).'''
        H_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            EFr = -1*band._q_sign*(EF-delta)
            H_tot += band.fetch('_CCRH', args=(EFr, T))
        return H_tot
    
    def compile(self, EF, T, max_level=2):
        '''Compile the object under specified Fermi energies (EF) and
        temperatures (T) to avoid redundant integration computations.
        The `max_level` parameter of integer type specifies the highest
        exponent for caching data, with a default value of 2.'''

        for band, delta in zip(self.bands, self.deltas):
            EFr = -1*band._q_sign*(EF-delta)
            band.compile(EFr, T, max_level)
        return super().compile(EF, T, max_level)
    
    def clear(self):
        '''Clear the cached data.'''
        for band in self.bands:
            band.clear()
        return super().clear()
    
    def Kbip(self, EF=None, T=None):
        '''Bipolar thermal conductivity, in W/(m.K).'''
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


class APSSPB(BaseBand):
    '''
    A class for describing single parabolic band (SPB) model when the
    acoustic phonon scattering (APS) mechanism predominates. In this
    model, there are three key parameters determine thermoelectric
    properties: (a) the effective mass of the density of states
    :math:`m_d^{\\ask}`, (b) the intrinsic electrical conductivity
    :math:`\\simga_0`, and (c) the ratio of longitudinal to transverse
    effective masses :math:`K^{\\ask}`. These parameters correspond
    to class attributes m_d, sigma0, and Kmass, respectively. The core
    of constructing the class is obtaining the values of these parameters.

    Attributes
    ----------
    m_d : float
        Effective mass of the density of states in :math:`m_e`, primarily
        influencing carrier concentration calculations. It should be a
        positive float.
    sigma0 : float
        Intrinsic electrical conductivity in `S/cm`, the core parameter
        influencing thermoelectric transport properties. It should be a
        positive float.
    Kmass : float
        The ratio of longitudinal to transverse effective mass, affecting
        calculations related to Hall coefficients. It should be a positive
        float.
    '''

    m_d = 1         # m_e
    sigma0 = 1      # S/cm
    Kmass = 1       # m1/m2
    def __init__(self, m_d=1, sigma0=1, Kmass=1):
        '''
        Initialize an instance by the specified parameters.

        Parameters
        ----------
        m_d : float, optional
            Effective mass of the density of states in static electron mass,
            by default 1.
        sigma0 : float, optional
            Intrinsic electrical conductivity in S/cm, by default 1.
        Kmass : float, optional
            The ratio of longitudinal to transverse effective mass, by
            default 1.
        '''
        self.m_d = m_d
        self.sigma0 = sigma0
        self.Kmass = Kmass
        
    def dos(self, E):
        '''Density of states, in 1E19 state/(eV.cm^3).'''
        factor = 1E-25      # state/(eV.m^3) --> 1E19 state/(eV.cm^3)
        g0 = np.power(2*self.m_d*m_e*q, 3/2)/(2*np.pi*np.pi* np.power(hbar, 3))
        return factor * g0 * np.sqrt(E)
    
    def trs(self, E, T):
        '''Transport distribution function, in S/cm.'''
        return self.sigma0 * E/(kB_eV*T)
    
    def hall(self, E, T):
        '''Hall transport distribution function, in S.cm/(V.s), i.e.
        [S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)].'''
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
        factor = 4.020521639724753      # [S/cm] / [cm^2/(V.s)]
        return self.sigma0 / factor
    
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

        # factor = np.sqrt(np.pi)/2 * q \
        #          * np.power(2*m_e*kB_eV*q*300, 3/2) \
        #          / (2*np.pi*np.pi*np.power(hbar, 3)) * 1E-4 /100  # S/cm
        factor = 4.020521639724753      # S/cm
        sigma0 = factor * UWT
        return cls(m_d=m_d, sigma0=sigma0, Kmass=Kmass)

    @classmethod
    def valuate_m_d(cls, dataS, dataN, dataT, hall=False, Kmass=1):
        spb = cls(Kmass=Kmass)
        dataEF = spb.slove_EF('S', dataS, dataT)
        if hall:
            N0 = spb.NH(dataEF, dataT)
        else:
            N0 = spb.N(dataEF, dataT)
        return np.power(dataN/N0, 2/3)

    @classmethod
    def valuate_L(cls, dataS):
        '''
        A class method to evaluate Lorenz numbers from Seebeck coefficients.

        Parameters
        ----------
        dataS : ndarray
            Absolute Seebeck coefficients in uV/K.

        Returns
        -------
        ndarray
            Lorenz numbers in 1E-8 W.Ohm/K^2.
        '''

        if np.any(dataS <= 0):
            raise ValueError('Non-negative values are required for dataS '
                             '(i.e. absolute Seebeck coefficient)')
        else:
            dataS = np.minimum(dataS, 1000)

        spb = cls(sigma0=100)
        TEMP = 1/kB_eV
        yita = spb.slove_EF('S', dataS, TEMP)
        L = spb.L(yita, TEMP)
        return spb.L(yita, TEMP)


class APSSKB(BaseBand):
    '''
    A class for describing single Kane band (SKB) model when the acoustic
    phonon scattering (APS) mechanism predominates. In contrast to the
    classical single parabolic band (SPB) model (see :class:`APSSPB`), an
    additional parameter describing the energy band shape, namely the bandgap
    (Eg), is introduced.

    Attributes
    ----------
    m_d : float
        Effective mass of the density of states in :math:`m_e`, primarily
        influencing carrier concentration calculations. It should be a
        positive float.
    sigma0 : float
        Intrinsic electrical conductivity in `S/cm`, the core parameter
        influencing thermoelectric transport properties. It should be a
        positive float.
    Eg : float
        Parameter bandgap in eV, which significantly influences various
        transport properties.
    Kmass : float
        The ratio of longitudinal to transverse effective mass, affecting
        calculations related to Hall coefficients. It should be a positive
        float.
    '''
    m_d = 1         # m_e
    sigma0 = 1      # S/cm
    Eg = 1          # eV
    Kmass = 1       # m1/m2
    def __init__(self, m_d=1, sigma0=1, Eg=1, Kmass=1):
        '''
        Initialize an instance by the specified parameters.

        Parameters
        ----------
        m_d : float, optional
            Effective mass of the density of states in static electron mass,
            by default 1.
        sigma0 : float, optional
            Intrinsic electrical conductivity in S/cm, by default 1.
        Eg : float, optional
            Bandgap in eV, by default 1.
        Kmass : float, optional
            The ratio of longitudinal to transverse effective mass, by
            default 1.
        '''
        self.m_d = m_d
        self.sigma0 = sigma0
        if Eg > 0:
            self.Eg = Eg
        else:
            raise ValueError('Eg should be greater than 0')
        self.Kmass = Kmass
        
    def dos(self, E):
        '''Density of states, in 1E19 state/(eV.cm^3).'''
        factor = 1E-25      # state/(eV.m^3) --> 1E19 state/(eV.cm^3)
        g0 = np.power(2*self.m_d*m_e*q, 3/2)/(2*np.pi*np.pi* np.power(hbar, 3))
        kane = np.sqrt(1+E/self.Eg) * (1+2*E/self.Eg)
        return factor * g0 * np.sqrt(E) * kane
    
    def trs(self, E, T):
        '''Transport distribution function, in S/cm.'''
        kane = 3*(1+E/self.Eg)/(np.power(1+2*E/self.Eg, 2)+2)
        return self.sigma0 * E/(kB_eV*T) * kane
    
    def hall(self, E, T):
        '''Hall transport distribution function, in S.cm/(V.s), i.e.
        [S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)].'''
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
        factor = 4.020521639724753      # [S/cm] / [cm^2/(V.s)]
        return self.sigma0 / factor
    
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

        # factor = np.sqrt(np.pi)/2 * q \
        #          * np.power(2*m_e*kB_eV*q*300, 3/2) \
        #          / (2*np.pi*np.pi*np.power(hbar, 3)) \
        #          * 1E-6 # C/m^3 --> C/cm^3
        factor = 4.020521639724753      # [S/cm] / [cm^2/(V.s)]
        sigma0 = factor * UWT
        return cls(m_d=m_d, sigma0=sigma0, Eg=Eg, Kmass=Kmass)