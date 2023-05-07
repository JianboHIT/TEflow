from abc import ABC, abstractmethod
from scipy.integrate import quad
import numpy as np

from .utils import kB, q0


class BaseBand(ABC):
    _q_sign = 1
    _caching = None
    
    @abstractmethod
    def dos(self, E):
        '''1E19 state/(eV.cm^3)'''
        pass
    
    @abstractmethod
    def trs(self, E, T):
        '''S/cm'''
        pass
    
    @abstractmethod
    def hall(self, E, T):
        '''[S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)]'''
        pass
    
    def _N(self, EF, T):
        '''1E19 cm^(-3)'''
        # x = E/(kB*T),  E = x*kB*T
        kernel = lambda E, _EF, _T: self.dos(E) \
                                    * self.fx((E-_EF)/(kB*_T))
        itg = lambda _EF, _T: quad(kernel, 0, np.inf, args=(_EF, _T))[0]
        return np.vectorize(itg)(EF, T)
    
    def _K_n(self, __n, EF, T):
        '''S/cm'''
        # x = E/(kB*T),  E = x*kB*T
        __n = round(__n)
        kernel = lambda x, _EF, _T: self.trs(x*kB*_T, _T) \
                                    * self.dfx(x - _EF/(kB*_T), __n)
        itg = lambda _EF, _T: quad(kernel, 0, np.inf, args=(_EF, _T))[0]
        return np.vectorize(itg)(EF, T)
    
    def _CCRH(self, EF, T):
        '''[S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)]'''
        kernel = lambda x, _EF, _T: self.hall(x*kB*_T, _T) \
                                    * self.dfx(x - _EF/(kB*_T))
        itg = lambda _EF, _T: quad(kernel, 0, np.inf, args=(_EF, _T))[0]
        return self._q_sign * np.vectorize(itg)(EF, T)
    
    def compile(self, EF, T, max_level=2):
        self._caching = {
            '_N': self._N(EF, T),
            '_K_n': [self._K_n(i, EF, T) for i in range(max_level+1)],
            '_CCRH': self._CCRH(EF, T),
        }
    
    def clear(self):
        self._caching = None
    
    def compute(self, __prop, args=(), index=None):
        if self._caching:
            if __prop not in self._caching:
                raise KeyError(f'Failed to read uncompiled {__prop}')
            if index is None:
                return self._caching[__prop]
            else:
                return self._caching[__prop][index]
        else:
            return getattr(self, __prop)(*args)
    
    def __getitem__(self, key):
        if not self._caching:
            raise RuntimeError('Uncompiled class')
        elif not hasattr(self, key):
            raise KeyError(f'Failed to read undefined {key}')
        elif key.startswith('_'):
            raise KeyError(f'Failed to read protected {key}')
        elif key in {'dos', 'trs', 'hall', 'compile', 'clear', 'compute'}:
            raise KeyError(f'Failed to read metadata {key}')
        else:
            return getattr(self, key)()
    
    def N(self, EF=None, T=None):
        '''1E19 cm^(-3)'''
        return self.compute('_N', args=(EF, T))
    
    def K_0(self, EF=None, T=None):
        '''S/cm'''
        return self.compute('_K_n', args=(0, EF, T), index=0)
    
    def K_1(self, EF=None, T=None):
        '''S/cm'''
        return self.compute('_K_n', args=(1, EF, T), index=1)
    
    def K_2(self, EF=None, T=None):
        '''S/cm'''
        return self.compute('_K_n', args=(2, EF, T), index=2)
    
    def CCRH(self, EF=None, T=None):
        '''[S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)]'''
        return self.compute('_CCRH', args=(EF, T))
    
    def C(self, EF=None, T=None):
        '''S/cm'''
        p0 = self.K_0(EF, T)
        return p0
    
    def CS(self, EF=None, T=None):
        '''[S/cm]*[uV/K]'''
        p1 = self.K_1(EF, T)
        return self._q_sign * 1E6 * kB * p1
    
    def S(self, EF=None, T=None):
        '''uV/K'''
        p0 = self.K_0(EF, T)
        p1 = self.K_1(EF, T)
        return self._q_sign * 1E6 * kB * p1/p0
    
    def PF(self, EF=None, T=None):
        '''uW/(cm.K^2)'''
        p0 = self.K_0(EF, T)
        p1 = self.K_1(EF, T)
        pr = np.power(p1, 2) / p0
        return 1E6 * kB * kB * pr
    
    def L(self, EF=None, T=None):
        '''1E-8 W.Ohm/K^2'''
        p0 = self.K_0(EF, T)
        p1 = self.K_1(EF, T)
        p2 = self.K_2(EF, T)
        pr = p2/p0 - np.power(p1/p0, 2)
        return 1E8 * kB * kB * pr
    
    def Ke(self, EF=None, T=None):
        '''W/(m.K)'''
        p0 = self.K_0(EF, T)
        p1 = self.K_1(EF, T)
        p2 = self.K_2(EF, T)
        pr = p2 - np.power(p1, 2)/p0
        return 1E2 * kB * kB * pr * T
    
    def U(self, EF=None, T=None):
        '''cm^2/(V.s)'''
        pC = self.K_0(EF, T)     # S/cm
        pN = self.N(EF, T)          # 1E19 cm^-3
        pQ = self._q_sign * q0
        return pC/(pQ*pN)
    
    def RH(self, EF=None, T=None):
        '''cm^3/C'''
        return self.CCRH(EF, T)/np.power(self.K_0(EF, T), 2)
    
    def UH(self, EF=None, T=None):
        '''cm^2/(V.s)'''
        return self.CCRH(EF, T)/self.K_0(EF, T)
    
    def NH(self, EF=None, T=None):
        '''1E19 cm^-3'''
        pQ = self._q_sign * q0
        return np.power(self.K_0(EF, T), 2)/self.CCRH(EF, T)/pQ
    
    @staticmethod
    def fx(x):
        return 1/2*(1-np.tanh(x/2))
    
    @staticmethod
    def dfx(x, k=0):
        return np.power(x, k) * 1/4*(1-np.power(np.tanh(x/2), 2))


class MultiBand(BaseBand):
    def __init__(self, bands=(), offsets=(), bands2=(), offsets2=()):
        dsp = 'Length of {} is not the same as the number of {}'
        if len(bands) != len(offsets):
            raise ValueError(dsp.format('bands', 'offsets'))
        if len(bands2) != len(offsets2):
            raise ValueError(dsp.format('bands2', 'offsets2'))
        self._bands = bands
        self._bands2 = bands2
        self._offsets = offsets
        self._offsets2 = offsets2
    
    @property
    def bands(self):
        return tuple(self._bands), tuple(self._bands2)
    
    @property
    def offsets(self):
        return tuple(self._offsets), tuple(self._offsets2)
    
    def dos(self, E):
        dos_tot = 0
        for bd, dt in zip(self._bands, self._offsets):
            dos_tot += bd.dos(np.maximum(E-dt, 0))
        for bd, dt in zip(self._bands2, self._offsets2):
            dos_tot += bd.dos(np.maximum(dt-E, 0))
        return dos_tot
    
    def trs(self, E, T):
        trs_tot = 0
        for bd, dt in zip(self._bands, self._offsets):
            trs_tot += bd.trs(np.maximum(E-dt, 0), T)
        for bd, dt in zip(self._bands2, self._offsets2):
            trs_tot += bd.trs(np.maximum(dt-E, 0), T)
        return trs_tot
    
    def hall(self, E, T):
        hall_tot = 0
        for bd, dt in zip(self._bands, self._offsets):
            hall_tot += bd.hall(np.maximum(E-dt, 0), T)
        for bd, dt in zip(self._bands2, self._offsets2):
            hall_tot += bd.hall(np.maximum(dt-E, 0), T)
        return hall_tot
    
    def _N(self, EF, T):
        N_tot, N_tot2 = 0, 0
        for bd, dt in zip(self._bands, self._offsets):
            N_tot += bd.compute('_N', args=(EF-dt, T))
        for bd, dt in zip(self._bands2, self._offsets2):
            N_tot2 += bd.compute('_N', args=(dt-EF, T))
        return N_tot + N_tot2
    
    def _K_n(self, __n, EF, T):
        K_tot, K_tot2 = 0, 0
        for bd, dt in zip(self._bands, self._offsets):
            K_tot += bd.compute('_K_n', args=(__n, EF-dt, T), index=__n)
        for bd, dt in zip(self._bands2, self._offsets2):
            K_tot2 += bd.compute('_K_n', args=(__n, dt-EF, T), index=__n)
        return K_tot + np.power(-1, __n) * K_tot2
    
    def _CCRH(self, EF, T):
        H_tot, H_tot2 = 0, 0
        for bd, dt in zip(self._bands, self._offsets):
            H_tot += bd.compute('_CCRH', args=(EF-dt, T))
        for bd, dt in zip(self._bands2, self._offsets2):
            H_tot2 += bd.compute('_CCRH', args=(dt-EF, T))
        return H_tot - H_tot2
    
    def compile(self, EF, T, max_level=2):
        for bd, dt in zip(self._bands, self._offsets):
            bd.compile(EF-dt, T, max_level)
        for bd, dt in zip(self._bands2, self._offsets2):
            bd.compile(dt-EF, T, max_level)
        return super().compile(EF, T, max_level)
    
    def clear(self):
        for bd in self._bands:
            bd.clear()
        for bd in self._bands2:
            bd.clear()
        return super().clear()
