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
    def __init__(self, cbands=(), cdeltas=(), 
                       vbands=(), vdeltas=(),):
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
        dos_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            Er = np.maximum(-1*band._q_sign*(E-delta), 0)
            dos_tot += band.dos(Er)
        return dos_tot
    
    def trs(self, E, T):
        trs_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            Er = np.maximum(-1*band._q_sign*(E-delta), 0)
            trs_tot += band.trs(Er, T)
        return trs_tot
    
    def hall(self, E, T):
        hall_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            Er = np.maximum(-1*band._q_sign*(E-delta), 0)
            hall_tot += band.hall(Er, T)
        return hall_tot
    
    def _N(self, EF, T):
        N_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            EFr = -1*band._q_sign*(EF-delta)
            N_tot += band.compute('_N', args=(EFr, T))
        return N_tot
    
    def _K_n(self, __n, EF, T):
        K_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            EFr = -1*band._q_sign*(EF-delta)
            K_tot += np.power(band._q_sign, __n) \
                     * band.compute('_K_n', args=(__n, EFr, T), index=__n)
        return K_tot
    
    def _CCRH(self, EF, T):
        H_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            EFr = -1*band._q_sign*(EF-delta)
            H_tot += band.compute('_CCRH', args=(EFr, T))
        return H_tot
    
    def compile(self, EF, T, max_level=2):
        for band, delta in zip(self.bands, self.deltas):
            EFr = -1*band._q_sign*(EF-delta)
            band.compile(EFr, T, max_level)
        return super().compile(EF, T, max_level)
    
    def clear(self):
        for band in self.bands:
            band.clear()
        return super().clear()