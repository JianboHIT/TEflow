from abc import ABC, abstractmethod
from scipy.integrate import quad
import numpy as np

from .utils import kB, q0


class BaseBand(ABC):
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
    
    def __N(self, EF, T):
        '''1E19 cm^(-3)'''
        # x = E/(kB*T),  E = x*kB*T
        kernel = lambda E: self.dos(E) * self.fx((E-EF)/(kB*T))
        return quad(kernel, 0, np.inf)[0]
    
    def N(self, EF, T):
        '''1E19 cm^(-3)'''
        if self._caching:
            return self._caching['N']
        else:
            return np.vectorize(self.__N)(EF, T)
    
    def __K_n(self, __n, EF, T):
        '''S/cm'''
        # x = E/(kB*T),  E = x*kB*T
        kernel = lambda x: self.trs(x*kB*T, T) * self.dfx(x - EF/(kB*T), __n)
        return quad(kernel, 0, np.inf)[0]
    
    def K_n(self, __n, EF, T):
        '''S/cm'''
        if self._caching:
            return self._caching['K_n'][__n]
        else:
            return np.vectorize(self.__K_n, excluded=['__n', ])(__n, EF, T)

    def __CCRH(self, EF, T):
        '''[S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)]'''
        kernel = lambda x: self.hall(x*kB*T, T) * self.dfx(x - EF/(kB*T))
        return quad(kernel, 0, np.inf)[0]
    
    def CCRH(self, EF, T):
        '''[S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)]'''
        if self._caching:
            return self._caching['CCRH']
        else:
            return np.vectorize(self.__CCRH)(EF, T)
    
    def compile(self, EF, T, max_level=2):
        self._caching = {
            'args': (EF, T),
            'N': self.N(EF, T),
            'CCRH': self.CCRH(EF, T),
            'K_n': [self.K_n(i, EF, T) for i in range(max_level+1)],
            'property': ('N', 'CCRH', 'C', 'CS', 'S', 'PF', 'L', 'Ke', 'U', 'RH'),
        }
    
    def clear(self):
        self._caching = None
    
    def __getattribute__(self, attr):
        caching = super().__getattribute__('_caching')
        if caching and (attr in caching['property']):
            return super().__getattribute__(attr)(*caching['args'])
        else:
            return super().__getattribute__(attr)
    
    def C(self, EF, T):
        '''S/cm'''
        p0 = self.K_n(0, EF, T)
        return p0
    
    def CS(self, EF, T):
        '''[S/cm]*[uV/K]'''
        p1 = self.K_n(1, EF, T)
        return 1E6 * kB * p1
    
    def S(self, EF, T):
        '''uV/K'''
        p0 = self.K_n(0, EF, T)
        p1 = self.K_n(1, EF, T)
        return 1E6 * kB * p1/p0
    
    def PF(self, EF, T):
        '''uW/(cm.K^2)'''
        p0 = self.K_n(0, EF, T)
        p1 = self.K_n(1, EF, T)
        pr = np.power(p1, 2) / p0
        return 1E6 * kB * kB * pr
    
    def L(self, EF, T):
        '''1E-8 W.Ohm/K^2'''
        p0 = self.K_n(0, EF, T)
        p1 = self.K_n(1, EF, T)
        p2 = self.K_n(2, EF, T)
        pr = p2/p0 - np.power(p1/p0, 2)
        return 1E8 * kB * kB * pr
    
    def Ke(self, EF, T):
        '''W/(m.K)'''
        p0 = self.K_n(0, EF, T)
        p1 = self.K_n(1, EF, T)
        p2 = self.K_n(2, EF, T)
        pr = p2 - np.power(p1, 2)/p0
        return 1E2 * kB * kB * pr * T
    
    def U(self, EF, T):
        '''cm^2/(V.s)'''
        pC = self.K_n(0, EF, T)     # S/cm
        pN = self.N(EF, T)          # 1E19 cm^-3
        return pC/(q0*pN)
    
    def RH(self, EF, T):
        '''cm^3/C'''
        return self.CCRH(EF, T)/np.power(self.K_n(0, EF, T), 2)
    
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
    
    def K_n(self, __n, EF, T):
        if self._caching:
            K_tot = self._caching['K_n'][__n]
        else:
            K_tot = 0
            for bd, dt in zip(self._bands, self._offsets):
                K_tot += bd.K_n(__n, EF-dt, T)
            for bd, dt in zip(self._bands2, self._offsets2):
                K_tot += np.power(-1, __n) * bd.K_n(__n, dt-EF, T)
        return K_tot
    
    def N(self, EF, T):
        if self._caching:
            N_tot = self._caching['N']
        else:
            N_tot = 0
            for bd, dt in zip(self._bands, self._offsets):
                N_tot += bd.N(EF-dt, T)
            for bd, dt in zip(self._bands2, self._offsets2):
                N_tot += bd.N(dt-EF, T)
        return N_tot
    
    def CCRH(self, EF, T):
        if self._caching:
            H_tot = self._caching['CCRH']
        else:
            H_tot = 0
            for bd, dt in zip(self._bands, self._offsets):
                H_tot += bd.CCRH(EF-dt, T)
            for bd, dt in zip(self._bands2, self._offsets2):
                H_tot += bd.CCRH(dt-EF, T)
        return H_tot