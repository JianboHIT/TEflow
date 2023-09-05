from abc import ABC, abstractmethod
from scipy.integrate import quad
from scipy.optimize import root_scalar
import numpy as np

from .utils import kB_eV, e0


_CUTOFF_FD = 30

def _quad_fd(kernel, Y, T):
    # kernel(x, Y, T)
    def itg(y, t):
        left  = quad(kernel,
                     np.maximum(0, y-_CUTOFF_FD),
                     np.maximum(0, y),
                     args=(y, t))[0]
        right = quad(kernel, 
                     np.maximum(0, y),
                     np.maximum(0, y+_CUTOFF_FD),
                     args=(y, t))[0]
        return left + right
    return np.vectorize(itg)(Y, T)


class BaseBand(ABC):
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
        # x = E/(kB_eV*T),  E = x*kB_eV*T
        kernel = lambda E, _EF, _T: self.dos(E) \
                                    * self.fx((E-_EF)/(kB_eV*_T))
        itg = lambda _EF, _T: quad(kernel, 0, np.inf, args=(_EF, _T))[0]
        return np.vectorize(itg)(EF, T)
    
    def _K_n(self, __n, EF, T):
        '''S/cm'''
        # x = E/(kB_eV*T),  E = x*kB_eV*T, Y = EF/(kB_eV*T)
        n = round(__n)
        def kernel(x, Y, T):
            return self.trs(x*kB_eV*T, T) * self.dfx(x-Y, n)
        return _quad_fd(kernel, EF/(kB_eV*T), T)
    
    def _CCRH(self, EF, T):
        '''[S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)]'''
        def kernel(x, Y, T):
            return self.hall(x*kB_eV*T, T) * self.dfx(x-Y)
        return self._q_sign * _quad_fd(kernel, EF/(kB_eV*T), T)
   
    def compile(self, EF, T, max_level=2):
        self._caching = {
            '_EF': EF,
            '_T': T,
            '_N': self._N(EF, T),
            '_K_n': [self._K_n(i, EF, T) for i in range(max_level+1)],
            '_CCRH': self._CCRH(EF, T),
        }
    
    def clear(self):
        self._caching = None
    
    def fetch(self, _prop, args=(), index=None, default=None):
        if self._caching:
            if _prop not in self._caching:
                raise KeyError(f'Failed to read uncompiled {_prop}')
            if any(arg is not None for arg in args):
                raise ValueError(f'Unusable arguments for cached {_prop}')
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
        '''1E19 cm^(-3)'''
        return self.fetch('_N', args=(EF, T))
    
    def K_0(self, EF=None, T=None):
        '''S/cm'''
        return self.fetch('_K_n', args=(EF, T), index=0)
    
    def K_1(self, EF=None, T=None):
        '''S/cm'''
        return self.fetch('_K_n', args=(EF, T), index=1)
    
    def K_2(self, EF=None, T=None):
        '''S/cm'''
        return self.fetch('_K_n', args=(EF, T), index=2)
    
    def CCRH(self, EF=None, T=None):
        '''[S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)]'''
        return self.fetch('_CCRH', args=(EF, T))
    
    def C(self, EF=None, T=None):
        '''S/cm'''
        p0 = self.K_0(EF, T)
        return p0
    
    def CS(self, EF=None, T=None):
        '''[S/cm]*[uV/K]'''
        p1 = self.K_1(EF, T)
        return self._q_sign * 1E6 * kB_eV * p1
    
    def S(self, EF=None, T=None):
        '''uV/K'''
        p0 = self.K_0(EF, T)
        p1 = self.K_1(EF, T)
        return self._q_sign * 1E6 * kB_eV * p1/p0
    
    def PF(self, EF=None, T=None):
        '''uW/(cm.K^2)'''
        p0 = self.K_0(EF, T)
        p1 = self.K_1(EF, T)
        pr = np.power(p1, 2) / p0
        return 1E6 * kB_eV * kB_eV * pr
    
    def CL(self, EF=None, T=None):
        '''[W/(m.K)] / [T]'''
        p0 = self.K_0(EF, T)
        p1 = self.K_1(EF, T)
        p2 = self.K_2(EF, T)
        pr = p2 - np.power(p1, 2)/p0
        return 1E2 * kB_eV * kB_eV * pr
    
    def L(self, EF=None, T=None):
        '''1E-8 W.Ohm/K^2'''
        p0 = self.K_0(EF, T)
        p1 = self.K_1(EF, T)
        p2 = self.K_2(EF, T)
        pr = p2/p0 - np.power(p1/p0, 2)
        return 1E8 * kB_eV * kB_eV * pr
    
    def Ke(self, EF=None, T=None):
        '''W/(m.K)'''
        p0 = self.K_0(EF, T)
        p1 = self.K_1(EF, T)
        p2 = self.K_2(EF, T)
        pr = p2 - np.power(p1, 2)/p0
        pT = self.fetch('_T', default=T)
        return 1E2 * kB_eV * kB_eV * pr * pT
    
    def U(self, EF=None, T=None):
        '''cm^2/(V.s)'''
        pC = self.K_0(EF, T)     # S/cm
        pN = self.N(EF, T)          # 1E19 cm^-3
        pQ = self._q_sign * e0
        return pC/(pQ*pN*1E19)
    
    def RH(self, EF=None, T=None):
        '''cm^3/C'''
        return self.CCRH(EF, T)/np.power(self.K_0(EF, T), 2)
    
    def UH(self, EF=None, T=None):
        '''cm^2/(V.s)'''
        return self.CCRH(EF, T)/self.K_0(EF, T)
    
    def NH(self, EF=None, T=None):
        '''1E19 cm^-3'''
        pQ = self._q_sign * e0
        return 1E-19*np.power(self.K_0(EF, T), 2)/self.CCRH(EF, T)/pQ

    def slove_EF(self, prop, value, T, near=0, between=None, **kwargs):
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
        return 1/2*(1-np.tanh(x/2))
    
    @staticmethod
    def dfx(x, k=0):
        return np.power(x, k) * 1/4*(1-np.power(np.tanh(x/2), 2))


class MultiBand(BaseBand):
    cacheable = BaseBand.cacheable | {'Kbip'}

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
            N_tot += band.fetch('_N', args=(EFr, T))
        return N_tot
    
    def _K_n(self, __n, EF, T):
        K_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            EFr = -1*band._q_sign*(EF-delta)
            K_tot += np.power(band._q_sign, __n) \
                     * band.fetch('_K_n', args=(EFr, T), index=__n)
        return K_tot
    
    def _CCRH(self, EF, T):
        H_tot = 0
        for band, delta in zip(self.bands, self.deltas):
            EFr = -1*band._q_sign*(EF-delta)
            H_tot += band.fetch('_CCRH', args=(EFr, T))
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
    
    def Kbip(self, EF=None, T=None):
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
