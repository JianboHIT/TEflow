#!/usr/bin/env python3

from abc import ABC, abstractmethod
from scipy.integrate import cumtrapz, trapz
import numpy as np


class BaseDevice(ABC):
    paras = dict()
    options = dict()
    profiles = dict()
    _itgs = None
    _mdfs = None
    _wgts = None
    
    def __init__(self, **paras):
        pass
        
    @abstractmethod
    def build(self, **options):
        pass
    
    @abstractmethod
    def simulate(self, Jx='optimal'):
        pass
    
    def __getattr__(self, name):
        if name in self.paras.keys():
            return self.paras[name]
        elif name in self.profiles.keys():
            return self.profiles[name]
        else:
            return super.__getattr__(name)
    
    @classmethod
    def _get_itgs(cls, datas, isCum=True):
        '''
        Calculate integrals of TE properties.

        Parameters
        ----------
        datas : list | ndarray
            TE datas like [T, C, S, K]
        isCum : bool, optional
            Whether to cumulatively integrate using the composite trapezoidal rule (default: True).

        Returns
        -------
        dict of ndarray
            Sc, Sh, Tc, Th, deltaT, itg_Rho, itg_S, itg_K, itg_RhoT, itg_ST
        '''
        
        T, C, S, K = datas
        Rho = 1E4 / C                       # S/cm to uOhm.m
        integral = cumtrapz if isCum else trapz
        
        itgs = dict()
        itgs['_isCum']  = isCum
        itgs['Sc']     = S[0]
        itgs['Sh']     = S[1:] if isCum else S[-1]
        itgs['Tc']     = T[0]
        itgs['Th']     = T[1:] if isCum else T[-1]
        itgs['deltaT'] = itgs['Th'] - itgs['Tc']
        itgs['Rho']    = integral(Rho, T)         # <Rho>
        itgs['S']      = integral(S, T)           # <S>
        itgs['K']      = integral(K, T)           # <K>
        itgs['RhoT']   = integral(T*Rho, T)       # <T*Rho>
        itgs['ST']     = integral(T*S, T)         # <T*S>
        cls._itgs = itgs
        return itgs
    
    @classmethod
    def _get_mdfs(cls, itgs):
        mdfs = dict()
        mdfs['_isCum'] = itgs['_isCum']
        mdfs['Rho']  = itgs['RhoT'] / itgs['Rho']   # <T*Rho>/<Rho>
        mdfs['S']  = itgs['ST'] / itgs['S']         # <T*S>/<S>
        mdfs['RhoW'] = itgs['RhoT'] - itgs['Tc']*itgs['Rho']  # <(T-Tc)*Rho>, weighted Rho
        mdfs['SW'] = 2*itgs['ST'] - itgs['Tc']*itgs['S']      # <(2T-Tc)*S>, weighted S
        mdfs['RhoT'] = mdfs['Rho'] - itgs['Tc']     # <(T-Tc)*Rho>/<Rho>, weighted DT/2 by Rho
        mdfs['ST'] = 2*mdfs['S'] - itgs['Tc']       # <(2T-Tc)*S>/<S>, weighted Th by S
        mdfs['ST_RhoT_0'] = mdfs['ST']          # modified Th
        mdfs['ST_RhoT_1'] = mdfs['ST_RhoT_0'] - mdfs['RhoT']    # modified Tave
        mdfs['ST_RhoT_2'] = mdfs['ST_RhoT_1'] - mdfs['RhoT']    # modified Tc
        cls._mdfs = mdfs
        return mdfs
    
    @classmethod
    def _get_wgts(cls, itgs, mdfs):
        if itgs['_isCum'] != mdfs['_isCum']:
            dsp = 'IsCums of itgs ({}) and mdfs ({}) are incompatible.'
            raise RuntimeError(dsp.format(itgs['_isCum'], mdfs['_isCum']))
        
        Rho_w = mdfs['Rho']
        TSc = itgs['Tc'] * itgs['Sc']
        TSh = itgs['Th'] * itgs['Sh']
        Tau_w = (TSh*itgs['deltaT'] - mdfs['SW']) / (TSh - TSc - itgs['S'])
        
        wgts = dict()
        wgts['_isCum'] = itgs['_isCum']
        wgts['W_J'] = Rho_w / itgs['deltaT']
        wgts['W_T'] = Tau_w / itgs['deltaT']
        wgts['alpha_0'] = mdfs['ST_RhoT_0'] / itgs['Sh']
        wgts['alpha_1'] = mdfs['ST_RhoT_1'] / itgs['Sh']
        wgts['alpha_2'] = mdfs['ST_RhoT_2'] / itgs['Sh']
        cls._wgts = wgts
        return wgts
 

class Generator(BaseDevice):
    paras = {
        'TEdatas': None,        # TE datas
        'L': 1,                 # length of TE leg
    }
    options = {
        'isCum': True,          # calculate only at the T_max (False) or all-T (True, default)
        'calWeights': False,    # whether to calculate dimensionless weight factors of Joule and Thomson heat
        'returnProfiles': False,
    }
    def __init__(self, **paras):
        self.paras.update(paras)

    def build(self, **options):
        self.options.update(options)
        options = self.options
        
        L = self.paras['L']
        datas = self.paras['TEdatas']
        
        itgs = self._get_itgs(datas, isCum=options['isCum'])
        mdfs = self._get_mdfs(itgs)
        
        deltaT = itgs['deltaT']
        PFeng = 1E-6 * itgs['S']*itgs['S']/itgs['Rho']
        Zeng = PFeng / itgs['K']
        ZTeng = Zeng * deltaT
        ZTp = Zeng * mdfs['ST_RhoT_1']
        m_opt = np.sqrt(1+ZTp)
        V_oc = 1E3 * itgs['S']        # mV
        Jd_sc = 1E5 * deltaT*itgs['S']/(itgs['Rho']*L)   # A/cm^2
        
        self.profiles = {
            '_isCum': options['isCum'],
            'deltaT': deltaT,
            'PFeng': PFeng,
            'ZTeng': ZTeng,
            'Zeng': Zeng,
            'ZTp': ZTp,
            'm_opt': m_opt,
            'V_oc': V_oc,
            'Jd_sc': Jd_sc,
        }
        
        if options['calWeights']:
            wgts = self._get_wgts(itgs, mdfs)
            self.profiles.update(wgts)
        
        if options['returnProfiles']:
            return self.profiles
        
    def simulate(self, Jd_r='optimal', numPoints=101):
        if isinstance(Jd_r, str):
            if Jd_r.lower().startswith('o'):
                Jd_r = None
            elif Jd_r.lower().startswith('s'):
                Jd_r = np.linspace(0, 1, numPoints)
        else:
            Jd_r = np.array(Jd_r)
        
        L = self.paras['L']
        PFeng = self.profiles['PFeng']
        Zeng = self.profiles['Zeng']
        deltaT = self.profiles['deltaT']
        mdfs = self._mdfs
        Qx = PFeng*deltaT/L
        
        outputs = dict()
        if Jd_r is None:
            m_opt = self.profiles['m_opt']
            outputs['Pd'] = 0.1 * Qx / 4     # W/cm^2, L = 1 mm
            outputs['Yita'] = 100 * deltaT * (m_opt-1)/(mdfs['ST_RhoT_0']*m_opt+mdfs['ST_RhoT_2'])
        else:
            if self.profiles['_isCum']:
                Jd_r = np.reshape(Jd_r, (-1,1))
            Vout_r = 1-Jd_r
            Pd_r = Jd_r * Vout_r
            Qhot_r = (1/self.profiles['Zeng'] + mdfs['ST'] * Jd_r - mdfs['RhoT']*Jd_r*Jd_r) / deltaT
            outputs['Jd'] = self.profiles['Jd_sc'] * Jd_r
            outputs['Vout'] = self.profiles['V_oc'] * Vout_r
            outputs['Pd'] = Qx * Pd_r
            outputs['Qhot'] = Qx * Qhot_r
            outputs['Yita'] = 100 * Pd_r/Qhot_r
        return outputs