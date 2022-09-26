#!/usr/bin/env python3

import logging
from abc import ABC, abstractmethod
from pprint import pformat
from scipy.integrate import cumtrapz, trapz
import numpy as np

LEVEL = logging.DEBUG
FORMAT = '%(asctime)s [%(levelname)s @ %(name)s]: %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(format=FORMAT, level=LEVEL, datefmt=DATEFMT)
logger = logging.getLogger(__name__)

class BaseDevice(ABC):
    paras = dict()
    options = dict()
    profiles = dict()
    outputs = dict()
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
    
    @classmethod
    @abstractmethod
    def valuate(cls, **kwargs):
        pass
    
    def __getattr__(self, name):
        if name in self.paras.keys():
            return self.paras[name]
        elif name in self.profiles.keys():
            return self.profiles[name]
        elif name in self.outputs.keys():
            return self.outputs[name]
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
        logger.info('Begin initialization ...')
        self.paras.update(paras)
        logger.info('Read parameters of device')
        logger.debug('Parameters:\n%s', pformat(self.paras))
        logger.info('Finish initialization')

    def build(self, **options):
        logger.info('Begin building process ...')
        self.options.update(options)
        options = self.options
        logger.info('Read options')
        logger.debug('Options:\n%s', pformat(options))
        
        L = self.paras['L']
        datas = self.paras['TEdatas']
        
        itgs = self._get_itgs(datas, isCum=options['isCum'])
        mdfs = self._get_mdfs(itgs)
        logger.info('Calculate integrals and modification factors')
        
        deltaT = itgs['deltaT']
        PFeng = 1E-6 * itgs['S']*itgs['S']/itgs['Rho']
        Zeng = PFeng / itgs['K']
        ZTeng = Zeng * deltaT
        ZTp = Zeng * mdfs['ST_RhoT_1']
        m_opt = np.sqrt(1+ZTp)
        Voc = 1E-3 * itgs['S']        # mV
        Jsc = 1E-1 * deltaT*itgs['S']/(itgs['Rho']*L)   # A/cm^2
        Qx = 0.1 * PFeng*deltaT/L      # W/cm^2, Qflux
        
        self.profiles = {
            '_isCum': options['isCum'],
            'deltaT': deltaT,
            'PFeng': PFeng,
            'ZTeng': ZTeng,
            'Zeng': Zeng,
            'ZTp': ZTp,
            'm_opt': m_opt,
            'Voc': Voc,
            'Jsc': Jsc,
            'Qx': Qx,
        }
        logger.info('Calculate profiles of device')
        logger.debug('Keys of profiles:\n%s', pformat(self.profiles.keys()))
        
        if options['calWeights']:
            wgts = self._get_wgts(itgs, mdfs)
            self.profiles.update(wgts)
            logger.info('Calculate weight factors')
        else:
            logger.debug('Ingore calculation of weight factors')
        
        logger.info('Finish building process')
        if options['returnProfiles']:
            return self.profiles
        
    def simulate(self, Jd_r='optimal', numPoints=101, returnOutputs=False):
        logger.info('Begin simulating ...')
        if isinstance(Jd_r, str):
            if Jd_r.lower().startswith('o'):
                Jd_r = None
                logger.info('Work under optimial current density')
            elif Jd_r.lower().startswith('s'):
                Jd_r = np.linspace(0, 1, numPoints)
                logger.info('Work under auto-limition of current density')
        else:
            Jd_r = np.array(Jd_r)
            logger.info('Work under assigned current density')
        
        deltaT = self.profiles['deltaT']
        Qx = self.profiles['Qx']
        mdfs = self._mdfs
        logger.info('Read out deltaT, Qx, and mdfs')
        
        outputs = dict()
        if Jd_r is None:
            m_opt = self.profiles['m_opt']
            outputs['Pd'] = 1/4 * Qx     # W/cm^2
            outputs['Yita'] = 100 * deltaT * (m_opt-1)/(mdfs['ST_RhoT_0']*m_opt+mdfs['ST_RhoT_2'])
            logger.info('Calculate Pd and Yita')
        else:
            if self.profiles['_isCum']:
                Jd_r = np.reshape(Jd_r, (-1,1))
                logger.debug('Reshape Jd_r to (-1,1)')
            Vout_r = 1-Jd_r
            Pd_r = Jd_r * Vout_r
            Qhot_rt = (1/self.profiles['Zeng'] + mdfs['ST'] * Jd_r - mdfs['RhoT']*Jd_r*Jd_r)
            outputs['Jd'] = self.profiles['Jsc'] * Jd_r
            outputs['Vout'] = self.profiles['Voc'] * Vout_r
            outputs['Pd'] = Qx * Pd_r
            outputs['Qhot'] = Qx/deltaT * Qhot_rt
            outputs['Yita'] = 100 * deltaT * Pd_r / Qhot_rt
            logger.info('Calculate Jd, Vout, Pd, Qhot, and Yita')
        
        self.outputs = outputs
        logger.info('Finish simulating process')
        if returnOutputs:
            return outputs
    
    @classmethod
    def valuate(cls, datas, L=1):
        logger.info('Quick calculate engineering output performace')
        logger.info('Initializing ...')
        gen = cls(TEdatas=datas, L=L)
        
        logger.info('Building device ...')
        gen.build()
        
        logger.info('Simulating ...')
        gen.simulate()
        
        logger.info('Finished. (PFeng, ZTeng, Pd, Yita)')
        return gen.PFeng, gen.ZTeng, gen.Pd, gen.Yita
        
        