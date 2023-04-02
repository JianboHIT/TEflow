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

import logging
import numpy as np
from pprint import pformat
from abc import ABC, abstractmethod
from scipy.integrate import cumtrapz
from numpy.polynomial import Polynomial

from .utils import AttrDict

logger = logging.getLogger(__name__)

class BaseDevice(ABC):
    '''
    An abstract class for computing engineering performance of thermoelectric device
    '''
    
    paras    = dict()   # input-init
    options  = dict()   # input-build
    configs  = dict()   # input-simulate
    profiles = dict()   # output-build
    outputs  = dict()   # output-simulate
    
    def __init__(self, **paras):
        self.paras.update(paras)
        pass
    
    def __getattr__(self, name):
        if name in self.paras:
            return self.paras[name]
        elif name in self.profiles:
            return self.profiles[name]
        elif name in self.outputs:
            return self.outputs[name]
        else:
            return super().__getattribute__(name)
     
    @abstractmethod
    def build(self, **options):
        self.options.update(options)
        return self.profiles
    
    @abstractmethod
    def simulate(self, **configs):
        self.configs.update(configs)
        return self.outputs
    
    @classmethod
    @abstractmethod
    def valuate(cls, **kwargs):
        pass

class GenCore(BaseDevice):
    '''
    An abstract class for computing engineering performance of thermoelectric generator
    '''
    paras = {
        'isPoly': False,        # mode
        'TEdatas': None,        # TE datas
        'Tc': None,             # temperature at cold side
        'Th': None,             # temperature at hot side
        'L': 1,                 # length of TE leg in mm
        'A': 100,               # the cross-sectional area in mm^2, default 1 cm^2
    }
    options = {
        'calWeights': False,    # whether to calculate dimensionless weight factors
        'returnProfiles': False,
    }
    configs = {
        'I_r': 'YitaMax',       # 'YitaMax' | 'PowerMax' | 'scan'(or 'sweep') | array_like
        'numPoints': 101,
        'returnOutputs': False,
    }
    
    def _build(self, options):
        '''
        options : dict
            'calWeights': False,    # whether to calculate dimensionless weight factors
            'returnProfiles': False,
        '''

        self.options.update(options)
        options = self.options
        logger.info('Read options ...')
        logger.debug('Options:\n%s', pformat(options))
        
        itgs = self._get_itgs(datas=self.paras['TEdatas'],
                              Tc=self.paras['Tc'],
                              Th=self.paras['Th'],
                              isPoly=self.paras['isPoly'])

        mdfs = self._get_mdfs(itgs)
        logger.info('Calculate integrals and modification factors')
        
        L = self.paras['L']     # mm
        A = self.paras['A']     # mm^2
        profiles = self._get_prfs(itgs, mdfs, L, A)
        logger.info('Calculate profiles of device')
        logger.debug('Keys of profiles:\n%s', pformat(list(profiles.keys())))
        
        if options['calWeights']:
            wgts = self._get_wgts(itgs, mdfs)
            profiles['wgts'] = wgts
            logger.info('Calculate weight factors')
        else:
            logger.debug('Ingore calculation of weight factors')
        
        self.profiles = profiles
        if options['returnProfiles']:
            return profiles
        else:
            return None
        
    def _simulate(self, configs):
        '''
        configs : dict
            'I_r': 'YitaMax',       # 'YitaMax' | 'PowerMax' | 'scan'(or 'sweep') | array_like
            'numPoints': 101,
            'returnOutputs': False,
        '''

        self.configs.update(configs)
        configs = self.configs
        logger.info('Read configs ...')
        logger.debug('Configs:\n%s', pformat(configs))
        
        # parse I_r
        I_r = configs['I_r']
        if isinstance(I_r, str):
            if I_r.lower().startswith('y'):
                m_opt = self.profiles['m_opt']
                I_r = 1/(1+m_opt)
                logger.info('Work under optimized efficiency')
            elif I_r.lower().startswith('p'):
                I_r = np.array([1/2,])
                logger.info('Work under optimized output power')
            elif I_r.lower().startswith('s'):
                shp = [-1,]+[1,]*self.profiles['deltaT']
                numPoints = configs['numPoints']
                I_r = np.linspace(0, 1, numPoints).reshape(shp)
                logger.info('Work under auto-limition of current')
            else:
                raise ValueError('Invalid value of I_r')
        else:
            I_r = np.atleast_1d(I_r)
            logger.info('Work under assigned reduced current')
        logger.debug('Shape of I_r: %s', str(I_r.shape))
        
        logger.info('Load profiles datas of device')
        prfs = self.profiles
        deltaT = prfs['deltaT']
        Zeng   = prfs['Zeng']
        Isc    = prfs['Isc']
        Voc    = prfs['Voc']
        Qx     = prfs['Qx']
        logger.debug('Read out deltaT, Zeng, Isc, Voc, Qx')
        
        mdfs = prfs['mdfs']
        mdf_ST = mdfs['ST']
        mdf_RhoT = mdfs['RhoT']
        logger.debug('Read out mdf_ST, mdf_RhoT')
        
        outputs = AttrDict()
        Vout_r = 1-I_r
        Pout_r = I_r * Vout_r
        Qhot_rt = (1/Zeng + mdf_ST * I_r - mdf_RhoT*I_r*I_r)
        outputs['I'] = Isc * I_r
        outputs['Vout'] = Voc * Vout_r
        outputs['Pout'] = Qx * Pout_r 
        outputs['Qhot'] = Qx * Qhot_rt/deltaT
        outputs['Yita'] = 100 * outputs['Pout'] / outputs['Qhot']
        logger.info('Calculate Vout, Pout, Qhot, and Yita')
        
        self.outputs = outputs
        if configs['returnOutputs']:
            return outputs
        else:
            return None
    
    @staticmethod
    def _get_itgs(datas, Tc, Th, isPoly):
        '''
        calculate integrals of porperties
            datas = datas_TCSK  if isPoly is False
            datas = datas_RhoSK if isPoly is True
        '''
        
        itgs = AttrDict()
        itgs['Tc']     = Tc
        itgs['Th']     = Th
        itgs['deltaT'] = Th - Tc
        
        if isPoly:
            Rho, S, K = datas
            T = Polynomial([0,1])
            RhoT = T * Rho
            ST   = T * Rho
            Rho_itg = Rho.integ()
            S_itg = S.integ()
            K_itg = K.integ()
            RhoT_itg = RhoT.integ()
            ST_itg = ST.integ()
            
            itgs['Sc'] = S(Tc)
            itgs['Sh'] = S(Th)
            itgs['Rho']  = Rho_itg(Th)  - Rho_itg(Tc)       # <Rho>
            itgs['S']    = S_itg(Th)    - S_itg(Tc)         # <S>
            itgs['K']    = K_itg(Th)    - K_itg(Tc)         # <K>
            itgs['RhoT'] = RhoT_itg(Th) - RhoT_itg(Tc)      # <T*Rho>
            itgs['ST']   = ST_itg(Th)   - ST_itg(Tc)        # <T*S>
        else:
            T, C, S, K = datas
            Rho = 1E4 / C           # S/cm to uOhm.m
            RhoT = T * Rho
            ST   = T * S
            Rho_itg = cumtrapz(Rho, T, initial=0)
            S_itg = cumtrapz(S, T, initial=0)
            K_itg = cumtrapz(K, T, initial=0)
            RhoT_itg = cumtrapz(RhoT, T, initial=0)
            ST_itg = cumtrapz(ST, T, initial=0)
            
            interp = lambda Pint, Ti: np.interp(Ti, T, Pint)
            
            itgs['Sc'] = interp(S, Tc)
            itgs['Sh'] = interp(S, Th)
            itgs['Rho']  = interp(Rho_itg, Th)  - interp(Rho_itg, Tc)
            itgs['S']    = interp(S_itg, Th)    - interp(S_itg, Tc)
            itgs['K']    = interp(K_itg, Th)    - interp(K_itg, Tc)
            itgs['RhoT'] = interp(RhoT_itg, Th) - interp(RhoT_itg, Tc)
            itgs['ST']   = interp(ST_itg, Th)   - interp(ST_itg, Tc)
        return itgs

    @staticmethod
    def _get_mdfs(itgs):
        '''
        calculate modification factors
        '''
        
        mdfs = AttrDict()
        mdfs['Rho']  = itgs['RhoT'] / itgs['Rho']               # <T*Rho>/<Rho>
        mdfs['S']    =   itgs['ST'] / itgs['S']                 # <T*S>/<S>
        mdfs['RhoW'] = itgs['RhoT'] - itgs['Tc']*itgs['Rho']    # <(T-Tc)*Rho>, weighted Rho
        mdfs['SW']   = 2*itgs['ST'] - itgs['Tc']*itgs['S']      # <(2T-Tc)*S>, weighted S
        mdfs['RhoT'] = mdfs['Rho'] - itgs['Tc']       # <(T-Tc)*Rho>/<Rho>, weighted DT/2 by Rho
        mdfs['ST']   = 2*mdfs['S'] - itgs['Tc']       # <(2T-Tc)*S>/<S>, weighted Th by S
        mdfs['ST_RhoT_0'] = mdfs['ST']                          # modified Th
        mdfs['ST_RhoT_1'] = mdfs['ST_RhoT_0'] - mdfs['RhoT']    # modified Tave
        mdfs['ST_RhoT_2'] = mdfs['ST_RhoT_1'] - mdfs['RhoT']    # modified Tc
        return mdfs
    
    @staticmethod
    def _get_wgts(itgs, mdfs):
        '''
        calculate dimensionless weight factors
        '''
        
        Rho_w = mdfs['Rho']
        TSc   = itgs['Tc'] * itgs['Sc']
        TSh   = itgs['Th'] * itgs['Sh']
        Tau_w = (TSh*itgs['deltaT'] - mdfs['SW']) / (TSh - TSc - itgs['S'])
        
        wgts = AttrDict()
        wgts['W_J'] = Rho_w / itgs['deltaT']
        wgts['W_T'] = Tau_w / itgs['deltaT']
        wgts['alpha_0'] = mdfs['ST_RhoT_0'] / itgs['Sh']
        wgts['alpha_1'] = mdfs['ST_RhoT_1'] / itgs['Sh']
        wgts['alpha_2'] = mdfs['ST_RhoT_2'] / itgs['Sh']
        return wgts
    
    @staticmethod
    def _get_prfs(itgs, mdfs, L, A):
        '''
        calculate profiles of device
        '''
        
        deltaT = itgs['deltaT']
        PFeng = 1E-6 * itgs['S']*itgs['S']/itgs['Rho']
        Zeng = PFeng / itgs['K']
        
        prfs = AttrDict()
        prfs['deltaT'] = deltaT
        prfs['PFeng']  = PFeng
        prfs['Zeng']   = Zeng
        prfs['ZTeng']  = Zeng * deltaT
        prfs['ZTp']    = Zeng * mdfs['ST_RhoT_1']
        prfs['m_opt']  = np.sqrt(1+prfs['ZTp'])
        prfs['Voc']    = 1E-3 * itgs['S']        # mV
        prfs['Isc']    = 1E-3 * deltaT*itgs['S']/(itgs['Rho']*L)*A   # A
        prfs['Qx']     = 1E-3 * PFeng*deltaT/L*A        # W
        prfs['itgs']   = itgs
        prfs['mdfs']   = mdfs
        prfs['wgts']   = None
        return prfs

class GenLeg(GenCore):
    '''
    Simulate thermoelectric leg of generator
    '''
    def __init__(self, **paras):
        '''
        paras : dict
            'isPoly': False,        # mode
            'TEdatas': None,        # TE datas
            'Tc': None,             # temperature at cold side
            'Th': None,             # temperature at hot side
            'L': 1,                 # length of TE leg in mm
            'A': 100,               # the cross-sectional area in mm^2, default 1 cm^2        
        '''
        
        logger.info('Begin initialization of {} ...'.format(self.__class__.__name__))
        
        # check TEdatas
        isPoly = paras.get('isPoly', self.paras['isPoly'])
        TEdatas = paras['TEdatas']
        if isPoly:
            Rho, S, K = TEdatas
            check = lambda x: isinstance(x, Polynomial)
            if not all(map(check, [Rho, S, K])):
                raise ValueError('TEdatas requires three numpy.polynomial.Polynomial.')
            else:
                logger.info('Read datas of TE properties in polynomial ...')
                logger.debug('Value of TEdatas:\n%s', str(TEdatas))
        else:
            T, C, S, K = TEdatas
            datas = np.vstack([T,C,S,K]).T
            logger.info('Read datas of TE properties ...')
            logger.debug('Value of TEdatas:\n%s', str(datas))
        
        # check Tc, Th
        for Ti in ('Tc', 'Th'):
            if Ti in paras:
                paras[Ti] = np.atleast_1d(paras[Ti])
                if len(paras[Ti]) == 1:
                    logger.info('{} is at {} K'.format(Ti, paras[Ti][0]))
                else:
                    logger.info('{} are at {}..{} K'.format(Ti, paras[Ti][0], paras[Ti][-1]))
            else:
                 raise ValueError('{} is required.'.format(Ti))  
        
        # update paras and check Length, Area
        self.paras.update(paras)
        logger.info('Length of TE leg: {} mm'.format(self.paras['L']))
        logger.info('Area of TE leg: {} mm^2'.format(self.paras['A']))
        logger.info('Finish initialization')
    
    def build(self, **options):
        '''
        options : dict
            'calWeights': False,    # whether to calculate dimensionless weight factors
            'returnProfiles': False,
        '''

        logger.info('Begin building process ...')
        profiles = self._build(options)
        logger.info('Finish building process')
        return profiles
    
    def simulate(self, **configs):
        '''
        configs : dict
            'I_r': 'YitaMax',       # 'YitaMax' | 'PowerMax' | 'scan'(or 'sweep') | array_like
            'numPoints': 101,
            'returnOutputs': False,
        '''
        
        logger.info('Begin simulating ...')
        outputs = self._simulate(configs)
        logger.info('Finish simulating process')
        return outputs

    @classmethod
    def valuate(cls, datas_TCSK, L=1):
        '''
        Convenient entry to evaluate thermoelectric leg of generator
        '''
        
        T = datas_TCSK[0]
        Tc, Th = T[0], T[1:]
        
        # initialling
        rst = AttrDict()
        logger.debug('Invoke valuate() method of %s', cls.__name__)
        gen = cls(TEdatas=datas_TCSK, Tc=Tc, Th=Th, L=L)
        
        # build device
        prfs = gen.build(returnProfiles=True)
        rst['deltaT'] = prfs.deltaT
        rst['PFeng']  = prfs.PFeng
        rst['ZTeng']  = prfs.ZTeng
        logger.info('Read out deltaT, PFeng, ZTeng')
        
        # to maximize Pout
        logger.info('To get maximal Pout')
        outs = gen.simulate(I_r='PowerMax', returnOutputs=True)
        rst['Pout'] = outs.Pout
        logger.info('Read out Pout')
        
        # to maximize Yita
        logger.info('To get maximal Yita')
        outs = gen.simulate(I_r='YitaMax', returnOutputs=True)
        rst['Yita'] = outs.Yita
        logger.info('Read out Yita')

        # results
        logger.debug('Exit valuate() method of %s', cls.__name__)
        return rst

class GenPair(GenCore):
    '''
    Simulate thermoelectric p-n pair of generator
    '''
    _paras = {
        'TEdatas_p': None,          # TE datas of p-type leg
        'TEdatas_n': None,          # TE datas of n-type leg
        'ratioLength': 1,           # Ln/Lp: float | array_like
        'ratioArea': 'ZTengMax',    # An/Ap: 'ZTengMax' | 'PFengMax' | array_like
    }
    def __init__(self, **paras):
        '''
        paras : dict
            'isPoly': False,        # mode
            'TEdatas': None,        # TE datas (just a place-holder)
            'TEdatas_p': None,      # TE datas of p-type leg
            'TEdatas_n': None,      # TE datas of n-type leg
            'Tc': None,             # temperature at cold side
            'Th': None,             # temperature at hot side
            'L': 1,                 # length of p-type TE leg in mm
            'A': 100,               # total cross-sectional area in mm^2, default 1 cm^2
            'ratioLength': 1,           # Ln/Lp: float | array_like
            'ratioArea': 'ZTengMax',    # An/Ap: 'ZTengMax' | 'PFengMax' | array_like
        '''
        self.paras.update(self._paras)      # update default parameters
        logger.info('Begin initialization of {} ...'.format(self.__class__.__name__))
        
        # check TEdatas_p, TEdatas_n
        isPoly = paras.get('isPoly', self.paras['isPoly'])
        TEdatas_np = dict(TEdatas_n=paras['TEdatas_n'],
                          TEdatas_p=paras['TEdatas_p'])
        
        if isPoly:
            for leg, datas_RhoSK in TEdatas_np.items():
                Rho, S, K = datas_RhoSK
                check = lambda x: isinstance(x, Polynomial)
                if not all(map(check, [Rho, S, K])):
                    dsp = '{} requires three numpy.polynomial.Polynomial.'
                    raise ValueError(dsp.format(leg))
                else:
                    dsp = 'Read TE properties datas of {}-type leg in polynomial ...'
                    logger.info(dsp.format(leg[-1]))
                    logger.debug('Value of %s:\n%s', leg, str(datas_RhoSK))
        else:
            for leg, datas_TCSK in TEdatas_np.items():
                T, C, S, K = datas_TCSK
                datas = np.vstack([T,C,S,K]).T
                dsp = 'Read TE properties datas of {}-type leg ...'
                logger.info(dsp.format(leg[-1]))
                logger.debug('Value of %s:\n%s', leg, str(datas))
        
        # check Tc, Th
        for Ti in ('Tc', 'Th'):
            if Ti in paras:
                paras[Ti] = np.atleast_1d(paras[Ti])
                if len(paras[Ti]) == 1:
                    logger.info('{} is at {} K'.format(Ti, paras[Ti][0]))
                else:
                    logger.info('{} are at {}..{} K'.format(Ti, paras[Ti][0], paras[Ti][-1]))
            else:
                 raise ValueError('{} is required.'.format(Ti))  
        
        # update paras and check Length, Area
        self.paras.update(paras)
        dsps = {'L': 'Length of p-type TE leg: {} mm',
                'A': 'Total cross-sectional area of TE p-n pair: {} mm^2',
                'ratioLength': 'Ratio of length (Ln/Lp): {}',
                'ratioArea': 'Ratio of Area (An/Ap): {}',}
        for key, dsp in dsps.items():
            logger.info(dsp.format(self.paras[key]))
        logger.info('Finish initialization')
        
    def build(self, **options):
        '''
        options : dict
            'calWeights': False,    # whether to calculate dimensionless weight factors
            'returnProfiles': False,
        '''
        
        logger.info('Begin building process ...')
        
        logger.info('Determine the configuration of p-n pair ...')
        rL = self.paras['ratioLength']
        rA = self.paras['ratioArea']
        datas_p = self.paras['TEdatas_p']
        datas_n = self.paras['TEdatas_n']
        if isinstance(rA, str):
            Tc = self.paras['Tc']
            Th = self.paras['Th']
            isPoly = self.paras['isPoly']
            itgs_p = super()._get_itgs(datas_p, Tc, Th, isPoly)
            itgs_n = super()._get_itgs(datas_n, Tc, Th, isPoly)
            if rA.lower().startswith('z'):
                rA = rL * np.sqrt(itgs_p['Rho']/itgs_n['Rho'] * itgs_n['K']/itgs_p['K'])
                logger.info('Obtain the configuration to maximize ZTeng')
                I_r = 'YitaMax'
                self.configs['I_r'] = I_r
                logger.debug("update default value of config['I_r'] to %s", I_r)
            elif rA.lower().startswith('p'):
                rA = np.sqrt(rL * itgs_p['Rho']/itgs_n['Rho'])
                logger.info('Obtain the configuration to maximize PFeng')
                I_r = 'PowerMax'
                self.configs['I_r'] = 'PowerMax'
                logger.debug("update default value of config['I_r'] to %s", I_r)
            else:
                raise ValueError('Invalid value of ratioArea')
        else:
            rA = np.array(rA)
            logger.info('Obtain the configuration under assigned size')
        self.paras['TEdatas'] = (itgs_p, itgs_n, rL, rA)
        logger.debug("Set paras['TEdatas'] as (itgs_p, itgs_n, rL, rA) form")

        profiles = self._build(options)
        logger.info('Finish building process')
        return profiles

    def simulate(self, **configs):
        '''
        configs : dict
            'I_r': 'YitaMax',       # 'YitaMax' | 'PowerMax' | 'scan'(or 'sweep') | array_like
            'numPoints': 101,
            'returnOutputs': False,
        '''
        
        logger.info('Begin simulating ...')
        outputs = self._simulate(configs)
        logger.info('Finish simulating process')
        return outputs
    
    @staticmethod
    def _get_itgs(datas, Tc, Th, isPoly):
        itgs_p, itgs_n, rL, rA = datas
        
        fx_t = lambda p, n: (p+n)/2
        fx_s = lambda p, n: p - n
        fx_r = lambda p, n: p*(1+rA) + n*rL/rA*(1+rA)
        fx_k = lambda p, n: p/(1+rA) + n/(1+rA)*rA/rL
        
        combs = [[fx_t, ('Tc', 'Th', 'deltaT')],
                 [fx_s, ('Sc', 'Sh', 'S', 'ST')],
                 [fx_r, ('Rho', 'RhoT')],
                 [fx_k, ('K',)]]
        
        itgs = AttrDict()
        for fx, props in combs:
            for prop in props:
                itgs[prop] = fx(itgs_p[prop], itgs_n[prop])
        return itgs
    
    @classmethod
    def valuate(cls, datas_p_TCSK, datas_n_TCSK, L=1):
        '''
        Convenient entry to evaluate thermoelectric p-n pair of generator
        '''
        
        T = datas_p_TCSK[0]
        Tc, Th = T[0], T[1:]
        
        # initialling
        rst = AttrDict()
        logger.debug('Invoke valuate() method of %s', cls.__name__)
        gen = cls(TEdatas_p=datas_p_TCSK, 
                  TEdatas_n=datas_n_TCSK,
                  Tc=Tc, Th=Th, L=L)
        
        # to maximize Yita
        logger.info('>>> To get maximal Yita <<<')
        gen.paras['ratioArea'] = 'ZTengMax'
        prfs = gen.build(returnProfiles=True)
        rst['deltaT'] = prfs.deltaT
        rst['ZTeng']  = prfs.ZTeng
        logger.info('Read out deltaT and ZTeng')
        
        outs = gen.simulate(I_r='YitaMax', returnOutputs=True)
        rst['Yita'] = outs.Yita
        logger.info('Read out Yita')
        
        # to maximize Pout
        logger.info('>>> To get maximal Pout <<<')
        gen.paras['ratioArea'] = 'PFengMax'
        prfs = gen.build(returnProfiles=True)
        rst['PFeng']  = prfs.PFeng
        logger.info('Read out deltaT, PFeng, ZTeng')
        
        outs = gen.simulate(I_r='PowerMax', returnOutputs=True)
        rst['Pout'] = outs.Pout
        logger.info('Read out Pout')

        # results
        logger.debug('Exit valuate() method of %s', cls.__name__)
        return rst
    