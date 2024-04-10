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
    def __init__(self, **paras):
        # self.profiles = AttrDict()    # initialize an empty profiles property
        pass

    @abstractmethod
    def build(self, **options):
        # self.profiles = AttrDict(...) # get profiles property
        # return self.profiles
        pass

    @abstractmethod
    def simulate(self, **configs):
        # profiles = self.profiles      # read out profiles property
        # outputs = AttrDict(...)       # get output
        # return output
        pass

    @classmethod
    @abstractmethod
    def valuate(cls, **kwargs):
        pass

class GenCore(BaseDevice):
    '''
    An abstract class for computing engineering performance of thermoelectric generator
    '''
    def __init__(self, TEdatas, Tc, Th, L=1, A=100, isPoly=False):
        self.TEdatas=TEdatas     # TE datas
        self.Tc=Tc               # temperature at cold side
        self.Th=Th               # temperature at hot side
        self.L=L                 # length of TE leg in mm
        self.A=A                 # the cross-sectional area in mm^2, default 1 cm^2
        self.isPoly=isPoly       # mode
        self.profiles = AttrDict()

    def _build(self, calWeights=False):
        # options = {
        #     'calWeights': False,    # whether to calculate dimensionless weight factors
        # }
        itgs = self._get_itgs()
        mdfs = self._get_mdfs(itgs)
        logger.info('Calculate integrals and modification factors')

        profiles = self._get_prfs(itgs, mdfs)
        logger.info('Calculate profiles of device')

        if calWeights:
            profiles['wgts'] = self._get_wgts(itgs, mdfs)
            logger.info('Calculate weight factors')
        else:
            logger.debug('Ingore calculation of weight factors')

        self.profiles = profiles
        logger.debug('Update profiles:\n%s', pformat(list(profiles.keys())))
        return profiles
    
    def _simulate(self, I_r='YitaMax', numPoints=101):
        # configs = {
        #     'I_r': 'YitaMax',       # 'YitaMax' | 'PowerMax' | 'scan'(or 'sweep') | array_like
        #     'numPoints': 101,
        # }

        prfs = self.profiles
        if not prfs:
            raise RuntimeError('Failed to read out profiles of device! '
                               'Try to build it at first ...')

        # parse I_r
        if isinstance(I_r, str):
            if I_r.lower().startswith('y'):
                m_opt = prfs['m_opt']
                I_r = 1/(1+m_opt)
                logger.info('Work under optimized efficiency')
            elif I_r.lower().startswith('p'):
                I_r = np.array([1/2,])
                logger.info('Work under optimized output power')
            elif I_r.lower().startswith('s'):
                shp = [-1,]+[1,]*prfs['deltaT']
                I_r = np.linspace(0, 1, numPoints).reshape(shp)
                logger.info('Work under auto-limition of current')
            else:
                raise ValueError('Invalid value of I_r')
        else:
            I_r = np.atleast_1d(I_r)
            logger.info('Work under assigned reduced current')
        logger.debug('Shape of I_r: %s', str(I_r.shape))

        # get details of profiles of device
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
        return outputs
    
    def _get_itgs(self):
        '''
        calculate integrals of porperties
        '''
        Tc = np.array(self.Tc)
        Th = np.array(self.Th)

        itgs = AttrDict()
        itgs['Tc']     = Tc
        itgs['Th']     = Th
        itgs['deltaT'] = Th - Tc
        
        if self.isPoly:
            Rho, S, K = self.TEdatas
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
            T, C, S, K = self.TEdatas
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

    def _get_mdfs(self, itgs=None):
        '''
        calculate modification factors
        '''
        if itgs is None:
            itgs = self._get_itgs()

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
    
    def _get_wgts(self, itgs=None, mdfs=None):
        '''
        calculate dimensionless weight factors
        '''
        if itgs is None:
            itgs = self._get_itgs()
        
        if mdfs is None:
            mdfs = self._get_mdfs(itgs)

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

    def _get_prfs(self, itgs=None, mdfs=None):
        '''
        calculate profiles of device
        '''
        if itgs is None:
            itgs = self._get_itgs()
        
        if mdfs is None:
            mdfs = self._get_mdfs(itgs)

        L, A = self.L, self.A

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
    def __init__(self, TEdatas, Tc, Th, L=1, A=100, isPoly=False):
        logger.info('Begin initialization of {} ...'.format(self.__class__.__name__))

        # check TEdatas
        if isPoly:
            Rho, S, K = TEdatas
            if not all(isinstance(x, Polynomial) for x in [Rho, S, K]):
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
        Tc = np.atleast_1d(Tc)
        if len(Tc) == 1:
            logger.info('Tc is at {} K'.format(Tc[0]))
        else:
            logger.info('Tc are at {}..{} K'.format(Tc[0], Tc[-1]))
        
        Th = np.atleast_1d(Th)
        if len(Th) == 1:
            logger.info('Th is at {} K'.format(Th[0]))
        else:
            logger.info('Th are at {}..{} K'.format(Th[0], Th[-1]))
        
        # check Length, Area
        logger.info('Length of TE leg: {} mm'.format(L))
        logger.info('Area of TE leg: {} mm^2'.format(A))

        # update paras
        super().__init__(TEdatas, Tc, Th, L, A, isPoly)
        logger.info('Finish initialization')
    
    def build(self, calWeights=False):
        options = dict(
            calWeights=calWeights,  # whether to calculate dimensionless weight factors
        )
        logger.info('Begin building process ...')
        logger.debug('Options:\n%s', pformat(options))

        profiles = self._build(calWeights=calWeights)
        logger.info('Finish building process')
        return profiles
    
    def simulate(self, I_r='YitaMax', numPoints=101):
        configs = dict(
            I_r=I_r,                # 'YitaMax' | 'PowerMax' | 'scan'(or 'sweep') | array_like
            numPoints=numPoints,
        )
        logger.info('Begin simulating ...')
        logger.debug('Configs:\n%s', pformat(configs))

        outputs = self._simulate(I_r=I_r, numPoints=numPoints)
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
        rst = AttrDict(Tc=Tc*np.ones_like(Th), Th=Th)
        logger.debug('Invoke valuate() method of %s', cls.__name__)
        gen = cls(TEdatas=datas_TCSK, Tc=Tc, Th=Th, L=L)
        
        # build device
        prfs = gen.build()
        rst['deltaT'] = prfs.deltaT
        rst['PFeng']  = prfs.PFeng
        rst['ZTeng']  = prfs.ZTeng
        logger.info('Read out deltaT, PFeng, ZTeng')
        
        # to maximize Pout
        logger.info('To get maximal Pout')
        outs = gen.simulate(I_r='PowerMax')
        rst['Pout'] = outs.Pout
        logger.info('Read out Pout')
        
        # to maximize Yita
        logger.info('To get maximal Yita')
        outs = gen.simulate(I_r='YitaMax')
        rst['Yita'] = outs.Yita
        logger.info('Read out Yita')

        # results
        logger.debug('Exit valuate() method of %s', cls.__name__)
        return rst

class GenPair(GenCore):
    '''
    Simulate thermoelectric p-n pair of generator
    '''
    def __init__(self, TEdatas_p, TEdatas_n, Tc, Th, L=1, A=100, isPoly=False):
        logger.info('Begin initialization of {} ...'.format(self.__class__.__name__))

        # check TEdatas_p, TEdatas_n
        TEdatas_pn = dict(TEdatas_p=TEdatas_p, TEdatas_n=TEdatas_n)
        if isPoly:
            for leg, datas_RhoSK in TEdatas_pn.items():
                Rho, S, K = datas_RhoSK
                if not all(isinstance(x, Polynomial) for x in [Rho, S, K]):
                    dsp = '{} requires three numpy.polynomial.Polynomial.'
                    raise ValueError(dsp.format(leg))
                else:
                    dsp = 'Read TE properties datas of {}-type leg in polynomial ...'
                    logger.info(dsp.format(leg[-1]))
                    logger.debug('Value of %s:\n%s', leg, str(datas_RhoSK))
        else:
            for leg, datas_TCSK in TEdatas_pn.items():
                T, C, S, K = datas_TCSK
                datas = np.vstack([T,C,S,K]).T
                dsp = 'Read TE properties datas of {}-type leg ...'
                logger.info(dsp.format(leg[-1]))
                logger.debug('Value of %s:\n%s', leg, str(datas))
        
        # check Tc, Th
        Tc = np.atleast_1d(Tc)
        if len(Tc) == 1:
            logger.info('Tc is at {} K'.format(Tc[0]))
        else:
            logger.info('Tc are at {}..{} K'.format(Tc[0], Tc[-1]))
        
        Th = np.atleast_1d(Th)
        if len(Th) == 1:
            logger.info('Th is at {} K'.format(Th[0]))
        else:
            logger.info('Th are at {}..{} K'.format(Th[0], Th[-1]))
        
        # check Length, Area
        logger.info('Length of TE leg: {} mm'.format(L))
        logger.info('Area of TE leg: {} mm^2'.format(A))

        # update paras: TEdatas, _rL, _rA are initialed to None
        super().__init__(None, Tc, Th, L, A, isPoly)
        self.TEdatas_p = TEdatas_p
        self.TEdatas_n = TEdatas_n
        self._rL = None
        self._rA = None
        logger.info('Finish initialization')

    def build(self, ratioLength=1, ratioArea='ZTengMax', calWeights=False):
        options = dict(
            ratioLength=ratioLength,# Ln/Lp: float | array_like
            ratioArea=ratioArea,    # An/Ap: 'ZTengMax' | 'PFengMax' | array_like
            calWeights=calWeights,  # whether to calculate dimensionless weight factors
        )
        logger.info('Begin building process ...')
        logger.debug('Options:\n%s', pformat(options))

        self._rL = ratioLength
        self._rA = ratioArea
        profiles = self._build(options)
        profiles['rL'] = self._rL   # get the real value
        profiles['rA'] = self._rA   # get the real value
        logger.info('Finish building process')
        return profiles
    
    def simulate(self, I_r='YitaMax', numPoints=101):
        configs = dict(
            I_r=I_r,                # 'YitaMax' | 'PowerMax' | 'scan'(or 'sweep') | array_like
            numPoints=numPoints,
        )
        logger.info('Begin simulating ...')
        logger.debug('Configs:\n%s', pformat(configs))

        outputs = self._simulate(I_r=I_r, numPoints=numPoints)
        logger.info('Finish simulating process')
        return outputs
    
    def _get_itgs(self):
        # calculate itgs of each leg
        self.TEdatas = self.TEdatas_p
        itgs_p = super()._get_itgs()

        self.TEdatas = self.TEdatas_n
        itgs_n = super()._get_itgs()

        self.TEdatas = None         # release to None
        logger.debug('Calculate itgs property of each leg')

        # parse rL & rA
        rL = np.array(self._rL)     # Ln/Lp: float | array_like
        rA = self._rA               # An/Ap: 'ZTengMax' | 'PFengMax' | array_like
        if isinstance(rA, str):
            if rA.lower().startswith('z'):
                rA = rL * np.sqrt(itgs_p['Rho']/itgs_n['Rho'] * itgs_n['K']/itgs_p['K'])
                logger.info('Obtain the configuration to maximize ZTeng')
            elif rA.lower().startswith('p'):
                rA = np.sqrt(rL * itgs_p['Rho']/itgs_n['Rho'])
                logger.info('Obtain the configuration to maximize PFeng')
            else:
                raise ValueError('Invalid value of ratioArea')
        else:
            rA = np.array(rA)
            logger.info('Obtain the configuration under assigned size')
        
        # update properties: _rL and _rA
        self._rL = rL
        self._rA = rA
        logger.debug('Parse configuration of the device:\n    rL: %s\n    rA: %s',
                     pformat(rL), pformat(rA))

        # merge itgs_p and itgs_n to the equivalent itgs
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
        rst = AttrDict(Tc=Tc*np.ones_like(Th), Th=Th)
        logger.debug('Invoke valuate() method of %s', cls.__name__)
        gen = cls(TEdatas_p=datas_p_TCSK, 
                  TEdatas_n=datas_n_TCSK,
                  Tc=Tc, Th=Th, L=L)
        
        # to maximize Yita
        logger.info('>>> To get maximal Yita <<<')
        prfs = gen.build(ratioArea='ZTengMax')
        rst['deltaT'] = prfs.deltaT
        rst['ZTeng']  = prfs.ZTeng
        logger.info('Read out deltaT and ZTeng')
        
        outs = gen.simulate(I_r='YitaMax')
        rst['Yita'] = outs.Yita
        logger.info('Read out Yita')
        
        # to maximize Pout
        logger.info('>>> To get maximal Pout <<<')
        prfs = gen.build(ratioArea='PFengMax')
        rst['PFeng']  = prfs.PFeng
        logger.info('Read out deltaT, PFeng, ZTeng')
        
        outs = gen.simulate(I_r='PowerMax')
        rst['Pout'] = outs.Pout
        logger.info('Read out Pout')

        # results
        logger.debug('Exit valuate() method of %s', cls.__name__)
        return rst
    