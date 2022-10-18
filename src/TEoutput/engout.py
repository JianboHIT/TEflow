import logging
import numpy as np
from pprint import pformat
from abc import ABC, abstractmethod
from scipy.integrate import cumtrapz
from numpy.polynomial import Polynomial

from .utils import AttrDict

logger = logging.getLogger(__name__)

def get_itgs(datas, Tc, Th, isPoly=False):
    '''
    calculate integrals of porperties
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

def get_mdfs(itgs):
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

def get_wgts(itgs, mdfs):    
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

class BaseDevice(ABC):
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
    
class Generator(BaseDevice):
    paras = {
        'isPoly': False,            # mode
        'TEdatas': (None, None),    # TE datas
        'Tc': None,
        'Th': None,
        'L': 1,                 # length of TE leg
        'A': None,
        'Rc': None,
        'Kc': None,
        'isPair': True,
        'hasContact': True,
    }
    def __new__(cls, **paras):
        if 'isPair' in paras:
            isPair = paras['isPair']
        else:
            TEdatas = paras.get('TEdatas', None)
            if TEdatas is None:
                raise ValueError('TEdatas is required to create object.')
            elif len(TEdatas) == 2:
                isPair = True
            else:
                isPair = False
        
        if 'hasContact' in paras:
            hasContact = paras['hasContact']
        else:
            Rc = paras.get('Rc', None)
            Kc = paras.get('Kc', None)
            if (Rc is None) and (Kc is None):
                hasContact = False
            else:
                hasContact = True
        
        if isPair:
            if hasContact:
                # return super().__new__(cls)
                pass
            else:
                # return GenCoupleCore(**paras)
                pass
        else:
            if hasContact:
                # return GenElement(**paras)
                pass
            else:
                return GenElementCore(**paras)
            
        dsp = ('%s will become a convenience class, and '
               'the original one is called GenElementCore now.')
        raise NotImplementedError(dsp, cls.__name__)


class GenElementCore(BaseDevice):
    paras = {
        'isPoly': False,        # mode
        'TEdatas': None,        # TE datas
        'Tc': None,
        'Th': None,
        'L': 1,                 # length of TE leg in mm
        'A': 100,               # the cross-sectional area in mm^2, default 1 cm^2
    }
    options = {
        'calWeights': False,    # whether to calculate dimensionless weight factors
        'returnProfiles': False,
    }
    configs = {
        'I_r': 'optimal',       # 'optimal' | 'scan'(or 'sweep') | array_like
        'numPoints': 101,
        'returnOutputs': False,
    }
    
    def __init__(self, **paras):
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
                logger.info('Read datas of TE properties in polynomial')
                logger.debug('Value of TEdatas:\n%s', pformat(TEdatas))
        else:
            T, C, S, K = TEdatas
            datas = np.vstack([T,C,S,K]).T
            # TODO: Tc, Th_max check and cutrange
            logger.info('Read datas of TE properties')
            logger.debug('Value of TEdatas:\n%s', pformat(datas))
        
        # check Tc, Th
        for Ti in ('Tc', 'Th'):
            if Ti in paras:
                paras[Ti] = np.atleast_1d(paras[Ti])
                if len(paras[Ti]) == 1:
                    logger.info('{} is at {}'.format(Ti, paras[Ti][0]))
                else:
                    logger.info('{} are at \n{}'.format(Ti, paras[Ti]))
            else:
                 raise ValueError('{} is required.'.format(Ti))  
        
        # update paras and check Length, Area
        self.paras.update(paras)
        logger.info('Length of TE leg: {} mm'.format(self.paras['L']))
        logger.info('Area of TE leg: {} mm^2'.format(self.paras['A']))
        logger.info('Finish initialization')

    def build(self, **options):
        logger.info('Begin building process ...')
        self.options.update(options)
        options = self.options
        logger.info('Read options')
        logger.debug('Options:\n%s', pformat(options))
        
        itgs = get_itgs(datas=self.paras['TEdatas'],
                        Tc=self.paras['Tc'],
                        Th=self.paras['Th'],
                        isPoly=self.paras['isPoly'])

        mdfs = get_mdfs(itgs)
        logger.info('Calculate integrals and modification factors')
        
        L = self.paras['L']         # mm
        A = self.paras['A'] / 100   # mm^2 to cm^2
        deltaT = itgs['deltaT']
        PFeng = 1E-6 * itgs['S']*itgs['S']/itgs['Rho']
        Zeng = PFeng / itgs['K']
        ZTeng = Zeng * deltaT
        ZTp = Zeng * mdfs['ST_RhoT_1']
        m_opt = np.sqrt(1+ZTp)
        Voc = 1E-3 * itgs['S']        # mV
        Jsc = 1E-1 * deltaT*itgs['S']/(itgs['Rho']*L)   # A/cm^2
        qx = 0.1 * PFeng*deltaT/L      # W/cm^2, q_flux
        Isc = A*Jsc
        Qx = A*qx
        
        profiles = AttrDict()
        profiles['deltaT'] = deltaT
        profiles['PFeng']  = PFeng
        profiles['ZTeng']  = ZTeng
        profiles['Zeng']   = Zeng
        profiles['ZTp']    = ZTp
        profiles['m_opt']  = m_opt
        profiles['Voc']    = Voc
        profiles['Isc']    = Isc
        profiles['Qx']     = Qx
        profiles['itgs']   = itgs
        profiles['mdfs']   = mdfs
        logger.info('Calculate profiles of device')
        logger.debug('Keys of profiles:\n%s', pformat(profiles.keys()))
        
        if options['calWeights']:
            wgts = get_wgts(itgs, mdfs)
            profiles.update(wgts)
            logger.info('Calculate weight factors')
        else:
            logger.debug('Ingore calculation of weight factors')
        
        self.profiles = profiles
        if options['returnProfiles']:
            return profiles
        logger.info('Finish building process')
        
    def simulate(self, **configs):
        logger.info('Begin simulating ...')
        self.configs.update(configs)
        configs = self.configs
        
        # I_r='optimal', numPoints=101, returnOutputs=False
        I_r = configs['I_r']      # 'optimal' | 'scan'(or 'sweep') | array_like
        if isinstance(I_r, str):
            if I_r.lower().startswith('o'):
                I_r = None
                logger.info('Work under optimial current')
            elif I_r.lower().startswith('s'):
                numPoints = configs['numPoints']
                I_r = np.linspace(0, 1, numPoints)
                logger.info('Work under auto-limition of current')
        else:
            I_r = np.array(I_r)
            logger.info('Work under assigned reduced current')
        
        deltaT = self.profiles['deltaT']
        Qx = self.profiles['Qx']
        mdfs = self.profiles['mdfs']
        logger.info('Read out deltaT, Qx, and mdfs')
        
        outputs = AttrDict()
        if I_r is None:
            m_opt = self.profiles['m_opt']
            ST_RhoT_0 = mdfs['ST_RhoT_0']
            ST_RhoT_2 = mdfs['ST_RhoT_2']
            outputs['Pout'] = 1/4 * Qx     # W
            outputs['Yita'] = 100 * deltaT * (m_opt-1)/(ST_RhoT_0*m_opt+ST_RhoT_2)
            logger.info('Calculate Pout and Yita')
        else:
            if deltaT.size > 1:
                shp = [-1,] + [1,]*deltaT.ndim
                I_r = np.reshape(I_r, shp)
                logger.debug('Reshape I_r to {}'.format(pformat(shp)))
            Vout_r = 1-I_r
            Pout_r = I_r * Vout_r
            Qhot_rt = (1/self.profiles['Zeng'] + mdfs['ST'] * I_r - mdfs['RhoT']*I_r*I_r)
            outputs['I'] = self.profiles['Isc'] * I_r
            outputs['Vout'] = self.profiles['Voc'] * Vout_r
            outputs['Pout'] = Qx * Pout_r 
            outputs['Qhot'] = Qx * Qhot_rt/deltaT
            outputs['Yita'] = 100 * outputs['Pout'] / outputs['Qhot']
            logger.info('Calculate Jd, Vout, Pout, Qhot, and Yita')
        
        self.outputs = outputs
        if configs['returnOutputs']:
            return outputs
        logger.info('Finish simulating process')
    
    @classmethod
    def valuate(cls, datas_TCSK, L=1):
        T = datas_TCSK[0]
        Tc, Th = T[0], T[1:]
        gen = cls(TEdatas=datas_TCSK, Tc=Tc, Th=Th, L=L)
        gen.build()
        rst = gen.simulate(returnOutputs=True)
        return gen.deltaT, gen.PFeng, gen.ZTeng, rst.Pout, rst.Yita
        
class GenElement(BaseDevice):
    paras = {
        'isPoly': False,        # mode
        'TEdatas': None,        # TE datas
        'Tc': None,
        'Th': None,
        'L': 1,                 # length of TE leg in mm
        'A': 100,               # the cross-sectional area in mm^2, default 1 cm^2
        'Rc': None,
        'Kc': None,
    }
    def __init__(self, **paras):
        raise NotImplementedError('Under-developed')
