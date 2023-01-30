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
from pprint import pformat
import numpy as np
from scipy.optimize import minimize_scalar

from .utils import AttrDict

logger = logging.getLogger(__name__)

def cal_Phi(u, S, T):
    '''
    calculate thermoelectric potential Phi

    Parameters
    ----------
    u : float
        the relative current density in [1/V]
    S : float
        Seebeck coefficient
    T : float
        temperature

    Returns
    -------
    Phi : float
        thermoelectric potential (Phi) in [V]
    '''
    Phi = 1E-6 * S*T + 1/u
    return Phi

def cal_Yita(u,datas,allTemp=False):
    '''
    calculate Yita by TE datas and initial u

    Parameters
    ----------
    u : float
        the relative current density in [1/V]
    datas : list | ndarray
        TE datas like [T, C, S, K]
    allTemp : bool, optional
        calculate Yita at all temperatures (allTemp=True), or only at the hot temperature (allTemp=False, default)

    Returns
    -------
    Yita : float | ndarray
        efficiency (Yita) in [%]
    '''
    T, C, S, K = datas
    C = 1E2 * C
    S = 1E-6 * S

    Phi_0 = S[0]*T[0] + 1/u
    Yitas = [0,]
    for i in range(1,len(T)):
        p_sqrt = 1 - u**2 * (K[i]/C[i] + K[i-1]/C[i-1]) * (T[i] - T[i-1])
        p_last = (T[i] + T[i-1])/2 * (S[i] - S[i-1])
        u_inv = np.sqrt(p_sqrt)/u - p_last
        u = 1/u_inv
        Phi_i = S[i]*T[i] + 1/u
        Yitas.append( 1-Phi_0/Phi_i )
    Yitas = 100 * np.array(Yitas)
    
    if allTemp:
        return Yitas
    else:
        return Yitas[-1]

def cal_ZTdev(Yita, Tc, Th):
    '''
    calculate ZTdev by a given Yita and corresponding temperatures at cold and hot sides

    Parameters
    ----------
    Yita : float | ndarray
        efficiency (Yita) in [%]
    Tc : float
        temperature at cold side
    Th : float
        temperature at hot side

    Returns
    -------
    ZTdev : float | ndarray
        device ZT
    '''
    # ZTdev = np.power((Th-Tc*(1-Yita/100))/(Th*(1-Yita/100)-Tc), 2) - 1
    sub_1 = Th-Tc*(1-Yita/100)
    sub_2 = Th*(1-Yita/100)-Tc
    sub = np.divide(sub_1, sub_2, out=np.ones_like(sub_1), where=(np.abs(sub_2) > 1E-3))
    ZTdev = np.power(sub, 2) - 1
    return ZTdev

def cal_opt_u(datas, details=False, returnYita=False):
    '''
    calculate optimal u to max Yita

    Parameters
    ----------
    datas : list | ndarray
        TE datas like [T, C, S, K]
    details : bool, optional
        return the actual OptimizeResult object that contains detailed optimization results (details=True), 
        or only the solution of the optimization and/or values of objective function (details=False, default)
    returnYita : bool, optional
        whether to return the corresponding single-point Yita at optimal u, by default False.
        Note: this will only work if details=False.

    Returns
    -------
    rst : float | tuple | OptimizeResult
        result depends on input parameters
    '''
    _, C, S, K = datas

    if S[0] > 0:
        u_min = 1E-2
        u_max = 1E-4 * S[0] * C[0] / K[0]
    else:
        u_min = 1E-4 * S[0] * C[0] / K[0]
        u_max = -1E-2
    nega_Yita = lambda u, datas: (-1) * cal_Yita(u, datas)
    rst = minimize_scalar(fun=nega_Yita, 
                          bounds=(u_min, u_max), 
                          args=(datas,), 
                          method='bounded')
    
    logger.info('Optimize u to maximize Yita')
    logger.debug('Returned OptimizeResult: \n%s', pformat(rst))
    if details:
        return rst
    else:
        if rst.success:
            if returnYita:
                x = rst.x
                y = rst.fun
                return x, (-1) * y
            else:
                return rst.x
        else:
            return None

def cal_opt_Yita(datas, allTemp=True):
    '''
    calculate maximum at the optimal u

    Parameters
    ----------
    datas : list | ndarray
        TE datas like [T, C, S, K]
    allTemp : bool, optional
        calculate maximum Yita at all temperatures (allTemp=True, default), 
        or only at the hot temperature (allTemp=False)

    Returns
    -------
    Yita_opt : float | ndarray
        maximum Yita
    '''
    u_opt = cal_opt_u(datas)
    Yita_opt = cal_Yita(u_opt, datas, allTemp=allTemp)
    logger.info('Calculate corresponding Yita at the optimized u')
    return Yita_opt

def valuate(datas_TCSK, allTemp=True):
    '''
    calculate ZTdev by TE datas

    Parameters
    ----------
    datas_TCSK : list | ndarray
        TE datas like [T, C, S, K]
    allTemp : bool, optional
        pass to cal_opt_Yita()

    Returns
    -------
    rst : AttrDict
        deltaT : float | ndarray
            temperature difference in [K]
        ZTdev : float | ndarray
            device ZT
        Yita : float | ndarray
            optimal Yita in [%]
    '''
    
    dsp = 'all' if allTemp else 'max.'
    logger.info('Calculate ZTdev by TE datas at %s temperature', dsp)
    
    rst = AttrDict()
    T = datas_TCSK[0]
    rst['deltaT'] = T - T[0]
    rst['Yita'] = cal_opt_Yita(datas_TCSK, allTemp=allTemp)
    rst['ZTdev'] = cal_ZTdev(rst['Yita'], Tc=T[0], Th=T)
    
    logger.info('Finish calculation of ZTdev and relative')
    return rst

