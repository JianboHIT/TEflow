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

import numpy as np

from .core import BaseBand
from .utils import kB_eV, m_e, hbar, e0


class APSSPB(BaseBand):
    m_d = 1         # m_e
    sigma0 = 1      # S/cm
    Kmass = 1       # m1/m2
    def __init__(self, m_d=1, sigma0=1, Kmass=1):
        self.m_d = m_d
        self.sigma0 = sigma0
        self.Kmass = Kmass
        
    def dos(self, E):
        factor = 1E-25      # state/(eV.m^3) --> 1E19 state/(eV.cm^3)
        g0 = np.power(2*self.m_d*m_e*e0, 3/2)/(2*np.pi*np.pi* np.power(hbar, 3))
        return factor * g0 * np.sqrt(E)
    
    def trs(self, E, T):
        return self.sigma0 * E/(kB_eV*T)
    
    def hall(self, E, T):
        N0 = np.power(2*self.m_d*m_e*kB_eV*e0*T, 3/2)/(2*np.pi*np.pi* np.power(hbar, 3))
        facotr = np.power(self.sigma0, 2) / (2/3 * 1E-6*N0 * e0)  # N0: m^-3 --> cm^-3
        return self.Kstar * facotr * np.sqrt(E/(kB_eV*T))
    
    @property
    def Kstar(self):
        K = self.Kmass
        return 3*K*(K+2)/np.power(2*K+1, 2)
    
    @property
    def UWT(self):
        factor = 4.020521639724753      # [S/cm] / [cm^2/(V.s)]
        return self.sigma0 / factor
    
    @classmethod
    def from_DP(cls, m1=1, m2=None, Nv=1, Cii=1, Ed=1):
        # m1, m2: m_e
        # Cii: GPa
        # Ed: eV
        
        m2 = (m2 or m1)
        m_c = 3/(1/m1+2/m2)
        m_d = np.cbrt(Nv*Nv * m1*m2*m2)
        # factor = (2*e0*e0*hbar*1E9) / (3*m_e*np.pi*e0*e0) /100    # S/cm
        factor = 245.66655370009886     # S/cm
        sigma0 = factor * (Nv*Cii)/(m_c*Ed*Ed)
        return cls(m_d=m_d, sigma0=sigma0, Kmass=m1/m2)
    
    @classmethod
    def from_UWT(cls, m_d=1, UWT=1, Kmass=1):
        # m_d: m_e
        # UWT: cm^2/(V.s)
        # factor = np.sqrt(np.pi)/2 * e0 \
        #          * np.power(2*m_e*kB_eV*e0*300, 3/2) \
        #          / (2*np.pi*np.pi*np.power(hbar, 3)) * 1E-4 /100  # S/cm
        factor = 4.020521639724753      # S/cm
        sigma0 = factor * UWT
        return cls(m_d=m_d, sigma0=sigma0, Kmass=Kmass)

    @classmethod
    def slove_m_d(cls, dataS, dataN, dataT, hall=False, Kmass=1):
        spb = cls(Kmass=Kmass)
        dataEF = spb.slove_EF('S', dataS, dataT)
        if hall:
            N0 = spb.NH(dataEF, dataT)
        else:
            N0 = spb.N(dataEF, dataT)
        return np.power(dataN/N0, 2/3)

    @classmethod
    def slove_L(cls, dataS):
        if np.any(dataS <= 0):
            raise ValueError('Non-negative values are required for dataS '
                             '(i.e. absolute Seebeck coefficient)')
        else:
            dataS = np.minimum(dataS, 1000)

        spb = cls(sigma0=100)
        TEMP = 1/kB_eV
        yita = spb.slove_EF('S', dataS, TEMP)
        L = spb.L(yita, TEMP)
        return spb.L(yita, TEMP)


class APSSKB(BaseBand):
    m_d = 1         # m_e
    sigma0 = 1      # S/cm
    Eg = 1          # eV
    Kmass = 1       # m1/m2
    def __init__(self, m_d=1, sigma0=1, Eg=1, Kmass=1):
        self.m_d = m_d
        self.sigma0 = sigma0
        if Eg > 0:
            self.Eg = Eg
        else:
            raise ValueError('Eg should be greater than 0')
        self.Kmass = Kmass
        
    def dos(self, E):
        factor = 1E-25      # state/(eV.m^3) --> 1E19 state/(eV.cm^3)
        g0 = np.power(2*self.m_d*m_e*e0, 3/2)/(2*np.pi*np.pi* np.power(hbar, 3))
        kane = np.sqrt(1+E/self.Eg) * (1+2*E/self.Eg)
        return factor * g0 * np.sqrt(E) * kane
    
    def trs(self, E, T):
        kane = 3*(1+E/self.Eg)/(np.power(1+2*E/self.Eg, 2)+2)
        return self.sigma0 * E/(kB_eV*T) * kane
    
    def hall(self, E, T):
        N0 = np.power(2*self.m_d*m_e*kB_eV*e0*T, 3/2)/(2*np.pi*np.pi* np.power(hbar, 3))
        facotr = np.power(self.sigma0, 2) / (2/3 * 1E-6*N0 * e0)  # N0: m^-3 --> cm^-3
        kane = 9*np.sqrt(1+E/self.Eg)/np.power(np.power(1+2*E/self.Eg, 2)+2, 2)
        return self.Kstar * facotr * np.sqrt(E/(kB_eV*T)) * kane
    
    @property
    def Kstar(self):
        K = self.Kmass
        return 3*K*(K+2)/np.power(2*K+1, 2)
    
    @property
    def UWT(self):
        factor = 4.020521639724753      # [S/cm] / [cm^2/(V.s)]
        return self.sigma0 / factor
    
    @classmethod
    def from_DP(cls, m1=1, m2=None, Nv=1, Cii=1, Ed=1, Eg=1):
        # m1, m2: m_e
        # Cii: GPa
        # Ed: eV
        # Eg: eV
        
        m2 = (m2 or m1)
        m_c = 3/(1/m1+2/m2)
        m_d = np.cbrt(Nv*Nv * m1*m2*m2)
        # factor = (2*e0*e0*hbar*1E9) / (3*m_e*np.pi*e0*e0) /100    # S/cm
        factor = 245.66655370009886     # S/cm
        sigma0 = factor * (Nv*Cii)/(m_c*Ed*Ed)
        return cls(m_d=m_d, sigma0=sigma0, Eg=Eg, Kmass=m1/m2)
    
    @classmethod
    def from_UWT(cls, m_d=1, UWT=1, Eg=1, Kmass=1):
        # m_d: m_e
        # UWT: cm^2/(V.s)
        # Eg: eV
        # factor = np.sqrt(np.pi)/2 * e0 \
        #          * np.power(2*m_e*kB_eV*e0*300, 3/2) \
        #          / (2*np.pi*np.pi*np.power(hbar, 3)) \
        #          * 1E-6 # C/m^3 --> C/cm^3
        factor = 4.020521639724753      # [S/cm] / [cm^2/(V.s)]
        sigma0 = factor * UWT
        return cls(m_d=m_d, sigma0=sigma0, Eg=Eg, Kmass=Kmass)
