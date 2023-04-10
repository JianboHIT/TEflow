import numpy as np

from .core import BaseBand
from .utils import kB, m_e, hbar, e0 


class APSSKB(BaseBand):
    m_d = 1
    sigma0 = 1
    Eg = 1
    def __init__(self, m_d=1, sigma0=1, Eg=1):
        self.m_d = m_d
        self.sigma0 = sigma0
        if Eg > 0:
            self.Eg = Eg
        else:
            raise ValueError('Eg should be greater than 0')
        
    def dos(self, E):
        factor = 1E-25      # state/(eV.m^3) --> 1E19 state/(eV.cm^3)
        g0 = np.power(2*self.m_d*m_e*e0, 3/2)/(2*np.pi*np.pi* np.power(hbar, 3))
        kane = np.sqrt(1+E/self.Eg) * (1+2*E/self.Eg)
        return factor * g0 * np.sqrt(E) * kane
    
    def trs(self, E, T):
        kane = 3*(1+E/self.Eg)/(np.power(1+2*E/self.Eg, 2)+2)
        return self.sigma0 * E/(kB*T) * kane
    
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
        return cls(m_d=m_d, sigma0=sigma0, Eg=Eg)
    
    @classmethod
    def from_UWT(cls, m_d=1, UWT=1, Eg=1):
        # m_d: m_e
        # UWT: cm^2/(V.s)
        # Eg: eV
        # factor = np.sqrt(np.pi)/2 * e0 \
        #          * np.power(2*m_e*kB*e0*300, 3/2) \
        #          / (2*np.pi*np.pi*np.power(hbar, 3)) * 1E-4 /100  # S/cm
        factor = 4.020521639724753      # S/cm
        sigma0 = factor * UWT
        return cls(m_d=m_d, sigma0=sigma0, Eg=Eg)