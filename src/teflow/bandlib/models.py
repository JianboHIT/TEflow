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
from .misc import kB_eV, m_e, hbar, q


class APSSPB(BaseBand):
    '''
    A class for describing single parabolic band (SPB) model when the
    acoustic phonon scattering (APS) mechanism predominates. In this
    model, there are three key parameters determine thermoelectric
    properties: (a) the effective mass of the density of states
    :math:`m_d^{\\ask}`, (b) the intrinsic electrical conductivity
    :math:`\\simga_0`, and (c) the ratio of longitudinal to transverse
    effective masses :math:`K^{\\ask}`. These parameters correspond
    to class attributes m_d, sigma0, and Kmass, respectively. The core
    of constructing the class is obtaining the values of these parameters.

    Attributes
    ----------
    m_d : float
        Effective mass of the density of states in :math:`m_e`, primarily
        influencing carrier concentration calculations. It should be a
        positive float.
    sigma0 : float
        Intrinsic electrical conductivity in `S/cm`, the core parameter
        influencing thermoelectric transport properties. It should be a
        positive float.
    Kmass : float
        The ratio of longitudinal to transverse effective mass, affecting
        calculations related to Hall coefficients. It should be a positive
        float.
    '''

    m_d = 1         # m_e
    sigma0 = 1      # S/cm
    Kmass = 1       # m1/m2
    def __init__(self, m_d=1, sigma0=1, Kmass=1):
        '''
        Initialize an instance by the specified parameters.

        Parameters
        ----------
        m_d : float, optional
            Effective mass of the density of states in static electron mass,
            by default 1.
        sigma0 : float, optional
            Intrinsic electrical conductivity in S/cm, by default 1.
        Kmass : float, optional
            The ratio of longitudinal to transverse effective mass, by
            default 1.
        '''
        self.m_d = m_d
        self.sigma0 = sigma0
        self.Kmass = Kmass
        
    def dos(self, E):
        '''Density of states, in 1E19 state/(eV.cm^3).'''
        factor = 1E-25      # state/(eV.m^3) --> 1E19 state/(eV.cm^3)
        g0 = np.power(2*self.m_d*m_e*q, 3/2)/(2*np.pi*np.pi* np.power(hbar, 3))
        return factor * g0 * np.sqrt(E)
    
    def trs(self, E, T):
        '''Transport distribution function, in S/cm.'''
        return self.sigma0 * E/(kB_eV*T)
    
    def hall(self, E, T):
        '''Hall transport distribution function, in S.cm/(V.s), i.e.
        [S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)].'''
        N0 = np.power(2*self.m_d*m_e*kB_eV*q*T, 3/2)/(2*np.pi*np.pi* np.power(hbar, 3))
        facotr = np.power(self.sigma0, 2) / (2/3 * 1E-6*N0 * q)  # N0: m^-3 --> cm^-3
        return self.Kstar * facotr * np.sqrt(E/(kB_eV*T))
    
    @property
    def Kstar(self):
        '''Anisotropy factor of effective mass for Hall effect.'''
        K = self.Kmass
        return 3*K*(K+2)/np.power(2*K+1, 2)
    
    @property
    def UWT(self):
        '''Temperature-independent weighted mobility, in cm^2/(V.s).'''
        factor = 4.020521639724753      # [S/cm] / [cm^2/(V.s)]
        return self.sigma0 / factor
    
    @classmethod
    def from_DP(cls, m1=1, m2=None, Nv=1, Cii=1, Ed=1):
        '''
        Construct the class based on the Deformation Potential
        (DP) theory.

        Parameters
        ----------
        m1 : float, optional
            The ratio of the longitudinal effective mass to the static
            electron mass, by default 1.
        m2 : float or None, optional
            The ratio of the transverse effective mass to the static
            electron mass. If None (default), it is set equal to m1.
        Nv : int, optional
            Valley degeneracy, by default 1.
        Cii : float, optional
            Elastic constant in GPa, by default 1.
        Ed : float, optional
            Deformation potential in eV, by default 1.
        '''

        m2 = (m2 or m1)
        m_c = 3/(1/m1+2/m2)
        m_d = np.cbrt(Nv*Nv * m1*m2*m2)
        # factor = (2*q*q*hbar*1E9) / (3*m_e*np.pi*q*q) /100    # S/cm
        factor = 245.66655370009886     # S/cm
        sigma0 = factor * (Nv*Cii)/(m_c*Ed*Ed)
        return cls(m_d=m_d, sigma0=sigma0, Kmass=m1/m2)
    
    @classmethod
    def from_UWT(cls, m_d=1, UWT=1, Kmass=1):
        '''
        Construct the class based on temperature-independent weighted
        mobility.

        Parameters
        ----------
        m_d : float, optional
            Effective mass of the density of states in static electron mass,
            by default 1.
        UWT : float, optional
            Temperature-independent weighted mobility in cm^2/(V.s), by
            default 1.
        Kmass : float, optional
            The ratio of longitudinal to transverse effective mass, by
            default 1.
        '''

        # factor = np.sqrt(np.pi)/2 * q \
        #          * np.power(2*m_e*kB_eV*q*300, 3/2) \
        #          / (2*np.pi*np.pi*np.power(hbar, 3)) * 1E-4 /100  # S/cm
        factor = 4.020521639724753      # S/cm
        sigma0 = factor * UWT
        return cls(m_d=m_d, sigma0=sigma0, Kmass=Kmass)

    @classmethod
    def valuate_m_d(cls, dataS, dataN, dataT, hall=False, Kmass=1):
        spb = cls(Kmass=Kmass)
        dataEF = spb.slove_EF('S', dataS, dataT)
        if hall:
            N0 = spb.NH(dataEF, dataT)
        else:
            N0 = spb.N(dataEF, dataT)
        return np.power(dataN/N0, 2/3)

    @classmethod
    def valuate_L(cls, dataS):
        '''
        A class method to evaluate Lorenz numbers from Seebeck coefficients.

        Parameters
        ----------
        dataS : ndarray
            Absolute Seebeck coefficients in uV/K.

        Returns
        -------
        ndarray
            Lorenz numbers in 1E-8 W.Ohm/K^2.
        '''

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
    '''
    A class for describing single Kane band (SKB) model when the acoustic
    phonon scattering (APS) mechanism predominates. In contrast to the
    classical single parabolic band (SPB) model (see :class:`APSSPB`), an
    additional parameter describing the energy band shape, namely the bandgap
    (Eg), is introduced.

    Attributes
    ----------
    m_d : float
        Effective mass of the density of states in :math:`m_e`, primarily
        influencing carrier concentration calculations. It should be a
        positive float.
    sigma0 : float
        Intrinsic electrical conductivity in `S/cm`, the core parameter
        influencing thermoelectric transport properties. It should be a
        positive float.
    Eg : float
        Parameter bandgap in eV, which significantly influences various
        transport properties.
    Kmass : float
        The ratio of longitudinal to transverse effective mass, affecting
        calculations related to Hall coefficients. It should be a positive
        float.
    '''
    m_d = 1         # m_e
    sigma0 = 1      # S/cm
    Eg = 1          # eV
    Kmass = 1       # m1/m2
    def __init__(self, m_d=1, sigma0=1, Eg=1, Kmass=1):
        '''
        Initialize an instance by the specified parameters.

        Parameters
        ----------
        m_d : float, optional
            Effective mass of the density of states in static electron mass,
            by default 1.
        sigma0 : float, optional
            Intrinsic electrical conductivity in S/cm, by default 1.
        Eg : float, optional
            Bandgap in eV, by default 1.
        Kmass : float, optional
            The ratio of longitudinal to transverse effective mass, by
            default 1.
        '''
        self.m_d = m_d
        self.sigma0 = sigma0
        if Eg > 0:
            self.Eg = Eg
        else:
            raise ValueError('Eg should be greater than 0')
        self.Kmass = Kmass
        
    def dos(self, E):
        '''Density of states, in 1E19 state/(eV.cm^3).'''
        factor = 1E-25      # state/(eV.m^3) --> 1E19 state/(eV.cm^3)
        g0 = np.power(2*self.m_d*m_e*q, 3/2)/(2*np.pi*np.pi* np.power(hbar, 3))
        kane = np.sqrt(1+E/self.Eg) * (1+2*E/self.Eg)
        return factor * g0 * np.sqrt(E) * kane
    
    def trs(self, E, T):
        '''Transport distribution function, in S/cm.'''
        kane = 3*(1+E/self.Eg)/(np.power(1+2*E/self.Eg, 2)+2)
        return self.sigma0 * E/(kB_eV*T) * kane
    
    def hall(self, E, T):
        '''Hall transport distribution function, in S.cm/(V.s), i.e.
        [S/cm]^2 * [cm^3/C] = [S/cm] * [cm^2/(V.s)].'''
        N0 = np.power(2*self.m_d*m_e*kB_eV*q*T, 3/2)/(2*np.pi*np.pi* np.power(hbar, 3))
        facotr = np.power(self.sigma0, 2) / (2/3 * 1E-6*N0 * q)  # N0: m^-3 --> cm^-3
        kane = 9*np.sqrt(1+E/self.Eg)/np.power(np.power(1+2*E/self.Eg, 2)+2, 2)
        return self.Kstar * facotr * np.sqrt(E/(kB_eV*T)) * kane
    
    @property
    def Kstar(self):
        '''Anisotropy factor of effective mass for Hall effect.'''
        K = self.Kmass
        return 3*K*(K+2)/np.power(2*K+1, 2)
    
    @property
    def UWT(self):
        '''Temperature-independent weighted mobility, in cm^2/(V.s).'''
        factor = 4.020521639724753      # [S/cm] / [cm^2/(V.s)]
        return self.sigma0 / factor
    
    @classmethod
    def from_DP(cls, m1=1, m2=None, Nv=1, Cii=1, Ed=1, Eg=1):
        '''
        Construct the class based on the Deformation Potential
        (DP) theory.

        Parameters
        ----------
        m1 : float, optional
            The ratio of the longitudinal effective mass to the static
            electron mass, by default 1.
        m2 : float or None, optional
            The ratio of the transverse effective mass to the static
            electron mass. If None (default), it is set equal to m1.
        Nv : int, optional
            Valley degeneracy, by default 1.
        Cii : float, optional
            Elastic constant in GPa, by default 1.
        Ed : float, optional
            Deformation potential in eV, by default 1.
        Eg : float, optional
            Bandgap in eV, by default 1.
        '''
        
        m2 = (m2 or m1)
        m_c = 3/(1/m1+2/m2)
        m_d = np.cbrt(Nv*Nv * m1*m2*m2)
        # factor = (2*q*q*hbar*1E9) / (3*m_e*np.pi*q*q) /100    # S/cm
        factor = 245.66655370009886     # S/cm
        sigma0 = factor * (Nv*Cii)/(m_c*Ed*Ed)
        return cls(m_d=m_d, sigma0=sigma0, Eg=Eg, Kmass=m1/m2)
    
    @classmethod
    def from_UWT(cls, m_d=1, UWT=1, Eg=1, Kmass=1):
        '''
        Construct the class based on temperature-independent weighted
        mobility.

        Parameters
        ----------
        m_d : float, optional
            Effective mass of the density of states in static electron mass,
            by default 1.
        UWT : float, optional
            Temperature-independent weighted mobility in cm^2/(V.s), by
            default 1.
        Eg : float, optional
            Bandgap in eV, by default 1.
        Kmass : float, optional
            The ratio of longitudinal to transverse effective mass, by
            default 1.
        '''

        # factor = np.sqrt(np.pi)/2 * q \
        #          * np.power(2*m_e*kB_eV*q*300, 3/2) \
        #          / (2*np.pi*np.pi*np.power(hbar, 3)) \
        #          * 1E-6 # C/m^3 --> C/cm^3
        factor = 4.020521639724753      # [S/cm] / [cm^2/(V.s)]
        sigma0 = factor * UWT
        return cls(m_d=m_d, sigma0=sigma0, Eg=Eg, Kmass=Kmass)
