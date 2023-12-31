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

from abc import ABC, abstractmethod
from collections import OrderedDict
from scipy.optimize import curve_fit
import numpy as np

from .analysis import vquad
from .utils import AttrDict


class Variable:
    '''
    A class to represent a variable that notifies subscribers upon
    value changes, used for synchronizing variables within a model.
    '''
    def __init__(self, tag, initial=1, lower=0, upper=1000, scale=1):
        '''
        Initialize a new Variable instance:

        Parameters
        ----------
        tag : str
            Sets the :attr:`tag` attribute.
        initial : float, optional
            The initial guess of :attr:`value` attribute, defaults to 1.
        lower : float, optional
            Sets the :attr:`lower` attribute, defaults to 0.
        upper : float, optional
            Sets the :attr:`upper` attribute, defaults to 1000.
        scale : float, optional
            Sets the :attr:`scale` attribute, defaults to 1.

        Raises
        ------
        ValueError
            If the initial value is not between the lower and upper bounds.
        '''
        if lower <= initial <= upper:
            self._value = initial
            self.lower = lower
            self.upper = upper
        else:
            raise ValueError('Initial value must be between the lower and upper bounds.')
        self.tag = tag
        self.scale = scale
        self.subscribers = []

    @property
    def value(self):
        '''
        The current value of the variable. Setting a new value will
        automatically trigger the :meth:`notify` method.

        Raises
        ------
        ValueError
            If the new value is not between the lower and upper bounds.
        '''
        return self._value

    @value.setter
    def value(self, new_value):
        if self.lower <= new_value <= self.upper:
            self._value = new_value
        else:
            raise ValueError('The value should be between '
                             f'{self.lower} and {self.upper}.')
        self.notify()

    def register(self, subscriber, name: str):
        '''
        Registers a subscriber that will be notified of value changes.

        Parameters
        ----------
        instance : object
            The instance to notify when the variable changes.
        name : str
            The parameter name in the instance that will be updated.
        '''
        self.subscribers.append((subscriber, name))

    def notify(self):
        '''
        Notifies all registered subscribers about the updated value.
        '''
        for sub, name in self.subscribers:
            sub[name] = self.value * self.scale


class Parameters(OrderedDict):
    '''
    A dictionary-like container to manage model parameters, 
    automatically synchronizing any contained Variable instances.
    '''
    def __init__(self, **parameters):
        for name, val in parameters.items():
            if isinstance(val, Variable):
                val.register(self, name)
                parameters[name] = val.value * val.scale
        super().__init__(**parameters)


class BaseKappaModel(ABC):
    '''
    Abstract base class for kappa models.

    This class serves as a template for kappa models, providing a framework
    for implementing model-specific calculations and fitting procedures.
    '''
    def __init__(self, **parameters):
        '''
        Parameters
        ----------
        **parameters : dict
            A dictionary of parameters to initialize the model. These parameters
            are managed by a `Parameters` instance, allowing for synchronization 
            with `Variable` objects if used.
        '''
        self.paras = Parameters(**parameters)
        self.__vars = []

    @abstractmethod
    def __call__(self, T):
        '''
        The core method to be implemented in subclasses. It should calculate 
        and return the kappa value for the given temperature T.
        '''
        pass
    
    def _cal(self, T, *args):
        '''
        Calculates the kappa value for a given temperature T and specific
        variable values. It necessitates that the quantity of values
        provided in `args` aligns with the number of variables in the model,
        and a mismatch triggers a ValueError. This method is designed
        primarily for internal usage within the :meth:`fit` method. 
        Generally, in standard workflows, direct manipulation of variable
        values through their `value` attribute is more conventional.
        For routine kappa value calculations, invoking the model instance
        directly as `kappa = model(T)`. Therefore, the explicit invocation
        of this method with variable values is seldom required in regular
        use cases.
        '''
        if len(args) != len(self.__vars):
            raise ValueError('Dismatch mumber between args and variables')
        for arg, var in zip(args, self.__vars):
            var.value = arg
        return self(T)
    
    def fit(self, dataT, dataK, *, variables=(), **kwargs):
        '''
        Fits the model to the provided data using specified variables.
        In the absence of specified variables, this method directly calculates
        and returns the kappa values for the given temperatures based on
        the current parameters. This method primarily utilizes the
        `scipy.optimize.curve_fit` function, with `kwargs` providing
        additional options to tailor the fitting process. However,
        note that 'p0' and 'bounds' parameters are superseded by the
        settings in the Variable instances, and should be adjusted 
        through their respective attributes, not via `kwargs`.
        '''
        if len(variables) == 0:
            return self(dataT)
        if not all(isinstance(v, Variable) for v in variables):
            raise ValueError(f"Only {__name__}.Variable objects are supported")
        self.__vars = variables
        if ('p0' in kwargs) or ('bounds' in kwargs):
            raise ValueError("'p0' or 'bounds' can only be from Variable")
        p0 = [var.value for var in variables]
        lower = [var.lower for var in variables]
        upper = [var.upper for var in variables]
        return curve_fit(self._cal, dataT, dataK,
                         p0=p0, bounds=(lower, upper), **kwargs)


class KappaDebye(BaseKappaModel):
    '''
    .. math::
        
        \\kappa = \\frac{1}{3} \\int_0^{\\omega_D}
                  \\tau v_s^2 g(\\omega)
                  \\hbar \\omega \\frac{\\partial f}{\\partial T}
                  d\\omega
    
    '''
    _EPS = 1E-6
    _hbar_kB = 7.638232577577646    # hbar/kB * 1E12
    def __init__(self, vs, td, components):
        '''
        Parameters
        ----------
        vs : float
            Sound velocity (:math:`v_s`), in km/s.
        td : float
            Debye temperature (:math:`\\Theta`), in Kelvin.
        components : list
            Components of the model, such as scattering mechanism.
        '''
        super().__init__(vs=vs, td=td)
        scattering = []
        for component in components:
            if isinstance(component, BaseScattering):
                scattering.append(component)
            else:
                # Unknown type of component
                # more type of component will be developed
                obj = f'{__name__}.BaseScattering objects'
                raise NotImplementedError(f'Only {obj} are supported now')
        self._scattering = scattering
    
    @property
    def wcut(self):
        '''
        Cut-off (angular) frequency, i.e.
        :math:`\\hbar \\omega_{cut} = k_B \\Theta`,
        in rad/ps.
        '''
        return self.paras['td'] / self._hbar_kB   # td / (hbar/kB * 1E12)
    
    def scattering(self, w, T, with_total=True):
        '''
        Scattering rate, in THz.
        '''
        ftot = AttrDict((s.tag, s(w, T)) for s in self._scattering)
        if with_total:
            ftot['total'] = sum(ftot.values())
        return ftot
    
    def spectral(self, w, T, accumulate=True):
        '''
        Spectral thermal conductivity :math:`\\kappa_s(\\omega)` in ps.W/(m.K):
        
        .. math::
            
            \\kappa_s(\\omega) = \\tau
                \\frac{\\omega ^2}{2 v_s \\pi ^2}
                k_B \\left ( \\frac{\\hbar \\omega}{k_B T}\\right )^2
                \\cfrac{\\exp \\left( \\cfrac{\\hbar \\omega}{k_B T} \\right )}{
                    \\left [
                        \\exp \\left( \\cfrac{\\hbar \\omega}{k_B T} - 1\\right)
                    \\right ]^2
                }
            \\text{, then }
            \\kappa = \\int_0^{\\omega_D} \\kappa_s(\\omega) d\\omega

        Parameters
        ----------
        w : ndarray
            Phonon angular frequency in rad/ps.
        T : ndarray
            Temperatures in K.
        '''
        w, T = np.broadcast_arrays(np.maximum(self._EPS, w), T)
        out = [[scat.tag, scat(w, T)] for scat in self._scattering]
        if accumulate:
            for i in range(1, len(out)):
                out[i][0] = out[i-1][0] + '+' + out[i][0]
                out[i][1] = out[i-1][1] + out[i][1]
        factor = np.where(w <= self.wcut, self._spectral_factor(w, T), 0)
        for i in range(len(out)):
            out[i][1] = factor/out[i][1]
        return AttrDict(out)

    def _spectral_factor(self, w, T):
        # kappa_s = factor * tau = factor / ftot
        kB = 0.01380649  # sacled by: w^2 / vs, i.e. 1E24/1E3 kB = 1E21 kB
        x = w/T * self._hbar_kB                        # hbar*(w*1E12)/(kB*T)
        wt = np.power(x, 2) * np.exp(x) / np.power(np.exp(x)-1, 2) * kB
        return np.power(w/np.pi, 2)/(2*self.paras['vs']) * wt

    def _spectral_sum(self, w, T, n:int=None):
        # for direct sum calculation, see :meth:`spectral` for each scattering
        scattering = self._scattering[:n] if n else self._scattering
        ftot = sum(scat(w, T) for scat in scattering)   # in THz
        return self._spectral_factor(w, T) / ftot

    def __call__(self, T, n:int=None):
        return vquad(self._spectral_sum, self._EPS, self.wcut, args=(T, n))[0]


class BaseScattering(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'tag'):
            raise NotImplementedError("Subclasses must define a class attribute 'tag'")

    def __init__(self, **parameters):
        # uncommon parameters in SI.
        self.paras = Parameters(**parameters)
    
    @abstractmethod
    def __call__(self, w, T):
        # w in rad/ps, T in K
        # return: tau^(-1), in THz
        pass


class ThreePhonon(BaseScattering):
    '''
    .. math::
        
        \\tau_i^{-1} = A \\omega ^2 T
        
    '''
    tag = 'PH'
    def __init__(self, A):
        super().__init__(A=A)
    
    def __call__(self, w, T):
        w, T = np.broadcast_arrays(w, T)
        return 1E+12*self.paras['A']*np.power(w, 2)*T


class PointDefect(BaseScattering):
    '''
    .. math::
        
        \\tau_i^{-1} = B \\omega ^4
        
    '''
    tag = 'PD'
    def __init__(self, B):
        super().__init__(B=B)
    
    def __call__(self, w, T):
        w, _ = np.broadcast_arrays(w, T)
        return 1E+36*self.paras['B']*np.power(w, 4)


class GrainBoundary(BaseScattering):
    '''
    .. math::
        
        \\tau_i^{-1} = \\frac{v_s}{L}
        
    '''
    tag = 'GB'
    def __init__(self, vs, L):
        # vs: in km/s
        # L: um
        super().__init__(vs=vs, L=L)
    
    def __call__(self, w, T):
        broadcasted = np.broadcast(w, T)
        return 1E-3 * self.paras['vs'] / self.paras['L'] * np.ones(broadcasted.shape)


class KappaBipolar(BaseKappaModel):
    '''
    .. math::
        \\kappa_{bip} = \\kappa_{bp} \\left ( \\frac{T}{T_{amb}} \\right ) ^p
            \\exp \\left( -\\frac{E_g}{2k_BT} \\right)
        \\text{, where } T_{amb} = 300 K

    Ref: O. C. Yelgel et al., Phys. Rev. B, 85 125207, 2012.
    '''
    def __init__(self, Kbp, Eg, p=1):
        '''
        Parameters
        ----------
        Kbp : float
            Adjustable parameter (:math:`\\kappa_{bp}`), in W/(m.K).
        Eg : float
            Band gap, in eV.
        p : float, optional
            Adjustable parameter, by default 1.
        '''
        super().__init__(Kbp=Kbp, Eg=Eg, p=p)
    
    def __call__(self, T):
        kB_eV = 8.617333262145179e-05   # kB under eV/K, i.e.,  kB/e0
        Tref = np.divide(T, 300)
        exponent = np.exp(-self.paras['Eg']/(2*kB_eV*T))
        return self.paras['Kbp'] * np.power(Tref, self.paras['p']) * exponent


class KappaPowerLaw(BaseKappaModel):
    '''
    .. math:: 
    
        \\kappa = \\kappa_{amb} \\left ( \\frac{T_{amb}}{T} \\right ) ^n
                  + \\kappa_0 \\text{, where } T_{amb} = 300 K
        
    '''
    def __init__(self, Kamb, n=1, K0=0):
        '''
        Parameters
        ----------
        Kamb : float
            Ambient thermal conductivity (:math:`\\kappa_{amb}`),
            corresponds to the thermal conductivity at the ambient temperature
            (300 K) with the background offset excluded.
        n : float, optional
            The exponent in the expression, by default 1.
        K0 : float, optional
            Background offset (:math:`\\kappa_0`), by default 0.
        '''
        super().__init__(Kamb=Kamb, n=n, K0=K0)
    
    def __call__(self, T):
        pow = np.power(np.divide(300, T), self.paras['n'])
        return self.paras['Kamb'] * pow + self.paras['K0']
