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
from collections.abc import Callable

from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz
import numpy as np

from .analysis import vquad
from .utils import AttrDict


class Variable:
    '''
    A class to represent a variable that notifies subscribers upon
    value changes, used for synchronizing variables within a model.
    '''
    def __init__(self, tag, initial=1, lower=0, upper=1000, scale=1, *,
                 constraint=None, depends=()):
        '''
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
        constraint : callable, optional
            A callable object used to compute the value of the current
            variable from other dependent variables.
        depends : list or tuple of Variables, optional
            Variables required by the `constraint`.

        Raises
        ------
        ValueError
            If the initial value is not between the lower and upper bounds,
            or if `constraint` is provided but is not a callable,
            or if any element in `depends` is not an instance of Variable.
        '''
        if lower <= initial <= upper:
            self._value = initial
            self.lower = lower
            self.upper = upper
        else:
            raise ValueError('Initial value must be between the lower and upper bounds.')
        self.tag = tag
        self.scale = scale
        self._subscribers = []
        if (constraint is not None) and (not isinstance(constraint, Callable)):
            raise ValueError("'constraint' must be a Callable object if provide")
        self._compute = constraint
        if not all(isinstance(d, type(self)) for d in depends):
            raise ValueError("'depends' must be a set of Variable")
        self._depends = tuple(depends)

    @property
    def constrained(self):
        '''
        Constraint status (read-only).
        '''
        return bool(self._compute)

    @property
    def value(self):
        '''
        The current value of the variable. Manually setting a new value will
        automatically trigger the :meth:`notify` method.

        Raises
        ------
        ValueError
            If the new value is not between the lower and upper bounds.
        '''
        if self.constrained:
            return self._compute(*(v.value for v in self._depends))
        else:
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
        self._subscribers.append((subscriber, name))

    def notify(self):
        '''
        Notifies all registered subscribers about the updated value.
        '''
        for sub, name in self._subscribers:
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
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'tag'):
            raise NotImplementedError("Subclasses must define a class attribute 'tag'")

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
        self.__ders = []

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
        for der in self.__ders:
            der.notify()
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
        if not all(isinstance(v, Variable) for v in variables):
            raise ValueError(f"Only {__name__}.Variable objects are supported")
        vars_ = [var for var in variables if not var.constrained]
        ders_ = [var for var in variables if var.constrained]
        if len(vars_) == 0:
            return self(dataT)
        self.__vars = vars_
        self.__ders = ders_
        if ('p0' in kwargs) or ('bounds' in kwargs):
            raise ValueError("'p0' or 'bounds' can only be from Variable")
        p0 = [var.value for var in vars_]
        lower = [var.lower for var in vars_]
        upper = [var.upper for var in vars_]
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
    tag = 'DEBYE'
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
        scattering = OrderedDict()
        additional = OrderedDict()
        for component in components:
            if isinstance(component, BaseScattering):
                if component.tag in scattering:
                    raise ValueError('Duplicated scattering mechanism is not allowed')
                scattering[component.tag] = component
            elif isinstance(component, BaseKappaModel):
                if component.tag in additional:
                    raise ValueError('Duplicated submodel is not allowed')
                additional[component.tag] = component
            else:
                # Unknown type of component
                # more type of component will be developed
                objs = ', '.join([
                    f'{__name__}.BaseScattering',
                    f'{__name__}.BaseKappaModel',
                ])
                raise NotImplementedError(f'Supported objects: {objs}')
        self._scattering = scattering
        self._additional = additional
    
    @property
    def wd(self):
        '''
        Debye cut-off (angular) frequency :math:`\\omega_D`
        (given by :math:`\\hbar \\omega_D = k_B \\Theta`)
        in rad/ps.
        '''
        return self.paras['td'] / self._hbar_kB   # td / (hbar/kB * 1E12)
    
    def scattering(self, w, T, with_total=True):
        '''
        Scattering rate, in THz.
        '''
        ftot = AttrDict((tag, s(w, T)) for tag, s in self._scattering.items())
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
        out = [[tag, scat(w, T)] for tag, scat in self._scattering.items()]
        if accumulate:
            for i in range(1, len(out)):
                out[i][0] = out[i-1][0] + '+' + out[i][0]
                out[i][1] = out[i-1][1] + out[i][1]
        factor = np.where(w <= self.wd, self._spectral_factor(w, T), 0)
        for i in range(len(out)):
            out[i][1] = factor/out[i][1]
        return AttrDict(out)

    def cumulative(self, w, T, accumulate=True, axis=-1):
        '''
        Cumulative thermal conductivity :math:`\\kappa_c(\\omega)` in W/(m.K):

        .. math ::

            \\kappa_c(\\omega) = \\int_0^{\\omega} \\kappa_s(\\omega) d\\omega

        '''
        spec = self.spectral(w, T, accumulate=accumulate)
        for key, val in spec.items():
            spec[key] = cumtrapz(val, w, initial=0, axis=axis)
        return spec

    def cumulative_mfp_t(self, mfp, T:float, accumulate=True, nbatch:int=1000):
        '''
        Calculates the cumulative thermal conductivity (in W/(m.K)) across phonon
        mean-free-path (:math:`\\lambda`, in nm) at a specified temperature point:

        .. math::

            \\kappa_c(\\lambda; T) = \\int_{\\lambda(\\omega; T) \\le \\lambda}
            \\kappa_s(\\omega; T) d\\omega
        '''
        # only scalar Temperature (T) input is supported currently
        T = np.atleast_1d(T)
        if T.size != 1:
            raise ValueError("Only scalar 'T' is supported now.")
        mfp = np.asarray(mfp)
        dw = self.wd / nbatch
        if dw < 10*self._EPS:
            raise ValueError("'nbatch' is too large to compute mfp weight.")
        w_left = np.arange(nbatch) * dw
        w_right = w_left + dw
        w_centre = (w_left + w_right) / 2
        w_left[0] = self._EPS
        spec = self.spectral(w_centre, T, accumulate=accumulate)
        weight = dw * np.asarray(list(spec.values()))   # (Nscat, nbatch)
        ftot_left = np.array([scat(w_left, T) for scat in self._scattering.values()])
        ftot_right = np.array([scat(w_right, T) for scat in self._scattering.values()])
        if accumulate:
            ftot_left = np.cumsum(ftot_left, axis=0)
            ftot_right = np.cumsum(ftot_right, axis=0)
        mfp_left = self.paras['vs'] / ftot_left     # [km/s]/[1/ps] = nm
        mfp_right = self.paras['vs'] / ftot_right   # [km/s]/[1/ps] = nm
        mfp_ave = (mfp_left+mfp_right)/2            # (Nscat, nbatch)
        mfp_wd = np.maximum(np.absolute(mfp_left-mfp_right), 10*self._EPS)
        # spec_mfp = AttrDict()
        cum_mfp = AttrDict()
        mfp_ = mfp[..., None]
        for key, ave, wd, w in zip(spec.keys(), mfp_ave, mfp_wd, weight):
            # wd *= 10    # enable smooth
            # spec = np.where((mfp_>ave-wd/2) & (mfp_<ave+wd/2), w/wd, 0)
            # spec_mfp[key] = np.sum(spec, axis=-1)
            cum = w/wd * np.minimum(np.maximum(mfp_-ave-wd/2, 0), wd)
            cum_mfp[key] = np.sum(cum, axis=-1)
        return cum_mfp

    def _spectral_factor(self, w, T):
        # kappa_s = factor * tau = factor / ftot
        kB = 0.01380649  # sacled by: w^2 / vs, i.e. 1E24/1E3 kB = 1E21 kB
        x = w/T * self._hbar_kB                        # hbar*(w*1E12)/(kB*T)
        wt = np.power(x, 2) * np.exp(x) / np.power(np.exp(x)-1, 2) * kB
        return np.power(w/np.pi, 2)/(2*self.paras['vs']) * wt

    def _spectral_sum(self, w, T, n:int=0):
        # for direct sum calculation, see :meth:`spectral` for each scattering
        if n:
            scattering = list(self._scattering.values())[:int(n)]
        else:
            scattering = self._scattering.values()
        ftot = sum(scat(w, T) for scat in scattering)   # in THz
        return self._spectral_factor(w, T) / ftot

    def __call__(self, T, nscat:int=None):
        if nscat is None:
            return self(T, 0) + sum(other(T) for other in self._additional.values())
        return vquad(self._spectral_sum, self._EPS, self.wd, args=(T, nscat))[0]


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
        
        \\tau_i^{-1} = A \\frac{k_B V_a^{1/3} \\gamma^2}{M_0 M_a v_s^3} \\omega ^2 T
                       \\exp \\left( -\\frac{[\\Theta]}{3T} \\right)
                     = A^{\\prime} \\omega ^2 T
                       \\exp \\left( -\\frac{[\\Theta]}{3T} \\right)

    (Hint: To diable the exponential term, simply set Theta (:math:`[\\Theta]`)
    to 0, even though it typically signifies the Debye temperature.)
    '''
    tag = 'PH'
    def __init__(self, gm=None, vs=None, Va=None, Ma=None, A=1, Theta=0, *, coef=None):
        if coef is None:
            super().__init__(gm=gm, vs=vs, Va=Va, Ma=Ma, A=A, Theta=Theta)
            self._direct_coef = False
        else:
            super().__init__(coef=coef, Theta=Theta)
            self._direct_coef = True
        for k, v in self.paras.items():
            if v is None:
                raise ValueError(f"Parameter '{k}' is required for {self.tag}")
    
    def __call__(self, w, T):
        w, T = np.broadcast_arrays(w, T)
        exponent = np.exp(-self.paras['Theta']/(3*T))
        return 1E+12*self.coef*np.power(w, 2)*T * exponent

    @property
    def coef(self):
        '''Value of :math:`A^{\\prime}`'''
        if self._direct_coef:
            return self.paras['coef']
        else:
            # coef = A*(kB*Va**(1/3)*gm**2) / (M0*Ma*vs**3)
            #          -23-30/3             - (-27  +3 *3 )     --> -15
            return 1.380649 / 1.67492749804 * 1E-15 * self.paras['A'] *\
                np.power(self.paras['Va'], 1/3) * np.power(self.paras['gm'], 2) /\
                (self.paras['Ma'] * np.power(self.paras['vs'], 3))


class PointDefect(BaseScattering):
    '''
    .. math::
        
        \\tau_i^{-1} = \\frac{V_a \\Gamma}{4 \\pi v_s^3} \\omega ^4
                     = B^{\\prime} \\omega ^4
        
    '''
    tag = 'PD'
    def __init__(self, vs=None, Va=None, G=None, *, coef=None):
        if coef is None:
            super().__init__(vs=vs, Va=Va, G=G)
            self._direct_coef = False
        else:
            super().__init__(coef=coef)
            self._direct_coef = True
        for k, v in self.paras.items():
            if v is None:
                raise ValueError(f"Parameter '{k}' is required for {self.tag}")
    
    def __call__(self, w, T):
        w, _ = np.broadcast_arrays(w, T)
        return 1E+36*self.coef*np.power(w, 4)

    @property
    def coef(self):
        '''Value of :math:`B^{\\prime}`'''
        if self._direct_coef:
            return self.paras['coef']
        else:
            # coef = Va*G/(4*pi*vs**3)
            #       -30  -(     3 *3 )       --> -36
            return 1E-36 * self.paras['Va'] * self.paras['G'] /\
                (4*np.pi*np.power(self.paras['vs'], 3))


class GrainBoundary(BaseScattering):
    '''
    .. math::
        
        \\tau_i^{-1} = \\alpha \\frac{v_s}{L}
        
    '''
    tag = 'GB'
    def __init__(self, vs, L, alpha=1):
        # vs: in km/s
        # L: um
        super().__init__(vs=vs, L=L, alpha=alpha)
    
    def __call__(self, w, T):
        L, *_ = np.broadcast_arrays(self.paras['L'], w, T)
        return 1E-3 * self.paras['alpha'] * self.paras['vs'] / L


class CahillScattering(BaseScattering):
    '''
    .. math::

        \\tau_i^{-1} = \\alpha \\frac{\\omega}{\\pi}
    '''
    tag = 'CAHILL'
    def __init__(self, alpha=1):
        super().__init__(alpha=alpha)

    def __call__(self, w, T):
        w, _ = np.broadcast_arrays(w, T)
        return self.paras['alpha'] * w/np.pi


class KappaBipolar(BaseKappaModel):
    '''
    .. math::
        \\kappa_{bip} = \\kappa_{bp} \\left ( \\frac{T}{T_{amb}} \\right ) ^p
            \\exp \\left( -\\frac{E_g}{2k_BT} \\right)
        \\text{, where } T_{amb} = 300 K

    Ref: O. C. Yelgel et al., Phys. Rev. B, 85 125207, 2012.
    '''
    tag = 'BIPOLAR'
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
    tag = 'POWERLAW'
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


class KappaKlemens(BaseKappaModel):
    '''
    .. math::

        \\frac{\\kappa(x)}{\\kappa_{pure}} =
        \\frac{\\tan^{-1}(u)}{u} \\text{, where }
        u^2 = \\cfrac{\\pi \\Theta V_a}{2\\hbar v_s^2}
        \\kappa_{pure}\\Gamma_0 \\cdot x (1-x)
    '''
    tag = 'KLEMENS'
    def __init__(self, Kpure, vs, td, G0, Va):
        '''
        Parameters
        ----------
        Kpure : float
            Kappa of the crystal with disorder (:math:`\\kappa_{pure}`),
            in W/(m.K).
        vs : float
            Sound velocity (:math:`v_s`), in km/s.
        td : float
            Debye temperature (:math:`\\Theta`), in Kelvin.
        G0 : float
            The factor of disorder scaling parameter (:math:`\\Gamma_0`).
        Va : float
            Average volume per atom (:math:`V_a`), in cubic angstroms (A^3).
        '''
        super().__init__(Kpure=Kpure, vs=vs, td=td, G0=G0, Va=Va)

    def u2(self, X):
        '''
        Calculates :math:`u^2` from :math:`x` based on model parameters.
        Users can customize the expression as needed by overriding this method.
        '''
        X = np.asarray(X)
        # hbar = 1.054571817e-34
        # hbar = hbar * (1E3 * 1E3) /1E-30
        hbar = 105.4571817
        factor = (np.pi * self.paras['td'] * self.paras['Va']) \
            /(2*hbar*np.power(self.paras['vs'],2)) \
            * self.paras['Kpure'] * self.paras['G0']
        return factor * X * (1-X)

    def __call__(self, X):
        u = np.sqrt(self.u2(X))
        ratio = np.where(np.abs(u) < 1E-6, 1, np.arctan(u)/u)
        return self.paras['Kpure'] * ratio

    def _cal(self, X, *args):
        '''
        Overrides :class:`BaseKappaModel`: replaces temperature variable with
        composition variable (X).
        '''
        return super()._cal(X, *args)

    def fit(self, dataX, dataK, *, variables=(), **kwargs):
        '''
        Overrides :class:`BaseKappaModel`: Fits dataX rather than dataT.
        '''
        return super().fit(dataX, dataK, variables=variables, **kwargs)
