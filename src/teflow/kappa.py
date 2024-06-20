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

import re
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable

from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz
import numpy as np

from .mathext import vquad
from .utils import AttrDict, ExecWrapper, CfgParser

logger = logging.getLogger(__name__)


class Variable:
    '''
    A singleton class based on the 'tag' attribute to represent
    a variable that notifies subscribers upon value changes,
    used for synchronizing variables across models.
    '''
    __instances = {}
    def __new__(cls, tag, *args, **kwargs):
        if tag not in cls.__instances:
            instance = super().__new__(cls)
            cls.__instances[tag] = instance
        return cls.__instances[tag]

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
        if hasattr(self, '_value'):
            return
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
        self._depends = Parameters(**{f'__{tag}_{i}':v for i, v in enumerate(depends)})

    def __str__(self):
        dsp = '{0.__class__.__name__}: {0.value} ? {0.scale} '\
              '<{0.lower},{0.upper}> {1} {0.tag}'
        return dsp.format(self, '@&' if self.constrained else '@')

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
            # return self._compute(*(v.value for v in self._depends))
            return self._compute(*self._depends.values())
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
            The name of parameter that will be updated.
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
    A dictionary-like container for managing model parameters, 
    automatically synchronizing with any contained :class:`Variable` instances. 
    Note that, the communication with :class:`Variable` instances is designed
    to be passive to enhance performance of code. As such, changes to
    :class:`Variable` will reflect in all registered Parameters instances,
    but direct modifications to Parameters only affect itself.
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

    Attributes
    ----------
    paras : Parameters
        All parameters of the model.
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
        if not scattering:
            raise ValueError('No scattering mechanism is provided')
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

    @classmethod
    def vs_to_td(cls, vs, Va):
        '''
        .. math ::

            \\Theta = \\frac{\\hbar\\omega_D}{k_B} = \\frac{\\hbar}{k_B}
                \\left( \\frac{6\\pi^2}{V_a} \\right) ^ {1/3} v_s

        Parameters
        ----------
        vs : float
            Average sound velocity (:math:`v_s`), in km/s.
        Va : float
            Average volume per atom (:math:`V_a`), in cubic angstroms (A^3).

        Returns
        -------
        float
            Debye temperature (:math:`\\Theta`), in Kelvin.
        '''
        # wd = np.power(6*np.pi*np.pi/(Va*1E-30), 1/3) * vs*1E3 = ... * 1E13
        return cls._hbar_kB * 10 * np.power(6*np.pi*np.pi/Va, 1/3) * vs

    @staticmethod
    def vs_to_gm(v1, v2):
        '''
        .. math::

            \\gamma = \\frac{3}{2} \\frac{1+\\mu}{2-3\\mu}
            \\text{, where }
            \\frac{v_1}{v_2} = \\sqrt{\\frac{2-2\\mu}{1-2\\mu}}

        Ref: D. S. Sanditov et al., Tech. Phys., 56 1619, 2011.

        Parameters
        ----------
        v1 : float
            Longitudinal sound velocity in km/s.
        v2 : float
            Transverse sound velocity in km/s.

        Returns
        -------
        float
            Gruneisen parameter (:math:`\\gamma`).
        '''
        r = np.power(np.divide(v1, v2), 2)  # (v1/v2)^2
        # mu = (r - 2)/(2*r - 2)
        # gm = 3/2 * (1+mu)/(2-3*mu)
        return 3/2 * (3*r-4)/(r+2)

    @staticmethod
    def vs_mean(v1, v2, p:int=-3):
        '''
        .. math::

            v_{s,p} = \\left( \\frac{v_1^p+2v_2^p}{3} \\right)^{1/p}

        Parameters
        ----------
        v1 : float
            Longitudinal sound velocity in km/s.
        v2 : float
            Transverse sound velocity in km/s.
        p : int, optional
            The exponent, by default -3.

        Returns
        -------
        float
            Generalized mean of sound velocities.
        '''
        v1, v2 = np.broadcast_arrays(v1, v2)
        vel = np.array([v1, v2, v2], dtype='float64')
        if p == 0:
            return np.power(np.prod(vel), 1/3)
        else:
            return np.power(np.mean(np.power(vel, p)), 1/p)


class BaseScattering(ABC):
    '''
    Abstract base class for scattering rate (in THz, i.e. 1/ps) model
    :math:`\\tau_i^{-1}(\\omega, T)`.

    Attributes
    ----------
    paras : Parameters
        All parameters of the model.
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

    Hint: To diable the exponential term, simply set Theta (:math:`[\\Theta]`)
    to 0, even though it typically signifies the Debye temperature.
    '''
    tag = 'PH'
    def __init__(self, gm=None, vs=None, Va=None, Ma=None, A=1, Theta=0, *, coef=None):
        '''
        Parameters
        ----------
        gm : float
            Gruneisen parameter (:math:`\\gamma`), dimensionless.
        vs : float
            Average sound velocity (:math:`v_s`), in km/s.
        Va : float
            Average volume per atom (:math:`V_a`), in cubic angstroms (A^3).
        Ma : float
            Average atomic mass per atom (:math:`M_a`), in atomic mass units (amu).
        A : float, optional
            Dimensionless adjustable parameter, by default 1.
        Theta : float, optional
            Debye temperature (:math:`[\\Theta]`) in Kelvin.
            If set to 0 (default), the exponential term will be disabled.
        coef : float, optional
            A comprehensive adjustable parameter (:math:`A^{\\prime}`).
            When set, all parameters except `Theta` become non-effective.
            Typically, its magnitude is around 10^(-18). By default, it is None.
        '''
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
        '''
        Parameters
        ----------
        vs : float
            Average sound velocity (:math:`v_s`), in km/s.
        Va : float
            Average volume per atom (:math:`V_a`), in cubic angstroms (A^3).
        G : float
            Dimensionless disorder scattering parameter (:math:`\\Gamma`)
        coef : float, optional
            A comprehensive adjustable parameter (:math:`B^{\\prime}`).
            When set, all parameters (`vs`, `Va`, and `G`)  become non-effective.
            Typically, its magnitude is around 10^(-41). By default, it is None.
        '''
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
            #       -30  -(     3 *3 )       --> -39
            return 1E-39 * self.paras['Va'] * self.paras['G'] /\
                (4*np.pi*np.power(self.paras['vs'], 3))


class GrainBoundary(BaseScattering):
    '''
    .. math::
        
        \\tau_i^{-1} = \\alpha \\frac{v_s}{L}
        
    '''
    tag = 'GB'
    def __init__(self, vs, L, alpha=1):
        '''
        Parameters
        ----------
        vs : float
            Average sound velocity (:math:`v_s`), in km/s.
        L : float
            Average grain size (:math:`L`), in um.
        alpha : float, optional
            Dimensionless adjustable parameter (:math:`\\alpha`), by default 1.
        '''
        super().__init__(vs=vs, L=L, alpha=alpha)
    
    def __call__(self, w, T):
        L, *_ = np.broadcast_arrays(self.paras['L'], w, T)
        return 1E-3 * self.paras['alpha'] * self.paras['vs'] / L


class Nanoparticles(BaseScattering):
    '''
    .. math::

        \\tau_i^{-1} = v_s \\left( \\sigma_{geometrical}^{-1}
            + \\sigma_{Rayleigh}^{-1} \\right)^{-1}N_1

    .. math ::

        \\sigma_{geometrical} = 2 \\pi R^2

    .. math ::

        \\sigma_{Rayleigh} = \\frac{16}{9} \\pi R^2 \\left[
                \\alpha^2 \\left(\\frac{D_1-D_0}{4D_0}\\right)^2
                + 3 \\alpha^8 \\left(\\frac{Y_1-Y_0}{Y_0}\\right)^2
            \\right] \\left(\\frac{\\omega R}{v_s}\\right)^4

    .. math ::

        N_1 = \\frac{\\phi}{\\overline{V_1}} = \\cfrac{\\phi}{\\cfrac{4}{3}\\pi R^3}

    Hints:

    1. :math:`Y_0` and :math:`Y_1` indicate the force constants of the host and
    nanoparticle, respectively, as defined in the original reference
    (W. Kim et al., J. Appl. Phys., 99, 084306, 2006).
    Practically, they can approximately be replaced by the Young's modulus.

    2. :math:`\\alpha`, designated as the trigonometric ratio in the original
    paper, is typically assumed to be 1 in subsequent comprehensive works.
    '''
    tag = 'NP'
    def __init__(self, vs, R, phi, D0, D1, Y0=1, Y1=1, alpha=1):
        '''
        Parameters
        ----------
        vs : float
            Average sound velocity (:math:`v_s`), in km/s.
        R : float
            Average radius of the nanoparticles (:math:`R`), in nm.
        phi : float
            Volume fraction of the nanoparticles (:math:`\\phi`),
            in the range (0, 1) theoretically.
        D0 : float
            Mass density of the host material (:math:`D_0`), in g/cm^3.
        D1 : float
            Mass density of the nanoparticles (:math:`D_1`), in g/cm^3.
        Y0 : float, optional
            Young modulus of the host material (:math:`Y_0`) in GPa, by default 1.
        Y1 : float, optional
            Young modulus of the nanoparticles (:math:`Y_1`) in GPa, by default 1.
        alpha : float, optional
            The trigonometric ratio (:math:`\\alpha`), by default 1.
        '''
        super().__init__(vs=vs, R=R, phi=phi, D0=D0, D1=D1, Y0=Y0, Y1=Y1, alpha=alpha)

    def __call__(self, w, T):
        w, _ = np.broadcast_arrays(w, T)
        # geom = vs * 2*pi*R^2 * phi/(4/3*pi*R^3) = 3/2 * vs * phi / R
        geom = 3/2 * self.paras['vs'] * self.paras['phi'] / self.paras['R']
        Gm = 1/4 * np.power(self.paras['alpha'], 2)\
             * np.power(1-self.paras['D1']/self.paras['D0'], 2)
        Gy = 3 * np.power(self.paras['alpha'], 8)\
             * np.power(1-self.paras['Y1']/self.paras['Y0'], 2)
        # Gm = np.power(1-self.paras['D1']/self.paras['D0'], 2) \
        #      + np.power(1-self.paras['Y1']/self.paras['Y0'], 2)
        w4 = np.power(w*self.paras['R']/self.paras['vs'], 4)
        # ratio = 2/9*Gm*w4   # sigma_L/sigma_S = [4/9*pi*R^2*Gm*w4] / [2*pi*R^2]
        ratio = 8/9 * (Gm+Gy) * w4 # sigma_L/sigma_S = [16/9*pi*R^2*Gm*w4] / [2*pi*R^2]
        return geom * ratio/(1+ratio)


class Dislocations(BaseScattering):
    '''
    .. math::

        \\tau_i^{-1} = \\tau_{DC}^{-1} + \\tau_{DS}^{-1}

    .. math::

        \\tau_{DC}^{-1} = \\alpha N_d \\frac{V_a^{4/3}}{v_s^2} \\omega^3

    .. math::

        \\tau_{DS}^{-1} = \\frac{2^{11/2}}{3^{7/2}} \\alpha F N_d B_d^2 \\gamma^2 \\omega

    .. math::

        F = \\frac{1}{2} + \\frac{1}{24} \\left( \\frac{1-2\\mu}{1-\\mu} \\right)^2
            \\left[ 1 + \\sqrt{2} \\left( \\frac{v_1}{v_2} \\right)^2 \\right]^2

    Hints:

    1. :math:`\\alpha` is a weight factor to account for the mutual orientation of the
    direction of the temperature gradient and the dislocation line. For dislocations
    perpendicular to the temperature gradient :math:`\\alpha=1`, while for those parallel
    to the gradient :math:`\\alpha=0`. If dislocation lines are orientated at random with
    respect to the temperature gradient, the average value found by integration is
    :math:`\\alpha=0.55` (ref: P. G. Klemens, Proc. Phys. Soc. A 68 1113, 1955).

    2. Note that the parameter :math:`F` is directly determined by the Poisson's ratio
    (:math:`\\mu`) and the ratio of longitudinal to transverse sound velocities
    (:math:`{v_1}/{v_2}`). To simplify further, we can consider the relationship
    :math:`\\left( {v_1}/{v_2} \\right)^2 = ({2-2\\mu})/({1-2\\mu})`.
    For common bulk materials, :math:`\\mu` typically ranges between 1/5 and 1/3,
    corresponding to :math:`F` values between 0.962 and 1.034. Therefore, in the absence
    of detailed material properties, it is reasonable to approximate :math:`F` as 1.

    3. In the expression for :math:`\\tau_{DS}^{-1}`, representing phonon scattering by the
    static strain field, the constant of proportionality has seen variations in Klement's research.
    In the original 1955 publication (P. G. Klemens, Proc. Phys. Soc. A 68 1113, 1955),
    the value was calculated to be :math:`2^{3/2}/3^{7/2}(\\approx 0.06)`.
    This was subsequently revised in 1958 by a factor of 16, resulting in
    :math:`2^{11/2}/3^{7/2}(\\approx 0.97)`, primarily to address the underestimation
    of the dislocation strain field in the initial value
    (P. G. Klemens, Solid State Phys. Adv. Res. Appl. 7, 1-98, 1958).
    In the current program, the revised value is adopted.
    If different considerations are required, the magnitude of :math:`\\tau_{DS}^{-1}`
    can be directly adjusted by modifying the parameter :math:`F`.
    '''
    tag = 'DL'
    def __init__(self, Nd, vs, Va, Bd, gm, F=1, alpha=0.55):
        '''
        Parameters
        ----------
        Nd : float
            The number of dislocation lines per unit area (:math:`N_d`),
            in 1E10 cm^(-2).
        vs : float
            Average sound velocity (:math:`v_s`), in km/s.
        Va : float
            Average volume per atom (:math:`V_a`), in cubic angstroms (A^3).
        Bd : float
            The magnitudes of Burgers vector (:math:`B_d`), in angstroms.
        gm : float
            Effective Gruneisen parameter (:math:`\\gamma`), dimensionless.
        F : float, optional
            A material-dependent parameter (:math:`F`), by default 1.
        alpha : float, optional
            The weight factor (:math:`\\alpha`), by default 0.55.
        '''
        super().__init__(Nd=Nd, vs=vs, Va=Va, Bd=Bd, gm=gm, F=F, alpha=alpha)

    def __call__(self, w, T):
        C = 0.9676996514698976      # constant np.power(2, 11/2)/np.power(3, 7/2)
        w, _ = np.broadcast_arrays(w, T)
        DC = 1E-8*np.power(self.paras['Va'], 4/3)/self.paras['vs']*np.power(w, 3)
        DS = 1E-6*np.power(self.paras['Bd']*self.paras['gm'], 2) * w
        return self.paras['alpha']*self.paras['Nd']*(DC + C*self.paras['F']*DS)


class StackingFaults(BaseScattering):
    '''
    .. math::

        \\tau_i^{-1} = \\alpha N_{sf} \\frac{V_a^{2/3}}{v_s} \\gamma^2 \\omega^2

    Hint: The constant of proportionality (:math:`\\alpha`) varies
    across different publications. In the original 1957 publication
    (P. G. Klemens, Can. J. Phys., 35 441, 1957), the value was derived as
    :math:`1/18 \\times 4/3 (\\approx 0.074)`. This value was later revised
    to 0.7 (P. G. Klemens, Solid State Phys. Adv. Res. Appl. 7, 1-98, 1958),
    which is also the default value adopted by the current program.
    '''
    tag = 'SF'
    def __init__(self, Nsf, vs, Va, gm, alpha=0.7):
        '''
        Parameters
        ----------
        Nsf : float
            The number of stacking faults crossing a line of unit length
            (:math:`N_{sf}`), in 10^6 m^(-1), also known as 1/um.
        vs : float
            Average sound velocity (:math:`v_s`), in km/s.
        Va : float
            Average volume per atom (:math:`V_a`), in cubic angstroms (A^3).
        gm : float
            Effective Gruneisen parameter (:math:`\\gamma`), dimensionless.
        alpha : float, optional
            Dimensionless adjustable parameter (:math:`\\alpha`), by default 0.7.
        '''
        super().__init__(Nsf=Nsf, vs=vs, Va=Va, gm=gm, alpha=alpha)

    def __call__(self, w, T):
        w, _ = np.broadcast_arrays(w, T)
        return 1E-5 * self.paras['alpha']*self.paras['Nsf']\
            * np.power(self.paras['Va'], 2/3)/self.paras['vs']\
            * np.power(self.paras['gm']*w, 2)


class FreeElectrons(BaseScattering):
    '''
    .. math::

        \\tau_i^{-1} = \\frac{(E_d m_d^\\ast)^2}{2 \\pi \\hbar^3 D_0 v_s} R_{ep}\\ \\omega
            = C^\\prime R_{ep} \\ \\omega

    .. math::

        R_{ep} = 1-\\frac{k_BT}{\\hbar \\omega} \\ln
                \\frac{
                    1+\\exp[(\\frac{\\hbar^2 \\omega^2}{8m_d^\\ast v_s^2}
                             + \\frac{1}{2}m_d^\\ast v_s^2
                             + \\frac{1}{2}\\hbar\\omega - E_F)/k_BT]
                }{
                    1+\\exp[(\\frac{\\hbar^2 \\omega^2}{8m_d^\\ast v_s^2}
                             + \\frac{1}{2}m_d^\\ast v_s^2
                             - \\frac{1}{2}\\hbar\\omega - E_F)/k_BT]
                }

    Ref: J. M. Ziman, Philos. Mag. 1, 191, 1956.
    '''
    tag = 'EP'
    def __init__(self, vs=None, md=None, EF=None, Ed=None, D0=None, *, Rep=None, coef=None):
        '''
        Parameters
        ----------
        vs : float
            Average sound velocity (:math:`v_s`), in km/s.
        md : float
            Density-of-state effective mass (:math:`m_d^\\ast`), in :math:`m_e`.
        EF : float
            Fermi energy (:math:`E_F`), in eV.
        Ed : float
            Deformation potential (:math:`E_d`), in eV.
        D0 : float
            Mass density of the material (:math:`D_0`), in g/cm^3.
        Rep : float, optional
            A dimensionless parameter (:math:`R_{ep}`) in the range of (0, 1)
            related to Fermi level (or doping concentration). At ultra-high
            carrier concentrations, it tends towards 1. The default is None,
            indicating that it will be calculated based on Fermi level `EF`.
            If set, `EF` become non-effective.
        coef : float, optional
            A comprehensive adjustable parameter (:math:`C^{\\prime}`).
            When set, all parameters except `Rep` become non-effective.
            By default, it is None.
        '''
        if coef is None:
            if Rep is None:
                if any(v is None for v in [vs, md, EF, Ed, D0]):
                    raise ValueError('vs, md, EF, Ed, and D0 are required')
                super().__init__(vs=vs, md=md, EF=EF, Ed=Ed, D0=D0)
            else:
                if any(v is None for v in [vs, md, Ed, D0]):
                    raise ValueError('vs, md, Ed, and D0 are required')
                super().__init__(vs=vs, md=md, Ed=Ed, D0=D0, Rep=Rep)
        else:
            if Rep is None:
                if any(v is None for v in [vs, md, EF]):
                    raise ValueError('vs, md, and EF are required')
                super().__init__(vs=vs, md=md, EF=EF, coef=coef)
            else:
                super().__init__(Rep=Rep, coef=coef)

    def __call__(self, w, T):
        w, T = np.broadcast_arrays(w, T)
        return self.coef * self.Rep(w, T) * w

    @property
    def coef(self):
        '''Value of :math:`C^{\\prime}`'''
        return self.paras.get('coef') or self._cal_coef()

    def _cal_coef(self):
        # Ed^2 * md^2 / (2*pi* hbar^3 * D0 * vs)
        # factor = np.power(1.602176634e-19 * 9.1093837015e-31, 2) \
        #     / (2*np.pi*np.power(1.054571817e-34, 3)*1e3*1e3) # => 2.89061617105021e-3
        return 2.89061617105021e-3 \
            * np.power(self.paras['Ed']*self.paras['md'],2) \
            / (self.paras['D0']*self.paras['vs'])

    def Rep(self, w, T):
        '''Returen the provided `Rep`, or the value calculated by `EF` if not provided.'''
        return self.paras.get('Rep') or self._cal_Rep(w, T)

    def _cal_Rep(self, w, T):
        md, vs = self.paras['md'], self.paras['vs']
        w, T = np.broadcast_arrays(np.maximum(w, 1E-6), T)

        ###### x = (hbar*w) / (kB*T)
        # hbar/kB * 1E12 = 7.638232577577646
        x = 7.638232577577646 * w/T

        ###### Tele = EF/kB
        # kB / q = 8.617333262145179e-05
        Tele = np.divide(self.paras['EF'], 8.617333262145179e-05) # EF/(kB)

        ###### Tph hbar^2 * w^2 / (4^2 * 1/2 * m_d * vs^2 * kB) + 1/2 * m_d * vs^2 / kB
        ### => (hbar * w / 4 / kB)^2 / (Emv/kB) + (Emv/kB)
        # 1/2 * 9.1093837015e-31 * 1e3 * 1e3 / 1.380649e-23 # => 0.03298949878462954
        Tmv = 0.03298949878462954 * np.multiply(md, np.power(vs, 2))
        # np.power(1.054571817e-34/(4*1.380649e-23) * 1E12, 2) # => 3.6464123068230276
        Thw = 3.6464123068230276 * np.power(w, 2)
        Tph = Thw/Tmv + Tmv    # in K

        # y = (Tph - Tele) / T
        # return 1-1/x*np.log((1+np.exp(y+x/2))/(1+np.exp(y-x/2)))
        p = np.exp(np.minimum((Tph - Tele) / T, 23))    # p < 1E10
        return np.log((np.exp(x/2)+p)/(np.exp(-x/2)+p))/x


class CahillScattering(BaseScattering):
    '''
    .. math::

        \\tau_i^{-1} = \\alpha \\frac{\\omega}{\\pi}
    '''
    tag = 'CAHILL'
    def __init__(self, alpha=1):
        '''
        Parameters
        ----------
        alpha : float, optional
            Dimensionless adjustable parameter (:math:`\\alpha`), by default 1.
        '''
        super().__init__(alpha=alpha)

    def __call__(self, w, T):
        w, _ = np.broadcast_arrays(w, T)
        return self.paras['alpha'] * w/np.pi


class KappaSlack(BaseKappaModel):
    '''
    .. math::

        \\kappa = A \\frac{M_a V_a^{1/3} \\Theta^3}{\\gamma^2 N^{2/3}T}

    Hint: Here :math:`\\Theta` refers to the "traditional" Debye temperature
    (namely, that determined from the elastic constants or specific heat),
    not the acoustic-mode Debye temperature :math:`\\Theta_a`.
    Their relationship is :math:`\\Theta^3 = N \\cdot \\Theta_a^3`,
    where :math:`N` is the number of atoms per unit cell.
    '''
    tag = 'SLACK'
    def __init__(self, td, gm, Ma, Va, N=1, A=1E-6):
        '''
        Parameters
        ----------
        td : float
            Debye temperature (:math:`\\Theta`), in Kelvin.
        gm : float
            Gruneisen parameter (:math:`\\gamma`), dimensionless.
        Ma : float
            Average atomic mass per atom (:math:`M_a`), in atomic mass units (amu).
        Va : float
            Average volume per atom (:math:`V_a`), in cubic angstroms (A^3).
        N : int, optional
            The number of atoms per unit cell, by default 1.
        A : float or str, optional
            Adjustable parameter :math:`A` in the model, which can be set as
            either a constant value or the name of a predefined model for
            computation during initialization. The default value is 1E-6,
            which is its typical magnitude as well. To customize a new model,
            inherit from this class and implement a property named
            `coef_<modelname>`, then it will be automatically invoked.
        '''
        if isinstance(A, str):
            self._coef_model = A
            super().__init__(td=td, gm=gm, Ma=Ma, Va=Va, N=N)
        else:
            self._coef_model = 'Constant'
            super().__init__(td=td, gm=gm, Ma=Ma, Va=Va, N=N, A=A)

    def __call__(self, T):
        return self.coef * self.paras['Ma'] * np.power(self.paras['Va'], 1/3) *\
            np.power(self.paras['td'], 3) / np.power(self.paras['gm'], 2) /\
            np.power(self.paras['N'], 2/3) / T

    @property
    def coef(self):
        '''
        The value of parameter :math:`A` in the model.
        '''
        return getattr(self, f'coef_{self._coef_model}')

    @property
    def coef_Julian(self):
        '''
        .. math ::

            A = \\frac{2.43 \\times 10^{-6}}{1-0.514/\\gamma+0.228/\\gamma^2}

        Ref: C. L. Julian, Physical Review, 137 A128, 1965.
        '''
        gm = self.paras['gm']
        return 2.43E-6 / (1-0.514/gm+0.228/gm/gm)

    @property
    def coef_Qin(self):
        '''
        .. math ::

            A = \\frac{1}{1+1/\\gamma + 8.3 \\times 10^5/\\gamma^{2.4}}

        Ref: G. Qin et al., Mater. Adv., 3 6683, 2022.
        '''
        gm = self.paras['gm']
        return 1/(1+1/gm+8.3E5/np.power(gm, 2.4))

    @property
    def coef_Constant(self):
        '''
        This property works when the parameter `A` is specified as a
        numerical value during initialization, and returns the
        provided value of `A` directly.
        '''
        return self.paras['A']


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
    
        \\kappa = \\kappa_{amb} \\left ( \\frac{T}{T_{amb}} \\right ) ^n
                  + \\kappa_0 \\text{, where } T_{amb} = 300 K
        
    '''
    tag = 'POWERLAW'
    def __init__(self, Kamb, n=-1, K0=0):
        '''
        Parameters
        ----------
        Kamb : float
            Ambient thermal conductivity (:math:`\\kappa_{amb}`),
            corresponds to the thermal conductivity at the ambient temperature
            (300 K) with the background offset excluded.
        n : float, optional
            The exponent in the expression, by default -1.
        K0 : float, optional
            Background offset (:math:`\\kappa_0`), by default 0.
        '''
        super().__init__(Kamb=Kamb, n=n, K0=K0)
    
    def __call__(self, T):
        pow = np.float_power(np.divide(T, 300), self.paras['n'])
        return self.paras['Kamb'] * pow + self.paras['K0']


class KappaKlemens(BaseKappaModel):
    '''
    .. math::

        \\frac{\\kappa(x)}{\\kappa_{pure}} =
        \\frac{\\tan^{-1}(u)}{u} \\text{, where }
        u^2 = \\cfrac{\\pi \\Theta V_a}{2\\hbar v_s^2}
        \\kappa_{pure}\\Gamma_0 \\cdot x (1-x)

    Ref: P. G. Klemens, Phys. Rev., 119 507-509, 1960.
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
        u = np.sqrt(np.maximum(self.u2(X), 1E-12))
        return self.paras['Kpure'] * np.arctan(u)/u

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


EXECMETA = {
    'DEBYE': ExecWrapper(KappaDebye,
        args=['vs', 'td', 'components',],
    ),
    'PH': ExecWrapper(ThreePhonon,
        args=['gm', 'vs', 'Va', 'Ma',],
        opts=['A', 'Theta',],
    ),
    'PHX': ExecWrapper(ThreePhonon,
        args=['coef',],
        opts=['Theta',],
    ),
    'PD': ExecWrapper(PointDefect,
        args=['vs', 'Va', 'G',],
    ),
    'PDX': ExecWrapper(PointDefect,
        args=['coef',],
    ),
    'GB': ExecWrapper(GrainBoundary,
        args=['vs', 'L',],
        opts=['alpha',],
    ),
    'NP': ExecWrapper(Nanoparticles,
        args=['vs', 'R', 'phi', 'D0', 'D1',],
        opts=['Y0', 'Y1', 'zeta',],
    ),
    'DL': ExecWrapper(Dislocations,
        args=['Nd', 'vs', 'Va', 'Bd', 'gm',],
        opts=['F', 'alpha',],
    ),
    'SF': ExecWrapper(StackingFaults,
        args=['Nsf', 'vs', 'Va', 'gm',],
        opts=['alpha',],
    ),
    'EP': ExecWrapper(FreeElectrons,
        args=['vs', 'md', 'EF', 'Ed', 'D0',],
    ),
    'EPR': ExecWrapper(FreeElectrons,
        args=['vs', 'md', 'Ed', 'D0', 'Rep',],
    ),
    'EPX': ExecWrapper(FreeElectrons,
        args=['vs', 'md', 'EF', 'coef',],
    ),
    'EPRX': ExecWrapper(FreeElectrons,
        args=['Rep', 'coef',],
    ),
    'CAHILL': ExecWrapper(CahillScattering,
        opts=['alpha',],
    ),
    'SLACK': ExecWrapper(KappaSlack,
        args=['td', 'gm', 'Ma', 'Va', 'A',],
        opts=['N', ],
    ),
    'BIPOLAR': ExecWrapper(KappaBipolar,
        args=['Kbp', 'Eg',],
        opts=['p',],
    ),
    'POWERLAW': ExecWrapper(KappaPowerLaw,
        args=['Kamb',],
        opts=['n', 'K0',],
    ),
    'KLEMENS': ExecWrapper(KappaKlemens,
        args=['Kpure', 'vs', 'td', 'G0', 'Va',],
    ),
}


def parse_KappaFit(filename, specify=None):
    '''Parse kappa model from configuration file'''
    config = CfgParser()
    with open(filename, 'r') as f:
        config.read_file(f)
        logger.info(f'Read configuration from {filename}')

    entry = config['entry']
    logger.debug('Found entry section')

    if specify:
        entry.update(specify)
        logger.debug('Update specify setting to entry:\n  %s' % specify)

    # parse variables
    tags = []
    fitting = []
    varbdict = OrderedDict()
    if 'variables' in config:
        logger.debug('Found variables section')
        varbpat = re.compile(r'\s*(?P<initial>\S+)?\s*\?\s*(?P<scale>[^\s<@]+)?\s*'
                             r'(<\s*(?P<lower>\S+)?\s*,\s*(?P<upper>\S+)?\s*>)?')
        for tag, val in config['variables'].items():
            m = varbpat.match(val)
            if m:
                logger.debug(f'{tag}: {m.groupdict()}')     # parsed raw text
                setting = {k:float(v) for k,v in m.groupdict().items() if v}
                varb = Variable(tag, **setting)
                tags.append(f'{tag}?')
                fitting.append(varb)
            else:
                tags.append(tag)
                try:
                    varb = float(val)
                except ValueError:
                    raise ValueError(f"Failed to parse variable '{tag}'")
            varbdict[tag] = varb
        logger.info(f'Variables: {", ".join(tags)}')
    else:
        logger.debug(('Not found variables section'))

    # parse model
    model_ = entry.get('model')
    if model_ is None:
        raise ValueError("Parameter '%s' is required in entry section!", 'model')
    logger.info(f"Parse model of kappa: {model_}")

    refpat = re.compile(r'^\s*@\s*(?P<tag>\S+)?\s*$')
    def parse_submodel(modelname):
        subsect, mtype = config.pmatch(modelname)
        logger.debug('Build %s.%s with variables:', modelname, mtype)
        if mtype not in EXECMETA:
            raise TypeError(modelname)
        kwargs = {}
        for key, val in subsect.items():
            if not val.strip():
                raise ValueError(f"{mtype} :: '{key}' is empty")

            # some special
            if key == 'components':
                comps_ = subsect.getlist(key)
                logger.debug(f'    {key} = [{", ".join(comps_)}]')
                kwargs[key] = [parse_submodel(comp) for comp in comps_]
                continue

            refmatch = refpat.match(val)    # match referenced variable
            if refmatch:
                tag = refmatch.groupdict().get('tag') or key
                if tag not in varbdict:
                    raise ValueError(f'Variable {tag} not found')
                varb = varbdict[tag]
                if hasattr(varb, 'tag'):
                    logger.debug(f' -> {key} = {str(varb).split(": ")[1]}')
                else:
                    logger.debug(f'  * {key} = {varb} @ {tag}')
                kwargs[key] = varbdict[tag]
            else:
                logger.debug(f'    {key} = {val}')
                try:
                    kwargs[key] = float(val)
                except ValueError:
                    kwargs[key] = val
        modelobj = EXECMETA[mtype].execute(**kwargs)
        if modelobj.tag != modelname:
            modelobj.tag = modelname
        return modelobj

    model = parse_submodel(model_)

    # parse dataX, dataY; fit model
    if fitting:
        expdata_ = entry.getarray('expdata')
        if expdata_ is None:
            raise ValueError("Parameter 'expdata' is required in entry section!")

        dataX, dataY, *_ = expdata_
        logger.info('Read experimental data successfully, run fitting ...')

        model.fit(dataX, dataY, variables=fitting)
        logger.info('Fitting results:')
        logger.info('='*50)
        dsp = '{0.tag:>10s} :  {0.value:<16.6g} X {0.scale:<16.6g}'
        for varb in fitting:
            logger.warning(dsp.format(varb))
        logger.info('='*50)
    else:
        dataX, dataY = None, None

    if fitting and entry.getboolean('substituted', False):
        with open(filename, 'r') as f:
            subs = f.readlines()

        numstart = None
        numend = len(subs)
        for numline, line in enumerate(subs):
            matched = config.SECTCRE.match(line.strip())
            if matched:
                if numstart is not None:
                    numend = numline
                    break
                if matched.group('header') == 'variables':
                    numstart = numline

        logger.debug(f"Section 'variables' range: {numstart} to {numend}")
        linepat = re.compile(r'(?<==).*?\?')
        for varb in fitting:
            tag = varb.tag
            value = varb.value * varb.scale
            for num in range(numstart, numend):
                matched = re.match(rf'{tag}\s*=.*?\?', subs[num].strip())
                if matched:
                    logger.debug(f'Overwrite value of {tag}: {value:.6g}')
                    subs[num] = linepat.sub(f' {value:.6g} # ?', subs[num], 1)
                    break
            else:
                raise RuntimeError(f'Failed to overwrite variable: {tag}')
        logger.info('Generate substituted configuration file')
    else:
        subs = None

    # parse predX and other options
    npoints=abs(entry.getint('npoints', 101))
    predX_ = entry.getseq('predict', None)
    if predX_:
        predX = np.array(predX_)
    else:
        margin=abs(entry.getfloat('margin', 0.05))
        if dataX is not None:
            xmin = (1+margin) * dataX.min() - margin * dataX.max()
            xmax = (1+margin) * dataX.max() - margin * dataX.min()
        else:
            raise RuntimeError("Failed to parse sample points for prediction")
        predX = np.linspace(xmin, xmax, npoints)
        logger.info(f'Predict from {xmin:.4g} to {xmax:.4g} with {npoints} points')
    predY = [model(predX),]
    predL = [f'Kappa-{model.tag}',]
    options = AttrDict(
        dataX=dataX,
        dataY=dataY,
        predX=predX,
        predY=predY,
        predL=predL,
        subs=subs,
    )

    if model.tag != KappaDebye.tag:
        return model, fitting, options
    # only for Debye model
    logger.debug('KappaDebye model is detected.')
    if entry.getboolean('splitkappa', True):
        logger.info('Split kappa to scattering mechanism and additional model')
        predL[0] = 'Kappa-Total'
        tag_scats = []
        for i, itag in enumerate(model._scattering):
            tag_scats.append(itag)
            predY.append(model(predX, nscat=i+1))
            predL.append('+'.join(tag_scats))
        for tag_model, submodel in model._additional.items():
            predY.append(submodel(predX))
            predL.append(tag_model)
    temper = entry.getfloat('temperature', 300)
    dsp_temp = f'Caluclate %s at {temper} K'
    funit_ = entry.get('frequnit', '2pi.THz')
    try:
        funit = float(funit_)
        funit_ = f'{funit:.6g}'
    except ValueError:
        if re.match(r'THz', funit_, re.IGNORECASE):
            # f = w/(2*pi)
            funit = 2*np.pi
            funit_ = 'THz'
        elif re.match(r'(omega_|w_?)d|norm', funit_, re.IGNORECASE):
            # wr = w/wd
            funit = model.wd
            funit_ = 'Normalized'
        elif re.match(r'reduce|x', funit_, re.IGNORECASE):
            # x = (hb*w)/(kB*T) = w/[T/(hb/kB)]
            funit = temper/model._hbar_kB
            funit_ = 'Reduced'
        elif re.match(r'meV', funit_, re.IGNORECASE):
            # meV = (hb*w)/q = w/[q/hb*1E-15]
            funit = 1.602176634/1.054571817 # q/hbar
            funit_ = 'meV'
        else:
            funit = 1
            funit_ = '2pi.THz'
    logger.debug(f'Using the unit of phonon frequency: {funit_}')
    if entry.getboolean('scattering', True):
        logger.info(dsp_temp, 'scattering rate (in THz) of phonon')
        rateX = np.linspace(0, model.wd, npoints)
        scattering = model.scattering(rateX, temper, with_total=True)
        options['rateX'] = rateX / funit
        options['rateY'] = list(scattering.values())
        options['rateL'] = list(scattering.keys())
    if entry.getboolean('spectral', True):
        logger.info(dsp_temp, 'spectral kappa on frequency')
        specX = np.linspace(0, model.wd, npoints)
        spectrals = model.spectral(specX, temper)
        options['specX'] = specX / funit
        options['specY'] = [v*funit for v in spectrals.values()]
        options['specL'] = list(spectrals.keys())
    if entry.getboolean('cumulate', True):
        logger.info(dsp_temp, 'cumulative kappa on mean-free-path (in nm)')
        cumuX = np.logspace(-1.3, 4.7, npoints)
        cumulates = model.cumulative_mfp_t(cumuX, temper)
        options['cumuX'] = cumuX
        options['cumuY'] = list(cumulates.values())
        options['cumuL'] = list(cumulates.keys())
    return model, fitting, options
