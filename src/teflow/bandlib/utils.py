from functools import wraps, partial
import numpy as np


kB = 8.617333262145179e-05      # eV/K
m_e = 9.1093837015e-31          # kg
hbar = 1.054571817e-34          # J.s
e0 = 1.602176634e-19            # C
q0 = 1.602176634                # 1E-19 C

def vectorize(__pyfunc=None, **kwargs):
    '''
    A wraper of numpy.vectorize 

    Parameters
    ----------
    __pyfunc : callable, optional
        A python callable object

    Returns
    -------
    A wraped numpy.vectorize
    
    Example 1:
    @vectorize
    def func(...):
        ...
    
    Example 2:
    @vectorize(otype=...)
    def func(...):
        ...
        
    Example 2:
    func = vectorize(func, ...)
    '''
    if __pyfunc is None:
        return partial(vectorize, **kwargs)
    else:
        return wraps(__pyfunc)(np.vectorize(__pyfunc, **kwargs))

