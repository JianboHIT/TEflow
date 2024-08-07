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

import numpy as np


def project3d(theta_rad, phi_rad, C11, C22, C33, C23=0, C13=0, C12=0):
    '''
    Projects a symmetric tensor onto a specified direction in 3D space.

    Parameters
    ----------
    theta_rad : float
        The azimuthal angle in radians, measured from the positive x-axis in the x-y plane.
    phi_rad : float
        The polar angle in radians, measured from the positive z-axis.
    C11 : array-like-
        The tensor component C11.
    C22 : array-like
        The tensor component C22.
    C33 : array-like
        The tensor component C33.
    C23 : array-like, optional
        The tensor component C23 (= C32). Default is 0.
    C13 : array-like, optional
        The tensor component C13 (= C31). Default is 0.
    C12 : array-like, optional
        The tensor component C12 (= C21). Default is 0.

    Returns
    -------
    array-like
        The projected value(s) of the tensor in the specified direction.
    '''
    cx = np.sin(phi_rad) * np.cos(theta_rad)
    cy = np.sin(phi_rad) * np.sin(theta_rad)
    cz = np.cos(phi_rad)

    return cx*cx * C11 + cy*cy * C22 + cz*cz * C33 \
           + 2 * (cy*cz * C23 + cz*cx * C13 + cx*cy * C12)

def anilay(C_list, S_list, K_list, widths=None):
    '''
    Calculate effective properties of anisotropic layered composite materials.

    This function computes the effective electrical conductivity, Seebeck
    coefficient, and thermal conductivity in both vertical and parallel
    directions for materials composed of layered anisotropic composites.

    Parameters
    ----------
    C_list : array_like
        List or array of electrical conductivities for each layer.
    S_list : array_like
        List or array of Seebeck coefficients for each layer.
    K_list : array_like
        List or array of thermal conductivities for each layer.
    widths : array_like, optional
        List or array of layer widths. If not provided, all layers are assumed
        to have equal width, by default None.

    Returns
    -------
    tuple
        A tuple containing six elements:
        - Effective electrical conductivity in the vertical direction.
        - Effective electrical conductivity in the parallel direction.
        - Effective Seebeck coefficient in the vertical direction.
        - Effective Seebeck coefficient in the parallel direction.
        - Effective thermal conductivity in the vertical direction.
        - Effective thermal conductivity in the parallel direction.

    Notes
    -----
    The input arguments will be automatically broadcasted, so their shapes
    must be compatible.
    '''
    C, S, K, w = np.broadcast_arrays(C_list, S_list, K_list, widths or 1.0)
    return (
        np.sum(w)/np.sum(w/C),
        np.sum(w*C)/np.sum(w),
        np.sum(w/K*S)/np.sum(w/K),
        np.sum(w*C*S)/np.sum(w*C),
        np.sum(w)/np.sum(w/K),
        np.sum(w*K)/np.sum(w),
    )
