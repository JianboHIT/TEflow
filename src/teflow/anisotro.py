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


def tens2seq(matrix, use_upper=None):
    '''
    Converts a symmetric 3x3 tensor matrix to Voigt sequential form.

    Parameters
    ----------
    matrix : (3,3) array-like
        A 3x3 symmetric tensor matrix.
    use_upper : bool or None, optional
        Determines which part of the matrix to use for the off-diagonal elements:
        - If True, uses the upper triangular part.
        - If False, uses the lower triangular part.
        - If None (default), averages the upper and lower triangular parts.

    Returns
    -------
    tuple of array-like
        The tensor in Voigt sequential form as (C11, C22, C33, C23, C13, C12).
    '''
    matrix = np.asarray(matrix)
    C11 = matrix[0, 0]
    C22 = matrix[1, 1]
    C33 = matrix[2, 2]

    if use_upper is None:
        C23 = (matrix[1, 2] + matrix[2, 1]) / 2
        C13 = (matrix[0, 2] + matrix[2, 0]) / 2
        C12 = (matrix[0, 1] + matrix[1, 0]) / 2
    elif use_upper:
        C23 = matrix[1, 2]  # C23 from upper triangle
        C13 = matrix[0, 2]  # C13 from upper triangle
        C12 = matrix[0, 1]  # C12 from upper triangle
    else:
        C23 = matrix[2, 1]  # C23 from lower triangle
        C13 = matrix[2, 0]  # C13 from lower triangle
        C12 = matrix[1, 0]  # C12 from lower triangle
    return C11, C22, C33, C23, C13, C12

def seq2tens(C11, C22, C33, C23=0, C13=0, C12=0):
    '''
    Converts a tensor from Voigt sequential form to its 3x3 matrix form.

    Parameters
    ----------
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
    (3,3) array
        The tensor in 3x3 matrix form.
    '''
    return np.asarray([[C11, C12, C13],
                       [C12, C22, C23],
                       [C13, C23, C33]])

def comb_rotations(theta_y, theta_z):
    '''
    Combines two rotations: one around the y-axis and one around the z-axis,
    into a single rotation matrix.

    Parameters
    ----------
    theta_y : float
        The rotation angle around the y-axis (in radians).
    theta_z : float
        The rotation angle around the z-axis (in radians).

    Returns
    -------
    (3,3) array
        The combined rotation matrix obtained by first rotating around the
        y-axis by `theta_y` and then around the z-axis by `theta_z`.
    '''
    R_y = np.array([[ np.cos(theta_y), 0, np.sin(theta_y)],
                    [        0       , 1,         0       ],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])

    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z),  np.cos(theta_z), 0],
                    [       0       ,         0       , 1]])
    return R_z @ R_y

def project3d(theta_rad, phi_rad, C11, C22, C33, C23=0, C13=0, C12=0):
    '''
    Projects a symmetric tensor onto a specified direction in 3D space.

    Parameters
    ----------
    theta_rad : float
        The azimuthal angle in radians, measured from the positive x-axis in the x-y plane.
    phi_rad : float
        The polar angle in radians, measured from the positive z-axis.
    C11 : array-like
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
