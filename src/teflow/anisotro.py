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
    array-like
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
    return np.asarray([C11, C22, C33, C23, C13, C12])

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

def project3d(theta_rad, phi_rad, /, C11, C22, C33, C23=0, C13=0, C12=0):
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

def rotate3d(rotation, /, C11, C22, C33, C23=0, C13=0, C12=0, *, orient=None):
    '''
    Rotates a symmetric tensor according to the specified rotation.

    This function allows you to rotate a symmetric tensor either by using a
    predefined rotation matrix or by specifying an orientation and a rotation angle.
    Mathematically, this operation can be expressed by the matrix transformation:

    .. math::

        C^{\prime} = R C R^T

    where :math:`R` is the rotation matrix, :math:`C` is the original tensor,
    and :math:`C^{\prime}` is the rotated tensor.
    This operation preserves the symmetry of the tensor.

    Parameters
    ----------
    rotation : (3,3) array_like, or float
        Interpreted as a (3,3) rotation matrix if `orient` is None (default).
        Otherwise, `rotation` is treated as the rotation angle in radians.
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
    orient : (3,) array-like, or {'x', 'y', 'z'}, optional
        Specifies the direction of rotation. If None (default), `rotation`
        is assumed to be a rotation matrix. When `orient` is provided,
        it defines the axis, and `rotation` is interpreted as the corresponding
        angle in radians. For simplicity, you can use 'x', 'y', or 'z' to indicate
        the standard Cartesian axes instead of providing a 3D vector.

    Returns
    -------
    array-like
        The rotated tensor in Voigt sequential form as (C11, C22, C33,
        C23, C13, C12).

    Examples
    --------
    Consider a material with anisotropic properties, represented by a
    diagonal tensor with values 1, 2, and 4 along its principal directions.
    When this material is rotated 30 degrees around the y-axis and then
    45 degrees around the z-axis, the tensor describing the material's
    properties transforms as follows:

    .. code-block:: python

        >>> T = [1, 2, 4]
        >>> Ty = rotate3d(np.radians(30), *T, orient='y')
        >>> Tyz = rotate3d(np.radians(45), *Ty, orient='z')
        >>> print(Tyz)
        ...

    To achieve the same final orientation with a single equivalent rotation,
    we can compute the combined rotation matrix (see :func:`comb_rotations`):

    .. code-block:: python

        >>> R = comb_rotations(np.pi/6, np.pi/4)

    Applying this combined rotation matrix directly to the original tensor gives:

    .. code-block:: python

        >>> Tr = rotate3d(R, *T)
        >>> print(Tr)
        ...

    The results of `Tyz` and `Tr` should be identical.
    '''
    if orient is None:
        rot_mat = np.asarray(rotation)
        R11, R12, R13 = rot_mat[0]
        R21, R22, R23 = rot_mat[1]
        R31, R32, R33 = rot_mat[2]
    elif isinstance(orient, str):
        axis = str.lower(orient)[0]
        theta = np.asarray(rotation)
        R11, R12, R13 = np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta)
        R21, R22, R23 = np.zeros_like(theta), np.cos(theta), -np.sin(theta)
        R31, R32, R33 = np.zeros_like(theta), np.sin(theta), np.cos(theta)

        if axis == 'x':
            pass
        elif axis == 'y':
            R11, R22, R33 = R33, R11, R22
            R12, R23, R31 = R31, R12, R23
            R13, R21, R32 = R32, R13, R21
        elif axis == 'z':
            R11, R22, R33 = R22, R33, R11
            R12, R23, R31 = R23, R31, R12
            R13, R21, R32 = R21, R32, R13
        else:
            raise ValueError("Invalid rotate_axis. Choose from "
                             "'x', 'y', 'z' or provide a 3D vector.")
    else:
        ux, uy, uz = np.asarray(orient) / np.linalg.norm(orient)
        vcos, vsin = np.cos(rotation), np.sin(rotation)
        R11 = vcos + ux**2 * (1-vcos)
        R22 = vcos + uy**2 * (1-vcos)
        R33 = vcos + uz**2 * (1-vcos)
        R23 = uy * uz * (1-vcos) - ux * vsin
        R13 = ux * uz * (1-vcos) + uy * vsin
        R12 = ux * uy * (1-vcos) - uz * vsin
        R32 = uz * uy * (1-vcos) + ux * vsin
        R31 = uz * ux * (1-vcos) - uy * vsin
        R21 = uy * ux * (1-vcos) + uz * vsin

    # Compute each component analytically
    rot_C11 = (R11*R11 * C11 + R12*R12 * C22 + R13*R13 * C33 +
               2 * (R11*R12*C12 + R11*R13*C13 + R12*R13*C23))

    rot_C22 = (R21**2 * C11 + R22**2 * C22 + R23**2 * C33 +
               2 * (R21*R22*C12 + R21*R23*C13 + R22*R23*C23))

    rot_C33 = (R31**2 * C11 + R32**2 * C22 + R33**2 * C33 +
               2 * (R31*R32*C12 + R31*R33*C13 + R32*R33*C23))

    rot_C12 = (R11*R21 * C11 + R12*R22 * C22 + R13*R23 * C33 +
               (R11*R22 + R12*R21) * C12 +
               (R11*R23 + R13*R21) * C13 +
               (R12*R23 + R13*R22) * C23)

    rot_C13 = (R11*R31 * C11 + R12*R32 * C22 + R13*R33 * C33 +
               (R11*R32 + R12*R31) * C12 +
               (R11*R33 + R13*R31) * C13 +
               (R12*R33 + R13*R32) * C23)

    rot_C23 = (R21*R31 * C11 + R22*R32 * C22 + R23*R33 * C33 +
               (R21*R32 + R22*R31) * C12 +
               (R21*R33 + R23*R31) * C13 +
               (R22*R33 + R23*R32) * C23)
    return np.asarray([rot_C11, rot_C22, rot_C33, rot_C23, rot_C13, rot_C12])

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
