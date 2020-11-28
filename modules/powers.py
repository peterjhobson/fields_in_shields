# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:29:01 2019

@author: PJ Hobson

Calculate power weightings from Eq.(41) of https://arxiv.org/abs/2006.02981.
"""

import numpy as np
import scipy.special as special
import mpmath as mp


def pn0(rho_c, z_prime_c1, z_prime_c2, args):
    """Calculate the zonal power contribution for a cylindrical coil.

    Keyword arguments:
    rho_c -- radius of cylindrical coil
    z_prime_c1 -- top position of cylindrical coil
    z_prime_c2 -- bottom position of cylindrical coil
    args -- res -- resistance of coil
         -- t -- thickness of coil
    """

    res, t = args

    return (res * rho_c / t) * np.pi * (z_prime_c1 - z_prime_c2)


def pnm(rho_c, z_prime_c1, z_prime_c2, n, m, args):
    """Calculate the tesseral power contribution for a cylindrical coil. Order n may be implemented as an array for a
    fixed degree m.

    Keyword arguments:
    rho_c -- radius of cylindrical coil
    z_prime_c1 -- top position of cylindrical coil
    z_prime_c2 -- bottom position of cylindrical coil
    n -- order of spherical harmonic
    m -- degree of spherical harmonic
    args -- res -- resistance of coil
         -- t -- thickness of coil
    """
    res, t = args

    return ((res * rho_c / t) * (np.pi * (z_prime_c1 - z_prime_c2) / 2
                                 + m ** 2 * (z_prime_c1 - z_prime_c2) ** 3 / (2 * np.pi * n ** 2 * rho_c ** 2)))


def pn0_planar(rho_p, rho_n0, args):
    """Calculate the zonal power contribution for a planar coil.

    Keyword arguments:
    rho_p -- radius of planar coil
    rho_n0 -- nth zero of the first bessel function of order 0
    args -- res -- resistance of coil
         -- t -- thickness of coil
    """
    res, t = args

    return (np.pi * res * rho_p ** 2 / t) * rho_n0 ** 2 * special.j1(rho_n0) ** 2


def pn1_planar(rho_p, rho_n1, args):
    """Calculate the first order tesseral power contribution for a planar coil.

    Keyword arguments:
    rho_p -- radius of planar coil
    rho_n0 -- nth zero of the first bessel function of order 1
    args -- res -- resistance of coil
         -- t -- thickness of coil
    """

    res, t = args

    return (np.pi * res * rho_p ** 2 / (2 * t)) * rho_n1 ** 2 * special.j0(rho_n1) ** 2


def pn2_planar(rho_p, rho_n2, args):
    """Calculate the second order tesseral power contribution for a planar coil.

    Keyword arguments:
    rho_p -- radius of planar coil
    rho_n0 -- nth zero of the first bessel function of order 2
    args -- res -- resistance of coil
         -- t -- thickness of coil
    """
    res, t = args

    return ((np.pi * res * rho_p ** 2 / (2 * t)) * (
            (rho_n2 ** 2 - 4) * special.j0(rho_n2) ** 2 -
            2 * ((rho_n2 ** 2 - 8) / rho_n2) * special.j0(rho_n2) * special.j1(rho_n2) +
            ((rho_n2 ** 4 - 16) / rho_n2 ** 2) * special.j1(rho_n2) ** 2))


def pn3_planar(rho_p, rho_n3, args):
    """Calculate the third order tesseral power contribution for a planar coil.

    Keyword arguments:
    rho_p -- radius of planar coil
    rho_n0 -- nth zero of the first bessel function of order 3
    args -- res -- resistance of coil
         -- t -- thickness of coil
    """
    res, t = args

    return ((np.pi * res * rho_p ** 2 / (2 * t)) * (
            ((rho_n3 ** 4 - 96) / rho_n3 ** 2) * special.j0(rho_n3) ** 2 +
            -48 * ((rho_n3 ** 2 - 8) / rho_n3 ** 3) * special.j0(rho_n3) * special.j1(rho_n3) +
            ((rho_n3 ** 6 - 10 * rho_n3 ** 4 + 96 * rho_n3 ** 2 - 384) / rho_n3 ** 4) * special.j1(rho_n3) ** 2))


def pnm_planar(rho_p, rho_nm, m, args):
    """Calculate the nth order tesseral power contribution for a planar coil. Is not designed for numpy arrays.

    Keyword arguments:
    rho_p -- radius of planar coil
    rho_nm -- nth zero of the first bessel function of order m
    args -- res -- resistance of coil
         -- t -- thickness of coil
    """

    res, t = args

    return float((np.pi * res * rho_p ** 2 / t) * (rho_nm / 2) ** (2 * m) * (
            1 / (np.math.factorial(m) * np.math.factorial(m - 1))) * (
                         mp.hyper([m, m + 1 / 2], [m + 1, m + 1, 2 * m + 1], -rho_nm ** 2) +
                         -(rho_nm ** 2 / (2 * (m + 1) ** 2)) * mp.hyper([m + 1 / 2, m + 1, m + 1],
                                                                        [m, m + 2, m + 2, 2 * m + 1], -rho_nm ** 2)))
