# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:29:01 2019

@author: PJ Hobson

Calculate power weightings from Eq.(41) of https://arxiv.org/abs/2006.02981.
"""

import numpy as np


def pn0(rho_c, z_prime_c1, z_prime_c2, args):
    """Calculate the zonal power contribution.

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
    """Calculate the tesseral power contribution. Order n may be implemented as an array for a fixed degree m.

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
