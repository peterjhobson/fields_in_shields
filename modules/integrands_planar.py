# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:29:01 2019

@author: PJ Hobson

Calculate integrands and summation arguments for planar coil calculation. Note that the
arguments have been converted from exponential to sinusoid and odd/even limits of integrals have been applied to
speed-up calculation.
"""

import numpy as np
import scipy.special as special

np.seterr(divide='ignore', invalid='ignore')


def Br_cmp(k, args):
    coords, n, m, rho_c, z_prime, rho_nm = args

    return (-k ** 2 * np.sign(coords[2] - z_prime) * np.exp(-k * abs(coords[2] - z_prime)) *
            special.jvp(m, k * coords[0]) * special.jv(m, k * rho_c) / (k ** 2 * rho_c ** 2 - rho_nm ** 2))


def Bphi_cmp(k, args):
    coords, n, m, rho_c, z_prime, rho_nm = args

    return (+m * k * np.sign(coords[2] - z_prime) * np.exp(-k * abs(coords[2] - z_prime)) *
            special.jv(m, k * coords[0]) * special.jv(m, k * rho_c) / (k ** 2 * rho_c ** 2 - rho_nm ** 2))


def Bz_cmp(k, args):
    coords, n, m, rho_c, z_prime, rho_nm = args

    return (+k ** 2 * np.exp(-k * abs(coords[2] - z_prime)) *
            special.jv(m, k * coords[0]) * special.jv(m, k * rho_c) / (k ** 2 * rho_c ** 2 - rho_nm ** 2))


def Rr_cmp(k, args):
    coords, n, m, rho_c, z_prime, rho_nm, a, L = args

    return ((-2 / np.pi) * (k * abs(k) / (abs(k) ** 2 * rho_c ** 2 + rho_nm ** 2)) * np.sin(k * (coords[2] - z_prime)) *
            special.ivp(m, abs(k) * coords[0]) * special.iv(m, abs(k) * rho_c) *
            special.kv(m, abs(k) * a) / special.iv(m, abs(k) * a))


def Rphi_cmp(k, args):
    coords, n, m, rho_c, z_prime, rho_nm, a, L = args

    return ((+2 / np.pi) * (m * k / (abs(k) ** 2 * rho_c ** 2 + rho_nm ** 2)) * np.sin(k * (coords[2] - z_prime)) *
            special.iv(m, abs(k) * coords[0]) * special.iv(m, abs(k) * rho_c) *
            special.kv(m, abs(k) * a) / special.iv(m, abs(k) * a))


def Rz_cmp(k, args):
    coords, n, m, rho_c, z_prime, rho_nm, a, L = args

    return ((-2 / np.pi) * (k ** 2 / (abs(k) ** 2 * rho_c ** 2 + rho_nm ** 2)) * np.cos(k * (coords[2] - z_prime)) *
            special.iv(m, abs(k) * coords[0]) * special.iv(m, abs(k) * rho_c) *
            special.kv(m, abs(k) * a) / special.iv(m, abs(k) * a))


def sigma_k(k, z, z_prime, L):
    return (np.sign(z - z_prime) * np.exp(-k * np.abs(z - z_prime)) -
            (2 / (np.exp(2 * k * L) - 1)) * (
                    np.exp(k * L) * np.sinh(k * (z + z_prime)) + np.sinh(k * (z - z_prime))))


def gamma_k(k, z, z_prime, L):
    return (np.exp(-k * abs(z - z_prime)) +
            (2 / (np.exp(2 * k * L) - 1)) * (np.exp(k * L) * np.cosh(k * (z + z_prime)) + np.cosh(k * (z - z_prime))))


def Br_mum(k, args):
    coords, n, m, rho_c, z_prime, rho_nm, a, L = args

    return (-k ** 2 * sigma_k(k, coords[2], z_prime, L) *
            special.jvp(m, k * coords[0]) * special.jv(m, k * rho_c) / (k ** 2 * rho_c ** 2 - rho_nm ** 2))


def Bphi_mum(k, args):
    coords, n, m, rho_c, z_prime, rho_nm, a, L = args

    return (+m * k * sigma_k(k, coords[2], z_prime, L) *
            special.jv(m, k * coords[0]) * special.jv(m, k * rho_c) / (k ** 2 * rho_c ** 2 - rho_nm ** 2))


def Bz_mum(k, args):
    coords, n, m, rho_c, z_prime, rho_nm, a, L = args

    return (+k ** 2 * gamma_k(k, coords[2], z_prime, L) *
            special.jv(m, k * coords[0]) * special.jv(m, k * rho_c) / (k ** 2 * rho_c ** 2 - rho_nm ** 2))


def S_p(p, z, z_prime, L):
    return (-1) ** p * np.sin(np.pi * p * (z + z_prime) / L) + np.sin(np.pi * p * (z - z_prime) / L)


def C_p(p, z, z_prime, L):
    return (-1) ** p * np.cos(np.pi * p * (z + z_prime) / L) + np.cos(np.pi * p * (z - z_prime) / L)


def Rr_mum(p, args):
    coords, n, m, rho_c, z_prime, rho_nm, a, L = args

    ppl = np.pi * p / L

    return (((-2 / L) * ppl * abs(ppl) / (abs(ppl) ** 2 * rho_c ** 2 + rho_nm ** 2)) * S_p(p, coords[2], z_prime, L) *
            special.ivp(m, abs(ppl) * coords[0]) * special.iv(m, abs(ppl) * rho_c) *
            special.kv(m, abs(ppl) * a) / special.iv(m, abs(ppl) * a))


def Rphi_mum(p, args):
    coords, n, m, rho_c, z_prime, rho_nm, a, L = args

    ppl = np.pi * p / L

    return (((2 / L) * m * ppl / (abs(ppl) ** 2 * rho_c ** 2 + rho_nm ** 2)) * S_p(p, coords[2], z_prime, L) *
            special.iv(m, abs(ppl) * coords[0]) * special.iv(m, abs(ppl) * rho_c) *
            special.kv(m, abs(ppl) * a) / special.iv(m, abs(ppl) * a))


def Rz_mum(p, args):
    coords, n, m, rho_c, z_prime, rho_nm, a, L = args

    ppl = np.pi * p / L

    return (((-2 / L) * ppl ** 2 / (abs(ppl) ** 2 * rho_c ** 2 + rho_nm ** 2)) * C_p(p, coords[2], z_prime, L) *
            special.iv(m, abs(ppl) * coords[0]) * special.iv(m, abs(ppl) * rho_c) *
            special.kv(m, abs(ppl) * a) / special.iv(m, abs(ppl) * a))


def z_p(p_array, z_prime, L):
    return (-np.ones(p_array.shape[0])) ** p_array * z_prime - p_array * L
