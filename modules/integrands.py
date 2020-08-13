# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:29:01 2019

@author: PJ Hobson

Calculate integrands and summation arguments from Eq.(45-68) of https://arxiv.org/abs/2006.02981. Note that the
arguments have been converted from exponential to sinusoid and odd/even limits of integrals have been applied to
speed-up calculation.
"""

import numpy as np
import scipy.special as special

np.seterr(divide='ignore', invalid='ignore')


def C1(k, coords, n, p, rho_c, z_prime_c1, z_prime_c2, L):
    return (S_pk(k, coords[2], n, p, z_prime_c1, z_prime_c2, L) *
            (-4 * np.pi * 1e-7 * rho_c * n * (z_prime_c1 - z_prime_c2) * k) /
            (n ** 2 * np.pi ** 2 - (z_prime_c1 - z_prime_c2) ** 2 * k ** 2))


def C2(k, coords, n, p, rho_c, z_prime_c1, z_prime_c2, L):
    return (C_pk(k, coords[2], n, p, z_prime_c1, z_prime_c2, L) *
            (-4 * 1e-7 * rho_c * (-1) ** abs(p) * (z_prime_c1 - z_prime_c2) ** 2 * k ** 2) /
            (n ** 2 * np.pi ** 2 - (z_prime_c1 - z_prime_c2) ** 2 * k ** 2))


def C3(k, coords, n, p, rho_c, z_prime_c1, z_prime_c2, L):
    return (C_pk(k, coords[2], n, p, z_prime_c1, z_prime_c2, L) *
            (+4 * 1e-7 * rho_c * (-1) ** abs(p) * (z_prime_c1 - z_prime_c2) ** 2 * abs(k)) /
            (coords[0] * (n ** 2 * np.pi ** 2 - (z_prime_c1 - z_prime_c2) ** 2 * k ** 2)))


def C4(k, coords, n, p, rho_c, z_prime_c1, z_prime_c2, L):
    return (C_pk(k, coords[2], n, p, z_prime_c1, z_prime_c2, L) *
            (-4 * np.pi * 1e-7 * rho_c * n * (z_prime_c1 - z_prime_c2) * abs(k)) /
            (n ** 2 * np.pi ** 2 - (z_prime_c1 - z_prime_c2) ** 2 * k ** 2))


def C5(k, coords, n, p, rho_c, z_prime_c1, z_prime_c2, L):
    return (S_pk(k, coords[2], n, p, z_prime_c1, z_prime_c2, L) *
            (+4 * 1e-7 * rho_c * (-1) ** abs(p) * (z_prime_c1 - z_prime_c2) ** 2 * k * abs(k)) /
            (n ** 2 * np.pi ** 2 - (z_prime_c1 - z_prime_c2) ** 2 * k ** 2))


def C01(k, coords, n, rho_c, z_prime_c1, z_prime_c2):
    return (S_0k(k, coords[2], n, z_prime_c1, z_prime_c2) *
            (-4 * np.pi * 1e-7 * rho_c * n * (z_prime_c1 - z_prime_c2) * k) /
            (n ** 2 * np.pi ** 2 - (z_prime_c1 - z_prime_c2) ** 2 * k ** 2))


def C02(k, coords, n, rho_c, z_prime_c1, z_prime_c2):
    return (C_0k(k, coords[2], n, z_prime_c1, z_prime_c2) *
            (-4 * 1e-7 * rho_c * (z_prime_c1 - z_prime_c2) ** 2 * k ** 2) /
            (n ** 2 * np.pi ** 2 - (z_prime_c1 - z_prime_c2) ** 2 * k ** 2))


def C03(k, coords, n, rho_c, z_prime_c1, z_prime_c2):
    return (C_0k(k, coords[2], n, z_prime_c1, z_prime_c2) *
            (+4 * 1e-7 * rho_c * (z_prime_c1 - z_prime_c2) ** 2 * abs(k)) /
            (coords[0] * (n ** 2 * np.pi ** 2 - (z_prime_c1 - z_prime_c2) ** 2 * k ** 2)))


def C04(k, coords, n, rho_c, z_prime_c1, z_prime_c2):
    return (C_0k(k, coords[2], n, z_prime_c1, z_prime_c2) *
            (-4 * np.pi * 1e-7 * rho_c * n * (z_prime_c1 - z_prime_c2) * abs(k)) /
            (n ** 2 * np.pi ** 2 - (z_prime_c1 - z_prime_c2) ** 2 * k ** 2))


def C05(k, coords, n, rho_c, z_prime_c1, z_prime_c2):
    return (S_0k(k, coords[2], n, z_prime_c1, z_prime_c2) *
            (+4 * 1e-7 * rho_c * (z_prime_c1 - z_prime_c2) ** 2 * k * abs(k)) /
            (n ** 2 * np.pi ** 2 - (z_prime_c1 - z_prime_c2) ** 2 * k ** 2))


def S_pk(k, z, n, p, z_prime_c1, z_prime_c2, L):
    return (np.sin(k * (z - p * L - (-1) ** abs(p) * z_prime_c2)) +
            (-1) ** (n + 1) * np.sin(k * (z - p * L - (-1) ** abs(p) * z_prime_c1)))


def C_pk(k, z, n, p, z_prime_c1, z_prime_c2, L):
    return (np.cos(k * (z - p * L - (-1) ** abs(p) * z_prime_c2)) +
            (-1) ** (n + 1) * np.cos(k * (z - p * L - (-1) ** abs(p) * z_prime_c1)))


def S_0k(k, z, n, z_prime_c1, z_prime_c2):
    return np.sin(k * (z - z_prime_c2)) + (-1) ** (n + 1) * np.sin(k * (z - z_prime_c1))


def C_0k(k, z, n, z_prime_c1, z_prime_c2):
    return np.cos(k * (z - z_prime_c2)) + (-1) ** (n + 1) * np.cos(k * (z - z_prime_c1))


def F_cmp(k, args):
    coords, n, p, rho_c, z_prime_c1, z_prime_c2, a, L = args

    return (special.ivp(0, abs(k) * coords[0]) * (special.kvp(0, abs(k) * rho_c) - special.ivp(0, abs(k) * rho_c) *
                                                  special.kv(0, abs(k) * a) / special.iv(0, abs(k) * a)) *
            C1(k, coords, n, p, rho_c, z_prime_c1, z_prime_c2, L))


def G_cmp(k, args):
    coords, n, m, p, rho_c, z_prime_c1, z_prime_c2, a, L = args

    return (special.ivp(m, abs(k) * coords[0]) * (special.kvp(m, abs(k) * rho_c) - special.ivp(m, abs(k) * rho_c) *
                                                  special.kv(m, abs(k) * a) / special.iv(m, abs(k) * a)) *
            C2(k, coords, n, p, rho_c, z_prime_c1, z_prime_c2, L))


def H_cmp(k, args):
    coords, n, m, p, rho_c, z_prime_c1, z_prime_c2, a, L = args

    return (m * special.iv(m, abs(k) * coords[0]) * (special.kvp(m, abs(k) * rho_c) - special.ivp(m, abs(k) * rho_c) *
                                                     special.kv(m, abs(k) * a) / special.iv(m, abs(k) * a)) *
            C3(k, coords, n, p, rho_c, z_prime_c1, z_prime_c2, L))


def D_cmp(k, args):
    coords, n, p, rho_c, z_prime_c1, z_prime_c2, a, L = args

    return (special.iv(0, abs(k) * coords[0]) * (special.kvp(0, abs(k) * rho_c) - special.ivp(0, abs(k) * rho_c) *
                                                 special.kv(0, abs(k) * a) / special.iv(0, abs(k) * a)) *
            C4(k, coords, n, p, rho_c, z_prime_c1, z_prime_c2, L))


def S_cmp(k, args):
    coords, n, m, p, rho_c, z_prime_c1, z_prime_c2, a, L = args

    return (special.iv(m, abs(k) * coords[0]) * (special.kvp(m, abs(k) * rho_c) - special.ivp(m, abs(k) * rho_c) *
                                                 special.kv(m, abs(k) * a) / special.iv(m, abs(k) * a)) *
            C5(k, coords, n, p, rho_c, z_prime_c1, z_prime_c2, L))


def F_cmp_nomu(k, args):
    coords, n, rho_c, z_prime_c1, z_prime_c2 = args

    return (special.ivp(0, abs(k) * coords[0]) * special.kvp(0, abs(k) * rho_c) *
            C01(k, coords, n, rho_c, z_prime_c1, z_prime_c2))


def G_cmp_nomu(k, args):
    coords, n, m, rho_c, z_prime_c1, z_prime_c2 = args

    return (special.ivp(m, abs(k) * coords[0]) * special.kvp(m, abs(k) * rho_c) *
            C02(k, coords, n, rho_c, z_prime_c1, z_prime_c2))


def H_cmp_nomu(k, args):
    coords, n, m, rho_c, z_prime_c1, z_prime_c2 = args

    return (m * special.iv(m, abs(k) * coords[0]) * special.kvp(m, abs(k) * rho_c) *
            C03(k, coords, n, rho_c, z_prime_c1, z_prime_c2))


def D_cmp_nomu(k, args):
    coords, n, rho_c, z_prime_c1, z_prime_c2 = args

    return (special.iv(0, abs(k) * coords[0]) * special.kvp(0, abs(k) * rho_c) *
            C04(k, coords, n, rho_c, z_prime_c1, z_prime_c2))


def S_cmp_nomu(k, args):
    coords, n, m, rho_c, z_prime_c1, z_prime_c2 = args

    return (special.iv(m, abs(k) * coords[0]) * special.kvp(m, abs(k) * rho_c) *
            C05(k, coords, n, rho_c, z_prime_c1, z_prime_c2))


def C1_DC(p, coords, n, rho_c, z_prime_c1, z_prime_c2, L):
    s = (S_pp(p, coords[2], n, z_prime_c1, z_prime_c2, L) *
         (-4 * np.pi * 1e-7 * rho_c * n * (z_prime_c1 - z_prime_c2)) * p /
         (n ** 2 * L ** 2 - (z_prime_c1 - z_prime_c2) ** 2 * p ** 2))

    return np.nan_to_num(s, nan=0, posinf=0, neginf=0)


def C2_DC(p, coords, n, rho_c, z_prime_c1, z_prime_c2, L):
    s = (C_mp(p, coords[2], n, z_prime_c1, z_prime_c2, L) *
         (4 * np.pi * 1e-7 * rho_c * (z_prime_c1 - z_prime_c2) ** 2 * p ** 2) /
         (L * (n ** 2 * L ** 2 - (z_prime_c1 - z_prime_c2) ** 2 * p ** 2)))

    return np.nan_to_num(s, nan=0, posinf=0, neginf=0)


def C3_DC(p, coords, n, rho_c, z_prime_c1, z_prime_c2, L):
    s = (C_mp(p, coords[2], n, z_prime_c1, z_prime_c2, L) *
         (-4 * 1e-7 * rho_c * (z_prime_c1 - z_prime_c2) ** 2 * abs(p)) /
         (coords[0] * (n ** 2 * L ** 2 - (z_prime_c1 - z_prime_c2) ** 2 * p ** 2)))

    return np.nan_to_num(s, nan=0, posinf=0, neginf=0)


def C4_DC(p, coords, n, rho_c, z_prime_c1, z_prime_c2, L):
    s = (C_pp(p, coords[2], n, z_prime_c1, z_prime_c2, L) *
         (-4 * np.pi * 1e-7 * rho_c * n * (z_prime_c1 - z_prime_c2) * abs(p)) /
         (n ** 2 * L ** 2 - (z_prime_c1 - z_prime_c2) ** 2 * p ** 2))

    return np.nan_to_num(s, nan=0, posinf=0, neginf=0)


def C5_DC(p, coords, n, rho_c, z_prime_c1, z_prime_c2, L):
    s = (S_mp(p, coords[2], n, z_prime_c1, z_prime_c2, L) *
         (-4 * np.pi * 1e-7 * rho_c * (z_prime_c1 - z_prime_c2) ** 2 * p * abs(p)) /
         (L * (n ** 2 * L ** 2 - (z_prime_c1 - z_prime_c2) ** 2 * p ** 2)))

    return np.nan_to_num(s, nan=0, posinf=0, neginf=0)


def S_pp(p, z, n, z_prime_c1, z_prime_c2, L):
    ppl = np.pi * p / L

    return ((-1) ** p * (np.sin(ppl * (z + z_prime_c2)) + (-1) ** (n + 1) * np.sin(ppl * (z + z_prime_c1))) +
            (np.sin(ppl * (z - z_prime_c2)) + (-1) ** (n + 1) * np.sin(ppl * (z - z_prime_c1))))


def S_mp(p, z, n, z_prime_c1, z_prime_c2, L):
    ppl = np.pi * p / L

    return ((-1) ** p * (np.sin(ppl * (z + z_prime_c2)) + (-1) ** (n + 1) * np.sin(ppl * (z + z_prime_c1))) -
            (np.sin(ppl * (z - z_prime_c2)) + (-1) ** (n + 1) * np.sin(ppl * (z - z_prime_c1))))


def C_pp(p, z, n, z_prime_c1, z_prime_c2, L):
    ppl = np.pi * p / L

    return ((-1) ** p * (np.cos(ppl * (z + z_prime_c2)) + (-1) ** (n + 1) * np.cos(ppl * (z + z_prime_c1))) +
            (np.cos(ppl * (z - z_prime_c2)) + (-1) ** (n + 1) * np.cos(ppl * (z - z_prime_c1))))


def C_mp(p, z, n, z_prime_c1, z_prime_c2, L):
    ppl = np.pi * p / L

    return ((-1) ** p * (np.cos(ppl * (z + z_prime_c2)) + (-1) ** (n + 1) * np.cos(ppl * (z + z_prime_c1))) -
            (np.cos(ppl * (z - z_prime_c2)) + (-1) ** (n + 1) * np.cos(ppl * (z - z_prime_c1))))


def F_mum(p, args):
    coords, n, rho_c, z_prime_c1, z_prime_c2, a, L = args

    ppl = np.pi * p / L

    return (special.ivp(0, abs(ppl) * coords[0]) * (special.kvp(0, abs(ppl) * rho_c) -
                                                    special.ivp(0, abs(ppl) * rho_c) * special.kv(0, abs(
                ppl) * a) / special.iv(0, abs(ppl) * a)) *
            C1_DC(p, coords, n, rho_c, z_prime_c1, z_prime_c2, L))


def G_mum(p, args):
    coords, n, m, rho_c, z_prime_c1, z_prime_c2, a, L = args

    ppl = np.pi * p / L

    return (special.ivp(m, abs(ppl) * coords[0]) * (special.kvp(m, abs(ppl) * rho_c) -
                                                    special.ivp(m, abs(ppl) * rho_c) * special.kv(m, abs(
                ppl) * a) / special.iv(m, abs(ppl) * a)) *
            C2_DC(p, coords, n, rho_c, z_prime_c1, z_prime_c2, L))


def H_mum(p, args):
    coords, n, m, rho_c, z_prime_c1, z_prime_c2, a, L = args

    ppl = np.pi * p / L

    return (m * special.iv(m, abs(ppl) * coords[0]) * (special.kvp(m, abs(ppl) * rho_c) -
                                                       special.ivp(m, abs(ppl) * rho_c) * special.kv(m, abs(
                ppl) * a) / special.iv(m, abs(ppl) * a)) *
            C3_DC(p, coords, n, rho_c, z_prime_c1, z_prime_c2, L))


def D_mum(p, args):
    coords, n, rho_c, z_prime_c1, z_prime_c2, a, L = args

    ppl = np.pi * p / L

    return (special.iv(0, abs(ppl) * coords[0]) * (special.kvp(0, abs(ppl) * rho_c) -
                                                   special.ivp(0, abs(ppl) * rho_c) * special.kv(0, abs(
                ppl) * a) / special.iv(0, abs(ppl) * a)) *
            C4_DC(p, coords, n, rho_c, z_prime_c1, z_prime_c2, L))


def D_mum_0(args):
    n, z_prime_c1, z_prime_c2, L = args

    return 4 * 1e-7 * (1 + (-1) ** (n + 1)) * (z_prime_c1 - z_prime_c2) / (n * L)


def S_mum(p, args):
    coords, n, m, rho_c, z_prime_c1, z_prime_c2, a, L = args

    ppl = np.pi * p / L

    return (special.iv(m, abs(ppl) * coords[0]) * (special.kvp(m, abs(ppl) * rho_c) -
                                                   special.ivp(m, abs(ppl) * rho_c) * special.kv(m, abs(
                ppl) * a) / special.iv(m, abs(ppl) * a)) *
            C5_DC(p, coords, n, rho_c, z_prime_c1, z_prime_c2, L))
