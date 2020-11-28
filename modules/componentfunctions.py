# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:29:01 2019

@author: peter

Wrapper functions to complete each step of inverse calculation in https://arxiv.org/abs/2006.02981.
"""

from modules import integrands, integrands_planar, powers
import numpy as np
import scipy.integrate as integrate
from scipy import special


def integral_containers(coords_wire_cylindrical, rho_cs, z_prime_c1s, z_prime_c2s, params):
    """Calculate the integrals for each Fourier coefficient at each of the wire points on a cylinder inside a
    cylindrical closed perfect magnetic conductor.

    Keyword arguments:
    coords_wire_cylindrical -- coordinates of the arbitrary points in cylindrical coordinates
    rho_cs -- radii of cylindrical coils
    z_prime_c1s -- top positions of cylindrical coils
    z_prime_c2s -- bottom positions of cylindrical coils
    params -- N -- maximum order of Fourier series
           -- M -- maximum degree of Fourier series
           -- p_lim -- maximum summation of the Dirac comb
           -- a -- radius of shield cylinder
           -- L -- length of shield cylinder
    """

    N, M, p_lim, a, L = params

    n_coils = rho_cs.shape[0]
    Ns = np.arange(1, N + 1)
    Ms = np.arange(1, M + 1)
    Ps = np.arange(1, p_lim + 1)

    W0r_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, 1))  # (n_points, N, 1)
    W0z_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, 1))  # (n_points, N, 1)

    Wr_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Wphi_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Wz_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)

    Qr_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Qphi_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Qz_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)

    for j in range(n_coils):

        for s in range(coords_wire_cylindrical.shape[0]):

            cds = coords_wire_cylindrical[s]

            for i, n in enumerate(Ns):

                W0r_cmp[s][i + j * N][0] = np.nansum(integrands.F_mum(Ps, [cds, n, rho_cs[j],
                                                                           z_prime_c1s[j], z_prime_c2s[j], a, L]))
                W0z_cmp[s][i + j * N][0] = (np.nansum(integrands.D_mum(Ps, [cds, n, rho_cs[j],
                                                                            z_prime_c1s[j], z_prime_c2s[j], a, L])) +
                                            integrands.D_mum_0([n, z_prime_c1s[j], z_prime_c2s[j], L]))

                for k, m in enumerate(Ms):
                    G_int = np.nansum(
                        integrands.G_mum(Ps, [cds, n, m, rho_cs[j], z_prime_c1s[j], z_prime_c2s[j], a, L]))
                    H_int = np.nansum(
                        integrands.H_mum(Ps, [cds, n, m, rho_cs[j], z_prime_c1s[j], z_prime_c2s[j], a, L]))
                    S_int = np.nansum(
                        integrands.S_mum(Ps, [cds, n, m, rho_cs[j], z_prime_c1s[j], z_prime_c2s[j], a, L]))

                    Wr_cmp[s][i + j * N][k] = G_int * np.cos(m * cds[1])
                    Wphi_cmp[s][i + j * N][k] = H_int * np.sin(m * cds[1])
                    Wz_cmp[s][i + j * N][k] = S_int * np.cos(m * cds[1])

                    Qr_cmp[s][i + j * N][k] = G_int * np.sin(m * cds[1])
                    Qphi_cmp[s][i + j * N][k] = -H_int * np.cos(m * cds[1])
                    Qz_cmp[s][i + j * N][k] = S_int * np.sin(m * cds[1])

    W_W0 = np.concatenate((W0r_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N),
                           W0z_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N)), axis=0)

    W_W = np.concatenate((Wr_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                          Wphi_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                          Wz_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M)), axis=0)

    Q_Q = np.concatenate((Qr_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                          Qphi_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                          Qz_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M)), axis=0)

    return W_W0, W_W, Q_Q


def integral_containers_nomu(coords_wire_cylindrical, rho_cs, z_prime1s, z_prime2s, params):
    """Calculate the integrals for each Fourier coefficient at each of the wire points on a cylinder without the
    contribution of the cylindrical closed perfect magnetic conductor.

    Keyword arguments:
    coords_wire_cylindrical -- coordinates of the arbitrary points in cylindrical coordinates
    rho_cs -- radii of cylindrical coils
    z_prime_c1s -- top positions of cylindrical coils
    z_prime_c2s -- bottom positions of cylindrical coils
    params -- N -- maximum order of Fourier series
           -- M -- maximum degree of Fourier series
           -- k_lim -- maximum integral argument
    """

    N, M, k_lim = params

    n_coils = rho_cs.shape[0]
    Ns = np.arange(1, N + 1)
    Ms = np.arange(1, M + 1)

    W0r_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, 1))  # (n_points, N, 1)
    W0z_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, 1))  # (n_points, N, 1)

    Wr_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Wphi_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Wz_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)

    Qr_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Qphi_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Qz_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)

    for j in range(n_coils):

        for s in range(coords_wire_cylindrical.shape[0]):

            cds = coords_wire_cylindrical[s]

            for i, n in enumerate(Ns):

                W0r_cmp[s][i + j * N][0] = integrate.quad(integrands.F_cmp_nomu, 0, k_lim,
                                                          [cds, n, rho_cs[j], z_prime1s[j], z_prime2s[j]])[0]
                W0z_cmp[s][i + j * N][0] = integrate.quad(integrands.D_cmp_nomu, 0, k_lim,
                                                          [cds, n, rho_cs[j], z_prime1s[j], z_prime2s[j]])[0]

                for k, m in enumerate(Ms):
                    G_int = integrate.quad(integrands.G_cmp_nomu, 0, k_lim,
                                           [cds, n, m, rho_cs[j], z_prime1s[j], z_prime2s[j]])[0]

                    H_int = integrate.quad(integrands.H_cmp_nomu, 0, k_lim,
                                           [cds, n, m, rho_cs[j], z_prime1s[j], z_prime2s[j]])[0]

                    S_int = integrate.quad(integrands.S_cmp_nomu, 0, k_lim,
                                           [cds, n, m, rho_cs[j], z_prime1s[j], z_prime2s[j]])[0]

                    Wr_cmp[s][i + j * N][k] = G_int * np.cos(m * cds[1])
                    Wphi_cmp[s][i + j * N][k] = H_int * np.sin(m * cds[1])
                    Wz_cmp[s][i + j * N][k] = S_int * np.cos(m * cds[1])

                    Qr_cmp[s][i + j * N][k] = G_int * np.sin(m * cds[1])
                    Qphi_cmp[s][i + j * N][k] = -H_int * np.cos(m * cds[1])
                    Qz_cmp[s][i + j * N][k] = S_int * np.sin(m * cds[1])

    W_W0 = np.concatenate((W0r_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N),
                           W0z_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N)), axis=0)

    W_W = np.concatenate((Wr_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                          Wphi_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                          Wz_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M)), axis=0)

    Q_Q = np.concatenate((Qr_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                          Qphi_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                          Qz_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M)), axis=0)

    return W_W0, W_W, Q_Q


def integral_containers_planar(coords_wire_cylindrical, rho_ps, z_prime_ps, params):
    """Calculate the integrals for each Fourier coefficient at each of the wire points on a plane inside a
    cylindrical closed perfect magnetic conductor.

    Keyword arguments:
    coords_wire_cylindrical -- coordinates of the arbitrary points in cylindrical coordinates
    rho_ps -- radii of planar coils
    z_prime_ps -- positions of planar coils
    params -- N -- maximum order of Fourier series
           -- M -- maximum degree of Fourier series
           -- p_lim -- maximum summation of the Dirac comb
           -- a -- radius of shield cylinder
           -- L -- length of shield cylinder
    """

    N, M, p_lim, k_lim, a, L = params

    n_coils = rho_ps.shape[0]
    Ns = np.arange(1, N + 1)
    Ms = np.arange(1, M + 1)
    Ps = np.arange(1, p_lim + 1)

    X0r_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, 1))  # (n_points, N)
    X0z_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, 1))  # (n_points, N)

    Xr_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Xphi_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Xz_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)

    Cr_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Cphi_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Cz_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)

    prefs = 2 * np.pi * 1e-7 * rho_ps ** 3

    for s in range(coords_wire_cylindrical.shape[0]):

        cds = coords_wire_cylindrical[s]

        for i, n in enumerate(Ns):

            rho_n0 = special.jn_zeros(0, N)[i]
            jn0 = -special.jv(1, rho_n0)

            for j in range(n_coils):

                I0_r = \
                    integrate.quad(integrands_planar.Br_mum, 0, k_lim,
                                   [cds, n, 0, rho_ps[j], z_prime_ps[j], rho_n0, a, L])[
                        0]
                I0_z = \
                    integrate.quad(integrands_planar.Bz_mum, 0, k_lim,
                                   [cds, n, 0, rho_ps[j], z_prime_ps[j], rho_n0, a, L])[
                        0]

                M0_r = np.nansum(integrands_planar.Rr_mum(Ps, [cds, n, 0, rho_ps[j], z_prime_ps[j], rho_n0, a, L]))
                M0_z = np.nansum(integrands_planar.Rz_mum(Ps, [cds, n, 0, rho_ps[j], z_prime_ps[j], rho_n0, a, L]))

                X0r_cmp[s][i + j * N] = prefs[j] * rho_n0 * jn0 * (I0_r + M0_r)
                X0z_cmp[s][i + j * N] = prefs[j] * rho_n0 * jn0 * (I0_z + M0_z)

                for k, m in enumerate(Ms):
                    rho_nm = special.jn_zeros(m, N)[i]
                    jnm = -special.jv(m + 1, rho_nm)

                    I_r = integrate.quad(integrands_planar.Br_mum, 0, k_lim,
                                         [cds, n, m, rho_ps[j], z_prime_ps[j], rho_nm, a, L])[0]
                    I_phi = integrate.quad(integrands_planar.Bphi_mum, 0, k_lim,
                                           [cds, n, m, rho_ps[j], z_prime_ps[j], rho_nm, a, L])[0]
                    I_z = integrate.quad(integrands_planar.Bz_mum, 0, k_lim,
                                         [cds, n, m, rho_ps[j], z_prime_ps[j], rho_nm, a, L])[0]

                    M_r = np.nansum(integrands_planar.Rr_mum(Ps, [cds, n, m, rho_ps[j], z_prime_ps[j], rho_nm, a, L]))
                    M_phi = np.nansum(
                        integrands_planar.Rphi_mum(Ps, [cds, n, m, rho_ps[j], z_prime_ps[j], rho_nm, a, L]))
                    M_z = np.nansum(integrands_planar.Rz_mum(Ps, [cds, n, m, rho_ps[j], z_prime_ps[j], rho_nm, a, L]))

                    Xr_cmp[s][i + j * N][k] = prefs[j] * rho_nm * jnm * (I_r + M_r) * np.cos(m * cds[1])
                    Xphi_cmp[s][i + j * N][k] = (prefs[j] / cds[0]) * rho_nm * jnm * (I_phi + M_phi) * np.sin(
                        m * cds[1])
                    Xz_cmp[s][i + j * N][k] = prefs[j] * rho_nm * jnm * (I_z + M_z) * np.cos(m * cds[1])

                    Cr_cmp[s][i + j * N][k] = prefs[j] * rho_nm * jnm * (I_r + M_r) * np.sin(m * cds[1])
                    Cphi_cmp[s][i + j * N][k] = -(prefs[j] / cds[0]) * rho_nm * jnm * (I_phi + M_phi) * np.cos(
                        m * cds[1])
                    Cz_cmp[s][i + j * N][k] = prefs[j] * rho_nm * jnm * (I_z + M_z) * np.sin(m * cds[1])

    X_Xi0 = np.concatenate((X0r_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N),
                            X0z_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N)), axis=0)

    X_Xi = np.concatenate((Xr_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                           Xphi_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                           Xz_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M)), axis=0)

    C_Chi = np.concatenate((Cr_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                            Cphi_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                            Cz_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M)), axis=0)

    return X_Xi0, X_Xi, C_Chi


def integral_containers_planar_nomu(coords_wire_cylindrical, rho_ps, z_prime_ps, params):
    """Calculate the integrals for each Fourier coefficient at each of the wire points on a plane without the
    contribution of the cylindrical closed perfect magnetic conductor.

    Keyword arguments:
    coords_wire_cylindrical -- coordinates of the arbitrary points in cylindrical coordinates
    rho_ps -- radii of planar coils
    z_prime_ps -- positions of planar coils
    params -- N -- maximum order of Fourier series
           -- M -- maximum degree of Fourier series
           -- k_lim -- maximum integral argument
    """

    N, M, k_lim = params

    n_coils = rho_ps.shape[0]
    Ns = np.arange(1, N + 1)
    Ms = np.arange(1, M + 1)

    X0r_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, 1))  # (n_points, N)
    X0z_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, 1))  # (n_points, N)

    Xr_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Xphi_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Xz_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)

    Cr_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Cphi_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)
    Cz_cmp = np.zeros((coords_wire_cylindrical.shape[0], n_coils * N, M))  # (n_points, N, M)

    prefs = 2 * np.pi * 1e-7 * rho_ps ** 3

    for s in range(coords_wire_cylindrical.shape[0]):

        cds = coords_wire_cylindrical[s]

        for i, n in enumerate(Ns):

            rho_n0 = special.jn_zeros(0, N)[i]
            jn0 = -special.jv(1, rho_n0)

            for j in range(n_coils):

                I0_r = \
                    integrate.quad(integrands_planar.Br_cmp, 0, k_lim, [cds, n, 0, rho_ps[j], z_prime_ps[j], rho_n0])[0]
                I0_z = \
                    integrate.quad(integrands_planar.Bz_cmp, 0, k_lim, [cds, n, 0, rho_ps[j], z_prime_ps[j], rho_n0])[0]

                X0r_cmp[s][i + j * N] = prefs[j] * rho_n0 * jn0 * I0_r
                X0z_cmp[s][i + j * N] = prefs[j] * rho_n0 * jn0 * I0_z

                for k, m in enumerate(Ms):
                    rho_nm = special.jn_zeros(m, N)[i]
                    jnm = -special.jv(m + 1, rho_nm)

                    I_r = \
                        integrate.quad(integrands_planar.Br_cmp, 0, k_lim,
                                       [cds, n, m, rho_ps[j], z_prime_ps[j], rho_nm])[0]
                    I_phi = \
                        integrate.quad(integrands_planar.Bphi_cmp, 0, k_lim,
                                       [cds, n, m, rho_ps[j], z_prime_ps[j], rho_nm])[
                            0]
                    I_z = \
                        integrate.quad(integrands_planar.Bz_cmp, 0, k_lim,
                                       [cds, n, m, rho_ps[j], z_prime_ps[j], rho_nm])[0]

                    Xr_cmp[s][i + j * N][k] = prefs[j] * rho_nm * jnm * I_r * np.cos(m * cds[1])
                    Xphi_cmp[s][i + j * N][k] = (prefs[j] / cds[0]) * rho_nm * jnm * I_phi * np.sin(m * cds[1])
                    Xz_cmp[s][i + j * N][k] = prefs[j] * rho_nm * jnm * I_z * np.cos(m * cds[1])

                    Cr_cmp[s][i + j * N][k] = prefs[j] * rho_nm * jnm * I_r * np.sin(m * cds[1])
                    Cphi_cmp[s][i + j * N][k] = -(prefs[j] / cds[0]) * rho_nm * jnm * I_phi * np.cos(m * cds[1])
                    Cz_cmp[s][i + j * N][k] = prefs[j] * rho_nm * jnm * I_z * np.sin(m * cds[1])

    X_Xi0 = np.concatenate((X0r_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N),
                            X0z_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N)), axis=0)

    X_Xi = np.concatenate((Xr_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                           Xphi_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                           Xz_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M)), axis=0)

    C_Chi = np.concatenate((Cr_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                            Cphi_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M),
                            Cz_cmp.reshape(coords_wire_cylindrical.shape[0], n_coils * N * M)), axis=0)

    return X_Xi0, X_Xi, C_Chi


def power_matrix(rho_cs, z_prime_c1s, z_prime_c2s, params):
    """Calculate the power matrix to act as a Tikhonov regularization parameter for W/Q calculation.

    Keyword arguments:
    rho_cs -- radii of cylindrical coils
    z_prime_c1s -- top positions of cylindrical coils
    z_prime_c2s -- bottom positions of cylindrical coils
    params -- N -- maximum order of Fourier series
           -- M -- maximum degree of Fourier series
           -- res -- resistance of coil
           -- t -- thickness of coil
    """

    N, M, res, t = params

    n_coils = rho_cs.shape[0]
    Ns = np.arange(1, N + 1)
    Ms = np.arange(1, M + 1)
    PWRs = np.zeros((n_coils, N * M))

    for j in range(n_coils):

        for k, m in enumerate(Ms):
            PWRs[j, k::M] = powers.pnm(rho_cs[j], z_prime_c1s[j], z_prime_c2s[j], Ns, m, [res, t])

    return np.diag(PWRs.flatten())


def power_matrix_0(rho_cs, z_prime_c1s, z_prime_c2s, params):
    """Calculate the power matrix to act as a Tikhonov regularization parameter for W0 calculation.

    Keyword arguments:
    rho_cs -- radii of cylindrical coils
    z_prime_c1s -- top positions of cylindrical coils
    z_prime_c2s -- bottom positions of cylindrical coils
    params -- N -- maximum order of Fourier series
           -- M -- maximum degree of Fourier series
           -- res -- resistance of coil
           -- t -- thickness of coil
    """

    N, M, res, t = params

    n_coils = rho_cs.shape[0]
    PWRs = np.zeros((n_coils, N))

    for j in range(n_coils):
        PWRs[j, :] = powers.pn0(rho_cs[j], z_prime_c1s[j], z_prime_c2s[j], [res, t]) * np.ones(N)

    return np.diag(PWRs.flatten())


def power_matrix_planar(rho_ps, params):
    """Calculate the power matrix to act as a Tikhonov regularization parameter for X/C calculation.

    Keyword arguments:
    rho_ps -- radii of planar coils
    params -- N -- maximum order of Fourier series
           -- M -- maximum degree of Fourier series
           -- res -- resistance of coil
           -- t -- thickness of coil
    """

    N, M, res, t = params

    n_coils = rho_ps.shape[0]

    PWRs = np.zeros((n_coils, N * M))

    for j in range(n_coils):

        if M >= 1:
            PWRs[j, 0::M] = powers.pn1_planar(rho_ps[j], special.jn_zeros(1, N), [res, t])

        if M >= 2:
            PWRs[j, 1::M] = powers.pn2_planar(rho_ps[j], special.jn_zeros(2, N), [res, t])

        if M >= 3:
            PWRs[j, 2::M] = powers.pn3_planar(rho_ps[j], special.jn_zeros(3, N), [res, t])

        if M >= 4:
            for m in range(4, M + 1):

                rho_nm = special.jn_zeros(m, N)

                for i in range(rho_nm.shape[0]):
                    PWRs[j, (m - 1) + i * M] = powers.pnm_planar(rho_ps[j], rho_nm[i], m, [res, t])

    return np.diag(PWRs.flatten())


def power_matrix_planar_0(rho_ps, params):
    """Calculate the power matrix to act as a Tikhonov regularization parameter for X0 calculation.

    Keyword arguments:
    rho_ps -- radii of planar coils
    params -- N -- maximum order of Fourier series
           -- M -- maximum degree of Fourier series
           -- res -- resistance of coil
           -- t -- thickness of coil
    """

    N, M, res, t = params

    n_coils = rho_ps.shape[0]

    PWRs = np.zeros((n_coils, N))

    for j in range(n_coils):
        PWRs[j, :] = powers.pn0_planar(rho_ps[j], special.jn_zeros(0, N), [res, t])

    return np.diag(PWRs.flatten())


def fourier_calculation(int_container, B_desired_targetpoints_cylindrical, pwr_matrix, beta):
    """Use the method of least squares to calculate W/Q or X/C.

    Keyword arguments:
    int_container -- W_targetpoints / Q_targetpoints / X_targetpoints / C_targetpoints
    B_desired_targetpoints_cylindrical -- desired field at target points in cylindrical coordinates
    pwr_matrix -- output from power_matrix() / power_matrix_planar()
    beta -- Tikhonov regularization weighting
    """

    B_desi = np.concatenate((B_desired_targetpoints_cylindrical[:, 0],
                             B_desired_targetpoints_cylindrical[:, 1],
                             B_desired_targetpoints_cylindrical[:, 2]), axis=0)

    A = np.matmul(int_container.transpose(), int_container) + beta * pwr_matrix
    b = np.matmul(int_container.transpose(), B_desi)

    return np.matmul(np.linalg.pinv(A), b)


def fourier_calculation_0(int_container, B_desired_targetpoints_cylindrical, pwr_matrix, beta):
    """Use the method of least squares to calculate W0/X0.

    Keyword arguments:
    int_container -- W0_targetpoints / X0_targetpoints
    B_desired_targetpoints_cylindrical -- desired field at target points in cylindrical coordinates
    pwr_matrix -- output from power_matrix() / power_matrix_planar()
    beta -- Tikhonov regularization weighting
    """

    B_desi = np.concatenate((B_desired_targetpoints_cylindrical[:, 0],
                             B_desired_targetpoints_cylindrical[:, 2]), axis=0)

    A = np.matmul(int_container.transpose(), int_container) + beta * pwr_matrix
    b = np.matmul(int_container.transpose(), B_desi)

    return np.matmul(np.linalg.pinv(A), b)


def b_calculation(coords_cylindrical, W_W0, W0, W_W, W, Q_Q, Q):
    """Use the Fourier coefficients and integral containers to calculate the magnetic field at arbitrary integral
    container evaluation points. W/Q and X/C are interchangeable.

    Keyword arguments:
    coords_wire_cylindrical -- coordinates of the arbitrary points in cylindrical coordinates
    W0_targetpoints / W_targetpoints / Q_targetpoints -- output from integral_containers() or integral_containers_nomu()
    W0 / W / Q -- Fourier coefficients
    """

    Ban_0 = np.matmul(W_W0, W0)
    Ban = (np.matmul(W_W, W) + np.matmul(Q_Q, Q))

    B_anal_cylindrical = np.zeros(coords_cylindrical.shape)

    B_anal_cylindrical[:, 0] = Ban[:coords_cylindrical.shape[0]] + Ban_0[:coords_cylindrical.shape[0]]
    B_anal_cylindrical[:, 1] = Ban[coords_cylindrical.shape[0]:
                                   2 * coords_cylindrical.shape[0]]
    B_anal_cylindrical[:, 2] = Ban[2 * coords_cylindrical.shape[0]:] + Ban_0[coords_cylindrical.shape[0]:]

    return B_anal_cylindrical


def streamfunction_calculation(coords_wire_cylindrical, W0, W, Q, z_prime_c1, z_prime_c2, params):
    """Use the Fourier coefficients and integral containers to calculate the streamfunction on the cylindrical
    surface.

    Keyword arguments:
    coords_wire_cylindrical -- coordinates on cylindrical surface in cylindrical coordinates
    W0 / W / Q -- Fourier coefficients
    z_prime_c1 -- top position of cylindrical coil
    z_prime_c2 -- bottom position of cylindrical coil
    params -- N -- maximum order of Fourier series
           -- M -- maximum degree of Fourier series
    """

    N, M = params

    W0S_cmp = np.zeros((coords_wire_cylindrical.shape[0], N))
    WS_cmp = np.zeros((coords_wire_cylindrical.shape[0], N, M))
    QS_cmp = np.zeros((coords_wire_cylindrical.shape[0], N, M))

    for s in range(coords_wire_cylindrical.shape[0]):

        cds = coords_wire_cylindrical[s]
        lnp = (z_prime_c1 - z_prime_c2) / (np.arange(1, N + 1) * np.pi)

        W0S_cmp[s, :] = +lnp * np.cos(np.arange(1, N + 1) * np.pi * (cds[2] - z_prime_c2) /
                                      (z_prime_c1 - z_prime_c2))

        for k, q in enumerate(np.arange(1, M + 1)):
            WS_cmp[s, :, k] = -lnp * np.cos(q * cds[1]) * np.sin(np.arange(1, N + 1) * np.pi * (cds[2] - z_prime_c2) /
                                                                 (z_prime_c1 - z_prime_c2))
            QS_cmp[s, :, k] = -lnp * np.sin(q * cds[1]) * np.sin(np.arange(1, N + 1) * np.pi * (cds[2] - z_prime_c2) /
                                                                 (z_prime_c1 - z_prime_c2))

    return (np.matmul(W0S_cmp.reshape(coords_wire_cylindrical.shape[0], N), W0) +
            np.matmul(WS_cmp.reshape(coords_wire_cylindrical.shape[0], N * M), W) +
            np.matmul(QS_cmp.reshape(coords_wire_cylindrical.shape[0], N * M), Q))


def streamfunction_calculation_planar(coords_wire_cylindrical, X0, X, C, rho_p, params):
    """Use the Fourier coefficients and integral containers to calculate the streamfunction on the planar
    surface.

    Keyword arguments:
    coords_wire_cylindrical -- coordinates on planar surface in cylindrical coordinates
    X0, X, C -- Fourier coefficients
    rho_p -- radii of planar coil
    params -- N -- maximum order of Fourier series
           -- M -- maximum degree of Fourier series
    """

    N, M = params

    X0S_cmp = np.zeros((coords_wire_cylindrical.shape[0], N))  # (n_points, N)
    XS_cmp = np.zeros((coords_wire_cylindrical.shape[0], N, M))  # (n_points, N, M)
    CS_cmp = np.zeros((coords_wire_cylindrical.shape[0], N, M))  # (n_points, N, M)

    rho_ns0 = special.jn_zeros(0, N)

    for s in range(coords_wire_cylindrical.shape[0]):

        cds = coords_wire_cylindrical[s]
        jns0 = special.jv(0, rho_ns0 * cds[0] / rho_p)
        X0S_cmp[s, :] = rho_p * jns0

        for k, q in enumerate(np.arange(1, M + 1)):
            rho_nsq = special.jn_zeros(q, N)
            jnsq = special.jv(q, rho_nsq * cds[0] / rho_p)
            XS_cmp[s, :, k] = rho_p * jnsq * np.cos(q * cds[1])
            CS_cmp[s, :, k] = rho_p * jnsq * np.sin(q * cds[1])

    return ((np.matmul(X0S_cmp.reshape(coords_wire_cylindrical.shape[0], N), X0) +
             np.matmul(XS_cmp.reshape(coords_wire_cylindrical.shape[0], N * M), X) +
             np.matmul(CS_cmp.reshape(coords_wire_cylindrical.shape[0], N * M), C)))
