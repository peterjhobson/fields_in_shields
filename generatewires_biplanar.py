# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 11:41:39 2019
@author: PJ Hobson

Script to design fields in shields.
Design wire patterns on circular bi-planes inside a closed cylindrical magnetic shield to generate
a target field between the planes.
Change the parameters to adjust the coil, shield, and field.
The power regularization is only tuned to order N≤3 harmonics and below.
"""

# %% PREAMBLE ##########################################################################################################

import os
import glob
import time

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from modules import targetpoints, biotsavart, sphericalharmonics, componentfunctions

# CHOOSE YOUR SAVEPATH HERE!!!
savepath = os.path.join(os.getcwd(), 'simulations', 'Bx')
if not os.path.exists(savepath):
    os.makedirs(savepath)
else:
    print('***{} already exists***'.format(savepath))

if not os.path.exists(os.path.join(savepath, 'plots')):
    os.makedirs(os.path.join(savepath, 'plots'))

# %% PARAMETER SELECTION ###############################################################################################

# optiparams
N = 50  # maximum order of Fourier modes
M = 1  # maximum degree of Fourier modes (equal to degree of the harmonic)
k_maximum = 200  # upper bound to the 'k' integral (assumed to be infinity)
p_maximum = 25  # upper bound to the 'p' summation (assumed to be infinity)
beta = 1e-16 * (1e-3 / 1.68e-8)  # weighting of the power optimization term #[1]
# #only tuned to order N≤3 harmonics and below
res = 1.68e-8  # resistivity of the conductor #[Ohm m]
t = 1e-3  # thickness of the conductor #[m]
n_contours = 14  # number of contour levels to generate curve files
n_contours_BS = 100  # number of contour levels for BS law
B_des_ind = sphericalharmonics.b_sum(sphericalharmonics.b1p1(scale=1))  # desired magnetic field harmonic
# it's easiest to decompose field into spherical harmonics and add them together as appropriate

new_containers = True
# this is the slowest bit of the code -- if you're iterating the same N, M, target points and are
# just changing beta / field harmonic then you can keep this off

nomu_calculation = False
# do you want to calculate the integrals in the unshielded case?
# turning this on will require longer computation times

save_contours = False
# do you want to save the plotted contours in Cartesian coordinates?

# geometryparams
rho_ps = np.array((0.45, 0.45)).reshape(2)  # radii of planar coils #m
z_prime_ps = np.array((0.45, -0.45)).reshape(2)  # z position of planar coils #m
n_rho_p = 200  # number of phi points to sample #[1]
n_phi_p = 200  # number of z points to sample #[1]

a = 0.5  # radius of the passive shield #m
L = 1  # length of the passive shield #m

# %% VARIABLE INITIALIZATION ###########################################################################################
t0 = time.time()

# coils
coords_p1_cylindrical = targetpoints.wirepoints_planar([rho_ps[0], z_prime_ps[0]], [n_rho_p, n_phi_p])
coords_p2_cylindrical = targetpoints.wirepoints_planar([rho_ps[1], z_prime_ps[1]], [n_rho_p, n_phi_p])

# target points
rho_min_tp = 1e-6  # inner radius of target points #m
rho_max_tp = 0.01 * rho_ps[0]  # outer radius of target points #m
phi_min_tp = 0
# first angle of target points (rotated anticlockwise) #rads
phi_max_tp = 3 * np.pi / 2
# last angle of target points (rotated anticlockwise) #rads
z_min_tp = 0.55 * z_prime_ps[1]  # lowest axial position of target points #m
z_max_tp = 0.55 * z_prime_ps[0]  # highest axial position of target points #m
n_rho_tp = 2  # number of rho samples  #[1]
n_phi_tp = 4  # number of phi samples #[1]
n_z_tp = 9  # number of z samples #[1]
coords_tp_cylindrical = targetpoints.cylinder_grid_cylindrical(
    [rho_min_tp, rho_max_tp, phi_min_tp, phi_max_tp, z_min_tp, z_max_tp],
    [n_rho_tp, n_phi_tp, n_z_tp])

# x measurement points
l_x_mesx = 2 * max(rho_ps)  # length of x position of measurement points #m
l_y_mesx = 1e-6  # length of y position of measurement points #m
l_z_mesx = 1e-6  # length of z position of measurement points #m
n_x_mesx = 101  # number of x samples  #[1]
n_y_mesx = 1  # number of y samples #[1]
n_z_mesx = 1  # number of z samples #[1]
c0_mesx = np.array((0, 0, 0))
coords_mesx_cylindrical = targetpoints.cubic_grid_cylindrical([l_x_mesx, l_y_mesx, l_z_mesx],
                                                              [n_x_mesx, n_y_mesx, n_z_mesx], c0_mesx)

# z measurement points
l_x_mesz = 1e-6  # length of x position of measurement points #m
l_y_mesz = 1e-6  # length of y position of measurement points #m
l_z_mesz = z_prime_ps[0] - z_prime_ps[1]  # length of z position of measurement points #m
n_x_mesz = 1  # number of x samples  #[1]
n_y_mesz = 1  # number of y samples #[1]
n_z_mesz = 101  # number of z samples #[1]
c0_mesz = np.array((0, 0, (z_prime_ps[0] + z_prime_ps[1]) / 2))
coords_mesz_cylindrical = targetpoints.cubic_grid_cylindrical([l_x_mesz, l_y_mesz, l_z_mesz],
                                                              [n_x_mesz, n_y_mesz, n_z_mesz], c0_mesz)

B_des_tp_cartesian = sphericalharmonics.b_sphericalharmonics(targetpoints.poltocart(coords_tp_cylindrical), B_des_ind)
# Cartesian #Desired magnetic field #(n_points, (x, y, z))
B_des_tp_cylindrical = targetpoints.fieldcarttopol(B_des_tp_cartesian, coords_tp_cylindrical)
# cylindrical #Desired magnetic field #(n_points, (rho, phi, z))

B_des_mesx_cartesian = sphericalharmonics.b_sphericalharmonics(targetpoints.poltocart(coords_mesx_cylindrical),
                                                               B_des_ind)
# Cartesian #Desired magnetic field #(n_points, (x, y, z))
B_des_mesx_cylindrical = targetpoints.fieldcarttopol(B_des_mesx_cartesian, coords_mesx_cylindrical)
# cylindrical #Desired magnetic field #(n_points, (rho, phi, z))

B_des_mesz_cartesian = sphericalharmonics.b_sphericalharmonics(targetpoints.poltocart(coords_mesz_cylindrical),
                                                               B_des_ind)
# Cartesian #Desired magnetic field #(n_points, (x, y, z))
B_des_mesz_cylindrical = targetpoints.fieldcarttopol(B_des_mesz_cartesian, coords_mesz_cylindrical)
# cylindrical #Desired magnetic field #(n_points, (rho, phi, z))

np.savetxt(os.path.join(savepath, 'coil_params_planar_p1.csv'),
           np.array((rho_ps[0], z_prime_ps[0], a, L)), delimiter=",")
np.savetxt(os.path.join(savepath, 'coil_params_planar_p2.csv'),
           np.array((rho_ps[1], z_prime_ps[1], a, L)), delimiter=",")
np.savetxt(os.path.join(savepath, 'opti_params.csv'),
           np.array((N, M, p_maximum, beta, res, t)), delimiter=",")

t1 = time.time()

print('***Variables initialized in {:.3} s***'.format(t1 - t0))

# System set-up
Xcylinder, Zcylinder = np.meshgrid(np.linspace(-a, a, 100), np.linspace(-0.5 * L, 0.5 * L, 100))
Ycylinder = np.sqrt(a ** 2 - Xcylinder ** 2)
rstride = 20
cstride = 10

theta_ps = np.linspace(0, 2 * np.pi, 201)
x_p1 = rho_ps[0] * np.cos(theta_ps)
y_p1 = rho_ps[0] * np.sin(theta_ps)
x_p2 = rho_ps[1] * np.cos(theta_ps)
y_p2 = rho_ps[1] * np.sin(theta_ps)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(Xcylinder, Ycylinder, Zcylinder,
                 alpha=0.2, rstride=rstride, cstride=cstride, color='grey')
ax1.plot_surface(Xcylinder, -Ycylinder, Zcylinder,
                 alpha=0.2, rstride=rstride, cstride=cstride, color='grey')
ax1.plot(x_p1, y_p1, z_prime_ps[0], color='pink', linewidth=5)
ax1.plot(x_p2, y_p2, z_prime_ps[1], color='pink', linewidth=5)
ax1.scatter(targetpoints.poltocart(coords_tp_cylindrical)[:, 0],
            targetpoints.poltocart(coords_tp_cylindrical)[:, 1],
            targetpoints.poltocart(coords_tp_cylindrical)[:, 2], color='red')
ax1.scatter(targetpoints.poltocart(coords_mesx_cylindrical)[:, 0],
            targetpoints.poltocart(coords_mesx_cylindrical)[:, 1],
            targetpoints.poltocart(coords_mesx_cylindrical)[:, 2], color='blue')
ax1.scatter(targetpoints.poltocart(coords_mesz_cylindrical)[:, 0],
            targetpoints.poltocart(coords_mesz_cylindrical)[:, 1],
            targetpoints.poltocart(coords_mesz_cylindrical)[:, 2], color='green')
ax1.quiver(
    targetpoints.poltocart(coords_tp_cylindrical)[:, 0],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 1],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 2],
    B_des_tp_cartesian[:, 0],
    B_des_tp_cartesian[:, 1],
    B_des_tp_cartesian[:, 2],
    color='red', length=0.05)
ax1.set_xlabel('$x$' + r' $\mathrm{ (m)}$')
ax1.set_ylabel('$y$' + r' $\mathrm{ (m)}$')
ax1.set_zlabel('$z$' + r' $\mathrm{ (m)}$')
fig1.tight_layout()
plt.savefig(os.path.join(savepath, 'plots', 'set-up'), dpi=300)
fig1.show()

# %% IN SHIELD #######################################################################################################
t0 = time.time()

if new_containers:
    X0_targetpoints, X_targetpoints, C_targetpoints = componentfunctions.integral_containers_planar(
        coords_tp_cylindrical,
        rho_ps, z_prime_ps,
        [N, M, p_maximum,
         k_maximum, a, L])
    Cpwr_cmp_planar_0 = componentfunctions.power_matrix_planar_0(rho_ps, [N, M, res, t])
    Cpwr_cmp_planar = componentfunctions.power_matrix_planar(rho_ps, [N, M, res, t])

X0 = componentfunctions.fourier_calculation_0(X0_targetpoints, B_des_tp_cylindrical, Cpwr_cmp_planar_0, beta)
X = componentfunctions.fourier_calculation(X_targetpoints, B_des_tp_cylindrical, Cpwr_cmp_planar, beta)
C = componentfunctions.fourier_calculation(C_targetpoints, B_des_tp_cylindrical, Cpwr_cmp_planar, beta)

B_analytic_tp_cylindrical = componentfunctions.b_calculation(coords_tp_cylindrical, X0_targetpoints, X0,
                                                             X_targetpoints, X, C_targetpoints, C)
B_analytic_tp_cartesian = targetpoints.fieldpoltocart(B_analytic_tp_cylindrical, coords_tp_cylindrical)

np.savetxt(os.path.join(savepath, 'Xi0_p1.csv'), X0[:N].reshape(N, 1), delimiter=",")
np.savetxt(os.path.join(savepath, 'Xi0_p2.csv'), X0[N:].reshape(N, 1), delimiter=",")
np.savetxt(os.path.join(savepath, 'Xi_p1.csv'), X[:N * M].reshape(N, M), delimiter=",")
np.savetxt(os.path.join(savepath, 'Xi_p2.csv'), X[N * M:].reshape(N, M), delimiter=",")
np.savetxt(os.path.join(savepath, 'Chi_p1.csv'), C[:N * M].reshape(N, M), delimiter=",")
np.savetxt(os.path.join(savepath, 'Chi_p2.csv'), C[N * M:].reshape(N, M), delimiter=",")

np.savetxt(os.path.join(savepath, 'tp.csv'), targetpoints.poltocart(coords_tp_cylindrical), delimiter=",")
np.savetxt(os.path.join(savepath, 'B_desired_tp.csv'), B_des_tp_cartesian, delimiter=",")
np.savetxt(os.path.join(savepath, 'B_analytic_tp.csv'), B_analytic_tp_cartesian, delimiter=",")

t1 = time.time()

print('***Shield calculation completed in {:.3} s***'.format(t1 - t0))

# %% NO SHIELD #######################################################################################################
t0 = time.time()

if nomu_calculation:

    if new_containers:
        X0_targetpoints_nomu, X_targetpoints_nomu, C_targetpoints_nomu = componentfunctions.integral_containers_planar_nomu(
            coords_tp_cylindrical,
            rho_ps, z_prime_ps,
            [N, M, k_maximum])

    B_nomu_analytic_tp_cylindrical = componentfunctions.b_calculation(coords_tp_cylindrical, X0_targetpoints_nomu, X0,
                                                                      X_targetpoints_nomu, X, C_targetpoints_nomu, C)
    B_nomu_analytic_tp_cartesian = targetpoints.fieldpoltocart(B_nomu_analytic_tp_cylindrical, coords_tp_cylindrical)

    np.savetxt(os.path.join(savepath, 'B_nomu_analytic_tp.csv'), B_nomu_analytic_tp_cartesian, delimiter=",")

t1 = time.time()

print('***No shield calculation completed in {:.3} s***'.format(t1 - t0))

# %% STREAM-FUNCTION ###################################################################################################
t0 = time.time()

sf_p1 = componentfunctions.streamfunction_calculation_planar(coords_p1_cylindrical, X0[:N], X[:N * M], C[:N * M],
                                                             rho_ps[0], [N, M])
sf_p2 = componentfunctions.streamfunction_calculation_planar(coords_p2_cylindrical, X0[N:], X[N * M:], C[N * M:],
                                                             rho_ps[1], [N, M])
sfct_p1 = sf_p1.reshape(n_phi_p, n_rho_p)
sfct_p2 = sf_p2.reshape(n_phi_p, n_rho_p)

sf_max = max(np.amax(sfct_p1), np.amax(sfct_p2))
sf_min = min(np.amin(sfct_p1), np.amin(sfct_p2))
curr = (sf_max - sf_min) / n_contours_BS
levels = sf_min + (np.arange(1, n_contours_BS + 1) - 0.5) * curr

X_p, Y_p = np.meshgrid(np.linspace(0, n_rho_p - 1, n_rho_p), np.linspace(0, n_phi_p - 1, n_phi_p))
cnts_p1 = plt.contour(X_p, Y_p, sfct_p1, levels=levels).allsegs
cnts_p2 = plt.contour(X_p, Y_p, sfct_p2, levels=levels).allsegs
plt.close()

contours_p1_cartesian, contours_p1_cylindrical = targetpoints.poltocartcontour_planar(cnts_p1,
                                                                                      [rho_ps[0], z_prime_ps[0]],
                                                                                      [n_rho_p, n_phi_p])
contours_p2_cartesian, contours_p2_cylindrical = targetpoints.poltocartcontour_planar(cnts_p2,
                                                                                      [rho_ps[1], z_prime_ps[1]],
                                                                                      [n_rho_p, n_phi_p])
# Adding the contours together
allconts_cartesian = contours_p1_cartesian + contours_p2_cartesian
B_BS_tp_cartesian = biotsavart.b_contours(targetpoints.poltocart(coords_tp_cylindrical), allconts_cartesian, curr)

np.savetxt(os.path.join(savepath, 'B_BiotSavart_tp.csv'), B_BS_tp_cartesian, delimiter=",")

t1 = time.time()
print('***Optimum contours calculated in {:.3} s***'.format(t1 - t0))

# 2D plot of the contours on the surface. Adjust n_contours to change the number of contour levels.
levels_mod = sf_min + (np.arange(1, n_contours + 1) - 0.5) * (sf_max - sf_min) / n_contours
cnts_p1_mod = plt.contour(X_p, Y_p, sfct_p1, levels=levels_mod).allsegs
cnts_p2_mod = plt.contour(X_p, Y_p, sfct_p2, levels=levels_mod).allsegs
plt.close()

contours_p1_mod_cartesian, contours_p1_mod_cylindrical = targetpoints.poltocartcontour_planar(cnts_p1_mod,
                                                                                              [rho_ps[0],
                                                                                               z_prime_ps[0]],
                                                                                              [n_rho_p, n_phi_p])
contours_p2_mod_cartesian, contours_p2_mod_cylindrical = targetpoints.poltocartcontour_planar(cnts_p2_mod,
                                                                                              [rho_ps[1],
                                                                                               z_prime_ps[1]],
                                                                                              [n_rho_p, n_phi_p])
l_p1 = biotsavart.linestyle_planar(contours_p1_mod_cartesian)
l_p2 = biotsavart.linestyle_planar(contours_p2_mod_cartesian)

if save_contours:
    if not os.path.exists(os.path.join(savepath, 'contours_levels_{}'.format(n_contours))):
        os.makedirs(os.path.join(savepath, 'contours_levels_{}'.format(n_contours)))

    for f in glob.glob(os.path.join(savepath, 'contours_levels_{}'.format(n_contours), '*')):
        os.remove(f)

    for i in range(len(contours_p1_mod_cartesian)):
        np.savetxt(os.path.join(savepath, 'contours_levels_{}'.format(n_contours), '{}-plane_1.txt'.format(i)),
                   contours_p1_mod_cartesian[i],
                   delimiter=",")

    for i in range(len(contours_p2_mod_cartesian)):
        np.savetxt(os.path.join(savepath, 'contours_levels_{}'.format(n_contours), '{}-plane_2.txt'.format(i)),
                   contours_p2_mod_cartesian[i],
                   delimiter=",")

    print('***Contours saved to {}***'.format(os.path.join(savepath, 'contours_levels_{}'.format(n_contours))))

fig2, ax2 = plt.subplots(1, 1)
im2 = ax2.pcolormesh(targetpoints.poltocart(coords_p1_cylindrical)[:, 0].reshape(n_phi_p, n_rho_p),
                     targetpoints.poltocart(coords_p1_cylindrical)[:, 1].reshape(n_phi_p, n_rho_p),
                     sfct_p1, vmin=sf_min * 1.5, vmax=sf_max * 1.5, cmap=plt.cm.bwr)
for i in range(len(contours_p1_mod_cartesian)):
    ax2.plot(contours_p1_mod_cartesian[i][:, 0],
             contours_p1_mod_cartesian[i][:, 1], color='black', linestyle=l_p1[i])
ax2.set_aspect('equal', 'box')
ax2.set_xlabel(r'$x$' + r' $\mathrm{ (m)}$')
ax2.set_ylabel(r'$y$' + r' $\mathrm{ (m)}$')
ax2.minorticks_on()
fig2.tight_layout()
plt.savefig(os.path.join(savepath, 'plots', 'wiredesigns_plane_1'), dpi=300)
fig2.show()

fig3, ax3 = plt.subplots(1, 1)
im3 = ax3.pcolormesh(targetpoints.poltocart(coords_p2_cylindrical)[:, 0].reshape(n_phi_p, n_rho_p),
                     targetpoints.poltocart(coords_p2_cylindrical)[:, 1].reshape(n_phi_p, n_rho_p),
                     sfct_p2, vmin=sf_min * 1.5, vmax=sf_max * 1.5, cmap=plt.cm.bwr)
for i in range(len(contours_p2_mod_cartesian)):
    ax3.plot(contours_p2_mod_cartesian[i][:, 0],
             contours_p2_mod_cartesian[i][:, 1], color='black', linestyle=l_p2[i])
ax3.set_aspect('equal', 'box')
ax3.set_xlabel(r'$x$' + r' $\mathrm{ (m)}$')
ax3.set_ylabel(r'$y$' + r' $\mathrm{ (m)}$')
ax3.minorticks_on()
fig3.tight_layout()
plt.savefig(os.path.join(savepath, 'plots', 'wiredesigns_plane_2'), dpi=300)
fig3.show()

fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.plot_surface(Xcylinder, Ycylinder, Zcylinder,
                 alpha=0.2, rstride=rstride, cstride=cstride, color='grey')
ax4.plot_surface(Xcylinder, -Ycylinder, Zcylinder,
                 alpha=0.2, rstride=rstride, cstride=cstride, color='grey')
ax4.plot(x_p1, y_p1, z_prime_ps[0], color='pink', linewidth=5)
ax4.plot(x_p2, y_p2, z_prime_ps[1], color='pink', linewidth=5)
ax4.scatter(targetpoints.poltocart(coords_tp_cylindrical)[:, 0],
            targetpoints.poltocart(coords_tp_cylindrical)[:, 1],
            targetpoints.poltocart(coords_tp_cylindrical)[:, 2], color='red')
ax4.quiver(
    targetpoints.poltocart(coords_tp_cylindrical)[:, 0],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 1],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 2],
    B_des_tp_cartesian[:, 0],
    B_des_tp_cartesian[:, 1],
    B_des_tp_cartesian[:, 2],
    color='red', length=0.05)
for i in range(len(contours_p1_mod_cartesian)):
    ax4.plot(contours_p1_mod_cartesian[i][:, 0],
             contours_p1_mod_cartesian[i][:, 1],
             contours_p1_mod_cartesian[i][:, 2], color='black', linestyle=l_p1[i])
for i in range(len(contours_p2_mod_cartesian)):
    ax4.plot(contours_p2_mod_cartesian[i][:, 0],
             contours_p2_mod_cartesian[i][:, 1],
             contours_p2_mod_cartesian[i][:, 2], color='black', linestyle=l_p2[i])
ax4.set_xlabel('$x$' + r' $\mathrm{ (m)}$')
ax4.set_ylabel('$y$' + r' $\mathrm{ (m)}$')
ax4.set_zlabel('$z$' + r' $\mathrm{ (m)}$')
fig4.tight_layout()
plt.savefig(os.path.join(savepath, 'plots', 'contours-3D'), dpi=300)
fig4.show()

# 3D plot of the fields
fig5 = plt.figure()
ax5 = fig5.add_subplot(111, projection='3d')
ax5.quiver(
    targetpoints.poltocart(coords_tp_cylindrical)[:, 0],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 1],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 2],
    B_des_tp_cartesian[:, 0],
    B_des_tp_cartesian[:, 1],
    B_des_tp_cartesian[:, 2],
    length=0.002, color='black')
ax5.quiver(
    targetpoints.poltocart(coords_tp_cylindrical)[:, 0],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 1],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 2],
    B_analytic_tp_cartesian[:, 0],
    B_analytic_tp_cartesian[:, 1],
    B_analytic_tp_cartesian[:, 2],
    length=0.002, color='red')
if nomu_calculation:
    ax5.quiver(
        targetpoints.poltocart(coords_tp_cylindrical)[:, 0],
        targetpoints.poltocart(coords_tp_cylindrical)[:, 1],
        targetpoints.poltocart(coords_tp_cylindrical)[:, 2],
        B_nomu_analytic_tp_cartesian[:, 0],
        B_nomu_analytic_tp_cartesian[:, 1],
        B_nomu_analytic_tp_cartesian[:, 2],
        length=0.002, color='orange')
ax5.quiver(
    targetpoints.poltocart(coords_tp_cylindrical)[:, 0],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 1],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 2],
    B_BS_tp_cartesian[:, 0],
    B_BS_tp_cartesian[:, 1],
    B_BS_tp_cartesian[:, 2],
    length=0.002, color='green')
ax5.scatter(targetpoints.poltocart(coords_tp_cylindrical)[:, 0],
            targetpoints.poltocart(coords_tp_cylindrical)[:, 1],
            targetpoints.poltocart(coords_tp_cylindrical)[:, 2])
ax5.set_xlabel('$x$' + r' $\mathrm{ (m)}$')
ax5.set_ylabel('$y$' + r' $\mathrm{ (m)}$')
ax5.set_zlabel('$z$' + r' $\mathrm{ (m)}$')
fig5.tight_layout()
plt.savefig(os.path.join(savepath, 'plots', 'fields'), dpi=300)
fig5.show()

# %% RECALCULATION x-line ##############################################################################################
t0 = time.time()

if new_containers:
    X0_mesx, X_mesx, C_mesx = componentfunctions.integral_containers_planar(coords_mesx_cylindrical, rho_ps, z_prime_ps,
                                                                            [N, M, p_maximum, k_maximum, a, L])

    if nomu_calculation:
        X0_mesx_nomu, X_mesx_nomu, C_mesx_nomu = componentfunctions.integral_containers_planar_nomu(
            coords_mesx_cylindrical,
            rho_ps, z_prime_ps,
            [N, M, k_maximum])

B_analytic_mesx_cylindrical = componentfunctions.b_calculation(coords_mesx_cylindrical, X0_mesx, X0, X_mesx, X, C_mesx,
                                                               C)
B_analytic_mesx_cartesian = targetpoints.fieldpoltocart(B_analytic_mesx_cylindrical, coords_mesx_cylindrical)
B_BS_mesx_cartesian = (
        biotsavart.b_contours(targetpoints.poltocart(coords_mesx_cylindrical), contours_p1_cartesian, curr) +
        biotsavart.b_contours(targetpoints.poltocart(coords_mesx_cylindrical), contours_p2_cartesian, curr))

np.savetxt(os.path.join(savepath, 'mesx.csv'), targetpoints.poltocart(coords_mesx_cylindrical), delimiter=",")
np.savetxt(os.path.join(savepath, 'B_desired_mesx.csv'), B_des_mesx_cartesian, delimiter=",")
np.savetxt(os.path.join(savepath, 'B_analytic_mesx.csv'), B_analytic_mesx_cartesian, delimiter=",")
np.savetxt(os.path.join(savepath, 'B_BiotSavart_mesx.csv'), B_BS_mesx_cartesian, delimiter=",")

if nomu_calculation:
    B_nomu_analytic_mesx_cylindrical = componentfunctions.b_calculation(coords_mesx_cylindrical, X0_mesx_nomu, X0,
                                                                        X_mesx_nomu, X, C_mesx_nomu, C)
    B_nomu_analytic_mesx_cartesian = targetpoints.fieldpoltocart(B_nomu_analytic_mesx_cylindrical,
                                                                 coords_mesx_cylindrical)
    np.savetxt(os.path.join(savepath, 'B_nomu_analytic_mesx.csv'), B_nomu_analytic_mesx_cartesian, delimiter=",")

t1 = time.time()
print('***Calculation along x-line completed in {:.3} s***'.format(t1 - t0))

# Field along x-axis
B_ind = 2  # select index of field to plot
if B_ind == 0:
    fpx = 'x'
if B_ind == 1:
    fpx = 'y'
if B_ind == 2:
    fpx = 'z'

fig6, ax6 = plt.subplots(1, 1)
ax6.axvline(-rho_max_tp, color='grey', ls=':', zorder=0)
ax6.axvline(rho_max_tp, color='grey', ls=':', zorder=0)
ax6.plot(targetpoints.poltocart(coords_mesx_cylindrical)[:, 0], B_des_mesx_cartesian[:, B_ind], color='black')
ax6.plot(targetpoints.poltocart(coords_mesx_cylindrical)[:, 0], B_analytic_mesx_cartesian[:, B_ind], color='red',
         zorder=20)
ax6.plot(targetpoints.poltocart(coords_mesx_cylindrical)[:, 0], B_BS_mesx_cartesian[:, B_ind], color='green', ls='--',
         zorder=10)
if nomu_calculation:
    ax6.plot(targetpoints.poltocart(coords_mesx_cylindrical)[:, 0], B_nomu_analytic_mesx_cartesian[:, B_ind],
             color='orange')
ax6.set_xlabel('$x$' + r' $\mathrm{ (m)}$')
ax6.set_ylabel(r'$B_{{{}}}$'.format(fpx))
fig6.tight_layout()
plt.savefig(os.path.join(savepath, 'plots', 'x-line'), dpi=300)
fig6.show()

# %% RECALCULATION z-line ##############################################################################################
t0 = time.time()

if new_containers:
    X0_mesz, X_mesz, C_mesz = componentfunctions.integral_containers_planar(coords_mesz_cylindrical, rho_ps, z_prime_ps,
                                                                            [N, M, p_maximum, k_maximum, a, L])

    if nomu_calculation:
        X0_mesz_nomu, X_mesz_nomu, C_mesz_nomu = componentfunctions.integral_containers_planar_nomu(
            coords_mesz_cylindrical,
            rho_ps, z_prime_ps,
            [N, M, k_maximum])

B_analytic_mesz_cylindrical = componentfunctions.b_calculation(coords_mesz_cylindrical, X0_mesz, X0, X_mesz, X, C_mesz,
                                                               C)
B_analytic_mesz_cartesian = targetpoints.fieldpoltocart(B_analytic_mesz_cylindrical, coords_mesz_cylindrical)
B_BS_mesz_cartesian = (
        biotsavart.b_contours(targetpoints.poltocart(coords_mesz_cylindrical), contours_p1_cartesian, curr) +
        biotsavart.b_contours(targetpoints.poltocart(coords_mesz_cylindrical), contours_p2_cartesian, curr))

np.savetxt(os.path.join(savepath, 'mesz.csv'), targetpoints.poltocart(coords_mesz_cylindrical), delimiter=",")
np.savetxt(os.path.join(savepath, 'B_desired_mesz.csv'), B_des_mesz_cartesian, delimiter=",")
np.savetxt(os.path.join(savepath, 'B_analytic_mesz.csv'), B_analytic_mesz_cartesian, delimiter=",")
np.savetxt(os.path.join(savepath, 'B_BiotSavart_mesz.csv'), B_BS_mesz_cartesian, delimiter=",")

if nomu_calculation:
    B_nomu_analytic_mesz_cylindrical = componentfunctions.b_calculation(coords_mesz_cylindrical, X0_mesz_nomu, X0,
                                                                        X_mesz_nomu, X, C_mesz_nomu, C)
    B_nomu_analytic_mesz_cartesian = targetpoints.fieldpoltocart(B_nomu_analytic_mesz_cylindrical,
                                                                 coords_mesz_cylindrical)
    np.savetxt(os.path.join(savepath, 'B_nomu_analytic_mesz.csv'), B_nomu_analytic_mesz_cartesian, delimiter=",")

t1 = time.time()
print('***Calculation along z-line completed in {:.3} s***'.format(t1 - t0))

# Field along x-axis
B_ind = 0  # select index of field to plot
if B_ind == 0:
    fpx = 'x'
if B_ind == 1:
    fpx = 'y'
if B_ind == 2:
    fpx = 'z'

fig7, ax7 = plt.subplots(1, 1)
ax7.axvline(z_min_tp, color='grey', ls=':', zorder=0)
ax7.axvline(z_max_tp, color='grey', ls=':', zorder=0)
ax7.plot(targetpoints.poltocart(coords_mesz_cylindrical)[:, 2], B_des_mesz_cartesian[:, B_ind], color='black')
ax7.plot(targetpoints.poltocart(coords_mesz_cylindrical)[:, 2], B_analytic_mesz_cartesian[:, B_ind], color='red',
         zorder=20)
ax7.plot(targetpoints.poltocart(coords_mesz_cylindrical)[:, 2], B_BS_mesz_cartesian[:, B_ind], color='green', ls='--',
         zorder=10)
if nomu_calculation:
    ax7.plot(targetpoints.poltocart(coords_mesz_cylindrical)[:, 2], B_nomu_analytic_mesz_cartesian[:, B_ind],
             color='orange')
ax7.set_xlabel('$z$' + r' $\mathrm{ (m)}$')
ax7.set_ylabel(r'$B_{{{}}}$'.format(fpx))
ax7.set_xlim(z_min_tp - 0.1, z_max_tp + 0.1)
ax7.set_ylim(0, 2)
fig7.tight_layout()
plt.savefig(os.path.join(savepath, 'plots', 'z-line'), dpi=300)
fig7.show()
