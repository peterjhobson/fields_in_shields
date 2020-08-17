# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 11:41:39 2019
@author: PJ Hobson

Script to design fields in shields. Change the parameters to adjust the coil, shield, and field.
"""

# %% PREAMBLE ##########################################################################################################

import os
import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.join(os.getcwd(), 'modules'))
import targetpoints
import biotsavart
import sphericalharmonics
import componentfunctions

# CHOOSE YOUR SAVEPATH HERE!!!
savepath = os.path.join(os.getcwd(), 'examples', '2Bx + Bz')
if not os.path.exists(savepath):
    os.makedirs(savepath)
else:
    print('***{} already exists***'.format(savepath))

if not os.path.exists(os.path.join(savepath, 'plots')):
    os.makedirs(os.path.join(savepath, 'plots'))

# %% PARAMETER SELECTION ###############################################################################################

# optiparams
N = 100  # maximum order of Fourier modes
M = 1  # maximum degree of Fourier modes (equal to degree of the harmonic)
k_maximum = 50  # upper bound to the 'k' integral (assumed to be infinity)
p_maximum = 20  # upper bound to the 'p' summation (assumed to be infinity)
beta = 1e-16 * (1e-3 / 1.68e-8)  # weighting of the power optimization term #[1]
res = 1.68e-8  # resistivity of the conductor #[Ohm m]
t = 1e-3  # thickness of the conductor #[m]
n_contours = 250
B_des_ind = sphericalharmonics.b_sum(sphericalharmonics.b1p1(scale=2),
                                     sphericalharmonics.b10(scale=1))  # desired magnetic field harmonic
# it's easiest to decompose field into spherical harmonics and add them together as appropriate

new_containers = True
# this is the slowest bit of the code -- if you're iterating the same N, M, target points and are
# just changing beta / field harmonic then you can keep this off

nomu_calculation = True
# do you want to calculate the integrals in the unshielded case?
# turning this on will require longer computation times

# geometryparams
rho_cs = np.array(0.245).reshape(1)  # radii of coils #m
z_prime_c1s = np.array(0.475).reshape(1)  # top z position of coils #m
z_prime_c2s = -z_prime_c1s  # bottom z position of coils #m
n_phi_c = 400  # number of phi points to sample #[1]
n_z_c = 500  # number of z points to sample #[1]
a = 0.25  # radius of the passive shield #m
L = 1  # length of the passive shield #m

# %% VARIABLE INITIALIZATION ###########################################################################################
t0 = time.time()

# coils
coords_c1_cylindrical = targetpoints.wirepoints_cylindrical([rho_cs[0], z_prime_c1s[0], z_prime_c2s[0]],
                                                            [n_phi_c, n_z_c])

# target points
rho_min_tp = 0.01  # inner radius of target points #m
rho_max_tp = 0.5 * rho_cs[0]  # outer radius of target points #m
phi_min_tp = 0
# first angle of target points (rotated anticlockwise) #rads
phi_max_tp = 3 * np.pi / 2
# last angle of target points (rotated anticlockwise) #rads
z_min_tp = 0.5 * z_prime_c2s[0]  # lowest axial position of target points #m
z_max_tp = 0.5 * z_prime_c1s[0]  # highest axial position of target points #m
n_rho_tp = 4  # number of rho samples  #[1]
n_phi_tp = 4  # number of phi samples #[1]
n_z_tp = 5  # number of z samples #[1]
coords_tp_cylindrical = targetpoints.cylinder_grid_cylindrical(
    [rho_min_tp, rho_max_tp, phi_min_tp, phi_max_tp, z_min_tp, z_max_tp],
    [n_rho_tp, n_phi_tp, n_z_tp])

# x measurement points
l_x_mesx = 2 * a  # length of x position of measurement points #m
l_y_mesx = 1e-6  # length of y position of measurement points #m
l_z_mesx = 1e-6  # length of z position of measurement points #m
n_x_mesx = 201  # number of x samples  #[1]
n_y_mesx = 1  # number of y samples #[1]
n_z_mesx = 1  # number of z samples #[1]
c0_mesx = np.array((0, 0, 0))
coords_mesx_cylindrical = targetpoints.cubic_grid_cylindrical([l_x_mesx, l_y_mesx, l_z_mesx],
                                                              [n_x_mesx, n_y_mesx, n_z_mesx], c0_mesx)

# z measurement points
l_x_mesz = 1e-6  # length of x position of measurement points #m
l_y_mesz = 1e-6  # length of y position of measurement points #m
l_z_mesz = L  # length of z position of measurement points #m
n_x_mesz = 1  # number of x samples  #[1]
n_y_mesz = 1  # number of y samples #[1]
n_z_mesz = 201  # number of z samples #[1]
c0_mesz = np.array((0, 0, 0))
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

np.savetxt(os.path.join(savepath, 'coil_params.csv'),
           np.array((rho_cs[0], z_prime_c1s[0], z_prime_c2s[0], a, L)), delimiter=",")
np.savetxt(os.path.join(savepath, 'opti_params.csv'),
           np.array((N, M, p_maximum, beta, res, t)), delimiter=",")

t1 = time.time()

print('***Variables initialized in {:.3} s***'.format(t1 - t0))

# System set-up
Xcylinder, Zcylinder = np.meshgrid(np.linspace(-a, a, 100), np.linspace(-0.5 * L, 0.5 * L, 100))
Ycylinder = np.sqrt(a ** 2 - Xcylinder ** 2)
Xcoil, Zcoil = np.meshgrid(np.linspace(-rho_cs[0], rho_cs[0], 100), np.linspace(z_prime_c1s[0], z_prime_c2s[0], 100))
Ycoil = np.sqrt(rho_cs[0] ** 2 - Xcoil ** 2)
rstride = 20
cstride = 10

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(Xcylinder, Ycylinder, Zcylinder,
                 alpha=0.2, rstride=rstride, cstride=cstride, color='grey')
ax1.plot_surface(Xcylinder, -Ycylinder, Zcylinder,
                 alpha=0.2, rstride=rstride, cstride=cstride, color='grey')
ax1.plot_surface(Xcoil, Ycoil, Zcoil,
                 alpha=0.4, rstride=rstride, cstride=cstride, color='pink')
ax1.plot_surface(Xcoil, -Ycoil, Zcoil,
                 alpha=0.4, rstride=rstride, cstride=cstride, color='pink')
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
    color='red', length=0.02)
ax1.set_xlabel(r'$x$ (m)')
ax1.set_ylabel(r'$y$ (m)')
ax1.set_zlabel(r'$z$ (m)')
fig1.tight_layout()
plt.savefig(os.path.join(savepath, 'plots', 'set-up'), dpi=300)
fig1.show()

# %% IN SHIELD #######################################################################################################
t0 = time.time()

if new_containers:
    W0_targetpoints, W_targetpoints, Q_targetpoints = componentfunctions.integral_containers(coords_tp_cylindrical,
                                                                                             rho_cs, z_prime_c1s,
                                                                                             z_prime_c2s,
                                                                                             [N, M, p_maximum, a, L])
    Cpwr_cmp_0 = componentfunctions.power_matrix_0(rho_cs, z_prime_c1s, z_prime_c2s, [N, M, res, t])
    Cpwr_cmp = componentfunctions.power_matrix(rho_cs, z_prime_c1s, z_prime_c2s, [N, M, res, t])

W0 = componentfunctions.fourier_calculation_0(W0_targetpoints, B_des_tp_cylindrical, Cpwr_cmp_0, beta)
W = componentfunctions.fourier_calculation(W_targetpoints, B_des_tp_cylindrical, Cpwr_cmp, beta)
Q = componentfunctions.fourier_calculation(Q_targetpoints, B_des_tp_cylindrical, Cpwr_cmp, beta)

B_analytic_tp_cylindrical = componentfunctions.b_calculation(coords_tp_cylindrical, W0_targetpoints, W0, W_targetpoints,
                                                             W,
                                                             Q_targetpoints, Q)
B_analytic_tp_cartesian = targetpoints.fieldpoltocart(B_analytic_tp_cylindrical, coords_tp_cylindrical)

np.savetxt(os.path.join(savepath, 'W0.csv'), W0.reshape(N, 1), delimiter=",")
np.savetxt(os.path.join(savepath, 'W.csv'), W.reshape(N, M), delimiter=",")
np.savetxt(os.path.join(savepath, 'Q.csv'), Q.reshape(N, M), delimiter=",")

np.savetxt(os.path.join(savepath, 'tp.csv'), targetpoints.poltocart(coords_tp_cylindrical), delimiter=",")
np.savetxt(os.path.join(savepath, 'B_desired_tp.csv'), B_des_tp_cartesian, delimiter=",")
np.savetxt(os.path.join(savepath, 'B_analytic_tp.csv'), B_analytic_tp_cartesian, delimiter=",")

t1 = time.time()

print('***Shield calculation completed in {:.3} s***'.format(t1 - t0))

# %% NO SHIELD #######################################################################################################
if nomu_calculation:
    t0 = time.time()

    if new_containers:
        W0_targetpoints_nomu, W_targetpoints_nomu, Q_targetpoints_nomu = componentfunctions.integral_containers_nomu(
            coords_tp_cylindrical, rho_cs, z_prime_c1s, z_prime_c2s,
            [N, M, k_maximum])

    B_nomu_analytic_tp_cylindrical = componentfunctions.b_calculation(coords_tp_cylindrical, W0_targetpoints_nomu, W0,
                                                                      W_targetpoints_nomu, W, Q_targetpoints_nomu, Q)
    B_nomu_analytic_tp_cartesian = targetpoints.fieldpoltocart(B_nomu_analytic_tp_cylindrical, coords_tp_cylindrical)

    np.savetxt(os.path.join(savepath, 'B_nomu_analytic_tp.csv'), B_nomu_analytic_tp_cartesian, delimiter=",")

    t1 = time.time()

    print('***No shield calculation completed in {:.3} s***'.format(t1 - t0))

# %% STREAM-FUNCTION ###################################################################################################
t0 = time.time()

# Only implemented for one coil, but pretty simple to do it for more
sf_c1 = componentfunctions.streamfunction_calculation(coords_c1_cylindrical, W0, W, Q, z_prime_c1s[0], z_prime_c2s[0],
                                                      [N, M])
sfct_c1 = np.fliplr(sf_c1.reshape(n_phi_c, n_z_c))
curr_c1 = (sfct_c1.max() - sfct_c1.min()) / n_contours
levels_c1 = sfct_c1.min() + (np.arange(1, n_contours + 1) - 0.5) * curr_c1
X, Y = np.meshgrid(np.linspace(0, n_z_c-1, n_z_c), np.linspace(0, n_phi_c-1, n_phi_c))
cnts_c1 = plt.contour(X, Y, sfct_c1, levels=levels_c1).allsegs
plt.close()

contours_c1_cartesian, contours_c1_cylindrical = targetpoints.poltocartcontour(cnts_c1, [rho_cs[0], z_prime_c1s[0],
                                                                                         z_prime_c2s[0]],
                                                                               [n_phi_c, n_z_c])
B_BS_tp_cartesian = biotsavart.b_contours(targetpoints.poltocart(coords_tp_cylindrical), contours_c1_cartesian, curr_c1)

np.savetxt(os.path.join(savepath, 'B_BiotSavart_tp.csv'), B_BS_tp_cartesian, delimiter=",")

t1 = time.time()
print('***Optimum contours calculated in {:.3} s***'.format(t1 - t0))

# 2D plot of the contours on the surface. Adjust n_cont_c1_mod to change the number of contour levels.
n_phi_mod = 0
n_cont_c1_mod = 16
levels_mod_c1 = sfct_c1.min() + (np.arange(1, n_cont_c1_mod + 1) - 0.5) * (
        sfct_c1.max() - sfct_c1.min()) / n_cont_c1_mod
cnts_mod_c1 = plt.contour(sfct_c1, n_phi_mod, levels=levels_mod_c1).allsegs
plt.close()

contours_mod_c1_cartesian, contours_mod_c1_cylindrical = targetpoints.poltocartcontour(cnts_mod_c1, [rho_cs[0],
                                                                                                     z_prime_c1s[0],
                                                                                                     z_prime_c2s[0]],
                                                                                       [n_phi_c, n_z_c])
l_c1 = biotsavart.linestyle_cylindrical_modified(contours_mod_c1_cylindrical)

fig2, ax2 = plt.subplots(1, 1)
im = ax2.pcolormesh(coords_c1_cylindrical[:, 1].reshape(n_phi_c, n_z_c) / np.pi,
                    coords_c1_cylindrical[:, 2].reshape(n_phi_c, n_z_c) / (2 * z_prime_c1s[0]),
                    np.fliplr(sfct_c1),
                    vmin=np.min(sfct_c1) * 1.5, vmax=np.max(sfct_c1) * 1.5, cmap=plt.cm.bwr)
for i in range(len(contours_mod_c1_cartesian)):
    ax2.plot(contours_mod_c1_cylindrical[i][:, 1] / np.pi,
             contours_mod_c1_cylindrical[i][:, 2] / (2 * z_prime_c1s[0]),
             color='black', linestyle=l_c1[i], linewidth=1)
ax2.set_xticks(np.arange(0, 2.5, step=0.5))  # Set label locations.
ax2.set_yticks(np.arange(-0.4, 0.6, step=0.2))  # Set label locations.
ax2.set_xlim(0, 2)
ax2.set_ylim(-0.5, 0.5)
ax2.set_xlabel(r'${\phi}/\pi$')
ax2.set_ylabel(r'$z/L_c$')
ax2.minorticks_on()
fig2.tight_layout()
plt.savefig(os.path.join(savepath, 'plots', 'wiredesigns'), dpi=300)
fig2.show()

# 3D plot of the fields
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.quiver(
    targetpoints.poltocart(coords_tp_cylindrical)[:, 0],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 1],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 2],
    B_des_tp_cartesian[:, 0],
    B_des_tp_cartesian[:, 1],
    B_des_tp_cartesian[:, 2],
    length=0.02, color='black')
ax3.quiver(
    targetpoints.poltocart(coords_tp_cylindrical)[:, 0],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 1],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 2],
    B_analytic_tp_cartesian[:, 0],
    B_analytic_tp_cartesian[:, 1],
    B_analytic_tp_cartesian[:, 2],
    length=0.02, color='red')
if nomu_calculation:
    ax3.quiver(
        targetpoints.poltocart(coords_tp_cylindrical)[:, 0],
        targetpoints.poltocart(coords_tp_cylindrical)[:, 1],
        targetpoints.poltocart(coords_tp_cylindrical)[:, 2],
        B_nomu_analytic_tp_cartesian[:, 0],
        B_nomu_analytic_tp_cartesian[:, 1],
        B_nomu_analytic_tp_cartesian[:, 2],
        length=0.02, color='orange')
ax3.quiver(
    targetpoints.poltocart(coords_tp_cylindrical)[:, 0],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 1],
    targetpoints.poltocart(coords_tp_cylindrical)[:, 2],
    B_BS_tp_cartesian[:, 0],
    B_BS_tp_cartesian[:, 1],
    B_BS_tp_cartesian[:, 2],
    length=0.02, color='green')
ax3.scatter(targetpoints.poltocart(coords_tp_cylindrical)[:, 0],
            targetpoints.poltocart(coords_tp_cylindrical)[:, 1],
            targetpoints.poltocart(coords_tp_cylindrical)[:, 2])
ax3.set_xlabel(r'$x$ (m)')
ax3.set_ylabel(r'$y$ (m)')
ax3.set_zlabel(r'$z$ (m)')
fig3.tight_layout()
plt.savefig(os.path.join(savepath, 'plots', 'fields'), dpi=300)
fig3.show()

# %% RECALCULATION x-axis ##############################################################################################
t0 = time.time()

if new_containers:
    W0_mesx, W_mesx, Q_mesx = componentfunctions.integral_containers(coords_mesx_cylindrical, rho_cs, z_prime_c1s,
                                                                     z_prime_c2s,
                                                                     [N, M, p_maximum, a, L])
    if nomu_calculation:
        WO_mesx_nomu, W_mesx_nomu, Q_mesx_nomu = componentfunctions.integral_containers_nomu(coords_mesx_cylindrical,
                                                                                             rho_cs,
                                                                                             z_prime_c1s, z_prime_c2s,
                                                                                             [N, M, k_maximum])

B_analytic_mesx_cylindrical = componentfunctions.b_calculation(coords_mesx_cylindrical, W0_mesx, W0, W_mesx, W, Q_mesx,
                                                               Q)
B_analytic_mesx_cartesian = targetpoints.fieldpoltocart(B_analytic_mesx_cylindrical, coords_mesx_cylindrical)
B_BS_mesx_cartesian = biotsavart.b_contours(targetpoints.poltocart(coords_mesx_cylindrical), contours_c1_cartesian,
                                            curr_c1)

np.savetxt(os.path.join(savepath, 'mesx.csv'), targetpoints.poltocart(coords_mesx_cylindrical), delimiter=",")
np.savetxt(os.path.join(savepath, 'B_desired_mesx.csv'), B_des_mesx_cartesian, delimiter=",")
np.savetxt(os.path.join(savepath, 'B_analytic_mesx.csv'), B_analytic_mesx_cartesian, delimiter=",")
np.savetxt(os.path.join(savepath, 'B_BiotSavart_mesx.csv'), B_BS_mesx_cartesian, delimiter=",")

if nomu_calculation:
    B_nomu_analytic_mesx_cylindrical = componentfunctions.b_calculation(coords_mesx_cylindrical, WO_mesx_nomu, W0,
                                                                        W_mesx_nomu, W, Q_mesx_nomu, Q)
    B_nomu_analytic_mesx_cartesian = targetpoints.fieldpoltocart(B_nomu_analytic_mesx_cylindrical,
                                                                 coords_mesx_cylindrical)

    np.savetxt(os.path.join(savepath, 'B_nomu_analytic_mesx.csv'), B_nomu_analytic_mesx_cartesian, delimiter=",")

t1 = time.time()
print('***Calculation along x-axis completed in {:.3} s***'.format(t1 - t0))

# Field along x-axis
B_ind = 0  # select index of field to plot
if B_ind == 0:
    fpx = 'x'
if B_ind == 1:
    fpx = 'y'
if B_ind == 2:
    fpx = 'z'

fig4, ax4 = plt.subplots(1, 1)
ax4.plot(targetpoints.poltocart(coords_mesx_cylindrical)[:, 0], B_des_mesx_cartesian[:, B_ind], color='black')
ax4.plot(targetpoints.poltocart(coords_mesx_cylindrical)[:, 0], B_analytic_mesx_cartesian[:, B_ind], color='red',
         zorder=20)
ax4.plot(targetpoints.poltocart(coords_mesx_cylindrical)[:, 0], B_BS_mesx_cartesian[:, B_ind], color='green', ls='--',
         zorder=10)
if nomu_calculation:
    ax4.plot(targetpoints.poltocart(coords_mesx_cylindrical)[:, 0], B_nomu_analytic_mesx_cartesian[:, B_ind],
             color='orange')
ax4.set_xlabel('$x$' + r' $\mathrm{ (m)}$')
ax4.set_ylabel(r'$B_{{{}}}$'.format(fpx) + r' $\mathrm{ (T)}$')
ax4.set_xlim(-a, a)
fig4.tight_layout()
plt.savefig(os.path.join(savepath, 'plots', 'x-axis'), dpi=300)
fig4.show()

# %% RECALCULATION z-axis ##############################################################################################
t0 = time.time()

if new_containers:
    W0_mesz, W_mesz, Q_mesz = componentfunctions.integral_containers(coords_mesz_cylindrical, rho_cs, z_prime_c1s,
                                                                     z_prime_c2s,
                                                                     [N, M, p_maximum, a, L])
    if nomu_calculation:
        WO_mesz_nomu, W_mesz_nomu, Q_mesz_nomu = componentfunctions.integral_containers_nomu(coords_mesz_cylindrical,
                                                                                             rho_cs,
                                                                                             z_prime_c1s, z_prime_c2s,
                                                                                             [N, M, k_maximum])

B_analytic_mesz_cylindrical = componentfunctions.b_calculation(coords_mesz_cylindrical, W0_mesz, W0, W_mesz, W, Q_mesz,
                                                               Q)
B_analytic_mesz_cartesian = targetpoints.fieldpoltocart(B_analytic_mesz_cylindrical, coords_mesz_cylindrical)
B_BS_mesz_cartesian = biotsavart.b_contours(targetpoints.poltocart(coords_mesz_cylindrical), contours_c1_cartesian,
                                            curr_c1)

np.savetxt(os.path.join(savepath, 'mesz.csv'), targetpoints.poltocart(coords_mesz_cylindrical), delimiter=",")
np.savetxt(os.path.join(savepath, 'B_desired_mesz.csv'), B_des_mesz_cartesian, delimiter=",")
np.savetxt(os.path.join(savepath, 'B_analytic_mesz.csv'), B_analytic_mesz_cartesian, delimiter=",")
np.savetxt(os.path.join(savepath, 'B_BiotSavart_mesz.csv'), B_BS_mesz_cartesian, delimiter=",")

if nomu_calculation:
    B_nomu_analytic_mesz_cylindrical = componentfunctions.b_calculation(coords_mesz_cylindrical, WO_mesz_nomu, W0,
                                                                        W_mesz_nomu, W, Q_mesz_nomu, Q)
    B_nomu_analytic_mesz_cartesian = targetpoints.fieldpoltocart(B_nomu_analytic_mesz_cylindrical,
                                                                 coords_mesz_cylindrical)

    np.savetxt(os.path.join(savepath, 'B_nomu_analytic_mesz.csv'), B_nomu_analytic_mesz_cartesian, delimiter=",")

t1 = time.time()
print('***Calculation along z-axis completed in {:.3} s***'.format(t1 - t0))

# Field along z-axis
B_ind = 2  # select index of field to plot
if B_ind == 0:
    fpz = 'x'
if B_ind == 1:
    fpz = 'y'
if B_ind == 2:
    fpz = 'z'

fig5, ax5 = plt.subplots(1, 1)
ax5.plot(targetpoints.poltocart(coords_mesz_cylindrical)[:, 2], B_des_mesz_cartesian[:, B_ind], color='black')
ax5.plot(targetpoints.poltocart(coords_mesz_cylindrical)[:, 2], B_analytic_mesz_cartesian[:, B_ind], color='red',
         zorder=20)
ax5.plot(targetpoints.poltocart(coords_mesz_cylindrical)[:, 2], B_BS_mesz_cartesian[:, B_ind], color='green', ls='--',
         zorder=10)
if nomu_calculation:
    ax5.plot(targetpoints.poltocart(coords_mesz_cylindrical)[:, 2], B_nomu_analytic_mesz_cartesian[:, B_ind],
             color='orange')
ax5.set_xlabel('$z$' + r' $\mathrm{ (m)}$')
ax5.set_ylabel(r'$B_{{{}}}$'.format(fpz) + r' $\mathrm{ (T)}$')
ax5.set_xlim(-L / 2, L / 2)
fig5.tight_layout()
plt.savefig(os.path.join(savepath, 'plots', 'z-axis'), dpi=300)
fig5.show()
