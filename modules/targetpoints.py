# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:20:33 2019

@author: PJ Hobson

Generate target points and perform coordinate transformations.
"""

import numpy as np


def cubic_grid_cylindrical(dim, ns, coords_center=np.zeros(3), tol=1e-9):
    """Create a cubic grid of points with no zero points to prevent errors. Returns in cylindrical coordinates.

    Keyword arguments:
    dim -- lx -- length of grid in x
        -- ly -- length of grid in y
        -- lz -- length of grid in z
    ns -- nx -- number of points in x
       -- ny -- number of points in y
       -- nz -- number of points in z
    coords_center_cylindrical -- center point in Cartesian coordinates
    tol -- tolerance for shifting the coordinates to prevent zeros
    """

    lx, ly, lz = dim
    nx, ny, nz = ns

    if lx < tol:
        lx = tol

    if ly < tol:
        ly = tol

    if lz < tol:
        lz = tol

    nx = int(nx)
    ny = int(ny)
    nz = int(nz)

    xc, yc, zc = np.meshgrid(
        (np.linspace(0., lx, nx) +
         (coords_center[0] - 0.5 * lx) * np.ones(nx)),
        (np.linspace(0., ly, ny) +
         (coords_center[1] - 0.5 * ly) * np.ones(ny)),
        (np.linspace(0., lz, nz) +
         (coords_center[2] - 0.5 * lz) * np.ones(nz)))

    return np.stack((np.sqrt(xc ** 2 + yc ** 2).flatten(),
                     np.arctan2(yc, xc).flatten(),
                     zc.flatten()), axis=1)


def cylinder_grid_cylindrical(dim, ns, coords_center_cylindrical=np.zeros(3), tol=1e-9):
    """Create a cylindrical grid of points with no zero points to prevent errors. Returns in cylindrical coordinates.

    Keyword arguments:
    dim -- minrho -- minimum radial position of cylinder
        -- maxrho -- maximum radial position of cylinder
        -- minphi -- minimum azimuthal angle of cylinder
        -- maxphi -- maximum azimuthal angle of cylinder
        -- minz -- minimum radial position of cylinder
        -- maxz -- maximum radial position of cylinder
    ns -- nrho -- number of points radially
       -- nphi -- number of points azimuthally
       -- nz -- number of points axially
    coords_center_cylindrical -- center point in cylindrical coordinates
    tol -- tolerance for shifting the coordinates to prevent zeros
    """

    minrho, maxrho, minphi, maxphi, minz, maxz = dim
    nrho, nphi, nz = ns

    nrho = int(nrho)
    nphi = int(nphi)
    nz = int(nz)

    if minrho < tol:
        minrho = tol

    rhoc, phic, zc = np.meshgrid(
        np.linspace(minrho, maxrho, nrho) + coords_center_cylindrical[0] * np.ones(nrho),
        np.linspace(minphi, maxphi, nphi) + coords_center_cylindrical[1] * np.ones(nphi),
        np.linspace(minz, maxz, nz) + coords_center_cylindrical[2] * np.ones(nz))

    return np.stack((rhoc.flatten(), phic.flatten(), zc.flatten()), axis=1)


def wirepoints_cylindrical(dim, ns):
    """Create an open cylinder of points on a coil surface. Returns in cylindrical coordinates.

    Keyword arguments:
    dim -- rho_c -- radius of cylindrical coil
        -- z_prime_c1 -- top position of cylindrical coil
        -- z_prime_c2 -- bottom position of cylindrical coil
    ns -- nphi -- number of points azimuthally
       -- nz -- number of points axially
    """

    rho_c, z_prime_c1, z_prime_c2 = dim
    nphi, nz = ns

    nphi = int(nphi)
    nz = int(nz)

    rhoc, phic, zc = np.meshgrid(rho_c, np.linspace(0, 2 * np.pi, nphi),
                                 np.linspace(z_prime_c1, z_prime_c2, nz))

    return np.stack((rhoc.flatten(), phic.flatten(), zc.flatten()), axis=1)


def wirepoints_planar(dim, ns):
    """Create a plane of points on a coil surface. Returns in cylindrical coordinates.

    Keyword arguments:
    dim -- rho_p -- radius of planar coil
        -- z_prime_p -- position of planar coil
    ns -- nrho -- number of points radially
       -- nphi -- number of points azimuthally
    """

    rho_p, z_prime_p = dim
    nrho, nphi = ns

    nrho = int(nrho)
    nphi = int(nphi)

    rhoc, phic, zc = np.meshgrid(np.linspace(0, rho_p, nrho),
                                 np.linspace(0, 2 * np.pi, nphi),
                                 z_prime_p)

    return np.stack((rhoc.flatten(), phic.flatten(), zc.flatten()), axis=1)


def carttopol(input_vector):
    """Convert vector from Cartesian to cylindrical coordinates.

    Keyword arguments:
    input_vector -- Cartesian coordinates
    """
    output_vector = np.zeros(np.shape(input_vector))

    output_vector[:, 0] = np.sqrt(input_vector[:, 0] ** 2 + input_vector[:, 1] ** 2)
    output_vector[:, 1] = np.arctan2(input_vector[:, 1], input_vector[:, 0])
    output_vector[:, 2] = input_vector[:, 2]

    return output_vector


def poltocart(input_vector):
    """Convert vector from cylindrical to Cartesian coordinates.

    Keyword arguments:
    input_vector -- cylindrical coordinates
    """
    output_vector = np.zeros(np.shape(input_vector))

    output_vector[:, 0] = input_vector[:, 0] * np.cos(input_vector[:, 1])
    output_vector[:, 1] = input_vector[:, 0] * np.sin(input_vector[:, 1])
    output_vector[:, 2] = input_vector[:, 2]

    return output_vector


def fieldcarttopol(input_vector, input_coords_cylindrical):
    """Convert magnetic field from Cartesian to cylindrical coordinates.

    Keyword arguments:
    input_vector -- field in Cartesian coordinates
    input_coords_cylindrical -- input coordinates in cylindrical coordinates
    """

    output_vector = np.zeros(np.shape(input_vector))

    output_vector[:, 0] = (input_vector[:, 0] * np.cos(input_coords_cylindrical[:, 1]) +
                           input_vector[:, 1] * np.sin(input_coords_cylindrical[:, 1]))
    output_vector[:, 1] = (- input_vector[:, 0] * np.sin(input_coords_cylindrical[:, 1]) +
                           input_vector[:, 1] * np.cos(input_coords_cylindrical[:, 1]))
    output_vector[:, 2] = input_vector[:, 2]

    return output_vector


def fieldpoltocart(input_vector, input_coords_cylindrical):
    """Convert magnetic field from cylindrical to Cartesian coordinates.

    Keyword arguments:
    input_vector -- field in cylindrical coordinates
    input_coords_cylindrical -- input coordinates in cylindrical coordinates
    """

    output_vector = np.zeros(np.shape(input_vector))

    output_vector[:, 0] = (input_vector[:, 0] * np.cos(input_coords_cylindrical[:, 1]) -
                           input_vector[:, 1] * np.sin(input_coords_cylindrical[:, 1]))
    output_vector[:, 1] = (input_vector[:, 0] * np.sin(input_coords_cylindrical[:, 1]) +
                           input_vector[:, 1] * np.cos(input_coords_cylindrical[:, 1]))
    output_vector[:, 2] = input_vector[:, 2]

    return output_vector


def poltocartcontour(contour_list, dim, ns):
    """Convert a contour output from matplotlib.pyplot.contour() into a list of contour arrays on a cylindrical coil surface.
    Returns in Cartesian and cylindrical coordinates.

    Keyword arguments:
    contour_list -- list of contours in cylindrical coordinates
    dim -- rho_c -- radius of cylindrical coil
        -- z_prime_c1 -- top position of cylindrical coil
        -- z_prime_c2 -- bottom position of cylindrical coil
    ns -- nphi -- number of points azimuthally
       -- nz -- number of points axially
    """

    rho_c, z_prime_c1, z_prime_c2 = dim
    nphi, nz = ns

    contours = []
    contours_cylindrical = []

    for i in range(len(contour_list)):

        for j in range(len(contour_list[i])):

            if len(contour_list[i][j]) > 3:
                # length 3 would be start - finish - start
                # which will cancel necessarily

                c_phi = (contour_list[i][j][:, 1] * 2 * np.pi / (nphi - 1))
                c_z = (contour_list[i][j][:, 0] * (z_prime_c1 - z_prime_c2) / (nz - 1) + z_prime_c2)

                output_vector = np.zeros((c_z.shape[0], 3))
                output_vector[:, 0] = (rho_c * np.cos(c_phi))
                output_vector[:, 1] = (rho_c * np.sin(c_phi))
                output_vector[:, 2] = c_z
                contours.append(output_vector)

                output_vector_cylindrical = np.zeros((c_z.shape[0], 3))
                output_vector_cylindrical[:, 0] = np.ones(c_z.shape[0]) * rho_c
                output_vector_cylindrical[:, 1] = c_phi
                output_vector_cylindrical[:, 2] = c_z
                contours_cylindrical.append(output_vector_cylindrical)

    return contours, contours_cylindrical


def poltocartcontour_planar(contour_list, dim, ns):
    """Convert a contour output from matplotlib.pyplot.contour() into a list of contour arrays on a planar coil surface.
    Returns in Cartesian and cylindrical coordinates.

    Keyword arguments:
    contour_list -- list of contours in cylindrical coordinates
    dim -- rho_p -- radius of planar coil
        -- z_prime_p -- position of planar coil
    ns -- nrho -- number of points radially
       -- nphi -- number of points azimuthally
    """

    rho_p, z_prime_p = dim
    nrho, nphi = ns

    contours = []
    contours_cylindrical = []

    for i in range(len(contour_list)):

        for j in range(len(contour_list[i])):

            if len(contour_list[i][j]) > 3:
                # length 3 would be start - finish - start
                # which will cancel necessarily

                c_rho = (contour_list[i][j][:, 0] * rho_p / (nrho - 1))
                c_phi = (contour_list[i][j][:, 1] * 2 * np.pi / (nphi - 1))

                output_vector = np.zeros((c_rho.shape[0], 3))
                output_vector[:, 0] = (c_rho * np.cos(c_phi))
                output_vector[:, 1] = (c_rho * np.sin(c_phi))
                output_vector[:, 2] = (np.ones(c_rho.shape[0]) * z_prime_p)
                contours.append(output_vector)

                output_vector_cylindrical = np.zeros((c_rho.shape[0], 3))
                output_vector_cylindrical[:, 0] = c_rho
                output_vector_cylindrical[:, 1] = c_phi
                output_vector_cylindrical[:, 2] = np.ones(c_rho.shape[0]) * z_prime_p
                contours_cylindrical.append(output_vector_cylindrical)

    return contours, contours_cylindrical
