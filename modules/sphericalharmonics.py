# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:38:41 2019

@author: PJ Hobson

Calculate magnetic fields based on the derivatives of the real spherical harmonics.
"""

import numpy as np


# format [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
#        [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
#        [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]]

def b1p1(scale=1):
    """Grad(Y_11) harmonic. Returns output in the following format
    [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]].

    Keyword arguments:
    scale -- multiplying factor of the harmonic
    """

    return [[[scale], [0, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[0], [0, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[0], [0, 0, 0], [0, 0, 0, 0, 0, 0]]]


def b10(scale=1):
    """Grad(Y_10) harmonic. Returns output in the following format
    [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]].

    Keyword arguments:
    scale -- multiplying factor of the harmonic
    """
    return [[[0], [0, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[0], [0, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[scale], [0, 0, 0], [0, 0, 0, 0, 0, 0]]]


def b1m1(scale=1):
    """Grad(Y_1-1) harmonic. Returns output in the following format
    [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]].

    Keyword arguments:
    scale -- multiplying factor of the harmonic
    """
    return [[[0], [0, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[scale], [0, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[0], [0, 0, 0], [0, 0, 0, 0, 0, 0]]]


def b2p2(scale=1):
    """Grad(Y_22) harmonic. Returns output in the following format
    [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]].

    Keyword arguments:
    scale -- multiplying factor of the harmonic
    """
    return [[[0], [scale, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[0], [0, -scale, 0], [0, 0, 0, 0, 0, 0]],
            [[0], [0, 0, 0], [0, 0, 0, 0, 0, 0]]]


def b2p1(scale=1):
    """Grad(Y_21) harmonic. Returns output in the following format
    [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]].

    Keyword arguments:
    scale -- multiplying factor of the harmonic
    """
    return [[[0], [0, 0, scale], [0, 0, 0, 0, 0, 0]],
            [[0], [0, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[0], [scale, 0, 0], [0, 0, 0, 0, 0, 0]]]


def b20(scale=1):
    """Grad(Y_20) harmonic. Returns output in the following format
    [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]].

    Keyword arguments:
    scale -- multiplying factor of the harmonic
    """
    return [[[0], [-scale, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[0], [0, -scale, 0], [0, 0, 0, 0, 0, 0]],
            [[0], [0, 0, 2 * scale], [0, 0, 0, 0, 0, 0]]]


def b2m1(scale=1):
    """Grad(Y_2-1) harmonic. Returns output in the following format
    [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]].

    Keyword arguments:
    scale -- multiplying factor of the harmonic
    """
    return [[[0], [0, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[0], [0, 0, scale], [0, 0, 0, 0, 0, 0]],
            [[0], [0, scale, 0], [0, 0, 0, 0, 0, 0]]]


def b2m2(scale=1):
    """Grad(Y_2-2) harmonic. Returns output in the following format
    [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]].

    Keyword arguments:
    scale -- multiplying factor of the harmonic
    """
    return [[[0], [0, scale, 0], [0, 0, 0, 0, 0, 0]],
            [[0], [scale, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[0], [0, 0, 0], [0, 0, 0, 0, 0, 0]]]


def b3p3(scale=1):
    """Grad(Y_33) harmonic. Returns output in the following format
    [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]].

    Keyword arguments:
    scale -- multiplying factor of the harmonic
    """
    return [[[0], [0, 0, 0], [scale, 0, 0, -scale, 0, 0]],
            [[0], [0, 0, 0], [0, -2 * scale, 0, 0, 0, 0]],
            [[0], [0, 0, 0], [0, 0, 0, 0, 0, 0]]]


def b3p2(scale=1):
    """Grad(Y_32) harmonic. Returns output in the following format
    [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]].

    Keyword arguments:
    scale -- multiplying factor of the harmonic
    """
    return [[[0], [0, 0, 0], [0, 0, 2 * scale, 0, 0, 0]],
            [[0], [0, 0, 0], [0, 0, 0, 0, -2 * scale, 0]],
            [[0], [0, 0, 0], [scale, 0, 0, -scale, 0, 0]]]


def b3p1(scale=1):
    """Grad(Y_31) harmonic. Returns output in the following format
    [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]].

    Keyword arguments:
    scale -- multiplying factor of the harmonic
    """
    return [[[0], [0, 0, 0], [-3 * scale, 0, 0, -scale, 0, 4 * scale]],
            [[0], [0, 0, 0], [0, -2 * scale, 0, 0, 0, 0]],
            [[0], [0, 0, 0], [0, 0, 8 * scale, 0, 0, 0]]]


def b30(scale=1):
    """Grad(Y_30) harmonic. Returns output in the following format
    [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]].

    Keyword arguments:
    scale -- multiplying factor of the harmonic
    """
    return [[[0], [0, 0, 0], [0, 0, -2 * scale, 0, 0, 0]],
            [[0], [0, 0, 0], [0, 0, 0, 0, -2 * scale, 0]],
            [[0], [0, 0, 0], [-scale, 0, 0, -scale, 0, 2 * scale]]]


def b3m1(scale=1):
    """Grad(Y_3-1) harmonic. Returns output in the following format
    [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]].

    Keyword arguments:
    scale -- multiplying factor of the harmonic
    """
    return [[[0], [0, 0, 0], [0, -2 * scale, 0, 0, 0, 0]],
            [[0], [0, 0, 0], [-scale, 0, 0, -3 * scale, 0, 4 * scale]],
            [[0], [0, 0, 0], [0, 0, 0, 0, 8 * scale, 0]]]


def b3m2(scale=1):
    """Grad(Y_3-2) harmonic. Returns output in the following format
    [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]].

    Keyword arguments:
    scale -- multiplying factor of the harmonic
    """
    return [[[0], [0, 0, 0], [0, 0, 0, 0, scale, 0]],
            [[0], [0, 0, 0], [0, 0, scale, 0, 0, 0]],
            [[0], [0, 0, 0], [0, scale, 0, 0, 0, 0]]]


def b3m3(scale=1):
    """Grad(Y_3-3) harmonic. Returns output in the following format
    [[[x],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[y],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]],
    [[z],[x,y,z],[x^2,xy,xz,y^2,yz,z^2]]].

    Keyword arguments:
    scale -- multiplying factor of the harmonic
    """
    return [[[0], [0, 0, 0], [0, 2 * scale, 0, 0, 0, 0]],
            [[0], [0, 0, 0], [scale, 0, 0, -scale, 0, 0]],
            [[0], [0, 0, 0], [0, 0, 0, 0, 0, 0]]]


def b_sphericalharmonics(coords_target, ind):
    """Calculates the magnetic field at coords_target using spherical harmonic indices.

    Keyword arguments:
    coords_target -- target points in Cartesian coordinates
    ind -- spherical harmonic indices
    """

    b_sh = np.zeros(coords_target.shape)

    for i in range(b_sh.shape[0]):

        for j in range(b_sh.shape[1]):
            bi = ind[j]

            b_sh[i][j] = (bi[0][0] +
                          bi[1][0] * coords_target[i, 0] +
                          bi[1][1] * coords_target[i, 1] +
                          bi[1][2] * coords_target[i, 2] +
                          bi[2][0] * coords_target[i, 0] * coords_target[i, 0] +
                          bi[2][1] * coords_target[i, 0] * coords_target[i, 1] +
                          bi[2][2] * coords_target[i, 0] * coords_target[i, 2] +
                          bi[2][3] * coords_target[i, 1] * coords_target[i, 1] +
                          bi[2][4] * coords_target[i, 1] * coords_target[i, 2] +
                          bi[2][5] * coords_target[i, 2] * coords_target[i, 2])

    return b_sh
