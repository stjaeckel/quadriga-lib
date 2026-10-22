# SPDX-License-Identifier: Apache-2.0
# Part of quadriga-lib — see LICENSE for terms.
#
# Formula copy of what the library gives a closed mass-law sheet in air (html_docu/quadriga_lib_material_model.md 5.4, 6.5), written from
# src/ray_mesh_interact.cpp (Material::mass_X0, mass_TL, davy_sigma, Material::interface_gain, Material::tf, qd_plate) as built on
# 2026-10-19 (m > 0 switches the mass law on at 20 dB per decade, bc24b0c; no coiA; the finite panel: the column side).
# Column level, GHz as the library reads them; one face in air, one body, the plate rule:
#   T = a tau / (a + tau - a tau) * g_int,  R = 1 - a,  a = tf(f) (power law, clamped to [-1, 1], negative -> 0)
#   tau = 10^(-TL/10),  TL = mass_TL(X0(f, c), c, f, side),  g_int = 10^(-att (f/fRef)^attB / 10), no gain below 0 dB
#   side (m, default 3.29 as the library where the column is absent; 0 the infinite plate): the radiation cosine is
#   1 / sigma, Davy's forced radiation efficiency of a square of that side, with the gain 1 / (sigma c), clipped at one
# Checked against the library through the walker (2026-10-16/17): T within 0.0022 dB, R within
# 0.0005 dB, sheet and traced solid, scalar types 4 / 5 and EM type 1 alike; as built on 2026-10-19 (the 13 fits of
# the starting rows of page 11.4 at their lab side, type 4): sheet T within 0.0000 dB, traced solid within 0.0042 dB, R within 0.0000 dB
# (de60194).
#
# Switches V (dict, every key optional) for what the library does not do:
#   mfit  the mass column as a fitted slope, X0 = ((f/fRef) M)^(m/20) (the library before 2026-10-17)
#   noatt att = 0 (a dropped candidate: no att)
#   m13   'col': eta(f) = 1/coiQ + C (f/coiF)^-1/2 (column 'C'); 'davy': + m''/(485 sqrt f_Hz),
#         m'' = rho0 c thk / (pi fRef_Hz) (a dropped candidate: the installed loss factor in frequency)
#   m14   limiting angle in degrees: every cosine bounded below by cos(m14) (a candidate closed 2026-10-19,
#         superseded by the finite panel; kept to reproduce the knob study of 2026-10-16)
#   m15   the bending loss as a resistance: TL from |1 + X c (j(1 - w) + w eta)|^2, normalized as the library
#         (a pending proposal)
import math

FMAP = 299792458.0 / 343.0 * 1e-9          # GHz per acoustic Hz (the tests' r1::FMAP)
RHO0C = 1.204 * 343.0                      # rho0 c of air, rayl (the mapping's air)


def tf_of(M, fG):
    v = M.get('tf', 0.0) * (fG / M['fRef']) ** M.get('tfB', 0.0)
    return max(-1.0, min(1.0, v))


def ig_of(M, fG, V={}):
    att = 0.0 if V.get('noatt') else M.get('att', 0.0)
    loss = att * (fG / M['fRef']) ** M.get('attB', 0.0)
    if loss < 0:
        return 1.0
    return 10 ** (-0.1 * loss)


def eta_of(M, fG, V):
    cQ, cF = M.get('coiQ', 0.0), M.get('coiF', 0.0)
    e = 1.0 / cQ
    if V.get('m13') == 'col':
        e += M.get('C', 0.0) * (fG / cF) ** -0.5
    elif V.get('m13') == 'davy':
        fR_hz, f_hz = M['fRef'] / FMAP, fG / FMAP
        e += RHO0C * M['thk'] / (math.pi * fR_hz) / (485.0 * math.sqrt(f_hz))
    return e


def X0_parts(M, fG, c, V={}, normal=None):
    """(Xm, kappa): Xm the mass and resonance part, kappa the complex bending factor (1 - w) - j w eta (1 without)"""
    m = M.get('m', 0.0)
    if not (m > 0):
        return 0.0, 1.0
    if V.get('m14') is not None:
        c = max(c, math.cos(math.radians(V['m14'])))
    arg = (fG / M['fRef']) * (M['thk'] if normal is None else normal)
    if not (arg > 0):
        return 0.0, 1.0
    X = arg ** (0.05 * m) if V.get('mfit') else arg
    F, Q, S = M.get('resF', 0.0), M.get('resQ', 0.0), M.get('resS', 0.0)
    if F > 0 and Q > 0 and S > 0:
        x = fG * c / F
        X *= min(1.0, abs(complex(1 - x * x, x / Q)))
    elif F > 0 and Q > 0 and S < 0:
        y = F / fG
        X *= abs(complex(1 - y * y, -y / Q))
    cF, cQ = M.get('coiF', 0.0), M.get('coiQ', 0.0)
    k = 1.0
    if cF > 0 and cQ > 0:
        s2 = 1 - c * c
        w = (fG / cF) ** 2 * s2 * s2
        k = complex(1 - w, -w * eta_of(M, fG, V))
    return X, k


def davy_sigma(fG, g, side):
    """the library's davy_sigma: Davy's forced radiation efficiency of a square of the given side (2a = side,
    S = side^2) at incidence cosine g [Davy_2009_ForcedRadiationEfficiency, Eqs. 19, 28, 34, 36, 37], k of the mapped
    frequency (the mapping keeps wavelengths), ka held at 0.1 or above"""
    k = 2 * math.pi * fG / 0.299792458
    ka = max(0.5 * k * side, 0.1)
    p = min(1.0, 1.3 * math.sqrt(math.pi / (2 * ka)))
    h = 1.0 / (2.0 / 3.0 * math.sqrt(2 * ka / math.pi) - 0.124)
    q = 2 * math.pi / (k * k * side * side)
    x = g if g >= p else h - (h / p - 1.0) * g
    return 1.0 / math.sqrt(x * x + q * q)


def mass_TL(X, c, cross=0.0, fG=1.0, side=0.0):
    """the library's mass_TL; cross = 2 c Xm w eta, the resistive cross term of the bending loss as a resistance (zero in the library); side > 0 the
    finite panel (fG its frequency in GHz)"""
    if not (X > 1.0) or not (c > 0.0):
        return 0.0
    A = 20 * math.log10(X)
    if not (side > 0.0):
        if c >= 1.0:
            return A
        return A * math.log10(1 + (X * c) ** 2 + cross) / math.log10(1 + X * X)
    c = min(c, 1.0)
    sg = davy_sigma(fG, c, side)
    return max(0.0, A * math.log10(1 + (X / sg) ** 2 + cross) / math.log10(1 + X * X) + 10 * math.log10(sg * c))


def tau_of(M, fG, c, V={}):
    Xm, k = X0_parts(M, fG, c, V)
    if V.get('m14') is not None:
        c = max(c, math.cos(math.radians(V['m14'])))
    side = M.get('side', 3.29)
    cr = c if not (side > 0.0) else 1.0 / davy_sigma(fG, min(c, 1.0), side) if c > 0.0 else c
    cross = 2.0 * cr * Xm * (-k.imag) if V.get('m15') and k != 1.0 else 0.0   # -Im(kappa) = w eta
    return 10 ** (-0.1 * mass_TL(Xm * abs(k), c, cross, fG, side))


def sheet(M, f_hz, th, V={}):
    """T, R of the closed sheet in air at incidence th (rad)"""
    fG = f_hz * FMAP
    c = math.cos(th)
    a = max(0.0, tf_of(M, fG))
    t = tau_of(M, fG, c, V)
    T = a * t / (a + t - a * t) if a > 0 and t > 0 else 0.0
    return T * ig_of(M, fG, V), 1 - a
