# SPDX-License-Identifier: Apache-2.0
# Part of quadriga-lib — see LICENSE for terms.
#
# Formula copy of what the library gives a closed mass-law sheet in air (md/acoustic.md 2.1, 2.3), written from
# src/ray_mesh_interact.cpp (Material::mass_X0, mass_TL, interface_gain, tf, qd_plate) as built on 2026-10-17 (M10:
# m > 0 switches the mass law on at 20 dB per decade; D2-b: no coiA). Column level, GHz as the library reads them;
# one face in air, one body, the plate rule R1h:
#   T = a tau / (a + tau - a tau) * g_int,  R = 1 - a,  a = tf(f) (power law, clamped to [-1, 1], negative -> 0)
#   tau = 10^(-TL/10),  TL = mass_TL(X0(f, c), c),  g_int = 10^(-att (f/fRef)^attB / 10), no gain below 0 dB
# Checked against the library through the walker (md/m10_m15_report.md 0.1, 2.4): T within 0.0022 dB, R within
# 0.0005 dB, sheet and traced solid, scalar types 4 / 5 and EM type 1 alike.
#
# Switches V (dict, every key optional) for what the library does not do:
#   mfit  the mass column as a fitted slope, X0 = ((f/fRef) M)^(m/20) (the library before 2026-10-17)
#   noatt att = 0 (candidate M11, dropped)
#   m13   'col': eta(f) = 1/coiQ + C (f/coiF)^-1/2 (column 'C'); 'davy': + m''/(485 sqrt f_Hz),
#         m'' = rho0 c thk / (pi fRef_Hz) (candidate M13, dropped)
#   m14   limiting angle in degrees: the mass cosine bounded below by cos(m14) (candidate M14, open)
#   m15   the bending loss as a resistance: TL from |1 + X c (j(1 - w) + w eta)|^2, normalized as the library
#         (candidate M15, a proposal)
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


def mass_TL(X, c, cross=0.0):
    """the library's mass_TL; cross = 2 c Xm w eta, the resistive cross term of M15 (zero in the library)"""
    if not (X > 1.0) or not (c > 0.0):
        return 0.0
    A = 20 * math.log10(X)
    if c >= 1.0:
        return A
    return A * math.log10(1 + (X * c) ** 2 + cross) / math.log10(1 + X * X)


def tau_of(M, fG, c, V={}):
    Xm, k = X0_parts(M, fG, c, V)
    if V.get('m14') is not None:
        c = max(c, math.cos(math.radians(V['m14'])))
    cross = 2.0 * c * Xm * (-k.imag) if V.get('m15') and k != 1.0 else 0.0   # -Im(kappa) = w eta
    return 10 ** (-0.1 * mass_TL(Xm * abs(k), c, cross))


def sheet(M, f_hz, th, V={}):
    """T, R of the closed sheet in air at incidence th (rad)"""
    fG = f_hz * FMAP
    c = math.cos(th)
    a = max(0.0, tf_of(M, fG))
    t = tau_of(M, fG, c, V)
    T = a * t / (a + t - a * t) if a > 0 and t > 0 else 0.0
    return T * ig_of(M, fG, V), 1 - a
