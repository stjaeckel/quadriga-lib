# SPDX-License-Identifier: Apache-2.0
# Part of quadriga-lib — see LICENSE for terms.
#
# Reference models of sound transmission through plates in air (md/grazing_report.md; md/acoustic.md 2.1), independent
# of the library. Plane-wave acoustics, a plate in a rigid baffle between two half-spaces of air:
#   - Plate: the transfer impedance Z_p(f, k) of a thin (Kirchhoff) or thick (Mindlin: shear and rotary inertia) plate at a
#     trace wavenumber k; eta(f) the loss factor (eta_edge: internal plus a coupling loss m''/(Ce sqrt f) to the frame)
#   - den_of / pi_of: a layer list (plates and air cavities, by transfer matrices at the trace wavenumber) with a radiation
#     efficiency sigma on both outer faces; pi_of is tau(th) cos(th), the transmitted power per unit incident intensity:
#     sigma = 1/cos th is the infinite plate (tau_inf), a windowed sigma the finite plate
#   - sigma_win: the radiation efficiency of the forced wave cut to an a x b opening in a baffle and radiated by the Rayleigh
#     integral (spatial windowing, Villot, Guigou, Gagliardini, JSV 245 (2001) 433), one face; average it over the azimuth
#   - davy_sigma / davy_mean: Davy's closed form of that efficiency, azimuth averaged, and its diffuse mean
#     [Davy_2009_ForcedRadiationEfficiency, Eqs. 19, 28, 34, 36, 37, 43], for a rectangle (2a = 4S/U, S); the library
#     takes it for a square of side sqrt(area) (S = side^2, U = 4 side)
# Checks (python3 tools/acoustic/gzlib.py): the layered model against the closed form tau = |1 + Z cos/(2 rho c)|^-2; the
# 4|16|4 mass-air-mass resonance at normal incidence; the windowed sigma against 1/cos and, at grazing, against
# Leppington's H1(x) sqrt(2 kL) and Davy's form. sigma_win is slow at high frequency for large plates (a few hundred
# thousand points per evaluation at 8 kHz on 10 x 3 m): cache it per size if you need many angles.
import math
import numpy as np

RHO0, C0 = 1.2, 343.0
Z0 = RHO0 * C0


def dB(x):
    return 10 * np.log10(x)


# ------------------------------------------------------------------------------------------------ plates and layers
class Plate:
    """m kg/m2; D bending stiffness N m; h thickness m; nu Poisson's ratio; eta(f) the loss factor; thick: Mindlin with the
    shear correction kap"""

    def __init__(self, m, D, h, nu=0.2, eta=lambda f: 0.01, thick=False, kap=5.0 / 6.0):
        self.m, self.D, self.h, self.nu, self.eta, self.thick, self.kap = m, D, h, nu, eta, thick, kap

    def fc(self):
        return C0 ** 2 / (2 * math.pi) * math.sqrt(self.m / self.D)

    def Z(self, f, k):
        """transfer impedance p / v_n for a wave of trace wavenumber k (array)"""
        w = 2 * math.pi * f
        e = 1 + 1j * self.eta(f)
        Dc = self.D * e
        if not self.thick:
            return 1j * w * self.m - 1j * Dc * k ** 4 / w
        E = 12 * self.D * (1 - self.nu ** 2) / self.h ** 3
        G = E / (2 * (1 + self.nu)) * e
        rho = self.m / self.h
        I = self.h ** 3 / 12
        kGh = self.kap * G * self.h
        num = kGh * k ** 2 * (Dc * k ** 2 - rho * I * w ** 2)
        den = w * (Dc * k ** 2 + kGh - rho * I * w ** 2)
        return 1j * w * self.m - 1j * num / den


def bending_B(h, E, nu):
    return E * h ** 3 / (12 * (1 - nu * nu))


def D_from_fc(m, fc):
    return m * (C0 ** 2 / (2 * math.pi * fc)) ** 2


def eta_edge(eta_int, m, Ce):
    """internal loss plus the coupling loss to the frame, m''/(Ce sqrt f) (Craik, Applied Acoustics 14 (1981))"""
    return lambda f: eta_int + (m / (Ce * math.sqrt(f)) if Ce > 0 else 0.0)


class Air:
    """an air cavity of depth d; a small loss through the wavenumber k (1 - j eta / 2)"""

    def __init__(self, d, eta=0.01):
        self.d, self.eta = d, eta


def den_of(layers, f, kx, Zr):
    """2 p_i / v_out for the layer list (source side first) at frequency f and trace wavenumber kx, with the radiation
    impedance Zr on both outer faces: the source face sees the blocked pressure 2 p_i less its own radiation"""
    k = 2 * math.pi * f / C0
    T11, T12, T21, T22 = 1.0 + 0j, 0j, 0j, 1.0 + 0j
    for L in layers:
        if isinstance(L, Air):
            kc = k * (1 - 0.5j * L.eta)
            kz = np.sqrt(kc * kc - kx * kx + 0j)
            Zc = Z0 * kc / kz
            a11 = np.cos(kz * L.d); a12 = 1j * Zc * np.sin(kz * L.d); a21 = 1j * np.sin(kz * L.d) / Zc; a22 = a11
        else:
            a11, a12, a21, a22 = 1.0, L.Z(f, kx), 0.0, 1.0
        T11, T12, T21, T22 = T11 * a11 + T12 * a21, T11 * a12 + T12 * a22, T21 * a11 + T22 * a21, T21 * a12 + T22 * a22
    return T11 * Zr + T12 + Zr * (T21 * Zr + T22)


def pi_of(layers, f, th, sig):
    """tau(th) cos(th) with the one-face radiation efficiency sig of the forced wave (array like th)"""
    k = 2 * math.pi * f / C0
    Zr = Z0 * np.asarray(sig)
    return 4 * Z0 * Zr / np.abs(den_of(layers, f, k * np.sin(th), Zr)) ** 2


def tau_inf(layers, f, th):
    """the infinite construction: sigma = 1 / cos"""
    c = np.maximum(np.cos(th), 1e-300)
    return pi_of(layers, f, th, 1.0 / c) / c


# ------------------------------------------------------------------------------------------------ the windowed sigma
def gl(n, lo, hi):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


def kgrid(k0, a, b, P=6):
    """radiation-circle grid: kx midpoints on (-k0, k0), beta midpoints on (-pi/2, pi/2), ky = sqrt(k0^2 - kx^2) sin beta;
    dkx dky / sqrt(k0^2 - k^2) = dkx dbeta. P points per window period 2 pi / size"""
    nx = max(64, int(math.ceil(2 * k0 * a * P / (2 * math.pi))))
    nb = max(64, int(math.ceil(math.pi * k0 * b * P / (2 * math.pi))))
    kx = -k0 + (np.arange(nx) + 0.5) * (2 * k0 / nx)
    be = -math.pi / 2 + (np.arange(nb) + 0.5) * (math.pi / nb)
    s = np.sqrt(np.maximum(k0 ** 2 - kx ** 2, 0.0))
    ky = s[:, None] * np.sin(be)[None, :]
    return kx, ky, (2 * k0 / nx) * (math.pi / nb)


def sigma_win(f, th, ph, a, b, grid=None):
    """radiation efficiency (one face) of the forced wave exp(-j k sin th (x cos ph + y sin ph)) windowed to the a x b
    opening; grid from kgrid(k0, a, b) to reuse"""
    k0 = 2 * math.pi * f / C0
    kx, ky, dA = grid if grid is not None else kgrid(k0, a, b)
    kp = k0 * math.sin(th)
    win = lambda u, L: (L * np.sinc(0.5 * u * L / math.pi)) ** 2
    return k0 / (4 * math.pi ** 2 * a * b) * dA * float(np.sum(win(kx - kp * math.cos(ph), a)[:, None] * win(ky - kp * math.sin(ph), b)))


def sigma_win_mean(f, th, a, b, nph=8):
    """sigma_win averaged over the azimuth (rectangle symmetry: Gauss-Legendre on 0 .. 90 deg)"""
    k0 = 2 * math.pi * f / C0
    grid = kgrid(k0, a, b)
    ph, wp = gl(nph, 0.0, math.pi / 2)
    return sum(w * sigma_win(f, th, p, a, b, grid) for p, w in zip(ph, wp)) / wp.sum()


# ------------------------------------------------------------------------------------------------ Davy 2009
def davy_pars(k, S, U, w=1.3, beta=0.124):
    """[Davy_2009_ForcedRadiationEfficiency, Eqs. 19, 28, 34, 36] with 2a = 4S/U and the area S"""
    ka = k * 2 * S / U
    p = min(1.0, w * math.sqrt(math.pi / (2 * ka)))
    h = 1.0 / (2.0 / 3.0 * math.sqrt(2 * ka / math.pi) - beta)
    return p, h, h / p - 1.0, 2 * math.pi / (k * k * S)


def davy_sigma(k, S, U, th, n=2.0):
    """Eq. 37: the forced radiation efficiency at incidence th (array), azimuth averaged"""
    p, h, al, q = davy_pars(k, S, U)
    g = np.cos(th)
    return np.where(g >= p, 1.0 / (g ** n + q ** n) ** (1 / n), 1.0 / ((h - al * g) ** n + q ** n) ** (1 / n))


def davy_mean(k, S, U):
    """Eq. 43 (n = 2): <sigma> = int_0^90 sigma sin th dth; the infinite plate's limiting angle that gives the same diffuse
    mass law is cos th_l = exp(1/2 - <sigma>) (md/grazing_report.md 0.5)"""
    p, h, al, q = davy_pars(k, S, U)
    r = lambda x: x + math.sqrt(x * x + q * q)
    s = math.log(r(1.0) / r(p))
    return s + (math.log(r(h) / r(p)) / al if abs(al) > 1e-12 else p / math.sqrt(h * h + q * q))


def side_SU(side):
    """the library's one-number size: a square of side sqrt(area)"""
    return side * side, 4.0 * side


# ------------------------------------------------------------------------------------------------ checks
if __name__ == '__main__':
    glass = Plate(m=10.0, D=bending_B(0.004, 70e9, 0.22), h=0.004, nu=0.22, eta=lambda f: 0.015)
    th = np.radians(np.array([0.0, 30.0, 60.0, 85.0]))
    k500 = 2 * math.pi * 500.0 / C0
    closed = np.abs(1 + glass.Z(500.0, k500 * np.sin(th)) * np.cos(th) / (2 * Z0)) ** -2
    print('layered model against the closed form, glass 4 mm, 500 Hz, 0/30/60/85 deg: max rel. diff %.1e'
          % np.max(np.abs(tau_inf([glass], 500.0, th) / closed - 1)))
    f = np.exp(np.linspace(math.log(100), math.log(400), 20000))
    t = np.array([tau_inf([glass, Air(0.016), glass], x, np.array([0.0]))[0] for x in f])
    print('4|16|4 at normal incidence: transmission peak at %.1f Hz (md/acoustic.md 2.8: 211.3)' % f[np.argmax(t)])
    for a, b in ((1.23, 1.48), (4.0, 2.7)):
        S, U = a * b, 2 * (a + b)
        for fx in (500.0, 2000.0):
            k = 2 * math.pi * fx / C0
            s30, s90 = sigma_win_mean(fx, math.radians(30), a, b), sigma_win_mean(fx, math.pi / 2, a, b)
            L, W = min(a, b) / 2, max(a, b) / 2
            lep = (0.5 - 0.15 * L / W) * math.sqrt(2 * k * L)
            dav = float(davy_sigma(k, S, U, np.array([math.pi / 2]))[0])
            print('%.2f x %.2f m, %4.0f Hz: sigma(30 deg) cos 30 = %.4f; sigma(90 deg) %.3f, Leppington %+.2f dB, Davy %+.2f dB'
                  % (a, b, fx, s30 * math.cos(math.radians(30)), s90, dB(lep / s90), dB(dav / s90)))
