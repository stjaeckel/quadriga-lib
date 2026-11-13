// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for arrayant_interpolate.
//
// The suite pins the interpolated field values, the geometry of a rotated antenna (local angles, polarization
// rotation, projected distance), the handling of every argument and output, and the agreement of the GENERIC and the
// AVX2 kernel. The multi-frequency variant is pinned by test_arrayant_interpolate_multi.cpp; the functions that
// interpolate patterns internally (pattern rotation and combination, channel generation) by their own suites.
//
// Oracles:
//  - Two samples: between two complex samples the result is their spherical interpolation (amplitude and phase
//    linear in the position between the samples), or the linear interpolation for opposite phases.
//  - Smooth pattern: amplitude and phase of every element are linear functions of azimuth and elevation. Linear
//    amplitude and spherical phase interpolation reproduce such a function exactly between the samples, so the
//    closed-form function is the oracle at any angle away from the azimuth wrap at +/- pi.
//  - Rotated antenna: the direction is transformed into the local frame with the rotation matrix of
//    calc_rotation_matrix (library call); the local field comes from the smooth pattern; the polarization rotation
//    angle follows from the theta basis vectors of the global and the local frame, the projected distance from the
//    local direction and the element position. The signs of heading and tilt are pinned by hand values.
//  - The AVX2 kernel is compared with the GENERIC kernel on a 3GPP panel at random directions, with tolerances from
//    single-precision arithmetic, and at fixed directions (poles of the global and the local frame, azimuth wrap,
//    grid points), with tolerances pinned to the measured accuracy.
//
// Conventions: angles in rad; the smooth pattern has 3 elements on a 20 x 15 degree grid; test angles keep the local
// azimuth within +/- 170 degree and the local elevation within +/- 80 degree (except where a case says otherwise).

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "qd_test_diag.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

QD_TEST_DIAG_SWITCH("QD_DIAG_ANT")

namespace
{
    constexpr double pi = 3.14159265358979323846;
    using cxd = std::complex<double>;

    // Kernels to run: GENERIC always, AVX2 where the library and the CPU support it
    std::vector<int> kernels()
    {
        std::vector<int> k = {1};
        if (quadriga_lib::quadriga_lib_has_AVX2())
            k.push_back(2);
        return k;
    }

    // Absolute tolerances for results that are exact by construction (fields of magnitude up to 3, angles in rad):
    // - double precision (GENERIC with double): rounding of the trigonometric functions and of the phase difference
    //   acos(cos(dphi)) of two samples, a few 1e-15; 1e-12 leaves a wide margin.
    // - single precision (GENERIC with float, AVX2 with float and double): an angle carries 2 to 4 roundings of float
    //   epsilon times pi (4e-7 each, including the approximations of the AVX2 functions), at most about 1.5e-6 rad;
    //   a field carries about 10 roundings of float epsilon times its magnitude (4e-6) plus the angle error times
    //   the slope of the field (up to 4 per rad, 6e-6). 2e-6 rad and 1e-5 cover these bounds.
    template <typename dtype>
    double tol_field(int kernel)
    {
        return (kernel == 1 && std::is_same<dtype, double>::value) ? 1e-12 : 1e-5;
    }
    template <typename dtype>
    double tol_angle(int kernel)
    {
        return (kernel == 1 && std::is_same<dtype, double>::value) ? 1e-12 : 2e-6;
    }

    // Uniform random numbers in [lo, hi) from the raw output of a seeded generator (same values on all platforms)
    struct qd_uniform
    {
        std::mt19937 gen;
        explicit qd_uniform(unsigned seed) : gen(seed) {}
        double operator()(double lo, double hi) { return lo + (hi - lo) * (double)gen() / 4294967296.0; }
    };

    // All outputs of one call
    template <typename dtype>
    struct qd_result
    {
        arma::Mat<dtype> V_re, V_im, H_re, H_im, dist, az_loc, el_loc, gamma;
    };

    // Calls with all outputs; the optional inputs take nullptr without a cast (dtype is deduced from ant)
    template <typename T>
    struct qd_same
    {
        using type = T;
    };
    template <typename dtype>
    qd_result<dtype> run(const quadriga_lib::arrayant<dtype> &ant, const arma::Mat<dtype> &az, const arma::Mat<dtype> &el,
                         const arma::uvec *i_element, const arma::Cube<typename qd_same<dtype>::type> *orientation,
                         const arma::Mat<typename qd_same<dtype>::type> *element_pos, int kernel)
    {
        qd_result<dtype> r;
        quadriga_lib::arrayant_interpolate<dtype>(ant, az, el, r.V_re, r.V_im, r.H_re, r.H_im, i_element, orientation, element_pos,
                                                  &r.dist, &r.az_loc, &r.el_loc, &r.gamma, kernel);
        return r;
    }

    template <typename dtype>
    bool identical(const arma::Mat<dtype> &a, const arma::Mat<dtype> &b)
    {
        return arma::size(a) == arma::size(b) && std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(dtype)) == 0;
    }

    template <typename dtype>
    bool identical(const qd_result<dtype> &a, const qd_result<dtype> &b)
    {
        return identical(a.V_re, b.V_re) && identical(a.V_im, b.V_im) && identical(a.H_re, b.H_re) &&
               identical(a.H_im, b.H_im) && identical(a.dist, b.dist) && identical(a.az_loc, b.az_loc) &&
               identical(a.el_loc, b.el_loc) && identical(a.gamma, b.gamma);
    }

    // Wraps an angle difference to (-pi, pi]
    double wrap(double x)
    {
        return std::remainder(x, 2.0 * pi);
    }

    // ---- Two-sample patterns ----

    // One element with two samples A and B of e-theta and e-phi, along azimuth (grid {0, pi}) or along elevation
    // (grid {0, pi/2})
    template <typename dtype>
    quadriga_lib::arrayant<dtype> two_sample_ant(bool along_elevation, cxd VA, cxd VB, cxd HA, cxd HB)
    {
        quadriga_lib::arrayant<dtype> ant;
        const arma::uword n_el = along_elevation ? 2 : 1, n_az = along_elevation ? 1 : 2;
        ant.e_theta_re.set_size(n_el, n_az, 1), ant.e_theta_im.set_size(n_el, n_az, 1);
        ant.e_phi_re.set_size(n_el, n_az, 1), ant.e_phi_im.set_size(n_el, n_az, 1);
        const cxd V[2] = {VA, VB}, H[2] = {HA, HB};
        for (arma::uword i = 0; i < 2; ++i)
        {
            ant.e_theta_re(i) = (dtype)V[i].real(), ant.e_theta_im(i) = (dtype)V[i].imag();
            ant.e_phi_re(i) = (dtype)H[i].real(), ant.e_phi_im(i) = (dtype)H[i].imag();
        }
        ant.azimuth_grid = along_elevation ? arma::Col<dtype>({(dtype)0.0}) : arma::Col<dtype>({(dtype)0.0, (dtype)pi});
        ant.elevation_grid = along_elevation ? arma::Col<dtype>({(dtype)0.0, (dtype)(0.5 * pi)}) : arma::Col<dtype>({(dtype)0.0});
        return ant;
    }

    // Oracle between two samples at fraction t in [0, 1]: linear for opposite phases, otherwise amplitude linear and
    // phase linear along the shorter arc
    cxd two_sample_oracle(cxd A, cxd B, double t)
    {
        const double dphi = wrap(std::arg(B) - std::arg(A));
        if (std::abs(std::abs(dphi) - pi) < 1e-12)
            return (1.0 - t) * A + t * B;
        return std::polar((1.0 - t) * std::abs(A) + t * std::abs(B), std::arg(A) + t * dphi);
    }

    // ---- Smooth pattern ----

    // Amplitude a0 + a_az * az + a_el * el and phase p0 + p_az * az + p_el * el
    struct qd_lin_coef
    {
        double a0, a_az, a_el, p0, p_az, p_el;
    };
    constexpr qd_lin_coef smooth_V[3] = {{2.0, 0.25, -0.3, 0.3, 0.8, -0.5},
                                         {1.5, -0.2, 0.4, -1.0, -0.6, 1.2},
                                         {2.5, 0.1, 0.2, 2.0, 1.1, 0.7}};
    constexpr qd_lin_coef smooth_H[3] = {{1.2, -0.15, 0.25, -0.4, 0.5, 0.9},
                                         {2.2, 0.3, -0.1, 1.3, -1.0, -0.4},
                                         {0.9, 0.05, -0.2, 0.0, 0.3, -1.1}};

    cxd smooth_eval(const qd_lin_coef &c, double az, double el)
    {
        return std::polar(c.a0 + c.a_az * az + c.a_el * el, c.p0 + c.p_az * az + c.p_el * el);
    }

    // 3 elements on a 20 x 15 degree grid; the samples are evaluated at the grid values of dtype, so that the function
    // is exact for the grid the interpolation sees; distinct element positions
    template <typename dtype>
    quadriga_lib::arrayant<dtype> smooth_ant()
    {
        quadriga_lib::arrayant<dtype> ant;
        ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 19);
        ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-0.5 * pi, 0.5 * pi, 13);
        ant.e_theta_re.set_size(13, 19, 3), ant.e_theta_im.set_size(13, 19, 3);
        ant.e_phi_re.set_size(13, 19, 3), ant.e_phi_im.set_size(13, 19, 3);
        for (arma::uword e = 0; e < 3; ++e)
            for (arma::uword ia = 0; ia < 19; ++ia)
                for (arma::uword ie = 0; ie < 13; ++ie)
                {
                    const double az = (double)ant.azimuth_grid(ia), el = (double)ant.elevation_grid(ie);
                    const cxd V = smooth_eval(smooth_V[e], az, el), H = smooth_eval(smooth_H[e], az, el);
                    ant.e_theta_re(ie, ia, e) = (dtype)V.real(), ant.e_theta_im(ie, ia, e) = (dtype)V.imag();
                    ant.e_phi_re(ie, ia, e) = (dtype)H.real(), ant.e_phi_im(ie, ia, e) = (dtype)H.imag();
                }
        ant.element_pos = {{0.1, 0.0, 0.2}, {-0.2, 0.3, 0.1}, {0.05, -0.1, 0.4}};
        return ant;
    }

    // Expected outputs of one element and angle
    struct qd_expect
    {
        cxd V, H;
        double az_loc, el_loc, gamma, dist;
    };

    // Rotated-frame oracle for element e of the smooth pattern at the global direction (az, el); R: rotation matrix
    // from local to global coordinates, 9 values in column-major order; pos: element position in local coordinates
    // - local direction d = R^T c, local angles from d
    // - gamma: angle from the global theta basis vector to the local one (mapped to global coordinates), measured
    //   towards the global phi basis vector; the fields rotate by gamma: V = cos(gamma) V_loc - sin(gamma) H_loc,
    //   H = sin(gamma) V_loc + cos(gamma) H_loc
    // - dist = -(d . pos): elements closer to the source (positive projection) have a negative distance
    qd_expect rotated_oracle(arma::uword e, double az, double el, const double *R, const double *pos)
    {
        const double c[3] = {std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el)};
        double d[3];
        for (int i = 0; i < 3; ++i)
            d[i] = R[3 * i] * c[0] + R[3 * i + 1] * c[1] + R[3 * i + 2] * c[2]; // Column i of R dotted with c
        qd_expect x;
        x.el_loc = std::asin(std::max(-1.0, std::min(1.0, d[2])));
        x.az_loc = std::atan2(d[1], d[0]);

        const double eth_g[3] = {std::sin(el) * std::cos(az), std::sin(el) * std::sin(az), -std::cos(el)};
        const double eph_g[3] = {-std::sin(az), std::cos(az), 0.0};
        const double eth_l[3] = {std::sin(x.el_loc) * std::cos(x.az_loc), std::sin(x.el_loc) * std::sin(x.az_loc), -std::cos(x.el_loc)};
        double eth_lg[3];
        for (int i = 0; i < 3; ++i)
            eth_lg[i] = R[i] * eth_l[0] + R[3 + i] * eth_l[1] + R[6 + i] * eth_l[2];
        const double cg = eth_g[0] * eth_lg[0] + eth_g[1] * eth_lg[1] + eth_g[2] * eth_lg[2];
        const double sg = eph_g[0] * eth_lg[0] + eph_g[1] * eth_lg[1];
        x.gamma = std::atan2(sg, cg);

        const cxd VL = smooth_eval(smooth_V[e], x.az_loc, x.el_loc), HL = smooth_eval(smooth_H[e], x.az_loc, x.el_loc);
        x.V = cg * VL - sg * HL;
        x.H = sg * VL + cg * HL;
        x.dist = -(d[0] * pos[0] + d[1] * pos[1] + d[2] * pos[2]);
        return x;
    }

    // Largest deviations of a result from the expected values (index o + a * n_out); gamma compared modulo 2 pi
    struct qd_dev
    {
        double field = 0.0, az = 0.0, el = 0.0, gamma = 0.0, dist = 0.0;
    };

    template <typename dtype>
    qd_dev deviation(const qd_result<dtype> &r, const std::vector<qd_expect> &x, arma::uword n_out, arma::uword n_ang)
    {
        REQUIRE(r.V_re.n_rows == n_out);
        REQUIRE(r.V_re.n_cols == n_ang);
        REQUIRE(r.gamma.n_rows == n_out);
        REQUIRE(r.gamma.n_cols == n_ang);
        qd_dev d;
        for (arma::uword i = 0; i < n_out * n_ang; ++i)
        {
            const cxd V((double)r.V_re(i), (double)r.V_im(i)), H((double)r.H_re(i), (double)r.H_im(i));
            d.field = std::max(d.field, std::max(std::abs(V - x[i].V), std::abs(H - x[i].H)));
            d.az = std::max(d.az, std::abs(wrap((double)r.az_loc(i) - x[i].az_loc)));
            d.el = std::max(d.el, std::abs((double)r.el_loc(i) - x[i].el_loc));
            d.gamma = std::max(d.gamma, std::abs(wrap((double)r.gamma(i) - x[i].gamma)));
            d.dist = std::max(d.dist, std::abs((double)r.dist(i) - x[i].dist));
        }
        return d;
    }

    template <typename dtype>
    void check_deviation(const qd_dev &d, int kernel, const char *what)
    {
        QD_DIAG("%s,%s,%d,%.3e,%.3e,%.3e,%.3e,%.3e\n", what, std::is_same<dtype, float>::value ? "float" : "double",
                kernel, d.field, d.az, d.el, d.gamma, d.dist);
        CHECK(d.field < tol_field<dtype>(kernel));
        CHECK(d.az < tol_angle<dtype>(kernel));
        CHECK(d.el < tol_angle<dtype>(kernel));
        CHECK(d.gamma < tol_angle<dtype>(kernel));
        CHECK(d.dist < tol_angle<dtype>(kernel));
    }

    // Expected values of the smooth pattern for the angles az / el ([1, n_ang] or [n_out, n_ang]), element indices
    // i_element, orientation ([3, 1 or n_out, 1 or n_ang]) and positions pos [3, n_out]
    template <typename dtype>
    std::vector<qd_expect> smooth_expect(const arma::Mat<dtype> &az, const arma::Mat<dtype> &el, const arma::uvec &i_element,
                                         const arma::Cube<dtype> &orientation, const arma::Mat<dtype> &pos)
    {
        const arma::uword n_out = i_element.n_elem, n_ang = az.n_cols;
        const arma::cube R = quadriga_lib::calc_rotation_matrix(arma::conv_to<arma::cube>::from(orientation));
        std::vector<qd_expect> x(n_out * n_ang);
        for (arma::uword a = 0; a < n_ang; ++a)
            for (arma::uword o = 0; o < n_out; ++o)
            {
                const arma::uword r = az.n_rows == 1 ? 0 : o;
                const double p[3] = {(double)pos(0, o), (double)pos(1, o), (double)pos(2, o)};
                const double *Rp = R.slice_colptr(R.n_slices == 1 ? 0 : a, R.n_cols == 1 ? 0 : o);
                x[o + a * n_out] = rotated_oracle(i_element(o), (double)az(r, a), (double)el(r, a), Rp, p);
                REQUIRE(std::abs(x[o + a * n_out].az_loc) < 170.0 * pi / 180.0); // Fixture: away from the azimuth wrap
                REQUIRE(std::abs(x[o + a * n_out].el_loc) < 80.0 * pi / 180.0);  // Fixture: away from the poles
            }
        return x;
    }

    template <typename dtype>
    arma::Mat<dtype> random_angles(qd_uniform &rnd, arma::uword n_rows, arma::uword n_cols, double lo, double hi)
    {
        arma::Mat<dtype> x(n_rows, n_cols);
        for (auto &v : x)
            v = (dtype)rnd(lo, hi);
        return x;
    }

    // ---- 3GPP panel for the kernel comparison ----

    // Copy of an arrayant in another precision (the comparisons run both precisions on the same pattern samples)
    template <typename T, typename S>
    quadriga_lib::arrayant<T> convert(const quadriga_lib::arrayant<S> &a)
    {
        quadriga_lib::arrayant<T> b;
        b.e_theta_re = arma::conv_to<arma::Cube<T>>::from(a.e_theta_re), b.e_theta_im = arma::conv_to<arma::Cube<T>>::from(a.e_theta_im);
        b.e_phi_re = arma::conv_to<arma::Cube<T>>::from(a.e_phi_re), b.e_phi_im = arma::conv_to<arma::Cube<T>>::from(a.e_phi_im);
        b.azimuth_grid = arma::conv_to<arma::Col<T>>::from(a.azimuth_grid), b.elevation_grid = arma::conv_to<arma::Col<T>>::from(a.elevation_grid);
        b.element_pos = arma::conv_to<arma::Mat<T>>::from(a.element_pos);
        b.coupling_re = arma::conv_to<arma::Mat<T>>::from(a.coupling_re), b.coupling_im = arma::conv_to<arma::Mat<T>>::from(a.coupling_im);
        return b;
    }

    // Largest magnitude M and largest slope S (per rad, from neighbouring samples) of the four pattern components
    template <typename dtype>
    void pattern_scale(const quadriga_lib::arrayant<dtype> &ant, double &M, double &S)
    {
        M = 0.0, S = 0.0;
        const arma::Cube<dtype> *P[4] = {&ant.e_theta_re, &ant.e_theta_im, &ant.e_phi_re, &ant.e_phi_im};
        for (const auto *p : P)
            for (arma::uword s = 0; s < p->n_slices; ++s)
                for (arma::uword c = 0; c < p->n_cols; ++c)
                    for (arma::uword r = 0; r < p->n_rows; ++r)
                    {
                        const double v = (double)(*p)(r, c, s);
                        M = std::max(M, std::abs(v));
                        if (r > 0)
                            S = std::max(S, std::abs(v - (double)(*p)(r - 1, c, s)) / (double)(ant.elevation_grid(r) - ant.elevation_grid(r - 1)));
                        if (c > 0)
                            S = std::max(S, std::abs(v - (double)(*p)(r, c - 1, s)) / (double)(ant.azimuth_grid(c) - ant.azimuth_grid(c - 1)));
                    }
    }

    // Compares a result computed with single precision with a double-precision reference, sample by sample.
    // Tolerances from single-precision arithmetic: 4 roundings of float epsilon on the angles (times pi, in rad) and
    // on the field (times M for the value and pi S for the angle error), amplified near the local poles by the
    // conditioning of the local azimuth 1 + 1 / cos(el_loc), with el_loc of the reference; the distance carries the
    // angle error times the largest element distance pos_max. Returns the largest ratio of deviation to tolerance.
    template <typename dtype>
    double compare_single_precision(const qd_result<dtype> &a, const qd_result<double> &ref, double M, double S,
                                    double pos_max, const char *what)
    {
        const double eps = 4.0 * (double)std::numeric_limits<float>::epsilon();
        double q_field = 0.0, q_angle = 0.0, q_dist = 0.0, d_field = 0.0, d_angle = 0.0, d_dist = 0.0;
        REQUIRE(arma::size(a.V_re) == arma::size(ref.V_re));
        CHECK(arma::abs(ref.V_re).max() > 0.1); // Not a trivial (zero) result
        for (arma::uword i = 0; i < ref.V_re.n_elem; ++i)
        {
            const double cond = 1.0 + 1.0 / std::max(std::cos(ref.el_loc(i)), 1e-12);
            const double tf = eps * (M + pi * S) * cond, ta = eps * pi * cond, td = eps * pos_max * (1.0 + pi * cond);
            const double df = std::max({std::abs((double)a.V_re(i) - ref.V_re(i)), std::abs((double)a.V_im(i) - ref.V_im(i)),
                                        std::abs((double)a.H_re(i) - ref.H_re(i)), std::abs((double)a.H_im(i) - ref.H_im(i))});
            const double da = std::max({std::abs(wrap((double)a.az_loc(i) - ref.az_loc(i))),
                                        std::abs((double)a.el_loc(i) - ref.el_loc(i)),
                                        std::abs(wrap((double)a.gamma(i) - ref.gamma(i)))});
            const double dd = std::abs((double)a.dist(i) - ref.dist(i));
            d_field = std::max(d_field, df), d_angle = std::max(d_angle, da), d_dist = std::max(d_dist, dd);
            q_field = std::max(q_field, df / tf), q_angle = std::max(q_angle, da / ta), q_dist = std::max(q_dist, dd / td);
        }
        QD_DIAG("%s,%s,%.3e,%.3f,%.3e,%.3f,%.3e,%.3f\n", what, std::is_same<dtype, float>::value ? "float" : "double",
                d_field, q_field, d_angle, q_angle, d_dist, q_dist);
        return std::max({q_field, q_angle, q_dist});
    }
} // namespace

TEST_CASE("arrayant_interpolate - two samples along azimuth and elevation give the hand values")
{
    QD_DIAG("\n# diag: two-sample patterns, largest deviation from the hand values\n");
    QD_DIAG("case,dtype,kernel,d_field,d_az,d_el,d_gamma,d_dist\n");

    // Linear case: e-theta in opposite phase (linear fallback), e-phi in phase (amplitude interpolation);
    // spherical case: e-theta with a phase step of 90 degree, e-phi with a phase and an amplitude step
    const cxd lin[4] = {{-2.0, -1.0}, {2.0, 1.0}, {3.0, 6.0}, {1.0, 2.0}};
    const cxd sph[4] = {{1.0, 0.0}, {0.0, 1.0}, {-2.0, 0.0}, {0.0, -1.0}};

    auto check = [](auto dtype_tag, bool along_el, const cxd *s, const char *what)
    {
        using dtype = decltype(dtype_tag);
        const auto ant = two_sample_ant<dtype>(along_el, s[0], s[1], s[2], s[3]);
        const double span = along_el ? 0.5 * pi : pi;
        arma::Mat<dtype> az(1, 4), el(1, 4);
        std::vector<qd_expect> x(4);
        for (arma::uword a = 0; a < 4; ++a)
        {
            const double t = 0.25 * (double)a, other = 0.5 * ((double)a - 1.5); // Position on the grid; other angle
            az(a) = (dtype)(along_el ? other : t * span), el(a) = (dtype)(along_el ? t * span : other);
            x[a] = {two_sample_oracle(s[0], s[1], t), two_sample_oracle(s[2], s[3], t), (double)az(a), (double)el(a), 0.0, 0.0};
        }
        for (int kernel : kernels())
        {
            const auto r = run(ant, az, el, nullptr, nullptr, nullptr, kernel);
            check_deviation<dtype>(deviation(r, x, 1, 4), kernel, what);
        }
    };
    for (bool along_el : {false, true})
    {
        check(0.0f, along_el, lin, along_el ? "linear el" : "linear az");
        check(0.0, along_el, lin, along_el ? "linear el" : "linear az");
        check(0.0f, along_el, sph, along_el ? "spherical el" : "spherical az");
        check(0.0, along_el, sph, along_el ? "spherical el" : "spherical az");
    }
}

TEST_CASE("arrayant_interpolate - samples on the grid are reproduced")
{
    QD_DIAG("\n# diag: smooth pattern at the grid points, largest deviation\n");
    QD_DIAG("case,dtype,kernel,d_field,d_az,d_el,d_gamma,d_dist\n");

    auto check = [](auto dtype_tag)
    {
        using dtype = decltype(dtype_tag);
        const auto ant = smooth_ant<dtype>();
        // Grid points without the azimuth wrap (+/- pi) and the poles
        arma::Mat<dtype> az(1, 17 * 11), el(1, 17 * 11);
        for (arma::uword ia = 0; ia < 17; ++ia)
            for (arma::uword ie = 0; ie < 11; ++ie)
                az(ia * 11 + ie) = ant.azimuth_grid(ia + 1), el(ia * 11 + ie) = ant.elevation_grid(ie + 1);
        const arma::uvec i_all = {0, 1, 2};
        const auto x = smooth_expect(az, el, i_all, arma::Cube<dtype>(3, 1, 1, arma::fill::zeros), ant.element_pos);
        for (int kernel : kernels())
            check_deviation<dtype>(deviation(run(ant, az, el, nullptr, nullptr, nullptr, kernel), x, 3, az.n_cols), kernel, "grid");
    };
    check(0.0f);
    check(0.0);
}

TEST_CASE("arrayant_interpolate - a smooth pattern is reproduced between the samples")
{
    QD_DIAG("\n# diag: smooth pattern at random angles, no rotation, largest deviation\n");
    QD_DIAG("case,dtype,kernel,d_field,d_az,d_el,d_gamma,d_dist\n");

    auto check = [](auto dtype_tag)
    {
        using dtype = decltype(dtype_tag);
        const auto ant = smooth_ant<dtype>();
        const arma::uvec i_all = {0, 1, 2};
        const arma::Cube<dtype> no_rotation(3, 1, 1, arma::fill::zeros);
        qd_uniform rnd(11);
        for (arma::uword n_rows : {1, 3}) // Planar and per-element angles
        {
            const auto az = random_angles<dtype>(rnd, n_rows, 50, -170.0 * pi / 180.0, 170.0 * pi / 180.0);
            const auto el = random_angles<dtype>(rnd, n_rows, 50, -80.0 * pi / 180.0, 80.0 * pi / 180.0);
            const auto x = smooth_expect(az, el, i_all, no_rotation, ant.element_pos);
            for (int kernel : kernels())
                check_deviation<dtype>(deviation(run(ant, az, el, nullptr, nullptr, nullptr, kernel), x, 3, 50), kernel,
                                       n_rows == 1 ? "planar" : "per-element");
        }
    };
    check(0.0f);
    check(0.0);
}

TEST_CASE("arrayant_interpolate - heading and tilt shift the local angles")
{
    // Heading -pi/8 adds pi/8 to the local azimuth; tilt +pi/16 (a downtilt, 3GPP TR 38.901) adds pi/16 to the local
    // elevation (hand values)
    QD_DIAG("\n# diag: heading and tilt, largest deviation from the hand values\n");
    QD_DIAG("case,dtype,kernel,d_field,d_az,d_el,d_gamma,d_dist\n");
    const cxd s[4] = {{1.0, 0.0}, {0.0, 1.0}, {-2.0, 0.0}, {0.0, -1.0}};

    auto check = [&s](auto dtype_tag, bool tilt)
    {
        using dtype = decltype(dtype_tag);
        const auto ant = two_sample_ant<dtype>(tilt, s[0], s[1], s[2], s[3]);
        const double shift = tilt ? pi / 16.0 : pi / 8.0, span = tilt ? 0.5 * pi : pi;
        arma::Cube<dtype> ori(3, 1, 1, arma::fill::zeros);
        ori(tilt ? 1 : 2) = (dtype)(tilt ? shift : -shift);
        arma::Mat<dtype> az(1, 4, arma::fill::zeros), el(1, 4, arma::fill::zeros);
        std::vector<qd_expect> x(4);
        for (arma::uword a = 0; a < 4; ++a)
        {
            const double ang = (double)a * span / 4.0, t = (ang + shift) / span;
            (tilt ? el : az)(a) = (dtype)ang;
            x[a] = {two_sample_oracle(s[0], s[1], t), two_sample_oracle(s[2], s[3], t), tilt ? 0.0 : ang + shift, tilt ? ang + shift : 0.0, 0.0, 0.0};
        }
        for (int kernel : kernels())
            check_deviation<dtype>(deviation(run(ant, az, el, nullptr, &ori, nullptr, kernel), x, 1, 4), kernel, tilt ? "tilt" : "heading");
    };
    for (bool tilt : {false, true})
        check(0.0f, tilt), check(0.0, tilt);
}

TEST_CASE("arrayant_interpolate - bank turns the polarization")
{
    // Vertically polarized element (e-theta = 1 + 1j at azimuth 0), banked by +pi/4 and -pi/4 (two outputs of the same
    // element, per-element orientation): the field turns by gamma = +pi/4 and -pi/4 into the horizontal component
    auto check = [](auto dtype_tag)
    {
        using dtype = decltype(dtype_tag);
        const auto ant = two_sample_ant<dtype>(false, {1.0, 1.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0});
        const arma::Mat<dtype> az(1, 1, arma::fill::zeros), el(1, 1, arma::fill::zeros);
        const arma::uvec i_element = {0, 0};
        arma::Cube<dtype> ori(3, 2, 1, arma::fill::zeros);
        ori(0, 0, 0) = (dtype)(0.25 * pi), ori(0, 1, 0) = (dtype)(-0.25 * pi);
        const double rs2 = 1.0 / std::sqrt(2.0);
        std::vector<qd_expect> x = {{{rs2, rs2}, {rs2, rs2}, 0.0, 0.0, 0.25 * pi, 0.0},
                                    {{rs2, rs2}, {-rs2, -rs2}, 0.0, 0.0, -0.25 * pi, 0.0}};
        for (int kernel : kernels())
            check_deviation<dtype>(deviation(run(ant, az, el, &i_element, &ori, nullptr, kernel), x, 2, 1), kernel, "bank");
    };
    check(0.0f);
    check(0.0);
}

TEST_CASE("arrayant_interpolate - every orientation shape matches the rotated-frame oracle")
{
    QD_DIAG("\n# diag: rotated antenna, largest deviation from the rotated-frame oracle\n");
    QD_DIAG("case,dtype,kernel,d_field,d_az,d_el,d_gamma,d_dist\n");

    auto check = [](auto dtype_tag, arma::uword n_ori_cols, arma::uword n_ori_slices, arma::uword n_ang_rows, const char *what)
    {
        using dtype = decltype(dtype_tag);
        const auto ant = smooth_ant<dtype>();
        const arma::uvec i_element = {2, 0, 1, 0}; // Permuted, with a duplicate
        const arma::uword n_out = 4, n_ang = 20;
        qd_uniform rnd(23);
        const auto az = random_angles<dtype>(rnd, n_ang_rows, n_ang, -90.0 * pi / 180.0, 90.0 * pi / 180.0);
        const auto el = random_angles<dtype>(rnd, n_ang_rows, n_ang, -30.0 * pi / 180.0, 30.0 * pi / 180.0);
        arma::Cube<dtype> ori(3, n_ori_cols, n_ori_slices);
        for (auto &v : ori)
            v = (dtype)rnd(-0.35, 0.35);
        const arma::Mat<dtype> pos = {{0.3, -0.1, 0.0, 0.2}, {0.1, 0.2, -0.3, 0.0}, {-0.2, 0.0, 0.1, 0.4}};
        const auto x = smooth_expect(az, el, i_element, ori, pos);
        for (int kernel : kernels())
            check_deviation<dtype>(deviation(run(ant, az, el, &i_element, &ori, &pos, kernel), x, n_out, n_ang), kernel, what);
    };

    SECTION("[3, 1, 1], planar angles")
    {
        check(0.0f, 1, 1, 1, "[3,1,1]"), check(0.0, 1, 1, 1, "[3,1,1]");
    }
    SECTION("[3, n_out, 1], per-element angles")
    {
        check(0.0f, 4, 1, 4, "[3,n_out,1]"), check(0.0, 4, 1, 4, "[3,n_out,1]");
    }
    SECTION("[3, 1, n_ang], planar angles")
    {
        check(0.0f, 1, 20, 1, "[3,1,n_ang]"), check(0.0, 1, 20, 1, "[3,1,n_ang]");
    }
    SECTION("[3, n_out, n_ang], per-element angles")
    {
        check(0.0f, 4, 20, 4, "[3,n_out,n_ang]"), check(0.0, 4, 20, 4, "[3,n_out,n_ang]");
    }
}

TEST_CASE("arrayant_interpolate - the projected distance follows the element positions")
{
    // Three outputs of one element at the unit positions x, y, z; the distance is minus the projection of the
    // position onto the direction of arrival (hand values)
    auto check = [](auto dtype_tag)
    {
        using dtype = decltype(dtype_tag);
        quadriga_lib::arrayant<dtype> ant;
        ant.e_theta_re.ones(1, 1, 1), ant.e_theta_im.zeros(1, 1, 1), ant.e_phi_re.zeros(1, 1, 1), ant.e_phi_im.zeros(1, 1, 1);
        ant.azimuth_grid = {(dtype)0.0}, ant.elevation_grid = {(dtype)0.0};
        const arma::uvec i_element = {0, 0, 0};
        arma::Mat<dtype> pos(3, 3, arma::fill::eye);
        const double rs2 = 1.0 / std::sqrt(2.0);

        struct qd_dir
        {
            double az, el, dx, dy, dz;
        };
        const qd_dir dirs[4] = {{0.0, 0.0, -1.0, 0.0, 0.0},
                                {0.75 * pi, 0.0, rs2, -rs2, 0.0},
                                {0.0, -0.25 * pi, -rs2, 0.0, rs2},
                                {0.0, -0.5 * pi, 0.0, 0.0, 1.0}};
        for (int kernel : kernels())
            for (const auto &d : dirs)
            {
                INFO("kernel " << kernel << ", az " << d.az << ", el " << d.el);
                const arma::Mat<dtype> az(1, 1, arma::fill::value((dtype)d.az)), el(1, 1, arma::fill::value((dtype)d.el));
                const auto r = run(ant, az, el, &i_element, nullptr, &pos, kernel);
                const arma::vec expect = {d.dx, d.dy, d.dz};
                CHECK(arma::approx_equal(arma::conv_to<arma::vec>::from(r.dist), expect, "absdiff", tol_angle<dtype>(kernel)));
            }
    };
    check(0.0f);
    check(0.0);
}

TEST_CASE("arrayant_interpolate - element indices: omitted, permuted and repeated")
{
    auto check = [](auto dtype_tag)
    {
        using dtype = decltype(dtype_tag);
        const auto ant = smooth_ant<dtype>();
        qd_uniform rnd(5);
        const auto az = random_angles<dtype>(rnd, 1, 12, -2.5, 2.5), el = random_angles<dtype>(rnd, 1, 12, -1.0, 1.0);
        const arma::Cube<dtype> no_rotation(3, 1, 1, arma::fill::zeros);
        for (int kernel : kernels())
        {
            INFO("kernel " << kernel);
            // Omitted (nullptr or empty): all elements in order, positions of the arrayant
            const arma::uvec i_all = {0, 1, 2}, i_empty;
            const auto r_null = run(ant, az, el, nullptr, nullptr, nullptr, kernel);
            const auto r_empty = run(ant, az, el, &i_empty, nullptr, nullptr, kernel);
            CHECK(identical(r_null, r_empty));
            auto d = deviation(r_null, smooth_expect(az, el, i_all, no_rotation, ant.element_pos), 3, 12);
            check_deviation<dtype>(d, kernel, "omitted");

            // Permuted and repeated: rows follow the indices, positions are the columns i_element of ant.element_pos
            const arma::uvec i_perm = {2, 0, 2, 1, 2};
            const arma::Mat<dtype> pos_perm = ant.element_pos.cols(i_perm);
            d = deviation(run(ant, az, el, &i_perm, nullptr, nullptr, kernel), smooth_expect(az, el, i_perm, no_rotation, pos_perm), 5, 12);
            check_deviation<dtype>(d, kernel, "permuted");
        }
    };
    check(0.0f);
    check(0.0);
}

TEST_CASE("arrayant_interpolate - element positions: given, from the arrayant, or zero")
{
    auto check = [](auto dtype_tag)
    {
        using dtype = decltype(dtype_tag);
        auto ant = smooth_ant<dtype>();
        qd_uniform rnd(7);
        const auto az = random_angles<dtype>(rnd, 1, 9, -2.5, 2.5), el = random_angles<dtype>(rnd, 1, 9, -1.0, 1.0);
        const arma::Cube<dtype> no_rotation(3, 1, 1, arma::fill::zeros);
        const arma::uvec i_element = {1, 2};
        const arma::Mat<dtype> pos = {{0.5, -0.25}, {0.125, 0.75}, {-0.375, 0.0625}}, pos_empty;
        for (int kernel : kernels())
        {
            INFO("kernel " << kernel);
            // Given positions replace those of the arrayant
            auto r = run(ant, az, el, &i_element, nullptr, &pos, kernel);
            check_deviation<dtype>(deviation(r, smooth_expect(az, el, i_element, no_rotation, pos), 2, 9), kernel, "pos given");

            // Omitted (nullptr or empty): the columns i_element of ant.element_pos
            r = run(ant, az, el, &i_element, nullptr, nullptr, kernel);
            const arma::Mat<dtype> pos_ant = ant.element_pos.cols(i_element);
            check_deviation<dtype>(deviation(r, smooth_expect(az, el, i_element, no_rotation, pos_ant), 2, 9), kernel, "pos of ant");
            CHECK(identical(r, run(ant, az, el, &i_element, nullptr, &pos_empty, kernel)));

            // Omitted with an empty ant.element_pos: all elements at the origin, zero distance
            auto ant_no_pos = ant;
            ant_no_pos.element_pos.reset();
            r = run(ant_no_pos, az, el, &i_element, nullptr, nullptr, kernel);
            CHECK(arma::all(arma::vectorise(r.dist) == (dtype)0.0));
            CHECK(identical(r.V_re, run(ant, az, el, &i_element, nullptr, nullptr, kernel).V_re));
        }
    };
    check(0.0f);
    check(0.0);
}

TEST_CASE("arrayant_interpolate - empty angle input gives empty outputs")
{
    const auto ant = smooth_ant<double>();
    const arma::uvec i_element = {0, 2};
    for (int kernel : kernels())
        for (arma::uword n_rows : {1, 2})
        {
            INFO("kernel " << kernel << ", rows " << n_rows);
            const arma::mat az(n_rows, 0), el(n_rows, 0);
            qd_result<double> r;
            for (arma::mat *m : {&r.V_re, &r.V_im, &r.H_re, &r.H_im, &r.dist, &r.az_loc, &r.el_loc, &r.gamma})
                m->ones(3, 3); // Stale content of another size
            quadriga_lib::arrayant_interpolate<double>(ant, az, el, r.V_re, r.V_im, r.H_re, r.H_im, &i_element, nullptr, nullptr,
                                               &r.dist, &r.az_loc, &r.el_loc, &r.gamma, kernel);
            for (arma::mat *m : {&r.V_re, &r.V_im, &r.H_re, &r.H_im, &r.dist, &r.az_loc, &r.el_loc, &r.gamma})
            {
                CHECK(m->n_rows == 2);
                CHECK(m->n_cols == 0);
            }
        }
}

TEST_CASE("arrayant_interpolate - optional outputs alone and together give the same values")
{
    const auto ant = smooth_ant<double>();
    qd_uniform rnd(3);
    const auto az = random_angles<double>(rnd, 3, 10, -2.5, 2.5), el = random_angles<double>(rnd, 3, 10, -1.0, 1.0);
    arma::cube ori(3, 1, 1);
    ori(0) = 0.2, ori(1) = -0.1, ori(2) = 0.3;
    for (int kernel : kernels())
    {
        INFO("kernel " << kernel);
        const auto all = run(ant, az, el, nullptr, &ori, nullptr, kernel);
        CHECK(arma::abs(all.V_re).max() > 0.1); // Not a trivial (zero) result
        CHECK(arma::abs(all.gamma).max() > 0.01);

        arma::mat V_re, V_im, H_re, H_im, x;
        quadriga_lib::arrayant_interpolate<double>(ant, az, el, V_re, V_im, H_re, H_im, nullptr, &ori, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, kernel);
        CHECK((identical(V_re, all.V_re) && identical(V_im, all.V_im) && identical(H_re, all.H_re) && identical(H_im, all.H_im)));

        quadriga_lib::arrayant_interpolate<double>(ant, az, el, V_re, V_im, H_re, H_im, nullptr, &ori, nullptr, &x, nullptr, nullptr, nullptr, kernel);
        CHECK(identical(x, all.dist));
        quadriga_lib::arrayant_interpolate<double>(ant, az, el, V_re, V_im, H_re, H_im, nullptr, &ori, nullptr, nullptr, &x, nullptr, nullptr, kernel);
        CHECK(identical(x, all.az_loc));
        quadriga_lib::arrayant_interpolate<double>(ant, az, el, V_re, V_im, H_re, H_im, nullptr, &ori, nullptr, nullptr, nullptr, &x, nullptr, kernel);
        CHECK(identical(x, all.el_loc));
        quadriga_lib::arrayant_interpolate<double>(ant, az, el, V_re, V_im, H_re, H_im, nullptr, &ori, nullptr, nullptr, nullptr, nullptr, &x, kernel);
        CHECK(identical(x, all.gamma));
        CHECK((identical(V_re, all.V_re) && identical(H_im, all.H_im)));
    }
}

TEST_CASE("arrayant_interpolate - pre-sized outputs are written in place, others are resized and overwritten")
{
    const auto ant = smooth_ant<double>();
    qd_uniform rnd(9);
    const auto az = random_angles<double>(rnd, 1, 6, -2.5, 2.5), el = random_angles<double>(rnd, 1, 6, -1.0, 1.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int kernel : kernels())
    {
        INFO("kernel " << kernel);
        const auto ref = run(ant, az, el, nullptr, nullptr, nullptr, kernel);
        CHECK(arma::abs(ref.V_re).max() > 0.1); // Not a trivial (zero) result

        // Correct size, stale content: written in place and completely
        qd_result<double> r;
        std::vector<arma::mat *> outs = {&r.V_re, &r.V_im, &r.H_re, &r.H_im, &r.dist, &r.az_loc, &r.el_loc, &r.gamma};
        std::vector<const double *> ptr;
        for (arma::mat *m : outs)
            m->set_size(3, 6), m->fill(nan), ptr.push_back(m->memptr());
        quadriga_lib::arrayant_interpolate<double>(ant, az, el, r.V_re, r.V_im, r.H_re, r.H_im, nullptr, nullptr, nullptr,
                                           &r.dist, &r.az_loc, &r.el_loc, &r.gamma, kernel);
        for (size_t i = 0; i < outs.size(); ++i)
            CHECK(outs[i]->memptr() == ptr[i]);
        CHECK(identical(r, ref));

        // Wrong size, stale content: resized and overwritten
        for (arma::mat *m : outs)
            m->set_size(2, 7), m->fill(nan);
        quadriga_lib::arrayant_interpolate<double>(ant, az, el, r.V_re, r.V_im, r.H_re, r.H_im, nullptr, nullptr, nullptr,
                                           &r.dist, &r.az_loc, &r.el_loc, &r.gamma, kernel);
        CHECK(identical(r, ref));
    }
}

TEST_CASE("arrayant_interpolate - NaN and infinite angles give NaN")
{
    // Non-finite angles give NaN in all outputs of that angle and element, without an exception; finite angles in the
    // same call are not affected
    auto check = [](auto dtype_tag)
    {
        using dtype = decltype(dtype_tag);
        const dtype nan = std::numeric_limits<dtype>::quiet_NaN(), inf = std::numeric_limits<dtype>::infinity();
        auto ant = quadriga_lib::generate_arrayant_xpol<dtype>(5.0); // 37 x 73 grid, 2 elements
        ant.element_pos = {{0.1, -0.1}, {0.2, 0.0}, {0.0, 0.3}};
        for (int kernel : kernels())
        {
            INFO("kernel " << kernel);
            // Planar angles: columns 1 to 6 are non-finite, columns 0 and 7 are finite
            const arma::Mat<dtype> az = {{0.3, nan, inf, -inf, 0.2, 0.2, 0.2, -2.0}};
            const arma::Mat<dtype> el = {{0.1, 0.0, 0.0, 0.0, nan, inf, -inf, 0.5}};
            const auto r = run(ant, az, el, nullptr, nullptr, nullptr, kernel);
            const arma::Mat<dtype> az_f = {{0.3, -2.0}}, el_f = {{0.1, 0.5}};
            const auto r_f = run(ant, az_f, el_f, nullptr, nullptr, nullptr, kernel);
            for (arma::uword o = 0; o < 2; ++o)
            {
                for (arma::uword a = 1; a < 7; ++a)
                    for (const arma::Mat<dtype> *m : {&r.V_re, &r.V_im, &r.H_re, &r.H_im, &r.dist, &r.az_loc, &r.el_loc, &r.gamma})
                        CHECK(std::isnan((*m)(o, a)));
                CHECK(std::abs((double)r.V_re(o, 0) - (double)r_f.V_re(o, 0)) < tol_field<dtype>(kernel));
                CHECK(std::abs((double)r.H_im(o, 7) - (double)r_f.H_im(o, 1)) < tol_field<dtype>(kernel));
            }

            // Per-element angles: non-finite angles for some elements only
            const arma::Mat<dtype> az2 = {{nan, 0.3, 0.1, inf}, {0.3, -inf, 0.2, 0.0}};
            const arma::Mat<dtype> el2 = {{0.1, 0.1, nan, 0.0}, {0.1, 0.1, 0.0, -inf}};
            const int bad[2][4] = {{1, 0, 1, 1}, {0, 1, 0, 1}};
            const auto r2 = run(ant, az2, el2, nullptr, nullptr, nullptr, kernel);
            for (arma::uword o = 0; o < 2; ++o)
                for (arma::uword a = 0; a < 4; ++a)
                    for (const arma::Mat<dtype> *m : {&r2.V_re, &r2.H_im, &r2.dist, &r2.gamma})
                        CHECK(std::isnan((*m)(o, a)) == (bad[o][a] == 1));
        }
    };
    check(0.0f);
    check(0.0);
}

TEST_CASE("arrayant_interpolate - angles outside the nominal ranges")
{
    // Azimuth shifted by +/- 2 pi gives the same direction and the same result. Elevation beyond +/- pi/2 by x
    // gives the direction (az + pi, +/- (pi/2 - x)), where both polarization basis vectors point the other way:
    // the fields change sign, gamma changes by pi, the local angles and the distance stay the same.
    const auto ant = smooth_ant<double>();
    qd_uniform rnd(17);
    arma::cube ori(3, 1, 1);
    ori(0) = 0.1, ori(1) = 0.2, ori(2) = -0.15;
    const arma::mat az = random_angles<double>(rnd, 1, 8, -1.2, 1.2);
    const arma::mat x = random_angles<double>(rnd, 1, 8, 0.5, 1.0);
    const arma::mat el_in = 0.5 * pi - x, el_out = 0.5 * pi + x;
    for (int kernel : kernels())
    {
        INFO("kernel " << kernel);
        const double tf = tol_field<double>(kernel), ta = tol_angle<double>(kernel);
        const auto r = run(ant, az, el_in, nullptr, &ori, nullptr, kernel);
        for (double shift : {2.0 * pi, -2.0 * pi})
        {
            const auto s = run(ant, arma::mat(az + shift), el_in, nullptr, &ori, nullptr, kernel);
            CHECK(arma::approx_equal(s.V_re, r.V_re, "absdiff", tf));
            CHECK(arma::approx_equal(s.H_im, r.H_im, "absdiff", tf));
            CHECK(arma::approx_equal(s.az_loc, r.az_loc, "absdiff", ta));
            CHECK(arma::approx_equal(s.gamma, r.gamma, "absdiff", ta));
        }
        for (double sign : {1.0, -1.0})
        {
            const auto r_in = run(ant, arma::mat(az + pi), arma::mat(sign * el_in), nullptr, &ori, nullptr, kernel);
            const auto r_out = run(ant, az, arma::mat(sign * el_out), nullptr, &ori, nullptr, kernel);
            CHECK(arma::approx_equal(r_out.V_re, arma::mat(-r_in.V_re), "absdiff", tf));
            CHECK(arma::approx_equal(r_out.V_im, arma::mat(-r_in.V_im), "absdiff", tf));
            CHECK(arma::approx_equal(r_out.H_re, arma::mat(-r_in.H_re), "absdiff", tf));
            CHECK(arma::approx_equal(r_out.H_im, arma::mat(-r_in.H_im), "absdiff", tf));
            CHECK(arma::approx_equal(r_out.az_loc, r_in.az_loc, "absdiff", ta));
            CHECK(arma::approx_equal(r_out.el_loc, r_in.el_loc, "absdiff", ta));
            CHECK(arma::approx_equal(r_out.dist, r_in.dist, "absdiff", ta));
            for (arma::uword i = 0; i < r_in.gamma.n_elem; ++i)
                CHECK(std::abs(wrap(r_out.gamma(i) - r_in.gamma(i) - pi)) < ta);
        }
    }
}

TEST_CASE("arrayant_interpolate - GENERIC and AVX2 agree on a 3GPP panel")
{
    // A dual-polarized 2 x 2 panel (8 elements, 1 degree grid) with 11 outputs (crossing the AVX2 vector width of 8
    // with a tail of 3), random angles with the elevation inside +/- (90 - 0.01) degree and random orientation. The
    // sections run the three AVX2 code paths: per element and angle (per-element angles or per-angle orientation),
    // vectorized over elements (planar angles, 8 or more outputs) and vectorized over angles (planar angles, fewer
    // than 8 outputs); their sizes exceed the OpenMP limits of both kernels. GENERIC with double is the reference;
    // tolerances from single-precision arithmetic, see compare_single_precision.
    if (!quadriga_lib::quadriga_lib_has_AVX2())
        return;

    auto check = [](auto dtype_tag, arma::uword n_out, arma::uword n_ang_rows, arma::uword n_ang, arma::uword n_ori_slices, const char *what)
    {
        using dtype = decltype(dtype_tag);
        const auto ant_gen = quadriga_lib::generate_arrayant_3GPP<double>(2, 2, 3.0e9, 3);
        const auto ant = convert<dtype>(ant_gen);
        const auto ant_d = convert<double>(ant); // Reference on the samples the dtype run sees
        double M, S;
        pattern_scale(ant_d, M, S);
        const double pos_max = arma::abs(ant_d.element_pos).max();

        const arma::uvec i_full = {7, 0, 3, 3, 5, 1, 6, 2, 4, 0, 7};
        const arma::uvec i_element = i_full.head(n_out);
        qd_uniform rnd(31);
        const double el_max = (90.0 - 0.01) * pi / 180.0;
        const arma::mat az_d = random_angles<double>(rnd, n_ang_rows, n_ang, -pi, pi);
        const arma::mat el_d = random_angles<double>(rnd, n_ang_rows, n_ang, -el_max, el_max);
        arma::cube ori_d(3, n_out, n_ori_slices);
        for (auto &v : ori_d)
            v = rnd(-pi, pi);
        const arma::Mat<dtype> az = arma::conv_to<arma::Mat<dtype>>::from(az_d), el = arma::conv_to<arma::Mat<dtype>>::from(el_d);
        const arma::Cube<dtype> ori = arma::conv_to<arma::Cube<dtype>>::from(ori_d);

        // Reference from the inputs as the dtype run sees them
        const arma::cube ori_ref = arma::conv_to<arma::cube>::from(ori);
        const auto ref = run(ant_d, arma::conv_to<arma::mat>::from(az), arma::conv_to<arma::mat>::from(el), &i_element,
                             &ori_ref, nullptr, 1);
        const auto avx = run(ant, az, el, &i_element, &ori, nullptr, 2);
        CHECK(compare_single_precision(avx, ref, M, S, pos_max, what) < 1.0);
    };

    QD_DIAG("\n# diag: AVX2 against GENERIC (double), 3GPP panel; d_* = largest deviation, q_* = largest deviation / tolerance\n");
    QD_DIAG("path,dtype,d_field,q_field,d_angle,q_angle,d_dist,q_dist\n");
    SECTION("per element and angle, per-element angles")
    {
        check(0.0f, 11, 11, 100, 1, "per-element angles"), check(0.0, 11, 11, 100, 1, "per-element angles");
    }
    SECTION("per element and angle, per-angle orientation")
    {
        check(0.0f, 11, 1, 100, 100, "per-angle orientation"), check(0.0, 11, 1, 100, 100, "per-angle orientation");
    }
    SECTION("vectorized over elements")
    {
        check(0.0f, 11, 1, 100, 1, "over elements"), check(0.0, 11, 1, 100, 1, "over elements");
    }
    SECTION("vectorized over angles")
    {
        check(0.0f, 3, 1, 150, 1, "over angles"), check(0.0, 3, 1, 150, 1, "over angles");
    }
}

TEST_CASE("arrayant_interpolate - AVX2 accuracy at fixed directions and at the poles")
{
    // Fixed directions where single-precision arithmetic is weakest: the poles of the global frame (zenith, nadir),
    // the pole of the local frame (antenna tilted by 90 degree) and 1e-3 rad next to it, the azimuth wrap, grid points
    // and midpoints, and a pattern whose grid is built in single precision. Reference: GENERIC with double on the same
    // double inputs. Each case runs through the three AVX2 code paths (8 outputs: vectorized over elements; 3 outputs:
    // vectorized over angles; per-element angles: per element and angle), with float and double. Tolerances: 3 to 5
    // times the largest deviation measured on an AVX2 CPU, at least about one single-precision rounding (1e-7 for
    // field and power, 2e-7 rad for angles); field and power (|V|^2 + |H|^2) relative to the peak amplitude M and M^2
    // of the pattern, angles in rad, distance in m (element positions up to 0.4 m). At the exact local pole the local
    // azimuth is undefined (inf: not compared), so the field components depend on the azimuth each kernel picks; the
    // power does not. Near the local pole the azimuth error grows as about 2e-7 rad divided by the distance from it.
    if (!quadriga_lib::quadriga_lib_has_AVX2())
        return;

    const double inf = std::numeric_limits<double>::infinity(), deg = pi / 180.0;
    std::vector<double> az_gen, el_gen, az_8, el_pole, el_near, az_grid, el_grid, az_fgrid, el_fgrid;
    for (int i = 0; i < 16; ++i)
        az_gen.push_back((-170.0 + 21.3 * i) * deg), el_gen.push_back((-75.0 + 9.7 * i) * deg);
    for (int i = 0; i < 8; ++i)
    {
        const double sgn = i % 2 ? -1.0 : 1.0, half = i % 2 ? 0.5 : 0.0;
        az_8.push_back((-170.0 + 45.0 * i) * deg), el_pole.push_back(sgn * 0.5 * pi), el_near.push_back(sgn * (0.5 * pi - 1e-3));
        az_grid.push_back((-150.0 + 37.0 * i + half) * deg), el_grid.push_back((-60.0 + 15.0 * i + half) * deg);
    }
    for (int k = -20; k <= 20; ++k)
        az_fgrid.push_back(89.0 * deg + 2.5e-6 * k), el_fgrid.push_back(0.1);

    // Patterns: 0 = 3GPP element, 1 = the same element generated in float, 2 = cross-polarized isotropic radiator,
    // 3 = half-wave dipole; all on the default 1 degree grid, in both precisions
    const quadriga_lib::arrayant<double> patterns_d[4] = {quadriga_lib::generate_arrayant_3GPP<double>(),
                                                          convert<double>(quadriga_lib::generate_arrayant_3GPP<float>()),
                                                          quadriga_lib::generate_arrayant_xpol<double>(),
                                                          quadriga_lib::generate_arrayant_half_wave_dipole<double>()};
    const quadriga_lib::arrayant<float> patterns_f[4] = {convert<float>(patterns_d[0]), convert<float>(patterns_d[1]),
                                                         convert<float>(patterns_d[2]), convert<float>(patterns_d[3])};
    struct qd_case
    {
        const char *name;
        int pattern;
        std::vector<double> az, el;
        double bank, tilt, heading;
        double t_field, t_power, t_az, t_el, t_gamma, t_dist;
    };
    const qd_case cases[] = {
        {"generic", 0, az_gen, el_gen, 0.3, -0.2, 0.7, 1e-6, 1e-6, 4e-6, 6e-7, 4e-6, 6e-7},
        {"global poles", 2, az_8, el_pole, 0.0, 0.0, 0.0, 5e-7, 1e-6, 5e-7, 2e-7, 2e-7, 1.5e-7},
        {"local pole", 2, {0.0, pi, 0.0, pi}, {0.0, 0.0, 0.0, 0.0}, 0.0, -0.5 * pi, 0.0, inf, 2e-6, inf, 2e-7, inf, 3e-7},
        {"1e-3 from the local pole", 2, {0.0, pi, 0.0, pi}, {1e-3, 1e-3, -1e-3, -1e-3}, 0.0, -0.5 * pi, 0.0, 3e-4, 2e-6, 3e-4, 1e-4, 3e-4, 3e-7},
        {"1e-3 from the poles, dipole", 3, az_8, el_near, 0.0, 0.0, 0.0, 7e-5, 1.5e-7, 5e-7, 1e-4, 2e-7, 3e-7},
        {"azimuth wrap", 0, {pi - 1e-6, -pi + 1e-6, pi - 0.3 * deg, -pi + 0.3 * deg, pi}, {0.1, -0.2, 0.3, 0.0, 0.5}, 0.0, 0.0, 0.0, 1e-7, 1e-7, 4e-7, 2e-7, 2e-7, 5e-7},
        {"grid points and midpoints", 0, az_grid, el_grid, 0.0, 0.0, 0.0, 3e-6, 5e-6, 3e-7, 4e-7, 2e-7, 5e-7},
        {"grid built in float", 1, az_fgrid, el_fgrid, 0.0, 0.0, 0.0, 2e-7, 1e-7, 2e-7, 2e-7, 2e-7, 4e-7}};

    QD_DIAG("\n# diag: AVX2 at fixed directions against GENERIC (double); d_field / M, d_power / M^2, angles in rad, dist in m\n");
    QD_DIAG("case,path,dtype,d_field,d_power,d_az,d_el,d_gamma,d_dist\n");
    auto check = [](const auto &ant, const qd_case &c, const quadriga_lib::arrayant<double> &ant_d)
    {
        using dtype = typename std::decay<decltype(ant.e_theta_re)>::type::elem_type;
        double M = 0.0;
        for (arma::uword i = 0; i < ant_d.e_theta_re.n_elem; ++i)
            M = std::max({M, std::hypot(ant_d.e_theta_re(i), ant_d.e_theta_im(i)), std::hypot(ant_d.e_phi_re(i), ant_d.e_phi_im(i))});
        const arma::uword n_ang = c.az.size();
        arma::cube ori_d(3, 1, 1);
        ori_d(0) = c.bank, ori_d(1) = c.tilt, ori_d(2) = c.heading;
        const arma::Cube<dtype> ori = arma::conv_to<arma::Cube<dtype>>::from(ori_d);

        struct qd_path
        {
            const char *name;
            arma::uword n_out, n_rows;
        };
        const qd_path paths[3] = {{"over elements", 8, 1}, {"over angles", 3, 1}, {"per element and angle", 3, 3}};
        for (const auto &p : paths)
        {
            arma::uvec i_element(p.n_out);
            for (arma::uword o = 0; o < p.n_out; ++o)
                i_element(o) = o % ant_d.e_theta_re.n_slices;
            arma::mat az_d(p.n_rows, n_ang), el_d(p.n_rows, n_ang), pos_d(3, p.n_out);
            for (arma::uword a = 0; a < n_ang; ++a)
                az_d.col(a).fill(c.az[a]), el_d.col(a).fill(c.el[a]);
            for (arma::uword o = 0; o < p.n_out; ++o)
                pos_d(0, o) = 0.1 * (double)o - 0.3, pos_d(1, o) = 0.2 - 0.05 * (double)o, pos_d(2, o) = 0.04 * (double)o;
            const arma::Mat<dtype> pos = arma::conv_to<arma::Mat<dtype>>::from(pos_d);
            const auto ref = run(ant_d, az_d, el_d, &i_element, &ori_d, &pos_d, 1);
            const auto r = run(ant, arma::conv_to<arma::Mat<dtype>>::from(az_d), arma::conv_to<arma::Mat<dtype>>::from(el_d), &i_element, &ori, &pos, 2);

            double d_field = 0.0, d_power = 0.0, d_az = 0.0, d_el = 0.0, d_gamma = 0.0, d_dist = 0.0;
            for (arma::uword i = 0; i < ref.V_re.n_elem; ++i)
            {
                const cxd V((double)r.V_re(i), (double)r.V_im(i)), H((double)r.H_re(i), (double)r.H_im(i));
                const cxd Vr(ref.V_re(i), ref.V_im(i)), Hr(ref.H_re(i), ref.H_im(i));
                d_field = std::max({d_field, std::abs(V - Vr) / M, std::abs(H - Hr) / M});
                d_power = std::max(d_power, std::abs(std::norm(V) + std::norm(H) - std::norm(Vr) - std::norm(Hr)) / (M * M));
                d_az = std::max(d_az, std::abs(wrap((double)r.az_loc(i) - ref.az_loc(i))));
                d_el = std::max(d_el, std::abs((double)r.el_loc(i) - ref.el_loc(i)));
                d_gamma = std::max(d_gamma, std::abs(wrap((double)r.gamma(i) - ref.gamma(i))));
                d_dist = std::max(d_dist, std::abs((double)r.dist(i) - ref.dist(i)));
            }
            QD_DIAG("%s,%s,%s,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e\n", c.name, p.name, std::is_same<dtype, float>::value ? "float" : "double",
                    d_field, d_power, d_az, d_el, d_gamma, d_dist);
            INFO(c.name << ", " << p.name << ", " << (std::is_same<dtype, float>::value ? "float" : "double"));
            CHECK(d_power < c.t_power); // Also where the fields are not compared
            CHECK((std::isinf(c.t_field) || d_field < c.t_field));
            CHECK((std::isinf(c.t_az) || d_az < c.t_az));
            CHECK(d_el < c.t_el);
            CHECK((std::isinf(c.t_gamma) || d_gamma < c.t_gamma));
            CHECK(d_dist < c.t_dist);
        }
    };
    for (const auto &c : cases)
    {
        check(patterns_f[c.pattern], c, patterns_d[c.pattern]);
        check(patterns_d[c.pattern], c, patterns_d[c.pattern]);
    }
}

TEST_CASE("arrayant_interpolate - float and double agree")
{
    // GENERIC with float against GENERIC with double on the same samples of the 3GPP panel; tolerances as in the
    // kernel comparison
    QD_DIAG("\n# diag: GENERIC float against GENERIC double, 3GPP panel\n");
    QD_DIAG("path,dtype,d_field,q_field,d_angle,q_angle,d_dist,q_dist\n");
    const auto ant_f = convert<float>(quadriga_lib::generate_arrayant_3GPP<double>(2, 2, 3.0e9, 3));
    const auto ant_d = convert<double>(ant_f); // Same samples in both precisions
    double M, S;
    pattern_scale(ant_d, M, S);
    qd_uniform rnd(37);
    const double el_max = (90.0 - 0.01) * pi / 180.0;
    const arma::fmat az = random_angles<float>(rnd, 8, 40, -pi, pi), el = random_angles<float>(rnd, 8, 40, -el_max, el_max);
    arma::fcube ori(3, 8, 1);
    for (auto &v : ori)
        v = (float)rnd(-pi, pi);
    const arma::cube ori_d = arma::conv_to<arma::cube>::from(ori);
    const auto r_f = run(ant_f, az, el, nullptr, &ori, nullptr, 1);
    const auto r_d = run(ant_d, arma::conv_to<arma::mat>::from(az), arma::conv_to<arma::mat>::from(el), nullptr, &ori_d, nullptr, 1);
    CHECK(compare_single_precision(r_f, r_d, M, S, arma::abs(ant_d.element_pos).max(), "GENERIC") < 1.0);
}

TEST_CASE("arrayant_interpolate - repeated calls are identical and a large call equals single-angle calls")
{
    // Large enough for the OpenMP regions of both kernels; for AVX2 the orientation does not vary per angle, since
    // slicing a per-angle orientation to one angle selects another AVX2 code path
    const auto ant = quadriga_lib::generate_arrayant_3GPP<double>(2, 2, 3.0e9, 3, 0.0, 0.5, 1, 1, 0.5, 0.5, nullptr, 5.0);
    const arma::uvec i_element = {7, 0, 3, 3, 5, 1, 6, 2, 4, 0, 7};
    const arma::uword n_ang = 300;
    qd_uniform rnd(41);
    for (int kernel : kernels())
        for (arma::uword n_rows : {1, 11})
        {
            INFO("kernel " << kernel << ", angle rows " << n_rows);
            const auto az = random_angles<double>(rnd, n_rows, n_ang, -pi, pi);
            const auto el = random_angles<double>(rnd, n_rows, n_ang, -1.5, 1.5);
            arma::cube ori(3, 11, 1);
            for (auto &v : ori)
                v = rnd(-1.0, 1.0);
            const auto r1 = run(ant, az, el, &i_element, &ori, nullptr, kernel);
            const auto r2 = run(ant, az, el, &i_element, &ori, nullptr, kernel);
            CHECK(arma::abs(r1.V_re).max() > 0.1); // Not a trivial (zero) result
            CHECK(identical(r1, r2));

            qd_result<double> cat;
            for (arma::mat *m : {&cat.V_re, &cat.V_im, &cat.H_re, &cat.H_im, &cat.dist, &cat.az_loc, &cat.el_loc, &cat.gamma})
                m->set_size(11, n_ang);
            for (arma::uword a = 0; a < n_ang; ++a)
            {
                const auto s = run(ant, arma::mat(az.col(a)), arma::mat(el.col(a)), &i_element, &ori, nullptr, kernel);
                cat.V_re.col(a) = s.V_re, cat.V_im.col(a) = s.V_im, cat.H_re.col(a) = s.H_re, cat.H_im.col(a) = s.H_im;
                cat.dist.col(a) = s.dist, cat.az_loc.col(a) = s.az_loc, cat.el_loc.col(a) = s.el_loc, cat.gamma.col(a) = s.gamma;
            }
            CHECK(identical(r1, cat));
        }

    // GENERIC with a per-angle orientation: slicing selects no other code path
    const auto az = random_angles<double>(rnd, 1, n_ang, -pi, pi), el = random_angles<double>(rnd, 1, n_ang, -1.5, 1.5);
    arma::cube ori(3, 11, n_ang);
    for (auto &v : ori)
        v = rnd(-1.0, 1.0);
    const auto r1 = run(ant, az, el, &i_element, &ori, nullptr, 1);
    bool all_equal = true;
    for (arma::uword a = 0; a < n_ang; ++a)
    {
        arma::cube ori_a(3, 11, 1);
        ori_a.slice(0) = ori.slice(a);
        const auto s = run(ant, arma::mat(az.col(a)), arma::mat(el.col(a)), &i_element, &ori_a, nullptr, 1);
        all_equal = all_equal && identical(arma::mat(r1.V_re.col(a)), s.V_re) && identical(arma::mat(r1.gamma.col(a)), s.gamma);
    }
    CHECK(all_equal);
}

TEST_CASE("arrayant_interpolate - use_kernel selects the kernel")
{
    const auto ant = smooth_ant<double>();
    qd_uniform rnd(43);
    const auto az = random_angles<double>(rnd, 1, 20, -2.5, 2.5), el = random_angles<double>(rnd, 1, 20, -1.0, 1.0);
    const auto r0 = run(ant, az, el, nullptr, nullptr, nullptr, 0);
    const auto r1 = run(ant, az, el, nullptr, nullptr, nullptr, 1);
    CHECK(identical(r0, r1)); // Auto selects GENERIC

    if (quadriga_lib::quadriga_lib_has_AVX2())
    {
        const auto r2 = run(ant, az, el, nullptr, nullptr, nullptr, 2);
        CHECK(!identical(r2.V_re, r1.V_re)); // AVX2 computes in single precision
    }
    else
    {
        CHECK_THROWS_AS(run(ant, az, el, nullptr, nullptr, nullptr, 2), std::invalid_argument);
    }
}

TEST_CASE("arrayant_interpolate - input validation")
{
    auto ant = smooth_ant<double>();
    const arma::mat az = {{0.1, 0.2, 0.3, 0.4}}, el = {{0.0, 0.1, -0.1, 0.2}};
    const arma::uvec i_element = {2, 0, 1};
    const arma::cube ori(3, 3, 4, arma::fill::zeros);
    const arma::mat pos(3, 3, arma::fill::zeros);
    arma::mat V_re, V_im, H_re, H_im, dist, az_loc, el_loc, gamma;

    CHECK_NOTHROW(quadriga_lib::arrayant_interpolate<double>(ant, az, el, V_re, V_im, H_re, H_im, &i_element, &ori, &pos,
                                                     &dist, &az_loc, &el_loc, &gamma, 1));

    // An invalid call leaves the outputs untouched
    V_re.fill(7.0);
    const double *V_re_ptr = V_re.memptr();
    auto call = [&](const quadriga_lib::arrayant<double> &a, const arma::mat &azi, const arma::mat &ele,
                    const arma::uvec *ie, const arma::cube *o, const arma::mat *p, int k)
    {
        quadriga_lib::arrayant_interpolate<double>(a, azi, ele, V_re, V_im, H_re, H_im, ie, o, p, &dist, &az_loc, &el_loc, &gamma, k);
    };

    SECTION("ant")
    {
        auto bad = ant;
        bad.e_theta_im.set_size(13, 18, 3);
        CHECK_THROWS_AS(call(bad, az, el, &i_element, &ori, &pos, 1), std::invalid_argument);
    }
    SECTION("azimuth")
    {
        CHECK_THROWS_AS(call(ant, arma::mat(0, 4), arma::mat(0, 4), &i_element, &ori, &pos, 1), std::invalid_argument);
        CHECK_THROWS_AS(call(ant, arma::mat(2, 4), arma::mat(2, 4), &i_element, &ori, &pos, 1), std::invalid_argument);
    }
    SECTION("elevation")
    {
        CHECK_THROWS_AS(call(ant, az, arma::mat(1, 3), &i_element, &ori, &pos, 1), std::invalid_argument);
        CHECK_THROWS_AS(call(ant, az, arma::mat(3, 4), &i_element, &ori, &pos, 1), std::invalid_argument);
    }
    SECTION("i_element")
    {
        const arma::uvec bad = {2, 3, 1};
        CHECK_THROWS_AS(call(ant, az, el, &bad, &ori, &pos, 1), std::invalid_argument);
    }
    SECTION("orientation")
    {
        for (const arma::cube &bad : {arma::cube(2, 3, 4), arma::cube(3, 2, 4), arma::cube(3, 3, 3)})
            CHECK_THROWS_AS(call(ant, az, el, &i_element, &bad, &pos, 1), std::invalid_argument);
    }
    SECTION("element_pos")
    {
        for (const arma::mat &bad : {arma::mat(2, 3), arma::mat(3, 2)})
            CHECK_THROWS_AS(call(ant, az, el, &i_element, &ori, &bad, 1), std::invalid_argument);
    }
    SECTION("use_kernel")
    {
        CHECK_THROWS_AS(call(ant, az, el, &i_element, &ori, &pos, -1), std::invalid_argument);
        CHECK_THROWS_AS(call(ant, az, el, &i_element, &ori, &pos, 3), std::invalid_argument);
    }
    SECTION("an output is an input or shares memory with one")
    {
        arma::mat az_copy = az;
        CHECK_THROWS_AS(quadriga_lib::arrayant_interpolate<double>(ant, az_copy, el, az_copy, V_im, H_re, H_im, &i_element), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::arrayant_interpolate<double>(ant, az_copy, el, V_re, V_im, H_re, H_im, &i_element, &ori, &pos,
                                                           nullptr, nullptr, &az_copy),
                        std::invalid_argument);
        arma::mat on_pos(const_cast<double *>(pos.memptr()), 3, 3, false, true);
        CHECK_THROWS_AS(quadriga_lib::arrayant_interpolate<double>(ant, az, el, on_pos, V_im, H_re, H_im, &i_element, &ori, &pos),
                        std::invalid_argument);
    }
    SECTION("an output is a data member of ant")
    {
        CHECK_THROWS_AS(quadriga_lib::arrayant_interpolate<double>(ant, az, el, ant.element_pos, V_im, H_re, H_im), std::invalid_argument);
        arma::mat on_pattern(ant.e_phi_im.memptr(), 3, 4, false, true);
        CHECK_THROWS_AS(quadriga_lib::arrayant_interpolate<double>(ant, az, el, V_re, V_im, H_re, H_im, nullptr, nullptr, nullptr,
                                                           nullptr, nullptr, nullptr, &on_pattern),
                        std::invalid_argument);
    }
    SECTION("two outputs are the same object")
    {
        CHECK_THROWS_AS(quadriga_lib::arrayant_interpolate<double>(ant, az, el, V_re, V_re, H_re, H_im), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::arrayant_interpolate<double>(ant, az, el, V_re, V_im, H_re, H_im, nullptr, nullptr, nullptr,
                                                           nullptr, &az_loc, nullptr, &az_loc),
                        std::invalid_argument);
    }

    CHECK(V_re.memptr() == V_re_ptr);
    CHECK(arma::all(arma::vectorise(V_re) == 7.0));
}
