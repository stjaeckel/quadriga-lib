// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <stdexcept>
#include <cstring>
#include <limits>

#include "quadriga_lib.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_arrayant.hpp"

// Cartesian unit vectors from azimuth and elevation angles [N], interleaved output [3, N]
template <typename dtype>
static inline void qd_geo2cart_interleaved(size_t N, dtype *cart_3xN, const dtype *az, const dtype *el)
{
    for (size_t i = 0; i < N; ++i)
    {
        const double ca = std::cos((double)az[i]), sa = std::sin((double)az[i]);
        const double ce = std::cos((double)el[i]), se = std::sin((double)el[i]);
        dtype *C = &cart_3xN[3 * i];
        C[0] = dtype(ce * ca), C[1] = dtype(ce * sa), C[2] = dtype(se);
    }
}

// Throws unless the center frequency is positive and finite
static void qd_check_center_freq(double center_freq)
{
    if (!std::isfinite(center_freq) || center_freq <= 0.0)
        throw std::invalid_argument("Input 'center_freq' must be positive and finite.");
}

// Resolved panel tuples of the TR 38.901 panel array: the fill rules of 'panel', 'spacing', 'ports', 'pol' and 'steer'
// in one place (see generate_antenna); 'steer' in degree with one row per panel, [Mg Ng, 2]
struct qd_panel_par
{
    arma::uword M = 1, N = 1, Mg = 1, Ng = 1, MP = 1, NP = 1;
    double dV = 0.5, dH = 0.5, dgV = 0.5, dgH = 0.5;
    unsigned pol = 1, model = 1;
    arma::mat steer;
};

static qd_panel_par qd_panel_fill(const arma::uvec &panel, const arma::vec &spacing, const arma::uvec &ports,
                                  const arma::uvec &pol, const arma::mat &steer)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    qd_panel_par q;

    if (panel.n_elem > 4)
        throw std::invalid_argument("Input 'panel' must have at most 4 entries.");
    arma::uword *count[4] = {&q.M, &q.N, &q.Mg, &q.Ng};
    for (arma::uword i = 0; i < panel.n_elem; ++i)
        if (panel[i] != 0)
            *count[i] = panel[i];
    if (double(q.M) * double(q.N) * double(q.Mg) * double(q.Ng) > 1.0e7) // Guards the index arithmetic
        throw std::invalid_argument("Input 'panel' must not give more than 1e7 elements per polarization.");

    if (ports.n_elem > 2)
        throw std::invalid_argument("Input 'ports' must have at most 2 entries.");
    q.MP = (ports.n_elem > 0 && ports[0] != 0) ? ports[0] : q.M;
    q.NP = (ports.n_elem > 1 && ports[1] != 0) ? ports[1] : q.N;
    if (q.M % q.MP != 0)
        throw std::invalid_argument("Input 'ports' (M_P) must divide 'panel' (M).");
    if (q.N % q.NP != 0)
        throw std::invalid_argument("Input 'ports' (N_P) must divide 'panel' (N).");

    if (spacing.n_elem > 4)
        throw std::invalid_argument("Input 'spacing' must have at most 4 entries.");
    const char *name[4] = {"d_V", "d_H", "d_gV", "d_gH"};
    double *d[4] = {&q.dV, &q.dH, &q.dgV, &q.dgH};
    for (arma::uword i = 0; i < 4; ++i)
    {
        const double x = i < spacing.n_elem ? spacing[i] : nan;
        if (std::isnan(x)) // Defaults, each from the entries before
            *d[i] = (i == 0) ? 0.5 : (i == 1 ? q.dV : (i == 2 ? double(q.M) * q.dV : double(q.N) * q.dH));
        else if (std::isfinite(x) && x >= 0.0)
            *d[i] = x;
        else
            throw std::invalid_argument(std::string("Input 'spacing' (") + name[i] + ") must be finite and >= 0.");
    }

    if (pol.n_elem > 2)
        throw std::invalid_argument("Input 'pol' must have at most 2 entries.");
    const arma::uword p = (pol.n_elem > 0 && pol[0] != 0) ? pol[0] : 1, model = (pol.n_elem > 1 && pol[1] != 0) ? pol[1] : 1;
    if (p >= 4 && p <= 6)
        throw std::invalid_argument("Input 'pol' (polarization) values 4 to 6 are reserved; elements are combined through 'ports'.");
    if (p > 6)
        throw std::invalid_argument("Input 'pol' (polarization) must be 1, 2 or 3.");
    if (model > 2)
        throw std::invalid_argument("Input 'pol' (model) must be 1 or 2.");
    q.pol = unsigned(p), q.model = unsigned(model);

    const arma::uword n_panels = q.Mg * q.Ng, n_cols = steer.is_empty() ? 0 : steer.n_cols;
    if (n_cols > 2)
        throw std::invalid_argument("Input 'steer' must have 1 or 2 columns.");
    if (n_cols != 0 && steer.n_rows != 1 && steer.n_rows != n_panels)
        throw std::invalid_argument("Input 'steer' must have one row or one row per panel.");
    q.steer.zeros(n_panels, 2);
    for (arma::uword c = 0; c < n_cols; ++c)
        for (arma::uword i = 0; i < n_panels; ++i)
        {
            const double x = steer(steer.n_rows == 1 ? 0 : i, c);
            if (!std::isnan(x) && !(std::abs(x) <= 90.0))
                throw std::invalid_argument("Input 'steer' must have values between -90 and 90 degrees, or NaN for the default.");
            q.steer(i, c) = std::isnan(x) ? 0.0 : x;
            if (c == 0 && q.steer(i, c) != 0.0 && q.M == q.MP)
                throw std::invalid_argument("Input 'steer' has a non-zero tilt, but 'ports' gives one element per sub-array in the vertical; tilt needs combined elements.");
            if (c == 1 && q.steer(i, c) != 0.0 && q.N == q.NP)
                throw std::invalid_argument("Input 'steer' has a non-zero pan, but 'ports' gives one element per sub-array in the horizontal; pan needs combined elements.");
        }
    return q;
}

// Fill rule of the TR element tuple [phi_3dB, theta_3dB, SLA_V, A_max, G_max] (TR 38.901 Table 7.3-1, Table 7.3-2)
// with the defaults of a table: omitted or NaN entries take the default (theta_3dB = phi_3dB, A_max = SLA_V); an
// omitted G_max takes the default, NaN stays NaN (normalisation to the directivity); writes par[5]
static void qd_tr38901_element_fill(const arma::vec &element, double phi_3dB, double SLA_V, double G_max_dBi, double *par)
{
    if (element.n_elem > 5)
        throw std::invalid_argument("Input 'element' must have at most 5 entries.");
    const char *name[4] = {"phi_3dB", "theta_3dB", "SLA_V", "A_max"};
    for (arma::uword i = 0; i < 4; ++i)
    {
        const double dflt = (i == 0) ? phi_3dB : (i == 1 ? par[0] : (i == 2 ? SLA_V : par[2]));
        par[i] = (i < element.n_elem && !std::isnan(element[i])) ? element[i] : dflt;
        if (!(i < 2 ? par[i] > 0.0 : par[i] >= 0.0)) // +Inf: no taper (beamwidths), no limit (SLA_V, A_max)
            throw std::invalid_argument(std::string("Input 'element' (") + name[i] +
                                        (i < 2 ? ") must be > 0 degrees (Inf: no taper)." : ") must be >= 0, an attenuation in dB below the peak (Inf: no limit)."));
    }
    par[4] = element.n_elem > 4 ? element[4] : G_max_dBi;
    if (std::isinf(par[4]))
        throw std::invalid_argument("Input 'element' (G_max) must be finite, or NaN for the directivity.");
}

// Arrayant of n_elements on the regular grid of 'res' degrees (checked here; NaN or <= 0: 1), all fields zero, element
// positions zero, identity port weights; the caller fills the fields and marks the object valid
template <typename dtype>
static quadriga_lib::arrayant<dtype> qd_arrayant_grid(double res_in, arma::uword n_elements, const char *name)
{
    const double res = double(dtype(qd_arrayant_grid_resolution(res_in, 1.0, "res"))); // Grid size as in dtype
    const arma::uword no_az = arma::uword(360.0 / res) + 1ULL, no_el = arma::uword(180.0 / res) + 1ULL;
    const dtype pi = dtype(arma::datum::pi), pih = dtype(arma::datum::pi / 2.0);

    quadriga_lib::arrayant<dtype> ant;
    ant.name = name;
    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, no_az);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pih, pih, no_el);
    ant.e_theta_re.zeros(no_el, no_az, n_elements);
    ant.e_theta_im.zeros(no_el, no_az, n_elements);
    ant.e_phi_re.zeros(no_el, no_az, n_elements);
    ant.e_phi_im.zeros(no_el, no_az, n_elements);
    ant.element_pos.zeros(3, n_elements);
    ant.port_weights_re.eye(n_elements, n_elements);
    ant.port_weights_im.zeros(n_elements, n_elements);
    return ant;
}

// Isotropic radiator with vertical polarization
template <typename dtype>
static quadriga_lib::arrayant<dtype> qd_generate_arrayant_omni(double res)
{
    auto ant = qd_arrayant_grid<dtype>(res, 1, "omni");
    ant.e_theta_re.ones();
    qd_arrayant_mark_valid(ant);
    return ant;
}

// Two isotropic radiators with vertical and horizontal polarization
template <typename dtype>
static quadriga_lib::arrayant<dtype> qd_generate_arrayant_xpol(double res)
{
    auto ant = qd_arrayant_grid<dtype>(res, 2, "xpol");
    ant.e_theta_re.slice(0).ones();
    ant.e_phi_re.slice(1).ones();
    qd_arrayant_mark_valid(ant);
    return ant;
}

// Short dipole with vertical polarization
template <typename dtype>
static quadriga_lib::arrayant<dtype> qd_generate_arrayant_dipole(double res)
{
    auto ant = qd_arrayant_grid<dtype>(res, 1, "dipole");
    ant.e_theta_re.slice(0) = arma::repmat(arma::cos(dtype(0.999999) * ant.elevation_grid) * dtype(std::sqrt(1.499961)), 1, ant.n_azimuth());
    qd_arrayant_mark_valid(ant);
    return ant;
}

// Half-wave dipole with vertical polarization
template <typename dtype>
static quadriga_lib::arrayant<dtype> qd_generate_arrayant_half_wave_dipole(double res)
{
    auto ant = qd_arrayant_grid<dtype>(res, 1, "half-wave-dipole");
    // cos(pi/2 sin(el)) / cos(el), written without the cancellation of cos(pi/2 sin(el)) next to the poles
    const arma::Col<dtype> c = arma::cos(ant.elevation_grid), s = arma::abs(arma::sin(ant.elevation_grid));
    const arma::Col<dtype> f = arma::sin(dtype(arma::datum::pi / 2.0) * arma::square(c) / (dtype(1.0) + s)) / c * dtype(1.280968208215292);
    ant.e_theta_re.slice(0) = arma::repmat(f, 1, ant.n_azimuth());
    qd_arrayant_mark_valid(ant);
    return ant;
}

// Vertically polarized element with 3 dB beamwidths in degrees (> 0, Inf: flat) and the power floor rear_gain_lin in
// [0, 1) relative to the peak, normalized to its directivity; the inputs are checked by qd_custom_element_fill
template <typename dtype>
static quadriga_lib::arrayant<dtype> qd_generate_arrayant_custom(double az_3dB_in, double el_3dB_in, double rear_gain_lin_in, double res_in)
{
    const dtype az_3dB = dtype(az_3dB_in), el_3dB = dtype(el_3dB_in), rear_gain_lin = dtype(rear_gain_lin_in);
    quadriga_lib::arrayant<dtype> ant = qd_arrayant_grid<dtype>(res_in, 1, "custom");
    const arma::uword no_az = ant.n_azimuth(), no_el = ant.n_elevation();

    constexpr dtype zero = dtype(0.0), one = dtype(1.0), half = dtype(0.5),
                    limit = dtype(1e-7), step = dtype(-0.382), limit_inf = dtype(1e38);
    const dtype deg2rad = dtype(arma::datum::pi / 360.0);

    arma::Col<dtype> phi_sq = ant.azimuth_grid % ant.azimuth_grid;
    arma::Col<dtype> cos_theta = arma::cos(ant.elevation_grid);
    cos_theta.at(0) = zero, cos_theta.at(no_el - 1) = zero;
    arma::Col<dtype> az_3dB_rad(1), el_3dB_rad(1);
    az_3dB_rad.at(0) = az_3dB * deg2rad;
    el_3dB_rad.at(0) = el_3dB * deg2rad;

    // Calculate azimuth pattern cut; an infinite beamwidth gives a flat cut (exponent 0)
    dtype a = one, d = half, x = limit_inf, delta = limit_inf;
    arma::Col<dtype> C(no_az), D(no_el);
    arma::Mat<dtype> xn; // Pattern value at the 3 dB angle, [1, 1]
    if (std::isinf(az_3dB))
        a = zero;
    for (unsigned lp = 0; lp < 5000 && std::isfinite(az_3dB); ++lp)
    {
        dtype an = (lp == 0) ? a : a + d;
        delta = (lp == 0) ? limit_inf : std::abs(a - an);
        C = rear_gain_lin + (one - rear_gain_lin) * arma::exp(-an * phi_sq);
        quadriga_lib::interp_1D(ant.azimuth_grid, C, az_3dB_rad, xn);
        dtype xm = std::abs(xn.at(0) - half);
        a = xm < x ? an : a;
        d = xm < x ? d : step * d;
        x = xm < x ? xm : x;
        if (delta < limit)
            break;
    }
    C = arma::exp(-a * phi_sq);

    // Calculate elevation pattern cut
    a = std::isinf(el_3dB) ? zero : one, d = half, x = limit_inf, delta = limit_inf;
    for (unsigned lp = 0; lp < 5000 && std::isfinite(el_3dB); ++lp)
    {
        dtype an = (lp == 0) ? a : a + d;
        delta = (lp == 0) ? limit_inf : std::abs(a - an);
        D = arma::pow(cos_theta, an);
        quadriga_lib::interp_1D(ant.elevation_grid, D, el_3dB_rad, xn);
        dtype xm = std::abs(xn.at(0) - half);
        a = xm < x ? an : a;
        d = xm < x ? d : step * d;
        x = xm < x ? xm : x;
        if (delta < limit)
            break;
    }
    D = arma::pow(cos_theta, a);

    // Combined pattern
    dtype *ptr = ant.e_theta_re.memptr();
    for (dtype *col = C.begin(); col != C.end(); ++col)
        for (dtype *row = D.begin(); row != D.end(); ++row)
            *ptr++ = std::sqrt(rear_gain_lin + (one - rear_gain_lin) * *row * *col);

    qd_arrayant_mark_valid(ant);

    // Normalize to gain
    const arma::Col<dtype> directivity_dBi = quadriga_lib::calc_directivity(ant);
    dtype directivity = dtype(std::pow(10.0, 0.1 * double(directivity_dBi[0])));
    dtype p_max = ant.e_theta_re.max();
    p_max *= p_max;
    ant.e_theta_re *= std::sqrt(directivity / p_max);

    return ant;
}

// One-element, vertically polarized TR 38.901 element (Table 7.3-1 shape) from the parameters of qd_tr38901_element_fill:
// field sqrt(10^((G_max + A_dB) / 10)); G_max NaN: the peak power equals the directivity; 'res' in degree, resolved
template <typename dtype>
static quadriga_lib::arrayant<dtype> qd_tr38901_element(const double *par, double res)
{
    quadriga_lib::arrayant<dtype> ant = qd_arrayant_grid<dtype>(res, 1, "3gpp");
    const double rad2deg = 180.0 / arma::datum::pi, G_max = std::isnan(par[4]) ? 0.0 : par[4];
    dtype *ptr = ant.e_theta_re.memptr();
    for (const dtype az : ant.azimuth_grid)
    {
        const double h = double(az) * rad2deg / par[0], A_H = std::min(12.0 * h * h, par[3]);
        for (const dtype el : ant.elevation_grid)
        {
            const double v = double(el) * rad2deg / par[1], A_V = std::min(12.0 * v * v, par[2]); // theta - 90 deg = -el
            *ptr++ = dtype(std::pow(10.0, 0.05 * (G_max - std::min(A_V + A_H, par[3]))));
        }
    }
    if (std::isnan(par[4]))
    {
        const double directivity = std::pow(10.0, 0.1 * double(quadriga_lib::calc_directivity(ant)[0]));
        const double p_max = double(ant.e_theta_re.max());
        ant.e_theta_re *= dtype(std::sqrt(directivity) / p_max);
    }
    qd_arrayant_mark_valid(ant);
    return ant;
}

// Polarization of TR 38.901 clause 7.3.2 applied in place to a valid one-element arrayant: 'pol' 1 (vertical), 2
// (zeta = 0 and 90 deg) or 3 (zeta = +45 and -45 deg) gives 1 or 2 elements; 'model' 1 turns the element (pattern and
// polarization) by zeta about its boresight (eq. 7.3-3), 'model' 2 sets F_theta = sqrt(A') cos(zeta), F_phi = sqrt(A') sin(zeta) with
// A' = |F_theta|^2 + |F_phi|^2 of the element (eq. 7.3-4, 7.3-5)
template <typename dtype>
static void qd_apply_polarization(quadriga_lib::arrayant<dtype> &ant, unsigned pol, unsigned model)
{
    const arma::uword n_el = (pol == 1) ? 1 : 2;
    if (n_el == 2)
        quadriga_lib::copy_element(ant, 0, {1});

    if (model == 1)
    {
        const double zeta[2] = {pol == 3 ? arma::datum::pi / 4.0 : 0.0, pol == 3 ? -arma::datum::pi / 4.0 : arma::datum::pi / 2.0};
        for (arma::uword i = 0; i < n_el; ++i)
            if (zeta[i] != 0.0)
                quadriga_lib::rotate_pattern(ant, arma::Col<dtype>{dtype(zeta[i]), dtype(0.0), dtype(0.0)}, 3, arma::uvec{i});
        return;
    }

    const double s = std::sqrt(0.5);
    const double cos_zeta[2] = {pol == 3 ? s : 1.0, pol == 3 ? s : 0.0}, sin_zeta[2] = {pol == 3 ? s : 0.0, pol == 3 ? -s : 1.0};
    const arma::uword n_val = ant.n_elevation() * ant.n_azimuth();
    const dtype *t_re = ant.e_theta_re.memptr(), *t_im = ant.e_theta_im.memptr(), *p_re = ant.e_phi_re.memptr(), *p_im = ant.e_phi_im.memptr();
    arma::vec a(n_val); // sqrt(A')
    for (arma::uword k = 0; k < n_val; ++k)
        a[k] = std::sqrt(double(t_re[k]) * double(t_re[k]) + double(t_im[k]) * double(t_im[k]) +
                         double(p_re[k]) * double(p_re[k]) + double(p_im[k]) * double(p_im[k]));
    for (arma::uword i = 0; i < n_el; ++i)
    {
        dtype *tr = ant.e_theta_re.slice_memptr(i), *pr = ant.e_phi_re.slice_memptr(i);
        for (arma::uword k = 0; k < n_val; ++k)
            tr[k] = dtype(a[k] * cos_zeta[i]), pr[k] = dtype(a[k] * sin_zeta[i]);
    }
    ant.e_theta_im.zeros();
    ant.e_phi_im.zeros();
}

// The polarization set of an array type: the TR element of 'par' ('tr_element') or the omni element on the grid of
// 'res', or the field patterns of 'pattern' ('regular_grid': resampled to a regular grid); a one-element set gets the
// polarizations of 'pol' and 'model'
template <typename dtype>
static quadriga_lib::arrayant<dtype> qd_polarized_element(bool tr_element, const double *par, bool element_given,
                                                          const quadriga_lib::arrayant<dtype> &pattern, unsigned pol, unsigned model, double res,
                                                          bool regular_grid = false)
{
    quadriga_lib::arrayant<dtype> elem;
    if (pattern.n_elements() == 0)
    {
        res = qd_arrayant_grid_resolution(res, 1.0, "res");
        elem = tr_element ? qd_tr38901_element<dtype>(par, res) : qd_generate_arrayant_omni<dtype>(res);
    }
    else
    {
        if (element_given)
            throw std::invalid_argument("Inputs 'element' and 'pattern' cannot be given together.");
        elem = pattern;
        const std::string error_message = elem.is_valid(); // Deep check
        if (!error_message.empty())
            throw std::invalid_argument("Input 'pattern' is not valid: " + error_message);
        if (elem.n_elements() > 1 && (pol != 1 || model != 1))
            throw std::invalid_argument("Input 'pol' must be [1, 1] for a 'pattern' with more than one element.");
        if (regular_grid) // Turning an element on its own grid needs samples in all directions
            quadriga_lib::rotate_pattern(elem, arma::Col<dtype>(3, arma::fill::zeros), 0);
    }
    if (elem.n_elements() == 1) // Only the field patterns are used
    {
        elem.element_pos.zeros(3, 1), elem.port_weights_re.ones(1, 1), elem.port_weights_im.zeros(1, 1);
        qd_apply_polarization(elem, pol, model);
    }
    return elem;
}

// Panel array of TR 38.901 clause 7.3 from the polarization set 'elem' (P elements, valid; only its radiation pattern
// is used): positions and sub-array weights follow from the indices (p, m, n, mg, ng), see generate_antenna;
// 'apply_weights' combines each port at its sub-array centre, from one combined sub-array per panel
template <typename dtype>
static quadriga_lib::arrayant<dtype> qd_arrayant_panel(const quadriga_lib::arrayant<dtype> &elem, double center_freq,
                                                       const qd_panel_par &q, bool apply_weights)
{
    const arma::uword P = elem.n_elements(), K = q.M / q.MP, L = q.N / q.NP;
    const bool combine = apply_weights && K * L > 1;
    const arma::uword n_el = P * q.M * q.N * q.Mg * q.Ng, n_port = n_el / (K * L), n_out = combine ? n_port : n_el;
    const arma::uword n_val = elem.n_elevation() * elem.n_azimuth();
    const double lambda = qd_speed_of_light / center_freq, two_pi = 2.0 * arma::datum::pi, deg2rad = arma::datum::pi / 180.0;
    const double w_abs = 1.0 / std::sqrt(double(K * L));

    quadriga_lib::arrayant<dtype> ant;
    ant.name = "3gpp";
    ant.center_freq = dtype(center_freq);
    ant.azimuth_grid = elem.azimuth_grid;
    ant.elevation_grid = elem.elevation_grid;
    ant.e_theta_re.set_size(elem.n_elevation(), elem.n_azimuth(), n_out);
    ant.e_theta_im.set_size(elem.n_elevation(), elem.n_azimuth(), n_out);
    ant.e_phi_re.set_size(elem.n_elevation(), elem.n_azimuth(), n_out);
    ant.e_phi_im.set_size(elem.n_elevation(), elem.n_azimuth(), n_out);
    ant.element_pos.zeros(3, n_out);
    if (combine)
        ant.port_weights_re.eye(n_port, n_port);
    else
        ant.port_weights_re.zeros(n_el, n_port);
    ant.port_weights_im.zeros(ant.port_weights_re.n_rows, n_port);

    arma::uword e = 0;
    for (arma::uword ng = 0; ng < q.Ng; ++ng)
        for (arma::uword mg = 0; mg < q.Mg; ++mg)
        {
            const double tilt = q.steer(mg + q.Mg * ng, 0) * deg2rad, pan = q.steer(mg + q.Mg * ng, 1) * deg2rad;
            const double zg = (double(mg) - 0.5 * double(q.Mg - 1)) * q.dgV, yg = (double(ng) - 0.5 * double(q.Ng - 1)) * q.dgH;
            quadriga_lib::arrayant<dtype> sub; // combine: the K x L sub-array of this panel, centred on the origin, as P ports
            if (combine)
            {
                qd_panel_par qs = q;
                qs.M = K, qs.N = L, qs.Mg = 1, qs.Ng = 1, qs.MP = 1, qs.NP = 1, qs.steer = q.steer.row(mg + q.Mg * ng);
                sub = quadriga_lib::apply_port_weights(qd_arrayant_panel(elem, center_freq, qs, false));
            }
            const quadriga_lib::arrayant<dtype> &src = combine ? sub : elem; // Patterns of the elements or ports
            for (arma::uword n = 0; n < q.N; ++n)
                for (arma::uword m = 0; m < q.M; ++m)
                {
                    if (combine && (m % K != 0 || n % L != 0)) // One output per sub-array
                        continue;
                    // Element (k, l) of the sub-array (TXRU) of port j; eq. 7.3-1 with the phase reference at the sub-array centre
                    const double k = double(m % K) - 0.5 * double(K - 1), l = double(n % L) - 0.5 * double(L - 1);
                    const double phase = two_pi * (k * q.dV * std::sin(tilt) - l * q.dH * std::sin(pan));
                    const double z = (double(m) - 0.5 * double(q.M - 1)) * q.dV + zg, y = (double(n) - 0.5 * double(q.N - 1)) * q.dH + yg;
                    const double zc = (double(m / K * K) + 0.5 * double(K - 1) - 0.5 * double(q.M - 1)) * q.dV + zg;
                    const double yc = (double(n / L * L) + 0.5 * double(L - 1) - 0.5 * double(q.N - 1)) * q.dH + yg;
                    const arma::uword j0 = P * (m / K + q.MP * (n / L + q.NP * (mg + q.Mg * ng)));
                    for (arma::uword p = 0; p < P; ++p)
                    {
                        const arma::uword i = combine ? j0 + p : e++;
                        std::memcpy(ant.e_theta_re.slice_memptr(i), src.e_theta_re.slice_memptr(p), n_val * sizeof(dtype));
                        std::memcpy(ant.e_theta_im.slice_memptr(i), src.e_theta_im.slice_memptr(p), n_val * sizeof(dtype));
                        std::memcpy(ant.e_phi_re.slice_memptr(i), src.e_phi_re.slice_memptr(p), n_val * sizeof(dtype));
                        std::memcpy(ant.e_phi_im.slice_memptr(i), src.e_phi_im.slice_memptr(p), n_val * sizeof(dtype));
                        ant.element_pos(1, i) = dtype((combine ? yc : y) * lambda), ant.element_pos(2, i) = dtype((combine ? zc : z) * lambda);
                        if (!combine)
                            ant.port_weights_re(i, j0 + p) = dtype(w_abs * std::cos(phase)), ant.port_weights_im(i, j0 + p) = dtype(w_abs * std::sin(phase));
                    }
                }
        }
    if (!ant.element_pos.is_finite())
        throw std::invalid_argument("Inputs 'spacing' and 'center_freq' give non-finite element positions.");
    qd_arrayant_mark_valid(ant);
    return ant;
}

// The 3GPP and ULA types: the panel array of qd_arrayant_panel with the TR element ('tr_element') or the omni element,
// or with the field patterns of 'pattern'
template <typename dtype>
static quadriga_lib::arrayant<dtype> qd_generate_panel_array(bool tr_element, const arma::uvec &panel, double center_freq,
                                                             const arma::vec &spacing, const arma::uvec &ports, const arma::uvec &pol,
                                                             const arma::mat &steer, const arma::vec &element,
                                                             const quadriga_lib::arrayant<dtype> &pattern, double res, bool apply_weights)
{
    qd_check_center_freq(center_freq);
    const qd_panel_par q = qd_panel_fill(panel, spacing, ports, pol, steer);
    double par[5];
    qd_tr38901_element_fill(element, 65.0, 30.0, 8.0, par);
    const auto elem = qd_polarized_element<dtype>(tr_element, par, element.n_elem != 0, pattern, q.pol, q.model, res);
    return qd_arrayant_panel(elem, center_freq, q, apply_weights);
}

// Handheld UT and CPE of TR 38.901 clause 7.3.0: 'P' elements per entry of 'locations', at the TR location labels.
// Handheld (Fig. 7.3-2, Table A.6-1/-2): each element is rolled about its boresight and turned to the bearing of its
// location, which is eq. 7.3-6 to 7.3-8 with the pole of the local frame defined. CPE (Fig. 7.3-6 seen from the front):
// the elements keep their orientation and face +x.
template <typename dtype>
static quadriga_lib::arrayant<dtype> qd_generate_device_array(bool handheld, const arma::vec &element, const arma::uvec &pol,
                                                              const quadriga_lib::arrayant<dtype> &pattern, const arma::vec &device_size,
                                                              const arma::uvec &locations, double res)
{
    qd_panel_par q = qd_panel_fill(arma::uvec(), arma::vec(), arma::uvec(), pol, arma::mat());
    if (handheld && q.pol > 1 && (pol.n_elem < 2 || pol[1] == 0)) // Two field patterns of the handheld model are those of Model-2
        q.model = 2;
    double par[5];
    if (handheld)
        qd_tr38901_element_fill(element, 125.0, 22.5, 5.3, par); // TR 38.901 Table 7.3-2
    else
        qd_tr38901_element_fill(element, 90.0, 25.0, 5.0, par); // TR 38.803 Table 5.2.3.3-1
    const auto elem = qd_polarized_element<dtype>(true, par, element.n_elem != 0, pattern, q.pol, q.model, res, handheld);

    if (device_size.n_elem > 3)
        throw std::invalid_argument("Input 'device_size' must have at most 3 entries.");
    double size[3] = {handheld ? 0.15 : 0.0, handheld ? 0.07 : 0.2, handheld ? 0.0 : 0.2};
    for (arma::uword i = 0; i < device_size.n_elem; ++i)
        if (!std::isnan(device_size[i]))
            size[i] = device_size[i];
    for (const double s : size)
        if (!(s >= 0.0 && s <= double(std::numeric_limits<dtype>::max())))
            throw std::invalid_argument("Input 'device_size' must be finite and >= 0.");
    if (handheld && (size[0] == 0.0 || size[1] == 0.0 || size[2] != 0.0))
        throw std::invalid_argument("Input 'device_size' of a handheld UT must have X > 0, Y > 0 and Z = 0.");
    if (!handheld && size[0] != 0.0)
        throw std::invalid_argument("Input 'device_size' of a CPE must have X = 0.");

    const arma::uword n_labels = handheld ? 8 : 9;
    const arma::uvec labels = locations.is_empty() ? arma::uvec(arma::regspace<arma::uvec>(1, n_labels)) : locations;
    if (labels.min() < 1 || labels.max() > n_labels)
        throw std::invalid_argument(std::string("Input 'locations' must have labels from 1 to ") + (handheld ? "8." : "9."));

    const arma::uword P = elem.n_elements();
    q.M = labels.n_elem, q.MP = labels.n_elem; // One column of elements, placed below
    quadriga_lib::arrayant<dtype> ant = qd_arrayant_panel(elem, 299792458.0, q, false);
    ant.name = handheld ? "3gpp-ut" : "3gpp-cpe";

    // Signs of the coordinates per label: x and y of the handheld, y and z = -x of the CPE
    static const int sign_x[9] = {-1, 0, 1, 1, 1, 0, -1, -1, 0}, sign_y[9] = {-1, -1, -1, 0, 1, 1, 1, 0, 0};
    const double roll = (P == 2 && q.pol != 3 ? 0.25 : 0.5) * arma::datum::pi; // TR 38.901 Table A.6-2 for two field patterns, else A.6-1
    for (arma::uword l = 0; l < labels.n_elem; ++l)
    {
        const arma::uword u = labels[l] - 1;
        const double sx = double(sign_x[u]), sy = double(sign_y[u]);
        const double x = 0.5 * sx * size[0] + 0.0, y = 0.5 * sy * size[1] + 0.0, z = 0.0 - 0.5 * sx * size[2]; // Handheld Z and CPE X are 0, never -0
        if (handheld)
        {
            const arma::Col<dtype> orientation = {dtype(roll), dtype(0.0), dtype(std::atan2(y, x))};
            quadriga_lib::rotate_pattern(ant, orientation, 3, arma::regspace<arma::uvec>(P * l, P * l + P - 1));
        }
        ant.element_pos.cols(P * l, P * l + P - 1).each_col() = arma::Col<dtype>{dtype(x), dtype(y), dtype(z)};
    }
    qd_arrayant_mark_valid(ant);
    return ant;
}

// Fill rule of the custom-shape element tuple [az_3dB, el_3dB, floor_dB]: omitted or NaN beamwidths take bw_default,
// floor_dB (attenuation below the peak, positive) omitted or NaN takes Inf = no floor; returns the arguments of
// qd_generate_arrayant_custom
static void qd_custom_element_fill(const arma::vec &element, double bw_default, double &az_3dB, double &el_3dB, double &rear_gain_lin)
{
    if (element.n_elem > 3)
        throw std::invalid_argument("Input 'element' must have at most 3 entries.");
    az_3dB = (element.n_elem > 0 && !std::isnan(element[0])) ? element[0] : bw_default;
    el_3dB = (element.n_elem > 1 && !std::isnan(element[1])) ? element[1] : bw_default;
    if (!(az_3dB > 0.0))
        throw std::invalid_argument("Input 'element' (az_3dB) must be > 0 degrees (Inf: no taper).");
    if (!(el_3dB > 0.0))
        throw std::invalid_argument("Input 'element' (el_3dB) must be > 0 degrees (Inf: no taper).");
    const double floor_dB = (element.n_elem > 2 && !std::isnan(element[2])) ? element[2] : std::numeric_limits<double>::infinity();
    rear_gain_lin = std::pow(10.0, -0.1 * floor_dB); // Inf: 0
    if (!(floor_dB > 0.0) || rear_gain_lin >= 1.0)   // also floor_dB below the double resolution of 1 - 10^(-x/10)
        throw std::invalid_argument("Input 'element' (floor_dB) must be > 0, the attenuation of the floor in dB below the peak (Inf: no floor).");
}

// Multibeam panel array: custom-shape element and maximum-ratio-transmission weights for the directions in beams
template <typename dtype>
static quadriga_lib::arrayant<dtype> qd_generate_arrayant_multibeam(const arma::uvec &panel, double center_freq, const arma::vec &spacing,
                                                                    const arma::uvec &pol, const arma::vec &element, const arma::mat &beams,
                                                                    double res_in, bool separate_beams, bool apply_weights)
{
    qd_check_center_freq(center_freq);
    const double res = qd_arrayant_grid_resolution(res_in, 1.0, "res");
    const qd_panel_par q = qd_panel_fill(panel, spacing, {}, pol, {});
    double az_3dB, el_3dB, rear_gain_lin;
    qd_custom_element_fill(element, 120.0, az_3dB, el_3dB, rear_gain_lin);

    if (!beams.is_empty() && beams.n_rows != 2 && beams.n_rows != 3)
        throw std::invalid_argument("Input 'beams' must have 2 or 3 rows.");
    if (!beams.is_finite())
        throw std::invalid_argument("Input 'beams' must be finite.");
    const arma::uword n_beams = beams.is_empty() ? 1 : beams.n_cols;
    arma::Col<dtype> az(n_beams, arma::fill::zeros), el(n_beams, arma::fill::zeros), weight(n_beams, arma::fill::ones);
    for (arma::uword i = 0; i < beams.n_cols; ++i) // Degree to rad
    {
        az[i] = dtype(beams(0, i) * arma::datum::pi / 180.0), el[i] = dtype(beams(1, i) * arma::datum::pi / 180.0);
        weight[i] = beams.n_rows == 3 ? dtype(beams(2, i)) : dtype(1.0);
    }

    const dtype zero = dtype(0.0);
    const dtype dist = dtype(5000.0 * qd_speed_of_light / center_freq); // Normalized distance = 5000 wavelengths
    arma::Mat<dtype> cart(3, n_beams, arma::fill::none);
    qd_geo2cart_interleaved<dtype>(n_beams, cart.memptr(), az.memptr(), el.memptr());

    auto pattern = qd_generate_arrayant_custom<dtype>(az_3dB, el_3dB, rear_gain_lin, res);

    // Vertically polarized array with one port per element for the MRT weights
    qd_panel_par q1 = q;
    q1.pol = 1, q1.model = 1;
    quadriga_lib::arrayant<dtype> ant = qd_arrayant_panel(pattern, center_freq, q1, false);

    arma::uword n_elements = ant.n_elements();
    arma::Mat<dtype> cpl_eye = ant.port_weights_re;
    arma::Mat<dtype> cpl_zero = ant.port_weights_im;

    auto probe = qd_generate_arrayant_omni<dtype>(30.0);

    // Phase shifters
    arma::Mat<dtype> cpl_re, cpl_im;

    // Get MRT weights
    const arma::Col<dtype> path_gain = {1.0};
    const arma::Col<dtype> path_length = {1000.0};
    const arma::Col<dtype> path_angle = {0.0};
    const arma::Col<dtype> zero_3(3, arma::fill::zeros); // TX position and orientations
    arma::Mat<dtype> path_pol = {1.0, zero, zero, zero, zero, zero, -1.0, zero};
    path_pol = path_pol.t();

    arma::Cube<dtype> coeff_re(1, n_elements, n_beams);
    arma::Cube<dtype> coeff_im(1, n_elements, n_beams);

    // Variables for the optimization loop
    arma::Col<dtype> gain(n_beams);
    arma::Mat<dtype> search_direction(2, n_beams, arma::fill::value((dtype)0.05)); // Step size in rad
    arma::Mat<dtype> angles(2, n_beams);
    for (arma::uword i_beam = 0; i_beam < n_beams; ++i_beam)
        angles(0, i_beam) = az[i_beam], angles(1, i_beam) = el[i_beam];

    // Beam pointing optimization loop
    arma::uword lp_max = separate_beams ? 7 : 25;
    for (arma::uword lp = 0; lp < lp_max; ++lp)
    {
        // Set the angles for testing
        arma::Mat<dtype> test_angles = angles;
        if (lp % 2 == 1) // Test azimuth
            for (arma::uword i_beam = 0; i_beam < n_beams; ++i_beam)
                test_angles(0, i_beam) += search_direction(0, i_beam);
        else if (lp != 0) // Test elevation
            for (arma::uword i_beam = 0; i_beam < n_beams; ++i_beam)
                test_angles(1, i_beam) += search_direction(1, i_beam);

        // Calculate the coefficients for each beam
        ant.port_weights_re = cpl_eye;
        ant.port_weights_im = cpl_zero;
        for (arma::uword i_beam = 0; i_beam < n_beams; ++i_beam)
        {
            arma::Col<dtype> aod = {test_angles(0, i_beam)};
            arma::Col<dtype> eod = {test_angles(1, i_beam)};

            arma::Cube<dtype> coeff_re_tmp, coeff_im_tmp, delay_tmp;

            // Receiver positions
            dtype R[3];
            qd_geo2cart_interleaved<dtype>(1, R, aod.memptr(), eod.memptr());
            qd_multiply_scalar(dist, R, 3);

            const arma::Col<dtype> r_pos = {R[0], R[1], R[2]};
            quadriga_lib::get_channels_planar(ant, probe, zero_3, zero_3, r_pos, zero_3, aod, eod,
                                              path_angle, path_angle, path_gain, path_length, path_pol,
                                              coeff_re_tmp, coeff_im_tmp, delay_tmp, center_freq);

            for (arma::uword i_elem = 0; i_elem < n_elements; ++i_elem)
            {
                coeff_re(0, i_elem, i_beam) = coeff_re_tmp[i_elem] * weight[i_beam];
                coeff_im(0, i_elem, i_beam) = coeff_im_tmp[i_elem] * weight[i_beam];
            }
        }

        // Calculate phase shifts
        if (separate_beams)
        {
            ant.port_weights_re.set_size(n_elements, n_beams);
            ant.port_weights_im.set_size(n_elements, n_beams);
            for (arma::uword i_beam = 0; i_beam < n_beams; ++i_beam)
                for (arma::uword i_elem = 0; i_elem < n_elements; ++i_elem)
                {
                    std::complex<dtype> w(coeff_re(0, i_elem, i_beam), coeff_im(0, i_elem, i_beam));

                    dtype mag = std::abs(w);
                    if (mag > std::numeric_limits<dtype>::epsilon() * dtype(1e3))
                        w /= mag;
                    else
                        w = std::complex<dtype>(dtype(1.0), dtype(0.0));

                    ant.port_weights_re(i_elem, i_beam) = w.real();
                    ant.port_weights_im(i_elem, i_beam) = -w.imag(); // Use conjugate
                }
        }
        else // Combine the coefficients into a single beam
        {
            ant.port_weights_re.set_size(n_elements, 1);
            ant.port_weights_im.set_size(n_elements, 1);
            for (arma::uword i_elem = 0; i_elem < n_elements; ++i_elem)
            {
                std::complex<double> w(0.0, 0.0); // Sum over the beams in double precision
                for (arma::uword i_beam = 0; i_beam < n_beams; ++i_beam)
                    w += std::complex<double>(coeff_re(0, i_elem, i_beam), coeff_im(0, i_elem, i_beam));

                const double mag = std::abs(w);
                if (mag > double(std::numeric_limits<dtype>::epsilon()) * 1e3)
                    w /= mag;
                else
                    w = std::complex<double>(1.0, 0.0);

                ant.port_weights_re[i_elem] = dtype(w.real());
                ant.port_weights_im[i_elem] = dtype(-w.imag()); // Use conjugate
            }
        }

        // Calculate gain per beam
        for (arma::uword i_beam = 0; i_beam < n_beams; ++i_beam)
        {
            arma::Col<dtype> aod = {az[i_beam]};
            arma::Col<dtype> eod = {el[i_beam]};
            arma::Cube<dtype> coeff_re_tmp, coeff_im_tmp, delay_tmp;

            dtype Rx = cart.at(0, i_beam) * dist;
            dtype Ry = cart.at(1, i_beam) * dist;
            dtype Rz = cart.at(2, i_beam) * dist;

            const arma::Col<dtype> r_pos = {Rx, Ry, Rz};
            quadriga_lib::get_channels_planar(ant, probe, zero_3, zero_3, r_pos, zero_3, aod, eod,
                                              path_angle, path_angle, path_gain, path_length, path_pol,
                                              coeff_re_tmp, coeff_im_tmp, delay_tmp, center_freq);

            dtype a = separate_beams ? coeff_re_tmp[i_beam] : coeff_re_tmp[0];
            dtype b = separate_beams ? coeff_im_tmp[i_beam] : coeff_im_tmp[0];
            dtype gain_new = a * a + b * b;

            // Update search direction for next iteration. The first iteration tests the start angles and is always
            // stored, also when the gain is 0 (e.g. a beam towards a pole, where the element pattern is 0).
            if (gain_new > gain[i_beam] || lp == 0) // Update gain and angles, keep search direction
            {
                gain[i_beam] = gain_new;
                angles(0, i_beam) = test_angles(0, i_beam);
                angles(1, i_beam) = test_angles(1, i_beam);
                cpl_re = ant.port_weights_re;
                cpl_im = ant.port_weights_im;
            }
            else // Discard new angles, change search direction and step size
            {
                if (lp % 2 == 1) // azimuth
                    search_direction(0, i_beam) *= (dtype)-0.382;
                else // elevation
                    search_direction(1, i_beam) *= (dtype)-0.382;
            }
        }
    }

    // Polarized array: the weights of a beam apply to every polarization; ports with the polarization fastest
    qd_apply_polarization(pattern, q.pol, q.model);
    const arma::uword P = pattern.n_elements(), n_out = cpl_re.n_cols;
    ant = qd_arrayant_panel(pattern, center_freq, q, false);
    ant.port_weights_re.zeros(P * n_elements, P * n_out);
    ant.port_weights_im.zeros(P * n_elements, P * n_out);
    for (arma::uword o = 0; o < n_out; ++o)
        for (arma::uword i = 0; i < n_elements; ++i)
            for (arma::uword p = 0; p < P; ++p)
                ant.port_weights_re(p + P * i, p + P * o) = cpl_re(i, o), ant.port_weights_im(p + P * i, p + P * o) = cpl_im(i, o);
    ant.name = "multibeam";
    qd_arrayant_mark_valid(ant);

    if (apply_weights)
        ant = quadriga_lib::apply_port_weights(ant);

    return ant;
}

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# generate_antenna
Generate an antenna array of a given type

- Tuples (`element`, `panel`, `spacing`, `pol`, `ports`, `steer`): entries left off at the end or set to NaN (0 in `panel`, `pol`, `ports`) take their default, e.g. `panel = {4, 8}` is one panel of 4 rows and 8 columns.
- Panel arrays: elements are ordered polarization fastest, then row `m` (upwards along z), column `n` (along +y), panel row `mg`, panel column `ng`, centred on the origin; ports in the same order with `m_p`, `n_p`. The port weights `[n_elements, n_ports]` holds the weights; `apply_weights` combines each port into one element at its sub-array centre.
- Sub-array weights (`ula`, `3GPP`; TR 38.901 eq. 7.3-1, along y by analogy): a port feeds `K = M / M_P` × `L = N / N_P` elements; element `(k, l)` gets `exp(j 2π ((k - (K-1)/2) d_V sin(tilt) - (l - (L-1)/2) d_H sin(pan))) / sqrt(K L)`; a positive tilt points the beam down, a positive pan towards +y.
- Polarization (TR 38.901 clause 7.3.2): `pol` 2 has the slants ζ = 0° and 90°, `pol` 3 ζ = +45° and -45° (element 0 first). Model-1 turns the element by ζ about its boresight (eq. 7.3-3); Model-2 sets F_θ = sqrt(A') cos ζ and F_φ = sqrt(A') sin ζ with A' = |F_θ|² + |F_φ|² of the element (eq. 7.3-4, 7.3-5).
- `multibeam`: maximum-ratio-transmission weights for the directions in `beams` (exact for one beam, approximate for several; only the ratios of the beam weights matter); one port per polarization, with `separate_beams` one per polarization and beam.
- Handheld UT (`3GPP-UT`, TR 38.901 clause 7.3): `P` elements at each entry of `locations` (Fig. 7.3-2), polarization fastest; each element is rolled about its boresight by 90° (45° for two field patterns: `pol` 2 or a two-element `pattern`) and turned to the bearing `atan2(y, x)` of its location (Tables A.6-1, A.6-2); a `pattern` on an irregular grid is first resampled to a regular one. `pol` 1 and 2 with their default model are the TR model (eq. 7.3-6 to 7.3-8), other values are extensions. For an element symmetric under a 180° roll (every `element` tuple), the other z'' direction of Fig. 7.3-7 is the same antenna with a negated port weight.
- CPE (`3GPP-CPE`): `P` elements at each entry of `locations` (Fig. 7.3-6 seen from the front, in the y-z plane), facing +x. TR 38.901 defines no CPE element; the default is the UE element of TR 38.803 Table 5.2.3.3-1.
- An input that the type does not take throws.

## Declaration:
```
quadriga_lib::arrayant<dtype> quadriga_lib::generate_antenna(
    const std::string &type,
    double res = NAN,
    double center_freq = 299792458.0,
    const arma::vec &element = {},
    const arma::uvec &panel = {},
    const arma::vec &spacing = {},
    const arma::uvec &pol = {},
    const arma::uvec &ports = {},
    const arma::mat &steer = {},
    const arma::mat &beams = {},
    const arma::vec &device_size = {},
    const arma::uvec &locations = {},
    const quadriga_lib::arrayant<dtype> &pattern = quadriga_lib::arrayant<dtype>(),
    bool separate_beams = false,
    bool apply_weights = false);
```

## Inputs:
- **`type`** — Antenna type:<br><br>
   | `type`                       | Antenna                                                                    | Inputs besides `res`, `center_freq`                                                |
   | ---------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
   | `omni`                       | Isotropic element, vertical polarization                                   | none                                                                               |
   | `xpol`                       | Two isotropic elements, vertical and horizontal polarization               | none                                                                               |
   | `dipole` (or `short-dipole`) | Short dipole, vertical polarization                                        | none                                                                               |
   | `half-wave-dipole`           | Half-wave dipole, vertical polarization                                    | none                                                                               |
   | `custom`                     | One element with given 3 dB beamwidths, peak gain equal to its directivity | `element`                                                                          |
   | `ula`                        | Uniform linear array of omni elements                                      | `panel`, `spacing`, `pol`, `ports`, `steer`, `pattern`, `apply_weights`            |
   | `3GPP` (or `3gpp`)           | Panel array of 3GPP TR 38.901 clause 7.3                                   | `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `pattern`, `apply_weights` |
   | `multibeam`                  | Panel array with beams towards given directions                            | `element`, `panel`, `spacing`, `pol`, `beams`, `separate_beams`, `apply_weights`   |
   | `3GPP-UT` (or `3gpp-ut`)     | Handheld UT of 3GPP TR 38.901 clause 7.3                                   | `element`, `pol`, `pattern`, `device_size`, `locations`                            |
   | `3GPP-CPE` (or `3gpp-cpe`)   | CPE of 3GPP TR 38.901 clause 7.3                                           | `element`, `pol`, `pattern`, `device_size`, `locations`                            |
- **`res`** *(optional)* — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; NaN or <= 0: 10° for `omni` and `xpol`, 1° otherwise; ignored if `pattern` is given
- **`center_freq`** *(optional)* — Center frequency in Hz; positive and finite; NaN: 299792458 (wavelength 1 m)
- **`element`** *(optional)* — Element parameters per type:
  - `custom` `[az_3dB, el_3dB, floor_dB]`: 3 dB beamwidths in degrees (> 0, Inf: no taper); floor of the power pattern as a positive attenuation in dB below the peak (10: 10 dB below the peak; Inf: no floor); default `[90, 90, Inf]`
  - `multibeam`: the same, default `[120, 120, Inf]`
  - `3GPP` `[phi_3dB, theta_3dB, SLA_V, A_max, G_max]` of TR 38.901 Table 7.3-1: beamwidths in degrees (> 0, Inf: no taper); SLA_V and A_max as positive attenuations in dB below the peak, as in the TR (>= 0, Inf: no limit); G_max in dBi (finite or NaN); default `[65, 65, 30, 30, 8]`; entries left off at the end or set to NaN take their default, where `theta_3dB` defaults to `phi_3dB` and `A_max` to `SLA_V` (e.g. `[90]` gives `[90, 90, 30, 30, 8]`); `G_max` left off is 8 dBi, NaN sets the peak gain to the directivity; not together with `pattern`
  - `3GPP-UT`: the same tuple, default `[125, 125, 22.5, 22.5, 5.3]` (Table 7.3-2); `3GPP-CPE`: default `[90, 90, 25, 25, 5]`; `[Inf, Inf, NaN, NaN, 0]` is isotropic
- **`panel`** *(optional)* — `[M, N, Mg, Ng]`: elements per column, columns per panel, panels per column, panels per row; default 1; at most 1e7 elements per polarization (`ula`: `panel = 8` is a vertical column, `[1, 8]` a horizontal row)
- **`spacing`** *(optional)* — `[d_V, d_H, d_gV, d_gH]` in wavelengths: vertical and horizontal element spacing, vertical and horizontal panel spacing; finite and >= 0 (0 places elements or panels at the same position); default `[0.5, d_V, M d_V, N d_H]`
- **`pol`** *(optional)* — `[polarization, model]`. Polarization: 1 = one element (ζ = 0°, vertical), 2 = two elements (ζ = 0° and 90°), 3 = two elements (ζ = +45° and -45°). Model of TR 38.901 clause 7.3.2: 1 = the element pattern is turned by ζ about its boresight (Model-1, eq. 7.3-3), 2 = the power pattern is kept and the field is split by ζ (Model-2, eq. 7.3-4, 7.3-5). Default `[1, 1]`; for `3GPP-UT` with polarization 2 or 3 the default model is 2, which gives the reference fields of the handheld model (eq. 7.3-6).
- **`ports`** *(optional)* — `[M_P, N_P]`: ports (TXRUs) per column and per row of a panel and polarization; must divide `M` and `N`; default `[M, N]` (one port per element)
- **`steer`** *(optional)* — `[tilt, pan]` in degrees, -90 to 90: electrical downtilt and pan of the sub-array weights; `[1, 2]` for all panels or `[Mg Ng, 2]` with one row per panel (panel order `mg`, `ng`); one column: pan 0; default 0; a non-zero tilt needs `M_P < M`, a non-zero pan `N_P < N`
- **`beams`** *(optional)* — `[2, n_beams]` or `[3, n_beams]`: azimuth in degrees; elevation in degrees; weight (linear; 2 rows: 1); finite; default `[0; 0; 1]`
- **`device_size`** *(optional)* — `[X, Y, Z]` in m (depth, width, height); NaN or left off: the default; `3GPP-UT` `[0.15, 0.07, 0]` with X, Y > 0 and Z = 0, `3GPP-CPE` `[0, 0.2, 0.2]` with X = 0 and Y, Z >= 0
- **`locations`** *(optional)* — TR location labels (`3GPP-UT` 1 to 8, `3GPP-CPE` 1 to 9), not 0-based indices; any order, repeats allowed; default all
- **`pattern`** *(optional)* — Element to use instead of the default element: its field patterns (`e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` with their grids) are copied to every element position; its `element_pos`, `coupling` and `center_freq` are not used, the generator sets them; a one-element pattern gets the polarizations of `pol`, one with several elements is a polarization set used as given and needs `pol` [1, 1]; empty (no elements): the element of the type
- **`separate_beams`** *(optional)* — If `true`, one port per beam (weights ignored)
- **`apply_weights`** *(optional)* — If `true`, combine each port into one pattern at its sub-array centre

## Returns:
- `quadriga_lib::arrayant<dtype>` — Antenna array object; panel arrays have `P M N Mg Ng` elements and `P M_P N_P Mg Ng` ports (`ula`, `3GPP`); with `apply_weights`, one element per port

## Example:
```
// Vertically polarized omni antenna, 1 degree grid; custom element with 30 x 20 degree beamwidth at 299792458 Hz
auto omni = quadriga_lib::generate_antenna("omni", 1.0);
auto beam = quadriga_lib::generate_antenna("custom", 1.0, NAN, {30.0, 20.0});

// Table 7.8-2, BS config 1: two 4 x 4 panels 2.5 wavelengths apart, +/-45 deg Model-2, one port per polarization
// and panel, 12 degree electrical downtilt
auto bs = quadriga_lib::generate_antenna("3GPP", 1.0, 6.0e9, {}, {4, 4, 1, 2}, {0.5, 0.5, 2.5, 2.5}, {3, 2}, {1, 1}, {12.0, 0.0});

// Handheld UT at the locations (1, 7, 3, 5) of Table 7.8-2A, Config B
auto ut = quadriga_lib::generate_antenna("3GPP-UT", 1.0, 7.0e9, {}, {}, {}, {}, {}, {}, {}, {}, {1, 7, 3, 5});
```

## See also:
- [[apply_port_weights]] (combines the ports into patterns)
- [[rotate_pattern]] (turns an antenna)
MD!*/

template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_antenna(const std::string &type, double res, double center_freq,
                                                             const arma::vec &element, const arma::uvec &panel, const arma::vec &spacing,
                                                             const arma::uvec &pol, const arma::uvec &ports, const arma::mat &steer,
                                                             const arma::mat &beams, const arma::vec &device_size, const arma::uvec &locations, const arrayant<dtype> &pattern,
                                                             bool separate_beams, bool apply_weights)
{
    // Inputs per type, as bit masks over the optional inputs below
    const char *name[12] = {"panel", "spacing", "ports", "pol", "steer", "element", "pattern", "beams", "separate_beams", "apply_weights",
                            "device_size", "locations"};
    const bool given[12] = {!panel.is_empty(), !spacing.is_empty(), !ports.is_empty(), !pol.is_empty(), !steer.is_empty(),
                            !element.is_empty(), pattern.n_elements() != 0, !beams.is_empty(), separate_beams, apply_weights,
                            !device_size.is_empty(), !locations.is_empty()};
    const unsigned ula = 0x5F, tgpp = 0x7F | 0x200, multibeam = 0x3AB, device = 0x8 | 0x20 | 0x40 | 0x400 | 0x800;
    const bool handheld = type == "3GPP-UT" || type == "3gpp-ut", cpe = type == "3GPP-CPE" || type == "3gpp-cpe";
    unsigned uses;
    if (type == "omni" || type == "xpol" || type == "dipole" || type == "short-dipole" || type == "half-wave-dipole")
        uses = 0;
    else if (type == "custom")
        uses = 0x20;
    else if (type == "ula")
        uses = ula | 0x200;
    else if (type == "3GPP" || type == "3gpp")
        uses = tgpp;
    else if (type == "multibeam")
        uses = multibeam;
    else if (handheld || cpe)
        uses = device;
    else
        throw std::invalid_argument("Input 'type' has an unsupported value '" + type + "'.");
    for (unsigned i = 0; i < 12; ++i)
        if (given[i] && !(uses & (1u << i)))
            throw std::invalid_argument(std::string("Input '") + name[i] + "' is not used by type '" + type + "'.");

    if (std::isnan(center_freq))
        center_freq = 299792458.0;
    qd_check_center_freq(center_freq);
    if (std::isnan(res) || (res <= 0.0 && std::isfinite(res)))
        res = (type == "omni" || type == "xpol") ? 10.0 : 1.0;

    quadriga_lib::arrayant<dtype> ant;
    if (type == "omni")
        ant = qd_generate_arrayant_omni<dtype>(res);
    else if (type == "xpol")
        ant = qd_generate_arrayant_xpol<dtype>(res);
    else if (type == "dipole" || type == "short-dipole")
        ant = qd_generate_arrayant_dipole<dtype>(res);
    else if (type == "half-wave-dipole")
        ant = qd_generate_arrayant_half_wave_dipole<dtype>(res);
    else if (type == "custom")
    {
        double az_3dB, el_3dB, rear_gain_lin;
        qd_custom_element_fill(element, 90.0, az_3dB, el_3dB, rear_gain_lin);
        ant = qd_generate_arrayant_custom<dtype>(az_3dB, el_3dB, rear_gain_lin, res);
    }
    else if (type == "ula")
    {
        ant = qd_generate_panel_array<dtype>(false, panel, center_freq, spacing, ports, pol, steer, {}, pattern, res, apply_weights);
        ant.name = "ula";
    }
    else if (type == "multibeam")
        ant = qd_generate_arrayant_multibeam<dtype>(panel, center_freq, spacing, pol, element, beams, res, separate_beams, apply_weights);
    else if (handheld || cpe)
        ant = qd_generate_device_array<dtype>(handheld, element, pol, pattern, device_size, locations, res);
    else
        ant = qd_generate_panel_array<dtype>(true, panel, center_freq, spacing, ports, pol, steer, element, pattern, res, apply_weights);

    ant.center_freq = dtype(center_freq);
    return ant;
}

template quadriga_lib::arrayant<float> quadriga_lib::generate_antenna(const std::string &type, double res, double center_freq,
                                                                      const arma::vec &element, const arma::uvec &panel, const arma::vec &spacing,
                                                                      const arma::uvec &pol, const arma::uvec &ports, const arma::mat &steer,
                                                                      const arma::mat &beams, const arma::vec &device_size, const arma::uvec &locations, const arrayant<float> &pattern,
                                                                      bool separate_beams, bool apply_weights);

template quadriga_lib::arrayant<double> quadriga_lib::generate_antenna(const std::string &type, double res, double center_freq,
                                                                       const arma::vec &element, const arma::uvec &panel, const arma::vec &spacing,
                                                                       const arma::uvec &pol, const arma::uvec &ports, const arma::mat &steer,
                                                                       const arma::mat &beams, const arma::vec &device_size, const arma::uvec &locations, const arrayant<double> &pattern,
                                                                       bool separate_beams, bool apply_weights);
