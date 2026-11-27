// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_arrayant.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_arrayant.hpp"

#include <cmath>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Power-weighted center of the elements of each port, [3, n_ports]; a port without finite power stays at the origin
template <typename dtype>
static arma::Mat<dtype> qd_port_centers(const arma::Mat<dtype> &element_pos, const arma::Mat<std::complex<dtype>> &pw)
{
    arma::Mat<dtype> centers(3, pw.n_cols, arma::fill::zeros);
    for (arma::uword o = 0ULL; o < pw.n_cols; ++o)
    {
        double total = 0.0, sum[3] = {0.0, 0.0, 0.0};
        for (arma::uword i = 0ULL; i < pw.n_rows; ++i)
        {
            const double p = (double)std::norm(pw.at(i, o));
            total += p;
            for (arma::uword d = 0ULL; d < 3ULL; ++d)
                sum[d] += p * (double)element_pos.at(d, i);
        }
        if (total > 0.0 && std::isfinite(total))
            for (arma::uword d = 0ULL; d < 3ULL; ++d)
                centers.at(d, o) = dtype(sum[d] / total);
    }
    return centers;
}

// Moves the phase reference of each port from the origin to its center; V and H are [n_ang, n_ports]
template <typename dtype>
static void qd_move_port_phase(arma::Mat<std::complex<dtype>> &V, arma::Mat<std::complex<dtype>> &H, const arma::Mat<dtype> &centers,
                               const arma::Mat<dtype> &azimuth, const arma::Mat<dtype> &elevation, double wave_no)
{
    for (arma::uword o = 0ULL; o < V.n_cols; ++o)
        for (arma::uword a = 0ULL; a < V.n_rows; ++a)
        {
            const double az = (double)azimuth[a], el = (double)elevation[a], cel = std::cos(el);
            const double proj = centers.at(0, o) * cel * std::cos(az) + centers.at(1, o) * cel * std::sin(az) + centers.at(2, o) * std::sin(el);
            const std::complex<dtype> shift(dtype(std::cos(wave_no * proj)), dtype(-std::sin(wave_no * proj))); // exp(j k d) with d = -proj
            V.at(a, o) *= shift, H.at(a, o) *= shift;
        }
}

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# apply_port_weights
Apply the port weights of an antenna array: one pattern per port

- Combines the elements of an array into one pattern per port, for beamforming and to speed up MIMO channel computation
- Each port pattern is the weighted sum of the element fields from the section overview, evaluated on the output grid
  and including the phase of each element position
- The result has one element per port and identity port weights; its elements are at the origin, or at the centers
  of their ports with `center_ports`

## Declarations:
```
quadriga_lib::arrayant<dtype> quadriga_lib::apply_port_weights(
    const arrayant<dtype> &ant,
    const double_list &azimuth_grid_new = {},
    const double_list &elevation_grid_new = {},
    double propagation_speed = 299792458.0,
    bool center_ports = false);

std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::apply_port_weights(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const double_list &azimuth_grid_new = {},
    const double_list &elevation_grid_new = {},
    const double_list &freq_new = {},
    double propagation_speed = 299792458.0,
    bool center_ports = false);
```

## Inputs:
- **`ant`** — Valid arrayant
- **`arrayant_vec`** — Vector form: non-empty vector of arrayant objects that passes [[is_valid]]
- **`azimuth_grid_new`** *(optional)* — Output azimuth grid in rad; empty: the input grid (of the first entry)
- **`elevation_grid_new`** *(optional)* — Output elevation grid in rad; empty: the input grid (of the first entry)
- **`freq_new`** *(optional)* — Vector form: output frequencies in Hz, positive and finite; empty: the `center_freq` of
  the entries, one output per input entry; `[n_freq_new]`
- **`propagation_speed`** *(optional)* — Wave propagation speed $c$ in m/s, positive and finite; the default is the
  speed of light for radio waves, use about 343 for acoustic arrays (e.g. from [[generate_speaker]])
- **`center_ports`** *(optional)* — If `true`, places each port $p$ at the power-weighted center of its elements,
  $\sum_i \lvert w_{i,p} \rvert^2 x_i / \sum_i \lvert w_{i,p} \rvert^2$, with the phase referenced there; a port with
  zero weights stays at the origin. A plane-wave channel is unchanged, and the slower phase over angle makes
  interpolation of coupled sub-arrays more accurate

## Returns:
- New `arrayant` with `n_ports` elements, each holding the pattern of that port; `name` is kept, and so is `center_freq`
  in the single form; vector form: `n_freq_new` such entries, each with its output frequency as `center_freq`

## Notes:
- **Frequency interpolation** (vector form): the fields are interpolated over frequency as in [[interpolate_pattern]],
  the port weights spherically between the two entries that bracket the output frequency. Output frequencies outside the
  entries take the nearest entry. The position phase uses the output frequency

## See also:
- [[generate_antenna]] (creates arrays with port weights)
- [[interpolate_pattern]] (evaluates the element patterns without the port weights)
- [[calc_directivity]], [[calc_beamwidth]] (work per element, so they measure the port patterns of the result)
- [[rotate_pattern]] (turns the combined pattern)
MD!*/

template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::apply_port_weights(const arrayant<dtype> &ant,
                                                                     const double_list &azimuth_grid_new_in,
                                                                     const double_list &elevation_grid_new_in,
                                                                     double propagation_speed_in,
                                                                     bool center_ports)
{
    const arma::Col<dtype> azimuth_grid_new = azimuth_grid_new_in.as<dtype>();
    const arma::Col<dtype> elevation_grid_new = elevation_grid_new_in.as<dtype>();
    const dtype propagation_speed = dtype(propagation_speed_in);
    std::string error_message = qd_arrayant_check(ant);
    if (error_message.length() != 0)
        throw std::invalid_argument("Input 'ant' is not valid: " + error_message);

    if (!(propagation_speed > dtype(0.0)) || !std::isfinite(propagation_speed))
        throw std::invalid_argument("Input 'propagation_speed' must be positive and finite.");

    // Get output angular grid
    bool new_azgrid = azimuth_grid_new.n_elem != 0;
    bool new_elgrid = elevation_grid_new.n_elem != 0;

    if (new_azgrid && !qd_in_range(azimuth_grid_new.memptr(), azimuth_grid_new.n_elem, dtype(-3.1415930), dtype(3.1415930), true, true))
        throw std::invalid_argument("Input 'azimuth_grid_new' must be sorted and in [-pi, pi].");

    if (new_elgrid && !qd_in_range(elevation_grid_new.memptr(), elevation_grid_new.n_elem, dtype(-1.5707965), dtype(1.5707965), true, true))
        throw std::invalid_argument("Input 'elevation_grid_new' must be sorted and in [-pi/2, pi/2].");

    const dtype *p_azimuth_grid = (new_azgrid) ? azimuth_grid_new.memptr() : ant.azimuth_grid.memptr();
    const dtype *p_elevation_grid = (new_elgrid) ? elevation_grid_new.memptr() : ant.elevation_grid.memptr();

    arma::uword n_azimuth_out = (new_azgrid) ? azimuth_grid_new.n_elem : ant.azimuth_grid.n_elem;
    arma::uword n_elevation_out = (new_elgrid) ? elevation_grid_new.n_elem : ant.elevation_grid.n_elem;
    arma::uword n_ang = n_azimuth_out * n_elevation_out;

    // Create list of angles for pattern interpolation
    arma::Mat<dtype> azimuth(1, n_ang, arma::fill::none);
    arma::Mat<dtype> elevation(1, n_ang, arma::fill::none);
    {
        dtype *p_azimuth = azimuth.memptr(), *p_elevation = elevation.memptr();
        for (arma::uword ia = 0ULL; ia < n_azimuth_out; ++ia)
            for (arma::uword ie = 0ULL; ie < n_elevation_out; ++ie)
                *p_azimuth++ = p_azimuth_grid[ia], *p_elevation++ = p_elevation_grid[ie];
    }

    // Get element positions
    arma::uword n_elements = ant.e_theta_re.n_slices;
    arma::Mat<dtype> element_pos_empty(3, n_elements, arma::fill::zeros);
    const auto p_element_pos = ant.element_pos.empty() ? &element_pos_empty : &ant.element_pos;

    // Interpolate the pattern data
    const arma::uvec i_element = arma::regspace<arma::uvec>(0, n_elements - 1);
    const arma::Cube<dtype> orientation(3, 1, 1, arma::fill::zeros);
    arma::Mat<dtype> V_re(n_elements, n_ang), V_im(n_elements, n_ang), H_re(n_elements, n_ang), H_im(n_elements, n_ang), dist(n_elements, n_ang);

    qd_interpolate_pattern_dispatch<dtype>(ant, azimuth, elevation, i_element, orientation, *p_element_pos,
                                            V_re, V_im, H_re, H_im, &dist, nullptr, nullptr, nullptr, 1);

    // Apply phase shift caused by element positions
    double lambda = (double)propagation_speed / (double)ant.center_freq;
    dtype wave_no = dtype(2.0 * arma::datum::pi / lambda);
    arma::Mat<std::complex<dtype>> phase(arma::cos(wave_no * dist), arma::sin(-wave_no * dist));
    arma::Mat<std::complex<dtype>> Vi(V_re, V_im), Hi(H_re, H_im);
    Vi = Vi % phase, Hi = Hi % phase;

    // Apply port weights
    arma::uword n_ports_out = ant.port_weights_re.empty() ? n_elements : ant.port_weights_re.n_cols;
    arma::Mat<std::complex<dtype>> Vo(n_ang, n_ports_out), Ho(n_ang, n_ports_out);
    arma::Mat<dtype> port_pos(3, n_ports_out, arma::fill::zeros);
    if (!ant.port_weights_re.empty())
    {
        arma::Mat<dtype> port_weights_im_empty(n_elements, n_ports_out, arma::fill::zeros);
        const auto p_port_weights_im = ant.port_weights_im.empty() ? &port_weights_im_empty : &ant.port_weights_im;
        arma::Mat<std::complex<dtype>> pw(ant.port_weights_re, *p_port_weights_im);
        if (center_ports)
            port_pos = qd_port_centers(*p_element_pos, pw);

        for (arma::uword i = 0ULL; i < n_elements; ++i)
        {
            arma::Col<std::complex<dtype>> vi = Vi.row(i).as_col(), hi = Hi.row(i).as_col();
            for (arma::uword o = 0ULL; o < n_ports_out; ++o)
            {
                std::complex<dtype> cpl = pw.at(i, o);
                Vo.col(o) += vi * cpl, Ho.col(o) += hi * cpl;
            }
        }
    }
    else // Identity port weights: one output per element, the angles along the rows
    {
        Vo = Vi.st(), Ho = Hi.st();
        if (center_ports)
            port_pos = *p_element_pos;
    }
    if (center_ports)
        qd_move_port_phase(Vo, Ho, port_pos, azimuth, elevation, (double)wave_no);

    // Write output data
    quadriga_lib::arrayant<dtype> output;
    {
        output.set_size(n_elevation_out, n_azimuth_out, n_ports_out, n_ports_out);

        std::memcpy(output.azimuth_grid.memptr(), p_azimuth_grid, n_azimuth_out * sizeof(dtype));
        std::memcpy(output.elevation_grid.memptr(), p_elevation_grid, n_elevation_out * sizeof(dtype));

        arma::Mat<dtype> cpy = arma::real(Vo);
        arma::uword n_bytes = n_elevation_out * n_azimuth_out * n_ports_out * sizeof(dtype);
        std::memcpy(output.e_theta_re.memptr(), cpy.memptr(), n_bytes);

        cpy = arma::imag(Vo);
        std::memcpy(output.e_theta_im.memptr(), cpy.memptr(), n_bytes);

        cpy = arma::real(Ho);
        std::memcpy(output.e_phi_re.memptr(), cpy.memptr(), n_bytes);

        cpy = arma::imag(Ho);
        std::memcpy(output.e_phi_im.memptr(), cpy.memptr(), n_bytes);

        output.element_pos = port_pos;
        output.port_weights_re.eye();
        output.port_weights_im.zeros();
        output.center_freq = ant.center_freq;
        output.name = ant.name;

        qd_arrayant_mark_valid(output);
    }

    return output;
}

template quadriga_lib::arrayant<float> quadriga_lib::apply_port_weights(const arrayant<float> &, const double_list &, const double_list &, double, bool);
template quadriga_lib::arrayant<double> quadriga_lib::apply_port_weights(const arrayant<double> &, const double_list &, const double_list &, double, bool);

template <typename dtype>
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::apply_port_weights(
    const std::vector<arrayant<dtype>> &arrayant_vec,
    const double_list &azimuth_grid_new_in,
    const double_list &elevation_grid_new_in,
    const double_list &freq_new_in,
    double propagation_speed_in,
    bool center_ports)
{
    const arma::Col<dtype> azimuth_grid_new = azimuth_grid_new_in.as<dtype>();
    const arma::Col<dtype> elevation_grid_new = elevation_grid_new_in.as<dtype>();
    const arma::Col<dtype> freq_new = freq_new_in.as<dtype>();
    const dtype propagation_speed = dtype(propagation_speed_in);
    
    if (arrayant_vec.empty())
        throw std::invalid_argument("Input 'arrayant_vec' cannot be empty.");

    if (!(propagation_speed > dtype(0.0)) || !std::isfinite(propagation_speed))
        throw std::invalid_argument("Input 'propagation_speed' must be positive and finite.");

    // The entries are validated by interpolate_pattern below, before their pattern data is used
    arma::uword n_freq = (arma::uword)arrayant_vec.size();

    // Validate optional output grids
    bool new_azgrid = azimuth_grid_new.n_elem != 0;
    bool new_elgrid = elevation_grid_new.n_elem != 0;
    bool new_fgrid = freq_new.n_elem != 0;

    if (new_azgrid && !qd_in_range(azimuth_grid_new.memptr(), azimuth_grid_new.n_elem, dtype(-3.1415930), dtype(3.1415930), true, true))
        throw std::invalid_argument("Input 'azimuth_grid_new' must be sorted and in [-pi, pi].");

    if (new_elgrid && !qd_in_range(elevation_grid_new.memptr(), elevation_grid_new.n_elem, dtype(-1.5707965), dtype(1.5707965), true, true))
        throw std::invalid_argument("Input 'elevation_grid_new' must be sorted and in [-pi/2, pi/2].");

    if (new_fgrid)
    {
        const dtype *p_freq = freq_new.memptr();
        for (arma::uword i = 0; i < freq_new.n_elem; ++i)
            if (!(p_freq[i] > dtype(0.0)) || !std::isfinite(p_freq[i]))
                throw std::invalid_argument("Input 'freq_new' must be positive and finite.");
    }

    // Output grids
    const dtype *p_azimuth_grid = new_azgrid ? azimuth_grid_new.memptr() : arrayant_vec[0].azimuth_grid.memptr();
    const dtype *p_elevation_grid = new_elgrid ? elevation_grid_new.memptr() : arrayant_vec[0].elevation_grid.memptr();
    arma::uword n_azimuth_out = new_azgrid ? azimuth_grid_new.n_elem : arrayant_vec[0].azimuth_grid.n_elem;
    arma::uword n_elevation_out = new_elgrid ? elevation_grid_new.n_elem : arrayant_vec[0].elevation_grid.n_elem;
    arma::uword n_ang = n_azimuth_out * n_elevation_out;

    // Output frequency grid: default = per-entry center_freq in input order
    arma::Col<dtype> freq_grid;
    if (new_fgrid)
        freq_grid = freq_new;
    else
    {
        freq_grid.set_size(n_freq);
        for (arma::uword i = 0; i < n_freq; ++i)
            freq_grid(i) = arrayant_vec[i].center_freq;
    }
    arma::uword n_freq_new = freq_grid.n_elem;

    // Build flat (azimuth, elevation) pair list (elevation varies fastest)
    arma::Mat<dtype> azimuth(1, n_ang, arma::fill::none);
    arma::Mat<dtype> elevation(1, n_ang, arma::fill::none);
    {
        dtype *p_az = azimuth.memptr(), *p_el = elevation.memptr();
        for (arma::uword ia = 0; ia < n_azimuth_out; ++ia)
            for (arma::uword ie = 0; ie < n_elevation_out; ++ie)
                *p_az++ = p_azimuth_grid[ia], *p_el++ = p_elevation_grid[ie];
    }

    // Element positions (identical across input entries per is_valid)
    arma::uword n_elements = arrayant_vec[0].n_elements();
    arma::Mat<dtype> element_pos_used = arrayant_vec[0].element_pos.is_empty()
                                            ? arma::Mat<dtype>(3, n_elements, arma::fill::zeros)
                                            : arrayant_vec[0].element_pos;

    // Interpolate fields across angles and freq; pass zero element_pos so no phase is baked in
    arma::Mat<dtype> zero_pos(3, n_elements, arma::fill::zeros);
    arma::Cube<dtype> V_re, V_im, H_re, H_im;
    quadriga_lib::interpolate_pattern<dtype>(arrayant_vec, azimuth, elevation, V_re, V_im, H_re, H_im,
                                                    freq_grid, {}, {}, zero_pos, nullptr, nullptr, 1);

    // Geometric dist matrix: dist(e, a) = element_pos[:,e] · wave_direction(az[a], el[a])
    arma::Mat<dtype> dist(n_elements, n_ang, arma::fill::none);
    {
        const dtype *p_pos = element_pos_used.memptr();
        const dtype *p_az = azimuth.memptr();
        const dtype *p_el = elevation.memptr();
        dtype *p_dist = dist.memptr(); // column-major: dist(e, a)
        for (arma::uword a = 0; a < n_ang; ++a)
        {
            dtype caz = std::cos(p_az[a]), saz = std::sin(p_az[a]);
            dtype cel = std::cos(p_el[a]), sel = std::sin(p_el[a]);
            dtype dx = cel * caz, dy = cel * saz, dz = sel;
            for (arma::uword e = 0; e < n_elements; ++e)
            {
                dtype x = p_pos[3 * e + 0], y = p_pos[3 * e + 1], z = p_pos[3 * e + 2];
                *p_dist++ = -(x * dx + y * dy + z * dz);
            }
        }
    }

    // Port weights setup
    bool no_port_weights = arrayant_vec[0].port_weights_re.is_empty();
    arma::uword n_ports_out = no_port_weights ? n_elements : arrayant_vec[0].port_weights_re.n_cols;

    // Input center frequencies and sort order (for port weight interpolation)
    arma::Col<double> f_in_d(n_freq);
    for (arma::uword i = 0; i < n_freq; ++i)
        f_in_d(i) = (double)arrayant_vec[i].center_freq;
    arma::uvec sort_idx = arma::sort_index(f_in_d);
    arma::Col<double> f_in_sorted(n_freq);
    for (arma::uword i = 0; i < n_freq; ++i)
        f_in_sorted(i) = f_in_d(sort_idx(i));

    // Build output entries
    std::vector<quadriga_lib::arrayant<dtype>> output(n_freq_new);
    const double C0 = (double)propagation_speed;

    for (arma::uword f = 0; f < n_freq_new; ++f)
    {
        double f_new = (double)freq_grid(f);
        dtype wave_no = dtype(2.0 * arma::datum::pi * f_new / C0);

        // Port weights for this output frequency (linear interp of complex port weights, clamped)
        arma::Mat<std::complex<dtype>> pw;
        if (no_port_weights)
        {
            pw = arma::Mat<std::complex<dtype>>(n_elements, n_ports_out, arma::fill::zeros);
            for (arma::uword k = 0; k < n_elements; ++k)
                pw(k, k) = std::complex<dtype>(dtype(1.0), dtype(0.0));
        }
        else
        {
            arma::uword i_lo = sort_idx(0), i_hi = sort_idx(0);
            double w = 0.0;
            if (f_new <= f_in_sorted(0))
                i_lo = i_hi = sort_idx(0), w = 0.0;
            else if (f_new >= f_in_sorted(n_freq - 1))
                i_lo = i_hi = sort_idx(n_freq - 1), w = 0.0;
            else
                for (arma::uword i = 1; i < n_freq; ++i)
                    if (f_in_sorted(i) >= f_new)
                    {
                        i_lo = sort_idx(i - 1);
                        i_hi = sort_idx(i);
                        double f_lo = f_in_sorted(i - 1), f_hi = f_in_sorted(i);
                        w = (f_new - f_lo) / (f_hi - f_lo);
                        break;
                    }

            const arma::Mat<dtype> &cre_lo = arrayant_vec[i_lo].port_weights_re;
            const arma::Mat<dtype> &cre_hi = arrayant_vec[i_hi].port_weights_re;
            arma::Mat<dtype> cim_lo = arrayant_vec[i_lo].port_weights_im.is_empty()
                                          ? arma::Mat<dtype>(n_elements, n_ports_out, arma::fill::zeros)
                                          : arrayant_vec[i_lo].port_weights_im;
            arma::Mat<dtype> cim_hi = arrayant_vec[i_hi].port_weights_im.is_empty()
                                          ? arma::Mat<dtype>(n_elements, n_ports_out, arma::fill::zeros)
                                          : arrayant_vec[i_hi].port_weights_im;

            arma::Mat<dtype> cre(n_elements, n_ports_out);
            arma::Mat<dtype> cim(n_elements, n_ports_out);
            const dtype *p_rlo = cre_lo.memptr(), *p_rhi = cre_hi.memptr();
            const dtype *p_ilo = cim_lo.memptr(), *p_ihi = cim_hi.memptr();
            dtype *p_ro = cre.memptr(), *p_io = cim.memptr();
            const arma::uword nE = n_elements * n_ports_out;
            for (arma::uword k = 0; k < nE; ++k)
                qd_slerp_complex(p_rlo[k], p_ilo[k], p_rhi[k], p_ihi[k], dtype(w), p_ro[k], p_io[k]);
            pw = arma::Mat<std::complex<dtype>>(cre, cim);
        }

        // Apply element-position phase shift: exp(-j * k * dist)
        arma::Mat<dtype> wd = wave_no * dist; // [n_elements, n_ang]
        arma::Mat<std::complex<dtype>> phase(arma::cos(wd), arma::sin(-wd));

        arma::Mat<dtype> V_re_s = V_re.slice(f); // [n_elements, n_ang]
        arma::Mat<dtype> V_im_s = V_im.slice(f);
        arma::Mat<dtype> H_re_s = H_re.slice(f);
        arma::Mat<dtype> H_im_s = H_im.slice(f);
        arma::Mat<std::complex<dtype>> Vi(V_re_s, V_im_s);
        arma::Mat<std::complex<dtype>> Hi(H_re_s, H_im_s);
        Vi = Vi % phase;
        Hi = Hi % phase;

        // Apply port weights: Vo(ang, port) = sum_e pw(e, port) * Vi(e, ang)
        arma::Mat<std::complex<dtype>> Vo(n_ang, n_ports_out, arma::fill::zeros);
        arma::Mat<std::complex<dtype>> Ho(n_ang, n_ports_out, arma::fill::zeros);
        for (arma::uword i = 0; i < n_elements; ++i)
        {
            arma::Col<std::complex<dtype>> vi = Vi.row(i).as_col();
            arma::Col<std::complex<dtype>> hi = Hi.row(i).as_col();
            for (arma::uword o = 0; o < n_ports_out; ++o)
            {
                std::complex<dtype> cpl = pw.at(i, o);
                Vo.col(o) += vi * cpl;
                Ho.col(o) += hi * cpl;
            }
        }

        arma::Mat<dtype> port_pos(3, n_ports_out, arma::fill::zeros);
        if (center_ports)
        {
            port_pos = qd_port_centers(element_pos_used, pw);
            qd_move_port_phase(Vo, Ho, port_pos, azimuth, elevation, 2.0 * arma::datum::pi * f_new / C0);
        }

        // Write output entry
        quadriga_lib::arrayant<dtype> &out = output[f];
        out.set_size(n_elevation_out, n_azimuth_out, n_ports_out, n_ports_out);

        std::memcpy(out.azimuth_grid.memptr(), p_azimuth_grid, n_azimuth_out * sizeof(dtype));
        std::memcpy(out.elevation_grid.memptr(), p_elevation_grid, n_elevation_out * sizeof(dtype));

        arma::uword n_bytes = n_elevation_out * n_azimuth_out * n_ports_out * sizeof(dtype);
        arma::Mat<dtype> cpy = arma::real(Vo);
        std::memcpy(out.e_theta_re.memptr(), cpy.memptr(), n_bytes);
        cpy = arma::imag(Vo);
        std::memcpy(out.e_theta_im.memptr(), cpy.memptr(), n_bytes);
        cpy = arma::real(Ho);
        std::memcpy(out.e_phi_re.memptr(), cpy.memptr(), n_bytes);
        cpy = arma::imag(Ho);
        std::memcpy(out.e_phi_im.memptr(), cpy.memptr(), n_bytes);

        out.element_pos = port_pos;
        out.port_weights_re.eye();
        out.port_weights_im.zeros();
        out.center_freq = dtype(f_new);
        out.name = arrayant_vec[0].name;

        qd_arrayant_mark_valid(out);
    }

    return output;
}

template std::vector<quadriga_lib::arrayant<float>> quadriga_lib::apply_port_weights(
    const std::vector<quadriga_lib::arrayant<float>> &,
    const double_list &, const double_list &, const double_list &, double, bool);

template std::vector<quadriga_lib::arrayant<double>> quadriga_lib::apply_port_weights(
    const std::vector<quadriga_lib::arrayant<double>> &,
    const double_list &, const double_list &, const double_list &, double, bool);
