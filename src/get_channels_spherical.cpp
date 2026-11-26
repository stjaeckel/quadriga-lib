// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>

#include "quadriga_channel.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_arrayant.hpp"
#include "qd_helpers_channel.hpp"

// Cartesian unit vectors (x, y, z) from azimuth and elevation angles
template <typename dtype>
static inline void qd_geo2cart(size_t N, const dtype *az, const dtype *el, dtype *x, dtype *y, dtype *z)
{
    for (size_t i = 0; i < N; ++i)
    {
        const double ca = std::cos((double)az[i]), sa = std::sin((double)az[i]);
        const double ce = std::cos((double)el[i]), se = std::sin((double)el[i]);
        x[i] = dtype(ce * ca), y[i] = dtype(ce * sa), z[i] = dtype(se);
    }
}

// Azimuth and elevation angles of the vectors (x, y, z); az or el may be nullptr
template <typename dtype>
static inline void qd_cart2geo(size_t N, const dtype *x, const dtype *y, const dtype *z, dtype *az, dtype *el)
{
    for (size_t i = 0; i < N; ++i)
    {
        const double xx = (double)x[i], yy = (double)y[i], zz = (double)z[i];
        const double le = 1.0 / std::sqrt(xx * xx + yy * yy + zz * zz);
        if (az != nullptr)
            az[i] = (dtype)std::atan2(yy * le, xx * le);
        if (el != nullptr)
        {
            const double zn = zz * le;
            el[i] = (dtype)std::asin(zn > 1.0 ? 1.0 : zn);
        }
    }
}

/*!SECTION
Channel generators
SECTION!*/

/*!SECTION_DESC
The functions of this section compute the channel coefficients of a link from its propagation paths and the antenna
arrays at both ends. They apply the antenna patterns, the polarization and the motion of the antennas, and return the
coefficients and delays that the functions of section "Channel data and processing" work with. The coefficients are
given per port pair: `n_tx` and `n_rx` count the ports of the arrays, after their port weights.<br><br>

**Generators:**

- [[get_channels_spherical]]: paths given by their first- and last-bounce scatterers; each antenna element sees the path
  from its own position (spherical wavefronts), so it covers large arrays and short distances
- [[get_channels_planar]]: paths given by their departure and arrival angles; all elements of an array see the same
  direction (planar wavefronts)
- [[get_channels_multifreq]]: as [[get_channels_spherical]], at several frequencies with frequency-dependent antennas,
  path gains and polarization, for radio and acoustic waves
- [[get_channels_irs]]: paths over an intelligent reflecting surface (IRS), from two path sets that meet at the IRS array
- [[get_channels_ieee_indoor]]: complete channels of the IEEE TGn, TGac, TGah and TGax indoor models

The paths come from a ray tracer (e.g. [[qrt_file_read]]), from the geometry-based stochastic model, or from the user.
For complete channels of a network layout, see [[get_channels_layout]].<br><br>

**Terms** (full list in the [Glossary](cpp_api.html#glossary)):

- **Path gain**: the linear power gain of a path without antennas, `[n_path]`
- **Path length**: the length of a path from the Tx to the Rx phase center in m, which sets its delay
- **Polarization transfer matrix** `polmat`: the Jones matrix of each path, `[8, n_path]` with interleaved real and
  imaginary parts (Glossary, "Polarization")
- **Absolute delays**: `use_absolute_delays = true` keeps the LOS delay in all paths; otherwise the delays are
  relative to the LOS path
- **Vector form**: [[get_channels_planar]] and [[get_channels_spherical]] have same-name overloads that compute
  several snapshots at once, one entry per snapshot
SECTION_DESC!*/

/*!MD
# get_channels_spherical
Calculate MIMO channel coefficients and delays for spherical wave propagation

- Coefficients and delays of one link for each pair of a Tx and an Rx port and each path; every element sees a path
  from its own position (spherical wavefronts), so delays and phases follow the geometry of each element pair
- A path runs from the Tx over its first-bounce scatterer (FBS) and last-bounce scatterer (LBS) to the Rx; the patterns
  of both arrays are interpolated at the departure and arrival angles of each element pair ([[interpolate_pattern]])
- Path $l$ of an element pair at the distance $d$ along the path has the coefficient and delay

  $$a_l = \sqrt{g_l} \; \mathbf{F}_r^T \mathbf{M}_l \, \mathbf{F}_t \; e^{-j 2\pi f_c d / c}, \qquad \tau_l = d / c$$

  with the path gain $g_l$, the Jones matrix $\mathbf{M}_l$ (`polmat`), the patterns $\mathbf{F} = (F_\theta, F_\phi)^T$
  of the Rx and Tx element, the center frequency $f_c$ and the speed of light $c$; the port weights then combine the
  elements into ports
- Vector form: the path inputs and the outputs have one entry per snapshot; the snapshots are computed in parallel

## Declarations:
```
void quadriga_lib::get_channels_spherical(
    const quadriga_lib::arrayant<dtype> &tx_array,
    const quadriga_lib::arrayant<dtype> &rx_array,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &tx_orientation,
    const arma::Mat<dtype> &rx_pos,
    const arma::Mat<dtype> &rx_orientation,
    const arma::Mat<dtype> &fbs_pos,
    const arma::Mat<dtype> &lbs_pos,
    const arma::Col<dtype> &path_gain,
    const arma::Col<dtype> &path_length,
    const arma::Mat<dtype> &polmat,
    arma::Cube<dtype> &coeff_re,
    arma::Cube<dtype> &coeff_im,
    arma::Cube<dtype> &delay,
    double center_freq = 0.0,
    bool use_absolute_delays = false,
    bool add_fake_los_path = false,
    arma::Cube<dtype> *aod = nullptr,
    arma::Cube<dtype> *eod = nullptr,
    arma::Cube<dtype> *aoa = nullptr,
    arma::Cube<dtype> *eoa = nullptr,
    int use_kernel = 0);

void quadriga_lib::get_channels_spherical(
    const quadriga_lib::arrayant<dtype> &tx_array,
    const quadriga_lib::arrayant<dtype> &rx_array,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &tx_orientation,
    const arma::Mat<dtype> &rx_pos,
    const arma::Mat<dtype> &rx_orientation,
    const std::vector<arma::Mat<dtype>> &fbs_pos,
    const std::vector<arma::Mat<dtype>> &lbs_pos,
    const std::vector<arma::Col<dtype>> &path_gain,
    const std::vector<arma::Col<dtype>> &path_length,
    const std::vector<arma::Mat<dtype>> &polmat,
    std::vector<arma::Cube<dtype>> &coeff_re,
    std::vector<arma::Cube<dtype>> &coeff_im,
    std::vector<arma::Cube<dtype>> &delay,
    double center_freq = 0.0,
    bool use_absolute_delays = false,
    bool add_fake_los_path = false,
    std::vector<arma::Cube<dtype>> *aod = nullptr,
    std::vector<arma::Cube<dtype>> *eod = nullptr,
    std::vector<arma::Cube<dtype>> *aoa = nullptr,
    std::vector<arma::Cube<dtype>> *eoa = nullptr,
    std::vector<arma::Cube<std::complex<dtype>>> *coeff = nullptr,
    int use_kernel = 0);
```

## Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array, valid ([[arrayant]]); `n_tx` and `n_rx` ports
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m, finite; `[3, 1]` or `[1, 3]`; vector form: `[3, 1]` (all
  snapshots) or `[3, n_snap]`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientation, Euler angles (bank, tilt, heading) in rad,
  finite; sizes as `tx_pos`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m; `[3, n_path]`
- **`path_gain`** — Path gains, linear power; `[n_path]`
- **`path_length`** — Path lengths in m between the Tx and Rx phase centres; a value below the Tx-FBS-LBS-Rx polyline
  (e.g. 0) gives the polyline length (Notes); `[n_path]`
- **`polmat`** — Jones matrices, real and imaginary parts interleaved ([Glossary](cpp_api.html#glossary),
  "Polarization"); `[8, n_path]`
- **`center_freq`** *(optional)* — Center frequency in Hz, finite and >= 0; 0 gives the coefficients at f = 0 (Notes)
- **`use_absolute_delays`** *(optional)* — Keep the delay of the Tx-Rx distance in all delays; `false`: delays relative
  to it (Notes)
- **`add_fake_los_path`** *(optional)* — Put the LOS path into an added slot 0, with zero power if there is none (Notes)
- **`use_kernel`** *(optional)* — Kernel of the antenna interpolation and of the coefficient computation: 0 = auto
  (GENERIC in this version), 1 = GENERIC, 2 = AVX2 (single-precision arithmetic, also for `double`; throws if
  unavailable)

## Outputs:
- **`coeff_re`** / **`coeff_im`** — Real and imaginary part of the coefficients; `[n_rx, n_tx, n_out]` with
  `n_out = n_path`, or `n_path + 1` with `add_fake_los_path`
- **`delay`** — Delays in s; `[n_rx, n_tx, n_out]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** *(optional)* — Azimuth and elevation of departure and arrival in rad, GCS;
  `[n_rx, n_tx, n_out]`
- **`coeff`** *(optional)* — Vector form: complex coefficients, replacing `coeff_re` / `coeff_im` (then left empty)

## Notes:
- **NLOS path length**: the length between the phase centres is $\max(\ell, d_\mathrm{poly})$ with `path_length`
  $\ell$ and the length $d_\mathrm{poly}$ of the Tx-FBS-LBS-Rx polyline; the excess, e.g. from more than two
  interactions as in [[qrt_file_read]] or [[coord2path]], is added to the FBS-LBS segment
- **LOS path**: the path whose length is closest to the Tx-Rx distance, within 0.1 mm; its element pairs use the
  direct line between the elements instead of the FBS and LBS
- **`add_fake_los_path`**: slot 0 holds the LOS path, with zero power if there is none; a LOS path among the inputs moves
  to slot 0 and leaves a zero-power path in its own slot (input index + 1); the other paths keep their order
- **0 paths** (all path inputs empty): outputs `[n_rx, n_tx, 0]`, or only the zero-power LOS path with
  `add_fake_los_path`
- **Relative delays**: the delay of the Tx-Rx distance is subtracted from all delays. Element pairs closer to each other
  than the phase centres then get negative delays, down to about $-(D_t + D_r) / (2c)$ with the array sizes $D$;
  [[quantize_delays]] rejects delays at or below `-tap_spacing / 2`, so large arrays need absolute delays
- **`center_freq = 0`**: the delay phase is 1, the antenna and polarization phases remain; the delays are unchanged
- **Port weights**: the delays and angles of a port pair are means over its element pairs, weighted by the
  port-weight powers (the angles via the mean direction)
- **Vector form**: `n_snap` is the length of the path inputs; entries of the output vectors that already have the output
  size are written in place. An error names the lowest failing snapshot (`Snapshot <i>: `, 0-based); invalid arrays and
  `use_kernel` throw without this prefix, also for `n_snap = 0`

## See also:
- [[get_channels_planar]] (paths given by angles, planar wavefronts)
- [[get_channels_multifreq]] (several frequencies)
- [[get_channels_irs]] (paths over an IRS)
- [[qrt_file_read]] (path data of many snapshots)
- [[freq_response]] (frequency response of the coefficients)
- [[quantize_delays]] (delays on a tap grid)
MD!*/

template <typename dtype>
void quadriga_lib::get_channels_spherical(const quadriga_lib::arrayant<dtype> &tx_array, const quadriga_lib::arrayant<dtype> &rx_array,
                                          const arma::Mat<dtype> &tx_pos, const arma::Mat<dtype> &tx_orientation,
                                          const arma::Mat<dtype> &rx_pos, const arma::Mat<dtype> &rx_orientation,
                                          const arma::Mat<dtype> &fbs_pos, const arma::Mat<dtype> &lbs_pos,
                                          const arma::Col<dtype> &path_gain, const arma::Col<dtype> &path_length, const arma::Mat<dtype> &polmat,
                                          arma::Cube<dtype> &coeff_re, arma::Cube<dtype> &coeff_im, arma::Cube<dtype> &delay,
                                          double center_freq_in, bool use_absolute_delays, bool add_fake_los_path,
                                          arma::Cube<dtype> *aod, arma::Cube<dtype> *eod, arma::Cube<dtype> *aoa, arma::Cube<dtype> *eoa,
                                          int use_kernel)
{
    if (!std::isfinite(center_freq_in) || center_freq_in < 0.0 || !std::isfinite((dtype)center_freq_in))
        throw std::invalid_argument("Input 'center_freq' must be finite and >= 0.");
    const dtype center_freq = dtype(center_freq_in);
    constexpr dtype zero = (dtype)0.0;
    constexpr dtype rC = dtype(1.0 / qd_speed_of_light);
    dtype wavelength = (center_freq > zero) ? (dtype)qd_speed_of_light / center_freq : (dtype)1.0;
    dtype wave_number = dtype(2.0 * arma::datum::pi / qd_speed_of_light) * center_freq;

    // Positions and orientations
    const dtype *p_tp = qd_xyz(tx_pos, "tx_pos"), *p_to = qd_xyz(tx_orientation, "tx_orientation");
    const dtype *p_rp = qd_xyz(rx_pos, "rx_pos"), *p_ro = qd_xyz(rx_orientation, "rx_orientation");
    const dtype Tx = p_tp[0], Ty = p_tp[1], Tz = p_tp[2];
    const dtype Rx = p_rp[0], Ry = p_rp[1], Rz = p_rp[2];

    // Kernel of the antenna interpolation and the coefficient computation, checked before any early return
    const int kernel = qd_interpolate_pattern_kernel(use_kernel, std::max(tx_array.e_theta_re.n_elem, rx_array.e_theta_re.n_elem));

    qd_check_alias_list({&coeff_re, &coeff_im, &delay, aod, eod, aoa, eoa},
                        {"coeff_re", "coeff_im", "delay", "aod", "eod", "aoa", "eoa"},
                        &fbs_pos, &lbs_pos, &path_gain, &path_length, &polmat);

    // 0 paths with add_fake_los_path: the output holds only the zero-power LOS path. Its slot (delays and angles from
    // the TX-RX geometry) does not depend on the path data, so a dummy NLOS path (1 m longer than the LOS path, never
    // detected as LOS) fills the input; its own slot is dropped.
    if (add_fake_los_path && fbs_pos.n_elem == 0 && lbs_pos.n_elem == 0 && path_gain.n_elem == 0 &&
        path_length.n_elem == 0 && polmat.n_elem == 0)
    {
        const arma::Mat<dtype> pos_d = {(Tx + Rx) * (dtype)0.5, (Ty + Ry) * (dtype)0.5, (Tz + Rz) * (dtype)0.5};
        const arma::Mat<dtype> fbs_d = pos_d.t();
        const arma::Col<dtype> gain_d = {(dtype)1.0}, length_d = {qd_calc_length(Tx, Ty, Tz, Rx, Ry, Rz) + (dtype)1.0};
        arma::Mat<dtype> M_d(8, 1, arma::fill::zeros);
        M_d.at(0) = (dtype)1.0, M_d.at(6) = (dtype)-1.0;

        arma::Cube<dtype> cr, ci, dl, ad, ed, aa, ea;
        quadriga_lib::get_channels_spherical<dtype>(tx_array, rx_array, tx_pos, tx_orientation, rx_pos, rx_orientation,
                                                    fbs_d, fbs_d, gain_d, length_d, M_d, cr, ci, dl,
                                                    center_freq, use_absolute_delays, true,
                                                    aod == nullptr ? nullptr : &ad, eod == nullptr ? nullptr : &ed,
                                                    aoa == nullptr ? nullptr : &aa, eoa == nullptr ? nullptr : &ea,
                                                    use_kernel);

        // Copy slot 0 to the outputs
        auto copy_slot = [](const arma::Cube<dtype> &src, arma::Cube<dtype> *dst)
        {
            if (dst == nullptr)
                return;
            dst->set_size(src.n_rows, src.n_cols, 1ULL);
            std::memcpy(dst->memptr(), src.memptr(), src.n_rows * src.n_cols * sizeof(dtype));
        };
        copy_slot(cr, &coeff_re), copy_slot(ci, &coeff_im), copy_slot(dl, &delay);
        copy_slot(ad, aod), copy_slot(ed, eod), copy_slot(aa, aoa), copy_slot(ea, eoa);
        return;
    }

    qd_arrayant_check_tx_rx(tx_array, rx_array);

    // 0 paths without add_fake_los_path: empty outputs of size [n_rx_ports, n_tx_ports, 0]
    if (fbs_pos.n_elem == 0 && lbs_pos.n_elem == 0 && path_gain.n_elem == 0 && path_length.n_elem == 0 && polmat.n_elem == 0)
    {
        const arma::uword n_tx_ports = tx_array.n_ports(), n_rx_ports = rx_array.n_ports();
        for (arma::Cube<dtype> *out : {&coeff_re, &coeff_im, &delay, aod, eod, aoa, eoa})
            if (out != nullptr)
                out->set_size(n_rx_ports, n_tx_ports, 0ULL);
        return;
    }

    // Check if the number of paths is consistent in all inputs
    if (fbs_pos.n_elem == 0 || lbs_pos.n_elem == 0 || path_gain.n_elem == 0 || path_length.n_elem == 0 || polmat.n_elem == 0)
        throw std::invalid_argument("Inputs 'fbs_pos', 'lbs_pos', 'path_gain', 'path_length' and 'polmat' cannot be empty.");
    if (fbs_pos.n_rows != 3 || lbs_pos.n_rows != 3 || polmat.n_rows != 8)
        throw std::invalid_argument("Inputs 'fbs_pos' and 'lbs_pos' must have 3 rows; 'polmat' must have 8 rows.");

    const arma::uword n_path = fbs_pos.n_cols;                         // Number of paths
    const arma::uword n_out = add_fake_los_path ? n_path + 1 : n_path; // Number of output paths
    const arma::uword n_tx = tx_array.e_theta_re.n_slices;             // Number of TX antenna elements before the port weights
    const arma::uword n_rx = rx_array.e_theta_re.n_slices;             // Number of RX antenna elements before the port weights
    const arma::uword n_links = n_rx * n_tx;                           // Number of MIMO channel coefficients per path (n_rx * n_tx)
    const arma::uword n_tx_ports = tx_array.n_ports();                 // Number of TX antenna elements after the port weights
    const arma::uword n_rx_ports = rx_array.n_ports();                 // Number of RX antenna elements after the port weights
    const arma::uword n_ports = n_tx_ports * n_rx_ports;               // Total number of ports

    if (lbs_pos.n_cols != n_path || path_gain.n_elem != n_path || path_length.n_elem != n_path || polmat.n_cols != n_path)
        throw std::invalid_argument("Inputs 'fbs_pos', 'lbs_pos', 'path_gain', 'path_length' and 'polmat' must have the same number of columns (n_paths).");

    // Set the output size
    for (arma::Cube<dtype> *out : {&coeff_re, &coeff_im, &delay, aod, eod, aoa, eoa})
        if (out != nullptr)
            out->set_size(n_rx_ports, n_tx_ports, n_out);

    // Map output memory to internal representation
    arma::Cube<dtype> CR, CI, DL;        // Internal map for delays and coefficients
    arma::Mat<dtype> AOD, EOD, AOA, EOA; // Angles [n_links, n_out], used when not written to the outputs

    bool different_output_size = n_tx != n_tx_ports || n_rx != n_rx_ports;
    if (different_output_size) // Temporary internal data storage
    {
        CR.set_size(n_rx, n_tx, n_out);
        CI.set_size(n_rx, n_tx, n_out);
        DL.set_size(n_rx, n_tx, n_out);
    }
    if (different_output_size || aod == nullptr)
        AOD.set_size(n_links, n_out);
    if (different_output_size || eod == nullptr)
        EOD.set_size(n_links, n_out);
    if (different_output_size || aoa == nullptr)
        AOA.set_size(n_links, n_out);
    if (different_output_size || eoa == nullptr)
        EOA.set_size(n_links, n_out);

    dtype *p_coeff_re = different_output_size ? CR.memptr() : coeff_re.memptr();
    dtype *p_coeff_im = different_output_size ? CI.memptr() : coeff_im.memptr();
    dtype *p_delays = different_output_size ? DL.memptr() : delay.memptr();
    dtype *p_aod = (different_output_size || aod == nullptr) ? AOD.memptr() : aod->memptr();
    dtype *p_eod = (different_output_size || eod == nullptr) ? EOD.memptr() : eod->memptr();
    dtype *p_aoa = (different_output_size || aoa == nullptr) ? AOA.memptr() : aoa->memptr();
    dtype *p_eoa = (different_output_size || eoa == nullptr) ? EOA.memptr() : eoa->memptr();

    // Angles and element-pair path lengths, element index lists for the interpolation
    arma::uvec i_tx_element, i_rx_element;
    arma::Mat<dtype> tx_element_pos_interp, rx_element_pos_interp;
    dtype dist_rx_tx;
    const arma::uword true_los_path = qd_channels_spherical_geometry(tx_array.element_pos, rx_array.element_pos, n_tx, n_rx,
                                                                     p_tp, p_to, p_rp, p_ro, fbs_pos.memptr(), lbs_pos.memptr(),
                                                                     path_length.memptr(), n_out, add_fake_los_path,
                                                                     p_aod, p_eod, p_aoa, p_eoa, p_delays,
                                                                     i_tx_element, tx_element_pos_interp,
                                                                     i_rx_element, rx_element_pos_interp, dist_rx_tx);
    const dtype *p_gain = path_gain.memptr();

    // Port weights, skipped when it is the identity
    const bool apply_element_port_weights = different_output_size ||
                                        !qd_port_weights_is_identity(tx_array.port_weights_re, tx_array.port_weights_im) ||
                                        !qd_port_weights_is_identity(rx_array.port_weights_re, rx_array.port_weights_im);

    // Port-weight powers abs(cpl)^2, each column (port) normalized to sum 1
    arma::Mat<dtype> tx_port_weights_pwr(n_tx, n_tx_ports, arma::fill::none);
    arma::Mat<dtype> rx_port_weights_pwr(n_rx, n_rx_ports, arma::fill::none);
    if (apply_element_port_weights)
    {
        qd_power_mat(n_tx, n_tx_ports, tx_port_weights_pwr.memptr(), true, tx_array.port_weights_re.memptr(), tx_array.port_weights_im.memptr());
        qd_power_mat(n_rx, n_rx_ports, rx_port_weights_pwr.memptr(), true, rx_array.port_weights_re.memptr(), rx_array.port_weights_im.memptr());
    }

    // Batch-interpolate antenna responses for all paths at once
    // Angles are already stored as [n_links, n_out] in p_aod/p_eod/p_aoa/p_eoa — wrap zero-copy
    const arma::Mat<dtype> AOD_all(p_aod, n_links, n_out, false, true);
    const arma::Mat<dtype> EOD_all(p_eod, n_links, n_out, false, true);
    const arma::Mat<dtype> AOA_all(p_aoa, n_links, n_out, false, true);
    const arma::Mat<dtype> EOA_all(p_eoa, n_links, n_out, false, true);

    arma::Mat<dtype> Vt_re(n_links, n_out, arma::fill::none), Vt_im(n_links, n_out, arma::fill::none);
    arma::Mat<dtype> Ht_re(n_links, n_out, arma::fill::none), Ht_im(n_links, n_out, arma::fill::none);
    arma::Mat<dtype> Vr_re(n_links, n_out, arma::fill::none), Vr_im(n_links, n_out, arma::fill::none);
    arma::Mat<dtype> Hr_re(n_links, n_out, arma::fill::none), Hr_im(n_links, n_out, arma::fill::none);

    const arma::Cube<dtype> tx_ori(const_cast<dtype *>(p_to), 3, 1, 1), rx_ori(const_cast<dtype *>(p_ro), 3, 1, 1);
    qd_interpolate_pattern_dispatch<dtype>(tx_array, AOD_all, EOD_all, i_tx_element, tx_ori, tx_element_pos_interp,
                                            Vt_re, Vt_im, Ht_re, Ht_im, nullptr, nullptr, nullptr, nullptr, kernel);
    qd_interpolate_pattern_dispatch<dtype>(rx_array, AOA_all, EOA_all, i_rx_element, rx_ori, rx_element_pos_interp,
                                            Vr_re, Vr_im, Hr_re, Hr_im, nullptr, nullptr, nullptr, nullptr, kernel);

    // Calculate the MIMO channel coefficients for each path
    const int n_out_i = (int)n_out;
#pragma omp parallel for if (n_out > 1 && n_out * n_links >= 512)
    for (int i_out = 0; i_out < n_out_i; ++i_out)
    {
        const arma::uword j = (arma::uword)i_out;
        const arma::uword i = add_fake_los_path ? (j == 0 ? 0 : j - 1) : j;
        const arma::uword o_slice = j * n_links; // Slice offset

        // Get column pointers into the precomputed interpolation results
        const dtype *pM = polmat.colptr(i);
        const dtype *pVrr = Vr_re.colptr(j), *pVri = Vr_im.colptr(j),
                    *pHrr = Hr_re.colptr(j), *pHri = Hr_im.colptr(j),
                    *pVtr = Vt_re.colptr(j), *pVti = Vt_im.colptr(j),
                    *pHtr = Ht_re.colptr(j), *pHti = Ht_im.colptr(j);
        const dtype path_amplitude = (add_fake_los_path && j == 0) ? zero : std::sqrt(p_gain[i]);

        // Compute MIMO coefficients (Jones product + phase rotation)
#if BUILD_WITH_AVX2
        if (kernel == 2)
            qd_coeff_combine_avx2(pVrr, pVri, pHrr, pHri, pVtr, pVti, pHtr, pHti,
                                  pM, &p_delays[o_slice], wave_number, wavelength, path_amplitude,
                                  &p_coeff_re[o_slice], &p_coeff_im[o_slice], n_links);
        else
#endif
            qd_coeff_combine(pVrr, pVri, pHrr, pHri, pVtr, pVti, pHtr, pHti,
                             pM, &p_delays[o_slice], wave_number, wavelength, path_amplitude,
                             &p_coeff_re[o_slice], &p_coeff_im[o_slice], n_links);

        // Convert per-link path lengths [m] to propagation delays [s]
        for (arma::uword R = 0ULL; R < n_links; ++R)
        {
            dtype dl = use_absolute_delays ? p_delays[o_slice + R] : p_delays[o_slice + R] - dist_rx_tx;
            p_delays[o_slice + R] = dl * rC;
        }

        // Apply port weights
        if (apply_element_port_weights)
        {
            const arma::uword N = (n_ports > n_links) ? n_ports : n_links;
            std::vector<dtype> temp(4 * N);
            dtype *tempX = temp.data(), *tempY = tempX + N, *tempZ = tempY + N, *tempT = tempZ + N;

            // Without different_output_size the data is in the outputs and the port weights run in place
            qd_port_weights_apply_slot(rx_array.port_weights_re.memptr(), rx_array.port_weights_im.memptr(),
                                   tx_array.port_weights_re.memptr(), tx_array.port_weights_im.memptr(),
                                   rx_port_weights_pwr.memptr(), tx_port_weights_pwr.memptr(), n_rx, n_rx_ports, n_tx, n_tx_ports,
                                   &p_coeff_re[o_slice], &p_coeff_im[o_slice], &p_delays[o_slice],
                                   different_output_size ? coeff_re.slice_memptr(j) : &p_coeff_re[o_slice],
                                   different_output_size ? coeff_im.slice_memptr(j) : &p_coeff_im[o_slice],
                                   different_output_size ? delay.slice_memptr(j) : &p_delays[o_slice], temp);

            // Angles: the port weights apply to the x, y, z components of the unit direction vectors
            auto couple_angles = [&](const dtype *p_az, const dtype *p_el, arma::Cube<dtype> *az, arma::Cube<dtype> *el)
            {
                if (az == nullptr && el == nullptr)
                    return;
                qd_geo2cart(n_links, &p_az[o_slice], &p_el[o_slice], tempX, tempY, tempZ);
                qd_multiply_3_mat(rx_port_weights_pwr.memptr(), tempX, tx_port_weights_pwr.memptr(), tempT, n_rx, n_rx_ports, n_tx, n_tx_ports);
                qd_multiply_3_mat(rx_port_weights_pwr.memptr(), tempY, tx_port_weights_pwr.memptr(), tempX, n_rx, n_rx_ports, n_tx, n_tx_ports);
                qd_multiply_3_mat(rx_port_weights_pwr.memptr(), tempZ, tx_port_weights_pwr.memptr(), tempY, n_rx, n_rx_ports, n_tx, n_tx_ports);

                // Without different_output_size, p_az / p_el are the outputs (or internal memory)
                dtype *az_out = different_output_size ? (az ? az->slice_memptr(j) : nullptr) : const_cast<dtype *>(&p_az[o_slice]);
                dtype *el_out = different_output_size ? (el ? el->slice_memptr(j) : nullptr) : const_cast<dtype *>(&p_el[o_slice]);
                qd_cart2geo(n_ports, tempT, tempX, tempY, az_out, el_out);
            };
            couple_angles(p_aod, p_eod, aod, eod);
            couple_angles(p_aoa, p_eoa, aoa, eoa);
        }
    }

    // Set the true LOS path as the first path
    if (add_fake_los_path && true_los_path != 0ULL)
    {
        dtype *ptrR = coeff_re.slice_memptr(true_los_path);
        dtype *ptrI = coeff_im.slice_memptr(true_los_path);

        std::memcpy(coeff_re.slice_memptr(0), ptrR, n_ports * sizeof(dtype));
        std::memcpy(coeff_im.slice_memptr(0), ptrI, n_ports * sizeof(dtype));

        for (arma::uword i = 0ULL; i < n_ports; ++i)
            ptrR[i] = zero, ptrI[i] = zero;
    }
}

template void quadriga_lib::get_channels_spherical(const quadriga_lib::arrayant<float> &tx_array, const quadriga_lib::arrayant<float> &rx_array,
                                                   const arma::Mat<float> &tx_pos, const arma::Mat<float> &tx_orientation,
                                                   const arma::Mat<float> &rx_pos, const arma::Mat<float> &rx_orientation,
                                                   const arma::Mat<float> &fbs_pos, const arma::Mat<float> &lbs_pos,
                                                   const arma::Col<float> &path_gain, const arma::Col<float> &path_length, const arma::Mat<float> &polmat,
                                                   arma::Cube<float> &coeff_re, arma::Cube<float> &coeff_im, arma::Cube<float> &delay,
                                                   double center_freq, bool use_absolute_delays, bool add_fake_los_path,
                                                   arma::Cube<float> *aod, arma::Cube<float> *eod, arma::Cube<float> *aoa, arma::Cube<float> *eoa,
                                                   int use_kernel);

template void quadriga_lib::get_channels_spherical(const quadriga_lib::arrayant<double> &tx_array, const quadriga_lib::arrayant<double> &rx_array,
                                                   const arma::Mat<double> &tx_pos, const arma::Mat<double> &tx_orientation,
                                                   const arma::Mat<double> &rx_pos, const arma::Mat<double> &rx_orientation,
                                                   const arma::Mat<double> &fbs_pos, const arma::Mat<double> &lbs_pos,
                                                   const arma::Col<double> &path_gain, const arma::Col<double> &path_length, const arma::Mat<double> &polmat,
                                                   arma::Cube<double> &coeff_re, arma::Cube<double> &coeff_im, arma::Cube<double> &delay,
                                                   double center_freq, bool use_absolute_delays, bool add_fake_los_path,
                                                   arma::Cube<double> *aod, arma::Cube<double> *eod, arma::Cube<double> *aoa, arma::Cube<double> *eoa,
                                                   int use_kernel);


template <typename dtype>
void quadriga_lib::get_channels_spherical(const quadriga_lib::arrayant<dtype> &tx_array, const quadriga_lib::arrayant<dtype> &rx_array,
                                          const arma::Mat<dtype> &tx_pos, const arma::Mat<dtype> &tx_orientation,
                                          const arma::Mat<dtype> &rx_pos, const arma::Mat<dtype> &rx_orientation,
                                          const std::vector<arma::Mat<dtype>> &fbs_pos, const std::vector<arma::Mat<dtype>> &lbs_pos,
                                          const std::vector<arma::Col<dtype>> &path_gain, const std::vector<arma::Col<dtype>> &path_length,
                                          const std::vector<arma::Mat<dtype>> &polmat,
                                          std::vector<arma::Cube<dtype>> &coeff_re, std::vector<arma::Cube<dtype>> &coeff_im,
                                          std::vector<arma::Cube<dtype>> &delay,
                                          double center_freq_in, bool use_absolute_delays, bool add_fake_los_path,
                                          std::vector<arma::Cube<dtype>> *aod, std::vector<arma::Cube<dtype>> *eod,
                                          std::vector<arma::Cube<dtype>> *aoa, std::vector<arma::Cube<dtype>> *eoa,
                                          std::vector<arma::Cube<std::complex<dtype>>> *coeff, int use_kernel)
{
    if (!std::isfinite(center_freq_in) || center_freq_in < 0.0 || !std::isfinite((dtype)center_freq_in))
        throw std::invalid_argument("Input 'center_freq' must be finite and >= 0.");
    const dtype center_freq = dtype(center_freq_in);
    const size_t n_snap = fbs_pos.size();
    if (lbs_pos.size() != n_snap || path_gain.size() != n_snap || path_length.size() != n_snap || polmat.size() != n_snap)
        throw std::invalid_argument("Inputs 'fbs_pos', 'lbs_pos', 'path_gain', 'path_length' and 'polmat' must have the same length.");

    const bool parallel = qd_channels_vec_check(tx_array, rx_array, tx_pos, tx_orientation, rx_pos, rx_orientation, n_snap, [&](size_t s)
                                                { return fbs_pos[s].n_cols; }, use_kernel);

    // Resizing keeps the existing entries, which the single call writes in place if they have the output size
    for (std::vector<arma::Cube<dtype>> *out : {&delay, aod, eod, aoa, eoa})
        if (out != nullptr && out->size() != n_snap)
            out->resize(n_snap);

    qd_channels_snapshot_loop<dtype>(n_snap, parallel, coeff_re, coeff_im, coeff,
                                     [&](size_t s, arma::Cube<dtype> &cr, arma::Cube<dtype> &ci)
                                     {
                                         auto col = [s](const arma::Mat<dtype> &m)
                                         { return qd_channels_vec_col(m, s); };
                                         quadriga_lib::get_channels_spherical<dtype>(tx_array, rx_array, col(tx_pos), col(tx_orientation), col(rx_pos), col(rx_orientation),
                                                                                     fbs_pos[s], lbs_pos[s], path_gain[s], path_length[s], polmat[s],
                                                                                     cr, ci, delay[s], center_freq, use_absolute_delays, add_fake_los_path,
                                                                                     aod ? &aod->at(s) : nullptr, eod ? &eod->at(s) : nullptr,
                                                                                     aoa ? &aoa->at(s) : nullptr, eoa ? &eoa->at(s) : nullptr, use_kernel);
                                     });
}

template void quadriga_lib::get_channels_spherical(const quadriga_lib::arrayant<float> &tx_array, const quadriga_lib::arrayant<float> &rx_array,
                                                   const arma::Mat<float> &tx_pos, const arma::Mat<float> &tx_orientation,
                                                   const arma::Mat<float> &rx_pos, const arma::Mat<float> &rx_orientation,
                                                   const std::vector<arma::Mat<float>> &fbs_pos, const std::vector<arma::Mat<float>> &lbs_pos,
                                                   const std::vector<arma::Col<float>> &path_gain, const std::vector<arma::Col<float>> &path_length,
                                                   const std::vector<arma::Mat<float>> &polmat,
                                                   std::vector<arma::Cube<float>> &coeff_re, std::vector<arma::Cube<float>> &coeff_im,
                                                   std::vector<arma::Cube<float>> &delay,
                                                   double center_freq, bool use_absolute_delays, bool add_fake_los_path,
                                                   std::vector<arma::Cube<float>> *aod, std::vector<arma::Cube<float>> *eod,
                                                   std::vector<arma::Cube<float>> *aoa, std::vector<arma::Cube<float>> *eoa,
                                                   std::vector<arma::Cube<std::complex<float>>> *coeff, int use_kernel);

template void quadriga_lib::get_channels_spherical(const quadriga_lib::arrayant<double> &tx_array, const quadriga_lib::arrayant<double> &rx_array,
                                                   const arma::Mat<double> &tx_pos, const arma::Mat<double> &tx_orientation,
                                                   const arma::Mat<double> &rx_pos, const arma::Mat<double> &rx_orientation,
                                                   const std::vector<arma::Mat<double>> &fbs_pos, const std::vector<arma::Mat<double>> &lbs_pos,
                                                   const std::vector<arma::Col<double>> &path_gain, const std::vector<arma::Col<double>> &path_length,
                                                   const std::vector<arma::Mat<double>> &polmat,
                                                   std::vector<arma::Cube<double>> &coeff_re, std::vector<arma::Cube<double>> &coeff_im,
                                                   std::vector<arma::Cube<double>> &delay,
                                                   double center_freq, bool use_absolute_delays, bool add_fake_los_path,
                                                   std::vector<arma::Cube<double>> *aod, std::vector<arma::Cube<double>> *eod,
                                                   std::vector<arma::Cube<double>> *aoa, std::vector<arma::Cube<double>> *eoa,
                                                   std::vector<arma::Cube<std::complex<double>>> *coeff, int use_kernel);
