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

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# get_channels_planar
Calculate MIMO channel coefficients for planar wave paths

- Computes complex channel coefficients and delays for all TX/RX element pairs across `n_path` propagation paths.
- Interpolates antenna patterns for both arrays, accounting for element positions, orientation, and polarization.
- LOS path detection is distance-based (angles ignored).
- Polarization transfer matrix `polmat` must be normalized; rows are interleaved real/imag components.
- If `add_fake_los_path` is true, one output path is always added (output size `n_path+1`): slot 0 holds the LOS path,
  with zero power if no LOS path was found. LOS = the strongest path whose `path_length` is within 0.1 mm of the TX-RX
  distance; it is moved to slot 0 and its original slot (input index + 1) holds a zero-power path. All other paths
  keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `[n_rx, n_tx, 0]`; with `add_fake_los_path = true` they
  hold only this zero-power LOS path, with the LOS delays per antenna pair.
- Polarization coupling: `polmat` is the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation V/H),
  column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx),
  rows 6-7 = (H rx, H tx).
- `path_length` is used as given for the delay and phase of the phase centres (no geometry check).
- `center_freq = 0` gives the coefficients at f = 0: the delay phase exp(-j·2π·f·τ) is 1, the antenna and polarization phases remain; delays are computed as usual.
- `use_absolute_delays = false` subtracts the straight-line TX↔RX distance (between the phase centres) from all path
  lengths before converting to delay; per-element delays of pairs that are closer to each other than the phase
  centres are then negative, down to about -(half the TX array size + half the RX array size) / speed of light; [[quantize_delays]] rejects
  delays at or below `-tap_spacing / 2` (at 5 ns taps: half sizes adding up to more than about 0.75 m), use
  `use_absolute_delays = true` in that case.

## Declaration:
```
void quadriga_lib::get_channels_planar(
    const quadriga_lib::arrayant<dtype> &tx_array,
    const quadriga_lib::arrayant<dtype> &rx_array,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &tx_orientation,
    const arma::Mat<dtype> &rx_pos,
    const arma::Mat<dtype> &rx_orientation,
    const arma::Col<dtype> &aod,
    const arma::Col<dtype> &eod,
    const arma::Col<dtype> &aoa,
    const arma::Col<dtype> &eoa,
    const arma::Col<dtype> &path_gain,
    const arma::Col<dtype> &path_length,
    const arma::Mat<dtype> &polmat,
    arma::Cube<dtype> &coeff_re,
    arma::Cube<dtype> &coeff_im,
    arma::Cube<dtype> &delay,
    double center_freq = 0.0,
    bool use_absolute_delays = false,
    bool add_fake_los_path = false,
    arma::Col<dtype> *rx_Doppler = nullptr,
    int use_kernel = 0);
```

## Inputs:
- **`tx_array`** — Transmit antenna array; `n_tx` elements
- **`rx_array`** — Receive antenna array; `n_rx` elements
- **`tx_pos`** — Transmitter position in [m]; `[3, 1]` or `[1, 3]`
- **`tx_orientation`** — Transmitter orientation (bank, tilt, heading) in [rad]; `[3, 1]` or `[1, 3]`
- **`rx_pos`** — Receiver position in [m]; `[3, 1]` or `[1, 3]`
- **`rx_orientation`** — Receiver orientation (bank, tilt, heading) in [rad]; `[3, 1]` or `[1, 3]`
- **`aod`** — Departure azimuth angles; `[n_path]`
- **`eod`** — Departure elevation angles; `[n_path]`
- **`aoa`** — Arrival azimuth angles; `[n_path]`
- **`eoa`** — Arrival elevation angles; `[n_path]`
- **`path_gain`** — Path gains in linear scale; `[n_path]`
- **`path_length`** — Path lengths from TX to RX phase center; `[n_path]`
- **`polmat`** — Polarization transfer matrix, Jones row order as above; `[8, n_path]`
- **`center_freq`** *(optional)* — Center frequency; 0 gives the coefficients at f = 0 (see above)
- **`use_absolute_delays`** *(optional)* — Include LOS delay offset in all paths
- **`add_fake_los_path`** *(optional)* — Always add one output path; slot 0 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot
- **`use_kernel`** *(optional)* — Kernel of the antenna interpolation: 0 = auto, 1 = GENERIC, 2 = AVX2 (single
  precision, also for `double`; throws if unavailable); other values throw; auto selects GENERIC in this version, see
  [[arrayant_interpolate]]

## Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path(+1)]`
- **`rx_Doppler`** *(optional)* — Doppler weights for moving RX; positive = moving toward path, negative = away; `[n_path(+1)]`

## See also:
- [[get_channels_planar_vec]] (several snapshots in one call)
- [[get_channels_spherical]] (spherical wave variant accounting for per-element angle differences)
- [[get_channels_ieee_indoor]] (for generating IEEE compliant channels using `get_channels_planar` internally)
- [[baseband_freq_response]] (for calculating the frequency response)
- [[quantize_delays]] (for mapping delays to a fixed grid)
- [[arrayant]] (antenna array class)
MD!*/

template <typename dtype>
void quadriga_lib::get_channels_planar(const quadriga_lib::arrayant<dtype> &tx_array, const quadriga_lib::arrayant<dtype> &rx_array,
                                       const arma::Mat<dtype> &tx_pos, const arma::Mat<dtype> &tx_orientation,
                                       const arma::Mat<dtype> &rx_pos, const arma::Mat<dtype> &rx_orientation,
                                       const arma::Col<dtype> &aod, const arma::Col<dtype> &eod, const arma::Col<dtype> &aoa, const arma::Col<dtype> &eoa,
                                       const arma::Col<dtype> &path_gain, const arma::Col<dtype> &path_length, const arma::Mat<dtype> &polmat,
                                       arma::Cube<dtype> &coeff_re, arma::Cube<dtype> &coeff_im, arma::Cube<dtype> &delay,
                                       double center_freq_in, bool use_absolute_delays, bool add_fake_los_path,
                                       arma::Col<dtype> *rx_Doppler, int use_kernel)
{
    const dtype center_freq = dtype(center_freq_in);
    constexpr dtype los_limit = dtype(1.0e-4);
    constexpr dtype zero = dtype(0.0);
    constexpr dtype one = dtype(1.0);
    constexpr dtype rC = dtype(1.0 / qd_speed_of_light);
    dtype wavelength = center_freq > zero ? dtype(qd_speed_of_light) / center_freq : one;
    dtype wave_number = dtype(2.0 * arma::datum::pi / qd_speed_of_light) * center_freq;

    // Positions and orientations
    const dtype *p_tp = qd_xyz(tx_pos, "tx_pos"), *p_to = qd_xyz(tx_orientation, "tx_orientation");
    const dtype *p_rp = qd_xyz(rx_pos, "rx_pos"), *p_ro = qd_xyz(rx_orientation, "rx_orientation");
    const dtype Tx = p_tp[0], Ty = p_tp[1], Tz = p_tp[2];
    const dtype Rx = p_rp[0], Ry = p_rp[1], Rz = p_rp[2];

    // Kernel of the antenna interpolation, checked before any early return
    const int kernel = qd_arrayant_interpolate_kernel(use_kernel, std::max(tx_array.e_theta_re.n_elem, rx_array.e_theta_re.n_elem));

    qd_check_alias(rx_Doppler, "rx_Doppler", &aod, &eod, &aoa, &eoa, &path_gain, &path_length, &polmat);
    qd_check_alias_list({&coeff_re, &coeff_im, &delay}, {"coeff_re", "coeff_im", "delay"},
                        &aod, &eod, &aoa, &eoa, &path_gain, &path_length, &polmat);
    qd_check_alias_outputs(&coeff_re, "coeff_re", rx_Doppler, "rx_Doppler");
    qd_check_alias_outputs(&coeff_im, "coeff_im", rx_Doppler, "rx_Doppler");
    qd_check_alias_outputs(&delay, "delay", rx_Doppler, "rx_Doppler");

    // 0 paths with add_fake_los_path: the output holds only the zero-power LOS path. Its slot does not depend on the
    // path data, except that its delays use the antenna response in the direction of path 0: a dummy NLOS path in the
    // LOS direction (1 m longer, never detected as LOS) gives the delays of a LOS path; its own slot is dropped.
    if (add_fake_los_path && aod.n_elem == 0 && eod.n_elem == 0 && aoa.n_elem == 0 && eoa.n_elem == 0 &&
        path_gain.n_elem == 0 && path_length.n_elem == 0 && polmat.n_cols == 0)
    {
        const dtype x = Rx - Tx, y = Ry - Ty, z = Rz - Tz;
        const dtype d_xy = std::sqrt(x * x + y * y);
        const arma::Col<dtype> aod_d = {std::atan2(y, x)}, eod_d = {std::atan2(z, d_xy)};
        const arma::Col<dtype> aoa_d = {std::atan2(-y, -x)}, eoa_d = {-eod_d.at(0)};
        const arma::Col<dtype> gain_d = {one}, length_d = {qd_calc_length(Tx, Ty, Tz, Rx, Ry, Rz) + one};
        arma::Mat<dtype> M_d(8, 1, arma::fill::zeros);
        M_d.at(0) = one, M_d.at(6) = -one;

        arma::Cube<dtype> cr, ci, dl;
        arma::Col<dtype> doppler;
        quadriga_lib::get_channels_planar<dtype>(tx_array, rx_array, tx_pos, tx_orientation, rx_pos, rx_orientation,
                                                 aod_d, eod_d, aoa_d, eoa_d, gain_d, length_d, M_d,
                                                 cr, ci, dl, center_freq, use_absolute_delays, true,
                                                 rx_Doppler == nullptr ? nullptr : &doppler, kernel);

        // Copy slot 0 to the outputs
        auto copy_slot = [](const arma::Cube<dtype> &src, arma::Cube<dtype> &dst)
        {
            if (dst.n_rows != src.n_rows || dst.n_cols != src.n_cols || dst.n_slices != 1ULL)
                dst.set_size(src.n_rows, src.n_cols, 1ULL);
            std::memcpy(dst.memptr(), src.memptr(), src.n_rows * src.n_cols * sizeof(dtype));
        };
        copy_slot(cr, coeff_re), copy_slot(ci, coeff_im), copy_slot(dl, delay);
        if (rx_Doppler != nullptr)
        {
            if (rx_Doppler->n_elem != 1ULL)
                rx_Doppler->set_size(1ULL);
            rx_Doppler->at(0) = doppler.at(0);
        }
        return;
    }

    qd_arrayant_check_tx_rx(tx_array, rx_array);

    arma::uword n_path = aod.n_elem;                                // Number of paths
    arma::uword n_out = add_fake_los_path ? n_path + 1ULL : n_path; // Number of output paths
    arma::uword n_tx = tx_array.e_theta_re.n_slices;                // Number of TX antenna elements before coupling
    arma::uword n_rx = rx_array.e_theta_re.n_slices;                // Number of RX antenna elements before coupling
    arma::uword n_links = n_rx * n_tx;                              // Number of MIMO channel coefficients per path (n_rx * n_tx)
    arma::uword n_tx_ports = tx_array.n_ports();                    // Number of TX antenna elements after coupling
    arma::uword n_rx_ports = rx_array.n_ports();                    // Number of RX antenna elements after coupling

    // Check if the number of paths is consistent in all inputs
    if (eod.n_elem != n_path || aoa.n_elem != n_path || eoa.n_elem != n_path || path_gain.n_elem != n_path || path_length.n_elem != n_path || polmat.n_cols != n_path)
        throw std::invalid_argument("Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length', and 'polmat' must have the same number of columns (n_paths).");
    if (polmat.n_rows != 8ULL && n_path != 0ULL)
        throw std::invalid_argument("Polarization transfer matrix 'polmat' must have 8 rows.");

    // Set the output size
    if (coeff_re.n_rows != n_rx_ports || coeff_re.n_cols != n_tx_ports || coeff_re.n_slices != n_out)
        coeff_re.set_size(n_rx_ports, n_tx_ports, n_out);
    if (coeff_im.n_rows != n_rx_ports || coeff_im.n_cols != n_tx_ports || coeff_im.n_slices != n_out)
        coeff_im.set_size(n_rx_ports, n_tx_ports, n_out);
    if (delay.n_rows != n_rx_ports || delay.n_cols != n_tx_ports || delay.n_slices != n_out)
        delay.set_size(n_rx_ports, n_tx_ports, n_out);

    // 0 paths without add_fake_los_path: empty outputs of size [n_rx_ports, n_tx_ports, 0]
    if (n_out == 0ULL)
    {
        if (rx_Doppler != nullptr && rx_Doppler->n_elem != 0ULL)
            rx_Doppler->set_size(0ULL);
        return;
    }

    // Map output memory to internal representation
    arma::Cube<dtype> CR, CI, DL;

    bool different_output_size = n_tx != n_tx_ports || n_rx != n_rx_ports;
    if (different_output_size) // Temporary internal data storage
    {
        CR.set_size(n_rx, n_tx, n_out);
        CI.set_size(n_rx, n_tx, n_out);
        DL.set_size(n_rx, n_tx, n_out);
    }

    dtype *p_coeff_re = different_output_size ? CR.memptr() : coeff_re.memptr();
    dtype *p_coeff_im = different_output_size ? CI.memptr() : coeff_im.memptr();
    dtype *p_delays = different_output_size ? DL.memptr() : delay.memptr();

    // Antenna interpolation requires matrix of size [1, n_ang], 1 angle per path
    // Angles are mapped to correct value ranges during antenna interpolation
    const arma::Mat<dtype> AOD = arma::Mat<dtype>(const_cast<dtype *>(aod.memptr()), 1, n_path, false, true);
    const arma::Mat<dtype> EOD = arma::Mat<dtype>(const_cast<dtype *>(eod.memptr()), 1, n_path, false, true);
    const arma::Mat<dtype> AOA = arma::Mat<dtype>(const_cast<dtype *>(aoa.memptr()), 1, n_path, false, true);
    const arma::Mat<dtype> EOA = arma::Mat<dtype>(const_cast<dtype *>(eoa.memptr()), 1, n_path, false, true);

    const dtype *p_gain = path_gain.memptr();
    const dtype *p_length = path_length.memptr();

    const arma::Cube<dtype> tx_ori(const_cast<dtype *>(p_to), 3, 1, 1), rx_ori(const_cast<dtype *>(p_ro), 3, 1, 1);
    const dtype dist_rx_tx = qd_calc_length(Tx, Ty, Tz, Rx, Ry, Rz);

    // Interpolate the antenna patterns for all paths
    arma::Mat<dtype> Vt_re(n_tx, n_path, arma::fill::none), Vt_im(n_tx, n_path, arma::fill::none),
        Ht_re(n_tx, n_path, arma::fill::none), Ht_im(n_tx, n_path, arma::fill::none),
        Vr_re(n_rx, n_path, arma::fill::none), Vr_im(n_rx, n_path, arma::fill::none),
        Hr_re(n_rx, n_path, arma::fill::none), Hr_im(n_rx, n_path, arma::fill::none),
        Pt(n_tx, n_path, arma::fill::none), Pr(n_rx, n_path, arma::fill::none);
    arma::Mat<dtype> AOA_loc, EOA_loc;

    // To calculate the Doppler weights, we need the arrival in local antenna-coordinates
    if (rx_Doppler != nullptr)
    {
        AOA_loc.set_size(n_rx, n_path);
        EOA_loc.set_size(n_rx, n_path);
        if (rx_Doppler->n_elem != n_out)
            rx_Doppler->zeros(n_out);
        else if (add_fake_los_path)
            rx_Doppler->at(0) = zero;
    }

    arma::uvec i_element = arma::regspace<arma::uvec>(0, n_tx - 1);
    arma::Mat<dtype> element_pos_interp(3ULL, n_tx, arma::fill::zeros);
    if (tx_array.element_pos.n_elem != 0ULL)
        std::memcpy(element_pos_interp.memptr(), tx_array.element_pos.memptr(), 3ULL * n_tx * sizeof(dtype));

    qd_arrayant_interpolate_dispatch<dtype>(tx_array, AOD, EOD, i_element, tx_ori, element_pos_interp,
                                            Vt_re, Vt_im, Ht_re, Ht_im, &Pt, nullptr, nullptr, nullptr, kernel);

    i_element = arma::regspace<arma::uvec>(0, n_rx - 1);
    element_pos_interp.zeros(3ULL, n_rx);
    if (rx_array.element_pos.n_elem != 0ULL)
        std::memcpy(element_pos_interp.memptr(), rx_array.element_pos.memptr(), 3ULL * n_rx * sizeof(dtype));

    qd_arrayant_interpolate_dispatch<dtype>(rx_array, AOA, EOA, i_element, rx_ori, element_pos_interp,
                                            Vr_re, Vr_im, Hr_re, Hr_im, &Pr, rx_Doppler == nullptr ? nullptr : &AOA_loc,
                                            rx_Doppler == nullptr ? nullptr : &EOA_loc, nullptr, kernel);

    element_pos_interp.reset();

    // Calculate the Doppler weights
    if (rx_Doppler != nullptr)
    {
        dtype *pAz = AOA_loc.memptr(), *pEl = EOA_loc.memptr();
        dtype *pD = add_fake_los_path ? rx_Doppler->memptr() + 1 : rx_Doppler->memptr();
        for (auto i = 0ULL; i < n_path; ++i)
            pD[i] = std::cos(pAz[i * n_rx]) * std::cos(pEl[i * n_rx]);
    }

    // Calculate the MIMO channel coefficients for each path
    arma::uword true_los_path = 0ULL;
    dtype true_los_power = zero;
    for (arma::uword j = 0ULL; j < n_out; ++j)
    {
        arma::uword i = add_fake_los_path ? (j == 0ULL ? 0ULL : j - 1ULL) : j;

        const dtype *pM = polmat.colptr(i);
        dtype *pVrr = Vr_re.colptr(i), *pVri = Vr_im.colptr(i),
              *pHrr = Hr_re.colptr(i), *pHri = Hr_im.colptr(i),
              *pVtr = Vt_re.colptr(i), *pVti = Vt_im.colptr(i),
              *pHtr = Ht_re.colptr(i), *pHti = Ht_im.colptr(i);

        dtype *pPt = Pt.colptr(i), *pPr = Pr.colptr(i);

        dtype path_amplitude = add_fake_los_path && j == 0ULL ? zero : std::sqrt(p_gain[i]);
        dtype path_length_local = add_fake_los_path && j == 0ULL ? dist_rx_tx : p_length[i];

        // LOS path detection
        if (std::abs(path_length_local - dist_rx_tx) < los_limit && add_fake_los_path && j != 0ULL && p_gain[i] > true_los_power)
            true_los_path = j, true_los_power = p_gain[i];

        arma::uword O = j * n_links; // Slice offset
        for (arma::uword t = 0ULL; t < n_tx; ++t)
            for (arma::uword r = 0ULL; r < n_rx; ++r)
            {
                arma::uword R = t * n_rx + r;

                dtype re = zero, im = zero;
                re += pVrr[r] * pM[0] * pVtr[t] - pVri[r] * pM[1] * pVtr[t] - pVrr[r] * pM[1] * pVti[t] - pVri[r] * pM[0] * pVti[t];
                re += pHrr[r] * pM[2] * pVtr[t] - pHri[r] * pM[3] * pVtr[t] - pHrr[r] * pM[3] * pVti[t] - pHri[r] * pM[2] * pVti[t];
                re += pVrr[r] * pM[4] * pHtr[t] - pVri[r] * pM[5] * pHtr[t] - pVrr[r] * pM[5] * pHti[t] - pVri[r] * pM[4] * pHti[t];
                re += pHrr[r] * pM[6] * pHtr[t] - pHri[r] * pM[7] * pHtr[t] - pHrr[r] * pM[7] * pHti[t] - pHri[r] * pM[6] * pHti[t];

                im += pVrr[r] * pM[1] * pVtr[t] + pVri[r] * pM[0] * pVtr[t] + pVrr[r] * pM[0] * pVti[t] - pVri[r] * pM[1] * pVti[t];
                im += pHrr[r] * pM[3] * pVtr[t] + pHri[r] * pM[2] * pVtr[t] + pHrr[r] * pM[2] * pVti[t] - pHri[r] * pM[3] * pVti[t];
                im += pVrr[r] * pM[5] * pHtr[t] + pVri[r] * pM[4] * pHtr[t] + pVrr[r] * pM[4] * pHti[t] - pVri[r] * pM[5] * pHti[t];
                im += pHrr[r] * pM[7] * pHtr[t] + pHri[r] * pM[6] * pHtr[t] + pHrr[r] * pM[6] * pHti[t] - pHri[r] * pM[7] * pHti[t];

                dtype dl = pPt[t] + path_length_local + pPr[r];
                dtype phase = wave_number * std::fmod(dl, wavelength);
                dtype cp = std::cos(phase), sp = std::sin(phase);

                p_coeff_re[O + R] = (re * cp + im * sp) * path_amplitude;
                p_coeff_im[O + R] = (-re * sp + im * cp) * path_amplitude;

                dl = use_absolute_delays ? dl : dl - dist_rx_tx;
                p_delays[O + R] = dl * rC;
            }
    }

    // Set the true LOS path as the first path
    if (add_fake_los_path && true_los_path != 0ULL)
    {
        // Coefficients
        dtype *ptrR = different_output_size ? CR.slice_memptr(true_los_path) : coeff_re.slice_memptr(true_los_path);
        dtype *ptrI = different_output_size ? CI.slice_memptr(true_los_path) : coeff_im.slice_memptr(true_los_path);
        std::memcpy(p_coeff_re, ptrR, n_links * sizeof(dtype));
        std::memcpy(p_coeff_im, ptrI, n_links * sizeof(dtype));
        for (arma::uword i = 0ULL; i < n_links; ++i)
            ptrR[i] = zero, ptrI[i] = zero;

        // Delays
        dtype *ptrD = different_output_size ? DL.slice_memptr(true_los_path) : delay.slice_memptr(true_los_path);
        std::vector<dtype> tmp(p_delays, p_delays + n_links);
        std::memcpy(p_delays, ptrD, n_links * sizeof(dtype));
        std::memcpy(ptrD, tmp.data(), n_links * sizeof(dtype));

        // Doppler weights
        if (rx_Doppler != nullptr)
            std::swap(rx_Doppler->at(0), rx_Doppler->at(true_los_path));
    }

    // Apply antenna element coupling; without different_output_size the data is in the outputs and the coupling runs
    // in place
    if (rx_array.coupling_re.n_elem != 0ULL || rx_array.coupling_im.n_elem != 0ULL ||
        tx_array.coupling_re.n_elem != 0ULL || tx_array.coupling_im.n_elem != 0ULL)
    {
        // Coupling powers abs(cpl)^2, each column (port) normalized to sum 1
        std::vector<dtype> rx_cpl(n_rx * n_rx_ports), tx_cpl(n_tx * n_tx_ports), temp;
        qd_power_mat(n_rx, n_rx_ports, rx_cpl.data(), true, rx_array.coupling_re.memptr(), rx_array.coupling_im.memptr());
        qd_power_mat(n_tx, n_tx_ports, tx_cpl.data(), true, tx_array.coupling_re.memptr(), tx_array.coupling_im.memptr());

        for (arma::uword j = 0ULL; j < n_out; ++j)
        {
            const arma::uword o = j * n_links; // Slice offset
            qd_coupling_apply_slot(rx_array.coupling_re.memptr(), rx_array.coupling_im.memptr(),
                                   tx_array.coupling_re.memptr(), tx_array.coupling_im.memptr(),
                                   rx_cpl.data(), tx_cpl.data(), n_rx, n_rx_ports, n_tx, n_tx_ports,
                                   &p_coeff_re[o], &p_coeff_im[o], &p_delays[o],
                                   different_output_size ? coeff_re.slice_memptr(j) : &p_coeff_re[o],
                                   different_output_size ? coeff_im.slice_memptr(j) : &p_coeff_im[o],
                                   different_output_size ? delay.slice_memptr(j) : &p_delays[o], temp);
        }
    }
}

template void quadriga_lib::get_channels_planar(const quadriga_lib::arrayant<float> &tx_array, const quadriga_lib::arrayant<float> &rx_array,
                                                const arma::Mat<float> &tx_pos, const arma::Mat<float> &tx_orientation,
                                                const arma::Mat<float> &rx_pos, const arma::Mat<float> &rx_orientation,
                                                const arma::Col<float> &aod, const arma::Col<float> &eod, const arma::Col<float> &aoa, const arma::Col<float> &eoa,
                                                const arma::Col<float> &path_gain, const arma::Col<float> &path_length, const arma::Mat<float> &polmat,
                                                arma::Cube<float> &coeff_re, arma::Cube<float> &coeff_im, arma::Cube<float> &delay,
                                                double center_freq, bool use_absolute_delays, bool add_fake_los_path,
                                                arma::Col<float> *rx_Doppler, int use_kernel);

template void quadriga_lib::get_channels_planar(const quadriga_lib::arrayant<double> &tx_array, const quadriga_lib::arrayant<double> &rx_array,
                                                const arma::Mat<double> &tx_pos, const arma::Mat<double> &tx_orientation,
                                                const arma::Mat<double> &rx_pos, const arma::Mat<double> &rx_orientation,
                                                const arma::Col<double> &aod, const arma::Col<double> &eod, const arma::Col<double> &aoa, const arma::Col<double> &eoa,
                                                const arma::Col<double> &path_gain, const arma::Col<double> &path_length, const arma::Mat<double> &polmat,
                                                arma::Cube<double> &coeff_re, arma::Cube<double> &coeff_im, arma::Cube<double> &delay,
                                                double center_freq, bool use_absolute_delays, bool add_fake_los_path,
                                                arma::Col<double> *rx_Doppler, int use_kernel);

/*!MD
# get_channels_planar_vec
Calculate MIMO channel coefficients for planar wave paths for several snapshots

- Batch form of [[get_channels_planar]]: computes `n_snap` snapshots in one call, the snapshots in parallel (OpenMP);
  each snapshot gives the same result as a single call with its inputs.
- `n_snap` is the length of the path inputs; all seven must have the same length. A snapshot's paths follow the rules
  of [[get_channels_planar]], including 0 paths.
- Positions and orientations have one column per snapshot; a single column applies to all snapshots.
- The output vectors get `n_snap` entries. Entries that already have the output size are written in place, without
  reallocating their memory.
- An error in a snapshot throws `std::invalid_argument` with the message of the single call, preceded by
  `"Snapshot <i>: "` (0-based, the lowest failing index). Invalid antennas and `use_kernel` throw without the prefix,
  also when `n_snap = 0`.
- With `coeff`, the coefficients are returned as complex values in place of `coeff_re` / `coeff_im`, which are left
  empty. Each thread computes one snapshot at a time into its own split buffers, so the split data of all snapshots is
  never held at once.

## Declaration:
```
void quadriga_lib::get_channels_planar_vec(
    const quadriga_lib::arrayant<dtype> &tx_array,
    const quadriga_lib::arrayant<dtype> &rx_array,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &tx_orientation,
    const arma::Mat<dtype> &rx_pos,
    const arma::Mat<dtype> &rx_orientation,
    const std::vector<arma::Col<dtype>> &aod,
    const std::vector<arma::Col<dtype>> &eod,
    const std::vector<arma::Col<dtype>> &aoa,
    const std::vector<arma::Col<dtype>> &eoa,
    const std::vector<arma::Col<dtype>> &path_gain,
    const std::vector<arma::Col<dtype>> &path_length,
    const std::vector<arma::Mat<dtype>> &polmat,
    std::vector<arma::Cube<dtype>> &coeff_re,
    std::vector<arma::Cube<dtype>> &coeff_im,
    std::vector<arma::Cube<dtype>> &delay,
    double center_freq = 0.0,
    bool use_absolute_delays = false,
    bool add_fake_los_path = false,
    std::vector<arma::Col<dtype>> *rx_Doppler = nullptr,
    std::vector<arma::Cube<std::complex<dtype>>> *coeff = nullptr,
    int use_kernel = 0);
```

## Inputs:
- **`tx_array`**, **`rx_array`** — Transmit and receive antenna arrays, as in [[get_channels_planar]]
- **`tx_pos`** — Transmitter positions in [m], finite; `[3, n_snap]` or `[3, 1]`
- **`tx_orientation`** — Transmitter orientations (bank, tilt, heading) in [rad], finite; `[3, n_snap]` or `[3, 1]`
- **`rx_pos`** — Receiver positions in [m], finite; `[3, n_snap]` or `[3, 1]`
- **`rx_orientation`** — Receiver orientations (bank, tilt, heading) in [rad], finite; `[3, n_snap]` or `[3, 1]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Departure and arrival angles in [rad]; vectors of length `n_snap`,
  entries `[n_path]`
- **`path_gain`**, **`path_length`** — Path gains (linear) and path lengths; vectors of length `n_snap`, entries `[n_path]`
- **`polmat`** — Polarization transfer matrices; vector of length `n_snap`, entries `[8, n_path]`
- **`center_freq`**, **`use_absolute_delays`**, **`add_fake_los_path`**, **`use_kernel`** *(optional)* — As in
  [[get_channels_planar]], for all snapshots

## Outputs:
- **`coeff_re`**, **`coeff_im`**, **`delay`** — Vectors of length `n_snap`, entries `[n_rx, n_tx, n_path(+1)]`
- **`rx_Doppler`** *(optional)* — Doppler weights for moving RX; vector of length `n_snap`, entries `[n_path(+1)]`
- **`coeff`** *(optional)* — Complex coefficients, replacing `coeff_re` and `coeff_im` (then left empty); vector of
  length `n_snap`, entries `[n_rx, n_tx, n_path(+1)]`

## See also:
- [[get_channels_planar]] (single snapshot)
- [[get_channels_spherical_vec]] (spherical wave variant)
MD!*/

template <typename dtype>
void quadriga_lib::get_channels_planar_vec(const quadriga_lib::arrayant<dtype> &tx_array, const quadriga_lib::arrayant<dtype> &rx_array,
                                           const arma::Mat<dtype> &tx_pos, const arma::Mat<dtype> &tx_orientation,
                                           const arma::Mat<dtype> &rx_pos, const arma::Mat<dtype> &rx_orientation,
                                           const std::vector<arma::Col<dtype>> &aod, const std::vector<arma::Col<dtype>> &eod,
                                           const std::vector<arma::Col<dtype>> &aoa, const std::vector<arma::Col<dtype>> &eoa,
                                           const std::vector<arma::Col<dtype>> &path_gain, const std::vector<arma::Col<dtype>> &path_length,
                                           const std::vector<arma::Mat<dtype>> &polmat,
                                           std::vector<arma::Cube<dtype>> &coeff_re, std::vector<arma::Cube<dtype>> &coeff_im,
                                           std::vector<arma::Cube<dtype>> &delay,
                                           double center_freq_in, bool use_absolute_delays, bool add_fake_los_path,
                                           std::vector<arma::Col<dtype>> *rx_Doppler,
                                           std::vector<arma::Cube<std::complex<dtype>>> *coeff, int use_kernel)
{
    const dtype center_freq = dtype(center_freq_in);
    const size_t n_snap = aod.size();
    if (eod.size() != n_snap || aoa.size() != n_snap || eoa.size() != n_snap ||
        path_gain.size() != n_snap || path_length.size() != n_snap || polmat.size() != n_snap)
        throw std::invalid_argument("Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length' and 'polmat' must have the same length.");

    const bool parallel = qd_channels_vec_check(tx_array, rx_array, tx_pos, tx_orientation, rx_pos, rx_orientation, n_snap, [&](size_t s)
                                                { return aod[s].n_elem; }, use_kernel);

    // Resizing keeps the existing entries, which the single call writes in place if they have the output size
    if (delay.size() != n_snap)
        delay.resize(n_snap);
    if (rx_Doppler != nullptr && rx_Doppler->size() != n_snap)
        rx_Doppler->resize(n_snap);

    qd_channels_snapshot_loop<dtype>(n_snap, parallel, coeff_re, coeff_im, coeff,
                                     [&](size_t s, arma::Cube<dtype> &cr, arma::Cube<dtype> &ci)
                                     {
                                         auto col = [s](const arma::Mat<dtype> &m)
                                         { return qd_channels_vec_col(m, s); };
                                         quadriga_lib::get_channels_planar<dtype>(tx_array, rx_array, col(tx_pos), col(tx_orientation), col(rx_pos), col(rx_orientation),
                                                                                  aod[s], eod[s], aoa[s], eoa[s], path_gain[s], path_length[s], polmat[s],
                                                                                  cr, ci, delay[s], center_freq, use_absolute_delays, add_fake_los_path,
                                                                                  rx_Doppler ? &rx_Doppler->at(s) : nullptr, use_kernel);
                                     });
}

template void quadriga_lib::get_channels_planar_vec(const quadriga_lib::arrayant<float> &tx_array, const quadriga_lib::arrayant<float> &rx_array,
                                                    const arma::Mat<float> &tx_pos, const arma::Mat<float> &tx_orientation,
                                                    const arma::Mat<float> &rx_pos, const arma::Mat<float> &rx_orientation,
                                                    const std::vector<arma::Col<float>> &aod, const std::vector<arma::Col<float>> &eod,
                                                    const std::vector<arma::Col<float>> &aoa, const std::vector<arma::Col<float>> &eoa,
                                                    const std::vector<arma::Col<float>> &path_gain, const std::vector<arma::Col<float>> &path_length,
                                                    const std::vector<arma::Mat<float>> &polmat,
                                                    std::vector<arma::Cube<float>> &coeff_re, std::vector<arma::Cube<float>> &coeff_im,
                                                    std::vector<arma::Cube<float>> &delay,
                                                    double center_freq, bool use_absolute_delays, bool add_fake_los_path,
                                                    std::vector<arma::Col<float>> *rx_Doppler,
                                                    std::vector<arma::Cube<std::complex<float>>> *coeff, int use_kernel);

template void quadriga_lib::get_channels_planar_vec(const quadriga_lib::arrayant<double> &tx_array, const quadriga_lib::arrayant<double> &rx_array,
                                                    const arma::Mat<double> &tx_pos, const arma::Mat<double> &tx_orientation,
                                                    const arma::Mat<double> &rx_pos, const arma::Mat<double> &rx_orientation,
                                                    const std::vector<arma::Col<double>> &aod, const std::vector<arma::Col<double>> &eod,
                                                    const std::vector<arma::Col<double>> &aoa, const std::vector<arma::Col<double>> &eoa,
                                                    const std::vector<arma::Col<double>> &path_gain, const std::vector<arma::Col<double>> &path_length,
                                                    const std::vector<arma::Mat<double>> &polmat,
                                                    std::vector<arma::Cube<double>> &coeff_re, std::vector<arma::Cube<double>> &coeff_im,
                                                    std::vector<arma::Cube<double>> &delay,
                                                    double center_freq, bool use_absolute_delays, bool add_fake_los_path,
                                                    std::vector<arma::Col<double>> *rx_Doppler,
                                                    std::vector<arma::Cube<std::complex<double>>> *coeff, int use_kernel);
