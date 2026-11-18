// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

// IRS arrayant dict: the coupling matrix holds the IRS weights; without one, C++ raises instead of using identity
static inline quadriga_lib::arrayant<double> qd_irs_dict2arrayant(const py::dict &irs)
{
    auto ant = qd_python_dict2arrayant(irs, true);
    auto given = [&](const char *key)
    { return irs.contains(key) && !irs[key].is_none() && py::array::ensure(irs[key]).size() != 0; };
    if (!given("coupling_re") && !given("coupling_im"))
        ant.coupling_re.reset(), ant.coupling_im.reset();
    return ant;
}

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# channel.irs
Calculate MIMO channel coefficients for IRS-assisted communication

- Computes channel coefficients and delays from two path segments: TX → IRS and IRS → RX
- IRS is modeled as a passive array; phase shifts are defined via its coupling matrix; codebook entry selected by `i_irs`
- The IRS coupling matrix holds the IRS weights and must be given: an `irs_array` or `irs_array_2` dict without
  `coupling_re` / `coupling_im` raises, unlike in the other functions, where it stands for identity
- Polarization coupling is applied via the 8-row transfer matrices `M_1`, `M_2`: 2x2 Jones matrices (rows = receive
  polarisation V/H, columns = transmit polarisation V/H; the IRS receives in segment 1 and transmits in segment 2),
  column-major with Re/Im interleaved: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx),
  rows 6-7 = (H rx, H tx)
- Each segment is computed by [[channel.spherical]]: NLOS path lengths are `max(path_length, d_poly)` with the
  polyline length `d_poly` of the segment; the delay of a combined path is the sum of both segment delays
- Output paths `n_path_irs` are all combinations of segment 1 and segment 2 paths that exceed `threshold_dB`; the pair
  of segment-1 path `i_1` and segment-2 path `i_2` has index `i_1 · n_path_2 + i_2` in `active_path` and
  `active_path_out`, and the output keeps this order
- A segment without paths (all its path inputs empty) gives empty outputs `(n_rx, n_tx, 0)`
- If `active_path` is provided, it overrides `threshold_dB` for path selection
- `center_freq = 0` gives the coefficients at f = 0: the delay phase exp(-j·2π·f·τ) is 1, the antenna and polarization phases remain; delays are computed as usual
- If `use_absolute_delays == False`, the LOS delay between the TX and RX phase centres (direct distance, not via the IRS)
  is subtracted from all paths; per-element delays can be slightly negative when the IRS is close to the TX-RX line

## Usage:
```
coeff, delay, active_path_out = quadriga_lib.channel.irs( tx_array, rx_array, irs_array, \
    tx_pos, tx_orientation, rx_pos, rx_orientation, irs_pos, irs_orientation, \
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1, \
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2, \
    i_irs, threshold_dB, center_freq, use_absolute_delays, angles, irs_array_2, active_path, use_kernel )

# Departure and arrival angles as well
coeff, delay, active_path_out, aod, eod, aoa, eoa = quadriga_lib.channel.irs( ..., angles=True )
```

## Inputs:
- **`tx_array`** — Transmit arrayant dict; `n_tx` = number of ports after element coupling; see [[generate]]
- **`rx_array`** — Receive arrayant dict; `n_rx` = number of ports after element coupling; see [[generate]]
- **`irs_array`** — IRS arrayant dict (TX-facing side) with `n_irs` elements; the coupling matrix holds the IRS
  weights, one codebook entry per column, `(n_irs, n_codebook)`; see [[generate]]
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `(3,)`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `(3,)`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`irs_pos`** — IRS position in Cartesian coordinates; `(3,)`
- **`irs_orientation`** — IRS orientation as Euler angles (bank, tilt, heading); `(3,)`
- **`fbs_pos_1`** — First-bounce scatterer positions for TX → IRS paths; `(3, n_path_1)`
- **`lbs_pos_1`** — Last-bounce scatterer positions for TX → IRS paths; `(3, n_path_1)`
- **`path_gain_1`** — Path gains in linear scale for TX → IRS paths; `(n_path_1,)`
- **`path_length_1`** — Total path lengths from TX to IRS phase center; values shorter than the polyline (e.g. 0) use
  the polyline length; `(n_path_1,)`
- **`M_1`** — Polarization transfer matrix for TX → IRS paths, row order as above; `(8, n_path_1)`
- **`fbs_pos_2`** — First-bounce scatterer positions for IRS → RX paths; `(3, n_path_2)`
- **`lbs_pos_2`** — Last-bounce scatterer positions for IRS → RX paths; `(3, n_path_2)`
- **`path_gain_2`** — Path gains in linear scale for IRS → RX paths; `(n_path_2,)`
- **`path_length_2`** — Total path lengths from IRS to RX phase center; values shorter than the polyline (e.g. 0) use
  the polyline length; `(n_path_2,)`
- **`M_2`** — Polarization transfer matrix for IRS → RX paths, row order as above; `(8, n_path_2)`
- **`i_irs`** — IRS codebook port index, 0-based (one port per column of the IRS coupling matrix); default: `0`
- **`threshold_dB`** — Gain threshold in dB; path combinations below this are discarded; default: `-140.0`
- **`center_freq`** — Center frequency in Hz; 0 gives the coefficients at f = 0 (see above); default: `0.0`
- **`use_absolute_delays`** — If `True`, delays include the LOS component; default: `False`
- **`angles`** — If `True`, also return the departure and arrival angles; `False` saves their memory; default: `False`
- **`irs_array_2`** — Second IRS arrayant dict for the RX-facing side (asymmetric IRS), with the same number of
  elements and the same coupling size as `irs_array`; default: `None` (`irs_array` for both sides)
- **`active_path`** — Mask selecting active path pairs, bool or numeric (nonzero = active); overrides `threshold_dB`
  when not empty; `(n_path_1 · n_path_2,)`; default: `None`
- **`use_kernel`** — Kernel of both segments, as in [[channel.spherical]]: 0 = auto, 1 = GENERIC, 2 = AVX2 (single
  precision; raises if unavailable); auto selects GENERIC in this version; `int`; default: 0

## Outputs:
- **`coeff`** — Complex channel coefficients; `(n_rx, n_tx, n_path_irs)`
- **`delay`** — Propagation delays in seconds; `(n_rx, n_tx, n_path_irs)`
- **`active_path_out`** — Mask of the path combinations in the output (can be passed as `active_path` to a later
  call); `bool`; `(n_path_1 · n_path_2,)`
- **`aod`** — Azimuth of departure in rad; `(n_rx, n_tx, n_path_irs)`; only with `angles=True`
- **`eod`** — Elevation of departure in rad; same shape; only with `angles=True`
- **`aoa`** — Azimuth of arrival in rad; same shape; only with `angles=True`
- **`eoa`** — Elevation of arrival in rad; same shape; only with `angles=True`

## See also:
- [[channel.spherical]] (single-segment spherical-wave channel)
- [[channel.planar]] (single-segment planar-wave channel)
- [[generate]] (antenna array generator)
MD!*/

py::tuple get_channels_irs(const py::dict &tx_array,
                           const py::dict &rx_array,
                           const py::dict &irs_array,
                           const py::array_t<double> &tx_pos,
                           const py::array_t<double> &tx_orientation,
                           const py::array_t<double> &rx_pos,
                           const py::array_t<double> &rx_orientation,
                           const py::array_t<double> &irs_pos,
                           const py::array_t<double> &irs_orientation,
                           const py::array_t<double> &fbs_pos_1,
                           const py::array_t<double> &lbs_pos_1,
                           const py::array_t<double> &path_gain_1,
                           const py::array_t<double> &path_length_1,
                           const py::array_t<double> &M_1,
                           const py::array_t<double> &fbs_pos_2,
                           const py::array_t<double> &lbs_pos_2,
                           const py::array_t<double> &path_gain_2,
                           const py::array_t<double> &path_length_2,
                           const py::array_t<double> &M_2,
                           const arma::uword i_irs,
                           const double threshold_dB,
                           const double center_freq,
                           py::handle use_absolute_delays,
                           py::handle angles,
                           py::handle irs_array_2,
                           py::handle active_path,
                           const int use_kernel)
{
    const bool want_absolute = qd_python_flag(use_absolute_delays, false, "use_absolute_delays");
    const bool want_angles = qd_python_flag(angles, false, "angles");

    // Antenna arrays
    const auto tx_array_a = qd_python_dict2arrayant(tx_array, true);
    const auto rx_array_a = qd_python_dict2arrayant(rx_array, true);
    const auto irs_array_a = qd_irs_dict2arrayant(irs_array);

    quadriga_lib::arrayant<double> irs_array_2_a;
    if (!irs_array_2.is_none())
    {
        if (!py::isinstance<py::dict>(irs_array_2))
            throw py::type_error("Input 'irs_array_2' must be a dict or None.");
        irs_array_2_a = qd_irs_dict2arrayant(py::reinterpret_borrow<py::dict>(irs_array_2));
    }

    // Positions and orientations
    const auto tx_pos_a = qd_python_numpy2arma_Col(tx_pos, true, false, "tx_pos", 3);
    const auto tx_orientation_a = qd_python_numpy2arma_Col(tx_orientation, true, false, "tx_orientation", 3);
    const auto rx_pos_a = qd_python_numpy2arma_Col(rx_pos, true, false, "rx_pos", 3);
    const auto rx_orientation_a = qd_python_numpy2arma_Col(rx_orientation, true, false, "rx_orientation", 3);
    const auto irs_pos_a = qd_python_numpy2arma_Col(irs_pos, true, false, "irs_pos", 3);
    const auto irs_orientation_a = qd_python_numpy2arma_Col(irs_orientation, true, false, "irs_orientation", 3);

    // Path data of both segments, sizes are validated by the C++ core
    const auto fbs_pos_1_a = qd_python_numpy2arma_Mat(fbs_pos_1, true);
    const auto lbs_pos_1_a = qd_python_numpy2arma_Mat(lbs_pos_1, true);
    const auto path_gain_1_a = qd_python_numpy2arma_Col(path_gain_1, true);
    const auto path_length_1_a = qd_python_numpy2arma_Col(path_length_1, true);
    const auto M_1_a = qd_python_numpy2arma_Mat(M_1, true);
    const auto fbs_pos_2_a = qd_python_numpy2arma_Mat(fbs_pos_2, true);
    const auto lbs_pos_2_a = qd_python_numpy2arma_Mat(lbs_pos_2, true);
    const auto path_gain_2_a = qd_python_numpy2arma_Col(path_gain_2, true);
    const auto path_length_2_a = qd_python_numpy2arma_Col(path_length_2, true);
    const auto M_2_a = qd_python_numpy2arma_Mat(M_2, true);

    // Active path mask: nonzero = active
    const auto active_path_in = qd_python_numpy2arma_Col<double>(active_path, true);
    std::vector<bool> active_path_a(active_path_in.n_elem);
    for (arma::uword i = 0; i < active_path_in.n_elem; ++i)
        active_path_a[i] = active_path_in[i] != 0.0;

    const quadriga_lib::arrayant<double> *p_irs_array_2 = irs_array_2.is_none() ? nullptr : &irs_array_2_a;
    const std::vector<bool> *p_active_path = active_path_a.empty() ? nullptr : &active_path_a;

    // Output sizes depend on the path selection: copied after the call
    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;
    std::vector<bool> active_path_out;

    quadriga_lib::get_channels_irs<double>(tx_array_a, rx_array_a, irs_array_a,
                                           tx_pos_a, tx_orientation_a, rx_pos_a, rx_orientation_a,
                                           irs_pos_a, irs_orientation_a,
                                           fbs_pos_1_a, lbs_pos_1_a, path_gain_1_a, path_length_1_a, M_1_a,
                                           fbs_pos_2_a, lbs_pos_2_a, path_gain_2_a, path_length_2_a, M_2_a,
                                           coeff_re, coeff_im, delay,
                                           i_irs, threshold_dB, center_freq, want_absolute,
                                           want_angles ? &aod : nullptr, want_angles ? &eod : nullptr,
                                           want_angles ? &aoa : nullptr, want_angles ? &eoa : nullptr,
                                           p_irs_array_2, p_active_path, &active_path_out, use_kernel);

    py::array_t<bool> active_path_out_py((py::ssize_t)active_path_out.size());
    bool *p_out = active_path_out_py.mutable_data();
    for (size_t i = 0; i < active_path_out.size(); ++i)
        p_out[i] = active_path_out[i];

    py::list out;
    out.append(qd_python_copy2numpy<double, std::complex<double>>(&coeff_re, &coeff_im));
    out.append(qd_python_copy2numpy(&delay));
    out.append(active_path_out_py);
    if (want_angles)
        for (const auto *c : {&aod, &eod, &aoa, &eoa})
            out.append(qd_python_copy2numpy(c));

    return py::tuple(out);
}
