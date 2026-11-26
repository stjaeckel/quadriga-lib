// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

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

- **Path gain**: the linear power gain of a path without antennas, `[n_path, 1]`
- **Path length**: the length of a path from the Tx to the Rx phase center in m, which sets its delay
- **Polarization transfer matrix** `polmat`: the Jones matrix of each path, `[8, n_path]` with interleaved real and
  imaginary parts, rows 1-2 (rx V, tx V), 3-4 (rx H, tx V), 5-6 (rx V, tx H), 7-8 (rx H, tx H) (Glossary,
  "Polarization")
- **Absolute delays**: `use_absolute_delays = true` keeps the LOS delay in all paths; otherwise the delays are
  relative to the LOS path
- **Vector form**: [[get_channels_planar]] and [[get_channels_spherical]] take the path inputs of several snapshots as
  cell arrays, one cell per snapshot, and return cell arrays
SECTION_DESC!*/

/*!MD
# GET_CHANNELS_SPHERICAL
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
- Vector form: with cell arrays as path inputs, one cell per snapshot, the outputs are cell arrays; the snapshots are
  computed in parallel

## Usage:
```
[ coeff_re, coeff_im, delay, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_spherical( tx_array, ...
    rx_array, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, ...
    path_length, polmat, center_freq, use_absolute_delays, add_fake_los_path, use_kernel );
```

## Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array, struct, fields in [[generate_antenna]]; `n_tx` and `n_rx` ports
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m, finite; 3 elements; vector form: 3 elements (all snapshots) or
  `[3, n_snap]`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientation, Euler angles (bank, tilt, heading) in rad,
  finite; sizes as `tx_pos`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m; `[3, n_path]`; vector form: cell
  arrays with `n_snap` entries, as [[qrt_file_read]] returns them
- **`path_gain`** — Path gains, linear power; `[n_path, 1]`; vector form: cell array
- **`path_length`** — Path lengths in m between the Tx and Rx phase centres; a value below the Tx-FBS-LBS-Rx polyline
  (e.g. 0) gives the polyline length (Notes); `[n_path, 1]`; vector form: cell array
- **`polmat`** — Jones matrices, real and imaginary parts interleaved (section description); `[8, n_path]`; vector
  form: cell array
- **`center_freq`** — Center frequency in Hz, finite and >= 0; 0 gives the coefficients at f = 0 (Notes); default: 0
- **`use_absolute_delays`** — Keep the delay of the Tx-Rx distance in all delays; `false`: delays relative to it
  (Notes); default: `false`
- **`add_fake_los_path`** — Put the LOS path into an added slot 1, with zero power if there is none (Notes); default:
  `false`
- **`use_kernel`** — Kernel of the antenna interpolation and of the coefficient computation: 0 = auto (GENERIC in this
  version), 1 = GENERIC, 2 = AVX2 (single-precision arithmetic; error if unavailable); default: 0

## Outputs:
- **`coeff_re`**, **`coeff_im`** — Real and imaginary part of the coefficients; `[n_rx, n_tx, n_out]` with
  `n_out = n_path`, or `n_path + 1` with `add_fake_los_path`; vector form: cell arrays `[n_snap, 1]` of such arrays
- **`delay`** — Delays in s; `[n_rx, n_tx, n_out]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Azimuth and elevation of departure and arrival in rad, GCS;
  `[n_rx, n_tx, n_out]`

## Notes:
- **NLOS path length**: the length between the phase centres is $\max(\ell, d_\mathrm{poly})$ with `path_length`
  $\ell$ and the length $d_\mathrm{poly}$ of the Tx-FBS-LBS-Rx polyline; the excess, e.g. from more than two
  interactions as in [[qrt_file_read]], is added to the FBS-LBS segment
- **LOS path**: the path whose length is closest to the Tx-Rx distance, within 0.1 mm; its element pairs use the
  direct line between the elements instead of the FBS and LBS
- **`add_fake_los_path`**: slot 1 holds the LOS path, with zero power if there is none; a LOS path among the inputs moves
  to slot 1 and leaves a zero-power path in its own slot (input index + 1); the other paths keep their order
- **0 paths** (all path inputs empty): outputs `[n_rx, n_tx, 0]`, or only the zero-power LOS path with
  `add_fake_los_path`
- **Relative delays**: the delay of the Tx-Rx distance is subtracted from all delays. Element pairs closer to each other
  than the phase centres then get negative delays, down to about $-(D_t + D_r) / (2c)$ with the array sizes $D$;
  [[quantize_delays]] rejects delays at or below `-tap_spacing / 2`, so large arrays need absolute delays
- **`center_freq = 0`**: the delay phase is 1, the antenna and polarization phases remain; the delays are unchanged
- **Port weights**: the delays and angles of a port pair are means over its element pairs, weighted by the
  port-weight powers (the angles via the mean direction)
- **Vector form**: `n_snap` is the length of the cell arrays; an error names the lowest failing snapshot
  (`Snapshot <i>: `, 1-based); invalid arrays and `use_kernel` give errors without this prefix

## See also:
- [[get_channels_planar]] (paths given by angles, planar wavefronts)
- [[get_channels_multifreq]] (several frequencies)
- [[get_channels_irs]] (paths over an IRS)
- [[qrt_file_read]] (path data of many snapshots)
- [[freq_response]] (frequency response of the coefficients)
- [[quantize_delays]] (delays on a tap grid)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 11 || nrhs > 15)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    auto ant_tx = qd_mex_struct2arrayant(prhs[0], false, false, "tx_array");
    auto ant_rx = qd_mex_struct2arrayant(prhs[1], false, false, "rx_array");

    const double center_freq = (nrhs < 12) ? 0.0 : qd_mex_get_scalar<double>(prhs[11], "center_freq", 0.0);
    const bool use_absolute_delays = (nrhs < 13) ? false : qd_mex_get_scalar<bool>(prhs[12], "use_absolute_delays", false);
    const bool add_fake_los_path = (nrhs < 14) ? false : qd_mex_get_scalar<bool>(prhs[13], "add_fake_los_path", false);
    const int use_kernel = (nrhs < 15) ? 0 : qd_mex_get_scalar<int>(prhs[14], "use_kernel", 0);

    arma::uword n_ports_tx = ant_tx.n_ports();
    arma::uword n_ports_rx = ant_rx.n_ports();

    // Batch mode: path data as cell arrays, one entry per snapshot
    if (mxIsCell(prhs[6]))
    {
        // Numeric data of class double is not copied
        auto get_Mat = [](const mxArray *a, const char *name) { return qd_mex_get_Mat<double>(a, false, name); };
        auto get_Col = [](const mxArray *a, const char *name) { return qd_mex_get_Col<double>(a, false, name); };
        const char *error = "Inputs 'fbs_pos', 'lbs_pos', 'path_gain', 'path_length' and 'polmat' must all be cell arrays.";
        const auto fbs_pos = qd_mex_read_cell<arma::mat>(prhs[6], "fbs_pos", error, get_Mat);
        const auto lbs_pos = qd_mex_read_cell<arma::mat>(prhs[7], "lbs_pos", error, get_Mat);
        const auto path_gain = qd_mex_read_cell<arma::vec>(prhs[8], "path_gain", error, get_Col);
        const auto path_length = qd_mex_read_cell<arma::vec>(prhs[9], "path_length", error, get_Col);
        const auto polmat = qd_mex_read_cell<arma::mat>(prhs[10], "polmat", error, get_Mat);
        const auto tx_pos = qd_mex_get_xyz_batch(prhs[2], "tx_pos");
        const auto tx_orientation = qd_mex_get_xyz_batch(prhs[3], "tx_orientation");
        const auto rx_pos = qd_mex_get_xyz_batch(prhs[4], "rx_pos");
        const auto rx_orientation = qd_mex_get_xyz_batch(prhs[5], "rx_orientation");

        // Outputs: cell arrays, the C++ function writes into the MATLAB memory of their entries
        const size_t n_snap = fbs_pos.size();
        std::vector<arma::cube> out[7];
        for (int k = 0; k < std::max(nlhs, 1); ++k)
        {
            plhs[k] = mxCreateCellMatrix((mwSize)n_snap, 1);
            out[k].resize(n_snap);
            for (size_t s = 0; s < n_snap; ++s)
            {
                const arma::uword n_path = add_fake_los_path ? fbs_pos[s].n_cols + 1 : fbs_pos[s].n_cols;
                mxSetCell(plhs[k], (mwIndex)s, qd_mex_init_output(&out[k][s], n_ports_rx, n_ports_tx, n_path));
            }
        }

        CALL_QD(quadriga_lib::get_channels_spherical<double>(ant_tx, ant_rx, tx_pos, tx_orientation, rx_pos, rx_orientation,
                                                             fbs_pos, lbs_pos, path_gain, path_length, polmat,
                                                             out[0], out[1], out[2], center_freq, use_absolute_delays, add_fake_los_path,
                                                             (nlhs > 3) ? &out[3] : nullptr, (nlhs > 4) ? &out[4] : nullptr,
                                                             (nlhs > 5) ? &out[5] : nullptr, (nlhs > 6) ? &out[6] : nullptr, nullptr, use_kernel));
        return;
    }

    const auto fbs_pos = qd_mex_get_Mat<double>(prhs[6], false, "fbs_pos");
    const auto lbs_pos = qd_mex_get_Mat<double>(prhs[7], false, "lbs_pos");
    const auto path_gain = qd_mex_get_Col<double>(prhs[8], false, "path_gain");
    const auto path_length = qd_mex_get_Col<double>(prhs[9], false, "path_length");
    const auto polmat = qd_mex_get_Mat<double>(prhs[10], false, "polmat");
    const auto tx_pos = qd_mex_typecast_Col<double>(prhs[2], "tx_pos", 3);
    const auto tx_orientation = qd_mex_typecast_Col<double>(prhs[3], "tx_orientation", 3);
    const auto rx_pos = qd_mex_typecast_Col<double>(prhs[4], "rx_pos", 3);
    const auto rx_orientation = qd_mex_typecast_Col<double>(prhs[5], "rx_orientation", 3);

    arma::uword n_path = add_fake_los_path ? fbs_pos.n_cols + 1 : fbs_pos.n_cols;

    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;
    arma::cube *p_aod = nullptr, *p_eod = nullptr, *p_aoa = nullptr, *p_eoa = nullptr;

    plhs[0] = qd_mex_init_output(&coeff_re, n_ports_rx, n_ports_tx, n_path);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&coeff_im, n_ports_rx, n_ports_tx, n_path);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&delay, n_ports_rx, n_ports_tx, n_path);
    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&aod, n_ports_rx, n_ports_tx, n_path), p_aod = &aod;
    if (nlhs > 4)
        plhs[4] = qd_mex_init_output(&eod, n_ports_rx, n_ports_tx, n_path), p_eod = &eod;
    if (nlhs > 5)
        plhs[5] = qd_mex_init_output(&aoa, n_ports_rx, n_ports_tx, n_path), p_aoa = &aoa;
    if (nlhs > 6)
        plhs[6] = qd_mex_init_output(&eoa, n_ports_rx, n_ports_tx, n_path), p_eoa = &eoa;

    // 0 paths: C++ returns the zero-power LOS path with add_fake_los_path, else empty outputs
    CALL_QD(quadriga_lib::get_channels_spherical<double>(ant_tx, ant_rx,
                                                         tx_pos, tx_orientation, rx_pos, rx_orientation,
                                                         fbs_pos, lbs_pos, path_gain, path_length, polmat,
                                                         coeff_re, coeff_im, delay,
                                                         center_freq, use_absolute_delays, add_fake_los_path,
                                                         p_aod, p_eod, p_aoa, p_eoa, use_kernel));
}
