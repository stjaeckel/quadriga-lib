// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# GET_CHANNELS_SPHERICAL
Calculate MIMO channel coefficients and delays for spherical wave propagation

- Computes complex channel coefficients and propagation delays for all TX/RX element pairs and paths,
  using spherical wave assumption with per-element phase and delay.
- Interpolates antenna patterns for both arrays, accounting for element positions and array orientation
  (bank/tilt/heading Euler angles).
- Polarization coupling is applied via the 8-row transfer matrix `polmat`: the 2x2 Jones matrix (rows = RX polarisation
  V/H, columns = TX polarisation V/H), stored column-major with Re/Im interleaved: rows 1-2 = (V rx, V tx),
  rows 3-4 = (H rx, V tx), rows 5-6 = (V rx, H tx), rows 7-8 = (H rx, H tx).
- NLOS paths: the path length between the TX and RX phase centres is `max(path_length, d_poly)`, where `d_poly` is the
  TX-FBS-LBS-RX polyline length; the length beyond the polyline (interactions between FBS and LBS, e.g. paths of
  [[qrt_file_read]] with more than two interactions) is added to the FBS-LBS segment. Delays and
  phases follow this length; the per-element offsets follow the FBS / LBS geometry.
- `center_freq = 0` gives the coefficients at f = 0: the delay phase exp(-j·2π·f·τ) is 1, the antenna and polarization phases remain; delays are computed as usual.
- If `use_absolute_delays == false`, the LOS delay between the TX and RX phase centres is subtracted from all paths.
  Delays are per element pair, so pairs that are closer to each other than the phase centres get negative delays,
  down to about -(half the TX array size + half the RX array size) / speed of light (both arrays endfire).
  [[quantize_delays]] maps delays above `-tap_spacing / 2` to tap 0 and rejects lower ones (at 5 ns taps: half sizes
  adding up to more than about 0.75 m); use `use_absolute_delays = true` in that case.
- LOS detection: the path whose length is closest to the TX-RX distance, within 0.1 mm.
- If `add_fake_los_path == true`, one output path is always added (`n_path + 1` outputs): slot 1 holds the LOS path,
  with zero power if no LOS path was found. If a LOS path was found, it is moved to slot 1 and its original slot
  (input index + 1) holds a zero-power path. All other paths keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `[n_rx, n_tx, 0]`; with `add_fake_los_path = true` they
  hold only this zero-power LOS path, with the LOS delays and angles per antenna pair.

## Batch mode:
- If `fbs_pos` is a cell array, the function processes `n_snap` snapshots in one call, the snapshots in parallel
  (OpenMP).
- The path inputs `fbs_pos`, `lbs_pos`, `path_gain`, `path_length` and `polmat` are then cell arrays of length `n_snap`, as
  [[qrt_file_read]] returns them for one frequency.
- The position and orientation inputs are `[3, n_snap]`, one column per snapshot, or a 3-element vector for all
  snapshots.
- All outputs are cell arrays of length `n_snap`, `[n_snap, 1]`, whose entries have the shapes given below.
- Snapshots without any path produce an empty entry, or only the zero-power LOS path with `add_fake_los_path` (as for
  a single snapshot).

## Usage:
```
[ coeff_re, coeff_im, delay, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_spherical( tx_array, ...
    rx_array, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, ...
    path_length, polmat, center_freq, use_absolute_delays, add_fake_los_path, use_kernel );
```

## Inputs:
- **`tx_array`** — Transmit antenna array; `n_tx` = number of ports after element coupling, see [[arrayant_generate]]
- **`rx_array`** — Receive antenna array; `n_rx` = number of ports after element coupling, see [[arrayant_generate]]
- **`tx_pos`** — Transmitter position, finite; `[3, 1]`, or `[3, n_snap]` in batch mode
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading), finite; `[3, 1]`, or
  `[3, n_snap]` in batch mode
- **`rx_pos`** — Receiver position, finite; `[3, 1]`, or `[3, n_snap]` in batch mode
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading), finite; `[3, 1]`, or
  `[3, n_snap]` in batch mode
- **`fbs_pos`** — First-bounce scatterer positions; `[3, n_path]`, or a cell of length `n_snap` in batch mode
- **`lbs_pos`** — Last-bounce scatterer positions; `[3, n_path]`, or a cell of length `n_snap` in batch mode
- **`path_gain`** — Path gains in linear scale; `[n_path, 1]`, or a cell of length `n_snap` in batch mode
- **`path_length`** — Total path lengths from TX to RX phase center; values shorter than the TX-FBS-LBS-RX polyline
  (e.g. 0) use the polyline length; `[n_path, 1]`, or a cell of length `n_snap` in batch mode
- **`polmat`** — Polarization transfer matrix, Jones row order as above; `[8, n_path]`, or a cell of length `n_snap` in
  batch mode
- **`center_freq`** — Center frequency in Hz; 0 (or empty) gives the coefficients at f = 0 (see above); default: 0
- **`use_absolute_delays`** — If `true`, delays include the LOS component; Default: `false`
- **`add_fake_los_path`** — If `true`, always adds one output path; slot 1 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; Default: `false`
- **`use_kernel`** — Kernel of the antenna interpolation and of the coefficient computation: 0 = auto, 1 = GENERIC,
  2 = AVX2 (single precision; error if unavailable); auto selects GENERIC in this version, see
  [[arrayant_interpolate]]; default: 0

## Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path(+1)]`
- **`aod`** — Azimuth of departure; `[n_rx, n_tx, n_path(+1)]`
- **`eod`** — Elevation of departure; `[n_rx, n_tx, n_path(+1)]`
- **`aoa`** — Azimuth of arrival; `[n_rx, n_tx, n_path(+1)]`
- **`eoa`** — Elevation of arrival; `[n_rx, n_tx, n_path(+1)]`

## See also:
- [[get_channels_planar]] (planar wave variant)
- [[get_channels_irs]] (for IRS-assisted communication)
- [[arrayant_generate]] (antenna array generator)
- [[baseband_freq_response]] (for calculating the frequency response)
- [[quantize_delays]] (for mapping delays to a fixed grid)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 11 || nrhs > 15)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Parse antenna objects
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

        CALL_QD(quadriga_lib::get_channels_spherical_vec<double>(ant_tx, ant_rx, tx_pos, tx_orientation, rx_pos, rx_orientation,
                                                                 fbs_pos, lbs_pos, path_gain, path_length, polmat,
                                                                 out[0], out[1], out[2], center_freq, use_absolute_delays, add_fake_los_path,
                                                                 (nlhs > 3) ? &out[3] : nullptr, (nlhs > 4) ? &out[4] : nullptr,
                                                                 (nlhs > 5) ? &out[5] : nullptr, (nlhs > 6) ? &out[6] : nullptr, nullptr, use_kernel));
        return;
    }

    // Single snapshot
    const auto fbs_pos = qd_mex_get_Mat<double>(prhs[6], false, "fbs_pos");
    const auto lbs_pos = qd_mex_get_Mat<double>(prhs[7], false, "lbs_pos");
    const auto path_gain = qd_mex_get_Col<double>(prhs[8], false, "path_gain");
    const auto path_length = qd_mex_get_Col<double>(prhs[9], false, "path_length");
    const auto polmat = qd_mex_get_Mat<double>(prhs[10], false, "polmat");
    const auto tx_pos = qd_mex_typecast_Col<double>(prhs[2], "tx_pos", 3);
    const auto tx_orientation = qd_mex_typecast_Col<double>(prhs[3], "tx_orientation", 3);
    const auto rx_pos = qd_mex_typecast_Col<double>(prhs[4], "rx_pos", 3);
    const auto rx_orientation = qd_mex_typecast_Col<double>(prhs[5], "rx_orientation", 3);

    // Derived inputs
    arma::uword n_path = add_fake_los_path ? fbs_pos.n_cols + 1 : fbs_pos.n_cols;

    // Initialize output memory
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

    // Call the C++ function
    // 0 paths: C++ returns the zero-power LOS path with add_fake_los_path, else empty outputs
    CALL_QD(quadriga_lib::get_channels_spherical<double>(ant_tx, ant_rx,
                                                         tx_pos, tx_orientation, rx_pos, rx_orientation,
                                                         fbs_pos, lbs_pos, path_gain, path_length, polmat,
                                                         coeff_re, coeff_im, delay,
                                                         center_freq, use_absolute_delays, add_fake_los_path,
                                                         p_aod, p_eod, p_aoa, p_eoa, use_kernel));
}
