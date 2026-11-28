// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Channel generators
SECTION!*/

/*!MD
# GET_CHANNELS_PLANAR
Calculate MIMO channel coefficients and delays for planar wave propagation

- Coefficients and delays of one link for each pair of a Tx and an Rx port and each path; a path is given by its
  departure and arrival angles, which all elements of an array share (planar wavefronts)
- Path $l$ of the element pair $(t, r)$ has the coefficient and delay

  $$a_l = \sqrt{g_l} \; \mathbf{F}_r^T \mathbf{M}_l \, \mathbf{F}_t \; e^{-j 2\pi f_c d / c}, \quad \tau_l = d / c, \quad d = \ell_l + d_t + d_r$$

  with the path gain $g_l$, the Jones matrix $\mathbf{M}_l$ (`polmat`), the patterns $\mathbf{F} = (F_\theta, F_\phi)^T$
  at the path angles, the path length $\ell_l$ between the phase centres (used as given), the distances $d_t$, $d_r$ of
  the elements from the wavefront planes through the phase centres (`dist` of [[interpolate_pattern]]), the center
  frequency $f_c$ and the speed of light $c$; the port weights then combine the elements into ports
- The positions enter only through the Tx-Rx distance (relative delays, LOS path)
- Vector form: with cell arrays as path inputs, one cell per snapshot, the outputs are cell arrays; the snapshots are
  computed in parallel

## Usage:
```
[ coeff, delay, rx_Doppler ] = quadriga_lib.get_channels_planar( tx_array, rx_array, ...
    tx_pos, tx_orientation, rx_pos, rx_orientation, aod, eod, aoa, eoa, path_gain, path_length, polmat, ...
    center_freq, use_absolute_delays, add_fake_los_path, use_kernel );
```

## Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array, struct, fields in [[generate_antenna]]; `n_tx` and `n_rx` ports
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m, finite; 3 elements; vector form: 3 elements (all snapshots) or
  `[3, n_snap]`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientation, Euler angles (bank, tilt, heading) in rad,
  finite; sizes as `tx_pos`
- **`aod`**, **`eod`** — Azimuth and elevation of departure in rad, GCS; `[n_path, 1]`; vector form: cell arrays with
  `n_snap` entries
- **`aoa`**, **`eoa`** — Azimuth and elevation of arrival in rad, GCS (the direction from the Rx toward the incoming
  path); `[n_path, 1]`; vector form: cell arrays
- **`path_gain`** — Path gains, linear power; `[n_path, 1]`; vector form: cell array
- **`path_length`** — Path lengths in m between the Tx and Rx phase centres; `[n_path, 1]`; vector form: cell array
- **`polmat`** — Jones matrices, real and imaginary parts interleaved (section description); `[8, n_path]`; vector
  form: cell array
- **`center_freq`** — Center frequency in Hz, finite and >= 0; 0 gives the coefficients at f = 0 (Notes); default: 0
- **`use_absolute_delays`** — Keep the delay of the Tx-Rx distance in all delays; `false`: delays relative to it
  (Notes); default: `false`
- **`add_fake_los_path`** — Put the LOS path into an added slot 1, with zero power if there is none (Notes); default:
  `false`
- **`use_kernel`** — Kernel of the antenna interpolation: 0 = auto (GENERIC in this version), 1 = GENERIC, 2 = AVX2
  (single-precision arithmetic; error if unavailable); default: 0

## Outputs:
- **`coeff`** — Complex coefficients; `[n_rx, n_tx, n_out]` with `n_out = n_path`, or `n_path + 1` with
  `add_fake_los_path`; vector form: cell arrays `[n_snap, 1]` of such arrays
- **`delay`** — Delays in s; `[n_rx, n_tx, n_out]`
- **`rx_Doppler`** — Doppler weight of each path for an Rx moving along its local x-axis: the cosine of the angle
  between this axis and the arrival direction; the Doppler shift at the speed $v$ is $v f_c / c$ times this weight,
  positive when the Rx moves toward the incoming path; `[1, n_out]`

## Notes:
- **LOS path** (only with `add_fake_los_path`): the strongest path whose `path_length` is within 0.1 mm of the Tx-Rx
  distance; the angles are not checked
- **`add_fake_los_path`**: slot 1 holds the LOS path, with zero power (and `rx_Doppler` 0) if there is none; a LOS path
  among the inputs moves to slot 1 and leaves a zero-power path in its own slot (input index + 1); the other paths
  keep their order
- **0 paths** (all path inputs empty): outputs `[n_rx, n_tx, 0]`, or only the zero-power LOS path with
  `add_fake_los_path`
- **Relative delays**: the delay of the Tx-Rx distance is subtracted from all delays; element pairs can then get
  negative delays, as in [[get_channels_spherical]]
- **`center_freq = 0`**: the delay phase is 1, the antenna and polarization phases remain; the delays are unchanged
- **Port weights**: the delays of a port pair are means over its element pairs, weighted by the port-weight powers
- **Vector form**: as in [[get_channels_spherical]]

## See also:
- [[get_channels_spherical]] (paths given by scatterers, spherical wavefronts)
- [[get_channels_ieee_indoor]] (IEEE indoor channels, computed with this function)
- [[freq_response]] (frequency response of the coefficients)
- [[quantize_delays]] (delays on a tap grid)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 13 || nrhs > 17)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    auto ant_tx = qd_mex_struct2arrayant(prhs[0], false, false, "tx_array");
    auto ant_rx = qd_mex_struct2arrayant(prhs[1], false, false, "rx_array");

    double center_freq = (nrhs < 14) ? 0.0 : qd_mex_get_scalar<double>(prhs[13], "center_freq", 0.0);
    bool use_absolute_delays = (nrhs < 15) ? false : qd_mex_get_scalar<bool>(prhs[14], "use_absolute_delays", false);
    bool add_fake_los_path = (nrhs < 16) ? false : qd_mex_get_scalar<bool>(prhs[15], "add_fake_los_path", false);
    const int use_kernel = (nrhs < 17) ? 0 : qd_mex_get_scalar<int>(prhs[16], "use_kernel", 0);

    arma::uword n_ports_tx = ant_tx.n_ports();
    arma::uword n_ports_rx = ant_rx.n_ports();

    // Batch mode: path data as cell arrays, one entry per snapshot
    if (mxIsCell(prhs[6]))
    {
        // Numeric data of class double is not copied
        auto get_Mat = [](const mxArray *a, const char *name) { return qd_mex_get_Mat<double>(a, false, name); };
        auto get_Col = [](const mxArray *a, const char *name) { return qd_mex_get_Col<double>(a, false, name); };
        const char *error = "Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length' and 'polmat' must all be cell arrays.";
        const auto aod = qd_mex_read_cell<arma::vec>(prhs[6], "aod", error, get_Col);
        const auto eod = qd_mex_read_cell<arma::vec>(prhs[7], "eod", error, get_Col);
        const auto aoa = qd_mex_read_cell<arma::vec>(prhs[8], "aoa", error, get_Col);
        const auto eoa = qd_mex_read_cell<arma::vec>(prhs[9], "eoa", error, get_Col);
        const auto path_gain = qd_mex_read_cell<arma::vec>(prhs[10], "path_gain", error, get_Col);
        const auto path_length = qd_mex_read_cell<arma::vec>(prhs[11], "path_length", error, get_Col);
        const auto polmat = qd_mex_read_cell<arma::mat>(prhs[12], "polmat", error, get_Mat);
        const auto tx_pos = qd_mex_get_xyz_batch(prhs[2], "tx_pos");
        const auto tx_orientation = qd_mex_get_xyz_batch(prhs[3], "tx_orientation");
        const auto rx_pos = qd_mex_get_xyz_batch(prhs[4], "rx_pos");
        const auto rx_orientation = qd_mex_get_xyz_batch(prhs[5], "rx_orientation");

        // Outputs: cell arrays, the C++ function writes into the MATLAB memory of their entries (coeff: complex)
        const size_t n_snap = aod.size();
        std::vector<arma::cx_cube> coeff(n_snap);
        std::vector<arma::cube> coeff_re, coeff_im, delay(n_snap);
        std::vector<arma::vec> rx_Doppler(n_snap);
        for (int k = 0; k < std::max(nlhs, 1); ++k)
        {
            plhs[k] = mxCreateCellMatrix((mwSize)n_snap, 1);
            for (size_t s = 0; s < n_snap; ++s)
            {
                const arma::uword n_path = add_fake_los_path ? aod[s].n_elem + 1 : aod[s].n_elem;
                mxSetCell(plhs[k], (mwIndex)s, (k == 0)   ? qd_mex_init_output(&coeff[s], n_ports_rx, n_ports_tx, n_path)
                                               : (k == 1) ? qd_mex_init_output(&delay[s], n_ports_rx, n_ports_tx, n_path)
                                                          : qd_mex_init_output(&rx_Doppler[s], n_path, true));
            }
        }

        CALL_QD(quadriga_lib::get_channels_planar<double>(ant_tx, ant_rx, tx_pos, tx_orientation, rx_pos, rx_orientation,
                                                          aod, eod, aoa, eoa, path_gain, path_length, polmat,
                                                          coeff_re, coeff_im, delay, center_freq, use_absolute_delays, add_fake_los_path,
                                                          (nlhs > 2) ? &rx_Doppler : nullptr, &coeff, use_kernel));
        return;
    }

    const auto aod = qd_mex_get_Col<double>(prhs[6], false, "aod");
    const auto eod = qd_mex_get_Col<double>(prhs[7], false, "eod");
    const auto aoa = qd_mex_get_Col<double>(prhs[8], false, "aoa");
    const auto eoa = qd_mex_get_Col<double>(prhs[9], false, "eoa");
    const auto path_gain = qd_mex_get_Col<double>(prhs[10], false, "path_gain");
    const auto path_length = qd_mex_get_Col<double>(prhs[11], false, "path_length");
    const auto polmat = qd_mex_get_Mat<double>(prhs[12], false, "polmat");
    const auto tx_pos = qd_mex_typecast_Col<double>(prhs[2], "tx_pos", 3);
    const auto tx_orientation = qd_mex_typecast_Col<double>(prhs[3], "tx_orientation", 3);
    const auto rx_pos = qd_mex_typecast_Col<double>(prhs[4], "rx_pos", 3);
    const auto rx_orientation = qd_mex_typecast_Col<double>(prhs[5], "rx_orientation", 3);

    arma::uword n_path = add_fake_los_path ? aod.n_elem + 1 : aod.n_elem;

    arma::cube coeff_re, coeff_im, delay;
    arma::vec rx_Doppler;

    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&delay, n_ports_rx, n_ports_tx, n_path);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&rx_Doppler, n_path, true);

    // 0 paths: C++ returns the zero-power LOS path with add_fake_los_path, else empty outputs
    CALL_QD(quadriga_lib::get_channels_planar<double>(ant_tx, ant_rx,
                                                      tx_pos, tx_orientation, rx_pos, rx_orientation,
                                                      aod, eod, aoa, eoa, path_gain, path_length, polmat,
                                                      coeff_re, coeff_im, delay,
                                                      center_freq, use_absolute_delays, add_fake_los_path,
                                                      (nlhs > 2) ? &rx_Doppler : nullptr, use_kernel));

    // The single-snapshot form has split outputs only: copy them into one complex array
    plhs[0] = qd_mex_copy2matlab(&coeff_re, &coeff_im, true);
}
