// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Channel statistics
SECTION!*/

/*!MD
# acdf
Calculate the empirical averaged cumulative distribution function (CDF)

- Computes per-column empirical CDFs by histogramming into bins and taking the normalized cumulative sum
- Averaged CDF is obtained by quantile-space averaging: for a fine probability grid, x-values from each
  column CDF are averaged, then mapped back to the bin grid
- Quantile statistics (mean and std) are reported at the 0.1, 0.2, ..., 0.9 probability levels
- `Inf` and `NaN` values are excluded from computation
- A data set without any finite value gets a NaN column in `cdf_per_set` and is left out of `cdf_avg`, `mu`
  and `sig`; if no set has finite values (possible only with given `bins`), `cdf_avg`, `mu` and `sig` are NaN
- If `bins` is empty, equally spaced bins spanning the data range are generated

## Usage:
```
cdf_per_set, bins_out, cdf_avg, mu, sig = quadriga_lib.tools.acdf( data, bins, *, n_bins )
```

## Inputs:
- **`data`** — Input data matrix; each column is one independent data set; magnitudes below about 1e18, 
  larger values overflow in `sig`; `(n_samples, n_sets)`
- **`bins`** — Bin centers; used as-is if non-empty (must be finite and strictly ascending, else a
  `ValueError` is raised); if empty, equally spaced bins spanning the data range are generated;
  `(n_bins_in,)`; default: `None`
- **`n_bins`** — Number of bins when auto-generating; must be >= 2; ignored when non-empty `bins`
  are provided; default: 201

## Outputs:
- **`cdf_per_set`** — Individual CDFs; one per column of data (NaN for a set without finite values); `(n_bins_out, n_sets)`
- **`bins_out`** — Auto-generated bins; copy of `bins` when non-empty `bins` are provided;
  `(n_bins_out,)` with `n_bins_out = n_bins` or `n_bins_out = n_bins_in`
- **`cdf_avg`** — Averaged CDF via quantile-space averaging across the data sets with finite values; `(n_bins_out,)`
- **`mu`** — Mean of the 0.1–0.9 quantiles across the data sets with finite values; `(9,)`
- **`sig`** — Standard deviation of the 0.1–0.9 quantiles across the data sets with finite values; `(9,)`
MD!*/

py::tuple acdf(const py::array_t<double> &data,
               py::handle bins,
               arma::uword n_bins)
{
    const auto data_a = qd_python_numpy2arma_Mat<double>(data, true);
    const auto bins_in_a = qd_python_numpy2arma_Col<double>(bins, true);

    arma::uword n_bins_out = bins_in_a.empty() ? n_bins : bins_in_a.n_elem;
    arma::uword n_sets = data_a.n_cols;

    arma::mat cdf_per_set;
    arma::vec cdf_avg, bins_out, mu, sig;

    auto cdf_per_set_py = qd_python_init_output(n_bins_out, n_sets, &cdf_per_set);
    auto cdf_avg_py = qd_python_init_output(n_bins_out, &cdf_avg);
    auto mu_py = qd_python_init_output(9, &mu);
    auto sig_py = qd_python_init_output(9, &sig);

    if (!bins_in_a.empty())
        bins_out = bins_in_a;

    quadriga_lib::acdf<double>(data_a, &bins_out, &cdf_per_set, &cdf_avg, &mu, &sig, n_bins);

    auto bins_out_py = qd_python_copy2numpy(&bins_out);

    return py::make_tuple(cdf_per_set_py, bins_out_py, cdf_avg_py, mu_py, sig_py);
}

/*!MD
# calc_angular_spread
Calculate azimuth and elevation angular spreads, optionally with spherical wrapping

- Computes RMS azimuth and elevation angular spreads from power-weighted angles
- Inputs are lists of 1D arrays so each CIR can have a different number of paths
- RMS spread uses the 3GPP TR 38.901 second-moment definition of wrapped deviations from the
  circular mean
- When `wrapping = True`, the power-weighted mean direction is computed in Cartesian coordinates
  and all paths are rotated so the centroid lies on the equator before computing spreads,
  avoiding pole singularity artifacts
- When `wrapping = False`, spreads are computed directly from raw angles; `orientation` is zero
  and `phi`/`theta` equal the input `az`/`el`
- When `calc_bank_angle = True`, an optimal bank angle maximizing azimuth spread is derived
  analytically from eigenvectors of the 2x2 power-weighted covariance matrix; only used when
  `wrapping = True`
- When `quantize_deg > 0`, paths within that angular distance are grouped and their powers summed

## Usage:
```
azimuth_spread, elevation_spread, orientation, phi, theta = \
    quadriga_lib.tools.calc_angular_spread( az, el, powers, *, wrapping, calc_bank_angle, quantize_deg )
```

## Inputs:
- **`az`** — Azimuth angles; range -pi to pi; list of length `n_cir`, each element a 1D array of length `n_path`
- **`el`** — Elevation angles; range -pi/2 to pi/2; list of length `n_cir`, each element a 1D array of length `n_path`
- **`powers`** — Path powers; list of length `n_cir`, each element a 1D array of length `n_path`
- **`wrapping`** — If True, enables spherical rotation; default: False
- **`calc_bank_angle`** — If True, computes the optimal bank angle analytically; only used when `wrapping = True`; default: False
- **`quantize_deg`** — Angular quantization step in [deg]; paths within this distance are grouped; default: 0.0 (no quantization);
  otherwise it must be finite and >= 1e-3 deg, else a `ValueError` is raised

## Outputs:
- **`azimuth_spread`** — RMS azimuth angular spread; `(n_cir,)`
- **`elevation_spread`** — RMS elevation angular spread; `(n_cir,)`
- **`orientation`** — Euler angles [bank; tilt; heading] of an antenna pointing at the power-weighted mean
  direction (a positive tilt points down); `(3, n_cir)`
- **`phi`** — Rotated azimuth angles, the path azimuths in the local frame of that antenna; list of length `n_cir`
- **`theta`** — Rotated elevation angles; list of length `n_cir`
MD!*/

py::tuple calc_angular_spread(py::list az,
                              py::list el,
                              py::list powers,
                              py::handle wrapping,
                              py::handle calc_bank_angle,
                              double quantize_deg)
{
    const bool wrapping_b = qd_python_flag(wrapping, false, "wrapping");
    const bool calc_bank_angle_b = qd_python_flag(calc_bank_angle, false, "calc_bank_angle");

    std::vector<arma::vec> az_a = qd_python_list2vector_Col<double>(az);
    std::vector<arma::vec> el_a = qd_python_list2vector_Col<double>(el);
    std::vector<arma::vec> powers_a = qd_python_list2vector_Col<double>(powers);

    arma::vec azimuth_spread, elevation_spread;
    arma::mat orientation;
    std::vector<arma::vec> phi, theta;

    quadriga_lib::calc_angular_spread<double>(az_a, el_a, powers_a,
                                              &azimuth_spread, &elevation_spread, &orientation,
                                              &phi, &theta, wrapping_b, calc_bank_angle_b, quantize_deg);

    auto azimuth_spread_py = qd_python_copy2numpy(&azimuth_spread);
    auto elevation_spread_py = qd_python_copy2numpy(&elevation_spread);
    auto orientation_py = qd_python_copy2numpy(&orientation);
    auto phi_py = qd_python_copy2list(&phi);
    auto theta_py = qd_python_copy2list(&theta);

    return py::make_tuple(azimuth_spread_py, elevation_spread_py, orientation_py, phi_py, theta_py);
}

/*!MD
# calc_cross_polarization_ratio
Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases

- Computes aggregate XPR from polarization transfer matrices using the total-power-ratio method:
  co-pol and cross-pol powers are summed across all qualifying paths per CIR, and XPR is their ratio
- Inputs are lists so each CIR can have a different number of paths
- XPR is computed in both the linear V/H basis and the circular LHCP/RHCP basis
- `polmat` is the 2x2 Jones matrix `M` (rows = RX polarisation V/H, columns = TX polarisation V/H), stored column-major with
  interleaved real/imaginary parts, 8 rows per path: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx),
  rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx). Below, `M_<rx><tx>` names an entry by its receive and transmit
  polarisation: `M_vv` = rows 0-1, `M_hv` = rows 2-3, `M_vh` = rows 4-5, `M_hh` = rows 6-7
- V-XPR and H-XPR are ratios per receive polarisation: V-XPR is the power received in V from V transmission over
  the power received in V from H transmission; H-XPR likewise for the H receive polarisation
- The circular entries couple circular patterns through `polmat` as `channel.spherical` couples antenna patterns,
  `c(r, t) = r^T M t` (plain transpose), with the same circular vectors at both ends, each in that antenna's own
  (e_theta, e_phi) frame: L = (V, H) = (1, +j) / sqrt(2) (LHCP), R = (1, -j) / sqrt(2) (RHCP); names after the IEEE
  definition for the time convention exp(+j 2 pi f t) of the library's coefficients exp(-j 2 pi f tau).
  `M_<rx><tx>`: `M_LL = (M_vv - M_hh + j(M_vh + M_hv)) / 2`, `M_RR = (M_vv - M_hh - j(M_vh + M_hv)) / 2`,
  `M_LR = (M_vv + M_hh + j(M_hv - M_vh)) / 2` (receive L, transmit R), `M_RL = (M_vv + M_hh + j(M_vh - M_hv)) / 2`.
  Co-polar means the same circular vector at both ends: the LOS matrix `diag(1, -1)` is co-polar (handedness
  kept), `diag(1, 1)` (one mirrored component, as by a single reflection) is cross-polar
- LHCP-XPR and RHCP-XPR are ratios per receive polarisation, like V-XPR and H-XPR
- LOS paths are identified by comparing path length against the direct TX-RX distance `dTR`; paths with
  `path_length <= dTR + window_size` are excluded by default
- Normalization of `polmat` does not affect XPR (cancels in the ratio) but does affect `pg`; magnitudes in `polmat` above about 1e18 overflow and give NaN
- If cross-pol power is zero and co-pol is positive, XPR is set to infinity; if both are zero, XPR
  is set to 0
- TX/RX positions may be fixed `(3, 1)` or mobile `(3, n_cir)`

## Usage:
```
xpr, pg = quadriga_lib.tools.calc_cross_polarization_ratio( powers, polmat, path_length, tx_pos, rx_pos, \
    *, include_los, window_size )
```

## Inputs:
- **`powers`** — Path powers on linear scale per CIR; list of length `n_cir`, each element a 1D
  array of length `n_path`
- **`polmat`** — Polarization transfer matrices with interleaved real/imag parts per CIR; list of length
  `n_cir`, each element a 2D array of shape `(8, n_path)`
- **`path_length`** — Absolute TX-to-RX path lengths per CIR; list of length `n_cir`, each element
  a 1D array of length `n_path`
- **`tx_pos`** — Transmitter position [x; y; z]; `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`rx_pos`** — Receiver position [x; y; z]; `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`include_los`** — If True, includes LOS and near-LOS paths in the XPR calculation; default: False
- **`window_size`** — LOS exclusion window; paths with `path_length <= dTR + window_size` are excluded when
  `include_los = False`; default: 0.01

## Outputs:
- **`xpr`** — XPR on linear scale; `(n_cir, 6)`; columns (0-based):<br><br>
   | Col | Description                                                     |
   | :-: | --------------------------------------------------------------- |
   | 0   | Aggregate linear XPR (total V+H co-pol / total V+H cross-pol)   |
   | 1   | V-XPR: sum(abs(M_vv)^2) / sum(abs(M_vh)^2) (receive V)          |
   | 2   | H-XPR: sum(abs(M_hh)^2) / sum(abs(M_hv)^2) (receive H)          |
   | 3   | Aggregate circular XPR (total L+R co-pol / total L+R cross-pol) |
   | 4   | LHCP XPR: sum(abs(M_LL)^2) / sum(abs(M_LR)^2) (receive L)       |
   | 5   | RHCP XPR: sum(abs(M_RR)^2) / sum(abs(M_RL)^2) (receive R)       |
- **`pg`** — Total path gain summed over all paths (including LOS) as
  `0.5 · sum(powers · (abs(M_vv)^2 + abs(M_hv)^2 + abs(M_vh)^2 + abs(M_hh)^2))`; `(n_cir,)`
MD!*/

py::tuple calc_cross_polarization_ratio(py::list powers,
                                        py::list polmat,
                                        py::list path_length,
                                        const py::array_t<double> &tx_pos,
                                        const py::array_t<double> &rx_pos,
                                        py::handle include_los,
                                        double window_size)
{
    const bool include_los_b = qd_python_flag(include_los, false, "include_los");

    std::vector<arma::vec> powers_a = qd_python_list2vector_Col<double>(powers);
    std::vector<arma::mat> polmat_a = qd_python_list2vector_Mat<double>(polmat);
    std::vector<arma::vec> path_length_a = qd_python_list2vector_Col<double>(path_length);
    arma::mat tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    arma::mat rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);

    arma::mat xpr;
    arma::vec pg;

    quadriga_lib::calc_cross_polarization_ratio<double>(powers_a, polmat_a, path_length_a, tx_pos_a, rx_pos_a,
                                                        &xpr, &pg, include_los_b, window_size);

    auto xpr_py = qd_python_copy2numpy(&xpr);
    auto pg_py = qd_python_copy2numpy(&pg);

    return py::make_tuple(xpr_py, pg_py);
}

/*!MD
# calc_delay_spread
RMS delay spread from per-CIR delays and linear-scale powers

- Computes RMS delay spread from a set of delays and corresponding linear-scale powers per CIR
- Inputs are lists of 1D arrays so each CIR can have a different number of paths
- Paths with power below `p_max / 10^(0.1 · threshold_dB)` are excluded; the default of
  100 dB effectively includes all paths
- When `granularity > 0`, paths falling into the same delay bin of width `granularity` have their
  powers summed before the spread is computed

## Usage:
```
ds, mean_delay = quadriga_lib.tools.calc_delay_spread( delays, powers, *, threshold_dB, granularity )
```

## Inputs:
- **`delays`** — Delays per CIR; list of length `n_cir`, each element a 1D array of length `n_path`
- **`powers`** — Path powers on linear scale per CIR; list of length `n_cir`, each element a 1D
  array of length `n_path`
- **`threshold_dB`** — Power threshold in [dB] relative to the strongest path; paths below it
  are excluded; default: 100.0
- **`granularity`** — Bin width in [s] for grouping paths in the delay domain; default: 0.0 (no grouping); must be
  finite and non-negative; when > 0, all delays must be finite and the delay span / granularity must be below 2^53,
  else a `ValueError` is raised

## Outputs:
- **`ds`** — RMS delay spread per CIR; `(n_cir,)`
- **`mean_delay`** — Mean delay per CIR; `(n_cir,)`
MD!*/

py::tuple calc_delay_spread(py::list delays,
                            py::list powers,
                            double threshold_dB,
                            double granularity)
{
    std::vector<arma::vec> delays_a = qd_python_list2vector_Col<double>(delays);
    std::vector<arma::vec> powers_a = qd_python_list2vector_Col<double>(powers);

    arma::vec mean_delay;
    const arma::vec ds = quadriga_lib::calc_delay_spread<double>(delays_a, powers_a, threshold_dB, granularity, &mean_delay);

    auto ds_py = qd_python_copy2numpy(&ds);
    auto mean_delay_py = qd_python_copy2numpy(&mean_delay);

    return py::make_tuple(ds_py, mean_delay_py);
}

/*!MD
# calc_rician_k_factor
Calculate the Rician K-Factor from channel impulse response data

- Computes the Rician K-Factor (KF): ratio of LOS power to NLOS power per CIR
- Inputs are lists of 1D arrays so each CIR can have a different number of paths
- LOS paths are those with length ≤ `dTR + window_size`, where `dTR` is the direct TX-RX distance;
  remaining paths are NLOS
- If total NLOS power is zero, KF is set to infinity; if total LOS power is zero, KF is set to 0
- TX/RX positions may be fixed `(3,)` / `(3, 1)` (reused for all CIRs) or mobile `(3, n_cir)`

## Usage:
```
kf, pg = quadriga_lib.tools.calc_rician_k_factor( powers, path_length, tx_pos, rx_pos, *, window_size )
```

## Inputs:
- **`powers`** — Path powers on linear scale per CIR; list of length `n_cir`, each element a 1D
  array of length `n_path`
- **`path_length`** — Absolute TX-to-RX path lengths per CIR; list of length `n_cir`, each element
  a 1D array of length `n_path` matching `powers`
- **`tx_pos`** — Transmitter position [x; y; z]; `(3,)` or `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`rx_pos`** — Receiver position [x; y; z]; `(3,)` or `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`window_size`** — LOS window; paths with length ≤ `dTR + window_size` are treated as LOS;
  default: 0.01

## Outputs:
- **`kf`** — Rician K-Factor on linear scale; `(n_cir,)`
- **`pg`** — Total path gain (sum of all path powers) per CIR; `(n_cir,)`
MD!*/

py::tuple calc_rician_k_factor(py::list powers,
                               py::list path_length,
                               const py::array_t<double> &tx_pos,
                               const py::array_t<double> &rx_pos,
                               double window_size)
{
    std::vector<arma::vec> powers_a = qd_python_list2vector_Col<double>(powers);
    std::vector<arma::vec> path_length_a = qd_python_list2vector_Col<double>(path_length);
    arma::mat tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    arma::mat rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);

    arma::vec kf, pg;

    // All outputs are computed, also when the caller ignores some
    quadriga_lib::calc_rician_k_factor<double>(powers_a, path_length_a, tx_pos_a, rx_pos_a, &kf, &pg, window_size);

    auto kf_py = qd_python_copy2numpy(&kf);
    auto pg_py = qd_python_copy2numpy(&pg);

    return py::make_tuple(kf_py, pg_py);
}
