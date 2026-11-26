// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# calc_angular_spread
Calculate the RMS azimuth and elevation angular spreads of each CIR, optionally around the mean direction of its paths

- For one angle, with the path powers $p_l$ normalized to sum 1 over the paths of a CIR (the second-moment definition
  of 3GPP TR 38.901):

  $$\sigma = \sqrt{\sum_l p_l \, \Delta_l^2}, \quad \Delta_l = \arg e^{j (\phi_l - \bar\phi)}$$

  with the circular mean $\bar\phi = \arg \sum_l p_l \, e^{j \phi_l}$
- `wrapping = False`: the spreads of the given azimuth and elevation angles; near the poles, the azimuth spread
  overstates the angular spread
- `wrapping = True`: the paths are first rotated so that their power-weighted mean direction (the mean of the unit
  vectors) lies at azimuth 0 on the equator; `calc_bank_angle` then also rotates them about that direction so that
  the azimuth spread is largest (principal axis of the power-weighted covariance of the rotated angles); `orientation`
  is the direction of an antenna that sees the paths at the rotated angles `phi` and `theta`
- `quantize_deg` > 0: the deviations $\Delta_l$ are rounded to multiples of `quantize_deg` and the powers of equal
  deviations summed, for azimuth and elevation separately

## Usage:
```
azimuth_spread, elevation_spread, orientation, phi, theta = quadriga_lib.channel.calc_angular_spread( \
    az, el, powers, *, wrapping, calc_bank_angle, quantize_deg )
```

## Inputs:
- **`az`** — Path azimuths in rad; list of `n_cir` arrays `(n_path,)`, at least one CIR and one path per CIR; `n_path`
  may differ between the CIRs
- **`el`** — Path elevations in rad; list of `n_cir` arrays `(n_path,)`
- **`powers`** — Path powers, linear, with a positive sum per CIR; list of `n_cir` arrays `(n_path,)`
- **`wrapping`** — Rotate the paths to the equator before computing the spreads; default: `False`
- **`calc_bank_angle`** — Rotate them also about the mean direction, for the largest azimuth
  spread; only with `wrapping`; default: `False`
- **`quantize_deg`** — Quantization step of the angle deviations in deg; 0 = none, else finite and >= 1e-3; default: 0.0

## Outputs:
- **`azimuth_spread`** — RMS azimuth spread in rad; `(n_cir,)`
- **`elevation_spread`** — RMS elevation spread in rad; `(n_cir,)`
- **`orientation`** — Euler angles (bank, tilt, heading) in rad of an antenna pointing at the mean
  direction (a positive tilt points down), zero without `wrapping`; `(3, n_cir)`
- **`phi`** — Path azimuths in rad in the frame of that antenna (as [[interpolate_pattern]]
  returns them), the input `az` without `wrapping`; list of `n_cir` arrays `(n_path,)`
- **`theta`** — Path elevations in rad in that frame, the input `el` without `wrapping`;
  list of `n_cir` arrays `(n_path,)`

## See also:
- [[calc_delay_spread]] (delay spread of the same paths)
- [[interpolate_pattern]] (antenna response in the rotated frame)
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
# calc_xpr
Calculate the cross-polarization ratio (XPR) of each CIR in the linear and the circular polarization basis

- Total-power ratio: the co- and cross-polar powers of the paths of a CIR, weighted by the path powers $p_l$, are
  summed, and the XPR is their ratio; LOS paths are left out unless `include_los` is set
- Entries of the polarization transfer matrix $M$ of a path, first index receive, second transmit polarization:
  rows 0-1 of `polmat` = $M_{vv}$, 2-3 = $M_{hv}$ (H rx, V tx), 4-5 = $M_{vh}$ (V rx, H tx), 6-7 = $M_{hh}$, each as
  real and imaginary part
- Circular basis: $M_{rt} = \mathbf{r}^T M \, \mathbf{t}$ (plain transpose), as [[channel.spherical]]
  couples antenna patterns, with the same vectors at both ends, each in the $(e_\theta, e_\phi)$ frame of its antenna:
  $\mathbf{L} = (1, +j) / \sqrt{2}$ (LHCP) and $\mathbf{R} = (1, -j) / \sqrt{2}$ (RHCP) in (V, H), which gives

  $$M_{LL}, M_{RR} = \frac{1}{2} \left( M_{vv} - M_{hh} \pm j (M_{vh} + M_{hv}) \right)$$

  $$M_{LR}, M_{RL} = \frac{1}{2} \left( M_{vv} + M_{hh} \pm j (M_{hv} - M_{vh}) \right)$$

  (upper sign for the first entry)

- The names L and R follow the IEEE definition (rotation seen along the propagation direction) for the time
  convention $e^{+j 2\pi f t}$, which the coefficients $e^{-j 2\pi f \tau}$ of the library imply, with
  $e_\theta \times e_\phi$ the outward direction of the pattern
- Co-polar means the same circular vector at both ends: the LOS matrix $\mathrm{diag}(1, -1)$ of
  [[channel.spherical]] is co-polar (handedness kept), $\mathrm{diag}(1, 1)$ (one mirrored component, as
  by a single reflection) is cross-polar

## Usage:
```
xpr, pg = quadriga_lib.channel.calc_xpr( powers, polmat, path_length, tx_pos, rx_pos, *, include_los, \
    window_size )
```

## Inputs:
- **`powers`** — Path powers $p_l$, linear; list of `n_cir` arrays `(n_path,)`; `n_path` may differ between the CIRs
- **`polmat`** — Polarization transfer matrices, magnitudes below about 1e18; list of `n_cir` arrays `(8, n_path)`
- **`path_length`** — Path lengths in m; list of `n_cir` arrays `(n_path,)`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `(3,)` (fixed) or `(3, n_cir)`
- **`include_los`** — Include the LOS paths in the XPR; default: `False`
- **`window_size`** — LOS window in m, >= 0: paths with `path_length` <= $d_\mathrm{TR}$ +
  `window_size` are LOS, $d_\mathrm{TR}$ the Tx-Rx distance; default: 0.01

## Outputs:
- **`xpr`** — XPR, linear; `(n_cir, 6)`; columns:

  | Column | XPR |
  | ------ | --- |
  | 0 | Linear: $(P_{vv} + P_{hh}) / (P_{vh} + P_{hv})$ with $P_{rt} = \sum_l p_l \lvert M_{rt} \rvert^2$ |
  | 1 | V (receive V): $P_{vv} / P_{vh}$ |
  | 2 | H (receive H): $P_{hh} / P_{hv}$ |
  | 3 | Circular: $(P_{LL} + P_{RR}) / (P_{LR} + P_{RL})$ |
  | 4 | LHCP (receive L): $P_{LL} / P_{LR}$ |
  | 5 | RHCP (receive R): $P_{RR} / P_{RL}$ |

- **`pg`** — Path gain of all paths, LOS included,
  $\frac{1}{2} \sum_l p_l \sum_{r,t \in \{v,h\}} \lvert M_{rt} \rvert^2$; `(n_cir,)`

## Notes:
- A ratio with zero cross-polar power is infinite, or 0 if the co-polar power is zero too (also for a CIR without
  paths, which has `pg` = 0)
- A scaling of `polmat` does not change the XPR, but `pg`

## See also:
- [[calc_rician_k_factor]] (LOS and NLOS power with the same LOS window)
- [[channel.spherical]] (polarization transfer matrices of the paths)
MD!*/

py::tuple calc_xpr(py::list powers,
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

    quadriga_lib::calc_xpr<double>(powers_a, polmat_a, path_length_a, tx_pos_a, rx_pos_a,
                                   &xpr, &pg, include_los_b, window_size);

    auto xpr_py = qd_python_copy2numpy(&xpr);
    auto pg_py = qd_python_copy2numpy(&pg);

    return py::make_tuple(xpr_py, pg_py);
}

/*!MD
# calc_delay_spread
Calculate the RMS delay spread of each CIR from its path delays and powers

- With the path powers $p_l$ normalized to sum 1 over the paths of a CIR, the mean delay and the RMS delay spread are

  $$\bar\tau = \sum_l p_l \, \tau_l, \qquad \sigma_\tau = \sqrt{\sum_l p_l \, (\tau_l - \bar\tau)^2}$$

- Paths more than `threshold_dB` below the strongest path of their CIR are left out
- With `granularity` > 0, the delays are first rounded to the nearest multiple of `granularity`, and the powers of
  the paths with the same rounded delay are summed
- A CIR without power gives a spread and a mean delay of 0

## Usage:
```
ds, mean_delay = quadriga_lib.channel.calc_delay_spread( delays, powers, *, threshold_dB, granularity )
```

## Inputs:
- **`delays`** — Path delays in s; list of `n_cir` arrays `(n_path,)`, at least one CIR and one path per CIR; `n_path`
  may differ between the CIRs
- **`powers`** — Path powers, linear; list of `n_cir` arrays `(n_path,)`
- **`threshold_dB`** — Power range in dB below the strongest path that is kept, > 0; default: 100.0
- **`granularity`** — Spacing of the delay grid in s, finite and >= 0; 0 = no grid; with a grid, all
  delays must be finite and their span must be below $2^{53}$ grid steps; default: 0.0

## Outputs:
- **`ds`** — RMS delay spread $\sigma_\tau$ in s; `(n_cir,)`
- **`mean_delay`** — Mean delay $\bar\tau$ in s; `(n_cir,)`

## See also:
- [[quantize_delays]] (delays on a tap grid with interpolation)
- [[calc_angular_spread]] (angular spreads)
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
Calculate the Rician K-factor of each CIR, the ratio of the LOS to the NLOS power

- $K = P_\mathrm{LOS} / P_\mathrm{NLOS}$ with the sums of the path powers; LOS paths are those with a length
  `path_length` <= $d_\mathrm{TR}$ + `window_size`, $d_\mathrm{TR}$ the Tx-Rx distance, all others are NLOS
- Without NLOS power, $K$ is infinite, or 0 if the LOS power is zero too (also for a CIR without paths)

## Usage:
```
kf, pg = quadriga_lib.channel.calc_rician_k_factor( powers, path_length, tx_pos, rx_pos, *, window_size )
```

## Inputs:
- **`powers`** — Path powers, linear; list of `n_cir` arrays `(n_path,)`, at least one CIR; `n_path` may differ
  between the CIRs
- **`path_length`** — Path lengths in m; list of `n_cir` arrays `(n_path,)`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `(3,)` (fixed) or `(3, n_cir)`
- **`window_size`** — LOS window in m, >= 0; default: 0.01

## Outputs:
- **`kf`** — K-factor, linear; `(n_cir,)`
- **`pg`** — Sum of all path powers, linear; `(n_cir,)`

## See also:
- [[calc_xpr]] (XPR with the same LOS window)
- [[calc_delay_spread]] (delay spread of the same paths)
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
