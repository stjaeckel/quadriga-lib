# Channel statistics

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-channel-statistics

| Function | Summary |
| --- | --- |
| [acdf](#acdf) | Calculate the empirical averaged cumulative distribution function (CDF) |
| [calc_angular_spread](#calc_angular_spread) | Calculate azimuth and elevation angular spreads, optionally with spherical wrapping |
| [calc_cross_polarization_ratio](#calc_cross_polarization_ratio) | Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases |
| [calc_delay_spread](#calc_delay_spread) | Calculates RMS delay spread from per-CIR delays and linear-scale powers |
| [calc_rician_k_factor](#calc_rician_k_factor) | Calculate the Rician K-Factor from channel impulse response data |

---
<a id="acdf"></a>
## acdf
Calculate the empirical averaged cumulative distribution function (CDF)

Also in: [C++](../cpp/channel-statistics.md#acdf) · [MATLAB](../mex/channel-statistics.md#acdf)

- Computes per-column empirical CDFs by histogramming into bins and taking the normalized cumulative sum
- Averaged CDF is obtained by quantile-space averaging: for a fine probability grid, x-values from each
  column CDF are averaged, then mapped back to the bin grid
- Quantile statistics (mean and std) are reported at the 0.1, 0.2, ..., 0.9 probability levels
- `Inf` and `NaN` values are excluded from computation
- A data set without any finite value gets a NaN column in `cdf_per_set` and is left out of `cdf_avg`, `mu`
  and `sig`; if no set has finite values (possible only with given `bins`), `cdf_avg`, `mu` and `sig` are NaN
- If `bins` is empty, equally spaced bins spanning the data range are generated

### Usage:
```python
cdf_per_set, bins_out, cdf_avg, mu, sig = quadriga_lib.tools.acdf( data, bins, n_bins )
```

### Inputs:
- **`data`** — Input data matrix; each column is one independent data set; magnitudes below about 1e18, 
  larger values overflow in `sig`; `(n_samples, n_sets)`
- **`bins`** — Bin centers; used as-is if non-empty (must be finite and strictly ascending, else a
  `ValueError` is raised); if empty, equally spaced bins spanning the data range are generated;
  `(n_bins_in,)` or empty; if `None` or empty, bins are auto-generated.
- **`n_bins`** — Number of bins when auto-generating; must be >= 2; ignored when non-empty `bins_in`
  are provided; default: 201

### Outputs:
- **`cdf_per_set`** — Individual CDFs; one per column of data (NaN for a set without finite values); `(n_bins_out, n_sets)`
- **`bins_out`** — Auto-generated bins; copy of `bins` when non-empty `bins_in` are provided;
  `n_bins_out = n_bins` or `n_bins_out = n_bins_in`
- **`cdf_avg`** — Averaged CDF via quantile-space averaging across the data sets with finite values; `(n_bins_out,)`
- **`mu`** — Mean of the 0.1–0.9 quantiles across the data sets with finite values; `(9,)`
- **`sig`** — Standard deviation of the 0.1–0.9 quantiles across the data sets with finite values; `(9,)`

---
<a id="calc_angular_spread"></a>
## calc_angular_spread
Calculate azimuth and elevation angular spreads, optionally with spherical wrapping

Also in: [C++](../cpp/channel-statistics.md#calc_angular_spreads_sphere) · [MATLAB](../mex/channel-statistics.md#calc_angular_spread)

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
- When `quantize > 0`, paths within that angular distance are grouped and their powers summed

### Usage:
```python
azimuth_spread, elevation_spread, orientation, phi, theta = \
    quadriga_lib.tools.calc_angular_spread( az, el, powers, wrapping, calc_bank_angle, quantize )
```

### Inputs:
- **`az`** — Azimuth angles; range -pi to pi; list of length `n_cir`, each element a 1D array of length `n_path`
- **`el`** — Elevation angles; range -pi/2 to pi/2; list of length `n_cir`, each element a 1D array of length `n_path`
- **`powers`** — Path powers; list of length `n_cir`, each element a 1D array of length `n_path`
- **`wrapping`** — If True, enables spherical rotation; default: False
- **`calc_bank_angle`** — If True, computes the optimal bank angle analytically; only used when `wrapping = True`; default: False
- **`quantize`** — Angular quantization step in [deg]; paths within this distance are grouped; default: 0.0 (no quantization);
  otherwise it must be finite and >= 1e-3 deg, else a `ValueError` is raised

### Outputs:
- **`azimuth_spread`** — RMS azimuth angular spread; `(n_cir,)`
- **`elevation_spread`** — RMS elevation angular spread; `(n_cir,)`
- **`orientation`** — Euler angles [bank; tilt; heading] of an antenna pointing at the power-weighted mean
  direction (a positive tilt points down); `(3, n_cir)`
- **`phi`** — Rotated azimuth angles, the path azimuths in the local frame of that antenna; list of length `n_cir`
- **`theta`** — Rotated elevation angles; list of length `n_cir`

---
<a id="calc_cross_polarization_ratio"></a>
## calc_cross_polarization_ratio
Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases

Also in: [C++](../cpp/channel-statistics.md#calc_cross_polarization_ratio) · [MATLAB](../mex/channel-statistics.md#calc_cross_polarization_ratio)

- Computes aggregate XPR from polarization transfer matrices using the total-power-ratio method:
  co-pol and cross-pol powers are summed across all qualifying paths per CIR, and XPR is their ratio
- Inputs are lists so each CIR can have a different number of paths
- XPR is computed in both the linear V/H basis and the circular LHCP/RHCP basis
- `M` is the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation V/H), stored column-major with
  interleaved real/imaginary parts, 8 rows per path: rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx),
  rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx). Below, `M_<rx><tx>` names an entry by its receive and transmit
  polarisation: `M_vv` = rows 0-1, `M_hv` = rows 2-3, `M_vh` = rows 4-5, `M_hh` = rows 6-7
- V-XPR and H-XPR are ratios per receive polarisation: V-XPR is the power received in V from V transmission over
  the power received in V from H transmission; H-XPR likewise for the H receive polarisation
- The circular entries couple circular patterns through `M` as get_channels_spherical couples antenna patterns,
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
- Normalization of `M` does not affect XPR (cancels in the ratio) but does affect `pg`; magnitudes in `M` above about 1e18 overflow and give NaN
- If cross-pol power is zero and co-pol is positive, XPR is set to infinity; if both are zero, XPR
  is set to 0
- TX/RX positions may be fixed `(3, 1)` or mobile `(3, n_cir)`

### Usage:
```python
xpr, pg = quadriga_lib.tools.calc_cross_polarization_ratio( powers, M, path_length, tx_pos, rx_pos, \
    include_los, window_size )
```

### Inputs:
- **`powers`** — Path powers on linear scale per CIR; list of length `n_cir`, each element a 1D
  array of length `n_path`
- **`M`** — Polarization transfer matrices with interleaved real/imag parts per CIR; list of length
  `n_cir`, each element a 2D array of shape `(8, n_path)`
- **`path_length`** — Absolute TX-to-RX path lengths per CIR; list of length `n_cir`, each element
  a 1D array of length `n_path`
- **`tx_pos`** — Transmitter position [x; y; z]; `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`rx_pos`** — Receiver position [x; y; z]; `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`include_los`** — If True, includes LOS and near-LOS paths in the XPR calculation; default: False
- **`window_size`** — LOS exclusion window; paths with `path_length <= dTR + window_size` are excluded when
  `include_los = False`; default: 0.01

### Outputs:
- **`xpr`** — XPR on linear scale; `(n_cir, 6)`; columns (0-based):
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

---
<a id="calc_delay_spread"></a>
## calc_delay_spread
Calculates RMS delay spread from per-CIR delays and linear-scale powers

Also in: [C++](../cpp/channel-statistics.md#calc_delay_spread) · [MATLAB](../mex/channel-statistics.md#calc_delay_spread)

- Computes RMS delay spread from a set of delays and corresponding linear-scale powers per CIR
- Inputs are lists of 1D arrays so each CIR can have a different number of paths
- Paths with power below `p_max / 10^(0.1 · threshold)` are excluded; the default threshold of
  100 dB effectively includes all paths
- When `granularity > 0`, paths falling into the same delay bin of width `granularity` have their
  powers summed before the spread is computed

### Usage:
```python
ds, mean_delay = quadriga_lib.tools.calc_delay_spread( delays, powers, threshold, granularity )
```

### Inputs:
- **`delays`** — Delays per CIR; list of length `n_cir`, each element a 1D array of length `n_path`
- **`powers`** — Path powers on linear scale per CIR; list of length `n_cir`, each element a 1D
  array of length `n_path`
- **`threshold`** — Power threshold in [dB] relative to the strongest path; paths below threshold
  are excluded; default: 100.0
- **`granularity`** — Bin width in [s] for grouping paths in the delay domain; default: 0.0 (no grouping); must be
  finite and non-negative; when > 0, all delays must be finite and the delay span / granularity must be below 2^53,
  else a `ValueError` is raised

### Outputs:
- **`ds`** — RMS delay spread per CIR; `(n_cir,)`
- **`mean_delay`** — Mean delay per CIR; `(n_cir,)`

---
<a id="calc_rician_k_factor"></a>
## calc_rician_k_factor
Calculate the Rician K-Factor from channel impulse response data

Also in: [C++](../cpp/channel-statistics.md#calc_rician_k_factor) · [MATLAB](../mex/channel-statistics.md#calc_rician_k_factor)

- Computes the Rician K-Factor (KF): ratio of LOS power to NLOS power per CIR
- Inputs are lists of 1D arrays so each CIR can have a different number of paths
- LOS paths are those with length ≤ `dTR + window_size`, where `dTR` is the direct TX-RX distance;
  remaining paths are NLOS
- If total NLOS power is zero, KF is set to infinity; if total LOS power is zero, KF is set to 0
- TX/RX positions may be fixed `(3,)` / `(3, 1)` (reused for all CIRs) or mobile `(3, n_cir)`

### Usage:
```python
kf, pg = quadriga_lib.tools.calc_rician_k_factor( powers, path_length, tx_pos, rx_pos, window_size )
```

### Inputs:
- **`powers`** — Path powers on linear scale per CIR; list of length `n_cir`, each element a 1D
  array of length `n_path`
- **`path_length`** — Absolute TX-to-RX path lengths per CIR; list of length `n_cir`, each element
  a 1D array of length `n_path` matching `powers`
- **`tx_pos`** — Transmitter position [x; y; z]; `(3,)` or `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`rx_pos`** — Receiver position [x; y; z]; `(3,)` or `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`window_size`** — LOS window; paths with length ≤ `dTR + window_size` are treated as LOS;
  default: 0.01

### Outputs:
- **`kf`** — Rician K-Factor on linear scale; `(n_cir,)`
- **`pg`** — Total path gain (sum of all path powers) per CIR; `(n_cir,)`
