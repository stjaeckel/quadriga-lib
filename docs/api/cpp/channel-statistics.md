# Channel statistics

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-channel-statistics

| Function | Summary |
| --- | --- |
| [acdf](#acdf) | Calculate the empirical averaged cumulative distribution function (CDF) |
| [calc_angular_spreads_sphere](#calc_angular_spreads_sphere) | Calculate azimuth and elevation angular spreads, optionally with spherical wrapping |
| [calc_cross_polarization_ratio](#calc_cross_polarization_ratio) | Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases |
| [calc_delay_spread](#calc_delay_spread) | Calculates RMS delay spread from per-CIR delays and linear-scale powers |
| [calc_rician_k_factor](#calc_rician_k_factor) | Calculate the Rician K-Factor from channel impulse response data |

---
<a id="acdf"></a>
## acdf
Calculate the empirical averaged cumulative distribution function (CDF)

Also in: [MATLAB](../mex/channel-statistics.md#acdf) · [Python](../python/channel-statistics.md#acdf)

- Computes per-column empirical CDFs by histogramming into bins and taking the normalized cumulative sum
- Averaged CDF is obtained by quantile-space averaging: for a fine probability grid, x-values from each column CDF are averaged, 
  then mapped back to the bin grid
- Quantile statistics (mean and std) are reported at the 0.1, 0.2, ..., 0.9 probability levels
- `Inf` and `NaN` values are excluded from computation
- A data set without any finite value gets a NaN column in `cdf_per_set` and is left out of `cdf_avg`, `mu` and `sig`;
  if no set has finite values (possible only with user-supplied bins), `cdf_avg`, `mu` and `sig` are NaN
- If `bins` points to an empty vector, equally spaced bins spanning the data range are generated and stored back
  (data without any finite value then throws); if non-empty, those bin centers are used and must be finite and
  strictly ascending, else `std::invalid_argument` is thrown; if `nullptr`, bins are auto-generated internally

### Declaration:
```cpp
void quadriga_lib::acdf(const arma::Mat<dtype> &data,
    arma::Col<dtype> *bins = nullptr,
    arma::Mat<dtype> *cdf_per_set = nullptr,
    arma::Col<dtype> *cdf_avg = nullptr,
    arma::Col<dtype> *mu = nullptr,
    arma::Col<dtype> *sig = nullptr,
    arma::uword n_bins = 201);
```

### Inputs:
- **`data`** — Input data matrix; each column is one independent data set; `[n_samples, n_sets]`
- **`bins`** *(optional)* — Bin centers; auto-generated and stored back if pointing to empty vector, 
  used as-is if non-empty (finite, strictly ascending), ignored if `nullptr`; `[n_bins]`
- **`n_bins`** *(optional)* — Number of bins when auto-generating; must be >= 2; ignored when non-empty bins are provided

### Outputs:
- **`cdf_per_set`** *(optional)* — Individual CDFs, one per column of data (NaN for a set without finite values); `[n_bins, n_sets]`
- **`cdf_avg`** *(optional)* — Averaged CDF via quantile-space averaging across the data sets with finite values; `[n_bins]`
- **`mu`** *(optional)* — Mean of the 0.1–0.9 quantiles across the data sets with finite values, `[9]`
- **`sig`** *(optional)* — Standard deviation of the 0.1–0.9 quantiles across the data sets with finite values, `[9]`

---
<a id="calc_angular_spreads_sphere"></a>
## calc_angular_spreads_sphere
Calculate azimuth and elevation angular spreads, optionally with spherical wrapping

Also in: [MATLAB](../mex/channel-statistics.md#calc_angular_spread) · [Python](../python/channel-statistics.md#calc_angular_spread)

- Computes RMS azimuth and elevation angular spreads from power-weighted angles; each CIR may have a different number of paths.
- RMS spread formula: `sqrt(sum(pw .* d^2))` where `d` are wrapped deviations from the circular mean (3GPP TR 38.901 second-moment definition).
- With wrapping (`disable_wrapping = false`), the mean direction is computed in Cartesian coordinates and all paths are rotated so the centroid lies on the equator before computing spreads, avoiding pole singularity artifacts.
- When `calc_bank_angle = true` (with wrapping), an optimal bank angle maximizing azimuth spread is derived analytically from eigenvectors of the 2x2 power-weighted covariance matrix of centered angles.
- When `disable_wrapping = true` (the default), spreads are computed directly from raw angles; `orientation` will be zero and `phi`/`theta` equal the input `az`/`el`.
- When `quantize > 0`, paths within that angular distance are grouped and their powers summed before computing spreads;
  `quantize` must be 0 or a finite value >= 1e-3 deg (negative, NaN, Inf and smaller positive values throw).

### Declaration:
```cpp
void quadriga_lib::calc_angular_spreads_sphere(
    const std::vector<arma::Col<dtype>> &az,
    const std::vector<arma::Col<dtype>> &el,
    const std::vector<arma::Col<dtype>> &powers,
    arma::Col<dtype> *azimuth_spread = nullptr,
    arma::Col<dtype> *elevation_spread = nullptr,
    arma::Mat<dtype> *orientation = nullptr,
    std::vector<arma::Col<dtype>> *phi = nullptr,
    std::vector<arma::Col<dtype>> *theta = nullptr,
    bool disable_wrapping = true,
    bool calc_bank_angle = false,
    dtype quantize = (dtype)0);
```

### Inputs:
- **`az`** — Azimuth angles; range -pi to pi; `[n_cir]` vector, each element of length `n_path`
- **`el`** — Elevation angles; range -pi/2 to pi/2; same structure as `az`
- **`powers`** — Path powers in [W]; same structure as `az`
- **`disable_wrapping`** *(optional)* — If true, skips spherical rotation and computes spreads from raw angles; default:
  `true` (wrapping off, as in the MATLAB and Python wrappers)
- **`calc_bank_angle`** *(optional)* — If true, computes optimal bank angle analytically; only used when
  `disable_wrapping = false`; default: `false`
- **`quantize`** *(optional)* — Angular quantization step in [deg]; paths within this distance are grouped; 0 disables grouping;
  otherwise it must be finite and >= 1e-3 deg, else `std::invalid_argument` is thrown

### Outputs:
- **`azimuth_spread`** *(optional)* — RMS azimuth spread; `[n_cir]`
- **`elevation_spread`** *(optional)* — RMS elevation spread; `[n_cir]`
- **`orientation`** *(optional)* — Power-weighted mean orientation in Euler angles [bank; tilt; heading]; `[3, n_cir]`
- **`phi`** *(optional)* — Rotated azimuth angles; `[n_cir]` vector, each element of length `n_path`
- **`theta`** *(optional)* — Rotated elevation angles; same structure as `phi`

---
<a id="calc_cross_polarization_ratio"></a>
## calc_cross_polarization_ratio
Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases

Also in: [MATLAB](../mex/channel-statistics.md#calc_cross_polarization_ratio) · [Python](../python/channel-statistics.md#calc_cross_polarization_ratio)

- Computes aggregate XPR from polarization transfer matrices using the total-power-ratio method:
  co-pol and cross-pol powers are summed across all qualifying paths per CIR, and XPR is their ratio.
- XPR is computed in both the linear V/H basis and the circular LHCP/RHCP basis.
- LOS paths are identified by comparing path length against direct TX-RX distance `dTR`; paths with
  `path_length <= dTR + window_size` are excluded by default (`include_los = false`).
- Polarization transfer matrix `M` is the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation
  V/H), stored column-major with interleaved real/imaginary parts, 8 rows per path: rows 0-1 = (V rx, V tx),
  rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx). Below, `M_<rx><tx>` names an entry by its
  receive and transmit polarisation: `M_vv` = rows 0-1, `M_hv` = rows 2-3, `M_vh` = rows 4-5, `M_hh` = rows 6-7
- V-XPR and H-XPR are ratios per receive polarisation: V-XPR is the power received in V from V transmission over
  the power received in V from H transmission; H-XPR likewise for the H receive polarisation
- The circular entries couple circular patterns through `M` as [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) couples antenna patterns,
  `c(r, t) = r^T M t` (plain transpose), with the same circular vectors at both ends, each in that antenna's own
  (e_theta, e_phi) frame: L = (V, H) = (1, +j) / sqrt(2) (LHCP), R = (1, -j) / sqrt(2) (RHCP). The names follow the
  IEEE definition (rotation seen along the propagation direction) for the time convention exp(+j 2 pi f t), which the
  library's coefficients exp(-j 2 pi f tau) imply, and e_theta x e_phi = the outward direction of the pattern.
  `M_<rx><tx>`: `M_LL = (M_vv - M_hh + j(M_vh + M_hv)) / 2`, `M_RR = (M_vv - M_hh - j(M_vh + M_hv)) / 2`,
  `M_LR = (M_vv + M_hh + j(M_hv - M_vh)) / 2` (receive L, transmit R), `M_RL = (M_vv + M_hh + j(M_vh - M_hv)) / 2`.
  Co-polar means the same circular vector at both ends: the LOS matrix `diag(1, -1)` of [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) is
  co-polar (handedness kept), `diag(1, 1)` (one mirrored component, as by a single reflection) is cross-polar.
- LHCP-XPR and RHCP-XPR are ratios per receive polarisation, like V-XPR and H-XPR.
- Normalization of `M` does not affect XPR (cancels in ratio) but does affect `pg`.
- If cross-pol power is zero and co-pol is positive, XPR is set to infinity; if both are zero, XPR is set to 0.
- TX/RX positions may be fixed `[3, 1]` or mobile `[3, n_cir]`.

### Declaration:
```cpp
void quadriga_lib::calc_cross_polarization_ratio(
    const std::vector<arma::Col<dtype>> &powers,
    const std::vector<arma::Mat<dtype>> &M,
    const std::vector<arma::Col<dtype>> &path_length,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &rx_pos,
    arma::Mat<dtype> *xpr = nullptr,
    arma::Col<dtype> *pg = nullptr,
    bool include_los = false,
    dtype window_size = 0.01);
```

### Inputs:
- **`powers`** — Path powers in [W]; `[n_cir]` vector, each element of length `n_path`
- **`M`** — Polarization transfer matrices; `[n_cir]` vector, each element of size `[8, n_path]`
- **`path_length`** — Absolute TX-to-RX path lengths; same structure as `powers`
- **`tx_pos`** — Transmitter position [x; y; z]; `[3, 1]` or `[3, n_cir]`
- **`rx_pos`** — Receiver position [x; y; z]; `[3, 1]` or `[3, n_cir]`
- **`include_los`** *(optional)* — If true, includes LOS and near-LOS paths in the XPR calculation
- **`window_size`** *(optional)* — LOS exclusion window; paths with `path_length <= dTR + window_size` are excluded
  when `include_los = false`

### Outputs:
- **`xpr`** *(optional)* — XPR on linear scale; `[n_cir, 6]`; columns:
   | Col | Description                                                     |
   | --- | --------------------------------------------------------------- |
   | 0   | Aggregate linear XPR (total V+H co-pol / total V+H cross-pol)   |
   | 1   | V-XPR: sum(abs(M_vv)^2) / sum(abs(M_vh)^2) (receive V)          |
   | 2   | H-XPR: sum(abs(M_hh)^2) / sum(abs(M_hv)^2) (receive H)          |
   | 3   | Aggregate circular XPR (total L+R co-pol / total L+R cross-pol) |
   | 4   | LHCP XPR: sum(abs(M_LL)^2) / sum(abs(M_LR)^2) (receive L)       |
   | 5   | RHCP XPR: sum(abs(M_RR)^2) / sum(abs(M_RL)^2) (receive R)       |
- **`pg`** *(optional)* — Total path gain summed over all paths (including LOS) as
  `0.5 * sum(powers * (abs(M_vv)^2 + abs(M_hv)^2 + abs(M_vh)^2 + abs(M_hh)^2))`; `[n_cir]`

---
<a id="calc_delay_spread"></a>
## calc_delay_spread
Calculates RMS delay spread from per-CIR delays and linear-scale powers

Also in: [MATLAB](../mex/channel-statistics.md#calc_delay_spread) · [Python](../python/channel-statistics.md#calc_delay_spread)

- Paths with power below `p_max / 10^(0.1 * threshold)` are excluded; default threshold of 100 dB effectively includes all paths.
- When `granularity > 0`, paths falling into the same delay bin of width `granularity` have their powers summed before computing the spread; function recurses on the binned profile.
  The bins are placed on a grid common to all CIRs; only occupied bins are stored, so memory does not grow with the delay span.
  All delays must then be finite, and the global delay span divided by `granularity` must be below 2^53.

### Declaration:
```cpp
arma::Col<dtype> quadriga_lib::calc_delay_spread(
    const std::vector<arma::Col<dtype>> &delays,
    const std::vector<arma::Col<dtype>> &powers,
    dtype threshold = 100.0,
    dtype granularity = 0.0,
    arma::Col<dtype> *mean_delay = nullptr);
```

### Inputs:
- **`delays`** — Delays in [s] per CIR; `[n_cir]` vector, each element a column vector of length `n_path`
- **`powers`** — Path powers in linear scale [W]; same structure as `delays`
- **`threshold`** *(optional)* — Power threshold in [dB] relative to strongest path; paths below threshold are excluded
- **`granularity`** *(optional)* — Bin width in [s] for grouping paths in the delay domain; 0 disables grouping; must be finite
  and non-negative; when > 0, non-finite delays or a span / granularity of 2^53 or more throw `std::invalid_argument`

### Outputs:
- **`mean_delay`** *(optional)* — Mean delay in [s] per CIR; `[n_cir]`

### Returns:
- RMS delay spread in [s] for each CIR; `[n_cir]`

### See also:
- [quantize_delays](channel-functions.md#quantize_delays) (for mapping delays to a fixed tap grid)
- [calc_rician_k_factor](#calc_rician_k_factor) (for calculating K-factor)

---
<a id="calc_rician_k_factor"></a>
## calc_rician_k_factor
Calculate the Rician K-Factor from channel impulse response data

Also in: [MATLAB](../mex/channel-statistics.md#calc_rician_k_factor) · [Python](../python/channel-statistics.md#calc_rician_k_factor)

- KF = LOS power / NLOS power; LOS paths are those with length ≤ `dTR + window_size`, where `dTR` is the direct TX-RX distance.
- If total NLOS power is zero, KF is set to `HUGE_VAL`; if total LOS power is zero, KF is set to 0.
- TX/RX positions may be fixed `[3, 1]` (reused for all snapshots) or mobile `[3, n_cir]`.

### Declaration:
```cpp
void quadriga_lib::calc_rician_k_factor(
    const std::vector<arma::Col<dtype>> &powers,
    const std::vector<arma::Col<dtype>> &path_length,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &rx_pos,
    arma::Col<dtype> *kf = nullptr,
    arma::Col<dtype> *pg = nullptr,
    dtype window_size = 0.01);
```

### Inputs:
- **`powers`** — Path powers in [W]; `[n_cir]` vector, each element of length `n_path`
- **`path_length`** — Absolute TX-to-RX path lengths; same structure as `powers`
- **`tx_pos`** — Transmitter position in Cartesian coordinates [x; y; z]; `[3, 1]` or `[3, n_cir]`
- **`rx_pos`** — Receiver position in Cartesian coordinates [x; y; z]; `[3, 1]` or `[3, n_cir]`
- **`window_size`** *(optional)* — LOS window; paths with length ≤ `dTR + window_size` are treated as LOS

### Outputs:
- **`kf`** *(optional)* — Rician K-Factor on linear scale; `[n_cir]`
- **`pg`** *(optional)* — Total path gain (sum of all path powers) in [W]; `[n_cir]`
