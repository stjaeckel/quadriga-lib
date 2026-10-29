// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# ssf_clusters
Cluster delays, powers and angles of a set of links, TR 38.901 clause 7.5 steps 5 to 7

- Turns the large-scale parameters of [[lsp_generate]] and the random values of [[ssf_draw]] into the delays, powers and
  angles of the clusters of each link, by one of three methods:
- `"38.901"` (default): TR 38.901 clause 7.5 steps 5 to 7 (exponential delays, powers with per-cluster shadowing,
  inverse Gaussian and Laplacian angle mapping with the scaling factors C_phi and C_theta); one frequency only
- `"38.901_MF"`: the multi-frequency method of TR 38.901 clause 7.6.5.1 (nominal delays and angles from anchor spreads,
  frequency-dependent powers, scaling by the K-factor per frequency)
- `"quadriga"`: the cluster generation of QuaDRiGa (paper: Jaeckel, Raschkowski, Burkhardt, Thiele, "A Spatially
  Consistent Geometric D2D Small-Scale Fading Model for Multiple Frequencies", 2019): continuous in space,
  frequency-independent delays and angles for several frequencies, swap-reciprocal, exact delay spread
- Slots per link: slot 0 = LOS path (delay 0, geometric LOS angles); slot 1 = ground reflection if `GR_enabled` is > 0;
  then the `N = NumClusters - 1` NLOS clusters, slot `1 + G + n` for cluster `n` (`G = 1` with ground reflection, else
  0); `n_slot = NumClusters + G`. Slot `1 + G + n` always belongs to column `n` of each block of the random values (the
  clusters are not sorted)
- LOS condition: the scenario is a LOS scenario if `KF_mu > -30` (dB). In an NLOS scenario the LOS slot and the
  ground-reflection slot have power 0 whatever K-factor `lsp` holds, and every method uses its NLOS formulas. Legacy
  QuaDRiGa "NLOS" files with `KF_mu > -30` (e.g. `BERLIN_UMa_NLOS`, `QuaDRiGa_Industrial_NLOS`) therefore run the LOS
  formulas
- Units and conventions: delays in [s] relative to the LOS delay (absolute delay = d3D / c + delay); angles in [rad] in
  global coordinates, elevation convention (elevation = 90 deg - zenith); azimuths in [-pi, pi), elevations in [-pi/2,
  pi/2]; the LOS angles from the positions: AoD = atan2(dy, dx), EoD = atan2(dz, d2D), AoA = AoD + pi (wrapped), EoA =
  -EoD
- Powers are the powers of the channel coefficients of TR 38.901 eq. (7.5-30): LOS slot `K/(K+1)`, NLOS clusters
  `P_n/(K+1)` (sum `1/(K+1)`); in NLOS scenarios K = 0. Path loss and shadow fading are not included (step 12, the
  caller)
- Per-link number of clusters (TR 38.901 clause 7.6.15): `n_cluster(l)` NLOS clusters are active; the other slots get
  delay 0, power 0 and the LOS angles. Every normalisation, minimum, maximum and scaling runs over the active clusters;
  in `"38.901"` the scaling factors C_phi, C_theta use `n_cluster(l)`. `n_active(l)` counts the active clusters with a
  power > 0 at one frequency at least (removed clusters are not counted)
- Random values, `N` columns per block (see [[ssf_draw]]): the shared set `rnd` (block 0 delays, blocks 1 to 4 signs of
  AOD, AOA, ZOD, ZOA, blocks 5 to 8 angle offsets of AOD, AOA, ZOD, ZOA) and one per-frequency set (the cluster
  shadowing) per frequency in `rnd_freq`. `"38.901"` uses all nine blocks and the shadowing; `"38.901_MF"` uses blocks
  0, 5, 6, 7, 8 and the shadowing per frequency; `"quadriga"` uses blocks 0, 5, 6, 7, 8. Normal blocks are mapped to
  Uniform(0,1) by `0.5·erfc(-x/sqrt(2))` where a method needs uniform values
- Spatial consistency with `"38.901"` (TR 38.901 clause 7.6.3.1): the caller evaluates the sign and offset blocks (1 to
  8) at the drop or segment-start positions and reuses these values along the track; the delays and the shadowing follow
  the positions. The method still jumps along tracks (see the table below)
- Inputs used per method (inputs that a method does not use are ignored and not validated):

| Input | `38.901` | `38.901_MF` | `quadriga` |
| --- | --- | --- | --- |
| `rnd` blocks | 0 to 8 | 0, 5, 6, 7, 8 | 0, 5, 6, 7, 8 |
| `rnd_freq` | required (one frequency) | required | ignored |
| `lsp_anchor` | ignored | required | ignored |
| `eod_offset_deg` | NLOS scenarios: mu_offset,ZOD = -value | ignored | NLOS scenarios: mean over the frequencies |
| `abs_toa` | NLOS scenarios: added | NLOS scenarios: added | NLOS scenarios: shift to the offset if > 0 |
| `gr_eps` | TR 38.901 clause 7.6.8 | TR 38.901 clause 7.6.8 | coefficient powers; power split inside the scaling |
| `o2i` | NLOS: mean ZOA 90 deg | ignored | ignored |
| `n_cluster` | yes, also in C_phi, C_theta | yes | yes |

- Absolute time of arrival (TR 38.901 clause 7.6.9), NLOS scenarios only: `"38.901"` and `"38.901_MF"` add `abs_toa(l)`
  to every active NLOS delay (eq. (7.6-43); with `"38.901"` the first cluster is then at `abs_toa(l)`); `"quadriga"`
  shifts the NLOS delays so that the first starts at `abs_toa(l)` where it is > 0. In LOS scenarios `abs_toa` is ignored
  (eq. (7.6-44) has no offset)
- Ground reflection (TR 38.901 clause 7.6.8), on if `GR_enabled > 0`; the ground is the plane z = 0 (`tx_pos` and
  `rx_pos` must have z >= 0). Slot 1: delay `(d_GR - d3D)/c`, computed as `4·h_t·h_r/((d_GR + d3D)·c)` with
  `d_GR = sqrt(d2D² + (h_t + h_r)²)`; AoD and AoA of the LOS path; EoD = EoA = `-atan2(h_t + h_r, d2D)`. Reflection
  coefficients per frequency of eqs. (7.6-38), (7.6-39) with the permittivity `eps = conj(gr_eps)` (the sign of eq.
  (7.6-40); the imaginary part of `gr_eps` is >= 0 as from [[lsp_generate]]), `cos theta = -(h_t + h_r)/d_GR'`,
  `sin² theta = (d2D/d_GR')²`, `d_GR' = max(d_GR, 1e-5)`; a coefficient with a zero denominator (only at grazing with
  `eps = 1`) is 0. Polarisation-averaged reflection power `R² = (abs(R_par)² + abs(R_perp)²)/2`
- Ground-reflection power: `"38.901"`, `"38.901_MF"`: `K/(K+1)·(d3D/d_GR')²·R²` (eq. (7.6-32)), the other slots
  unchanged, so the total power is above 1 (no renormalisation; median +1.6 dB in UMi LOS at 6 GHz); `"quadriga"`:
  `K/(K+1)·R²` (QuaDRiGa's two-ray channel, eq. (7.6-32) without the factor d3D/d_GR); its spread scaling splits the LOS
  power into `(1 - R²)·K/(K+1)` and `R²·K/(K+1)`, so the delay spread computed from the slot powers of a
  ground-reflection link is about 0.85 of the input. In NLOS scenarios the ground-reflection power is 0
- Coefficients from the slot powers, the same for all methods: LOS path amplitude `sqrt(power)` with the polarisation
  `diag(1, -1)` of eq. (7.5-29); ground reflection amplitude `sqrt(power)` with the polarisation
  `diag(R_par, -R_perp)/R`, `R = sqrt(R²)`, which applies the reflection exactly once and gives eq. (7.6-32); `R² = 0`
  gives a zero coefficient. A consumer without polarisation uses `sqrt(power)·exp(j·arg(R_par))` (`arg(0) = 0`) for the
  reflection: it keeps the reflection phase (about pi near grazing), without which the two-ray interference is reversed.
  NLOS clusters: amplitude `sqrt(power)`
- Permittivity: real part of `gr_eps` >= 1 and imaginary part >= 0; [[lsp_generate]] gives real parts of 3.0 to 44
  between 0.5 and 100 GHz; its fit falls below 1 above about 1.3 THz, where this function rejects it
- `NumClusters = 1` (`N = 0`): the LOS path only, with power 1 in all methods and all scenarios (K is not used), plus
  the ground reflection if enabled, with the power of the method's rule for `K/(K+1) = 1`; `rnd` is empty, `rnd_freq`
  and `lsp_anchor` are not needed, `n_cluster` entries are 0, `n_active` is 0
- Method `"38.901"`, per link with `N_l` active clusters, in degrees; `u_n` = block 0, `s_n` the sign block and `y_n`
  the offset block of the angle type, `z_n` the shadowing value, `r_tau = r_DS`, `zeta = LNS_ksi`, `K` = KF of `lsp`
  (LOS scenarios): `tau'_n = -r_tau·DS·ln(u_n)`, `tau_n = tau'_n - min(tau')`; the first cluster `f` has the smallest
  `tau'_n` (lowest index on ties). LOS: `K_lim = max(10·lg K, -5)`,
  `C_tau = 0.7705 - 0.0433·K_lim + 0.0002·K_lim² + 0.000017·K_lim³`, delay `tau_n / C_tau` (eq. (7.5-4)); NLOS: delay
  `tau_n`. `P_n = P'_n / sum(P')`, `P'_n = exp(-tau_n·(r_tau - 1)/(r_tau·DS))·10^(-zeta·z_n/10)` (eqs. (7.5-5), (7.5-6),
  computed in the log domain). Clusters with `P_n < 10^(-2.5)·max(P)` are removed: power 0, delay and angles kept, no
  renormalisation. Output power `P_n/(K+1)`. Angle mapping with `Pa_n = P_n/(K+1) + (n = f ? K/(K+1) : 0)` (eq. (7.5-8);
  NLOS `Pa_n = P_n`): azimuth `a_n = 2·(AS/1.4)·sqrt(-ln(Pa_n/max(Pa)))/C_phi`, zenith
  `a_n = -ZS·ln(Pa_n/max(Pa))/C_theta`, with
  `C_phi = C_phi^NLOS(N_l)·(1.1035 - 0.028·K_lim - 0.002·K_lim² + 0.0001·K_lim³)` and
  `C_theta = C_theta^NLOS(N_l)·(1.3086 + 0.0339·K_lim - 0.0077·K_lim² + 0.0002·K_lim³)` in LOS (NLOS: the table values);
  `C_phi^NLOS`, `C_theta^NLOS` from TR 38.901 Tables 7.5-2 and 7.5-4, linearly interpolated between the table entries
  and constant beyond both ends. `v_n = X_n·a_n + Y_n`, `X_n = +1` if `s_n >= 0`, else -1, `Y_n = y_n·AS/7`. NLOS: AOD =
  LOS AOD + v_n, AOA = LOS AOA + v_n, ZOD = LOS ZOD + mu_offset,ZOD + v_n with `mu_offset,ZOD = -eod_offset_deg(l, 0)`,
  ZOA = (90 if `o2i(l)` else LOS ZOA) + v_n. LOS: angle = LOS angle + v_n - v_f (eqs. (7.5-12), (7.5-17): the first
  cluster at delay 0 in the LOS direction; `o2i` and the ZOD offset have no effect). Zenith angles are wrapped to [0,
  360] and mapped to `360 - theta` in [180, 360] (ZOA and ZOD), azimuths wrapped
- Deviations of `"38.901"` from the TR text: the K-factor in eqs. (7.5-3), (7.5-10), (7.5-15) is limited to >= -5 dB
  (below, C_theta turns negative near -10 dB); for links far below -5 dB the ASA is then about 0.86 of the input. The
  delays are not sorted: the TR's sort of eq. (7.5-2) only relabels the clusters, and keeping the cluster of each field
  column keeps the clusters stable under spatial consistency
- Method `"38.901_MF"` (TR 38.901 clause 7.6.5.1 as written), per link with `N_l` active clusters, in degrees;
  `r = 1.5`; anchor spreads `DS0`, `ASD0`, `ASA0`, `ZSD0`, `ZSA0` from `lsp_anchor`; DS, ASD, ASA, ZSD, ZSA and `K` from
  `lsp` at frequency k (`K = 0` in NLOS scenarios); `u_n` = block 0, `y_n` the offset block of the angle type (blocks 5
  to 8: Y_n, Z_n of the TR and, through `V = Phi(y)`, V_n, W_n), `z_{n,k}` the shadowing at frequency k,
  `zeta = LNS_ksi`. Step 5', the same at all frequencies: `tau'_n = -r·DS0·ln(u_n)`; azimuths
  `phi'_n = arg(exp(-j·r·AS0·y_n))`, i.e. `-r·AS0·y_n` wrapped to [-180, 180); zeniths
  `theta'_n = arg(exp(-j·r·ZS0·sgn(y_n)·ln(erfc(abs(y_n)/sqrt(2)))/sqrt(2)))`, which equals the TR's
  `sgn(V_n - 0.5)·ln(1 - 2·abs(V_n - 0.5))` with full precision near V = 0 or 1. LOS: cluster 0 (the TR's cluster 1) has
  `tau'_0 = 0` and all four nominal angles 0. Step 6' per frequency:
  `P'_n = exp(-tau'_n·g_DS - (phi'_AOD,n·g_ASD/sqrt(2))² - (phi'_AOA,n·g_ASA/sqrt(2))² -`
  `sqrt(2)·abs(theta'_ZOD,n)·g_ZSD - sqrt(2)·abs(theta'_ZOA,n)·g_ZSA)·10^(-zeta·z_{n,k}/10)`
  (eq. (7.6.30a)) with `g_DS = max(r·DS0 - DS, 0)/(DS·r·DS0)`, `g_ASD = sqrt(max((r·ASD0)² - ASD², 0))/(ASD·r·ASD0)`
  (ASA likewise), `g_ZSD = max(r·ZSD0 - ZSD, 0)/(ZSD·r·ZSD0)` (ZSA likewise); a g factor is 0 where the spread at k is
  at least r times the anchor spread. `P_n = P'_n/sum(P')` (log domain), output power `P_n/(K+1)`; with the LOS slot
  `K/(K+1)` this is eq. (7.6.30h). Step 7' per frequency: delay `sqrt(1 + K/2)·tau'_n` (LOS) or `tau'_n` (NLOS), plus
  `abs_toa` in NLOS scenarios; angle = LOS angle + `sqrt(1 + K)·angle'_n` (LOS) or LOS angle + `angle'_n` (NLOS) for
  AOD, AOA, ZOD, ZOA (eqs. (7.6.30i) to (7.6.30m) without the ray terms); azimuths wrapped, zenith angles folded as in
  `"38.901"`
- `"38.901_MF"` follows the TR text literally: the nominal angles are wrapped in step 5' and scaled by `sqrt(1 + K)` in
  step 7', so in LOS an angle that crosses +-180 deg in step 5' jumps by `sqrt(1 + K)·360` deg before the final wrap
  (about 0.05 to 0.07 jumps per metre along LOS tracks); in LOS the delays and angles depend on the frequency through K
  (they are equal at all frequencies if K is); the NLOS delays are not shifted to start at 0 (no cluster at delay 0 in
  NLOS); no sort, no -25 dB removal, no C factors, no ZOD offset (mean EoD about 2 to 3 deg off the clause 7.5
  statistics in UMi and UMa NLOS), no O2I rule; the per-cluster shadowing is drawn per frequency (TR clause 7.6.5); the
  anchor LSPs (TR: e.g. at 2 GHz) are an input; the zenith fold of clause 7.5 step 7 is applied to the final zenith
  angles, for which clause 7.6.5.1 states no rule
- Method `"quadriga"` (the paper's section II with the changes below), per link with `N_l` active clusters, in rad;
  `u_n` = block 0, `y^x_n` the offset block of angle type x (blocks 5 to 8: AOD, AOA, EOD, EOA); DS, the four angular
  spreads `AS_x` and `K` from `lsp` at frequency k (`K = 0` in NLOS scenarios); `R²_k` the reflection power and `G = 1`
  with ground reflection, else 0. Delays and angles are the same at all frequencies. Initial values (paper eqs. (1),
  (4), (5)): `tau~_n = -ln(u_n)` (unitless), `a~^x_n = (Phi(y^x_n) - 0.5)·pi` in (-pi/2, pi/2),
  `Phi(y) = 0.5·erfc(-y/sqrt(2))`. Centres: the LOS azimuths; the elevations `(1 - w)·LOS elevation + w·E_GR` with `w`
  the mean of `R²_k` over the frequencies in LOS scenarios with ground reflection, else `w = 0`. LOS path and ground
  reflection enter the scaling as fixed offsets (azimuth 0, elevation their elevation minus the centre). Powers per
  frequency (paper eqs. (6) to (12)): `P_n` proportional to
  `exp(-tau~_n·g_DS - (a~^AOD_n)²·g_AOD - (a~^AOA_n)²·g_AOA -`
  `abs(a~^EOD_n)·g_EOD - abs(a~^EOA_n)·g_EOA)`, normalised to
  `1/(K+1)`; no per-cluster shadowing. One frequency: `g_DS = r_tau - 1`, `g_AOD = g_AOA = 0.56`,
  `g_EOD = g_EOA = 0.76`. Several frequencies: `g_DS = -1.5·ln(1.2·d - 0.15)`, `d = DS_k/(max DS + min DS)` limited to
  [0.15, 0.85] (`r_tau - 1` if DS does not depend on the frequency); azimuth `g = -2.2·ln(1.5·b - 0.35)`, elevation
  `g = -3.4·ln(1.2·b - 0.1)`, `b = max(0.75·AS_k/max AS, 0.25)`. Scaling weights: LOS `K/(K+1)·(1 - G·R²)`, ground
  reflection `K/(K+1)·G·R²`, clusters `P_n` (sum 1). Delay scale per frequency (paper eqs. (13), (14)): the factor `s_k`
  for which the weighted delays (LOS 0, ground reflection its delay, clusters `s_k·tau~_n`) have the delay spread DS_k,
  in closed form (1 if the clusters have no spread; with ground reflection at least the factor that minimises the
  spread); delays `s·tau~_n` in [s] with `s` the mean of `s_k`. Angle scale per type and frequency: the input spread
  over the wrapped spread of the initial angles, then up to 9 updates of the factor in dB until the linear
  power-weighted spread of the scaled cluster offsets together with the fixed offsets equals the input to 1e-4 dB; the
  mean over the frequencies, at most 3 (azimuth) and 1.5 (elevation) (paper eq. (18)). Scaled elevation offsets are
  mirrored at +-pi/2. Angles: AOD, AOA = LOS azimuth + offset (wrapped); EOD = centre + `delta` + offset, EOA = centre +
  offset, both mirrored at the poles (the azimuth is kept); `delta` = the mean of `eod_offset_deg` over the frequencies
  in NLOS scenarios (TR eq. (7.5-19) with mu_offset,ZOD = -offset), 0 in LOS scenarios. Absolute TOA in NLOS scenarios:
  where `abs_toa(l) > 0` the NLOS delays are shifted so that the first is at `abs_toa(l)`
- Changes of `"quadriga"` against the paper and QuaDRiGa: the LOS direction is added to the scaled offsets (azimuth
  wrapped, elevation mirrored at the poles) instead of the 3-D rotation of the paper (which widens the elevation spread
  of departures); the EoD offset of the TR is added in NLOS scenarios; no weakening of the arrival elevations for links
  with a large height difference; no pull of the NLOS elevations towards the ground reflection in NLOS scenarios; no
  fixed angles for one or two NLOS clusters. `o2i` is not used
- Properties of `"quadriga"`: at one frequency the delay spread of the output (all slots) equals DS, also with the
  absolute TOA; with ground reflection the scaling uses the split weights above, so the delay spread from the slot
  powers is smaller (about 0.85 of DS in UMi and UMa LOS at 6 GHz, medium ground), and the elevation spreads cannot fall
  below the spread of the LOS path and the reflection alone (the target is then not reached). Below the caps the linear
  power-weighted spread of the angle offsets equals the input; with the circular definition of
  [[calc_angular_spread]] the medians are 1.00 for ASD, ESD, ESA and ASA in NLOS, while in LOS the LOS power
  bounds the ASA (median 0.86 UMi, 0.59 UMa). With several frequencies the averaged scales leave a per-frequency spread
  error of about +-15 to 20 %. Continuous along tracks; delays and angles independent of the frequency; the relation
  between angle offset and power is weaker than in clause 7.5
- Swap reciprocity: with mirrored random values (dual-mobility values of [[ssf_draw]], or the blocks 1 and 2, 3 and 4, 5
  and 6, 7 and 8 exchanged) and mirrored LSPs (ASD and ASA, ESD and ESA exchanged, also in `lsp_anchor`), swapping
  `tx_pos` and `rx_pos` gives the same delays and powers with departure and arrival angles exchanged; `"38.901"` needs
  `eod_offset_deg` = 0 and no O2I link, `"quadriga"` `eod_offset_deg` = 0
- Properties of the methods (typical values for UMi and UMa at 6 and 30 GHz):

| Property | `38.901` | `38.901_MF` | `quadriga` |
| --- | --- | --- | --- |
| Output/input spread, median | DS 0.96-1.02, ASD 0.95-1.02, ASA 0.99 (NLOS) / 0.80-0.89 (LOS), ESD 0.97-1.04, ESA 0.97-1.03 | NLOS 0.88-0.94, LOS 0.44-0.77 | DS 1.00, ASD 1.00, ESD 1.00, ESA 1.00, ASA 1.00 (NLOS) / 0.59-0.86 (LOS) |
| Continuous along a track | no: in LOS the first cluster changes (0.03-0.11 jumps/m), the -25 dB rule switches clusters (0.03-0.19 /m), signs and offsets must be frozen per segment; nearby UTs see sign flips (0.5-1.5 jumps per m of separation) | NLOS yes; LOS no (0.05-0.07 /m, wrap before the scaling by sqrt(1+K)) | yes |
| Delays and angles equal at all frequencies | one frequency only | NLOS yes; LOS only if K does not depend on frequency | yes |
| Exact spreads | no (C factors calibrated) | no | DS exact at one frequency; AS exact below the caps (linear definition); averaged scales with several frequencies |
| Swap-reciprocal (mirrored fields and LSPs) | yes, if `eod_offset_deg` = 0 and no O2I link | yes | yes, if `eod_offset_deg` = 0 |
| Per-cluster shadowing | yes (zeta) | yes (Q_n) | no |
| TR terms mu_offset,ZOD (NLOS), O2I ZOA 90 deg | both | neither | offset only |
| `NumClusters = 1` | LOS path only (+ GR, eq. (7.6-32)) | LOS path only (+ GR, eq. (7.6-32)) | LOS path only (+ GR, R²) |
| Ground reflection (coefficient powers) | TR 7.6.8: K/(K+1)·(d3D/d_GR)²·R² | TR 7.6.8 | K/(K+1)·R²; power split inside the scaling |

- Recommended use: `"38.901"` for drop-based calibration at one frequency; `"quadriga"` for tracks, several frequencies
  or device-to-device links; `"38.901_MF"` where the multi-frequency method of the TR is required
- `scenpar`, `lsp`, `rnd` and `rnd_freq` must come from the same scenario ([[lsp_generate]], [[ssf_draw]]); this is not
  checked beyond the sizes. `scenpar` is completed with the defaults of missing keys and validated by
  [[scenpar_validate]]; its messages pass through unchanged
- `n_link = 0` (`rx_pos` of shape `(0, 3)`) gives outputs with 0 rows
- Validation order: `method`, positions, `scenpar`, `lsp`, one frequency for `"38.901"`, `rnd`, `rnd_freq`,
  `lsp_anchor`, `eod_offset_deg`, `abs_toa`, ground-reflection inputs and heights, `o2i`, `n_cluster`; all checks run
  before any output is written

## Usage:
```
delay, power, aod, eod, aoa, eoa, n_active = quadriga_lib.gbsm.ssf_clusters( scenpar, tx_pos, rx_pos, lsp, rnd, \
    method, rnd_freq, lsp_anchor, eod_offset_deg, abs_toa, gr_eps, o2i, n_cluster )
```

## Inputs:
- **`scenpar`** — Scenario parameters; dict as returned by [[conf_file_read]] or [[scenpar_validate]] (the
  entry `SubpathMethod` is ignored), or any subset of its keys; missing keys have their default value;
  `None` = all defaults; used: `NumClusters`, `NumClusters_min`, `KF_mu` (LOS condition), `r_DS`, `LNS_ksi`,
  `GR_enabled`
- **`tx_pos`** — Tx positions in m; one row is used for all links; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`lsp`** — Large-scale parameters from [[lsp_generate]], finite; DS, ASD, ASA, ESD, ESA (slices 0, 3, 4, 5, 6)
  > 0, KF (slice 1) >= 0; slices 2 and 7 are not used; `(n_link, n_freq, 8)`
- **`rnd`** — Shared cluster set from [[ssf_draw]] (`set_id = -1`), finite, columns 0 to N-1 in [0, 1];
  `(n_link, 9·N)` (any empty array for `N = 0`)
- **`method`** — `"38.901"`, `"38.901_MF"` or `"quadriga"`; default: `"38.901"`
- **`rnd_freq`** — Per-frequency sets from [[ssf_draw]] (`set_id = k` in slice `k`), finite; required by
  `"38.901"` and `"38.901_MF"` if N > 0; `(n_link, N, n_freq)`; default: None
- **`lsp_anchor`** — LSPs at the anchor frequency from [[lsp_generate]], finite, spreads > 0; required by
  `"38.901_MF"` if N > 0; `(n_link, 1, 8)`; default: None
- **`eod_offset_deg`** — EoD offset in deg from [[lsp_generate]], positive = up, finite; `None` = 0;
  `(n_link, n_freq)`; default: None
- **`abs_toa`** — Absolute TOA offset in s from [[lsp_generate]], finite and >= 0; `None` = 0; `(n_link,)`;
  default: None
- **`gr_eps`** — Relative permittivity of the ground, complex (as [[lsp_generate]] returns it with
  `complex=True`), real part >= 1 and imaginary part >= 0; a real array is accepted (imaginary part 0); required if
  `GR_enabled > 0`; `(n_link, n_freq)`; default: None
- **`o2i`** — O2I flag per link, bool or numeric array (nonzero = O2I; mean ZOA 90 deg in `"38.901"`); `None` = no
  O2I link; `(n_link,)`; default: None
- **`n_cluster`** — Number of active NLOS clusters per link, non-negative integers in [N_lo, N] with
  `N_lo = min(max(1, NumClusters_min - 1), N)` (TR 38.901 Table 7.6.15-1 where the file states the range), given
  as int or float array; `None` = N; `(n_link,)`; default: None

## Outputs:
- **`delay`** — Delays in s relative to the LOS delay; `(n_link, n_slot, n_fd)`, `n_fd = n_freq` for
  `"38.901_MF"`, else 1
- **`power`** — Coefficient powers (linear) per frequency; `(n_link, n_slot, n_freq)`
- **`aod`**, **`aoa`** — Azimuth of departure and arrival in rad, in [-pi, pi); `(n_link, n_slot, n_fd)`
- **`eod`**, **`eoa`** — Elevation of departure and arrival in rad, in [-pi/2, pi/2]; `(n_link, n_slot, n_fd)`
- **`n_active`** — Number of NLOS clusters with power > 0; uint64; `(n_link,)`

## See also:
- [[ssf_draw]] (the random values `rnd` and `rnd_freq`)
- [[ssf_init_phases]] (the phases of the random generators)
- [[lsp_generate]] (the LSPs, the EoD offset, the absolute TOA and the ground permittivity)
- [[lsf_draw]] (the random values of the LSPs)
- [[calc_delay_spread]], [[calc_angular_spread]] (spreads of the output)
MD!*/

py::tuple ssf_clusters(py::handle scenpar,
                       const py::array_t<double> &tx_pos,
                       const py::array_t<double> &rx_pos,
                       const py::array_t<double> &lsp,
                       const py::array_t<double> &rnd,
                       const std::string &method,
                       py::handle rnd_freq,
                       py::handle lsp_anchor,
                       py::handle eod_offset_deg,
                       py::handle abs_toa,
                       py::handle gr_eps,
                       py::handle o2i,
                       py::handle n_cluster)
{
    // Read inputs; the entry 'SubpathMethod' of the scenario dict is not used
    std::string subpath_method;
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, subpath_method, "scenpar");
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const auto lsp_a = qd_python_numpy2arma_Cube<double>(lsp, true);
    const auto rnd_a = qd_python_numpy2arma_Mat<double>(rnd, true);
    const auto rnd_freq_a = qd_python_numpy2arma_Cube<double>(rnd_freq, true);
    const auto lsp_anchor_a = qd_python_numpy2arma_Cube<double>(lsp_anchor, true);
    const auto eod_offset_deg_a = qd_python_numpy2arma_Mat<double>(eod_offset_deg, true);
    const auto abs_toa_a = qd_python_numpy2arma_Col<double>(abs_toa, true);
    const auto o2i_in = qd_python_numpy2arma_Col<double>(o2i, true);
    const auto n_cluster_in = qd_python_numpy2arma_Col<double>(n_cluster, true);

    // Complex permittivity -> real and imaginary part; a real array is cast to complex (imaginary part 0)
    arma::mat gr_eps_re, gr_eps_im;
    if (!gr_eps.is_none())
    {
        arma::cube re, im;
        qd_python_copy2arma(py::cast<py::array_t<std::complex<double>>>(gr_eps), re, im);
        gr_eps_re = arma::mat(re.memptr(), re.n_rows, re.n_cols * re.n_slices);
        gr_eps_im = arma::mat(im.memptr(), im.n_rows, im.n_cols * im.n_slices);
    }

    // O2I flags: nonzero = true
    std::vector<bool> o2i_a(o2i_in.n_elem);
    for (arma::uword i = 0; i < o2i_in.n_elem; ++i)
        o2i_a[i] = o2i_in[i] != 0.0;

    // Cluster numbers are counts: read as double, checked, converted
    arma::uvec n_cluster_a(n_cluster_in.n_elem);
    for (arma::uword i = 0; i < n_cluster_in.n_elem; ++i)
    {
        const double v = n_cluster_in[i];
        if (!(v >= 0.0 && v < 9007199254740992.0 && v == std::floor(v)))
            throw std::invalid_argument("Entries in 'n_cluster' must be non-negative integers.");
        n_cluster_a[i] = (arma::uword)v;
    }

    // Wrap optional pointers; None selects the default
    const arma::cube *p_rnd_freq = rnd_freq.is_none() ? nullptr : &rnd_freq_a;
    const arma::cube *p_lsp_anchor = lsp_anchor.is_none() ? nullptr : &lsp_anchor_a;
    const arma::mat *p_eod_offset_deg = eod_offset_deg.is_none() ? nullptr : &eod_offset_deg_a;
    const arma::vec *p_abs_toa = abs_toa.is_none() ? nullptr : &abs_toa_a;
    const arma::mat *p_gr_eps_re = gr_eps.is_none() ? nullptr : &gr_eps_re;
    const arma::mat *p_gr_eps_im = gr_eps.is_none() ? nullptr : &gr_eps_im;
    const std::vector<bool> *p_o2i = o2i.is_none() ? nullptr : &o2i_a;
    const arma::uvec *p_n_cluster = n_cluster.is_none() ? nullptr : &n_cluster_a;

    // Call library function; the output sizes depend on 'scenpar' and 'method', so the outputs are copied afterwards
    arma::cube delay, power, aod, eod, aoa, eoa;
    arma::uvec n_active;
    quadriga_lib::ssf_clusters(scenpar_a, tx_pos_a, rx_pos_a, lsp_a, rnd_a, delay, power, aod, eod, aoa, eoa, method,
                               p_rnd_freq, p_lsp_anchor, p_eod_offset_deg, p_abs_toa, p_gr_eps_re, p_gr_eps_im, p_o2i,
                               p_n_cluster, &n_active);

    return py::make_tuple(qd_python_copy2numpy(&delay), qd_python_copy2numpy(&power), qd_python_copy2numpy(&aod),
                          qd_python_copy2numpy(&eod), qd_python_copy2numpy(&aoa), qd_python_copy2numpy(&eoa),
                          qd_python_copy2numpy(&n_active));
}

// pybind11 declaration:
// m.def("ssf_clusters", &ssf_clusters,
//       py::arg("scenpar"),
//       py::arg("tx_pos").none(false),
//       py::arg("rx_pos").none(false),
//       py::arg("lsp").none(false),
//       py::arg("rnd").none(false),
//       py::arg("method") = "38.901",
//       py::arg("rnd_freq") = py::none(),
//       py::arg("lsp_anchor") = py::none(),
//       py::arg("eod_offset_deg") = py::none(),
//       py::arg("abs_toa") = py::none(),
//       py::arg("gr_eps") = py::none(),
//       py::arg("o2i") = py::none(),
//       py::arg("n_cluster") = py::none());
