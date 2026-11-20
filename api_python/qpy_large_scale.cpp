// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"
#include "qd_helpers_gbsm.hpp"

#include <limits>

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# lsp_generate
Large-scale parameters of a set of links from normalized random values

- Generates the 8 LSPs of TR 38.901 clause 7.5 step 4 per link and frequency from the random values of [[gbsm.draw]]
  (`"lsf"`), and the absolute TOA offset, the ground permittivity and the EoD offset
- Mean `mu` and standard deviation `sigma` in the log domain (lg for DS and the angular spreads, dB for KF, SF and XPR),
  with the key prefix `P` (`DS`, `KF`, `SF`, `AS_D`, `AS_A`, `ES_D`, `ES_A`, `XPR`), `d = max(d2D, 1e-4)`, `h = max(hBS,
  1e-4)`, `a = atan(h/d)` and `f' = max(center_freq/1e9, LSP_fc_min)` in GHz:
  - `mu = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)`
  - `sigma = max(0, max(0, P_sigma + P_delta·lg(P_omega + f')) + P_kappa·lg(d) + P_tau·lg(h) + P_beta·lg(a))`
  - SF has the mean 0. DS adds `lg(DS_VS_a·DS_VS + DS_VS_b)` if `DS_VS > 0` (V/S of the InF hall, TR 38.901 Table 7.5-6
    Part-3). ESD adds `ES_D_mu_A·d/1000 + ES_D_mu_B·hUT + ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0)` and is
    limited below by `ES_D_mu_min` (TR 38.901 Tables 7.5-7 to 7.5-12)
  - The multiplying coefficients must be finite; a frequency term with `P_gamma` or `P_delta` not 0 requires `P_omega +
    f' > 0`
- `lsp = 10^(mu + sigma·x)` for DS and the angular spreads, `10^(0.1·(mu + sigma·x))` for KF, SF and XPR; `x` are the
  columns 0 to 7 of `rnd`, cross-correlated by a root of the LSP cross-correlation matrix of `scenpar`, the same for all
  frequencies (TR 38.901 clause 7.6.5). `rnd = numpy.zeros((n_link, 11))` gives the median LSPs
- Absolute TOA offset (TR 38.901 clause 7.6.9), active if `absTOA_mu > -30`: `10^(absTOA_mu + absTOA_sigma·x)` with
  column 8 of `rnd`, bounded by `2·absTOA_Lmax/c` if `absTOA_Lmax > 0`
- The InF and InH NLOS files set `absTOA_Lmax`, and the InF files `DS_VS`, for the hall or office of their calibration
  scenario; for another hall set these keys in `scenpar`
- Ground permittivity: QuaDRiGa's fits for dry, medium and wet ground, interpolated linearly over column 9 of `rnd` (0
  dry, 0.5 medium, 1 wet) with `center_freq` not limited by `LSP_fc_min`; they are not the materials of TR 38.901 Table
  7.6.8-1
- EoD offset, the negative of the ZOD offset of TR 38.901 Tables 7.5-7 to 7.5-12: `EoD_offset_e + EoD_offset_e_gamma·lg
  f'`, plus `10^((EoD_offset_a + EoD_offset_a_gamma·lg f')·lg(max(EoD_offset_b, d)) + EoD_offset_c +
  EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT·hUT)` if `EoD_offset_b > 0`, plus `(atan((EoD_offset_hcl - h)/d) -
  atan((hUT - h)/d))·180/pi` if `hUT < EoD_offset_hcl`

## Usage:
```
lsp, abs_toa, gr_eps, eod_offset_deg, lsp_mu, lsp_sigma = quadriga_lib.gbsm.lsp_generate( scenpar, \
    center_freq, tx_pos, rx_pos, rnd, *, sf_sigma_dB, cholesky, clip_spreads )
```

## Inputs:
- **`scenpar`** — Scenario parameters; the LSP cross-correlation matrix must be positive definite
- **`center_freq`** — Carrier frequencies in Hz, finite and > 0; `(n_freq,)`
- **`tx_pos`** — Tx positions in m; `(3,)` or `(3, n_link)`
- **`rx_pos`** — Rx positions in m; `(3, n_link)`
- **`rnd`** — Random values from [[gbsm.draw]] (`"lsf"`), finite; column 9 in [0, 1] (the permittivity is always
  computed); column 10 is not used; `(n_link, 11)`
- **`sf_sigma_dB`** — SF standard deviation in dB (e.g. from [[path_loss]]), finite and >= 0; replaces the value from
  `scenpar`, also in `lsp_sigma`; `None` or empty = the scenario value; `(n_link, n_freq)`; default: None
- **`cholesky`** — `True`: Cholesky root in the order of TR 38.901 clause 7.5 step 4 (SF, KF, DS, ASD, ASA, ESD, ESA;
  XPR last); `False`: symmetric square root (legacy QuaDRiGa); default: True
- **`clip_spreads`** — `True`: limit ASD and ASA to 104 deg, ESD and ESA to 52 deg (TR 38.901 clause 7.5 step 4);
  `lsp_mu` and `lsp_sigma` are not affected; default: True

## Outputs:
- **`lsp`** — Large-scale parameters; slices: 0 DS in s; 1 KF, 2 SF and 7 XPR linear; 3 ASD, 4 ASA, 5 ESD, 6 ESA in
  deg; `(n_link, n_freq, 8)`
- **`abs_toa`** — Absolute TOA offset in s, 0 if not active; `(n_link,)`
- **`gr_eps`** — Relative permittivity of the ground, imaginary part >= 0; the complex value of TR 38.901 eq.
  (7.6-40) is `gr_eps.conj()`; complex128; `(n_link, n_freq)`
- **`eod_offset_deg`** — EoD offset in deg, positive = up; `(n_link, n_freq)`
- **`lsp_mu`** — Mean `mu` in the log domain, slices as `lsp`: lg of s for DS, lg of deg for the angular spreads, dB
  for KF, SF and XPR; required by [[ssf_rays]]; `(n_link, n_freq, 8)`
- **`lsp_sigma`** — Standard deviation `sigma` in the log domain of `lsp_mu`, >= 0; required by [[ssf_rays]];
  `(n_link, n_freq, 8)`

## See also:
- [[gbsm.draw]] (the random values `rnd`)
- [[path_loss]] (the SF standard deviation `sf_sigma_dB`)
- [[ssf_clusters]] (the clusters from the LSPs)
- [[ssf_rays]] (takes `lsp`, `lsp_mu` and `lsp_sigma`)
MD!*/

py::tuple lsp_generate(py::handle scenpar,
                       const py::array_t<double> &center_freq,
                       const py::array_t<double> &tx_pos,
                       const py::array_t<double> &rx_pos,
                       const py::array_t<double> &rnd,
                       py::handle sf_sigma_dB,
                       py::handle cholesky,
                       py::handle clip_spreads)
{
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, "scenpar");
    const bool cholesky_b = qd_python_flag(cholesky, true, "cholesky");
    const bool clip_spreads_b = qd_python_flag(clip_spreads, true, "clip_spreads");
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const auto rnd_a = qd_python_numpy2arma_Mat<double>(rnd, true);
    const auto sf_sigma_a = qd_python_numpy2arma_Mat<double>(sf_sigma_dB, true);

    const arma::uword n_link = qd_gbsm_n_link(rx_pos_a), n_freq = center_freq_a.n_elem;
    arma::cube lsp, lsp_mu, lsp_sigma;
    arma::vec abs_toa;
    arma::mat gr_eps_re, gr_eps_im, eod_offset_deg;

    auto lsp_py = qd_python_init_output(n_link, n_freq, (arma::uword)8, &lsp);
    auto abs_toa_py = qd_python_init_output(n_link, &abs_toa);
    auto eod_offset_deg_py = qd_python_init_output(n_link, n_freq, &eod_offset_deg);
    auto lsp_mu_py = qd_python_init_output(n_link, n_freq, (arma::uword)8, &lsp_mu);
    auto lsp_sigma_py = qd_python_init_output(n_link, n_freq, (arma::uword)8, &lsp_sigma);

    quadriga_lib::lsp_generate(scenpar_a, center_freq_a, tx_pos_a, rx_pos_a, rnd_a, lsp, sf_sigma_a, &abs_toa, &gr_eps_re,
                               &gr_eps_im, &eod_offset_deg, &lsp_mu, &lsp_sigma, cholesky_b, clip_spreads_b);

    // The permittivity is merged into one complex array (small: one value per link and frequency)
    auto gr_eps_py = qd_python_copy2numpy<double, std::complex<double>>(&gr_eps_re, &gr_eps_im);
    return py::make_tuple(lsp_py, abs_toa_py, gr_eps_py, eod_offset_deg_py, lsp_mu_py, lsp_sigma_py);
}

/*!MD
# path_loss
Path loss and shadow-fading standard deviation of a set of links

- Evaluates one of the 10 parametric path-loss models of the QuaDRiGa conf files for every link and carrier frequency;
  the TR 38.901 v19.5.0 conf files of the library reproduce Table 7.4.1-1 with them. The O2I penetration loss is not
  included (see [[o2i_loss]]). The models are evaluated as written for any positive frequency, also outside their
  validity range
- Notation: `f = center_freq/1e9` in GHz (not limited by `LSP_fc_min`); `d3`, `d2` the 3D and 2D distance, at least 0.1
  m; `hBS`, `hUT` the z coordinates of Tx and Rx; `lg = log10`; `PL1(x) = A1·lg(x) + B + C·lg(f) + D·x`; `sf` the
  shadow-fading std. Keys of `plpar` in parentheses: the required ones, after the semicolon the optional ones (`X = v`:
  default)
- Models (the key `model` of `plpar`):
  - `constant` (A): `A`
  - `logdist` (A, B, C; SF): `A·lg d3 + B + C·lg f`; `sf = SF`
  - `logdist_simple` (A, B): `A·lg d3 + B`
  - `winner_los` (A1, A2 different, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, sig1, sig2): `hb = max(hBS, 1.5)`, `hm =
    max(hUT, 1.5)`, `Gj = Bj + Cj·lg f + Dj·lg hb + Ej·lg hm`, breakpoint `bp = 10^((G2 - G1)/(A1 - A2))`; `Aj·lg d2 +
    Gj + Fj·hm` and `sf = sigj` with `j = 1` for `d2 <= bp`, else `j = 2`
  - `winner_nlos` (A1..A3, Ah1..Ah3, B1..B3, C1..C3, D1..D3, E1..E3, F1..F3): `hb`, `hm` as `winner_los`; `(Aj + Ahj·lg
    hb)·lg d2 + Bj + Cj·lg f + Dj·lg hb + Ej·lg hm + Fj·hm` with `j = 1` for `f < 1.5`, 2 for `f < 2`, else 3
  - `winner_pathloss` (A, B, C, X; SF): `A·lg d3 + B + C·lg(f/5) + X`; `sf = SF`
  - `dual_slope` (A1 or A, B, E; A2 = A1, C = 0, D = 0, hE = 0, hE_rand = 0, sig1, sig2): `hb = max(hBS, 1.1·hE, 0.1)`,
    `hm = max(hUT, 1.1·hE, 0.1)`, `dBP = E·f·(hb - hE)·(hm - hE)`, `dBP3 = sqrt(dBP² + (hb - hm)²)`; `PL1(d3)`, `sf =
    sig1` for `d2 <= dBP`, else `PL1(dBP3) + A2·lg(d3/dBP3)`, `sf = sig2`
  - `tripple_slope`, spelled as in QuaDRiGa (A1 or A, B, E1 or E; A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1, hE1 = 0,
    hE2 = 0, sig1..sig3): the `dual_slope` rule with the breakpoints `dBP1` of `E1, hE1` (at most `dBP2`) and `dBP2`
    of `E2, hE2`, the slopes `A1`, `A2`, `A3` and `sf = sig1`, `sig2`, `sig3`
  - `nlos` (An, Bn; Cn, Dn, D1n, D2n, D3n, En, E1n, E2n, E3n, Fn, G1n = 0, G2n = 1, LOSmax = 1, Ax, Bx, Cx, sig): `hb =
    max(hBS, 0.1)`, `hm = max(hUT, 0.1)`; `An·lg d3 + Bn + Cn·lg f + Dn·lg hb + D1n·lg(hb)/hb + D2n·lg(hb)/hb² +
    D3n·hb + En·lg hm + E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm + Fn·lg(hb)·lg(d3) + G1n·(lg(G2n·hm))²`; with
    `LOSmax != 0` (which requires the `dual_slope` keys) at least the `dual_slope` loss; with `Ax`, `Bx` or `Cx` at
    least `Ax·lg d3 + Bx + Cx·lg f` (e.g. the InF-SL term of InF-DL); `sf = sig`
  - `satellite` (A = 20, B = 32.45, C = 20, D = 0, usePLa = 1): `sa = min(hBS/d3, 1)`; `+Inf` for `sa <= 0`, else `A·lg
    d3 + B + C·lg f + D·lg(asin(sa))`, plus the zenith attenuation by atmospheric gases over `sa` (1 to 100 GHz,
    constant beyond) if `usePLa != 0`
  - no model (`plpar` is `None`, `{}` or has no entry `model`): 0
- Where the model gives no `sf` (an `SF` or `sig` key not set, or a model without one): the SF std of [[lsp_generate]]
  at the geometric distance, so that passing `sf_sigma_dB` to [[lsp_generate]] is neutral
- `hE_rand != 0` with `rnd_he` given: the random environment height of TR 38.901 Table 7.4.1-1 note 1, drawn from
  `rnd_he`, which makes it spatially consistent at the decorrelation distance `SF_lambda` (the TR draws it independently
  per link); the UT height is then clamped at `hE + 0.1` instead of `1.1·hE`. Along a track the loss beyond the
  breakpoint steps where hE changes; evaluate `rnd_he` once per track segment to avoid this
- Distances: the geometric distances of TR 38.901 eq. (7.4-1). With `d2d_in` (legacy QuaDRiGa), links with `d2d_in > 0`
  use the outdoor part of the distances instead, at least 10 m
- A required key that is missing is an error, naming `A1` for "A1 or A" and `E1` for "E1 or E"; every key that the model
  reads must have one finite value; the breakpoint factors `E`, `E1`, `E2` and `G2n` must be > 0

## Usage:
```
pl_dB, sf_sigma_dB = quadriga_lib.gbsm.path_loss( plpar, scenpar, center_freq, tx_pos, rx_pos, *, \
    d2d_in, rnd_he )
```

## Inputs:
- **`plpar`** — Path-loss parameters; dict as returned by [[conf_file_read]]: the entry `model` (str) and one real
  scalar or 1D array per further parameter, named without "PL_"; keys not read are ignored; `None` or `{}`: no model
  (loss 0)
- **`scenpar`** — Scenario parameters; the SF std terms and `LSP_fc_min` are used
- **`center_freq`** — Carrier frequencies in Hz, finite and > 0; `(n_freq,)`
- **`tx_pos`** — Tx positions in m; `(3,)` or `(3, n_link)`
- **`rx_pos`** — Rx positions in m; `(3, n_link)`
- **`d2d_in`** — Indoor 2D distance in m of the UT of each link, legacy QuaDRiGa only (leave it `None` for TR 38.901,
  which uses the geometric distance); finite and >= 0; 0 = outdoor link; `(n_link,)`; default: None
- **`rnd_he`** — Uniform(0,1) value per link for the random environment height, column 10 of [[gbsm.draw]] (`"lsf"`);
  `None` or empty = fixed `hE`; `(n_link,)`; default: None

## Outputs:
- **`pl_dB`** — Path loss in dB; `(n_link, n_freq)`
- **`sf_sigma_dB`** — Shadow-fading standard deviation in dB; `(n_link, n_freq)`

## See also:
- [[lsp_generate]] (takes `sf_sigma_dB`)
- [[o2i_loss]] (penetration loss, added by the caller)
MD!*/

py::tuple path_loss(py::handle plpar,
                    py::handle scenpar,
                    const py::array_t<double> &center_freq,
                    const py::array_t<double> &tx_pos,
                    const py::array_t<double> &rx_pos,
                    py::handle d2d_in,
                    py::handle rnd_he)
{
    // Read inputs; the entry 'model' of 'plpar' is the model
    std::string pl_model;
    std::vector<std::string> pl_order;
    const auto plpar_a = qd_python_dict2plpar(plpar, pl_model, pl_order, "plpar");
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, "scenpar");
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const auto d2d_in_a = qd_python_numpy2arma_Col<double>(d2d_in, true);
    const auto rnd_he_a = qd_python_numpy2arma_Col<double>(rnd_he, true);

    // Output allocation of the optional output
    arma::mat sf_sigma_dB;
    auto sf_sigma_py = qd_python_init_output(qd_gbsm_n_link(rx_pos_a), center_freq_a.n_elem, &sf_sigma_dB);

    const arma::mat pl_dB = quadriga_lib::path_loss(pl_model, plpar_a, scenpar_a, center_freq_a, tx_pos_a, rx_pos_a,
                                                    &sf_sigma_dB, d2d_in_a, rnd_he_a);

    return py::make_tuple(qd_python_copy2numpy(&pl_dB), sf_sigma_py);
}

/*!MD
# los_probability
LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models

- Evaluates the probability that a link is in line-of-sight (LOS) from its 2D distance and the BS and UT heights, and,
  when `rnd_g` and `center_freq` are given, the soft LOS state of TR 38.901 eq. (7.6-18); the LOS state itself (a draw
  against `p_los`) and its spatial correlation are the caller's
- `38.901_*`: TR 38.901 v19.5.0 Table 7.4.2-1 (`InH_Mixed`, `InH_Open`: mixed and open office). Where the table leaves
  cases open: all results are limited to [0, 1]; UMa uses `C' = 1` for `hUT > 23` m; InF-SH and InF-DH give 1 for
  `hUT >= h_clutter` and the rule of InF-SL and InF-DL for `hBS <= hUT`; SMa gives 1 for `hBS <= hUT` and uses
  `max(h_x - hUT, 0)` in all three terms (the TR only in the residential term; equal for `hUT < 15` m)
- `38.811_*`: TR 38.811 Table 6.6.1-1 over the elevation `atan2(hBS, d2d_out)`, linearly interpolated; `Suburban` and
  `Rural` share one table
- `QuaDRiGa_Industrial`: `min(1, exp(-(d2d_out + 20)/80)·exp(min(hBS, 8)/5 - 1))`; `LOS`: 1; `NLOS`: 0
- The `LOS_state_lambda` of the InF conf files is `d_clutter/2` of the TR defaults; with another `d_clutter` draw the
  LOS state at a matching decorrelation distance

## Usage:
```
p_los, los_soft = quadriga_lib.gbsm.los_probability( model, d2d_out, h_bs, h_ut, *, d_clutter, \
    r_clutter, h_clutter, rnd_g, center_freq )
```

## Inputs:
- **`model`** — Model name: `38.901_RMa`, `38.901_UMi`, `38.901_UMa`, `38.901_InH_Mixed`, `38.901_InH_Open`,
  `38.901_InF_SL`, `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_DH`, `38.901_InF_HH`, `38.901_SMa`,
  `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`, `38.811_Rural`, `QuaDRiGa_Industrial`, `LOS`, `NLOS`
- **`d2d_out`** — 2D outdoor distance in m, finite and >= 0: `d2D - d2d_in` for O2I UTs, clamped at 0 by the caller; the
  2D distance in indoor scenarios; `(n_link,)`
- **`h_bs`** — BS height in m, finite; float or `(n_link,)`
- **`h_ut`** — UT height in m, finite; float or `(n_link,)`
- **`d_clutter`** — Clutter size in m of the InF and SMa models; `None`, nan or <= 0: TR default (InF-SL and InF-SH 10
  m, InF-DL and InF-DH 2 m, SMa 30 m); default: None
- **`r_clutter`** — Clutter density of the InF models (default 0.2 for SL and SH, 0.6 for DL and DH), vegetation density
  of SMa (default 0; 0.1 sparse, 0.2 dense); in [0, 1); `None` or nan = default; default: None
- **`h_clutter`** — Clutter height in m of InF-SH (default 2 m) and InF-DH (6 m), >= 0; `None` or nan = default. The
  other models ignore the clutter arguments, which are validated for every model; default: None
- **`rnd_g`** — Normal(0,1) value per link, G of eq. (7.6-18); `(n_link,)`; default: None
- **`center_freq`** — Carrier frequencies in Hz, finite and > 0; `(n_freq,)`; default: None

## Outputs:
- **`p_los`** — LOS probability in [0, 1]; `(n_link,)`
- **`los_soft`** — Soft LOS state in [0, 1]; `> 0.5` exactly when `rnd_g > -F(p_los)` of eq. (7.6-18), which has the
  probability `p_los`; `(n_link, n_freq)`; `None` if `rnd_g` or `center_freq` is `None`

## See also:
- [[o2i_loss]] (indoor distance for `d2d_out`)
- [[sos.rand]], [[sos.randn]] (spatially consistent LOS-state draw and `rnd_g`)
MD!*/

py::tuple los_probability(const std::string &model,
                          const py::array_t<double> &d2d_out,
                          const py::array_t<double> &h_bs,
                          const py::array_t<double> &h_ut,
                          py::handle d_clutter,
                          py::handle r_clutter,
                          py::handle h_clutter,
                          py::handle rnd_g,
                          py::handle center_freq)
{
    // Clutter arguments: None = NaN (TR default of the model), else a real scalar
    auto clutter = [](py::handle value, const std::string &name)
    {
        if (value.is_none())
            return std::numeric_limits<double>::quiet_NaN();
        const std::string error_msg = "Input '" + name + "' must be a real scalar or None.";
        const auto a = qd_python_real_array(value, error_msg);
        if (a.size() != 1)
            throw std::invalid_argument(error_msg);
        return *a.data();
    };
    const double d_clutter_d = clutter(d_clutter, "d_clutter"), r_clutter_d = clutter(r_clutter, "r_clutter"),
                 h_clutter_d = clutter(h_clutter, "h_clutter");

    const auto d2d_out_a = qd_python_numpy2arma_Col<double>(d2d_out, true);
    const auto h_bs_a = qd_python_numpy2arma_Col<double>(h_bs, true);
    const auto h_ut_a = qd_python_numpy2arma_Col<double>(h_ut, true);
    const auto rnd_g_a = qd_python_numpy2arma_Col<double>(rnd_g, true);
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);

    // Output allocation; the soft LOS state needs both 'rnd_g' and 'center_freq', else it is returned as None
    const bool soft = !rnd_g.is_none() && !center_freq.is_none();
    arma::mat los_soft;
    py::object los_soft_py = py::none();
    if (soft)
        los_soft_py = qd_python_init_output(d2d_out_a.n_elem, center_freq_a.n_elem, &los_soft);

    arma::mat *p_los_soft = soft ? &los_soft : nullptr;

    const arma::vec p_los = quadriga_lib::los_probability(model, d2d_out_a, h_bs_a, h_ut_a, d_clutter_d, r_clutter_d,
                                                          h_clutter_d, rnd_g_a, center_freq_a, p_los_soft);

    return py::make_tuple(qd_python_copy2numpy(&p_los), los_soft_py);
}

/*!MD
# o2i_loss
O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3

- Draws for each UT one of the given penetration-loss models and evaluates its loss at each carrier frequency; the
  caller adds `loss_dB` to the path loss of the UT
- Models of TR 38.901 v19.5.0: `low`, `high`, `low_A` (Table 7.4.3-2 with the materials of Table 7.4.3-1), `legacy_6GHz`
  (Table 7.4.3-3, single-frequency UMa and UMi below 6 GHz), `car` and `car_metal` (clause 7.4.3.2, metallized windows)
- Model of UT `u`: the first model `j` with `rnd(u, 0)` below the sum of the weights up to `j`, else the last model with
  a weight > 0; a model with weight 0 is never chosen
- Indoor distance: the minimum of columns 1 and 2 of `rnd` times `d2d_in_max` (`low`, `high`, `low_A`, clause 7.4.3.1),
  column 1 times `d2d_in_max` (`legacy_6GHz`), 0 for the car models
- Draw `rnd` with [[sos.rand]] and [[sos.randn]] at the correlation distances of TR 38.901 clause 7.6.3.3: 50 m for
  column 0, 25 m for columns 1 and 2, 10 m for column 3
- TR applicability (Table 7.4.3-2): RMa `low` only, InF `high` only, UMa and UMi `low` and `high`, SMa `low`, `high` and
  `low_A`. SMa has 90 % residential and 10 % commercial buildings (Table 7.2-5) with different maximum indoor distances:
  call `o2i_loss` once per building type

## Usage:
```
loss_dB, d2d_in, model_ind = quadriga_lib.gbsm.o2i_loss( center_freq, rnd, models, weights, *, \
    d2d_in_max )
```

## Inputs:
- **`center_freq`** — Carrier frequencies in Hz, finite and > 0; `(n_freq,)`
- **`rnd`** — Random values per UT: columns 0 to 2 Uniform in [0, 1] (model choice, two indoor distances), column 3
  Normal(0,1), finite (deviation `sigma_P`, car loss); `(n_ut, 4)`, `n_ut = 0` allowed
- **`models`** — Models to mix, names as listed above; list of str or one str; `n_model` entries
- **`weights`** — Probability of each model, finite and >= 0, sum 1 within 1e-6; `(n_model,)`
- **`d2d_in_max`** — Maximum indoor 2D distance in m, finite and >= 0 (`scenpar["O2I_d2D_in_max"]` of the O2I conf
  files; TR: 25 m for UMa, UMi and SMa commercial buildings, 10 m for RMa and SMa residential buildings); default: 25.0

## Outputs:
- **`loss_dB`** — Penetration loss in dB; `(n_ut, n_freq)`
- **`d2d_in`** — Indoor 2D distance in m, 0 for the car models; `(n_ut,)`
- **`model_ind`** — Chosen model per UT, 0-based index into `models`; uint64; `(n_ut,)`

## See also:
- [[los_probability]] (uses the outdoor distance `d2D - d2d_in`)
- [[path_loss]] (basic path loss, legacy outdoor distance from `d2d_in`)
MD!*/

py::tuple o2i_loss(const py::array_t<double> &center_freq,
                   const py::array_t<double> &rnd,
                   py::handle models,
                   const py::array_t<double> &weights,
                   double d2d_in_max)
{
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);
    const auto rnd_a = qd_python_numpy2arma_Mat<double>(rnd, true);
    const auto models_a = qd_python_list2vector_Strings(models);
    const auto weights_a = qd_python_numpy2arma_Col<double>(weights, true);

    arma::vec d2d_in;
    arma::uvec model_ind;
    auto d2d_in_py = qd_python_init_output(rnd_a.n_rows, &d2d_in);
    auto model_ind_py = qd_python_init_output(rnd_a.n_rows, &model_ind);

    const arma::mat loss_dB = quadriga_lib::o2i_loss(center_freq_a, rnd_a, models_a, weights_a, d2d_in_max, &d2d_in,
                                                     &model_ind);

    return py::make_tuple(qd_python_copy2numpy(&loss_dB), d2d_in_py, model_ind_py);
}
