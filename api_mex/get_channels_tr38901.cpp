// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex_types.hpp"

/*!SECTION
Channel generators
SECTION!*/

/*!MD
# GET_CHANNELS_TR38901
Generate the channels of a TR 38.901 scenario in one call

- Channels between the UTs of a drop and the BSs of a 3GPP TR 38.901 scenario (UMa, UMi, RMa, SMa, InH, InF), with the
  given BS and UT antenna arrays; by default one channel per UT and frequency, to its serving BS sector
- The scenario sets the BS layout and the drop of the UTs as in TR 38.901 (table in Notes); optional inputs change
  single values, the time evolution and which links are returned
- Built on the layout functions ([[generate_layout]], [[init_link_states]], [[attach_terminals]], [[generate_rays]],
  [[get_channels_from_rays]]); `layout` returns the simulated layout, to continue there

## Usage:
```
[ chan, gain_dB, layout, index ] = quadriga_lib.get_channels_tr38901( bs_array, ut_array, scenario, ...
    center_freq, n_ut, seed, spacing, grid, area, bs, ut, indoor, time_grid, los, output, attach, links, ...
    link_gain, uplink, o2i_low, config_dirs );
```

## Inputs:
- **`bs_array`** — BS antenna array, struct, fields in [[generate_antenna]]; every sector uses it, turned to the sector heading
- **`ut_array`** — UT antenna array, struct, fields in [[generate_antenna]]
- **`scenario`** — `'UMa'`, `'UMi'`, `'RMa'`, `'SMa'`, `'InH_Open'`, `'InH_Mixed'`, `'InF_SL'`, `'InF_DL'`,
  `'InF_SH'`, `'InF_DH'` or `'InF_HH'`; also with the prefix `'38.901_'`
- **`center_freq`** — Carrier frequencies in Hz, ascending; one set of channels per frequency; default: 3.5e9;
  `[n_freq]`
- **`n_ut`** — UTs per cell (per BS sector), >= 1; default: 1
- **`seed`** — Seed of the random numbers: the same seed gives the same drop and channels; -1 = a random seed,
  which `layout.seed` returns; default: -1

Optional changes to the scenario layout (defaults in Notes). Give only the values to change; NaN keeps a default.
[[generate_layout]] describes all values:

- **`spacing`** — Inter-site distance in m, e.g. `200`; indoor: BS spacing in x and y; default: `[]`
- **`grid`** — Rings of sites around the centre site, e.g. `0` = one site, `2` = 19 sites; the
  number of sectors as third value, e.g. `[0, NaN, 1]`; indoor: BSs in x and y; default: `[]`
- **`area`** — Hall size in m, length, width, height (InH, InF); default: `[]`
- **`bs`** — BS height in m, e.g. `30`; then the mechanical downtilt and an offset of the sector
  headings in degrees; default: `[]`
- **`ut`** — UT height and minimum distance to the BS in m, speed in m/s, e.g. `[NaN, NaN, 30 / 3.6]` =
  all UTs at 30 km/h; then the orientation (bank, tilt, heading) in degrees; default: `[]`
- **`indoor`** — Share of the UTs in buildings, e.g. `0.5` (UMa, UMi, RMa, SMa); then the range of floors
  and the floor height; default: `[]`

Simulation settings:

- **`time_grid`** — `[T, dt]` in s: snapshots at 0, dt, 2 dt, … up to T, while the UTs move; `dt`
  defaults to 1 ms, e.g. `0.1` = 101 snapshots over 100 ms; `[]` = one snapshot (static channel); default: `[]`
- **`los`** — `'auto'`: LOS or NLOS as drawn from the LOS probability of the scenario; `'los'` or `'nlos'`: that state for
  all links, also of the UTs in buildings and to the other sectors, so it can change the serving sector (`'InF_HH'` is
  always LOS); default: `'auto'`
- **`output`** — `'taps'`: the rays of a cluster combined into taps; `'rays'`: every ray as a path, with angles
  and path gains; default: `'taps'`
- **`attach`** — How the serving sector is chosen: `'large_scale'` (coupling gain) or `'rsrp'` (Notes); default: `'large_scale'`
- **`links`** — Which links are returned: `'serving'`, the serving sector of each UT; `'site'`, all sectors of its
  serving site; `'all'`, all BS sectors, e.g. for interference studies (Outputs); default: `'serving'`
- **`link_gain`** — false: without path loss, building loss and shadow fading, so that the power is that of the
  antennas and the small-scale fading; `gain_dB` and the serving sector still include them; default: true
- **`uplink`** — true: the uplink, with the UT as Tx (Notes); default: false
- **`o2i_low`** — Share of buildings with the low-loss model of the building penetration loss, in [0, 1];
  NaN = the TR 38.901 value (50 % in UMa and UMi); not for InH, InF; default: NaN
- **`config_dirs`** — Folders with own scenario files (`.conf`), searched before the shipped ones; default: `{}`

## Outputs:
- **`chan`** — Channels as struct array, fields in [[hdf5_read_channel]]; what taps and rays hold: [[gbsm_single_link_channel]]
  - `'serving'`: one channel per UT and frequency, all UTs at the first frequency, then at the next: `chan(i_ut + (i_freq - 1) * n_ut_total)`, 1-based
  - `'site'`, `'all'`: one channel per BS sector, UT and frequency; the sectors of the first UT one after the other,
    then those of the next UT, then the next frequency; `index` names the BS sector, UT and frequency of each channel
  - UTs in drop order: the first `n_ut` were dropped in cell 1 (site BS1, sector 0), the next `n_ut` in cell 2, …;
    the serving sector is the one with the highest gain, often not the cell of the drop. The name says it too, e.g.
    `'BS5:0_UT11'` = site BS5, sector 0, UT 11
  - Coefficients and delays `[n_ut_port, n_bs_port, n_tap, n_snap]` (downlink; `uplink` swaps the first two)
- **`gain_dB`** — Coupling gain or RSRP in dB of each UT to each BS sector, e.g. for the SIR; `[n_ut_total, n_bs_node]`
- **`layout`** — The simulated layout: BS and UT positions, link states, pairing; fields in [[layout_csv_read]]
- **`index`** — BS sector (node), UT and frequency of each channel, 1-based; uint64; `[3, n_channel]`

## Example:
```
bs = quadriga_lib.generate_antenna( '3gpp', 10, 3.5e9, [], [4, 4, 1, 1] );
ut = quadriga_lib.generate_antenna( 'xpol' );
chan = quadriga_lib.get_channels_tr38901( bs, ut, 'UMa', 3.5e9, 10, 1 );                          % 19 sites
chan = quadriga_lib.get_channels_tr38901( bs, ut, 'UMi', 3.5e9, 10, 1, [], 0, [], [], [], [], 0.1 );   % one site, 100 ms
```

## Notes:
- **Scenario defaults** (TR 38.901 Tables 7.2-1 to 7.2-5 and 7.8-7); the UTs are at 1.5 m (InH 1 m, InF_HH 3 m),
  in buildings on their floor:

  | Scenario | BS layout | BS height | UTs indoors | UT speed |
  | --- | --- | --- | --- | --- |
  | UMa | 19 sites × 3 sectors, ISD 500 m, wrap-around | 25 m | 80 % | 3 km/h |
  | UMi | 19 sites × 3 sectors, ISD 200 m, wrap-around | 10 m | 80 % | 3 km/h |
  | RMa | 19 sites × 3 sectors, ISD 1732 m, wrap-around | 35 m | 50 %, the others in cars | 3 km/h, cars 40 km/h |
  | SMa | 19 sites × 3 sectors, ISD 1299 m, wrap-around | 35 m | 80 %, the others in cars | 3 km/h, cars 40 km/h |
  | InH_Open, InH_Mixed | Hall 120 × 50 m, 12 BSs 20 m apart | 3 m | all | 3 km/h |
  | InF_SL, InF_DH | Hall 120 × 60 × 10 m, 18 BSs 20 m apart | 1.5 m (SL), 8 m (DH) | all | 3 km/h |
  | InF_DL, InF_SH, InF_HH | Hall 300 × 150 × 10 m, 18 BSs 50 m apart | 1.5 m (DL), 8 m (SH, HH) | all | 3 km/h |

  `n_ut` UTs are dropped in each cell (`n_ut_total` = `n_ut` × the number of BS sectors); with the wrap-around, the
  outer cells see the interference of a large network, also with one site
- **Serving sector**: the BS sector with the highest coupling gain (`'large_scale'`, TR 38.901 Table 7.8-1) or RSRP
  (`'rsrp'`, Table 7.8-2) at the first frequency, see [[attach_terminals]]; `'rsrp'` and `links` = `'all'` generate
  the rays of all links, the others only those to the serving sites
- **Time evolution**: the UTs move from their drop position along their direction at their speed; delays and angles of
  the clusters stay, the phases follow the motion (mobility `'phase'` of [[get_channels_from_rays]])
- **Several frequencies**: the clusters come from the same random values at all frequencies, scaled by the spreads of
  each frequency (method `'38.901_MF'`, TR 38.901 clause 7.6.5)
- **Uplink**: the transposed downlink (reciprocal channel, the same Doppler shift), with the two ends swapped in all
  members and the name; `index` and `gain_dB` keep the downlink meaning. This is the uplink of a TDD system; for FDD,
  add the uplink frequency to `center_freq`
- **Memory**: about 90 kB per snapshot for the 57 UTs of the default UMa drop with 2 × 2 ports (16 to 25 taps);
  `links` = `'all'` multiplies it by the number of BS sectors; `'site'` by the sectors per site (3)
- The C++ outputs come after `chan` in their C++ order; the wrapper adds the folder of the shipped conf files to
  `config_dirs`

## See also:
- [[hdf5_read_channel]] (the fields of the channels)
- [[freq_response]] (frequency response of the channels at given subcarriers)
- [[generate_layout]] (the scenario layouts and their inputs in full)
- [[get_channels_layout]] (channels of any layout)
- [[get_channels_ieee_indoor]] (the IEEE indoor models)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3 || nrhs > 21)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const auto bs_array = qd_mex_struct2arrayant(prhs[0], false, false, "bs_array");
    const auto ut_array = qd_mex_struct2arrayant(prhs[1], false, false, "ut_array");
    const std::string scenario = qd_mex_get_string(prhs[2], "", "scenario");
    const arma::vec center_freq = (nrhs < 4 || mxIsEmpty(prhs[3])) ? arma::vec{3.5e9} : qd_mex_get_Col<double>(prhs[3], false, "center_freq");
    const arma::uword n_ut = (nrhs < 5) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[4], "n_ut", 1);
    const arma::sword seed = (nrhs < 6) ? -1 : (arma::sword)qd_mex_get_seed(prhs[5], "seed");
    const arma::vec spacing = qd_mex_layout_vec(nrhs, prhs, 6, "spacing");
    const arma::vec grid = qd_mex_layout_vec(nrhs, prhs, 7, "grid");
    const arma::vec area = qd_mex_layout_vec(nrhs, prhs, 8, "area");
    const arma::vec bs = qd_mex_layout_vec(nrhs, prhs, 9, "bs");
    const arma::vec ut = qd_mex_layout_vec(nrhs, prhs, 10, "ut");
    const arma::vec indoor = qd_mex_layout_vec(nrhs, prhs, 11, "indoor");
    const arma::vec time_grid = qd_mex_layout_vec(nrhs, prhs, 12, "time_grid");
    const std::string los = (nrhs < 14) ? std::string("auto") : qd_mex_get_string(prhs[13], "auto", "los");
    const std::string output = (nrhs < 15) ? std::string("taps") : qd_mex_get_string(prhs[14], "taps", "output");
    const std::string attach = (nrhs < 16) ? std::string("large_scale") : qd_mex_get_string(prhs[15], "large_scale", "attach");
    const std::string links = (nrhs < 17) ? std::string("serving") : qd_mex_get_string(prhs[16], "serving", "links");
    const bool link_gain = (nrhs < 18) ? true : qd_mex_get_scalar<bool>(prhs[17], "link_gain", true);
    const bool uplink = (nrhs < 19) ? false : qd_mex_get_scalar<bool>(prhs[18], "uplink", false);
    const double o2i_low = (nrhs < 20) ? NAN : qd_mex_get_scalar<double>(prhs[19], "o2i_low", NAN);
    const std::vector<std::string> config_dirs = qd_mex_layout_config_dirs((nrhs < 21) ? nullptr : prhs[20]);

    std::vector<quadriga_lib::channel<double>> chan;
    arma::mat gain_dB;
    quadriga_lib::layout layout;
    arma::umat index;
    CALL_QD(chan = quadriga_lib::get_channels_tr38901(bs_array, ut_array, scenario, center_freq, n_ut, seed, spacing, grid, area, bs, ut,
                                                      indoor, time_grid, los, output, attach, links, link_gain, uplink, o2i_low, config_dirs,
                                                      (nlhs > 1) ? &gain_dB : nullptr, (nlhs > 2) ? &layout : nullptr,
                                                      (nlhs > 3) ? &index : nullptr));

    plhs[0] = qd_mex_channel2struct(chan);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&gain_dB);
    if (nlhs > 2)
        plhs[2] = qd_mex_layout2struct(layout);
    if (nlhs > 3)
        plhs[3] = qd_mex_layout_index_out(index);
}
