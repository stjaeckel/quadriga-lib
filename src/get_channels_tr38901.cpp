// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_layout.hpp"
#include "qd_helpers_layout.hpp"

#include <cmath>
#include <any>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// The uplink of a downlink channel: the ends swapped, coefficients and delays transposed per path (reciprocal channel,
// no conjugation, so that the Doppler shift stays that of the downlink), the Jones matrices transposed
static void qd_uplink(quadriga_lib::channel<double> &c)
{
    const size_t u = c.name.find('_'); // "<bs>_<ut>"; the preset names have no '_'
    if (u != std::string::npos)
        c.name = c.name.substr(u + 1) + "_" + c.name.substr(0, u);
    for (size_t i = 0; i < c.par_names.size(); ++i) // Entries with one value per end, [Tx, Rx]
        if (c.par_names[i] == "b_med" || c.par_names[i] == "b_eff" || c.par_names[i] == "limit" || c.par_names[i] == "ratio")
            if (auto *v = std::any_cast<arma::vec>(&c.par_data[i]); v != nullptr && v->n_elem == 2)
                std::swap((*v)[0], (*v)[1]);
    std::swap(c.tx_pos, c.rx_pos);
    std::swap(c.tx_orientation, c.rx_orientation);
    std::swap(c.fbs_pos, c.lbs_pos);
    for (auto *v : {&c.coeff_re, &c.coeff_im, &c.delay})
        for (arma::cube &x : *v)
        {
            arma::cube t(x.n_cols, x.n_rows, x.n_slices, arma::fill::none);
            for (arma::uword p = 0; p < x.n_slices; ++p)
                t.slice(p) = x.slice(p).t();
            x = std::move(t);
        }
    for (arma::mat &a : c.path_angles) // {AOD, EOD, AOA, EOA}
        if (a.n_cols == 4)
            a = arma::join_rows(a.cols(2, 3), a.cols(0, 1));
    for (arma::mat &m : c.polmat) // Rows (V rx, V tx), (H rx, V tx), (V rx, H tx), (H rx, H tx), Re and Im each
        if (m.n_rows == 8)
            m = arma::join_cols(m.rows(0, 1), m.rows(4, 5), m.rows(2, 3), m.rows(6, 7));
}

/*!SECTION
Channel generators
SECTION!*/

/*!MD
# get_channels_tr38901
Generate the channels of a TR 38.901 scenario in one call

- Channels between the UTs of a drop and the BSs of a 3GPP TR 38.901 scenario (UMa, UMi, RMa, SMa, InH, InF), with the
  given BS and UT antenna arrays; by default one channel per UT and frequency, to its serving BS sector
- The scenario sets the BS layout and the drop of the UTs as in TR 38.901 (table in Notes); optional inputs change
  single values, the time evolution and which links are returned
- Built on the layout functions ([[generate_layout]], [[init_link_states]], [[attach_terminals]], [[generate_rays]],
  [[get_channels_from_rays]]); `layout_out` returns the simulated layout, to continue there

## Declaration:
```
std::vector<channel<double>> quadriga_lib::get_channels_tr38901(
    const arrayant<double> &bs_array,
    const arrayant<double> &ut_array,
    const std::string &scenario,
    const double_list &center_freq = {3.5e9},
    arma::uword n_ut = 1,
    arma::sword seed = -1,
    const double_list &spacing = {},
    const double_list &grid = {},
    const double_list &area = {},
    const double_list &bs = {},
    const double_list &ut = {},
    const double_list &indoor = {},
    const double_list &time_grid = {},
    const std::string &los = "auto",
    const std::string &output = "taps",
    const std::string &attach = "large_scale",
    const std::string &links = "serving",
    bool link_gain = true,
    bool uplink = false,
    double o2i_low = NAN,
    const std::vector<std::string> &config_dirs = {},
    arma::mat *gain_dB = nullptr,
    layout *layout_out = nullptr,
    arma::umat *index = nullptr);
```

## Inputs:
- **`bs_array`** — BS antenna array, valid ([[arrayant]]); every sector uses it, turned to the sector heading
- **`ut_array`** — UT antenna array, valid ([[arrayant]])
- **`scenario`** — `"UMa"`, `"UMi"`, `"RMa"`, `"SMa"`, `"InH_Open"`, `"InH_Mixed"`, `"InF_SL"`, `"InF_DL"`,
  `"InF_SH"`, `"InF_DH"` or `"InF_HH"`; also with the prefix `"38.901_"`
- **`center_freq`** *(optional)* — Carrier frequencies in Hz, ascending; one set of channels per frequency;
  `[n_freq]`
- **`n_ut`** *(optional)* — UTs per cell (per BS sector), >= 1
- **`seed`** *(optional)* — Seed of the random numbers: the same seed gives the same drop and channels; -1 = a random seed,
  which `layout_out.seed` returns

Optional changes to the scenario layout (defaults in Notes). Give only the values to change; NAN keeps a default.
[[generate_layout]] describes all values:

- **`spacing`** *(optional)* — Inter-site distance in m, e.g. `200`; indoor: BS spacing in x and y
- **`grid`** *(optional)* — Rings of sites around the centre site, e.g. `0` = one site, `2` = 19 sites; the
  number of sectors as third value, e.g. `{0, NAN, 1}`; indoor: BSs in x and y
- **`area`** *(optional)* — Hall size in m, length, width, height (InH, InF)
- **`bs`** *(optional)* — BS height in m, e.g. `30`; then the mechanical downtilt and an offset of the sector
  headings in degrees
- **`ut`** *(optional)* — UT height and minimum distance to the BS in m, speed in m/s, e.g. `{NAN, NAN, 30 / 3.6}` =
  all UTs at 30 km/h; then the orientation (bank, tilt, heading) in degrees
- **`indoor`** *(optional)* — Share of the UTs in buildings, e.g. `0.5` (UMa, UMi, RMa, SMa); then the range of floors
  and the floor height

Simulation settings:

- **`time_grid`** *(optional)* — `{T, dt}` in s: snapshots at 0, dt, 2 dt, … up to T, while the UTs move; `dt`
  defaults to 1 ms, e.g. `0.1` = 101 snapshots over 100 ms; omitted = one snapshot (static channel)
- **`los`** *(optional)* — `"auto"`: LOS or NLOS as drawn from the LOS probability of the scenario; `"los"` or `"nlos"`: that state for
  all links, also of the UTs in buildings and to the other sectors, so it can change the serving sector (`"InF_HH"` is
  always LOS)
- **`output`** *(optional)* — `"taps"`: the rays of a cluster combined into taps; `"rays"`: every ray as a path, with angles
  and path gains
- **`attach`** *(optional)* — How the serving sector is chosen: `"large_scale"` (coupling gain) or `"rsrp"` (Notes)
- **`links`** *(optional)* — Which links are returned: `"serving"`, the serving sector of each UT; `"site"`, all sectors of its
  serving site; `"all"`, all BS sectors, e.g. for interference studies (Returns)
- **`link_gain`** *(optional)* — false: without path loss, building loss and shadow fading, so that the power is that of the
  antennas and the small-scale fading; `gain_dB` and the serving sector still include them
- **`uplink`** *(optional)* — true: the uplink, with the UT as Tx (Notes)
- **`o2i_low`** *(optional)* — Share of buildings with the low-loss model of the building penetration loss, in [0, 1];
  NAN = the TR 38.901 value (50 % in UMa and UMi); not for InH, InF
- **`config_dirs`** *(optional)* — Folders with own scenario files (`.conf`), searched before the shipped ones

## Outputs:
- **`gain_dB`** *(optional)* — Coupling gain or RSRP in dB of each UT to each BS sector, e.g. for the SIR; `[n_ut_total, n_bs_node]`
- **`layout_out`** *(optional)* — The simulated layout: BS and UT positions, link states, pairing; see [[layout]]
- **`index`** *(optional)* — BS sector (node), UT and frequency of each channel, 0-based; `[3, n_channel]`

## Returns:
- Channels as [[channel]] objects; what taps and rays hold: [[gbsm_single_link_channel]]
  - `"serving"`: one channel per UT and frequency, all UTs at the first frequency, then at the next: `chan[i_ut + i_freq * n_ut_total]`, 0-based
  - `"site"`, `"all"`: one channel per BS sector, UT and frequency; the sectors of the first UT one after the other,
    then those of the next UT, then the next frequency; `index` names the BS sector, UT and frequency of each channel
  - UTs in drop order: the first `n_ut` were dropped in cell 0 (site BS1, sector 0), the next `n_ut` in cell 1, …;
    the serving sector is the one with the highest gain, often not the cell of the drop. The name says it too, e.g.
    `"BS5:0_UT11"` = site BS5, sector 0, UT 11
  - Coefficients and delays per snapshot `[n_ut_port, n_bs_port, n_tap]` (downlink; `uplink` swaps the first two)

## Example:
```
const auto bs = quadriga_lib::generate_antenna<double>("3gpp", 10.0, 3.5e9, {}, {4, 4, 1, 1});
const auto ut = quadriga_lib::generate_antenna<double>("xpol");
auto chan = quadriga_lib::get_channels_tr38901(bs, ut, "UMa", {3.5e9}, 10, 1);   // 19 sites, 10 UTs per cell
auto c2 = quadriga_lib::get_channels_tr38901(bs, ut, "UMi", 3.5e9, 10, 1, {}, 0, {}, {}, {}, {}, 0.1);   // one site, 100 ms
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
- **Serving sector**: the BS sector with the highest coupling gain (`"large_scale"`, TR 38.901 Table 7.8-1) or RSRP
  (`"rsrp"`, Table 7.8-2) at the first frequency, see [[attach_terminals]]; `"rsrp"` and `links` = `"all"` generate
  the rays of all links, the others only those to the serving sites
- **Time evolution**: the UTs move from their drop position along their direction at their speed; delays and angles of
  the clusters stay, the phases follow the motion (mobility `"phase"` of [[get_channels_from_rays]])
- **Several frequencies**: the clusters come from the same random values at all frequencies, scaled by the spreads of
  each frequency (method `"38.901_MF"`, TR 38.901 clause 7.6.5)
- **Uplink**: the transposed downlink (reciprocal channel, the same Doppler shift), with the two ends swapped in all
  members and the name; `index` and `gain_dB` keep the downlink meaning. This is the uplink of a TDD system; for FDD,
  add the uplink frequency to `center_freq`
- **Memory**: about 90 kB per snapshot for the 57 UTs of the default UMa drop with 2 × 2 ports (16 to 25 taps);
  `links` = `"all"` multiplies it by the number of BS sectors; `"site"` by the sectors per site (3)

## See also:
- [[channel]] (the returned objects)
- [[freq_response]] (frequency response of the channels at given subcarriers)
- [[generate_layout]] (the scenario layouts and their inputs in full)
- [[get_channels_layout]] (channels of any layout)
- [[get_channels_ieee_indoor]] (the IEEE indoor models)
MD!*/

std::vector<quadriga_lib::channel<double>> quadriga_lib::get_channels_tr38901(
    const arrayant<double> &bs_array, const arrayant<double> &ut_array, const std::string &scenario, const double_list &center_freq_in,
    arma::uword n_ut, arma::sword seed, const double_list &spacing_in, const double_list &grid_in, const double_list &area_in, const double_list &bs_in,
    const double_list &ut_in, const double_list &indoor_in, const double_list &time_grid_in, const std::string &los, const std::string &output,
    const std::string &attach, const std::string &links, bool link_gain, bool uplink, double o2i_low, const std::vector<std::string> &config_dirs,
    arma::mat *gain_dB, layout *layout_out, arma::umat *index)
{
    const arma::vec &center_freq = center_freq_in.vec();
    const arma::vec &spacing = spacing_in.vec();
    const arma::vec &grid = grid_in.vec();
    const arma::vec &area = area_in.vec();
    const arma::vec &bs = bs_in.vec();
    const arma::vec &ut = ut_in.vec();
    const arma::vec &indoor = indoor_in.vec();
    const arma::vec &time_grid = time_grid_in.vec();
    
    for (const auto &[a, name] : {std::make_pair(&bs_array, "bs_array"), std::make_pair(&ut_array, "ut_array")})
        if (const std::string e = a->is_valid(); !e.empty())
            throw std::invalid_argument("Input '" + std::string(name) + "': " + e);
    if (center_freq.is_empty())
        throw std::invalid_argument("Input 'center_freq' must not be empty.");
    if (n_ut == 0)
        throw std::invalid_argument("Input 'n_ut' must be at least 1.");
    if (time_grid.n_elem > 2)
        throw std::invalid_argument("Input 'time_grid' must have at most 2 entries: [observation_time, update_interval].");
    const double t_obs = time_grid.n_elem > 0 ? time_grid[0] : 0.0, t_upd = time_grid.n_elem > 1 ? time_grid[1] : 1.0e-3;
    if (!std::isfinite(t_obs) || t_obs < 0.0 || !std::isfinite(t_upd) || t_upd <= 0.0)
        throw std::invalid_argument("Input 'time_grid' must hold a finite observation time >= 0 and a finite update interval > 0.");
    if (t_obs / t_upd >= 4294967295.0)
        throw std::invalid_argument("Input 'time_grid' must give observation_time / update_interval below 2^32 - 1.");
    if (los != "auto" && los != "los" && los != "nlos")
        throw std::invalid_argument("Input 'los' must be 'auto', 'los' or 'nlos'.");
    if (output != "taps" && output != "rays")
        throw std::invalid_argument("Input 'output' must be 'taps' or 'rays'.");
    if (attach != "large_scale" && attach != "rsrp")
        throw std::invalid_argument("Input 'attach' must be 'large_scale' or 'rsrp'.");
    if (links != "serving" && links != "site" && links != "all")
        throw std::invalid_argument("Input 'links' must be 'serving', 'site' or 'all'.");
    if (!std::isnan(o2i_low) && !(o2i_low >= 0.0 && o2i_low <= 1.0))
        throw std::invalid_argument("Input 'o2i_low' must be in [0, 1], or NaN.");

    // The preset of the family; the indoor presets have only indoor UTs
    const std::string family = scenario.rfind("38.901_", 0) == 0 ? scenario.substr(7) : scenario;
    bool known = false;
    for (const char *f : {"UMa", "UMi", "RMa", "SMa", "InH_Open", "InH_Mixed", "InF_SL", "InF_DL", "InF_SH", "InF_DH", "InF_HH"})
        known = known || family == f;
    if (!known)
        throw std::invalid_argument("Input 'scenario' must be 'UMa', 'UMi', 'RMa', 'SMa', 'InH_Open', 'InH_Mixed', 'InF_SL', 'InF_DL', "
                                    "'InF_SH', 'InF_DH' or 'InF_HH', also with the prefix '38.901_'.");
    const std::string preset = "38.901_" + family;
    if (family.rfind("In", 0) == 0 && !std::isnan(o2i_low))
        throw std::invalid_argument("Input 'o2i_low' must be NaN for the indoor preset '" + preset + "'.");
    if (family == "InF_HH" && los == "nlos")
        throw std::invalid_argument("Input 'los' cannot be 'nlos' for 'InF_HH', which is always LOS.");

    layout L = generate_layout(preset, n_ut, center_freq, spacing, grid, area, bs, ut, indoor, {}, true, seed);
    set_antenna(L, bs_array, false);
    set_antenna(L, ut_array, true);
    init_link_states(L, center_freq.n_elem > 1 ? "38.901_MF" : "38.901", std::isnan(o2i_low) ? arma::vec() : arma::vec{o2i_low}, false,
                     2.0e9, "", -1, config_dirs);
    if (los != "auto")
        L.links.los.fill(los == "los" ? 1 : 0);

    // Serving sector of each UT; without "all", the pairs of its site and the rays of the links to that site
    const bool all = links == "all";
    std::vector<std::vector<gbsm_rays>> rays;
    if (attach == "rsrp" || all)
        rays = generate_rays(L);
    arma::uvec site;
    arma::mat g;
    const arma::uvec node = attach_terminals(L, attach, rays, &site, &g);
    const layout_links &K = L.links;
    std::vector<arma::uword> sel;
    for (arma::uword l = 0; l < K.n_link(); ++l)
        if (all || K.i_tx[l] == site[K.i_rx[l]])
            sel.push_back(l);
    if (!all)
        L.pairing = arma::join_cols(site.t(), arma::regspace<arma::urowvec>(0, site.n_elem - 1));
    if (attach == "rsrp" && !all)
    {
        std::vector<std::vector<gbsm_rays>> s;
        for (arma::uword l : sel)
            s.push_back(std::move(rays[l]));
        rays = std::move(s);
    }
    else if (!all)
        rays = generate_rays(L, sel);

    arma::vec times;
    if (t_obs > 0.0)
        times = arma::regspace(0.0, std::floor(t_obs / t_upd * (1.0 + 1e-12))) * t_upd; // 0.3 / 0.1 is 2.9999999999999996
    std::vector<channel<double>> out = qd_layout::channels_of_nodes(L, rays, sel, links == "serving" ? node : arma::uvec(), times, output,
                                                                    link_gain, index);
    if (uplink)
        for (auto &c : out)
            qd_uplink(c);
    if (gain_dB != nullptr)
        *gain_dB = std::move(g);
    if (layout_out != nullptr)
        *layout_out = std::move(L);
    return out;
}
