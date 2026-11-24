// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_layout.hpp"

#include <cmath>
#include <string>

/*!SECTION
Network layouts and mobility
SECTION!*/

/*!MD
# layout_links
State of the propagation links of one realisation of a layout

- Holds the discrete state of the links (line of sight, indoor state, scenario and random seeds), no large-scale parameters, phases or rays; it is not stored in the CSV files
- A propagation link is one site, one terminal and one segment of a track with one resolved scenario; its arrays have one entry per link

## Attributes:<br>
| Attribute | Size | Description |
| --------- | ---- | ----------- |
| `arma::u32_vec i_tx`, `i_rx`, `i_segment` | `[n_link]` | Site, terminal and segment of each link, 0-based |
| `arma::u32_vec image` | `[n_link]` | Wrap-around image of the site, 0 = the real site |
| `arma::u32_vec i_conf` | `[n_link]` | Index into `conf_names`, derived from scenario, LOS and indoor state |
| `std::vector<std::string> conf_names` | `[n_conf]` | Resolved conf names |
| `arma::u32_vec los` | `[n_link]` | 1 = line of sight |
| `arma::s32_vec indoor` | `[n_link]` | Indoor state at the start of the segment, as in [[layout_nodes]] |
| `arma::u32_vec floor_id` | `[n_link]` | Floor number - 1, 0 outdoors |
| `arma::vec d2d_in` | `[n_link]` | Indoor 2D distance in [m] |
| `arma::mat o2i_rnd` | `[n_terminal, 4]` | Random inputs of [[o2i_loss]] |
| `arma::u32_vec o2i_model` | `[n_terminal]` | Chosen O2I model, index into the list of the family |
| `arma::s64_vec gbsm_seed` | `[n_link]` | Seed of the random phases; equal for one site, image and conf |
| `arma::sword seed` | scalar | Master seed, -1 = not set |
| `std::string method`, `sos_preset` | | Method and SOS preset as used |
| `double anchor_freq` | scalar | Anchor frequency of the multi-frequency model in [Hz], default 2e9 |
| `bool pol_variability` | scalar | Polarization power variability, default false |
| `std::vector<std::string> conf_files` | `[n_conf]` | Resolved paths of the conf files; empty = not resolved |
| `unsigned long long layout_hash` | scalar | Hash of all layout members that determine the links |

## Member functions:<br>
| Function | Description |
| -------- | ----------- |
| `.n_link()` | Number of links |
| `.empty()` | `true` if there are no links |
| `.reset()` | Empties all members and restores the defaults |
| `.is_valid()` | `std::string`: empty if the object is valid, else an error message; checks the sizes, the indices into `conf_names` and the ranges |

## See also:
- [[layout]] (holds the links as `links`)
- [[o2i_loss]] (uses `o2i_rnd`)
MD!*/

arma::uword quadriga_lib::layout_links::n_link() const { return i_tx.n_elem; }

bool quadriga_lib::layout_links::empty() const { return i_tx.n_elem == 0; }

void quadriga_lib::layout_links::reset() { *this = layout_links(); }

std::string quadriga_lib::layout_links::is_valid() const
{
    const arma::uword n = i_tx.n_elem;
    if (i_rx.n_elem != n || i_segment.n_elem != n || image.n_elem != n || i_conf.n_elem != n || los.n_elem != n ||
        indoor.n_elem != n || floor_id.n_elem != n || d2d_in.n_elem != n || gbsm_seed.n_elem != n)
        return "The members 'i_tx', 'i_rx', 'i_segment', 'image', 'i_conf', 'los', 'indoor', 'floor_id', 'd2d_in' and 'gbsm_seed' must have one entry per link.";
    for (arma::uword i = 0; i < n; ++i)
    {
        if (i_conf[i] >= conf_names.size())
            return "Member 'i_conf' has an index outside 'conf_names'.";
        if (los[i] > 1)
            return "Member 'los' must be 0 or 1.";
        if (indoor[i] < -1 || indoor[i] > 3)
            return "Member 'indoor' must be -1, 0, 1, 2 or 3.";
        if (std::isinf(d2d_in[i]) || d2d_in[i] < 0.0)
            return "Member 'd2d_in' must be NaN or >= 0.";
    }
    if (!conf_files.empty() && conf_files.size() != conf_names.size())
        return "Member 'conf_files' must be empty or have one entry per conf name.";
    if ((!o2i_rnd.is_empty() && o2i_rnd.n_cols != 4) || o2i_rnd.n_rows != o2i_model.n_elem)
        return "Member 'o2i_rnd' must have 4 columns and one row per entry of 'o2i_model'.";
    if (seed < -1)
        return "Member 'seed' must be >= -1.";
    if (!(anchor_freq > 0.0) || !std::isfinite(anchor_freq))
        return "Member 'anchor_freq' must be positive and finite.";
    return "";
}
