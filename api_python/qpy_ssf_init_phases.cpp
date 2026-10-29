// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# ssf_init_phases
Draw the random phases of one small-scale fading generator set of a BS site and scenario

- The small-scale fading (SSF) random variables of the clusters (TR 38.901 clause 7.6.3.1) are spatially correlated
  sum-of-sinusoids generators in two kinds of sets, with `N = NumClusters - 1` NLOS clusters. Column `c = b·N + n`
  belongs to block `b` and cluster `n` (0-based):
- Shared cluster set (`set_id = -1`, frequency-independent), `9·N` generators: block 0 cluster delay (Uniform(0,1));
  blocks 1, 2 sign of AOD, AOA; 3, 4 sign of ZOD, ZOA; 5, 6 angle offset of AOD, AOA; 7, 8 angle offset of ZOD, ZOA
  (all Normal(0,1))
- Per-frequency set (`set_id = k >= 0`), `N` generators: block 0 cluster shadowing (Normal(0,1)). `k` is the index
  of the frequency in the caller's frequency list; the list must have the same order in every call
- This function draws the phases; [[ssf_draw]] evaluates them at the links. The caller stores `phase` and evaluates
  the same fields again along tracks or on maps
- Slice 1 (Tx side) is always the mirror of slice 0: column `b·N + n` of slice 1 is column `m(b)·N + n` of slice
  0, with `m` exchanging the blocks 1 and 2, 3 and 4, 5 and 6, 7 and 8 (departure and arrival of the same cluster);
  the delay block and the shadowing block mirror themselves
- Seed of the set, derived from `seed`, the seed of the large-scale set of the same site and scenario (see
  [[lsf_init_phases]]): `seed_set = splitmix64( splitmix64(seed) ^ tag ) >> 1` with `tag = 1` for the shared set
  and `tag = 2 + k` for the set of frequency `k`, and `splitmix64(x)`: `z = x + 0x9E3779B97F4A7C15`,
  `z = (z ^ (z >> 30)) · 0xBF58476D1CE4E5B9`, `z = (z ^ (z >> 27)) · 0x94D049BB133111EB`, result `z ^ (z >> 31)`
  (unsigned 64-bit arithmetic, wrapping). Slice 0 equals `quadriga_lib.sos.init_phases( n_coeff, n_gen, seed_set )`
  of [[sos.init_phases]], with `n_coeff` the number of sinusoids of `preset`. The sets therefore never repeat the
  large-scale phases of the same seed, and the sets of different frequencies are independent streams
- The same seed gives the same phases on all platforms; the caller keeps an explicit seed (or the phase arrays) to
  reproduce the clusters. `seed = -1` draws unrelated random phases from the system random device
- Each scenario of a site (link type LOS, NLOS, O2I) needs its own large-scale seed: TR 38.901 clause 7.6.3.4 makes
  the link types uncorrelated, and one seed would give the LOS and the NLOS scenario identical cluster fields
- `NumClusters = 1` (`N = 0`, LOS path only): both sets are empty, `phase` has shape `(n_coeff, 0, 2)`
- Append-only: a later version that appends blocks to a set keeps the existing columns and their phases;
  [[sos.init_phases]] fills column `g` from one stream independently of the number of columns
- No bias removal (the function takes no positions). `SC_lambda`, the decorrelation distance of all generators, is
  used only by [[ssf_draw]]
- `scenpar` is completed with the defaults of missing keys and validated as in [[scenpar_validate]]; its messages
  pass through unchanged. An invalid value of a key that is not used here throws as well
- Validation order: `seed`, `set_id`, `preset`, `scenpar`

## Usage:
```
phase = quadriga_lib.gbsm.ssf_init_phases( scenpar, seed, set_id, preset )
```

## Inputs:
- **`scenpar`** — Scenario parameters; dict as returned by [[conf_file_read]] or [[scenpar_validate]] (the
  entry `SubpathMethod` is ignored), or any subset of its keys; missing keys have their default value;
  `None` = all defaults; `NumClusters` sets the number of generators
- **`seed`** — Seed of the large-scale set of the same site and scenario; int from -1 to 2^63-1; -1 = system random
  device, a new set of phases on each call; default: -1
- **`set_id`** — `-1` = shared cluster set; `k >= 0` = set of frequency `k` (0-based); int; default: -1
- **`preset`** — Sum-of-sinusoids preset, see [[sos.preset]]; sets `n_coeff`; default: `"Comb256"`

## Output:
- **`phase`** — Phases in rad; slice 0 = Rx side, slice 1 = Tx side (mirror of slice 0); all values in
  [-pi, pi); `(n_coeff, n_gen, 2)` with `n_gen = 9·N` (shared set) or `N` (per-frequency set)

## See also:
- [[ssf_draw]] (evaluates the phases at the links)
- [[ssf_clusters]] (the clusters from the random values)
- [[lsf_init_phases]] (the large-scale set and its seed)
- [[sos.init_phases]] (the raw phases)
MD!*/

py::array_t<double> ssf_init_phases(py::handle scenpar,
                                    long long seed,
                                    long long set_id,
                                    const std::string &preset)
{
    // Read inputs; the entry 'SubpathMethod' of the scenario dict is not used
    std::string subpath_method;
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, subpath_method, "scenpar");

    // Call library function; the size of 'phase' depends on 'scenpar' and 'preset', so it is copied afterwards
    arma::cube phase;
    quadriga_lib::ssf_init_phases(scenpar_a, phase, (arma::sword)seed, (arma::sword)set_id, preset);

    return qd_python_copy2numpy(&phase);
}

// pybind11 declaration:
// m.def("ssf_init_phases", &ssf_init_phases,
//       py::arg("scenpar"),
//       py::arg("seed") = -1,
//       py::arg("set_id") = -1,
//       py::arg("preset") = "Comb256");
