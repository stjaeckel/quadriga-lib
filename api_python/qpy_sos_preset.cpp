// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_preset
Frequencies and ACF of a predefined sum-of-sinusoids generator

- Returns a precomputed frequency table for [[sos_val]] together with the autocorrelation
  function (ACF) it was fitted to
- Presets: "Exp128", "Exp256", "Exp512", "Exp1024", "Gauss128", "Gauss256", "Gauss512",
  "Gauss1024", "Comb128", "Comb256", "Comb512", "Comb1024". The number is the number of
  sinusoids, `n_coeff`. The name is matched exactly, including upper and lower case
- ACF types, with the distance `D` and the decorrelation distance `d`: `Exp`: `R = exp( -D/d )`;
  `Gauss`: `R = exp( -(D/d)^2 )`; `Comb`: `Gauss` for `D <= d`, `Exp` beyond.
  All three have `R(d) = exp(-1)`
- The ACF is returned as a table with 200 samples from 0 to `4.975·d` in steps of `0.025·d`
- The tables are stored for `d = 10 m`. For another decorrelation distance the distances are
  multiplied by `d/10` and the frequencies by `10/d`
- All tables are 3-dimensional and can be used with positions of 1, 2 or 3 columns
- No sinusoid has a wavelength above 20 decorrelation distances. The ACF of the tables stays near
  zero beyond the fitted range, and the mean of the generator over a large area is not offset by
  sinusoids of nearly zero frequency

## Usage:
```
freq, acf, dist = quadriga_lib.sos.preset( name, dist_decorr )
```

## Inputs:
- **`name`** — Name of the preset, e.g. "Comb256"; `str`
- **`dist_decorr`** — Decorrelation distance; must be > 0; default: 10.0

## Outputs:
- **`freq`** — Frequencies; `(n_coeff, 3)`
- **`acf`** — Target ACF; `(200,)`
- **`dist`** — Sample points of the ACF; `(200,)`

## See also:
- [[sos_init_phases]] (random phases for the generator)
- [[sos_val]] (evaluation)
- [[sos_generate]] (frequencies for other ACFs)
MD!*/

py::tuple sos_preset(const std::string &name,
                     double dist_decorr)
{
    // Output allocation; the number of sinusoids is known only after the call
    arma::mat freq;
    arma::vec acf, dist;
    auto acf_py = qd_python_init_output((arma::uword)200, &acf);
    auto dist_py = qd_python_init_output((arma::uword)200, &dist);

    // Call library function
    quadriga_lib::sos_preset<double>(name, freq, &acf, &dist, dist_decorr);

    // Copy to python
    auto freq_py = qd_python_copy2numpy(&freq);

    return py::make_tuple(freq_py, acf_py, dist_py);
}

// pybind11 declaration:
// m.def("preset", &sos_preset,
//       py::arg("name"),
//       py::arg("dist_decorr") = 10.0);
