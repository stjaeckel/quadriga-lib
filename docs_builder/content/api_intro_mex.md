+++
title = "MATLAB / Octave API"
description = "%%GAP: MATLAB / Octave API reference for Quadriga-Lib: array antennas, channel generation, statistics, the 3GPP stochastic model and site-specific simulation tools.%%"
lead = "%%GAP: Reference for the MATLAB / Octave interface, version {{ version }}. All functions live in the `+quadriga_lib` package.%%"
+++

- Each function has a 1-line short description, optional detailed notes, a Usage block, and Inputs/Outputs sections.
- Array sizes follow in backticks, e.g. `[n_rx, n_tx, n_path]`.
- The conventions and shared quantities of the C++ [Glossary](cpp_api.html#glossary) apply; the points below are specific to MATLAB / Octave.
- All functions live in the `+quadriga_lib` package and are called as `quadriga_lib.function_name(...)`.
- Numeric inputs accept any numeric type and are cast internally; default is `double` unless stated otherwise (e.g. `uint64` for some index arguments).
- Arguments have the names, order and defaults of the C++ function unless the function states otherwise.
- Inputs with a stated default, or marked *(optional)*, can be omitted or passed as `[]`; all others are required. Output arguments are optional via `nargout`; a call without assignment returns the first output as `ans`, and outputs after the first are returned only when requested.
- Invalid inputs (shape/domain) and problems with a file's access or content raise a MATLAB error with the identifier `quadriga_lib:CPPerror`.
- Index conventions: 1-based unless the field is explicitly called 0-based.
- The canonical arrayant struct is defined in [[arrayant_generate]]. Mandatory fields: `e_theta_re/im`, `e_phi_re/im`, `azimuth_grid`, `elevation_grid`. Optional field defaults: `element_pos = zeros(3, n_elements)`, identity coupling, `center_freq = 299792458`, empty `name`. A struct array represents a frequency-dependent (multi-frequency) model with one entry per frequency. Each call builds a new arrayant from the struct, so its data is checked in full on every call.
