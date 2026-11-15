+++
title = "C++ API"
description = "%%GAP: C++ API reference for Quadriga-Lib: array antennas, channel generation, statistics, the 3GPP stochastic model and site-specific simulation tools.%%"
lead = "%%GAP: Reference for the C++17 library, version {{ version }}. All functions live in the `quadriga_lib` namespace.%%"
+++

- Each function has a 1-line short description, optional detailed notes, a Declaration block, and Inputs/Outputs/Returns sections.
- Array sizes follow in backticks, e.g. `[n_rx, n_tx, n_path]`.
- All functions and classes live in the `quadriga_lib` namespace.
- Default include: `#include "quadriga_lib.hpp"`.
- Template parameter `dtype` is `float` or `double` unless stated.
- Pointer arguments: `nullptr` skips optional outputs; required inputs throw on `nullptr`.
- Output containers are resized automatically unless they already have the correct shape; this invalidates any prior pointers into their memory.
- An output cannot be the same object as an input or another output, nor share memory with one; such a call throws `std::invalid_argument`. Exceptions are stated in the function documentation.
- Invalid inputs (shape/domain) cause a `std::invalid_argument`; I/O failures a `std::runtime_error`.
- Conventions and the names of shared quantities: see the [Glossary](#glossary).

## Glossary

The conventions and shared quantities of all three APIs; the MATLAB / Octave and Python usage notes give what is specific to them.

**General conventions**

- Shapes: `[a, b, c]` means `[rows, cols, slices]`, `[rows, cols]` a matrix and `[n]` a vector; arrays are column-major (C++: `arma::Cube`, `arma::Mat`, `arma::Col` / `arma::Row`).
- Index base: 0-based unless the field is explicitly called "1-based" (which applies to `obj_ind`, `mtl_ind`, `fbs_ind`, `sbs_ind`, and QDANT `id`).
- Units: angles in radians (degrees only where stated, e.g. `*_deg`, `*_3dB`); distances in meters; frequencies in Hz; time in seconds; powers linear unless `_dB`.
- Coordinate system: GCS = right-handed Cartesian, meters. Euler angles are intrinsic Tait-Bryan in the order (bank=x, tilt=y, heading=z), applied as Rz·Ry·Rx, each a right-handed rotation as in 3GPP TR 38.901 (bearing, downtilt, slant), so that orientations from TR 38.901 are entered as published: a positive heading turns the x-axis (boresight) toward y, a positive tilt turns it down, a positive bank turns the y-axis toward z.
- Complex data: separate real and imaginary arrays (`_re` / `_im`), or interleaved along the rows where documented (e.g. `M`), for compatibility with older Octave versions, whose MEX interface has no interleaved complex arrays. Some functions also offer an optional `std::complex` output, which replaces the split pair (then left empty).
- Speed of light/sound defaults: `299792458.0` m/s (EM), `343.0` m/s (acoustic).

**Frequencies**

- `center_freq`: carrier frequency in Hz; scalar, or one value per snapshot or per frequency entry `[n_freq]`. Stored in HDF5 and QDANT files as `CenterFrequency`.
- `freq`: short name for a frequency in Hz; a value that must stay in GHz is named `freq_GHz`. For a frequency grid, `freq` holds the sample frequencies and `freq_new` the target frequencies.
- `spatial_freq`: spatial frequencies of a sum-of-sinusoids generator in 1/m, `[n_coeff, n_dim]`; limits `spatial_freq_min`, `spatial_freq_max`.

**Positions and orientations**

- `tx_pos`, `rx_pos`, `irs_pos`, `orig_pos`: positions in GCS in meters; `tx_orientation`, `rx_orientation`, `irs_orientation`, `orig_orientation`: Euler angles (bank, tilt, heading) in rad. C++: `arma::Mat`, `[3, 1]` for one item (inputs also take `[1, 3]`), `[3, n]` for many (e.g. one per snapshot); MATLAB / Octave and Python: a 3-element vector for one item. Exceptions with one row per item, `[n, 3]`: the ray-tracing state, meshes, point clouds, QRT CIR and origin position lists (`qrt_file_init`, `qrt_file_parse`), sum-of-sinusoids positions and the GBSM batch inputs.

**Polarization**

- `M`: polarization transfer matrix; 8 rows per path, interleaved real/imaginary, order `[ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH]`. A 2-row form `[ReVV, ImVV]` is used for scalar (acoustic) fields.

**Kernel selection**

- `use_kernel`: compute kernel; `0` = auto (CUDA if available and problem large enough, else AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2, `3` = CUDA. Throws if the requested kernel is unavailable.
- `gpu_id`: CUDA device ID; only read when `use_kernel` resolves to CUDA.
