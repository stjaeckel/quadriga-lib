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
- Armadillo types are column-major. Shape notation `[a, b, c]` means `[rows, cols, slices]` for `arma::Cube`; `[rows, cols]` for `arma::Mat`; `[n]` for `arma::Col`/`arma::Row`.
- Pointer arguments: `nullptr` skips optional outputs; required inputs throw on `nullptr`.
- Output containers are resized automatically unless they already have the correct shape; this invalidates any prior pointers into their memory.
- An output cannot be the same object as an input or another output, nor share memory with one; such a call throws `std::invalid_argument`. Exceptions are stated in the function documentation.
- Invalid inputs (shape/domain) cause a `std::invalid_argument`; I/O failures a `std::runtime_error`.
- Index conventions: 0-based unless the field is explicitly called "1-based" (which applies to `obj_ind`, `mtl_ind`, `fbs_ind`, `sbs_ind`, and QDANT `id`).
- Units: angles in radians (degrees only where stated, e.g. `*_deg`, `*_3dB`); distances in meters; frequencies in Hz; time in seconds; powers linear unless `_dB`.
- Coordinate system: GCS = right-handed Cartesian, meters. Euler angles are intrinsic Tait-Bryan in the order (bank=x, tilt=y, heading=z), applied as Rz·Ry·Rx, each a right-handed rotation as in 3GPP TR 38.901 (bearing, downtilt, slant): a positive heading turns the x-axis (boresight) toward y, a positive tilt turns it down, a positive bank turns the y-axis toward z.
- Polarization transfer matrix `M`: 8 rows per path, interleaved real/imaginary, order `[ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH]`. A 2-row form `[ReVV, ImVV]` is used for scalar (acoustic) fields.
- Speed of light/sound defaults: `299792458.0` m/s (EM), `343.0` m/s (acoustic).
- Kernel-selection parameters (`use_kernel`): `0` = auto (CUDA if available and problem large enough, else AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2, `3` = CUDA. Throws if the requested kernel is unavailable.
- `gpu_id` is only read when `use_kernel` resolves to CUDA.
