# General usage notes

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/mex_api.html#sec-general-usage-notes

- Each function has a 1-line short description, optional detailed notes, a Usage block, and Inputs/Outputs sections.
- Array sizes follow in backticks, e.g. `[n_rx, n_tx, n_path]`.
- All functions live in the `+quadriga_lib` package and are called as `quadriga_lib.function_name(...)`.
- Numeric inputs accept any numeric type and are cast internally; default is `double` unless stated otherwise (e.g. `uint64` for some index arguments).
- MATLAB arrays are column-major. Shape notation `[a, b, c]` means `[rows, cols, slices]` for 3-D arrays; `[rows, cols]` for 2-D matrices; `[n]` for vectors.
- Inputs with a stated default, or marked *(optional)*, can be omitted or passed as `[]`; all others are required. Output arguments are optional via `nargout`; outputs after the first are computed only when requested.
- Functions operating on arrayant data accept either a single struct (*struct mode*) or the individual fields as separate positional arguments (*split mode*). Split-mode signatures are single-frequency only; multi-frequency input requires struct mode.
- Invalid inputs (shape/domain) and I/O failures raise a MATLAB error with the identifier `quadriga_lib:CPPerror`.
- Index conventions: 1-based unless the field is explicitly called 0-based.
- Units: angles in radians (degrees only where stated, e.g. `*_deg`); distances in meters; frequencies in Hz; time in seconds; powers linear unless `_dB`.
- Coordinate system: GCS = right-handed Cartesian, meters. Euler angles are intrinsic Tait-Bryan in the order (bank=x, tilt=y, heading=z), applied as Rz·Ry·Rx.
- Polarization transfer matrix `M`: 8 rows per path, interleaved real/imaginary, order `[ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH]`. A 2-row form `[ReVV, ImVV]` is used for scalar (acoustic) fields.
- The canonical arrayant struct is defined in [arrayant_generate](array-antenna-functions.md#arrayant_generate). Mandatory fields: `e_theta_re/im`, `e_phi_re/im`, `azimuth_grid`, `elevation_grid`. Optional field defaults: `element_pos = zeros(3, n_elements)`, identity coupling, `center_freq = 299792458`, empty `name`. A struct array represents a frequency-dependent (multi-frequency) model with one entry per frequency.
- Speed of light/sound defaults: `299792458.0` m/s (EM), `343.0` m/s (acoustic).
- Kernel-selection parameters (`use_kernel`): `0` = auto (CUDA if available and problem large enough, else AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2, `3` = CUDA. Throws if the requested kernel is unavailable.
- `gpu_id` is only read when `use_kernel` resolves to CUDA.
