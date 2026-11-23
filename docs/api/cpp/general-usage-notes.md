# General usage notes

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-general-usage-notes

- Each function has a 1-line short description, optional detailed notes, a Declaration block, and Inputs/Outputs/Returns sections.
- Array sizes follow in backticks, e.g. `[n_rx, n_tx, n_path]`.
- All functions and classes live in the `quadriga_lib` namespace.
- Default include: `#include "quadriga_lib.hpp"`.
- Template parameter `dtype` is `float` or `double` unless stated. It is deduced from the data arguments; where no mandatory argument fixes it, it defaults to `double`. Real-valued scalar options are `double` for both types and are converted to `dtype` inside the function.
- Mandatory arguments are references. Optional inputs are references with an empty default; an empty container (`{}`) means "not given". Optional outputs are pointers with a `nullptr` default (not computed). The few optional inputs whose empty value has its own meaning are pointers with a `nullptr` default; their documentation says so. `ray_mesh_interact` and `ray_state_update` still take mandatory arguments by pointer and throw on `nullptr`.
- Output containers are resized automatically unless they already have the correct shape; this invalidates any prior pointers into their memory.
- An output argument cannot be the same object as an input or another output, nor share memory with one; such a call throws `std::invalid_argument`. Exceptions are stated in the function documentation. A returned result is a new object and may be assigned to an input.
- Invalid inputs (shape/domain, an index outside the layout of a file) cause a `std::invalid_argument`; a problem with a file's access or content (missing, unreadable, truncated, malformed, unsupported) a `std::runtime_error`.
- Conventions and the names of shared quantities: see the [Glossary](#glossary).

<a id="glossary"></a>
## Glossary

The conventions and shared quantities of all three APIs; the MATLAB / Octave and Python usage notes give what is specific to them.

**General conventions**

- Shapes: `[a, b, c]` means `[rows, cols, slices]`, `[rows, cols]` a matrix and `[n]` a vector; arrays are column-major (C++: `arma::Cube`, `arma::Mat`, `arma::Col` / `arma::Row`).
- Index base: 0-based unless the field is explicitly called "1-based" (which applies to `mtl_ind`, `fbs_ind`, `sbs_ind`, `point_ind`, `forward_ind`, `mesh_ind`, and QDANT `id`).
- Units: SI units, angles in radians, powers and gains linear. A value in dB, dBi or degrees carries the unit in its name (`threshold_dB`, `directivity_dBi`, `quantize_deg`); a value in the default unit carries no suffix. Other non-SI units keep their suffix, e.g. `speed_station_kmh`.
- Coordinate system: GCS = right-handed Cartesian, meters. Euler angles are intrinsic Tait-Bryan in the order (bank=x, tilt=y, heading=z), applied as Rz·Ry·Rx, each a right-handed rotation as in 3GPP TR 38.901 (bearing, downtilt, slant), so that orientations from TR 38.901 are entered as published: a positive heading turns the x-axis (boresight) toward y, a positive tilt turns it down, a positive bank turns the y-axis toward z.
- Complex data: separate real and imaginary arrays (`_re` / `_im`), or interleaved along the rows where documented (e.g. `polmat`), for compatibility with older Octave versions, whose MEX interface has no interleaved complex arrays. Some functions also offer an optional `std::complex` output, which replaces the split pair (then left empty).
- Speed of light/sound: `299792458.0` m/s (EM), `343.0` m/s (acoustic; the one value the library uses for sound: the acoustic frequency mapping, acoustic wavelengths and `generate_speaker`).
- Counts: `n_<thing>`, e.g. `n_path`, `n_snap`, `n_interact` (interaction points per path; mesh intersections per ray in the ray-tracing functions).
- Limits: `max_<things>` / `min_<things>` limit a count, e.g. `max_interactions`, `max_paths`, `min_segments`; a limit on a value names the quantity, e.g. `max_path_length`.
- Selectors: `i_<dimension>`, the index or indices of the items to use along one dimension, e.g. `i_element`, `i_snap`, `i_freq`, `i_cir`; for a copy, `i_source` and `i_target`.
- Index arrays: `<thing>_ind`, indices into other data, e.g. `obj_ind`, `mtl_ind`, `sub_mesh_ind`, `hit_ind`.
- Identifiers: `<thing>_id`, e.g. `channel_id`, `gpu_id`.
- Flags: a `bool` is named for what `true` does, never negated, and one concept has one name, e.g. `wrapping`, `uplink` (`true` = reverse link, e.g. in `qrt_file_read` the origin is the receiver).
- Not given / all: an omitted optional input (C++ an empty container, or `nullptr` for an input passed by pointer, MATLAB / Octave `[]`, Python `None`), never a value inside an index or ID range. Special values kept: `seed = -1` (system random device), `set_id = -1` (shared set; MATLAB / Octave `0`, the 1-based form of -1), `qdant_write` `id = 0` (append with a new ID), and `0` = unlimited for count limits.
- Non-finite inputs: NaN and Inf throw. Exceptions: a function for which a scan of its large data arrays would cost noticeable time may leave them unchecked and then states what non-finite values give; a scalar whose default depends on other inputs defaults to `NaN`, which selects that default; where the quantity is a setting for which <= 0 means "not set" (a size, an angle range, a frequency limit), values <= 0 select it too. A physical model parameter (a ratio, a distance, a spread) throws for values outside its range instead. Each such input documents the rule and the default, e.g. "NaN or <= 0: default 90°".
- Modes: integer codes (e.g. `usage`, `pol`, `interaction_type`, `normalize_polmat`, `fix_taps`, `axis`) or strings, each documented by its function.
- Results (C++): a function with one mandatory result returns it, e.g. `arma::vec ds = calc_delay_spread(delays, powers);`; its optional results are pointer arguments. Large results (e.g. `gbsm_draw`, `sos_val`, the `get_channels_*` functions) and the results of functions with several mandatory results are output arguments, so that the wrappers can write into MATLAB / numpy memory without a copy. Also returned are objects from readers, generators and factories (`arrayant`, `channel`, meshes, colormaps, path lists), a count, status or read object (also next to output arguments, e.g. `obj_file_read`, `qdant_read`, `hdf5_read_layout`), and a single computed scalar (e.g. `sos_calc_mse`).

**Frequencies**

- `center_freq`: carrier frequency in Hz; scalar, or one value per snapshot or per frequency entry `[n_freq]`. Stored in HDF5 files as `CenterFrequency` and in QDANT files as `center_freq`.
- `freq`: short name for a frequency in Hz; a value that must stay in GHz is named `freq_GHz`. For a frequency grid, `freq` holds the sample frequencies and `freq_new` the target frequencies.
- Acoustic frequency: in scalar (acoustic) mode every function takes the acoustic frequency in Hz. The ray-tracing core and the material model work with the EM-equivalent frequency `f · 299792458 / 343`, which has the same wavelength; the frequency columns of a material table (`fRef`, `resF`, `coiF`, `tfF`) are in EM-equivalent GHz (1 kHz = 0.87403 GHz).
- `spatial_freq`: spatial frequencies of a sum-of-sinusoids generator in 1/m, `[n_coeff, n_dim]`; limits `spatial_freq_min`, `spatial_freq_max`.

**Delays**

- `tap_spacing`: spacing of the delay taps in seconds; scalar.

**Positions and orientations**

- `tx_pos`, `rx_pos`, `irs_pos`, `orig_pos`: positions in GCS in meters; `tx_orientation`, `rx_orientation`, `irs_orientation`, `orig_orientation`: Euler angles (bank, tilt, heading) in rad. C++: `arma::Mat`, `[3, 1]` for one item (inputs also take `[1, 3]`), `[3, n]` for many (e.g. one per snapshot); MATLAB / Octave and Python: a 3-element vector for one item. Exceptions with one row per item, `[n, 3]`: the ray-tracing state, meshes, point clouds, QRT CIR and origin position lists (`qrt_file_init`, `qrt_file_parse`), and sum-of-sinusoids positions.

**Polarization**

- `polmat`: polarization transfer matrix, the 2×2 polarization matrix of TR 38.901 per path or ray, rows rx, columns tx, stored column-major; `[8, n]` real interleaved: rows 0–1 (rx V, tx V), 2–3 (rx H, tx V), 4–5 (rx V, tx H), 6–7 (rx H, tx H). In scalar (acoustic) mode `[2, n]` holds the real and imaginary part of the complex scalar coefficient. Labels such as `ReHV` name the rx polarization first. The HDF5 file stores it as `path_polarization`.

**Antenna arrays: elements, ports and port weights**

- Array: the whole `arrayant` object, an array of antenna or acoustic elements (an antenna array, a loudspeaker or a microphone array), with the radiation pattern (antennas) or directivity pattern (loudspeakers, microphones) of each element, the element positions and the port weights.
- Element: one radiator or sensor (antenna element, loudspeaker driver, microphone capsule), with a pattern, a position and one row of the port weights.
- Pattern: the direction-dependent field of an element, or of a port after `apply_port_weights`, sampled on an azimuth and elevation grid (`e_theta_re/im`, `e_phi_re/im`).
- Port: one column of the port weights; `apply_port_weights` turns ports into elements. `n_ports()` is the number of ports.
- Port weights: the fixed complex feed network that maps the elements to the ports, `[n_elements, n_ports]` (`port_weights_re`, `port_weights_im`); empty means identity (one port per element). It is not the mutual coupling between elements.

**Antenna array generators (TR 38.901 tuples)**

- Tuples: the panel parameters of TR 38.901 clause 7.3 are short arrays (C++: `arma::uvec` for counts and codes, `arma::vec` / `arma::mat` for real values). Trailing entries may be omitted; 0 (counts and codes) or NaN (real values) selects the default of that entry; more entries than defined throw. Entries in degrees or dB carry no `_deg` / `_dB` suffix (a decided deviation from the unit rule); the unit is given per entry here and in each function.
- `panel`: `[M, N, Mg, Ng]`: elements per column (along z), columns per panel (along y), panels per column, panels per row; default 1.
- `spacing`: `[d_V, d_H, d_gV, d_gH]` in wavelengths: vertical and horizontal element spacing, vertical and horizontal panel spacing (centre to centre); defaults 0.5, `d_V`, `M d_V`, `N d_H`.
- `ports`: `[M_P, N_P]`: ports (TXRUs) per column and per row of a panel and polarization; they divide `M` and `N`; default `[M, N]` (one port per element).
- `pol`: `[polarization, model]`: polarization 1 (vertical), 2 (H/V) or 3 (±45°), 4 to 6 reserved; model 1 or 2 (TR 38.901 Model-1, Model-2); default `[1, 1]`.
- `steer`: `[tilt, pan]` in degrees: electrical downtilt (positive down) and pan (positive towards +y) of the sub-array weights; one row for all panels or one row per panel; default 0.
- `element`: the element parameters of a generator type, unit per entry in its documentation, e.g. `[phi_3dB, theta_3dB, SLA_V, A_max, G_max]` in degrees, degrees, dB, dB and dBi for the TR element. Attenuations (`SLA_V`, `A_max`, `floor_dB`) are positive dB values below the peak, as in TR 38.901 (30 dB), and Inf removes the limit; an infinite beamwidth gives a flat cut.
- `beams`: `[2 or 3, n_beams]`: azimuth and elevation in degrees, weight (linear).

**Kernel selection**

- `use_kernel`: compute kernel of a function with more than one kernel; `0` = auto, `1` = GENERIC, `2` = AVX2, `3` = CUDA (only where a CUDA kernel exists). Auto selects per function, as stated in each function; where it is not the fastest kernel, the function says so. Throws if the requested kernel is unavailable.
- `gpu_id`: CUDA device ID; only read when `use_kernel` resolves to CUDA.
