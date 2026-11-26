# QRT path files

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/formats.html#qrt

| Section |
| --- |
| [Structure](#qrt-structure) |
| [Versions](#qrt-versions) |
| [Byte layout](#qrt-byte-layout) |
| [Stored values](#qrt-stored-values) |
| [Precision](#qrt-precision) |

A QRT file holds the propagation paths of a scene before antennas are applied: for each origin (a Tx position) and each CIR position, the interaction points of every path and its polarization transfer matrix (EM) or complex coefficient (scalar, e.g. acoustic), with the interaction (material) losses included. The paths come from a ray tracer or from the GBSM: [export_rays_qrt](../api/cpp/network-layouts-and-mobility.md#export_rays_qrt) writes the rays of a network layout. Any program can write the file with [qrt_file_init](../api/cpp/channel-data-and-processing.md#qrt_file_init) and [qrt_file_append](../api/cpp/channel-data-and-processing.md#qrt_file_append). The readers turn the paths into the inputs of the channel generators, which apply the antennas.

QRT and [HDF5 channel files](hdf5_channel.md) store different stages of a simulation:

| | QRT file | HDF5 channel file |
| --- | --- | --- |
| Content | Paths without antennas: interaction points and per-path coefficients | Channels with antennas: coefficients and delays per antenna pair, optional path data and metadata |
| Source | Ray tracers, the GBSM (`export_rays_qrt`) | Any channel: GBSM, layouts, channels made from QRT paths |
| Organization | Origins × CIRs; destinations group the CIRs (e.g. a track) | A grid of slots `(ix, iy, iz, iw)` with one channel each |
| Next step | Read the paths, then apply antennas ([get_channels_spherical](../api/cpp/channel-generators.md#get_channels_spherical)) | Frequency response, statistics, link-level simulation |

Related functions:

- C++ [qrt_file_init](../api/cpp/channel-data-and-processing.md#qrt_file_init), [qrt_file_append](../api/cpp/channel-data-and-processing.md#qrt_file_append), [qrt_paths_from_scatterers](../api/cpp/channel-data-and-processing.md#qrt_paths_from_scatterers), [qrt_file_parse](../api/cpp/channel-data-and-processing.md#qrt_file_parse), [qrt_file_read](../api/cpp/channel-data-and-processing.md#qrt_file_read), [qrt_file_read_raw](../api/cpp/channel-data-and-processing.md#qrt_file_read_raw), [qrt_read_cache_init](../api/cpp/channel-data-and-processing.md#qrt_read_cache_init), [export_rays_qrt](../api/cpp/network-layouts-and-mobility.md#export_rays_qrt)
- MATLAB / Octave [qrt_file_parse](../api/mex/channel-data-and-processing.md#qrt_file_parse), [qrt_file_read](../api/mex/channel-data-and-processing.md#qrt_file_read), [export_rays_qrt](../api/mex/network-layouts-and-mobility.md#export_rays_qrt)
- Python [qrt_file_parse](../api/python/channel-data-and-processing.md#qrt_file_parse), [qrt_file_read](../api/python/channel-data-and-processing.md#qrt_file_read), [export_rays_qrt](../api/python/network-layouts-and-mobility.md#export_rays_qrt)

<a id="qrt-structure"></a>
## Structure

- **Origin**: one end of the paths, usually a Tx. A file has `n_orig` origin slots, filled one after another by [qrt_file_append](../api/cpp/channel-data-and-processing.md#qrt_file_append); a slot not written yet has zero position and no paths.
- **CIR**: one position (and orientation) of the other end. Every origin has paths to the same `n_cir` CIR positions.
- **Destination**: a named run of consecutive CIRs, e.g. the snapshots of one Rx track; `cir_offset` is the index of its first CIR. A file without destinations reads as one destination `RX` that owns all CIRs.
- **Frequencies**: `n_freq` frequencies (1 to 127) in Hz; each path has one coefficient set per frequency.

The file is written in two steps: [qrt_file_init](../api/cpp/channel-data-and-processing.md#qrt_file_init) writes the header with the frequencies, the CIR positions, the destinations and the empty origin slots; each [qrt_file_append](../api/cpp/channel-data-and-processing.md#qrt_file_append) appends the data block of one origin at the end of the file and fills the origin's position, orientation and data offset in the header. With `uplink` the readers swap the ends, so the origin becomes the receiver.

<a id="qrt-versions"></a>
## Versions

| Version | ID string | Coefficients per path and frequency |
| ------- | --------- | ----------------------------------- |
| 5 | `#QRT-BINv05` | EM: polarization transfer matrix, 8 floats |
| 6 | `#QRT-BINv06` | Scalar (e.g. acoustic): one complex coefficient, 2 floats |
<a id="qrt-byte-layout"></a>
## Byte layout

Little-endian. `u8`, `u32`, `u64` are unsigned integers of 1, 4 and 8 bytes, `f32` is an IEEE single-precision float. Offsets are bytes from the start of the file. Arrays are written as columns: all x values, then all y values, and so on.

**Header**, at offset 0:

| Offset | Type | Name | Content |
| ------ | ---- | ---- | ------- |
| 0 | `char[11]` | ID | `#QRT-BINv05` or `#QRT-BINv06`, no terminating NUL |
| 11 | `u32` | `n_orig` | Number of origin slots, at least 1 |
| 15 | `u32` | `n_cir` | Number of CIRs per origin, may be 0 |
| 19 | `u32` | `n_dest` | Number of destinations, may be 0 |
| 23 | `u32` | `n_freq` | Number of frequencies, 1 to 127 |
| 27 | `f32[n_freq]` | `freq` | Frequencies in Hz |

**CIR table**, only if `n_cir > 0`:

| Type | Content |
| ---- | ------- |
| `u8` | Orientation format `fmt` of the CIRs (see below) |
| `f32[n_cir]` × 3 | Positions x, y, z in m |
| `f32[n_cir]` × 0 to 3 | Orientations in rad: bank if bit 2 of `fmt` is set, then tilt if bit 1, then heading if bit 0 |

**Destination table**:

| Type | Content |
| ---- | ------- |
| `u32[n_dest]` | `cir_offset`: 0-based index of the first CIR of each destination, non-decreasing, the first 0, at most `n_cir` |
| `n_dest` × (`u8` length, `char[length]`) | Destination names, at most 255 bytes, no terminating NUL |

**Origin table**:

| Type | Content |
| ---- | ------- |
| `u8` | Orientation format `fmt` of the origins; must be 7 (all three angles) in a file that [qrt_file_append](../api/cpp/channel-data-and-processing.md#qrt_file_append) extends |
| `f32[n_orig]` × 3 | Positions x, y, z in m; 0 for a slot not written yet |
| `f32[n_orig]` × 0 to 3 | Orientations in rad, as in the CIR table |
| `u64[n_orig]` | Offset of the data block of each origin; 0 = slot not written yet |

The orientation format, 0 to 7, is a bit mask of the stored angle columns: bit 2 bank (rotation about x), bit 1 tilt (about y), bit 0 heading (about z). A missing column is zero; 0 means no orientation, 7 all three angles.

**Origin data block**, at its offset, one per written origin:

| Type | Content |
| ---- | ------- |
| `u8`, `char[length]` | Length and name of the origin, at most 255 bytes |
| `u32` | Largest number of paths of the CIRs of this origin; informational, the readers do not use it |
| `u64[n_cir]` | Offset of the path data of each CIR |
| path data × `n_cir` | One record per CIR, back to back in CIR order; the readers reject records that overlap or share an offset |

**Path data** of one CIR with `n_path` paths, `n_int` the sum of the interaction counts:

| Type | Content |
| ---- | ------- |
| `u32` | `n_path`, may be 0 |
| `u8[n_path]` | Number of interaction points of each path, 0 = LOS, at most 255 |
| `f32[c · n_path · n_freq]` | Coefficients, `c` = 8 (version 5) or 2 (version 6): all paths of the first frequency, then the next; per path `c` consecutive values |
| `f32[3 · n_int]` | Interaction points x, y, z in m, point by point, the paths in order, each from the origin to the CIR |
| `u8[n_int]` | Interaction type code of each point, in the same order ([ray_state_update](../api/cpp/site-specific-simulation-tools.md#ray_state_update)) |

A version 5 coefficient set is the polarization transfer matrix in the order Re VV, Im VV, Re HV, Im HV, Re VH, Im VH, Re HH, Im HH (first letter rx, second tx; see `polmat` in the [Glossary](../api/cpp/general-usage-notes.md#glossary)); a version 6 set is the real and imaginary part of the complex coefficient.

<a id="qrt-stored-values"></a>
## Stored values

- Coefficients hold the interaction (material) losses and polarization changes only. The propagation loss and the delay of the path length are not stored: the readers compute the length of the polyline from the origin through the interaction points to the CIR position and add the loss ([qrt_file_read](../api/cpp/channel-data-and-processing.md#qrt_file_read), Notes).
- Paths of GBSM scatterers ([qrt_paths_from_scatterers](../api/cpp/channel-data-and-processing.md#qrt_paths_from_scatterers), [export_rays_qrt](../api/cpp/network-layouts-and-mobility.md#export_rays_qrt)) carry the code 193 at their scatterers; their LOS and ground-reflection paths look like ray-traced ones (no interaction, one reflection 129).
- The path length and the interaction counters of the C++ [path](../api/cpp/site-specific-simulation-tools.md#path) object are not stored.

<a id="qrt-precision"></a>
## Precision

All values are float32. Positions far from the origin of the coordinate system lose resolution, which shows as path-length and phase errors. Use local coordinates, shifted in x and y only (ground at z = 0). Measured path length errors: 2.7 µm, 0.31 mm and 1.0 mm for coordinates up to 150 m, 5 km and 20 km (phase errors of 0.003, 0.39 and 1.29 rad at 60 GHz).
