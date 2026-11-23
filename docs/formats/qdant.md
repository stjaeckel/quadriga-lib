# QuaDRiGa Array Antenna Exchange Format (QDANT)

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/formats.html#qdant

| Section | Topics |
| --- | --- |
| [\<layout>](#layout) | [Content](#content-2) · [Examples](#examples) |
| [\<arrayant>](#arrayant) | [Attributes](#attributes) · [Child elements](#child-elements) · [Validation & Formatting Rules (at a glance)](#validation-formatting-rules-at-a-glance) · [Quick Reference](#quick-reference) · [Tag names of earlier versions](#tag-names-of-earlier-versions) · [Example](#example) |

QDANT is an XML grammar for exchanging antenna array pattern data. Tag names are case-sensitive. A file contains a single `<qdant>` root element that may include an optional `<layout>` and one or more `<arrayant>` elements. A namespace is required on the root (with or without a prefix).

Coordinate convention (for interpreting pattern data):

- Spherical (polar-spherical) angles theta (elevation) and phi (azimuth).
- Elevation theta: −90° (down) … 0° (horizon) … +90° (up).
- Azimuth phi: −180° (west) … −90° (south) … 0° (east) … +90° (north) … +180° (west); counting is anti-clockwise.
- Pattern components are resolved along unit vectors ê_theta and ê_phi (far-field, no ê_r component).

See related read / write functions in the API documentation for details on how to use this format:

- C++ QDANT read function ([qdant_read](../api/cpp/antennas-transducers-and-patterns.md#qdant_read))
- C++ QDANT write function ([qdant_write](../api/cpp/antennas-transducers-and-patterns.md#qdant_write))
- MATLAB / Octave QDANT read function ([qdant_read](../api/mex/antennas-transducers-and-patterns.md#qdant_read))
- MATLAB / Octave QDANT write function ([qdant_write](../api/mex/antennas-transducers-and-patterns.md#qdant_write))
- Python QDANT read function ([qdant_read](../api/python/antennas-transducers-and-patterns.md#qdant_read))
- Python QDANT write function ([qdant_write](../api/python/antennas-transducers-and-patterns.md#qdant_write))

<a id="layout"></a>
## \<layout>

Defines how multiple \<arrayant> objects are arranged into an object array (optional)

<a id="content-2"></a>
### Content

- One or more column vectors, separated by spaces; entries within a vector are comma-separated.
- Values: Integers that must match the id of corresponding `<arrayant>` elements.

<a id="examples"></a>
### Examples

- Linear list: 1 2 3 (three columns with one entry each).
- 3×2 matrix of object IDs (3 columns, each a column vector of length 2): 1,2 1,2 2,1

<a id="arrayant"></a>
## \<arrayant>

Describes a single antenna array

<a id="attributes"></a>
### Attributes

- **`id`** (integer, required if multiple arrayants present)\
  Unique values per file; if `<layout>` is used, each id referenced there must exist.

<a id="child-elements"></a>
### Child elements

- **`<name>`** (string, optional)\
  Human-readable identifier.

- **`<center_freq>`** (Hz, integer or float, optional)\
  Default 300 MHz if omitted. When omitted, element positions are interpreted in wavelength multiples (i.e., meters scaled by λ at 300 MHz).

- **`<n_elements>`** (integer, required for multi-element arrays)\
  Number of elements in this array.

- **`<element_pos>`** (optional)\
  One or more position vectors \[x,y,z] in meters, relative to the phase center. Axis orientation: x = east, y = north, z = up. Formatting: Vectors separated by spaces; within each vector, components separated by commas. Requires `<n_elements>` to be defined and to match the number of vectors. Example: `0,0,0 0.05,0,0`

- **`<elevation_grid>`** (degrees, required)\
  List of elevation samples (−90 … +90). Example: `-90 -45 0 45 90`

- **`<azimuth_grid>`** (degrees, required)\
  List of azimuth samples (−180 … +180; 0 points east; anti-clockwise positive). Example: `-180 -90 0 90 180`

- **`<port_weights_abs>`** (optional; only with `<n_elements>`)\
  N×M absolute values of the port weights. Column vectors separated by spaces; entries within a column are separated by commas. Dimensions must be consistent with n_elements = N. M is the number of antenna ports. Example (N=2): `1,0 0,1`

- **`<port_weights_phase>`** (optional; only with `<n_elements>` and `<port_weights_abs>`)\
  N×M phases (degrees) of the port weights. Column vectors separated by spaces; entries within a column are separated by commas. Dimensions must be consistent with n_elements = N. Example (N=2): `0,0 0,0`

- **`<e_theta_mag>`** (optional)\
  Magnitude in \[dB] of the electric field aligned with the ê − θ vector of the spherical coordinate system. One line contains all azimuth values for one elevation angle. The order is given by `azimuth_grid` (columns) and `elevation_grid` (rows). If `n_elements` is greater than 1, an additional attribute `<e_theta_mag el="[no]">` must be provided. In the first example, the first line contains the values for the -90 degree elevation angle (down) and the last line contains the values for the 90 degree elevation (up). If there is no field in θ-direction, this XML-element can be omitted.

- **`<e_theta_phase>`** (optional; only with `<e_theta_mag>`)\
  Phase in \[degree] of the electric field aligned with the ê − θ vector. One line contains all azimuth values for one elevation angle. The order is given by `azimuth_grid` and `elevation_grid`. If `n_elements` is greater than 1, an additional attribute `<e_theta_phase el="[no]">` must be provided. e_theta_phase cannot be defined without defining e_theta_mag first. If all phases are 0, this XML-element can be omitted.

- **`<e_phi_mag>`** (optional)\
  Magnitude in \[dB] of the electric field aligned with the ê − φ vector of the spherical coordinate system.

- **`<e_phi_phase>`** (optional; only with `<e_phi_mag>`)\
  Phase in \[degree] of the electric field aligned with the ê − φ vector.

<a id="validation-formatting-rules-at-a-glance"></a>
### Validation & Formatting Rules (at a glance)

- Case-sensitive tag names.
- `<qdant>` must be present; it may contain 0 or 1 `<layout>` and 1+ `<arrayant>`.
- If `<layout>` is present, every integer in it must match an existing `<arrayant id="…">`
- If `n_elements` is provided:\
  (1) element_pos (if present) must provide exactly n_elements vectors.\
  (2) port_weights_abs and port_weights_phase (if present) must be N×M with N = n_elements.\
  (3) Pattern tags (e_theta_*, e_phi_*) must include el="1" … "N" for per-element data.
- Pattern blocks: Number of lines = length of `elevation_grid`. Numbers per line = length of `azimuth_grid`. Magnitudes in dB; phases in degrees.
- Omissions: If a component is everywhere zero, omit its *_mag (and its *_phase). If all phases are zero, *_phase may be omitted.
- Zeros and levels: a magnitude of `-inf` dB is an exact zero; there is no lower limit for other values. quadriga-lib writes every non-zero sample, and an element that is zero in both components as an all-`-inf` `e_theta_mag` node.
- Precision written by quadriga-lib (single-precision equivalent relative to the element peak): magnitudes (dB) and phases (degree) get the fewest decimals (0 to 6) that keep the error of each complex sample at or below 6e-8 times the peak amplitude of its element (over both components); magnitudes get 6 decimals within 20 dB of the peak and one fewer per 20 dB below it. Samples far below the peak are therefore less accurate relative to their own value (about 6e-4 at -80 dB, 6e-2 at -120 dB). Positions and port weights as the shortest text that reads back as the same float (at most 9 significant digits), grids rounded to 0.001 degree. A phase tag is omitted if all phases round to 0.
- `n_elements` must be backed by data: `element_pos`, `port_weights_abs`, or pattern tags with `el="N"`.
- Reader limits of quadriga-lib, over all entries read (a repeated ID counts once): elements without pattern tags read as zeros, and these zero patterns are limited to the larger of 2^24 values and 1 value per file byte; the identity port weights of a file without `port_weights_abs` (at most 4096 elements) are limited to the larger of 2^24 values and 128 values per file byte.

<a id="quick-reference"></a>
### Quick Reference

| Element              | Required | Type / Units      | Notes                                                                |
| -------------------- | -------: | ----------------- | -------------------------------------------------------------------- |
| `qdant`              |      Yes | –                 | Root; carries namespace                                              |
| `layout`             |       No | int matrix (text) | Columns separated by spaces; within a column, IDs comma-separated    |
| `arrayant@id`        |    Cond. | int               | Unique; referenced by `layout`                                       |
| `name`               |       No | string            | Free text                                                            |
| `center_freq`        |       No | Hz                | Default 300 MHz if absent                                            |
| `n_elements`         |    Cond. | int               | Required for multi-element arrays                                    |
| `element_pos`        |       No | meters            | Vectors `x,y,z`; vectors space-separated; components comma-separated |
| `elevation_grid`     |  **Yes** | degrees           | −90 … +90                                                            |
| `azimuth_grid`       |  **Yes** | degrees           | −180 … +180; 0° east; anti-clockwise positive                        |
| `port_weights_abs`   |       No | matrix            | N×M; columns space-sep; entries comma-sep                            |
| `port_weights_phase` |       No | matrix (deg)      | Requires `port_weights_abs`; same shape                              |
| `e_theta_mag`        |    Cond. | dB grid           | Rows = elevations; cols = azimuths; `el="n"` if N>1                  |
| `e_theta_phase`      |    Cond. | deg grid          | Only with `e_theta_mag`; omit if all zeros                           |
| `e_phi_mag`          |    Cond. | dB grid           | Rows = elevations; cols = azimuths; `el="n"` if N>1                  |
| `e_phi_phase`        |    Cond. | deg grid          | Only with `e_phi_mag`; omit if all zeros                             |

<a id="tag-names-of-earlier-versions"></a>
### Tag names of earlier versions

Files written by earlier versions of quadriga-lib stay valid: the reader accepts the earlier tag names below, and the writer uses only the current names. Updating a file leaves its other entries unchanged, so one file can hold both. Versions that know only the earlier names cannot read files with the current names. An entry that contains both names of one tag is an error. The file stores grids and phases in degrees and magnitudes in dB, also where a tag has the same name as an `arrayant` member in radians or linear scale.

| Earlier name      | Current name         |
| ----------------- | -------------------- |
| `CenterFrequency` | `center_freq`        |
| `NoElements`      | `n_elements`         |
| `ElementPosition` | `element_pos`        |
| `ElevationGrid`   | `elevation_grid`     |
| `AzimuthGrid`     | `azimuth_grid`       |
| `CouplingAbs`     | `port_weights_abs`   |
| `CouplingPhase`   | `port_weights_phase` |
| `EthetaMag`       | `e_theta_mag`        |
| `EthetaPhase`     | `e_theta_phase`      |
| `EphiMag`         | `e_phi_mag`          |
| `EphiPhase`       | `e_phi_phase`        |

<a id="example"></a>
### Example

2-Element Cross-Polarized Array (minimal pattern values):

```
<?xml version="1.0" encoding="UTF-8"?>
<qdant xmlns="http://www.quadriga-channel-model.de">
  <layout>1</layout>
  <arrayant id="1">
    <name>Simple XPOL</name>
    <center_freq>2600000000</center_freq>
    <n_elements>2</n_elements>
    <element_pos>0,0,0 0,0,0</element_pos>

    <elevation_grid>-90 0 90</elevation_grid>
    <azimuth_grid>-180 -90 0 90 180</azimuth_grid>

    <port_weights_abs>1,0 0,1</port_weights_abs>
    <port_weights_phase>0,0 0,0</port_weights_phase>

    <e_theta_mag el="1">
      0 0 0 0 0
      0 0 0 0 0
      0 0 0 0 0
    </e_theta_mag>
    <e_theta_phase el="1">
      0 0 0 0 0
      0 0 0 0 0
      0 0 0 0 0
    </e_theta_phase>

    <e_phi_mag el="2">
      0 0 0 0 0
      0 0 0 0 0
      0 0 0 0 0
    </e_phi_mag>
    <e_phi_phase el="2">
      0 0 0 0 0
      0 0 0 0 0
      0 0 0 0 0
    </e_phi_phase>
  </arrayant>
</qdant>
```
