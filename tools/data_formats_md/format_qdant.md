/*!SECTION
QuaDRiGa Array Antenna Exchange Format (QDANT)
SECTION!*/

/*!SECTION_DESC
QDANT is an XML grammar for exchanging array-antenna pattern data. Tag names are
case-sensitive. A file contains a single `<qdant>` root element that may
include an optional `<layout>` and one or more `<arrayant>`
elements. A namespace is required on the root (with or without a prefix).<br><br>

Coordinate convention (for interpreting pattern data):
- Spherical (polar–spherical) angles theta (elevation) and phi (azimuth).
- Elevation theta: −90° (down) … 0° (horizon) … +90° (up).
- Azimuth phi: −180° (west) … −90° (south) … 0° (east) … +90° (north) … +180° (west); counting is anti-clockwise.
- Pattern components are resolved along unit vectors ê_theta and ê_phi (far-field, no ê_r component).

See related read / write functions in the API documentation for details on how to use this format:
- C++ QDANT read function (<a href="cpp_api.html#qdant_read">qdant_read</a>)
- C++ QDANT write function (<a href="cpp_api.html#.qdant_write">arrayant.qdant_write</a>)
- MATLAB / Octave QDANT read function (<a href="mex_api.html#arrayant_qdant_read">arrayant_qdant_read</a>)
- MATLAB / Octave QDANT write function (<a href="mex_api.html#arrayant_qdant_write">arrayant_qdant_write</a>)
- Python QDANT read function (<a href="python_api.html#qdant_read">qdant_read</a>)
- Python QDANT write function (<a href="python_api.html#qdant_write">qdant_write</a>)

SECTION_DESC!*/

/*!MD
# &lt;layout&gt;
Defines how multiple &lt;arrayant&gt; objects are arranged into an object array (optional)

## Content:
- One or more column vectors, separated by spaces; entries within a vector are comma-separated.
- Values: Integers that must match the id of corresponding `<arrayant>` elements.

## Examples:
- Linear list: 1 2 3 (three columns with one entry each).
- 3×2 matrix of object IDs (3 columns, each a column vector of length 2): 1,2 1,2 2,1
MD!*/

/*!MD
# &lt;arrayant&gt;
Describes a single array antenna

## Attributes:
- **`id`** (integer, required if multiple arrayants present)<br> 
  Unique values per file; if `<layout>` is used, each id referenced there must exist.

## Child elements:
- **`<name>`** (string, optional)<br> 
  Human-readable identifier.

- **`<CenterFrequency>`** (Hz, integer or float, optional)<br>
  Default 300 MHz if omitted. When omitted, element positions are interpreted in wavelength multiples (i.e., meters scaled by λ at 300 MHz).

- **`<NoElements>`** (integer, required for multi-element arrays)<br>
  Number of elements in this array.

- **`<ElementPosition>`** (optional)<br>
    One or more position vectors [x,y,z] in meters, relative to the phase center. Axis orientation: 
    x = east, y = north, z = up. Formatting: Vectors separated by spaces; within each vector, components 
    separated by commas. Requires `<NoElements>` to be defined and to match the number of vectors.
    Example: `0,0,0 0.05,0,0`

- **`<ElevationGrid>`** (degrees, required)<br>
    List of elevation samples (−90 … +90). Example: `-90 -45 0 45 90`

- **`<AzimuthGrid>`** (degrees, required)<br>
    List of azimuth samples (−180 … +180; 0 points east; anti-clockwise positive). Example: `-180 -90 0 90 180`

- **`<CouplingAbs>`** (optional; only with `<NoElements>`)<br>
    N×M absolute values of the coupling matrix. Column vectors separated by spaces; entries within a 
    column are separated by commas. Dimensions must be consistent with NoElements = N. 
    M is the number of antenna ports. Example (N=2): `1,0 0,1`

- **`<CouplingPhase>`** (optional; only with `<NoElements>` and `<CouplingAbs>`)<br>
    N×N phases (degrees) for the same matrix. Column vectors separated by spaces; entries within a 
    column are separated by commas. Dimensions must be consistent with NoElements = N.
    Example (N=2): `0,0 0,0`

- **`<EthetaMag>`** (optional)<br>
    Magnitude in [dB] of the electric field aligned with the ê − θ vector of the spherical coordinate 
    system. One line contains all azimuth values for one elevation angle. The order is given by 
    `AzimuthGrid` (columns) and `ElevationGrid` (rows). If `NoElements` is greater than 1, an additional 
    attribute `<EthetaMag el="[no]">` must be provided. In the first example, the first line contains the 
    values for the -90 degree elevation angle (down) and the last line contains the values for the 90 
    degree elevation (up). If there is no field in θ-direction, this XML-element can be omitted.

- **`<EthetaPhase>`** (optional; only with `<EthetaMag>`)<br>
    Phase in [degree] of the electric field aligned with the ê − θ vector. One line contains all 
    azimuth values for one elevation angle. The order is given by `AzimuthGrid` and `ElevationGrid`. 
    If `NoElements` is greater than 1, an additional attribute `<EthetaPhase el="[no]">` must be provided. 
    EthetaPhase cannot be defined without defining EthetaMag first. If all phases are 0, this 
    XML-element can be omitted.

- **`<EphiMag>`** (optional)<br>
    Magnitude in [dB] of the electric field aligned with the ê − φ vector of the spherical coordinate system.

- **`<EphiPhase>`** (optional; only with `<EphiMag>`)<br>
    Phase in [degree] of the electric field aligned with the ê − φ vector.

## Validation & Formatting Rules (at a glance):
- Case-sensitive tag names.
- `<qdant>` must be present; it may contain 0 or 1 `<layout>` and 1+ `<arrayant>`.
- If `<layout>` is present, every integer in it must match an existing `<arrayant id="…">`
- If `NoElements` is provided:<br> 
  (1) ElementPosition (if present) must provide exactly NoElements vectors.<br>
  (2) CouplingAbs and CouplingPhase (if present) must be N×N with N = NoElements.<br>
  (3) Pattern tags (Etheta*, Ephi*) must include el="1" … "N" for per-element data.
- Pattern blocks: Number of lines = length of `ElevationGrid`. Numbers per line = length of `AzimuthGrid`.
  Magnitudes in dB; phases in degrees.
- Omissions: If a component is everywhere zero, omit its *Mag (and its *Phase). If all phases are zero, *Phase may be omitted.

## Quick Reference:
| Element           | Required | Type / Units      | Notes                                                                |
| ----------------- | -------: | ----------------- | -------------------------------------------------------------------- |
| `qdant`           |      Yes | –                 | Root; carries namespace                                              |
| `layout`          |       No | int matrix (text) | Columns separated by spaces; within a column, IDs comma-separated    |
| `arrayant@id`     |    Cond. | int               | Unique; referenced by `layout`                                       |
| `name`            |       No | string            | Free text                                                            |
| `CenterFrequency` |       No | Hz                | Default 300 MHz if absent                                            |
| `NoElements`      |    Cond. | int               | Required for multi-element arrays                                    |
| `ElementPosition` |       No | meters            | Vectors `x,y,z`; vectors space-separated; components comma-separated |
| `ElevationGrid`   |  **Yes** | degrees           | −90 … +90                                                            |
| `AzimuthGrid`     |  **Yes** | degrees           | −180 … +180; 0° east; anti-clockwise positive                        |
| `CouplingAbs`     |       No | matrix            | N×M; columns space-sep; entries comma-sep                            |
| `CouplingPhase`   |       No | matrix (deg)      | Requires `CouplingAbs`; same shape                                   |
| `EthetaMag`       |    Cond. | dB grid           | Rows = elevations; cols = azimuths; `el="n"` if N>1                  |
| `EthetaPhase`     |    Cond. | deg grid          | Only with `EthetaMag`; omit if all zeros                             |
| `EphiMag`         |    Cond. | dB grid           | Rows = elevations; cols = azimuths; `el="n"` if N>1                  |
| `EphiPhase`       |    Cond. | deg grid          | Only with `EphiMag`; omit if all zeros                               |

## Example:
2-Element Cross-Polarized Array (minimal pattern values):
```
<?xml version="1.0" encoding="UTF-8"?>
<qdant xmlns="http://www.quadriga-channel-model.de">
  <layout>1</layout>
  <arrayant id="1">
    <name>Simple XPOL</name>
    <CenterFrequency>2600000000</CenterFrequency>
    <NoElements>2</NoElements>
    <ElementPosition>0,0,0 0,0,0</ElementPosition>

    <ElevationGrid>-90 0 90</ElevationGrid>
    <AzimuthGrid>-180 -90 0 90 180</AzimuthGrid>

    <CouplingAbs>1,0 0,1</CouplingAbs>
    <CouplingPhase>0,0 0,0</CouplingPhase>

    <EthetaMag el="1">
      0 0 0 0 0
      0 0 0 0 0
      0 0 0 0 0
    </EthetaMag>
    <EthetaPhase el="1">
      0 0 0 0 0
      0 0 0 0 0
      0 0 0 0 0
    </EthetaPhase>

    <EphiMag el="2">
      0 0 0 0 0
      0 0 0 0 0
      0 0 0 0 0
    </EphiMag>
    <EphiPhase el="2">
      0 0 0 0 0
      0 0 0 0 0
      0 0 0 0 0
    </EphiPhase>
  </arrayant>
</qdant>
```


MD!*/