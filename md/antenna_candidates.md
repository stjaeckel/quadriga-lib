# Antenna model candidates

Antenna models that quadriga-lib may add after the TR 38.901 antenna work ([tr38901_antennas.md](tr38901_antennas.md)). Nothing here is decided or scheduled. New types follow the extension rules of tr38901_antennas.md, section 2.2, so they can be added after v1.0 without breaking calls.

Sources: the MATLAB-only generators of QuaDRiGa (`qd_arrayant.generate_antenna`), and a survey of standards and implementations (Sionna, ns-3, MATLAB Phased Array Toolbox, ITU-R) on 2026-11-16. Parameter details marked "survey" were not checked against the standard.

## Next candidate: circular aperture

- Model: circular aperture with the field 2·J₁(x)/x, x = k·a·sin θ (TR 38.811 clause 6.4.1 satellite antenna; VSAT terminal in TR 38.821; QuaDRiGa `parabolic`; ns-3 `CircularApertureAntennaModel`). The field is real with sign changes at the nulls, so it fits the complex storage.
- Parameters: aperture radius or diameter, sidelobe floor, G_max (in `element`); polarisation V, H, LHCP, RHCP, dual CP (circular codes from the reserved `pol` range); hexagonal multibeam layout for satellites (through `beams`).
- Open: the sampling grid. A beam much narrower than 1° needs a grid that is dense around boresight only (QuaDRiGa builds such a grid); a candidate is a second `res` entry. TR 38.811 is not covered by the TR 38.901 inventory and needs its own.

## Further candidates

| Model | Source | What it adds | Implication |
| ----- | ------ | ------------ | ----------- |
| Sector elements | TR 36.814 Table A.2.1.1-2, TR 36.873 clause 7.1, ITU-R M.2135, ITU-R M.2101 (survey) | the Table 7.3-1 formula family with other constants; M.2101 is the IMT sharing-study element and array | mostly presets of the 3GPP `element`; TR 36.814 puts the electrical tilt into the element (θ + tilt) and uses SLA_V = A_max − 5 dB, which needs one more `element` entry or the rotated-pattern approximation |
| Circular polarisation | QuaDRiGa `rhcp-dipole`, `lhcp-dipole`, `lhcp-rhcp-dipole`; MATLAB crossed dipole | crossed dipoles with coupling [1; ±j]/√2 | a few lines; uses reserved `pol` codes |
| Cosine-power and Gaussian elements | MATLAB `phased.CosineAntennaElement`, `phased.GaussianAntennaElement`; ns-3 `CosineAntennaModel` (azimuth only); IEEE 802.11ad steerable model (survey) | analytic element shapes | small; new `element` variants or presets of the custom shape |
| Vehicle antennas | TR 37.885 clause 6.1.4 (survey); QuaDRiGa `vehicular` | V2X: bumper, rooftop and bus antennas, FR1 and FR2 | option 1 is panels with orientations (as the UT work); option 2 needs the TR's tabulated patterns as data |

## Not planned

- ITU-R envelope masks (F.1336, F.699, F.1245, S.465, S.580, S.672, S.1528 rec. 1.2/1.3): gain envelopes in dB without phase, nulls or polarisation; stored as complex fields their phase has no meaning, so they suit power-based interference studies, not coherent channels. Exception: S.1528 rec. 1.4 (circular Taylor aperture) has a true field and could become a variant of the circular aperture; S.1528 was revised in 08/2026, so the clause needs checking.
- Aliases without new physics: QuaDRiGa `patch` (custom 90° × 90°), `ula2`, `ula4`, `ula8`, `parametric` (the custom shape with raw parameters), `multi` (covered by the `3GPP` type with a custom pattern, `ports = 1` and `steer`).
- QuaDRiGa `testarray` (31 elements for angle and XPR estimation): useful only together with an angle-estimation tool.
