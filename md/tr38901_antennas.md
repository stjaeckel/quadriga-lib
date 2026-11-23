# TR 38.901 antenna model: plan and decisions

Goal (maintainer, 2026-11-16): every TR 38.901 antenna item in the inventory is "Yes". One wrapper for all generators (`generate_antenna`) is kept. The coverage state per TR item is in [tr38901_inventory.md](tr38901_inventory.md), section 3.3; this document holds the plan and the decisions for the remaining work (TR v19.5.0).

Order: the BS session (section 2, done) first, then the UT session (section 3).

Done before: #34 in 8551c68. Both interpolation kernels treat dx² + dy² < (8 eps)² as the pole of the local frame and choose the local azimuth so that the local and the global θ unit vectors agree; float and double, GENERIC and AVX2 agree there. For the sessions below this means: turned elements (Model-1 slants, the UT rolls of Annex A.6) are defined on their axis, where they carry the unrotated polarisation. Model-1 is tested against eq. 7.3-3a/b on and off the axis.

Not part of either session: the antenna imbalance NOTE under Table 7.3-2 (a per-port random loss drawn separately for UL and DL: a channel-side draw), the ITU-R M.2135 sector pattern of the CDL spatial filter (7.7.4.1, CDL work).

## 1. Names of the two antenna types

Decided 2026-11-16: the panel type keeps the name `3GPP` (or `3gpp`; BS and panel UTs such as the UT config A of Table 7.8-2 / 7.8-2A, M = N = 1, P = 2 with X-pol 0°/+90°, and Config A of Table 7.8-6A, M = 2, N = 1, P = 2); the UT session adds `3GPP-UT` (or `3gpp-ut`) for the handheld UT and `3GPP-CPE` (or `3gpp-cpe`, section 3). Other implementations name their TR antennas after the document ("38.901" and "tr38901" in Sionna, `ThreeGppAntennaModel` in ns-3, `3gpp-3d` / `3gpp-mmw` in QuaDRiGa) or after NR (MATLAB `phased.NRRectangularPanelArray`, size `[M N Mg Ng]`, the order of `panel`); none uses BS, UE or UT. Rejected: `3GPP-BS` (the panel is not BS-only), no prefix (loses the reference to the TR shapes), `nr-bs` / `nr-ue` (TR 38.901 is not NR-specific and says UT).

## 2. BS session

Status: done (the panel type `3GPP` with the tuples below, Model-2, sub-array weights, steering per panel, G_max, the ULA on the panel path, multibeam with `beams`, the C++ dispatcher `generate_antenna` behind thin MEX and Python wrappers). The library documentation (doc block of `generate_antenna`, the glossary) is the reference for the interface; this section keeps what the UT session builds on.

Encoding (shared by all types). Parameters are TR tuples as arrays, the same in C++, MEX and Python. Trailing entries may be omitted; inside a tuple, NaN (real tuples) or 0 (integer tuples `panel`, `ports`, `pol`, C++ `arma::uvec`) selects the default of that entry. Inputs a type does not use throw. The tuples carry degrees and dB without a `_deg` / `_dB` suffix; the glossary states the unit per entry. New shared names get their glossary entry in the same commit (`device_size`, `locations` with the UT session).

- `element`: one meaning per entry for every type, beamwidths first: TR types `[φ_3dB, θ_3dB, SLA_V, A_max, G_max]` (omitted: θ_3dB = φ_3dB, A_max = SLA_V; G_max NaN normalises to the directivity), custom and multibeam `[az_3dB, el_3dB, floor_dB]`. Giving `element` and `pattern` together throws.
- `pol` `[polarisation, model]`: 1 vertical, 2 H/V, 3 ±45°; model 1 (eq. 7.3-3) or 2 (eq. 7.3-4/7.3-5); 4 to 6 are reserved and throw. On a one-element element (TR, omni or a one-element `pattern`), Model-1 turns the given fields by ζ (`pol` = 1 uses the pattern as given) and Model-2 builds F_θ = √A′ cos ζ, F_φ = √A′ sin ζ from A′ = |F_θ|² + |F_φ|²; a `pattern` with more elements requires `pol` = 1.
- G_max: the TR constant per type (8 dBi for `3GPP`, 5.3 dBi for the UT); the Table 7.3-1 shape has a directivity of 9.83 dBi, the Table 7.3-2 shape 5.33 dBi.
- Element and port order: polarisation fastest, then the array indices (for the panel m, n, Mg, Ng).

Building blocks in `src/generate_antenna.cpp` for the UT types: `qd_tr38901_element` (the Table 7.3-1 / 7.3-2 shape from `[φ_3dB, θ_3dB, SLA_V, A_max, G_max]`, filled by `qd_tr38901_element_fill`; the Table 7.3-2 constants are tested), `qd_apply_polarization` (Model-1 / Model-2), `qd_arrayant_panel` (positions and coupling of a panel from one element; used by `3GPP-CPE`), and the dispatcher `generate_antenna`, which fills every default in one place.

Test scope and limits (plan rule; the SPECs stay): Catch2 tests the fill rules once, other validation one section per tuple; each wrapper tests each position once on one type, with one loop over omitted / `[]` / None; Python layout and dtype tests only for array inputs that are new. Limits: exact (0) for omitted versus 0 / NaN versus the explicit default and for alternative input forms; 1e-12 relative (double) between two code paths; 1e-10 dB for the Table 7.3-1/7.3-2 shapes; 0.01 dB for G_max against the directivity; float against double 1e-5 absolute for patterns, 1e-6 relative to the aperture for positions.

Extension rules (decided 2026-11-16). After v1.0 the generator interface stays stable, and further antenna types (see [antenna_candidates.md](antenna_candidates.md)) are added without breaking calls:

- Reserved codes: `pol` 4 to 6 throw in v1.0 and are reserved (e.g. for circular polarisation later); a code is never given a new meaning once it had one.
- Type names: a type tied to one document carries its prefix (`3GPP`, `3GPP-UT`, `3GPP-CPE`, later e.g. `ITU-M2101`); a generic physical model has a plain name (`omni`, `dipole`, later e.g. `circular-aperture`).
- Tuples grow only at the end: a new trailing entry has a default, so existing calls keep their meaning (e.g. `res` may later become `[step_deg, max_angle_deg]` for narrow-beam grids).
- `element` is the per-type hook: a new type defines its own `element` entries instead of a new position; new positions, if needed, are appended with a default.
- No new Python outputs later (a new Python output is a breaking change, SPEC_PYTHON_API): what a layout system later needs from the generator (e.g. the resolved parameter record that regenerates an antenna) becomes a member of the returned arrayant.
- Regenerating antennas from stored generator parameters (planned for the layout system) needs the resolved parameters, not the omitted ones; the C++ dispatcher fills every default in one place, so it can report them. A custom `pattern` cannot be stored as numbers and needs a file reference.

## 3. UT session

Status: not started; after the BS session.

Types: `3GPP-UT` (handheld UT) and `3GPP-CPE` (section 1 and below).

CPE: decided 2026-11-16, a type of its own, `3GPP-CPE` (or `3gpp-cpe`); `3GPP-UT` is the handheld UT. Beyond the geometry the TR gives nothing for the CPE (no element pattern, orientation, polarisation or calibration; the general rule of clause 7.3.0 places the UT "element/array" at the candidate locations), and the 9 locations are a 3 × 3 grid with 10 cm pitch. So `3GPP-CPE` runs on the panel code path:
- a 3 × 3 panel with horizontal pitch Y/2 and vertical pitch Z/2 of `device_size` = [X, Y, Z] (default 0 × 20 × 20 cm, independent of λ);
- `locations` in the TR numbering of Fig. 7.3-6 selects the elements (default all); several antennas per location come from `pol` (e.g. a cross-polarised pair);
- `pol`, `element` and `pattern` with the panel meaning; no `z_dir`;
- our reading, to be documented: elements face +x (broadside of the square, the panel convention) and Fig. 7.3-6 is seen from the front as Fig. 7.3-1, which fixes the numbers to ±y (the TR says only "side view").

Rejected: a device choice inside `3GPP-UT` (`pol`, `element` and `z_dir` would change meaning with the device, and a `device` input duplicates the type string); no CPE in the generator (the caller maps the TR numbering, the inventory row stays partial).

What the TR gives for the two device classes (clause 7.3.0):

| | Handheld | CPE |
| --- | --- | --- |
| Geometry | 8 locations: corners and edge centres of a 15 × 7 cm rectangle in the device plane (Fig. 7.3-2, top-down view) | 9 locations: corners, edge centres and centre of a 20 × 20 cm vertical square (Fig. 7.3-6, side view); a location may hold several antennas |
| Element pattern | Table 7.3-2 | not given |
| Orientation | each antenna along centre → location; z'' rules of Fig. 7.3-7 / 7.3-8 | not given (the figure has no arrows) |
| Polarisation | handheld model, eq. 7.3-6 to 7.3-8 | not given |
| Calibration | Tables 7.8-2A (Config B), A.6-1/-2 | none |

Content of `3GPP-UT`: handheld positions from the device size (15 × 7 × 0 cm, candidate locations 1 to 8 of Fig. 7.3-2), per-antenna orientation (α_u, β_u, γ_u), one or two field patterns per location (eq. 7.3-6 to 7.3-8; two patterns: γ_u − 45°), Tables A.6-1 and A.6-2, element pattern of Table 7.3-2. The panel UT configs (UT config A of Tables 7.8-2 / 7.8-2A: M = N = 1, P = 2, X-pol 0°/+90°; Config A of Table 7.8-6A: M = 2, N = 1, P = 2) are the panel type with an isotropic element and Model-2, not a UT type. The device orientation Ω_UT (clause 7.3.0: β = 45°, 90° etc.) is the receiver orientation of the channel functions, not a generator input.

Rotation: each handheld antenna is the reference element rotated by (bank, tilt, heading) = (γ_u, β_u, α_u) with `rotate_pattern`, pattern and polarisation together (usage 0 or 3), which is eq. 7.1-7/7.1-8 with eq. 7.3-6; the session implements the rotation, not eq. 7.3-7/7.3-8 literally (their denominator vanishes on the z'' axis; the NOTE after eq. 7.3-8 allows any computation of ψ_u, and the pole handling of 8551c68 covers the axis). To be checked against eq. 7.1-7/7.1-8 in a test.

Test cases: Table 7.8-2A metric 5 (F_u,θ' and F_u,φ' in the UT LCS for Config D, single polarisation, locations 1 to 8) is the reference fixture for `3GPP-UT`; Table 7.8-2A Configs B (locations (1, 7, 3, 5)) and C (dual polarisation, 1 to 8); Table 7.8-6C Configs A and B.

Parameters of `3GPP-UT`, in the encoding of section 2 (`3GPP-CPE`: `device_size` and `locations` as below, the rest as the panel type). Shared with the BS types: `element` (Table 7.3-2: 125°, 125°, 22.5 dB, 22.5 dB, 5.3 dBi), `pol` (1 = one field pattern per location, 2 = two field patterns), `pattern` (custom or isotropic element; Tables 7.8-6A Config B and 7.8-6B use isotropic, Tables 7.8-2A Configs B to D and 7.8-6C the Table 7.3-2 element, which is the tested default), `center_freq`, `res`. Own positions, each with one meaning:

- `device_size` `[X, Y, Z]` in m; default (0.15, 0.07, 0) for handheld, (0, 0.2, 0.2) for CPE.
- `locations`: TR location labels (handheld 1 to 8, CPE 1 to 9), passed unchanged in all three APIs (labels, not 0-based indices); any length; default all.
- `z_dir`: the choice of the z'' direction per antenna (+1 or −1, Fig. 7.3-7, see H1); default the orientations of Tables A.6-1 and A.6-2.

Tables A.6-1/-2: positions on the device edge (x' = ±7.5 or 0 cm, y' = ±3.5 or 0 cm), bearings −155°, −90°, −25°, 0°, 25°, 90°, 155°, 180° for antennas 1 to 8, tilt 0, roll 90° (one field pattern) or 45° (two).

Decisions from the plan review (2026-11-16):

- H1. `z_dir` is kept: one entry per antenna, +1 (default) the orientations of Tables A.6-1/-2, −1 the other z'' direction of Fig. 7.3-7, which is γ_u + 180° (one field pattern: 90° → −90°; two field patterns: 45° → −135°).
- H2. With two field patterns per location, the θ'' pattern and then the φ'' pattern, adjacent per location; `locations` keeps the caller's order (Table 7.8-2A gives (1, 7, 3, 5), and the port numbering follows it).
- H3. Positions and bearings from the geometry for every device size (bearing atan2(y', x'): 25.02° for the default size against 25° in A.6; tested against A.6 within 0.05°); a handheld `device_size` with Z ≠ 0 throws.

References found by a survey on 2026-11-16 (not read yet):
- No open implementation of the handheld UT model was found (Sionna, ns-3, MATLAB and QuaDRiGa have generic UT antennas), so there is no reference code to compare with; checks go against the TR text and Tables A.6-1/-2.
- Reported by the survey: a draft CR on NaN values of eq. 7.3-7/7.3-8. The handheld polarised model is CR 0032, in the TR since v19.2.0 (Annex B); v19.5.0 has the NOTE after eq. 7.3-8 that ψ_u may be computed via eq. 7.1-10 to avoid division by 0. Covered by implementing the rotation (see above).
- Descriptions of the model: [arXiv 2507.19266](https://arxiv.org/pdf/2507.19266) (Rel-19 channel model overview), [arXiv 2509.10357](https://arxiv.org/pdf/2509.10357) (realistic UE antennas in the 3GPP channel model), [arXiv 2602.07623](https://arxiv.org/html/2602.07623v1) (implementation tutorial), [Nokia white paper](https://nokia.com/asset/215108) (UE antenna and near-field blocking models).
