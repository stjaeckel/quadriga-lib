# TR 38.901 antenna model: plan and decisions

Goal (maintainer, 2026-11-16): every TR 38.901 antenna item in the inventory is "Yes". One wrapper for all generators (`generate_antenna`) is kept. The coverage state per TR item is in [tr38901_inventory.md](tr38901_inventory.md), section 3.3; this document holds the decisions behind the generators (TR v19.5.0).

Both sessions are done: BS (section 2) and UT (section 3).

Done before: #34 in 8551c68. Both interpolation kernels treat dx² + dy² < (8 eps)² as the pole of the local frame and choose the local azimuth so that the local and the global θ unit vectors agree; float and double, GENERIC and AVX2 agree there. For the generators this means: turned elements (Model-1 slants, the UT rolls of Annex A.6) are defined on their axis, where they carry the unrotated polarisation. Model-1 is tested against eq. 7.3-3a/b on and off the axis.

Not part of either session: the antenna imbalance NOTE under Table 7.3-2 (a per-port random loss drawn separately for UL and DL: a channel-side draw), the ITU-R M.2135 sector pattern of the CDL spatial filter (7.7.4.1, CDL work).

## 1. Names of the two antenna types

Decided 2026-11-16: the panel type keeps the name `3GPP` (or `3gpp`; BS and panel UTs such as the UT config A of Table 7.8-2 / 7.8-2A, M = N = 1, P = 2 with X-pol 0°/+90°, and Config A of Table 7.8-6A, M = 2, N = 1, P = 2); the UT session adds `3GPP-UT` (or `3gpp-ut`) for the handheld UT and `3GPP-CPE` (or `3gpp-cpe`, section 3). Other implementations name their TR antennas after the document ("38.901" and "tr38901" in Sionna, `ThreeGppAntennaModel` in ns-3, `3gpp-3d` / `3gpp-mmw` in QuaDRiGa) or after NR (MATLAB `phased.NRRectangularPanelArray`, size `[M N Mg Ng]`, the order of `panel`); none uses BS, UE or UT. Rejected: `3GPP-BS` (the panel is not BS-only), no prefix (loses the reference to the TR shapes), `nr-bs` / `nr-ue` (TR 38.901 is not NR-specific and says UT).

## 2. BS session

Status: done (the panel type `3GPP` with the tuples below, Model-2, sub-array weights, steering per panel, G_max, the ULA on the panel path, multibeam with `beams`, the C++ dispatcher `generate_antenna` behind thin MEX and Python wrappers). The library documentation (doc block of `generate_antenna`, the glossary) is the reference for the interface; this section keeps what the UT session builds on.

Encoding (shared by all types). Parameters are TR tuples as arrays, the same in C++, MEX and Python. Trailing entries may be omitted; inside a tuple, NaN (real tuples) or 0 (integer tuples `panel`, `ports`, `pol`, C++ `arma::uvec`) selects the default of that entry. Inputs a type does not use throw. The tuples carry degrees and dB without a `_deg` / `_dB` suffix; the glossary states the unit per entry. New shared names get their glossary entry in the same commit.

- `element`: one meaning per entry for every type, beamwidths first: TR types `[φ_3dB, θ_3dB, SLA_V, A_max, G_max]` (omitted: θ_3dB = φ_3dB, A_max = SLA_V; G_max NaN normalises to the directivity), custom and multibeam `[az_3dB, el_3dB, floor_dB]`. Giving `element` and `pattern` together throws.
- `pol` `[polarisation, model]`: 1 vertical, 2 H/V, 3 ±45°; model 1 (eq. 7.3-3) or 2 (eq. 7.3-4/7.3-5); 4 to 6 are reserved and throw. On a one-element element (TR, omni or a one-element `pattern`), Model-1 turns the given fields by ζ (`pol` = 1 uses the pattern as given) and Model-2 builds F_θ = √A′ cos ζ, F_φ = √A′ sin ζ from A′ = |F_θ|² + |F_φ|²; a `pattern` with more elements requires `pol` = 1.
- G_max: the TR constant per type (8 dBi for `3GPP`, 5.3 dBi for the UT); the Table 7.3-1 shape has a directivity of 9.83 dBi, the Table 7.3-2 shape 5.33 dBi.
- Element and port order: polarisation fastest, then the array indices (for the panel m, n, Mg, Ng).

Test scope and limits (plan rule; the SPECs stay): Catch2 tests the fill rules once, other validation one section per tuple; each wrapper tests each position once on one type, with one loop over omitted / `[]` / None; Python layout and dtype tests only for array inputs that are new. Limits: exact (0) for omitted versus 0 / NaN versus the explicit default and for alternative input forms; 1e-12 relative (double) between two code paths; 1e-10 dB for the Table 7.3-1/7.3-2 shapes; 0.01 dB for G_max against the directivity; float against double 1e-5 absolute for patterns, 1e-6 relative to the aperture for positions.

Extension rules (decided 2026-11-16). After v1.0 the generator interface stays stable, and further antenna types (see [antenna_candidates.md](antenna_candidates.md)) are added without breaking calls:

- Reserved codes: `pol` 4 to 6 throw in v1.0 and are reserved (e.g. for circular polarisation later); a code is never given a new meaning once it had one.
- Type names: a type tied to one document carries its prefix (`3GPP`, `3GPP-UT`, `3GPP-CPE`, later e.g. `ITU-M2101`); a generic physical model has a plain name (`omni`, `dipole`, later e.g. `circular-aperture`).
- Tuples grow only at the end: a new trailing entry has a default, so existing calls keep their meaning (e.g. `res` may later become `[step_deg, max_angle_deg]` for narrow-beam grids).
- `element` is the per-type hook: a new type defines its own `element` entries instead of a new position; new positions, if needed, are appended with a default.
- No new Python outputs later (a new Python output is a breaking change, SPEC_PYTHON_API): what a layout system later needs from the generator (e.g. the resolved parameter record that regenerates an antenna) becomes a member of the returned arrayant.
- Regenerating antennas from stored generator parameters (planned for the layout system) needs the resolved parameters, not the omitted ones; the C++ dispatcher fills every default in one place, so it can report them. A custom `pattern` cannot be stored as numbers and needs a file reference.

## 3. UT session

Status: done (the types `3GPP-UT` and `3GPP-CPE` with `device_size` and `locations`). The library documentation is the reference for the interface and the inventory for the coverage; this section keeps the decisions behind them.

- CPE: a type of its own. The TR gives only the geometry, so the rest is our reading: elements face +x, Fig. 7.3-6 is seen from the front as Fig. 7.3-1, and the default element is the UE element of TR 38.803 Table 5.2.3.3-1 (90°, 25 dB, 5 dBi; directivity 7.44 dBi), chosen as the realistic patch-like element. `element` covers the others, e.g. `[Inf, Inf, NaN, NaN, 0]` for isotropic.
- No `z_dir` input (decided 2026-11-23): the other z'' direction of Fig. 7.3-7 is a roll by 180° about the boresight, which negates the field of every `element` tuple (a custom pattern without that symmetry is turned before it is passed). A caller negates the port weight of that antenna.
- `locations` keeps the caller's order and allows repeats, so that antennas can be generated first and edited afterwards. The labels are 1-based in all APIs because they are names.
- `pol` on the handheld: `pol` 1 and 2 with their default model are the TR model (Model-2 gives the two reference fields; with `pol` 1 the default keeps the fields of a custom `pattern`). Model-1 for two patterns and `pol` 3 are accepted as extensions for research use. A two-element `pattern` is read as the TR pair of θ'' and φ'' field patterns and gets the roll of 45°; a ±45° pair is passed in that basis.
- Rotation: each antenna is the reference element rotated by (γ_u, β_u, α_u) with `rotate_pattern`, not eq. 7.3-7/7.3-8 literally (their denominator vanishes on the z'' axis). Bearings come from the geometry for every device size.
- Reference data: `Calibration_UTant` in R1-2506406 holds F_θ' and F_φ' of metric 5 for all 8 antennas on a 5° grid; the test uses 8 directions of it.
