# TR 38.901 antenna model: plan and decisions

Goal (maintainer, 2026-11-16): every TR 38.901 antenna item in the inventory is "Yes". One wrapper for all generators (`arrayant_generate`) is kept. The coverage state per TR item is in [tr38901_inventory.md](tr38901_inventory.md), section 3.3; this document holds the plan and the decisions for the remaining work (TR v19.5.0).

Order: the BS session (section 2) first, then the UT session (section 3).

Done before: #34 in 6938691. Both interpolation kernels treat dx² + dy² < (8 eps)² as the pole of the local frame and choose the local azimuth so that the local and the global θ unit vectors agree; float and double, GENERIC and AVX2 agree there. For the sessions below this means: turned elements (Model-1 slants, the UT rolls of Annex A.6) are defined on their axis, where they carry the unrotated polarisation. A test of Model-1 against eq. 7.3-3a/b, on and off the axis, exists (`test_arrayant_generate.cpp`, "turned polarizations follow TR 38.901 Model-1"); the inventory row still says "off-boresight values are not" tested.

Not part of either session: the antenna imbalance NOTE under Table 7.3-2 (a per-port random loss drawn separately for UL and DL: a channel-side draw), the ITU-R M.2135 sector pattern of the CDL spatial filter (7.7.4.1, CDL work).

## 1. Names of the two antenna types

Decided 2026-11-16: the panel type keeps the name `3GPP` (or `3gpp`; BS and panel UTs such as the UT config A of Table 7.8-2 / 7.8-2A, M = N = 1, P = 2 with X-pol 0°/+90°, and Config A of Table 7.8-6A, M = 2, N = 1, P = 2); the UT session adds `3GPP-UT` (or `3gpp-ut`) for the handheld UT and `3GPP-CPE` (or `3gpp-cpe`, section 3). Other implementations name their TR antennas after the document ("38.901" and "tr38901" in Sionna, `ThreeGppAntennaModel` in ns-3, `3gpp-3d` / `3gpp-mmw` in QuaDRiGa) or after NR (MATLAB `phased.NRRectangularPanelArray`, size `[M N Mg Ng]`, the order of `panel`); none uses BS, UE or UT. Rejected: `3GPP-BS` (the panel is not BS-only), no prefix (loses the reference to the TR shapes), `nr-bs` / `nr-ue` (TR 38.901 is not NR-specific and says UT).

## 2. BS session

Status: not started; all decisions taken.

### 2.1 Gaps

Measured at 86c5b05; `generate_arrayant_3GPP` unchanged since, except the pole handling of 6938691.

- One `spacing` for d_H and d_V. Tables 7.8-6A and 7.8-6B (UMi, 7 GHz) use 24 × 32 elements with d_H = 0.5λ, d_V = 0.7λ.
- No TXRU sub-arrays: `pol` 4 to 6 combine all M elements of a column; Tables 7.8-6A to 7.8-6C give (M, N, P, M_g, N_g; M_P, N_P), e.g. M = 24 on M_P = 8, and Table 7.8-2A gives M_p, N_p per config.
- No horizontal combining or panning: Table 7.8-2 config 1 maps all 16 elements per polarisation and panel to one port with "panning angles of the two subarrays: (0, 0)".
- `dgv` and `dgh` default to 0.5λ, valid only for M = N = 1; with M = 4, Mg = 2 the panels overlap silently.
- NaN and Inf are not rejected (NaN `spacing` or Inf `dgv` give NaN positions, NaN `tilt` with `pol = 4` a broken pattern); negative values select the default silently.
- Model-2 (eq. 7.3-4, 7.3-5) has no option; Tables 7.8-2A and 7.8-6A to 7.8-6C use it, and Table 7.8-2 specifies "Model-2 in TR36.873".
- A 3GPP call with `az_3dB` gets a custom-shape element built by the wrappers, not a TR-shaped one; `pol` 2 and 3 are not applied to a custom element (the wrappers build the ULA polarisations themselves).

Correct already: the element pattern equals Table 7.3-1 within 2e-14 dB; element numbering matches Fig. 7.3-1 (m upwards along z, n along +y); ζ = +45° is element 0; panel spacings are centre-to-centre; port weights equal eq. 7.3-1 up to a common phase (the tilt sign is pinned by a test).

Wrappers today: MEX `arrayant_generate(type, …)` has 16 positional inputs shared by all types, position 10 is `tilt` (3GPP) or the beam matrix `dir` (multibeam), `az_3dB = 0` selects the 3GPP element. Python `arrayant.generate` has 21 named arguments; its documentation says keyword-only, but the registration has no `py::kw_only()`. MEX `dir` versus Python `beam_az`, `beam_el`, `beam_weight`; Python `apply_weights` has no MEX counterpart. The Python documentation gives the `az_3dB` default for `3gpp` as 67; the code default is 0, which selects the TR element.

### 2.2 Decisions

Parameters are TR tuples as arrays, the same in C++, MEX and Python. Trailing entries may be omitted; inside a tuple, NaN (real tuples) or 0 (integer tuples: `panel`, `ports`, `pol`) selects the default of that entry (the #77 rule for defaults that depend on other inputs; see G6):

| Tuple | Entries | Omitted entries |
| ----- | ------- | --------------- |
| `panel` | `[M, N, Mg, Ng]` | 1 |
| `spacing` | `[d_V, d_H, d_gV, d_gH]` in λ | d_H = d_V; d_gV = M·d_V, d_gH = N·d_H |
| `ports` | `[M_P, N_P]` | M_P = M, N_P = N (no combining); M_P must divide M, N_P must divide N |
| `pol` | `[polarisation, model]` | model 1 |
| `steer` | `[tilt, pan]` in degrees, one row or one row per panel (G5) | 0 |
| `element` | per type, see below | the TR table of the type |

Examples: `panel = 8` is an 8 × 1 column, `ports = 1` combines every column into one port, `ports = [1, 1]` gives one port per polarisation and panel (Table 7.8-2 config 1). A Table 7.8-6A row reads `panel = [24, 32], ports = [8, 32], spacing = [0.7, 0.5], pol = [3, 2]` (the ±45° slant is an assumption: Table 7.8-6A gives P = 2 and Model-2 but no slant; ±45° is stated for BS config 1 of Table 7.8-2). The order is vertical before horizontal and panel elements before panels, as Tables 7.8-6A to 7.8-6C (M, N, P, M_g, N_g; M_P, N_P); clause 7.3.0 and Table 7.8-2A write panels first, which would not allow `panel = M`.

- Panel spacing. Default M·d_V and N·d_H (the densest layout without overlap, one uniform array; the TR gives no default). A spacing given by the caller may overlap or coincide, down to 0 (co-located elements, e.g. for testing MIMO algorithms that degenerate for them). Negative values, Inf, and NaN other than as the default marker throw.
- Polarisation. `pol` is 1 (vertical), 2 (H/V) or 3 (±45°); `pol` 4 to 6 are removed (combining only through `ports`). The model is 1 (Model-1, eq. 7.3-3) or 2 (Model-2, eq. 7.3-4/7.3-5). The 3GPP type applies `pol` and the model to a custom element too (needed for the ULA and for the panel UT configs of Tables 7.8-2 and 7.8-6A: Model-2 with an isotropic element).
- Steering. `steer = [tilt, pan]` replaces `tilt`; `tilt` is the electrical downtilt of eq. 7.3-1 (θ_etilt = 90° + tilt); the horizontal weights for `pan` follow eq. 7.3-1 by analogy along y with d_H (a reading, not TR text). Multibeam gets its own position `beams` `[3, n_beams]` (az; el; weight). No position is shared between two quantities; `element`, `pol` and the defaults of `device_size` are defined per type (listed per type in the documentation).
- Element. One position `element` whose entries depend on the type: custom and multibeam `[az_3dB, el_3dB, rear_gain]` (the library's shape, normalised to the directivity); 3GPP and UT the TR constants (Table 7.3-1, Table 7.3-2). A custom-shape element on a 3GPP array stays available through `pattern`. The entries mix degrees, dB and linear values, so the documentation gives the unit per entry.
- G_max. The TR constant: 8 dBi (3GPP), 5.3 dBi (UT); NaN selects the normalisation to the directivity. Measured at 86c5b05: the Table 7.3-1 shape has a directivity of 9.83 dBi, so the TR BS element radiates 1.83 dB less than it receives; the Table 7.3-2 shape has 5.33 dBi. With other beamwidths and the default G_max the gain can exceed the directivity, which NaN avoids.
- ULA. Kept, on the 3GPP code path with every 3GPP tuple and the omni element as its default element. `panel = 8` is a vertical column; a horizontal ULA is `panel = [1, 8]`.
- C++. `generate_arrayant_3GPP` and the other generators take the tuples as containers by `const &` with a default (e.g. `const arma::uvec &panel = {}`, empty = all defaults), so that the three APIs read the same. This is a deliberate exception to "optional inputs by pointer" (`md/SPEC_CPP_API.md`); the BS session writes the exception into the SPEC. #69 has a comment (2026-11-16) so that the cleanup does not convert these arguments; #48 excludes the generators.
- Wrappers. One dispatcher `arrayant_generate` for all generators (a documented exception to #69's one wrapper per C++ function family). Positional calls of both wrappers change; Python names and order equal MEX. The frequency input is `center_freq` in all three APIs (#70; both wrappers still say `freq`). The NaN / Inf / negative rule replaces every silent range fix of the generators, also `center_freq <= 0` → default and the clamping of `res`.
- Glossary. The new shared names (`panel`, `spacing`, `ports`, `steer`, `element`, `beams`, `device_size`, `locations`) get glossary entries in the same commit (SPEC_CPP_API). The tuples carry degrees and dB without a `_deg` / `_dB` suffix; the glossary states the unit per entry (a deviation from the suffix convention, decided with the tuples).
- Test files follow the SPEC naming in the rewrite: `tests/mex_tests/test_generate_arrayant.m` becomes `test_arrayant_generate.m`; the two Python generator test files are merged.
- Implementation hint: positions and coupling follow directly from the indices (p, m, n, mg, ng) and the sub-array index, in one loop, instead of the four copy-element stacking loops of today; this makes `ports` and `pan` simple.

### 2.3 Open for the BS session

- Order of the TR constants in `element`. Proposed: beamwidths first, as the custom shape: `[φ_3dB, θ_3dB, SLA_V, A_max, G_max]` (omitted: θ_3dB = φ_3dB, A_max = SLA_V, G_max as above), so that the first two entries are azimuth and elevation beamwidth for every type. Alternative: vertical first, as Table 7.3-1.

Decisions from the plan review (2026-11-16), all decided:

- G1. Sub-array weights. Eq. 7.3-1 applies per TXRU over K = M/M_P elements: m restarts in each sub-array, normalisation 1/√K; the same for `pan` over N/N_P elements with 1/√(N/N_P); phase reference the sub-array centre (the weights equal eq. 7.3-1 up to a constant phase per port, which no calibration metric sees). By default the generator keeps all elements and writes the weights into the coupling matrix (the channel functions apply it; exact element positions, also for spherical waves and the near field of Table 7.8-6A); a flag combines each port into one pattern (today's `pol` 4 to 6 behaviour, faster channels, port at the sub-array centre), as multibeam's `apply_weights`.
- G2. Element and port order as today: polarisation fastest, then m (or m_p), n (or n_p), Mg, Ng. The TR is silent; documented as the library order.
- G3. `pol` and the model apply to a one-element `pattern`, with A' = |F_θ|² + |F_φ|² of that element (an existing polarisation is replaced). A `pattern` with more elements is the caller's own polarisation set: used as given, requires `pol = 1`, otherwise throws.
- G4. `element`: beamwidths first and one meaning per entry for every type: TR types `[φ_3dB, θ_3dB, SLA_V, A_max, G_max]` (omitted: θ_3dB = φ_3dB, A_max = SLA_V, G_max as above), custom and multibeam `[az_3dB, el_3dB, floor_dB]`, where entry 3 is a floor attenuation in dB for every type (the custom shape converts it, 10^(−x/10)); the custom default is NaN = no floor (today's `rear_gain_lin = 0`). Breaks custom calls that pass `rear_gain_lin`.
- G5. `steer` is `[n_panels, 2]` (one `[tilt, pan]` row per panel, panels in the order of G2) or one row, which applies to every panel; one column means pan 0. Positive `pan` turns the beam towards +y (positive azimuth). With a panel spacing of 0 (allowed, see above) this builds FR2 beam-steering antennas from co-located panels with their own steering angles. Table 7.8-2 config 1 is `steer = [t, 0; t, 0]` or one row.
- G6. Integer tuples (`panel`, `ports`, `pol`) are `arma::uvec`; 0 selects the default of an entry (a count cannot be 0; this is the #77 rule "where the quantity cannot be <= 0, values <= 0 do too"), e.g. `ports = [0, 1]` is "M_P default, N_P = 1". Real tuples (`spacing`, `steer`, `element`) use NaN. The C++ example in E4 stays `const arma::uvec &panel = {}`.
- G7. A C++ dispatcher `arrayant_generate(type, …)` holds the per-type logic (type switch, per-type defaults such as `res` 10° for omni and xpol, fill rules); the MEX and Python wrappers are 1:1 thin wrappers of it, so the #69 exception for the dispatcher goes away. The individual C++ generators stay.
- G8. Test scope for the generators (a plan rule; the SPECs stay): Catch2 tests the fill rules once through one generator (omitted entries, 0 / NaN, dependent defaults, Inf, negative, too long); other validation tests one section per tuple; each wrapper tests each position once on one type, with one loop over omitted / `[]` / None; Python layout and dtype tests only for `beams` and `pattern`.
- G9. Numeric limits: exact (0) for omitted versus 0 / NaN versus the explicit default and for alternative input forms (same arithmetic after the fill); 1e-12 relative (double) between two code paths (the ULA on the 3GPP path versus today's ULA, kept elements with coupling versus the combined port, Model-2 versus eq. 7.3-4/7.3-5); 1e-10 dB for the Table 7.3-1/7.3-2 shapes; 0.01 dB for G_max against the directivity (grid-limited); float against double 1e-5 absolute for patterns, 1e-6 relative to the aperture for positions, 1e-6·(1 + k·aperture) for coupling phases (a 64-element column at 0.7λ reaches k·z ≈ 280 rad, about 2e-5 in float).
- G10. Python mirrors the MEX positional structure: same order, positional and keyword calls both allowed (no `py::kw_only()`), so a call reads the same in both wrappers. The Python documentation's "keyword-only" sentence goes. Rejected: keyword-only after `type` (safer for long calls, but breaks the parity of the call shapes).
- G11. Multibeam: C++ and both wrappers take `beams` `[3, n_beams]` and the flags `separate_beams` and `apply_weights` (Python flags through `qd_python_flag`); the type `multibeam_sep` is removed.

Test cases from the TR: Table 7.8-2 config 1 (M = N = 4, P = 2, M_g = 1, N_g = 2, d = 0.5λ, d_g = 2.5λ, ±45°, one port per polarisation and panel) and config 2 (M = N = 2, P = 1, one port per element); Table 7.8-2A config 3 (8 × 16, M_p = 8, N_p = 16) and config 4 (64 × 16, M_p = N_p = 16); Table 7.8-6A config 1 ((24, 32, 2, 1, 1; 8, 32), d_H = 0.5λ, d_V = 0.7λ) and config 2 ((64, 16, 2, 1, 1; 8, 16)); Table 7.8-6C ((8, 16, 2, 1, 1; 8, 16)). All BS configs use Model-2.

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

Rotation: each handheld antenna is the reference element rotated by (bank, tilt, heading) = (γ_u, β_u, α_u) with `arrayant_rotate_pattern`, pattern and polarisation together (usage 0 or 3), which is eq. 7.1-7/7.1-8 with eq. 7.3-6; the session implements the rotation, not eq. 7.3-7/7.3-8 literally (their denominator vanishes on the z'' axis; the NOTE after eq. 7.3-8 allows any computation of ψ_u, and the pole handling of 6938691 covers the axis). To be checked against eq. 7.1-7/7.1-8 in a test.

Test cases: Table 7.8-2A metric 5 (F_u,θ' and F_u,φ' in the UT LCS for Config D, single polarisation, locations 1 to 8) is the reference fixture for `3GPP-UT`; Table 7.8-2A Configs B (locations (1, 7, 3, 5)) and C (dual polarisation, 1 to 8); Table 7.8-6C Configs A and B.

Parameters of `3GPP-UT`, in the encoding of section 2.2 (`3GPP-CPE`: `device_size` and `locations` as below, the rest as the panel type). Shared with the BS types: `element` (Table 7.3-2: 125°, 125°, 22.5 dB, 22.5 dB, 5.3 dBi), `pol` (1 = one field pattern per location, 2 = two field patterns), `pattern` (custom or isotropic element; Tables 7.8-6A Config B and 7.8-6B use isotropic, Tables 7.8-2A Configs B to D and 7.8-6C the Table 7.3-2 element, which is the tested default), `center_freq`, `res`. Own positions, each with one meaning:

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
