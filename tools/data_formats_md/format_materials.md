/*!SECTION
Material Properties
SECTION!*/

/*!SECTION_DESC
The material table describes the media of a scene for propagation, one row per material, keyed by the material's name. <a href="cpp_api.html#obj_file_read">obj_file_read</a> reads it from a CSV file (or uses its built-in table), assigns every triangle the row of its `.obj` material (`csv_ind`), and returns the row names in `csv_names` and the columns in `csv_prop`, a map from column name to a vector with one value per row. `csv_ind` is 1-based, so the value of face `i` is `csv_prop["a"][csv_ind[i] - 1]`; index 0 is no material. The ray functions take `csv_prop` as their material table (`mtl_prop`) and `csv_ind` as the material index of each face.<br><br>

This section describes the table and its file format. The model behind the columns (the formulas, the interface physics, sheets and traced solids, the electromagnetic and the acoustic use, the rules of use and the limits) is described on the page <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a>.<br><br>

Related functions:
- C++ OBJ reader (<a href="cpp_api.html#obj_file_read">obj_file_read</a>) and writer (<a href="cpp_api.html#obj_file_write">obj_file_write</a>)
- C++ ray / mesh interaction (<a href="cpp_api.html#ray_mesh_interact">ray_mesh_interact</a>)
- C++ ray tracer (<a href="cpp_api.html#ray_progress">ray_progress</a>)
- C++ diffraction gain (<a href="cpp_api.html#calc_diffraction_gain">calc_diffraction_gain</a>)
- C++ Mitsuba scene export (<a href="cpp_api.html#mitsuba_xml_file_write">mitsuba_xml_file_write</a>)
SECTION_DESC!*/

/*!MD
# 1. Material Table
The CSV form of the table, how it is read, and its 26 columns

## CSV form:
A material file is a comma-separated text file. The first line is the header with the column names; every further line is one material. Only the `name` column is required, at any position; every other column is numeric, and the column order does not matter. Example: a concrete wall traced as a solid and a 6 mm glass pane as a thin sheet:<br><br>

```
name,a,b,c,d,thk
itu_concrete,5.24,0,0.0462,0.7822,0
itu_glass,6.31,0,0.0036,1.3394,0.006
```

## Reader rules:
- Cells are separated by commas and trimmed of surrounding white space; there is no quoting, so a name or a value cannot contain a comma. A UTF-8 byte-order mark is skipped, and blank lines are ignored.
- Column names are case-sensitive. An empty column name in the header is an error.
- Every material needs a non-empty name, and names must be unique.
- An empty numeric cell, and a missing cell at the end of a line, reads as 0, not as the column's default. Where the default is not 0, fill every cell or leave the column out: a 0 in `fRef`, `a` or `e` is rejected, and a 0 in `side` selects the infinite plate. A cell that does not start with a number is an error.
- The file needs at least one material. A material file replaces the built-in table; the two are not merged.
- Columns the model does not know are returned in `csv_prop` and ignored by the ray functions.
- `obj_file_read` does not check the values. The functions that take the table check it when they read it: all columns have one length; `fRef`, `a` and `e` are positive; `c`, `g`, `thk`, `att`, `alpha`, `m`, `resF`, `resQ`, `coiF`, `coiQ`, `tfF` and `side` are not negative. A violation throws and names the column and the 1-based material. `ray_progress` takes at most 16383 materials.
- <a href="cpp_api.html#obj_file_write">obj_file_write</a> writes `csv_names` and `csv_prop` in the same form, as a `.csv` file named after the `.obj` file.

## Material lookup:
- Each face takes the material of its `usemtl` name: the row of the same name, else the row of its base name, the part before the first dot (`concrete.gray` and `concrete.001` both map to `concrete`). Several visual materials can so share one row of the table.
- A name that matches no row is an error when `csv_strict` is set; otherwise the face gets index 0, no material, which the ray functions treat as air.
- Row `k` of the table (counted from 1) is index `k` in `csv_ind`.
- Without a material file the built-in table is used (Section 2). With an empty `.obj` file name, only the table is read and returned.

## Columns:
Every column is optional apart from `name`; an absent column takes its default for every material. Frequencies are in GHz (the API takes frequencies in Hz); a value "at `fRef`" is scaled by its exponent as `(f/fRef)^exponent`. Wave media are the materials without the mass column (`m` = 0); `m > 0` switches a material to a mass-law material, a lumped plate with the mass law. The last column says which of the two a column acts on.<br><br>

| Key    | Unit | Default | Meaning                                                                                   | Acts on   |
| :----: | :--: | :-----: | ----------------------------------------------------------------------------------------- | --------- |
| fRef   | GHz  | 1       | reference frequency of every power law; of a mass-law material it also sets the surface mass | both   |
| a      | —    | 1       | relative permittivity ε′ at `fRef`                                                        | both      |
| b      | —    | 0       | frequency exponent of `a`                                                                 | both      |
| c      | S/m  | 0       | conductivity σ at `fRef`                                                                  | both      |
| d      | —    | 0       | frequency exponent of `c`                                                                 | both      |
| e      | —    | 1       | relative permeability μ′ at `fRef`                                                        | both      |
| f      | —    | 0       | frequency exponent of `e`                                                                 | both      |
| g      | S/m  | 0       | magnetic loss at `fRef` (the loss of μ, as `c` is that of ε)                              | both      |
| h      | —    | 0       | frequency exponent of `g`                                                                 | both      |
| thk    | m    | 0       | thickness: > 0 makes every face of the material a thin sheet of this thickness; 0 a traced solid | both |
| att    | dB   | 0       | penetration loss paid once per entry, at `fRef`                                           | both      |
| attB   | —    | 0       | frequency exponent of `att`                                                               | both      |
| alpha  | dB/m | 0       | absorption along the path inside the material, at `fRef`                                  | both      |
| alphaB | —    | 0       | frequency exponent of `alpha`                                                             | both      |
| m      | —    | 0       | the mass column: any value > 0 makes the material a mass-law material                     | switch    |
| resF   | GHz  | 0       | resonance frequency: a pole of ε (wave media); the plate's resonance (mass-law)           | both, differently |
| resQ   | —    | 0       | quality factor of that resonance                                                          | both      |
| resS   | —    | 0       | strength of the pole (wave media); its sign selects the form of the resonance (mass-law)  | both, differently |
| coiF   | GHz  | 0       | coincidence frequency, the plate's bending stiffness                                      | mass-law  |
| coiQ   | —    | 0       | 1 / loss factor of the bending                                                            | mass-law  |
| tf     | —    | 0       | transmission factor of the faces at `fRef`: > 0 leaks reflected power into transmission, a negative value forces reflection | both |
| tfB    | —    | 0       | frequency exponent of `tf`                                                                | both      |
| tfP    | —    | 0       | height of a resonant peak added to `tf`                                                   | both (a series impedance on mass-law faces) |
| tfF    | GHz  | 0       | frequency of that peak                                                                    | both      |
| tfQ    | —    | 0       | quality factor of that peak; a negative value selects the under-damped branch             | both      |
| side   | m    | 3.29    | panel size of a mass-law material, √(area); 0 the infinite plate                          | mass-law  |
MD!*/

/*!MD
# 2. Built-in EM Materials
The table obj_file_read uses when no material file is given

## Table:
Built-in material table used by <a href="cpp_api.html#obj_file_read">obj_file_read</a> when no material file is given. It defines only the columns `a`, `b`, `c`, `d` and `att`; every other column is absent and takes its default (`fRef` 1 GHz, μ = 1, no `thk`, so every material is a traced solid). A scene using only built-in materials therefore has the keys `a`, `b`, `c`, `d`, `att` in `csv_prop`. The range is the frequency range of the source in GHz.<br><br>

| Name                  | a     | b      | c       | d      | att  | Range (GHz) | Source         |
| --------------------- | ----: | -----: | ------: | -----: | ---: | ----------: | -------------- |
| air                   | 1.0   | 0.0    | 0.0     | 0.0    | 0.0  | 0.001–100   | P.2040-3       |
| vacuum                | 1.0   | 0.0    | 0.0     | 0.0    | 0.0  | 0.001–100   | P.2040-3       |
| textiles              | 1.5   | 0.0    | 5e-5    | 0.62   | 0.0  | 1–100       | estimate       |
| plastic               | 2.44  | 0.0    | 2.33e-5 | 1.0    | 0.0  | 1–100       | estimate       |
| ceramic               | 6.5   | 0.0    | 0.0023  | 1.32   | 0.0  | 1–100       | estimate       |
| sea_water             | 80.0  | -0.25  | 4.0     | 0.58   | 0.0  | 1–100       | estimate       |
| sea_ice               | 3.2   | -0.022 | 1.1     | 1.5    | 0.0  | 1–100       | estimate       |
| water                 | 80.0  | -0.18  | 0.6     | 1.52   | 0.0  | 1–20        | estimate       |
| water_ice             | 3.17  | -0.005 | 5.6e-5  | 1.7    | 0.0  | 1–20        | estimate       |
| itu_concrete          | 5.24  | 0.0    | 0.0462  | 0.7822 | 0.0  | 1–100       | P.2040-3       |
| itu_brick             | 3.91  | 0.0    | 0.0238  | 0.16   | 0.0  | 1–40        | P.2040-3       |
| itu_plasterboard      | 2.73  | 0.0    | 0.0085  | 0.9395 | 0.0  | 1–100       | P.2040-3       |
| itu_wood              | 1.99  | 0.0    | 0.0047  | 1.0718 | 0.0  | 0.001–100   | P.2040-3       |
| itu_glass             | 6.31  | 0.0    | 0.0036  | 1.3394 | 0.0  | 0.1–100     | P.2040-3       |
| itu_ceiling_board     | 1.48  | 0.0    | 0.0011  | 1.075  | 0.0  | 1–100       | P.2040-3       |
| itu_chipboard         | 2.58  | 0.0    | 0.0217  | 0.78   | 0.0  | 1–100       | P.2040-3       |
| itu_plywood           | 2.71  | 0.0    | 0.33    | 0.0    | 0.0  | 1–40        | P.2040-3       |
| itu_marble            | 7.074 | 0.0    | 0.0055  | 0.9262 | 0.0  | 1–60        | P.2040-3       |
| itu_floorboard        | 3.66  | 0.0    | 0.0044  | 1.3515 | 0.0  | 50–100      | P.2040-3       |
| itu_metal             | 1.0   | 0.0    | 1.0e7   | 0.0    | 0.0  | 1–100       | P.2040-3       |
| itu_very_dry_ground   | 3.0   | 0.0    | 0.00015 | 2.52   | 0.0  | 1–10        | P.2040-3       |
| itu_medium_dry_ground | 15.0  | -0.1   | 0.035   | 1.63   | 0.0  | 1–10        | P.2040-3       |
| itu_wet_ground        | 30.0  | -0.4   | 0.15    | 1.3    | 0.0  | 1–10        | P.2040-3       |
| itu_vegetation        | 1.0   | 0.0    | 1.0e-4  | 1.1    | 0.0  | 0.03–30     | P.833-9        |
| irr_glass             | 6.27  | 0.0    | 0.0043  | 1.1925 | 23.0 | 0.5–100     | TR 38.901      |

## Notes:
- The P.2040-3 rows are Rec. ITU-R P.2040-3, Table 3. Its frequency ranges are not hard limits but indicate the measurements behind each fit, except for the three ground types, which must not be used outside 1–10 GHz.
- `air` and `vacuum` are rows 1 and 2, identical (lossless free space). A face with no material (index 0) is also treated as air.
- `itu_vegetation` follows the specific attenuation of woodland in Rec. ITU-R P.833-9, Figure 2 (30 MHz to about 30 GHz).
- `irr_glass` (infrared-reflective glass) takes its `att` from 3GPP TR 38.901 V17.0.0, Table 7.4.3-1 (the constant 23 dB of its loss `23 + 0.3 f` dB) and is the only built-in material with a non-zero `att`.
- `textiles`, `plastic`, `ceramic`, `sea_water`, `sea_ice`, `water` and `water_ice` are estimates of the library, not values of a recommendation; the range is the band they were made for.
- There is no built-in acoustic table. Preliminary starting rows for rigid constructions are listed in <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md#114-starting-rows-for-rigid-constructions">Section 11.4 of the material-model page</a>.
MD!*/

/*!MD
# 3. Principled BSDF
Visual material parameters read from the companion .mtl file

## Parameters:
When an OBJ file references an `.mtl` library, the per-material Principled BSDF parameters can be returned as an `[n_mtl, 17]` matrix (`bsdf`). These describe visual appearance only and do not affect propagation. Each row corresponds to one entry of `mtl_names`; if no matching `.mtl` file is found, the matrix is empty. All color, alpha, and 0–1 parameters are clamped to [0, 1].<br><br>

| Index | Field                | Property                       | Range  | Default |
| :---: | -------------------- | ------------------------------ | :----: | :-----: |
| 0     | R                    | Base color, red                | 0 … 1  | 0.8     |
| 1     | G                    | Base color, green              | 0 … 1  | 0.8     |
| 2     | B                    | Base color, blue               | 0 … 1  | 0.8     |
| 3     | alpha                | Opacity                        | 0 … 1  | 1.0     |
| 4     | roughness            | Surface roughness              | 0 … 1  | 0.5     |
| 5     | metallic             | Metallic factor                | 0 … 1  | 0.0     |
| 6     | ior                  | Index of refraction            | ≥ 1    | 1.45    |
| 7     | specular             | Specular factor                | 0 … 1  | 0.5     |
| 8     | Re                   | Emission color, red            | 0 … 1  | 0.0     |
| 9     | Ge                   | Emission color, green          | 0 … 1  | 0.0     |
| 10    | Be                   | Emission color, blue           | 0 … 1  | 0.0     |
| 11    | sheen                | Sheen factor                   | 0 … 1  | 0.0     |
| 12    | clearcoat            | Clearcoat factor               | 0 … 1  | 0.0     |
| 13    | clearcoat_roughness  | Clearcoat roughness            | 0 … 1  | 0.0     |
| 14    | anisotropic          | Anisotropy factor              | 0 … 1  | 0.0     |
| 15    | anisotropic_rotation | Anisotropy rotation            | 0 … 1  | 0.0     |
| 16    | transmission         | Transmission factor            | 0 … 1  | 0.0     |

## Mapping from .mtl keywords:
| .mtl keyword | BSDF field             | Notes                                            |
| ------------ | ---------------------- | ------------------------------------------------ |
| `Kd`         | R, G, B                | Base color                                       |
| `Ke`         | Re, Ge, Be             | Emission color                                   |
| `Ka`         | metallic               | First component only                             |
| `Pm`         | metallic               | The later of `Ka` and `Pm` wins                  |
| `Ks`         | specular               | First component only                             |
| `d`          | alpha                  | —                                                |
| `Ni`         | ior                    | —                                                |
| `Ns`         | roughness              | Converted: `roughness = 1 − sqrt(Ns · 0.001)`    |
| `Pr`         | roughness              | The later of `Ns` and `Pr` wins                  |
| `Ps`         | sheen                  | —                                                |
| `Pc`         | clearcoat              | —                                                |
| `Pcr`        | clearcoat_roughness    | —                                                |
| `aniso`      | anisotropic            | —                                                |
| `anisor`     | anisotropic_rotation   | —                                                |
| `Tf`         | transmission           | First component only                             |
MD!*/

/*!MD
# 4. References
Sources of the built-in table and the description of the model

- <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a> — the model behind the columns, with its own references.
- <a target="_blank" rel="noopener noreferrer" href="https://www.itu.int/rec/R-REC-P.2040">Rec. ITU-R P.2040-3</a>, Effects of building materials and structures on radiowave propagation above about 100 MHz, 2023 — Table 3, the source of the P.2040-3 rows.
- <a target="_blank" rel="noopener noreferrer" href="https://www.itu.int/rec/R-REC-P.833">Rec. ITU-R P.833-9</a>, Attenuation in vegetation, 2016 — Figure 2, the source of `itu_vegetation`.
- 3GPP TR 38.901 V17.0.0, Study on channel model for frequencies from 0.5 to 100 GHz, 2022 — Table 7.4.3-1, material penetration losses, the source of the `att` of `irr_glass`.
MD!*/
