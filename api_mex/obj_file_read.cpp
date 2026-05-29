// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_tools.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# OBJ_FILE_READ
Read a Wavefront .obj file and extract geometry and material information

- Parses a triangulated Wavefront `.obj` file; quads and n-gons are not supported
- Materials applied per triangle via `usemtl` tag; unknown/missing materials default to `"vacuum"` (all
  parameters at their defaults: ε_r = 1, σ = 0, all loss and resonance terms disabled)
- Material name matching is case-sensitive
- Default materials follow ITU-R P.2040-3 Table 3 (1–40 GHz; ground materials limited to 1–10 GHz)
- Default material tag syntax: `usemtl itu_concrete` (or `itu_brick`, `itu_wood`, etc.)
- Custom material tag syntax: `usemtl Name::a:b:c:d:att:attB:alpha:alphaB:fRef:m:resF:resQ:resS:coiF:coiQ:coiA`
  - Trailing fields are optional; any omitted field falls back to its default (see the parameter table below)
  - Example (only ε and conductivity): `usemtl Glass::6.31:0:0.0036:1.3394`
- A material row has between 1 and 16 columns. Only column 0 (`a`) is required; every other column may be
  omitted and is then substituted with its default. The columns split into three roles:
  - **Interface reflection** (`a`, `b`, `c`, `d`, `resF`, `resQ`, `resS`) — set the complex permittivity ε, which
    fixes the Fresnel reflection coefficient and therefore the room-side absorption `1 − abs(R)²`.
    Applied once per surface hit, independent of path length.
  - **Interface transmission** (`att`, `attB`, `coiF`, `coiQ`, `coiA`) — a lumped through-surface loss in dB,
    applied once per transmission, independent of path length.
  - **In-medium attenuation** (`c`, `d` via ε, `alpha`, `alphaB`, `m`) — loss accumulated along the path
    traversed inside a body; depends on the in-medium distance.
- Frequency laws (`f` in GHz; `f/fRef` is the relative frequency, but `resF` and `coiF` are absolute GHz):<br><br>
  | Parameter  | Formula                                                      | Unit   | Meaning                                |
  | ---------- | ------------------------------------------------------------ | ------ | -------------------------------------- |
  | ε(f)       | `a·(f/fRef)^b + resS·resF² / (resF² − f² − i·(resF/resQ)·f)` | —      | relative permittivity (complex)        |
  | σ(f)       | `c·(f/fRef)^d`                                               | [S/m]  | conductivity                           |
  | att(f)     | `att·(f/fRef)^attB + coiA / (1 + (coiQ·(f − coiF)/coiF)²)`   | [dB]   | per-interface transmission loss        |
  | α(f)       | `alpha·(f/fRef)^alphaB`                                      | [dB/m] | in-medium loss × in-medium path length |
  | mass(f, L) | `max(0, m·log10((f/fRef)·L))`                                | [dB]   | in-medium, L = path length in metres   |
- **Permittivity resonance** (`resF`, `resQ`, `resS`): a Lorentz pole that adds a peak to absorption (acoustic α) 
  and a feature to reflection near `resF`; `resQ` sets sharpness (higher = narrower). Active only when `resF > 0` and
  `resS ≠ 0`. Models resonant dielectrics / frequency-selective media (EM) and Helmholtz / membrane absorbers (acoustic).
- **Coincidence term** (`coiF`, `coiQ`, `coiA`): a Lorentzian added to the transmission loss at `coiF`. 
  Negative `coiA` produces a transmission dip (acoustic coincidence / pass-band); positive `coiA` produces 
  a stop-band. Total loss is clamped to ≥ 0. Active only when `coiF > 0` and `coiA ≠ 0`.
- **Mass-law term** (`m`): a transmission loss that is logarithmic in both frequency and in-medium path 
  length. `m = 20` reproduces the acoustic mass law (+6 dB/octave and +6 dB per thickness doubling). Default 
  0 (EM through-loss is the linear `alpha` term). The imaginary sign of the ε resonance follows the library's 
  loss convention (consistent with σ).

## Usage:
```
[ mesh, mtl_prop, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names, bsdf ] = ...
    quadriga_lib.obj_file_read( fn, materials_csv, trim );
```

## Inputs:
- **`fn`** — Path to the `.obj` file
- **`materials_csv`** *(optional)* — Path to CSV file with custom material properties.
  Required columns: `name`, `a`. Optional columns (any order, any subset):
  `b`, `c`, `d`, `att`, `attB`, `alpha`, `alphaB`, `fRef`, `m`, `resF`, `resQ`, `resS`, `coiF`, `coiQ`, `coiA`.
  Missing optional columns and empty cells fall back to per-column defaults (`a` → 1, `fRef` → 1, all others → 0).
  If empty, ITU-R P.2040-3 defaults are used.
- **`trim`** *(optional, default = `true`)* — If `true`, `mtl_prop` is trimmed to the smallest width 
  that captures all non-default parameter values in the scene; if `false`, all 16 columns are returned.

## Outputs:
- **`mesh`** — Triangle vertex coordinates as `{X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3}` per row; `[n_mesh, 9]`
- **`mtl_prop`** — Material properties; `[n_mesh, n_cols]` with `1 ≤ n_cols ≤ 16` (depends on `trim`). Columns:<br><br>
  | Index | Symbol | Property                             | Units    | Default |
  | :---: | :----: | ------------------------------------ | :------: | :-----: |
  | 1     | a      | ε_r at fRef                          | —        | 1.0     |
  | 2     | b      | Frequency exponent for ε_r           | —        | 0       |
  | 3     | c      | σ at fRef                            | S/m      | 0       |
  | 4     | d      | Frequency exponent for σ             | —        | 0       |
  | 5     | att    | Penetration loss at fRef             | dB       | 0       |
  | 6     | attB   | Frequency exponent for att           | —        | 0       |
  | 7     | alpha  | In-medium absorption at fRef         | dB/m     | 0       |
  | 8     | alphaB | Frequency exponent for alpha         | —        | 0       |
  | 9     | fRef   | Reference frequency                  | GHz      | 1.0     |
  | 10    | m      | Mass-law transmission slope          | dB/decade| 0       |
  | 11    | resF   | Permittivity resonance frequency     | GHz      | 0       |
  | 12    | resQ   | Permittivity resonance quality factor| —        | 0       |
  | 13    | resS   | Permittivity resonance strength      | —        | 0       |
  | 14    | coiF   | Coincidence frequency                | GHz      | 0       |
  | 15    | coiQ   | Coincidence quality factor           | —        | 0       |
  | 16    | coiA   | Coincidence loss amplitude           | dB       | 0       |
- **`vert_list`** — All vertex positions in the file; `[n_vert, 3]`
- **`face_ind`** — 1-based indices into `vert_list` per triangle; uint64; `[n_mesh, 3]`
- **`obj_ind`** — 1-based object index per triangle; uint64; `[n_mesh]`
- **`mtl_ind`** — 1-based material index per triangle; uint64; `[n_mesh]`
- **`obj_names`** — Object names; cell array of strings; length = `max(obj_ind)`
- **`mtl_names`** — Material names; cell array of strings; length = `max(mtl_ind)`
- **`bsdf`** — Principled BSDF values from the `.mtl` file; `[n_mtl, 17]`; columns:<br><br>
   | Index | Property                  | Range | Default |
   | :---: | ------------------------- | :---: | ------: |
   | 1     | Base Color Red            | 0–1   | 0.8     |
   | 2     | Base Color Green          | 0–1   | 0.8     |
   | 3     | Base Color Blue           | 0–1   | 0.8     |
   | 4     | Transparency (alpha)      | 0–1   | 1.0     |
   | 5     | Roughness                 | 0–1   | 0.5     |
   | 6     | Metallic                  | 0–1   | 0.0     |
   | 7     | Index of refraction (IOR) | 0–4   | 1.45    |
   | 8     | Specular IOR adjustment   | 0–1   | 0.5     |
   | 9     | Emission Red              | 0–1   | 0.0     |
   | 10    | Emission Green            | 0–1   | 0.0     |
   | 11    | Emission Blue             | 0–1   | 0.0     |
   | 12    | Sheen                     | 0–1   | 0.0     |
   | 13    | Clearcoat                 | 0–1   | 0.0     |
   | 14    | Clearcoat roughness       | 0–1   | 0.0     |
   | 15    | Anisotropic               | 0–1   | 0.0     |
   | 16    | Anisotropic rotation      | 0–1   | 0.0     |
   | 17    | Transmission              | 0–1   | 0.0     |

## Default material table:
- Built-in materials use only columns 1–5 (`a`, `b`, `c`, `d`, `att`); `attB = alpha = alphaB = 0`, `fRef = 1 GHz`, and all extended parameters (`m`, `resF`, `resQ`, `resS`, `coiF`, `coiQ`, `coiA`) = 0. A scene using only built-in materials with `trim = true` therefore yields a 5-column `mtl_prop` (4 columns if no material sets `att`; only `irr_glass` does).<br><br>
  | Name                  | a     | b      | c       | d      | att  | max fGHz |
  | --------------------- | ----: | -----: | ------: | -----: | ---: | -------: |
  | vacuum / air          | 1.0   | 0.0    | 0.0     | 0.0    | 0.0  | 100      |
  | textiles              | 1.5   | 0.0    | 5e-5    | 0.62   | 0.0  | 100      |
  | plastic               | 2.44  | 0.0    | 2.33e-5 | 1.0    | 0.0  | 100      |
  | ceramic               | 6.5   | 0.0    | 0.0023  | 1.32   | 0.0  | 100      |
  | sea_water             | 80.0  | -0.25  | 4.0     | 0.58   | 0.0  | 100      |
  | sea_ice               | 3.2   | -0.022 | 1.1     | 1.5    | 0.0  | 100      |
  | water                 | 80.0  | -0.18  | 0.6     | 1.52   | 0.0  | 20       |
  | water_ice             | 3.17  | -0.005 | 5.6e-5  | 1.7    | 0.0  | 20       |
  | itu_concrete          | 5.24  | 0.0    | 0.0462  | 0.7822 | 0.0  | 100      |
  | itu_brick             | 3.91  | 0.0    | 0.0238  | 0.16   | 0.0  | 40       |
  | itu_plasterboard      | 2.73  | 0.0    | 0.0085  | 0.9395 | 0.0  | 100      |
  | itu_wood              | 1.99  | 0.0    | 0.0047  | 1.0718 | 0.0  | 100      |
  | itu_glass             | 6.31  | 0.0    | 0.0036  | 1.3394 | 0.0  | 100      |
  | itu_ceiling_board     | 1.48  | 0.0    | 0.0011  | 1.075  | 0.0  | 100      |
  | itu_chipboard         | 2.58  | 0.0    | 0.0217  | 0.78   | 0.0  | 100      |
  | itu_plywood           | 2.71  | 0.0    | 0.33    | 0.0    | 0.0  | 40       |
  | itu_marble            | 7.074 | 0.0    | 0.0055  | 0.9262 | 0.0  | 60       |
  | itu_floorboard        | 3.66  | 0.0    | 0.0044  | 1.3515 | 0.0  | 100      |
  | itu_metal             | 1.0   | 0.0    | 1.0e7   | 0.0    | 0.0  | 100      |
  | itu_very_dry_ground   | 3.0   | 0.0    | 0.00015 | 2.52   | 0.0  | 10       |
  | itu_medium_dry_ground | 15.0  | -0.1   | 0.035   | 1.63   | 0.0  | 10       |
  | itu_wet_ground        | 30.0  | -0.4   | 0.15    | 1.3    | 0.0  | 10       |
  | itu_vegetation        | 1.0   | 0.0    | 1.0e-4  | 1.1    | 0.0  | 100      |
  | irr_glass             | 6.27  | 0.0    | 0.0043  | 1.1925 | 23.0 | 100      |

## See also:
- [[obj_file_write]] (for writing OBJ files)
- [[triangle_mesh_segmentation]] (for calculating sub-meshes)
- [[triangle_mesh_aabb]] (for calculating bounding boxes)
- [[subdivide_triangles]] (for mesh subdivision)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs > 3 || nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 9)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    std::string fn = qd_mex_get_string(prhs[0]);
    std::string materials_csv = (nrhs < 2) ? "" : qd_mex_get_string(prhs[1]);
    bool trim = (nrhs < 3) ? true : (bool)mxGetScalar(prhs[2]);

    arma::mat mesh, mtl_prop, vert_list, bsdf;
    arma::umat face_ind;
    arma::uvec obj_ind, mtl_ind;
    std::vector<std::string> obj_names, mtl_names;

    CALL_QD(quadriga_lib::obj_file_read<double>(fn, &mesh, &mtl_prop, &vert_list, &face_ind, &obj_ind,
                                                &mtl_ind, &obj_names, &mtl_names, &bsdf, materials_csv, trim));

    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&mesh);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&mtl_prop);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&vert_list);
    if (nlhs > 3)
    {
        face_ind += 1;
        plhs[3] = qd_mex_copy2matlab(&face_ind);
    }
    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&obj_ind);
    if (nlhs > 5)
        plhs[5] = qd_mex_copy2matlab(&mtl_ind);
    if (nlhs > 6)
        plhs[6] = qd_mex_copy2matlab(&obj_names);
    if (nlhs > 7)
        plhs[7] = qd_mex_copy2matlab(&mtl_names);
    if (nlhs > 8)
        plhs[8] = qd_mex_copy2matlab(&bsdf);
}
