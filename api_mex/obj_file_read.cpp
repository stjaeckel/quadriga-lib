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
- Materials applied per triangle via `usemtl` tag; unknown or missing materials default to
  `"vacuum"` (ε_r = 1, σ = 0, Att = 0, α = 0)
- Material name matching is case-sensitive
- Default materials follow ITU-R P.2040-3 Table 3 (1–40 GHz; ground materials limited to 1–10 GHz)
- Default material tag syntax: `usemtl itu_concrete` (or `itu_brick`, `itu_wood`, etc.)
- Custom material tag syntax: `usemtl Name::a:b:c:d:att:attB:alpha:alphaB:fRef`<br>
  - ε_r(f)  = a · (f/fRef)^b           (relative permittivity)<br>
  - σ(f)    = c · (f/fRef)^d   [S/m]   (conductivity)<br>
  - Att(f)  = att · (f/fRef)^attB [dB] (fixed penetration loss)<br>
  - α(f)    = alpha · (f/fRef)^alphaB  [dB/m] (distance-dependent absorption)<br>
  - Trailing fields are optional; defaults are `b = c = d = att = attB = alpha = alphaB = 0`, `fRef = 1` GHz

## Usage:
```
[ mesh, mtl_prop, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names, bsdf ] = ...
    quadriga_lib.obj_file_read( fn );

% Use a custom material definition file
[ mesh, mtl_prop, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names, bsdf ] = ...
    quadriga_lib.obj_file_read( fn, materials_csv );
```

## Inputs:
- **`fn`** — Path to the `.obj` file
- **`materials_csv`** — Path to CSV file with custom material properties. Required columns: `name`, `a`. 
  Optional columns: `b`, `c`, `d`, `att`, `attB`, `alpha`, `alphaB`, `fRef`. Column order is flexible; 
  missing optional columns default to `0` (`fRef` → `1`). If empty, ITU-R P.2040-3 defaults are used.

## Outputs:
- **`mesh`** — Triangle vertex coordinates as `{X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3}` per row; `[n_mesh, 9]`
- **`mtl_prop`** — Material properties; `[n_mesh, 9]`; Columns:<br><br>
  | Index | Symbol | Property                                      |
  | :---: | :----: | --------------------------------------------- |
  | 1     | a      | ε_r at fRef                                   |
  | 2     | b      | Frequency exponent for ε_r                    |
  | 3     | c      | σ at fRef [S/m]                               |
  | 4     | d      | Frequency exponent for σ                      |
  | 5     | att    | Penetration loss at fRef [dB]                 |
  | 6     | attB   | Frequency exponent for att                    |
  | 7     | alpha  | Distance absorption at fRef [dB/m]            |
  | 8     | alphaB | Frequency exponent for alpha                  |
  | 9     | fRef   | Reference frequency [GHz]                     |
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
- For all defaults below: `attB = alpha = alphaB = 0` and `fRef = 1 GHz`:<br><br>
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
    if (nrhs > 2 || nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 9)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    std::string fn = qd_mex_get_string(prhs[0]);
    std::string materials_csv = (nrhs < 2) ? "" : qd_mex_get_string(prhs[1]);

    arma::mat mesh, mtl_prop, vert_list, bsdf;
    arma::umat face_ind;
    arma::uvec obj_ind, mtl_ind;
    std::vector<std::string> obj_names, mtl_names;

    CALL_QD(quadriga_lib::obj_file_read<double>(fn, &mesh, &mtl_prop, &vert_list, &face_ind, &obj_ind,
                                                &mtl_ind, &obj_names, &mtl_names, &bsdf, materials_csv));

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
