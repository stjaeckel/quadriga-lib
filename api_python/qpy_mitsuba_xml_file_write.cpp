// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# mitsuba_xml_file_write
Write geometry and material data to a Mitsuba 3 XML scene file.

## Description:
Converts a triangular surface mesh into the XML format understood by **Mitsuba 3**
<a href="https://www.mitsuba-renderer.org">www.mitsuba-renderer.org</a>.
The generated file can be loaded directly by **NVIDIA Sionna RT** for differentiable radio-propagation
simulations.<br><br>

- Converts a 3D geometry mesh into Mitsuba 3 XML format for use with rendering tools.
- Enables exporting models from `quadriga-lib` to be used with **Mitsuba 3** or **Sionna RT**:
- <a href="https://www.mitsuba-renderer.org">Mitsuba 3</a>: Research-oriented retargetable rendering system.
- <a href="https://developer.nvidia.com/sionna">NVIDIA Sionna</a>: Hardware-accelerated differentiable ray tracer for wireless propagation, built on Mitsuba 3.
- Supports grouping faces into named objects and assigning materials by name.
- Optionally maps materials to ITU default presets used by Sionna RT.

## Usage:
```
from quadriga_lib import RTtools
RTtools.mitsuba_xml_file_write( fn, vert_list, face_ind, obj_id, mtl_id, obj_names, mtl_names, bsdf, map_to_itu )
```

## Input Arguments:
- **`fn`**<br>
  Output file name (including path and `.xml` extension).

- **`vert_list`**<br>
  Vertex list, size `[n_vert, 3]`, each row is a vertex (x, y, z) in Cartesian coordinates [m].

- **`face_ind`**<br>
  Face indices (0-based), size `[n_mesh, 3]`, each row defines a triangle via vertex indices.

- **`obj_id`** (input)<br>
  Object indices (1-based), size `[n_mesh]`. Assigns each triangle to an object.

- **`mtl_id`** (input)<br>
  Material indices (1-based), size `[n_mesh]`. Assigns each triangle to a material.

- **`obj_names`**<br>
  List of object names. Length must be equal to `max(obj_ind)`.

- **`mtl_names`**<br>
  List of material names. Length must be equal to `max(mtl_ind)`.

- `**bsdf** = []` (optional input)<br>
  Material reflectivity data (BSDF parameters), size `[len(mtl_names), 17]`. If omitted, the `null` BSDF is used.
  Note that Sionna RT ignores all BSDF parameters. They are only used by the Mitsuma rendering system.
  See [[obj_file_read]] for a definition of the data fields.

- `**map_to_itu** = false` (optional input)<br>
  If true, maps material names to ITU-defined presets used by Sionna RT. Default: `false`

## See also:
- [[obj_file_read]]
MD!*/

void mitsuba_xml_file_write(const std::string &fn,             // Output file name
                            py::array_t<double> vert_list,     // Vertex list, size [n_vert, 3]
                            py::array_t<arma::uword> face_ind, // Face indices (0-based), size [n_mesh, 3]
                            py::array_t<arma::uword> obj_ind,  // Object indices (1-based), size [n_mesh]
                            py::array_t<arma::uword> mtl_ind,  // Material indices (1-based), size [n_mesh]
                            py::list obj_names,                // Object names, length = max(obj_ind)-1
                            py::list mtl_names,                // Material names, length = max(mtl_ind)-1
                            py::array_t<double> bsdf,          // BSDF data, size [mtl_names.size(), 17]
                            bool map_to_itu)                   // Optional mapping to ITU default materials used by Sionna
{
    const arma::mat vert_list_a = qd_python_numpy2arma_Mat(vert_list, true);
    const arma::umat face_ind_a = qd_python_numpy2arma_Mat(face_ind, true);
    const arma::uvec obj_ind_a = qd_python_numpy2arma_Col(obj_ind, true);
    const arma::uvec mtl_ind_a = qd_python_numpy2arma_Col(mtl_ind, true);
    const std::vector<std::string> obj_names_a = qd_python_list2vector_Strings(obj_names);
    const std::vector<std::string> mtl_names_a = qd_python_list2vector_Strings(mtl_names);
    const arma::mat bsdf_a = qd_python_numpy2arma_Mat(bsdf, true);

    quadriga_lib::mitsuba_xml_file_write(fn, vert_list_a, face_ind_a, obj_ind_a, mtl_ind_a, obj_names_a, mtl_names_a, bsdf_a, map_to_itu);
}
