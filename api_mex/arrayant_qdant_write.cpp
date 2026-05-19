// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_QDANT_WRITE
Writes array antenna data to QDANT files

- The QuaDRiGa array antenna exchange format (QDANT) is an XML format for storing antenna pattern data
- Multiple array antennas can be stored in the same file using distinct `id` values
- If writing to an existing file without specifying an `id`, the data is appended at the end and
  the returned `id_in_file` identifies its location in the file
- An optional `layout` can be provided to organize the data inside the file
- Passing a struct array (multiple elements) writes a frequency-dependent model with sequential
  1-based IDs; in this mode appending to an exisiting file is not allowed and will cause an error

## Usage:
```
% Arrayant as struct
id_in_file = quadriga_lib.arrayant_qdant_write( fn, arrayant, id, layout );

% Arrayant as separate inputs
id_in_file = quadriga_lib.arrayant_qdant_write( fn, [], id, layout, e_theta_re, e_theta_im, e_phi_re, ...
    e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq, name );
```

## Inputs:
- **`fn`** — Output QDANT filename; string; must not be empty
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [[arrayant_generate]]; 
  pass `[]` to provide the data via separate inputs instead; a struct array writes a frequency-dependent 
  model and requires `id` and `layout` to be omitted
- **`id`** — Target ID of the antenna inside the file; default: max-ID in existing file + 1 (or 1 if the file does not exist);
  ignored for multi-frequency model
- **`layout`** — Matrix organizing multiple antenna IDs within the file; must only reference IDs present in the file; uint32

## Inputs (separate arrayant data, required when `arrayant` is `[]`):
- **`e_theta_re`** — e-theta field component, real part; `[n_elevation, n_azimuth, n_elements]`
- **`e_theta_im`** — e-theta field component, imaginary part; `[n_elevation, n_azimuth, n_elements]`
- **`e_phi_re`** — e-phi field component, real part; `[n_elevation, n_azimuth, n_elements]`
- **`e_phi_im`** — e-phi field component, imaginary part; `[n_elevation, n_azimuth, n_elements]`
- **`azimuth_grid`** — Azimuth angles in rad, -π to π, sorted; `[n_azimuth]`
- **`elevation_grid`** — Elevation angles in rad, -π/2 to π/2, sorted; `[n_elevation]`
- **`element_pos`** — Element (x,y,z) positions; `[3, n_elements]`; default: zeros
- **`coupling_re`** — Coupling matrix, real part; `[n_elements, n_ports]`; default: identity
- **`coupling_im`** — Coupling matrix, imaginary part; `[n_elements, n_ports]`; default: zeros
- **`center_freq`** — Center frequency in Hz; default: 299792458
- **`name`** — Name of the array antenna object; string

## Outputs:
- **`id_in_file`** — ID assigned to the antenna in the file after writing; set to 0 in multi-frequency (struct array) mode

## See also:
- [[arrayant_qdant_read]] (for reading QDANT data)
- [[arrayant_generate]] (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format (<a href="formats.html#6cab4884">QDANT</a>)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 15)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    std::string fn = qd_mex_get_string(prhs[0]);
    unsigned id = (nrhs < 3) ? 0 : qd_mex_get_scalar<unsigned>(prhs[2], "id", 0);
    arma::u32_mat layout = (nrhs < 4) ? arma::u32_mat() : qd_mex_typecast_Mat<unsigned>(prhs[3], "layout");

    if (fn.empty())
         mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "File name cannot be empty.");

    // Assemble array antenna object
    auto ant = quadriga_lib::arrayant<double>();
    auto ant_multi = std::vector<quadriga_lib::arrayant<double>>();
    bool is_multi = false;

    if (nrhs <= 4 && mxIsStruct(prhs[1])) // Struct input
    {
        is_multi = mxGetNumberOfElements(prhs[1]) > 1;
        if (is_multi)
        {
            ant_multi = qd_mex_struct2arrayant_multi(prhs[1], true);
            if (id != 0 || !layout.empty())
                mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Inputs 'id' and 'layout' are not allowed for multi-frequency arrays.");
        }
        else
            ant = qd_mex_struct2arrayant(prhs[1], true);
    }
    else if (nrhs >= 10) // Separate inputs
    {
        if (mxIsStruct(prhs[1]))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Cannot mix struct input with separate arrayant inputs.");

        ant.e_theta_re = qd_mex_get_Cube<double>(prhs[4]);
        ant.e_theta_im = qd_mex_get_Cube<double>(prhs[5]);
        ant.e_phi_re = qd_mex_get_Cube<double>(prhs[6]);
        ant.e_phi_im = qd_mex_get_Cube<double>(prhs[7]);
        ant.azimuth_grid = qd_mex_get_Col<double>(prhs[8]);
        ant.elevation_grid = qd_mex_get_Col<double>(prhs[9]);

        if (nrhs > 10)
            ant.element_pos = qd_mex_get_Mat<double>(prhs[10]);

        if (nrhs > 11)
            ant.coupling_re = qd_mex_get_Mat<double>(prhs[11]);

        if (nrhs > 12)
            ant.coupling_im = qd_mex_get_Mat<double>(prhs[12]);

        if (nrhs > 13)
            ant.center_frequency = qd_mex_get_scalar<double>(prhs[13]);

        if (nrhs > 14)
            ant.name = qd_mex_get_string(prhs[14]);
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    // Check if files exisits in multi-freq mode and trow error if yes
    if (is_multi && std::filesystem::exists(fn))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "File exisits. Appending a multi-frequency arrays to an aeisting file is not allowed.");

    // Write to file
    if (is_multi)
        CALL_QD(qdant_write_multi(fn, ant_multi));
    else
        CALL_QD(id = ant.qdant_write(fn, id, layout));

    if (nlhs == 1)
        plhs[0] = qd_mex_copy2matlab(&id);
}
