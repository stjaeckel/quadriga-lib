// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_WRITE_CHANNEL
Write a channel object to an HDF5 file

- Writes structured channel data and optional unstructured datasets to a single 4D slot
- Creates the file with a sensible default storage layout if it does not yet exist; appends to existing files otherwise
- A warning is issued if the slot already contains data (it is overwritten or extended)
- Structured data is stored in single precision regardless of MATLAB input precision
- Unstructured datasets retain their MATLAB type and shape (see [[hdf5_write_dset]])

## Usage:
```
storage_dims = quadriga_lib.hdf5_write_channel( fn, location, par, chan );
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`location`** — Slot location inside the file; 1-based; vector with 1-4 elements, i.e. `[ix]`,
  `[ix, iy]`, `[ix, iy, iz]` or `[ix, iy, iz, iw]`; pass `[]` for default `[1, 1, 1, 1]`
- **`par`** — Unstructured data; struct whose field names become HDF5 dataset names (each prefixed with `par_`); pass `[]` to skip
- **`chan`** — Structured channel data; 1x1 struct whose field layout matches [[hdf5_read_channel]]; pass `[]` to skip

## Outputs:
- **`storage_dims`** *(optional)* — Storage layout dimensions of the file `[nx, ny, nz, nw]`; `[4]`; uint32

## Caveat:
- If the file does not exist, it is created with a default layout derived from the
  number of elements in `location`:
  - `[ix]` → `[max(ix, 65536), 1, 1, 1]`
  - `[ix, iy]` → `[max(ix, 1024), max(iy, 64), 1, 1]`
  - `[ix, iy, iz]` → `[max(ix, 256), max(iy, 16), max(iz, 16), 1]`
  - `[ix, iy, iz, iw]` → `[max(ix, 128), max(iy, 8), max(iz, 8), max(iw, 8)]`
- Use [[hdf5_create_file]] for a custom storage layout
- Use [[hdf5_reshape_layout]] to change the layout later
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs != 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read fn and location
    const std::string fn = qd_mex_get_string(prhs[0]);
    const arma::Col<unsigned> location = qd_mex_get_Col<unsigned>(prhs[1]);

    const arma::uword n_loc = location.is_empty() ? 1 : location.n_elem;
    unsigned ix = location.is_empty() ? 1 : location.at(0);
    unsigned iy = location.n_elem > 1 ? location.at(1) : 1;
    unsigned iz = location.n_elem > 2 ? location.at(2) : 1;
    unsigned iw = location.n_elem > 3 ? location.at(3) : 1;

    if (ix == 0 || iy == 0 || iz == 0 || iw == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'location' cannot contain zeros (1-based indexing).");

    // Read storage layout (returns [0,0,0,0] if file does not exist)
    arma::Col<unsigned> storage_space;
    CALL_QD(storage_space = quadriga_lib::hdf5_read_layout(fn));

    // Create file with a default layout if it does not exist (using 1-based ix/iy/iz/iw)
    if (storage_space.at(0) == 0)
    {
        unsigned nx = 1, ny = 1, nz = 1, nw = 1;
        if (n_loc <= 1)
            nx = ix > 65536 ? ix : 65536;
        else if (n_loc == 2)
            nx = ix > 1024 ? ix : 1024,
            ny = iy > 64 ? iy : 64;
        else if (n_loc == 3)
            nx = ix > 256 ? ix : 256,
            ny = iy > 16 ? iy : 16,
            nz = iz > 16 ? iz : 16;
        else // n_loc == 4
            nx = ix > 128 ? ix : 128,
            ny = iy > 8 ? iy : 8,
            nz = iz > 8 ? iz : 8,
            nw = iw > 8 ? iw : 8;

        CALL_QD(quadriga_lib::hdf5_create(fn, nx, ny, nz, nw));
        storage_space.at(0) = nx;
        storage_space.at(1) = ny;
        storage_space.at(2) = nz;
        storage_space.at(3) = nw;
    }

    // Convert to 0-based for the C++ write call
    --ix, --iy, --iz, --iw;

    // Build channel object
    quadriga_lib::channel<double> c;

    // Structured data from 'chan' struct (allow empty [] to mean "skip")
    if (mxGetNumberOfElements(prhs[3]) != 0)
    {
        if (!mxIsStruct(prhs[3]))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'chan' must be a struct.");

        std::vector<quadriga_lib::channel<double>> vec;
        CALL_QD(vec = qd_mex_struct2channel(prhs[3]));

        if (vec.size() > 1)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'chan' must be a 1x1 struct.");
        if (vec.size() == 1)
            c = std::move(vec[0]);
    }

    // Unstructured data from 'par' struct (allow empty [] to mean "skip")
    if (mxGetNumberOfElements(prhs[2]) != 0)
    {
        if (!mxIsStruct(prhs[2]))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'par' must be a struct.");

        const int n_fields = mxGetNumberOfFields(prhs[2]);
        for (int i = 0; i < n_fields; ++i)
        {
            mxArray *field_data = mxGetFieldByNumber(prhs[2], 0, i);
            if (mxGetNumberOfElements(field_data) == 0)
                continue;

            const std::string field_name = mxGetFieldNameByNumber(prhs[2], i);
            c.par_names.push_back(field_name);

            if (mxIsClass(field_data, "char"))
                c.par_data.push_back(std::any(qd_mex_get_string(field_data)));
            else
                c.par_data.push_back(qd_mex_anycast(field_data, "par." + field_name, false));
        }
    }

    // Prune zero-padded trailing columns in interact_coord (round-trip artifacts)
    const size_t n_snap = (size_t)c.n_snap();
    if (c.no_interact.size() == n_snap && c.interact_coord.size() == n_snap)
    {
        for (size_t s = 0; s < n_snap; ++s)
        {
            unsigned long long cnt = 0;
            for (auto v : c.no_interact[s])
                cnt += v;
            if (c.interact_coord[s].n_cols > cnt)
                c.interact_coord[s].resize(c.interact_coord[s].n_rows, cnt);
        }
    }

    // Write to HDF5
    int return_code = 0;
    CALL_QD(return_code = quadriga_lib::hdf5_write(&c, fn, ix, iy, iz, iw));

    if (return_code == 1)
        mexWarnMsgIdAndTxt("quadriga_lib:CPPerror",
                           "Modifying or overwriting existing dataset in file.");

    // Return storage space
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&storage_space, true);
}