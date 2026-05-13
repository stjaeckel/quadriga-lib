// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_WRITE_CHANNEL
Write one or more channel objects to an HDF5 file

- Writes a struct array of channels into 4D slots (one slot per array element)
- Optional unstructured data can be passed as a matching struct array
- Creates the file with a sensible default layout if it does not yet exist; appends to existing files otherwise
- A warning is issued if any selected slot already contains data (it is overwritten)
- Structured data is stored in single precision regardless of MATLAB input precision
- Unstructured datasets retain their MATLAB type and shape (see [[hdf5_write_dset]])
- Each scalar location input is broadcast to all `numel(chan)` channels; each vector input must have exactly `numel(chan)` elements
- If the file does not exist, it is created with layout `[max(numel(chan), max(ix)), max(iy), max(iz), max(iw)]`

## Usage:
```
storage_dims = quadriga_lib.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw );
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`chan`** — Structured channel data; non-empty struct array; field layout matches [[hdf5_read_channel]]
- **`par`** *(optional)* — Unstructured data; struct array of the same size as `chan`. Field names become HDF5 dataset
  names per slot (each prefixed with `par_`). Empty fields are skipped. Pass `[]` or omit to disable.
- **`ix`** *(optional)* — 1-based slot index along dimension X; scalar or vector of length `numel(chan)`; default `1:numel(chan)`
- **`iy`** *(optional)* — 1-based slot index along dimension Y; scalar or vector of length `numel(chan)`; default `1`
- **`iz`** *(optional)* — 1-based slot index along dimension Z; scalar or vector of length `numel(chan)`; default `1`
- **`iw`** *(optional)* — 1-based slot index along dimension W; scalar or vector of length `numel(chan)`; default `1`

## Outputs:
- **`storage_dims`** *(optional)* — Storage layout dimensions of the file `[nx, ny, nz, nw]`; `[4]`; uint32

## See also:
- [[hdf5_create_file]] (for creating a file with a custom storage layout)
- [[hdf5_reshape_layout]] (to change the layout later)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read filename
    const std::string fn = qd_mex_get_string(prhs[0]);

    // Parse 'chan' (mandatory, struct or struct array)
    if (!mxIsStruct(prhs[1]) || mxGetNumberOfElements(prhs[1]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'chan' must be a non-empty struct.");

    std::vector<quadriga_lib::channel<double>> channels;
    CALL_QD(channels = qd_mex_struct2channel(prhs[1], true));
    const arma::uword n_chan = (arma::uword)channels.size();

    // Parse 'par' (optional, struct array of same size as 'chan', or empty)
    const bool has_par = (nrhs > 2) && (mxGetNumberOfElements(prhs[2]) != 0);
    if (has_par)
    {
        if (!mxIsStruct(prhs[2]))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'par' must be a struct.");
        if ((arma::uword)mxGetNumberOfElements(prhs[2]) != n_chan)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'par' must have the same number of elements as 'chan'.");
    }

    // Parse location indices (scalar broadcast or vector of length n_chan)
    auto get_index = [&](int arg_idx, unsigned default_val) -> arma::Col<unsigned>
    {
        if (arg_idx >= nrhs || mxGetNumberOfElements(prhs[arg_idx]) == 0)
            return arma::Col<unsigned>(1).fill(default_val);
        return qd_mex_get_Col<unsigned>(prhs[arg_idx]);
    };

    arma::Col<unsigned> ix_vec, iy_vec, iz_vec, iw_vec;

    // ix: default = 1:n_chan
    if (nrhs > 3 && mxGetNumberOfElements(prhs[3]) != 0)
        ix_vec = qd_mex_get_Col<unsigned>(prhs[3]);
    else
    {
        ix_vec.set_size(n_chan);
        for (arma::uword i = 0; i < n_chan; ++i)
            ix_vec.at(i) = (unsigned)(i + 1);
    }
    iy_vec = get_index(4, 1);
    iz_vec = get_index(5, 1);
    iw_vec = get_index(6, 1);

    auto check_idx = [&](const arma::Col<unsigned> &v, const char *name)
    {
        if (v.n_elem != 1 && v.n_elem != n_chan)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input '%s' must be a scalar or a vector of length numel(chan).", name);
        if (arma::any(v == 0))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input '%s' cannot contain zeros (1-based indexing).", name);
    };
    check_idx(ix_vec, "ix");
    check_idx(iy_vec, "iy");
    check_idx(iz_vec, "iz");
    check_idx(iw_vec, "iw");

    // Read storage layout (returns [0,0,0,0] if file does not exist)
    arma::Col<unsigned> storage_space;
    CALL_QD(storage_space = quadriga_lib::hdf5_read_layout(fn));

    // Create file with a default layout if it does not exist
    if (storage_space.at(0) == 0)
    {
        unsigned nx = std::max((unsigned)n_chan, ix_vec.max());
        unsigned ny = iy_vec.max();
        unsigned nz = iz_vec.max();
        unsigned nw = iw_vec.max();
        CALL_QD(quadriga_lib::hdf5_create(fn, nx, ny, nz, nw));
        storage_space.at(0) = nx;
        storage_space.at(1) = ny;
        storage_space.at(2) = nz;
        storage_space.at(3) = nw;
    }

    // Cache 'par' field names (struct arrays share field layout)
    std::vector<std::string> par_field_names;
    int n_par_fields = 0;
    if (has_par)
    {
        n_par_fields = mxGetNumberOfFields(prhs[2]);
        par_field_names.reserve((size_t)n_par_fields);
        for (int i = 0; i < n_par_fields; ++i)
            par_field_names.emplace_back(mxGetFieldNameByNumber(prhs[2], i));
    }

    // Write each channel to its slot
    bool any_overwrite = false;
    for (arma::uword k = 0; k < n_chan; ++k)
    {
        quadriga_lib::channel<double> &c = channels[k];

        // Attach unstructured 'par' data for this channel
        if (has_par)
        {
            for (int i = 0; i < n_par_fields; ++i)
            {
                mxArray *field_data = mxGetFieldByNumber(prhs[2], (mwIndex)k, i);
                if (field_data == nullptr || mxGetNumberOfElements(field_data) == 0)
                    continue;

                const std::string &field_name = par_field_names[(size_t)i];
                c.par_names.push_back(field_name);

                if (mxIsClass(field_data, "char"))
                    c.par_data.push_back(std::any(qd_mex_get_string(field_data)));
                else
                    c.par_data.push_back(qd_mex_anycast(field_data, "par." + field_name, false));
            }
        }

        // Resolve location with scalar broadcasting
        unsigned ix = ix_vec.n_elem == 1 ? ix_vec.at(0) : ix_vec.at(k);
        unsigned iy = iy_vec.n_elem == 1 ? iy_vec.at(0) : iy_vec.at(k);
        unsigned iz = iz_vec.n_elem == 1 ? iz_vec.at(0) : iz_vec.at(k);
        unsigned iw = iw_vec.n_elem == 1 ? iw_vec.at(0) : iw_vec.at(k);

        // Write (convert to 0-based)
        int return_code = 0;
        CALL_QD(return_code = quadriga_lib::hdf5_write(&c, fn, ix - 1, iy - 1, iz - 1, iw - 1));
        if (return_code == 1)
            any_overwrite = true;
    }

    if (any_overwrite)
        mexWarnMsgIdAndTxt("quadriga_lib:CPPerror", "Modifying or overwriting existing dataset in file.");

    // Return storage space
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&storage_space, true);
}