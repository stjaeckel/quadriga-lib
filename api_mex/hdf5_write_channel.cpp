// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_WRITE_CHANNEL
Write one or more channel objects to an HDF5 file

- Writes a struct array of channels into 4D slots (one slot per array element)
- Optional unstructured data can be passed as a matching struct array
- Creates the file if it does not exist, with the layout that just holds all slots (largest index per dimension);
  use [[hdf5_create_file]] for another layout
- All inputs are checked before anything is written: every slot inside the layout of an existing file, no slot
  twice, all channels valid
- A slot that already contains data is replaced completely: its old structured data and all its unstructured
  datasets (also those written with [[hdf5_write_dset]]) are removed; `code` shows which slots were replaced. The
  file does not shrink; copy the channels into a new file to reclaim the space of the removed data
- One writer per file at a time; files are not locked
- Structured data is stored in single precision regardless of MATLAB input precision
- Unstructured datasets retain their MATLAB type and shape (see [[hdf5_write_dset]])
- Each scalar location input is broadcast to all `numel(chan)` channels; each vector input must have exactly `numel(chan)` elements

## Usage:
```
code = quadriga_lib.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw );
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`chan`** — Structured channel data; non-empty struct array; field layout matches [[hdf5_read_channel]]
- **`par`** — Unstructured data; struct array of the same size as `chan`. Field names become HDF5 dataset
  names per slot (each prefixed with `par_`). Empty fields are skipped. Pass `[]` or omit to disable.
  Field values follow the type rules of [[hdf5_write_dset]] (at most 3 dimensions).
- **`ix`** — 1-based slot index along dimension X; scalar or vector of length `numel(chan)`; default `1:numel(chan)`
- **`iy`** — 1-based slot index along dimension Y; scalar or vector of length `numel(chan)`; default `1`
- **`iz`** — 1-based slot index along dimension Z; scalar or vector of length `numel(chan)`; default `1`
- **`iw`** — 1-based slot index along dimension W; scalar or vector of length `numel(chan)`; default `1`

## Outputs:
- **`code`** — Per channel: `0` if the slot was empty, `1` if existing data was replaced; `[numel(chan), 1]`; int32

## See also:
- [[hdf5_create_file]] (for creating a file with a custom storage layout)
- [[hdf5_reshape_layout]] (to change the layout later)
- [[hdf5_read_channel]] (for reading channel data)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read filename
    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");

    // Parse 'chan' (mandatory, struct or struct array)
    if (!mxIsStruct(prhs[1]) || mxGetNumberOfElements(prhs[1]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'chan' must be a non-empty struct.");

    auto channels = qd_mex_struct2channel(prhs[1], false, false, "chan"); // C++ validates, naming the entry
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

    // Parse location indices, 1-based -> 0-based (scalar broadcast or vector of length n_chan)
    auto get_index = [&](int arg_idx, const char *name) -> arma::Col<unsigned>
    {
        if (arg_idx >= nrhs || mxGetNumberOfElements(prhs[arg_idx]) == 0)
            return arma::Col<unsigned>(1, arma::fill::zeros);
        return qd_mex_get_index_Col<unsigned>(prhs[arg_idx], name);
    };

    arma::Col<unsigned> ix_vec, iy_vec, iz_vec, iw_vec;

    // ix: default = 1:n_chan
    if (nrhs > 3 && mxGetNumberOfElements(prhs[3]) != 0)
        ix_vec = get_index(3, "ix");
    else
        ix_vec = arma::regspace<arma::Col<unsigned>>(0, 1, (unsigned)n_chan - 1);
    iy_vec = get_index(4, "iy");
    iz_vec = get_index(5, "iz");
    iw_vec = get_index(6, "iw");

    auto check_idx = [&](const arma::Col<unsigned> &v, const char *name)
    {
        if (v.n_elem != 1 && v.n_elem != n_chan)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input '%s' must be a scalar or a vector of length numel(chan).", name);
    };
    check_idx(ix_vec, "ix");
    check_idx(iy_vec, "iy");
    check_idx(iz_vec, "iz");
    check_idx(iw_vec, "iw");

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

    // Attach the unstructured 'par' data and resolve the slot of each channel (scalar broadcasting)
    arma::u32_mat slots(4, n_chan);
    for (arma::uword k = 0; k < n_chan; ++k)
    {
        quadriga_lib::channel<double> &c = channels[k];
        if (has_par)
            for (int i = 0; i < n_par_fields; ++i)
            {
                mxArray *field_data = mxGetFieldByNumber(prhs[2], (mwIndex)k, i);
                if (field_data == nullptr || mxGetNumberOfElements(field_data) == 0)
                    continue;

                const std::string &field_name = par_field_names[(size_t)i];
                c.par_names.push_back(field_name);

                if (mxIsClass(field_data, "char"))
                    c.par_data.push_back(std::any(qd_mex_get_string(field_data, "", "par." + field_name)));
                else
                    c.par_data.push_back(qd_mex_anycast(field_data, "par." + field_name, false));
            }

        slots.at(0, k) = ix_vec.n_elem == 1 ? ix_vec.at(0) : ix_vec.at(k);
        slots.at(1, k) = iy_vec.n_elem == 1 ? iy_vec.at(0) : iy_vec.at(k);
        slots.at(2, k) = iz_vec.n_elem == 1 ? iz_vec.at(0) : iz_vec.at(k);
        slots.at(3, k) = iw_vec.n_elem == 1 ? iw_vec.at(0) : iw_vec.at(k);
    }

    // Write all channels (C++ checks all inputs first and creates a missing file)
    arma::s32_vec code;
    CALL_QD(code = quadriga_lib::hdf5_write_channel_vec(fn, channels, slots));
    plhs[0] = qd_mex_copy2matlab(&code);
}
