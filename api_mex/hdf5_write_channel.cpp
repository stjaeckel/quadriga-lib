// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# HDF5_WRITE_CHANNEL
Write channel objects to one or several slots of an HDF5 channel file

- Stores `chan(k)` with the unstructured data `par(k)` in slot `(ix(k), iy(k), iz(k), iw(k))`; a scalar index
  applies to all channels
- A missing file is created with the layout that just holds the slots (the largest index per dimension);
  [[hdf5_create_file]] sets another layout
- A slot that holds data is replaced completely: its old channel and all its unstructured datasets, also those added
  with [[hdf5_write_dset]], are removed
- All inputs are checked before anything is written, so a failing check leaves the file unchanged

## Usage:
```
code = quadriga_lib.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw );
```

## Inputs:
- **`fn`** — File name
- **`chan`** — Channels to write, valid; struct array, fields in [[hdf5_read_channel]]; `n_chan` entries, at least 1
- **`par`** — Unstructured data; struct array of `n_chan` entries: a field name is a dataset name (stored with the
  prefix `par_`), the values follow the type rules of [[hdf5_write_dset]], empty fields are skipped; default: `[]`
  (none)
- **`ix`** — Slot index along x, 1-based; scalar or `n_chan` entries; default: `1:n_chan`
- **`iy`**, **`iz`**, **`iw`** — Slot indices along y, z and w, 1-based; scalar or `n_chan` entries; default: 1

## Output:
- **`code`** — `0` if the slot was empty, `1` if its data was replaced; int32; `[n_chan, 1]`

## Notes:
- A channel without data and without `par` entries replaces the data of an occupied slot (it then reads as an empty
  channel, and [[hdf5_read_layout]] reports it as written); it leaves an empty slot as it is and does not create a
  missing file
- The space of replaced data is not reused, so the file does not shrink; copying the channels into a new file reclaims
  it
- Files are not locked: one writer per file at a time, also for [[hdf5_write_dset]] and [[hdf5_reshape_layout]].
  Reading is safe while no one writes

## See also:
- [[hdf5_read_channel]] (reads the channels back)
- [[hdf5_create_file]] (creates a file with a given layout)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");

    if (!mxIsStruct(prhs[1]) || mxGetNumberOfElements(prhs[1]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'chan' must be a non-empty struct.");

    auto channels = qd_mex_struct2channel(prhs[1], false, false, "chan"); // C++ validates, naming the entry
    const arma::uword n_chan = (arma::uword)channels.size();

    // Parse 'par' (optional, struct array of same size as 'chan', or empty)
    const bool has_par = (nrhs > 2) && (mxGetNumberOfElements(prhs[2]) != 0);
    if (has_par)
    {
        qd_mex_check_struct(prhs[2], "par");
        if ((arma::uword)mxGetNumberOfElements(prhs[2]) != n_chan)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Inputs 'par' and 'chan' must have the same number of entries.");
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
                    c.par_data.push_back(qd_mex_anycast(field_data, "par." + field_name));
            }

        slots.at(0, k) = ix_vec.n_elem == 1 ? ix_vec.at(0) : ix_vec.at(k);
        slots.at(1, k) = iy_vec.n_elem == 1 ? iy_vec.at(0) : iy_vec.at(k);
        slots.at(2, k) = iz_vec.n_elem == 1 ? iz_vec.at(0) : iz_vec.at(k);
        slots.at(3, k) = iw_vec.n_elem == 1 ? iw_vec.at(0) : iw_vec.at(k);
    }

    // Write all channels (C++ checks all inputs first and creates a missing file)
    arma::s32_vec code;
    CALL_QD(code = quadriga_lib::hdf5_write_channel(fn, channels, slots));
    plhs[0] = qd_mex_copy2matlab(&code);
}
