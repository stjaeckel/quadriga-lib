// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# hdf5_write_channel
Write one or more channel objects to an HDF5 file

- Writes a list of channel dicts into 4D storage slots (one slot per list entry)
- `chan` may also be a single dict, in which case one channel is written
- Optional unstructured data (`par`) can be passed as a matching list of dicts (or a single dict)
- Creates the file if it does not exist, with the layout that just holds all slots (largest index + 1 per
  dimension); use [[hdf5_create_file]] for another layout
- All inputs are checked before anything is written: every slot inside the layout of an existing file, no slot
  twice, all channels valid
- A slot that already contains data is replaced completely: its old structured data and all its unstructured
  datasets (also those written with [[hdf5_write_dset]]) are removed; `code` shows which slots were replaced.
  The file does not shrink; copy the channels into a new file to reclaim the space of the removed data
- One writer per file at a time; files are not locked
- Structured data is stored in single precision regardless of the input precision
- Unstructured datasets retain their numpy dtype and shape; supported types as in [[hdf5_write_dset]]
- Each scalar index input is broadcast to all channels; each vector index must have one entry per channel
- Channel dict field layout matches [[hdf5_read_channel]]
- Per-snapshot fields accept the list model (list of arrays) or the stack model
  (single array with the snapshot index as the last axis), detected per field
- Coefficients may be passed as a complex `coeff`, or as separate real `coeff_re` and `coeff_im` (the two forms are mutually exclusive)
- Slot indices are 0-based

## Usage:
```
code = quadriga_lib.channel.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file
- **`chan`** — Structured channel data; a channel dict or a list of channel dicts; field layout matches [[hdf5_read_channel]]
- **`par`** — Unstructured data; a dict or a list of dicts with the same number of entries as `chan`. Dict keys become HDF5 dataset
  names per slot; `None` values are skipped. Pass `None` to disable; default: `None`
- **`ix`** — 0-based slot indices along dimension X; scalar or vector of length `len(chan)`; default: `0 ... len(chan)-1`
- **`iy`** — 0-based slot indices along dimension Y; scalar or vector of length `len(chan)`; default: 0
- **`iz`** — 0-based slot indices along dimension Z; scalar or vector of length `len(chan)`; default: 0
- **`iw`** — 0-based slot indices along dimension W; scalar or vector of length `len(chan)`; default: 0

## Outputs:
- **`code`** — Per channel: `0` if the slot was empty, `1` if existing data was replaced; `(len(chan),)`; int64

## See also:
- [[hdf5_create_file]] (for creating a file with a custom storage layout)
- [[hdf5_reshape_layout]] (to change the layout later)
- [[hdf5_read_channel]] (for reading channel data)
MD!*/

py::array_t<py::ssize_t> hdf5_write_channel(const py::handle &fn,
                                         py::handle chan,
                                         py::handle par,
                                         py::handle ix,
                                         py::handle iy,
                                         py::handle iz,
                                         py::handle iw)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    // Convert the channel input (single dict or list of dicts) to channel objects
    std::vector<quadriga_lib::channel<double>> channels = qd_python_list2channel(chan, false); // C++ validates, naming the entry
    const arma::uword n_chan = (arma::uword)channels.size();
    if (n_chan == 0)
        throw std::invalid_argument("Input 'chan' must contain at least one channel.");

    // Parse the unstructured 'par' data: dict, list of dicts, or None
    std::vector<py::dict> par_vec;
    if (!par.is_none())
    {
        if (py::isinstance<py::dict>(par))
            par_vec.emplace_back(py::reinterpret_borrow<py::dict>(par));
        else if (py::isinstance<py::list>(par))
        {
            for (py::handle item : py::reinterpret_borrow<py::list>(par))
            {
                if (!py::isinstance<py::dict>(item))
                    throw std::invalid_argument("Each 'par' list entry must be a dict.");
                par_vec.emplace_back(py::reinterpret_borrow<py::dict>(item));
            }
        }
        else
            throw std::invalid_argument("'par' must be a dict, a list of dicts, or None.");

        if ((arma::uword)par_vec.size() != n_chan)
            throw std::invalid_argument("'par' must have the same number of entries as 'chan'.");
    }

    // Resolve a slot-index argument (scalar / vector / None) to a 0-based index vector.
    // 'ix' defaults to 0 ... n_chan-1; the others default to a scalar 0.
    auto parse_idx = [&](py::handle h, const char *name, bool is_x) -> arma::Col<unsigned>
    {
        arma::Col<unsigned> v = qd_python_numpy2arma_Col<unsigned>(h, false, false, name);
        if (v.is_empty())
        {
            if (is_x)
            {
                v.set_size(n_chan);
                for (arma::uword i = 0; i < n_chan; ++i)
                    v.at(i) = (unsigned)i;
            }
            else
                v = arma::Col<unsigned>(1, arma::fill::zeros);
        }
        if (v.n_elem != 1 && v.n_elem != n_chan)
            throw std::invalid_argument(std::string("Input '") + name +
                                        "' must be a scalar or a vector of length len(chan).");
        return v;
    };

    const arma::Col<unsigned> ix_vec = parse_idx(ix, "ix", true);
    const arma::Col<unsigned> iy_vec = parse_idx(iy, "iy", false);
    const arma::Col<unsigned> iz_vec = parse_idx(iz, "iz", false);
    const arma::Col<unsigned> iw_vec = parse_idx(iw, "iw", false);

    // Attach the unstructured 'par' data and resolve the slot of each channel (scalar broadcasting)
    arma::u32_mat slots(4, n_chan);
    for (arma::uword k = 0; k < n_chan; ++k)
    {
        quadriga_lib::channel<double> &c = channels[k];
        if (!par_vec.empty())
            for (auto item : par_vec[k])
            {
                if (item.second.is_none()) // empty / missing field -> skip
                    continue;
                std::string field_name = py::cast<std::string>(item.first);
                c.par_names.push_back(field_name);
                c.par_data.push_back(qd_python_anycast(item.second, field_name));
            }

        slots.at(0, k) = ix_vec.n_elem == 1 ? ix_vec.at(0) : ix_vec.at(k);
        slots.at(1, k) = iy_vec.n_elem == 1 ? iy_vec.at(0) : iy_vec.at(k);
        slots.at(2, k) = iz_vec.n_elem == 1 ? iz_vec.at(0) : iz_vec.at(k);
        slots.at(3, k) = iw_vec.n_elem == 1 ? iw_vec.at(0) : iw_vec.at(k);
    }

    // Write all channels (C++ checks all inputs first and creates a missing file)
    const arma::s32_vec code = quadriga_lib::hdf5_write_channel_vec(fn_s, channels, slots);
    return qd_python_copy2numpy<int, py::ssize_t>(&code);
}
