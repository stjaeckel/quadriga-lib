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
- Creates the file with a default layout if it does not exist; appends to existing files otherwise
- A warning is issued if any selected slot already contains data (it is overwritten)
- Structured data is stored in single precision regardless of the input precision
- Unstructured datasets retain their numpy dtype and shape
- Each scalar index input is broadcast to all channels; each vector index must have one entry per channel
- If the file does not exist, the layout is `[max(len(chan), max(ix)+1), max(iy)+1, max(iz)+1, max(iw)+1]`
- Channel dict field layout matches [[hdf5_read_channel]]
- Per-snapshot fields accept the list model (list of arrays) or the stack model
  (single array with the snapshot index as the last axis), detected per field
- Coefficients may be passed as a complex `coeff`, or as separate real `coeff_re` and `coeff_im` (the two forms are mutually exclusive)
- Slot indices are 0-based

## Usage:
```
storage_dims = quadriga_lib.channel.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; str
- **`chan`** — Structured channel data; a channel dict or a list of channel dicts; field layout matches [[hdf5_read_channel]]
- **`par`** — Unstructured data; a dict or a list of dicts with the same number of entries as `chan`. Dict keys become HDF5 dataset
  names per slot; `None` values are skipped. Pass `None` to disable; default: `None`
- **`ix`** — 0-based slot indices along dimension X; scalar or vector of length `len(chan)`; default: `0 ... len(chan)-1`
- **`iy`** — 0-based slot indices along dimension Y; scalar or vector of length `len(chan)`; default: 0
- **`iz`** — 0-based slot indices along dimension Z; scalar or vector of length `len(chan)`; default: 0
- **`iw`** — 0-based slot indices along dimension W; scalar or vector of length `len(chan)`; default: 0

## Outputs:
- **`storage_dims`** — Storage layout dimensions of the file `[nx, ny, nz, nw]`; `(4,)`; uint32

## See also:
- [[hdf5_create_file]] (for creating a file with a custom storage layout)
- [[hdf5_reshape_layout]] (to change the layout later)
- [[hdf5_read_channel]] (for reading channel data)
MD!*/

py::array_t<unsigned> hdf5_write_channel(const std::string &fn,
                                         py::handle chan,
                                         py::handle par,
                                         py::handle ix,
                                         py::handle iy,
                                         py::handle iz,
                                         py::handle iw)
{
    // Convert the channel input (single dict or list of dicts) to channel objects
    std::vector<quadriga_lib::channel<double>> channels = qd_python_list2channel(chan, true);
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
        arma::Col<unsigned> v = qd_python_numpy2arma_Col<unsigned>(h, false);
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

    // Read the storage layout (returns [0,0,0,0] if the file does not exist)
    arma::Col<unsigned> storage_space = quadriga_lib::hdf5_read_layout(fn);

    // Create the file with a default layout if it does not exist
    if (storage_space.at(0) == 0)
    {
        unsigned nx = std::max((unsigned)n_chan, ix_vec.max() + 1u);
        unsigned ny = iy_vec.max() + 1u;
        unsigned nz = iz_vec.max() + 1u;
        unsigned nw = iw_vec.max() + 1u;
        quadriga_lib::hdf5_create(fn, nx, ny, nz, nw);
        storage_space.at(0) = nx;
        storage_space.at(1) = ny;
        storage_space.at(2) = nz;
        storage_space.at(3) = nw;
    }

    // Write each channel to its slot
    bool any_overwrite = false;
    for (arma::uword k = 0; k < n_chan; ++k)
    {
        quadriga_lib::channel<double> &c = channels[k];

        // Attach the unstructured 'par' data for this channel
        if (!par_vec.empty())
            for (auto item : par_vec[k])
            {
                if (item.second.is_none()) // empty / missing field -> skip
                    continue;
                std::string field_name = py::cast<std::string>(item.first);
                c.par_names.push_back(field_name);
                c.par_data.push_back(qd_python_anycast(item.second, field_name));
            }

        // Resolve the slot location with scalar broadcasting
        unsigned cx = ix_vec.n_elem == 1 ? ix_vec.at(0) : ix_vec.at(k);
        unsigned cy = iy_vec.n_elem == 1 ? iy_vec.at(0) : iy_vec.at(k);
        unsigned cz = iz_vec.n_elem == 1 ? iz_vec.at(0) : iz_vec.at(k);
        unsigned cw = iw_vec.n_elem == 1 ? iw_vec.at(0) : iw_vec.at(k);

        // Write (indices are 0-based, passed through unchanged)
        int return_code = quadriga_lib::hdf5_write(&c, fn, cx, cy, cz, cw);
        if (return_code == 1)
            any_overwrite = true;
    }

    // Warn if existing data was overwritten
    if (any_overwrite)
        py::module_::import("warnings")
            .attr("warn")("Modifying or overwriting existing dataset in file.");

    // Return the storage layout
    return qd_python_copy2numpy(&storage_space);
}

// pybind11 declaration:
// m.def("hdf5_write_channel", &hdf5_write_channel,
//       py::arg("fn"),
//       py::arg("chan"),
//       py::arg("par") = py::none(),
//       py::arg("ix") = py::none(),
//       py::arg("iy") = py::none(),
//       py::arg("iz") = py::none(),
//       py::arg("iw") = py::none());