// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# any_type_id
Get type ID and raw access from a `std::any` object

- Inspects a `std::any` object and returns an integer type identifier for its contents
- Optionally retrieves dimensions (rows, columns, slices) for Armadillo matrix/cube/vector types; for `std::string`, `dims[0]` is the string length, `dims[1]`/`dims[2]` are zero
- Optionally retrieves a raw `void*` to the internal data — not type-safe, bypasses `const` protection; use with caution

## Declaration:
```
int quadriga_lib::any_type_id(
    const std::any *data,
    unsigned long long *dims = nullptr,
    void **dataptr = nullptr);
```

## Inputs:
- **`data`** — Pointer to the `std::any` object to inspect

## Outputs:
- **`dims`** *(optional)* — Array of 3 values filled with `[rows, cols, slices]` of the contained Armadillo object
- **`dataptr`** *(optional)* — Receives a raw pointer to the object's internal data

## Returns:
- Integer type ID of the contained value:<br><br>
  | ID  | Type                      | ID  | Type                   | ID  | Type                      |
  | --- | ------------------------- | --- | ---------------------- | --- | ------------------------- |
  | -2  | `no value`                | -1  | `unsupported type`     | 9   | `std::string`             |
  | 10  | `float`                   | 11  | `double`               | 12  | `unsigned long long int`  |
  | 13  | `long long int`           | 14  | `unsigned int`         | 15  | `int`                     |
  | 20  | `arma::Mat<float>`        | 21  | `arma::Mat<double>`    | 22  | `arma::Mat<arma::uword>`  |
  | 23  | `arma::Mat<arma::sword>`  | 24  | `arma::Mat<unsigned>`  | 25  | `arma::Mat<int>`          |
  | 30  | `arma::Cube<float>`       | 31  | `arma::Cube<double>`   | 32  | `arma::Cube<arma::uword>` |
  | 33  | `arma::Cube<arma::sword>` | 34  | `arma::Cube<unsigned>` | 35  | `arma::Cube<int>`         |
  | 40  | `arma::Col<float>`        | 41  | `arma::Col<double>`    | 42  | `arma::Col<arma::uword>`  |
  | 43  | `arma::Col<arma::sword>`  | 44  | `arma::Col<unsigned>`  | 45  | `arma::Col<int>`          |
  | 50  | `arma::Row<float>`        | 51  | `arma::Row<double>`    | 52  | `arma::Row<arma::uword>`  |
  | 53  | `arma::Row<arma::sword>`  | 54  | `arma::Row<unsigned>`  | 55  | `arma::Row<int>`          |

## See also:
- [[hdf5_read_dset]] (uses `any_type_id` to read dataset from HDF5 file)
- [[hdf5_write_dset]] (HDF5 dataset writer)
MD!*/

// Returns type ID of a std::any field:
int quadriga_lib::any_type_id(const std::any *par_data, unsigned long long *dims, void **dataptr)
{
    if (par_data == nullptr || !par_data->has_value())
        return -2;

    if (dims != nullptr) // Set dims to scalar type
        dims[0] = 1ULL, dims[1] = 1ULL, dims[2] = 1ULL;

    if (par_data->type().name() == typeid(std::string).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<std::string>(par_data);
            if (dims != nullptr)
                dims[0] = (unsigned long long)data->length(), dims[1] = 0ULL, dims[2] = 0ULL;
            if (dataptr != nullptr)
                *dataptr = (void *)data->c_str();
        }
        return 9;
    }

    // Scalar types
    if (par_data->type().name() == typeid(float).name())
    {
        if (dataptr != nullptr)
            *dataptr = (void *)std::any_cast<float>(par_data);
        return 10;
    }

    if (par_data->type().name() == typeid(double).name())
    {
        if (dataptr != nullptr)
            *dataptr = (void *)std::any_cast<double>(par_data);
        return 11;
    }

    if (par_data->type().name() == typeid(unsigned long long int).name())
    {
        if (dataptr != nullptr)
            *dataptr = (void *)std::any_cast<unsigned long long int>(par_data);
        return 12;
    }

    if (par_data->type().name() == typeid(long long int).name())
    {
        if (dataptr != nullptr)
            *dataptr = (void *)std::any_cast<long long int>(par_data);
        return 13;
    }

    if (par_data->type().name() == typeid(unsigned int).name())
    {
        if (dataptr != nullptr)
            *dataptr = (void *)std::any_cast<unsigned int>(par_data);
        return 14;
    }

    if (par_data->type().name() == typeid(int).name())
    {
        if (dataptr != nullptr)
            *dataptr = (void *)std::any_cast<int>(par_data);
        return 15;
    }

    // Matrix types
    if (par_data->type().name() == typeid(arma::Mat<float>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Mat<float>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 20;
    }

    if (par_data->type().name() == typeid(arma::Mat<double>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Mat<double>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 21;
    }

    if (par_data->type().name() == typeid(arma::Mat<unsigned long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Mat<unsigned long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 22;
    }

    if (par_data->type().name() == typeid(arma::Mat<long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Mat<long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 23;
    }

    if (par_data->type().name() == typeid(arma::Mat<unsigned int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Mat<unsigned int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 24;
    }

    if (par_data->type().name() == typeid(arma::Mat<int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Mat<int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 25;
    }

    // Cube types
    if (par_data->type().name() == typeid(arma::Cube<float>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Cube<float>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols, dims[2] = data->n_slices;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 30;
    }

    if (par_data->type().name() == typeid(arma::Cube<double>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Cube<double>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols, dims[2] = data->n_slices;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 31;
    }

    if (par_data->type().name() == typeid(arma::Cube<unsigned long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Cube<unsigned long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols, dims[2] = data->n_slices;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 32;
    }

    if (par_data->type().name() == typeid(arma::Cube<long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Cube<long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols, dims[2] = data->n_slices;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 33;
    }

    if (par_data->type().name() == typeid(arma::Cube<unsigned int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Cube<unsigned int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols, dims[2] = data->n_slices;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 34;
    }

    if (par_data->type().name() == typeid(arma::Cube<int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Cube<int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols, dims[2] = data->n_slices;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 35;
    }

    // Column vectors
    if (par_data->type().name() == typeid(arma::Col<float>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Col<float>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 40;
    }

    if (par_data->type().name() == typeid(arma::Col<double>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Col<double>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 41;
    }

    if (par_data->type().name() == typeid(arma::Col<unsigned long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Col<unsigned long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 42;
    }

    if (par_data->type().name() == typeid(arma::Col<long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Col<long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 43;
    }

    if (par_data->type().name() == typeid(arma::Col<unsigned int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Col<unsigned int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 44;
    }

    if (par_data->type().name() == typeid(arma::Col<int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Col<int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 45;
    }

    // Row vectors
    if (par_data->type().name() == typeid(arma::Row<float>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Row<float>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 50;
    }

    if (par_data->type().name() == typeid(arma::Row<double>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Row<double>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 51;
    }

    if (par_data->type().name() == typeid(arma::Row<unsigned long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Row<unsigned long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 52;
    }

    if (par_data->type().name() == typeid(arma::Row<long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Row<long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 53;
    }

    if (par_data->type().name() == typeid(arma::Row<unsigned int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Row<unsigned int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 54;
    }

    if (par_data->type().name() == typeid(arma::Row<int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Row<int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 55;
    }
    return -1;
}
