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

#include "quadriga_channel.hpp"

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
                *dims = (unsigned long long)data->length();
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
