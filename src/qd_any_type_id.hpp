// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Library-internal: type ID, shape and a read-only pointer to the data of the value in a std::any

#ifndef quadriga_lib_any_type_id_H
#define quadriga_lib_any_type_id_H

#include <armadillo>
#include <any>
#include <string>

// Codes 10 + k (scalar), 20 + k (arma::Mat), 30 + k (arma::Cube), 40 + k (arma::Col), 50 + k (arma::Row) for the
// element type E with index k; -1 if 'a' holds none of them
template <typename E>
static inline int qd_any_type_id_elem(const std::any *a, int k, unsigned long long *dims, const void **data)
{
    if (const E *x = std::any_cast<E>(a))
        return *data = x, 10 + k;
    if (const arma::Mat<E> *x = std::any_cast<arma::Mat<E>>(a))
        return dims[0] = x->n_rows, dims[1] = x->n_cols, *data = x->memptr(), 20 + k;
    if (const arma::Cube<E> *x = std::any_cast<arma::Cube<E>>(a))
        return dims[0] = x->n_rows, dims[1] = x->n_cols, dims[2] = x->n_slices, *data = x->memptr(), 30 + k;
    if (const arma::Col<E> *x = std::any_cast<arma::Col<E>>(a))
        return dims[0] = x->n_elem, *data = x->memptr(), 40 + k;
    if (const arma::Row<E> *x = std::any_cast<arma::Row<E>>(a))
        return dims[0] = x->n_elem, *data = x->memptr(), 50 + k;
    return -1;
}

// Type ID of the value in a std::any (codes of quadriga_lib::any_type_id), its shape and a pointer to its data
// - dims: optional, 3 values [rows, cols, slices]; vectors give [n_elem, 1, 1], a string [length, 0, 0]
// - data: optional, the data of the contained object (characters of a string); cast it to the type of the code
static inline int qd_any_type_id(const std::any *a, unsigned long long *dims = nullptr, const void **data = nullptr)
{
    if (a == nullptr || !a->has_value())
        return -2;

    unsigned long long d[3];
    const void *p = nullptr;
    dims = dims == nullptr ? d : dims, data = data == nullptr ? &p : data;
    dims[0] = 1ULL, dims[1] = 1ULL, dims[2] = 1ULL;

    if (const std::string *s = std::any_cast<std::string>(a))
        return dims[0] = s->length(), dims[1] = 0ULL, dims[2] = 0ULL, *data = s->data(), 9;

    int id = -1;
    (void)((id = qd_any_type_id_elem<float>(a, 0, dims, data)) >= 0 ||
           (id = qd_any_type_id_elem<double>(a, 1, dims, data)) >= 0 ||
           (id = qd_any_type_id_elem<unsigned long long>(a, 2, dims, data)) >= 0 ||
           (id = qd_any_type_id_elem<long long>(a, 3, dims, data)) >= 0 ||
           (id = qd_any_type_id_elem<unsigned>(a, 4, dims, data)) >= 0 ||
           (id = qd_any_type_id_elem<int>(a, 5, dims, data)) >= 0);
    return id;
}

#endif
