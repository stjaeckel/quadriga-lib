// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Helpers shared by the sources of the site-specific simulation tools (ray tracing, meshes, point clouds)

#ifndef qd_helpers_site_hpp
#define qd_helpers_site_hpp

#include <armadillo>
#include <cmath>

// Cross product
template <typename dtype>
static inline void crossp(dtype vx, dtype vy, dtype vz,    // Vector V
                          dtype kx, dtype ky, dtype kz,    // Vector K
                          dtype *rx, dtype *ry, dtype *rz, // Result V x K
                          bool normalize_output = false)
{
    *rx = vy * kz - vz * ky; // x-component
    *ry = vz * kx - vx * kz; // y-component
    *rz = vx * ky - vy * kx; // z-component

    if (normalize_output)
    {
        dtype scl = (dtype)1.0 / std::sqrt(*rx * *rx + *ry * *ry + *rz * *rz);
        *rx *= scl, *ry *= scl, *rz *= scl;
    }
}

// Dot product
template <typename dtype>
static inline dtype dotp(dtype vx, dtype vy, dtype vz, // Vector V
                         dtype kx, dtype ky, dtype kz, // Vector K
                         bool normalize = false)       // Option to normalize
{
    dtype dot = vx * kx + vy * ky + vz * kz; // Dot product calculation

    if (normalize)
    {
        dtype v_mag = std::sqrt(vx * vx + vy * vy + vz * vz); // Magnitude of V
        dtype k_mag = std::sqrt(kx * kx + ky * ky + kz * kz); // Magnitude of K
        if (v_mag > 0 && k_mag > 0)                           // Normalize the dot product
            dot /= (v_mag * k_mag);
    }

    return dot;
}

// Repeat sequence of values + Optional typecast
template <typename dtypeIn, typename dtypeOut>
static void qd_repeat_sequence(const dtypeIn *sequence, size_t sequence_length, size_t repeat_value, size_t repeat_sequence, dtypeOut *output)
{
    size_t pos = 0;                                  // Position in output
    for (size_t rs = 0; rs < repeat_sequence; ++rs)  // Repeat sequence of values
        for (size_t v = 0; v < sequence_length; ++v) // Iterate through all values of the sequence
        {
            dtypeOut val = (dtypeOut)sequence[v];        // Type conversion
            for (size_t rv = 0; rv < repeat_value; ++rv) // Repeat each value
                output[pos++] = val;
        }
}

#endif
