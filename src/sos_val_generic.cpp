// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib_generic_functions.hpp"

#include <cmath>
#include <vector>

// Generic C++ implementation of the sum-of-sinusoids evaluation
void qd_SOS_GENERIC(const double *fx, const double *fy, const double *fz,
                    const double *wc, const double *ws,
                    const size_t n_coeff,
                    const size_t n_gen,
                    const double *px, const double *py, const double *pz,
                    const size_t n_pos,
                    double *val)
{
    const double two_pi = 6.283185307179586;
    const long long n_pos_ll = (long long)n_pos;

#pragma omp parallel
    {
        // Cosine and sine of all sinusoids at one position, shared by all generators
        std::vector<double> c(n_coeff), s(n_coeff);

#pragma omp for
        for (long long i_pos = 0; i_pos < n_pos_ll; ++i_pos)
        {
            const size_t ip = (size_t)i_pos;
            const double x = px[ip], y = py[ip], z = pz[ip];

            for (size_t n = 0; n < n_coeff; ++n)
            {
                double t = fx[n] * x + fy[n] * y + fz[n] * z; // Phase in cycles
                t -= std::floor(t + 0.5);                     // Reduced to [-0.5, 0.5)
                c[n] = std::cos(two_pi * t), s[n] = std::sin(two_pi * t);
            }

            for (size_t g = 0; g < n_gen; ++g)
            {
                const double *pc = &wc[g * n_coeff], *ps = &ws[g * n_coeff];
                double acc = 0.0;
                for (size_t n = 0; n < n_coeff; ++n)
                    acc += c[n] * pc[n] - s[n] * ps[n];
                val[g * n_pos + ip] = acc;
            }
        }
    }
}
