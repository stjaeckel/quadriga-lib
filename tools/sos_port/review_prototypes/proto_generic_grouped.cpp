// GENERIC variants. (a) sincos in one call, (b) accumulation of 4 generators at a time; the order of the
// additions per (position, generator) is unchanged, so the results must be bit-identical to qd_SOS_GENERIC.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cmath>
#include <cstddef>
#include <vector>
void var_SOS_GENERIC(const double *fx, const double *fy, const double *fz, const double *wc, const double *ws,
                     const size_t n_coeff, const size_t n_gen, const double *px, const double *py, const double *pz,
                     const size_t n_pos, double *val, int use_sincos)
{
    const double two_pi = 6.283185307179586;
    const long long n_pos_ll = (long long)n_pos;
#pragma omp parallel
    {
        std::vector<double> c(n_coeff), s(n_coeff);
#pragma omp for
        for (long long i_pos = 0; i_pos < n_pos_ll; ++i_pos)
        {
            const size_t ip = (size_t)i_pos;
            const double x = px[ip], y = py[ip], z = pz[ip];
            for (size_t n = 0; n < n_coeff; ++n)
            {
                double t = fx[n] * x + fy[n] * y + fz[n] * z;
                t -= std::floor(t + 0.5);
                if (use_sincos)
                    sincos(two_pi * t, &s[n], &c[n]);
                else
                    c[n] = std::cos(two_pi * t), s[n] = std::sin(two_pi * t);
            }
            size_t g = 0;
            for (; g + 4 <= n_gen; g += 4)
            {
                const double *c0 = &wc[g * n_coeff], *s0 = &ws[g * n_coeff], *c1 = c0 + n_coeff, *s1 = s0 + n_coeff;
                const double *c2 = c1 + n_coeff, *s2 = s1 + n_coeff, *c3 = c2 + n_coeff, *s3 = s2 + n_coeff;
                double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
                for (size_t n = 0; n < n_coeff; ++n)
                {
                    const double cn = c[n], sn = s[n];
                    a0 += cn * c0[n] - sn * s0[n], a1 += cn * c1[n] - sn * s1[n];
                    a2 += cn * c2[n] - sn * s2[n], a3 += cn * c3[n] - sn * s3[n];
                }
                val[g * n_pos + ip] = a0, val[(g + 1) * n_pos + ip] = a1, val[(g + 2) * n_pos + ip] = a2, val[(g + 3) * n_pos + ip] = a3;
            }
            for (; g < n_gen; ++g)
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
