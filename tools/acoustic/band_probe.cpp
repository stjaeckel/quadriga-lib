// SPDX-License-Identifier: Apache-2.0
// Part of quadriga-lib — see LICENSE for terms.
//
// Band probe for the acoustic add-on (md/acoustic.md 2.8, 11): one ray through ray_progress until no ray is left, a band
// of frequencies on the base frequency's geometry (the first one), every path that ends in air outside the structure
// summed at its polyline's point: T beyond the last face, R in front of the first. Scalar mode; tracer 5 =
// refraction_mode. Faces are planes x = const (normal towards -x, or +x when flipped), as ub::build makes them.
// Written for the M9 work (md/m9_fixtures_report.md, item 3): a band equals its single frequencies bitwise on a leaf on
// a box of air; the fixture is "ray_progress - a leaf on a box of air at every frequency of a band".
//
// stdin, one line per run: tag itype theta maxdepth start sink nm (a b c d e f g h thk tf tfB m fRef resF resQ resS att
// coiF coiQ alpha tfP tfF tfQ)*nm nf (x mat flip)*nf thr maxint floor_dB nfreq f_1 .. f_nfreq
//   itype 4 or 5; theta in degrees; maxdepth, start and sink are read and ignored; materials are numbered from 1 in
//   the order given (columns as in Material, ray_mesh_interact.cpp); thr the thin_slab_threshold, maxint the
//   interaction cap (at most 255), floor_dB the gain floor; frequencies in acoustic Hz (mapped by 8.7403e5).
//   tfP tfF tfQ: the peak of md/acoustic.md 2.6 (tfF in GHz like resF and coiF; 0 0 0 for none).
// stdout: P,tag,theta,tracer,f_base,f,nT,nR,rays,events,Tre,Tim,Rre,Rim (one line per frequency of the band).
// Build: tools/acoustic/build_band_probe.sh <output binary>, from the repository root.
#include "quadriga_lib.hpp"
#include <complex>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <omp.h>
int main()
{
    omp_set_num_threads(1);
    const double PI = 3.14159265358979323846, C0 = 299792458.0, FMAP = 8.7403e5;
    const std::complex<double> J(0.0, 1.0);
    typedef std::unordered_map<std::string, std::vector<float>> Pal;
    const char *COLS[] = {"a", "b", "c", "d", "e", "f", "g", "h", "thk", "tf", "tfB", "m", "fRef", "resF", "resQ", "resS", "att", "coiF", "coiQ", "alpha", "tfP", "tfF", "tfQ"};
    printf("P,tag,theta,tracer,f_base,f,nT,nR,rays,events,Tre,Tim,Rre,Rim\n");
    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ss(line);
        std::string tag;
        int itype, maxdepth, start, sink, nm, nfc;
        double thd;
        ss >> tag >> itype >> thd >> maxdepth >> start >> sink >> nm;
        Pal pal;
        for (const char *k : COLS)
            pal[k] = std::vector<float>(nm, 0.0f);
        for (int q = 0; q < nm; ++q)
            for (const char *k : COLS)
            {
                double v;
                ss >> v;
                pal[k][q] = (float)v;
            }
        ss >> nfc;
        std::vector<float> x(nfc);
        std::vector<int> mat(nfc), flip(nfc);
        for (int q = 0; q < nfc; ++q)
            ss >> x[q] >> mat[q] >> flip[q];
        double thr, floor_dB;
        int maxint, nf;
        ss >> thr >> maxint >> floor_dB >> nf;
        std::vector<double> fq(nf);
        for (int q = 0; q < nf; ++q)
            ss >> fq[q];
        if (ss.fail())
        {
            fprintf(stderr, "bad line\n");
            continue;
        }
        const float h = 200.0f;
        arma::fmat mesh(2 * nfc, 9);
        arma::uvec mi(2 * nfc);
        for (int k = 0; k < nfc; ++k)
        {
            if (!flip[k])
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], -h, h, x[k], h, -h}, mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], -h, h, x[k], h, h};
            else
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], h, -h, x[k], -h, h}, mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], h, h, x[k], -h, h};
            mi(2 * k) = (arma::uword)mat[k], mi(2 * k + 1) = (arma::uword)mat[k];
        }
        const float x_front = *std::min_element(x.begin(), x.end()), x_back = *std::max_element(x.begin(), x.end());
        const double th = thd * PI / 180.0, st = std::sin(th);
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        arma::frowvec kv = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf(nf);
        for (int q = 0; q < nf; ++q)
            cf(q) = (float)(fq[q] * FMAP);
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 1 + nf, arma::fill::zeros);
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * kv, dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * kv;
        arma::fcube pd(1, 3, nf);
        for (int q = 0; q < nf; ++q)
            pd.slice(q) = kv;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, nf, true);
        paths[0].set_length(r0);
        std::vector<std::complex<double>> T(nf, 0.0), R(nf, 0.0);
        int nT = 0, nR = 0;
        long rays = 0;
        for (int g = 0; g < 4000; ++g)
        {
            quadriga_lib::ray_progress(mesh, mi, pal, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths, nullptr, nullptr, nullptr, nullptr,
                                       (uint8_t)maxint, 255, 255, 0, (float)floor_dB, 3.0f, (float)thr, itype == 5, true, nullptr, nullptr, nullptr, nullptr);
            if (orig.n_rows == 0)
                break;
            rays += (long)orig.n_rows;
            for (arma::uword i = 0; i < orig.n_rows; ++i)
            {
                const bool fwd = dest(i, 0) > orig(i, 0);
                const size_t ns = paths[i].n_seg();
                if ((cur(i) & (short)0x3FFF) != 0 || ns != (size_t)(g + 1))
                    continue;
                const float *first = paths[i].coord(0), *last = paths[i].coord(ns - 1);
                const bool isT = fwd && last[0] > x_back - 1.0e-3f, isR = !fwd && last[0] < x_front + 1.0e-3f;
                if (!isT && !isR)
                    continue;
                double Lin = 0.0;
                for (size_t s = 1; s < ns; ++s)
                {
                    const float *p = paths[i].coord(s - 1), *q = paths[i].coord(s);
                    Lin += std::sqrt(((double)q[0] - p[0]) * ((double)q[0] - p[0]) + ((double)q[1] - p[1]) * ((double)q[1] - p[1]) + ((double)q[2] - p[2]) * ((double)q[2] - p[2]));
                }
                for (int q = 0; q < nf; ++q)
                {
                    const double k = 2.0 * PI * fq[q] * FMAP / C0;
                    const float *xc = paths[i].xpr_coeff(q);
                    std::complex<double> F = std::complex<double>((double)xc[0], (double)xc[1]) * std::exp(-J * k * Lin) * std::exp(J * k * st * ((double)last[1] - (double)first[1]));
                    if (isT)
                        T[q] += F;
                    else
                        R[q] += F;
                }
                if (isT)
                    ++nT;
                else
                    ++nR;
            }
        }
        for (int q = 0; q < nf; ++q)
            printf("P,%s,%.4f,%d,%.6f,%.6f,%d,%d,%ld,%ld,%.9g,%.9g,%.9g,%.9g\n", tag.c_str(), thd, itype, fq[0], fq[q], nT, nR, rays, rays,
                   T[q].real(), T[q].imag(), R[q].real(), R[q].imag());
        fflush(stdout);
    }
    return 0;
}
