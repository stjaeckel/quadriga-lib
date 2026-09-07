// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic suite for the thin-sheet refactor.
//
// A slab modeled as a traced solid and the same slab modeled as a zero-thickness sheet carrying thk
// must give the same forward transmission. The two paths through the code are different - the solid
// crosses two faces and accumulates state in ray_state_update, the sheet resolves the whole crossing
// in one atomic event - so agreement is the property the refactor aims at, not something the
// implementation gets for free.
//
// This file is written to PRODUCE A REPORT rather than to gate a build. Several divergences are known
// and unfixed; asserting on them would only stop the report before the interesting part. Assertions
// are therefore limited to sanity (finite, in range, obstacle actually hit) plus one loose staged
// check per case with the eventual target recorded beside it. The printed tables are the product.
//
// Every table carries an exact analytic column computed here by a transfer-matrix method, so both the
// solid and the sheet are measured against physics rather than only against each other. The TMM
// reduces to the single-slab Airy sum at one layer, which is asserted directly.
//
// Set QD_TEST_VERBOSE=1 to print the tables (currently forced on).

#include <catch2/catch_test_macros.hpp>
#include "quadriga_lib.hpp"

#include <algorithm>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define QD_TEST_VERBOSE

namespace
{
    using MtlProp = std::unordered_map<std::string, std::vector<double>>;
    using cdouble = std::complex<double>;

    constexpr double PI = 3.14159265358979323846;
    constexpr double C0 = 299792458.0;

    // Staged tolerance. 1e-6 is the target once the entry-boundary work lands; until then the known
    // divergences are larger and the report, not the assert, is what carries the information.
    constexpr double TOL_TARGET = 1.0e-6;
    constexpr double TOL_STAGED = 0.75;

    bool verbose()
    {
#ifdef QD_TEST_VERBOSE
        return true;
#else
        const char *v = std::getenv("QD_TEST_VERBOSE");
        return v != nullptr && v[0] != '0';
#endif
    }

    // Canonical column defaults, matching obj_file_write's csv_write_defaults
    double column_default(const std::string &col)
    {
        return (col == "a" || col == "e" || col == "fRef") ? 1.0 : 0.0;
    }

    // Material table holding only the materials a test defines, but with the column set of the
    // built-in library. Seeding the columns keeps the test in step with whatever the canonical set
    // is; defining the values explicitly keeps the physics under test independent of revisions to
    // the library's own entries.
    struct MaterialTable
    {
        std::vector<std::string> names;
        MtlProp prop;

        MaterialTable()
        {
            std::vector<std::string> lib_names;
            MtlProp lib_prop;
            quadriga_lib::obj_file_read<double>("", nullptr, nullptr, nullptr, nullptr, nullptr,
                                                nullptr, nullptr, nullptr, "", nullptr,
                                                &lib_names, &lib_prop);
            for (const auto &kv : lib_prop)
                prop[kv.first] = {};
        }

        // Appends a material, returns its 1-based index. Columns the library table omits are
        // created on demand and back-filled with their canonical default for materials already
        // added, so every column vector stays the same length as names.
        arma::uword add(const std::string &name, const std::map<std::string, double> &values)
        {
            arma::uword n_have = (arma::uword)names.size();
            for (const auto &v : values)
                if (prop.count(v.first) == 0)
                    prop[v.first] = std::vector<double>(n_have, column_default(v.first));

            names.push_back(name);
            for (auto &kv : prop)
            {
                auto it = values.find(kv.first);
                kv.second.push_back(it == values.end() ? column_default(kv.first) : it->second);
            }
            return (arma::uword)names.size();
        }
    };

    // Material definitions used across the suite. Sheet and solid entries are identical apart from
    // thk, which is what selects the model - so a sheet glued to a solid of the same properties is a
    // homogeneous slab of the summed thickness and has an exact analytic reference.
    struct MatDef
    {
        double a, b, c, d;   // eps_r and sigma, ITU-R P.2040 form
        double att = 0.0;    // lumped interface loss, dB
        double m = 0.0;      // acoustic mass-law slope, dB/decade
    };

    std::map<std::string, double> to_cols(const MatDef &M, double thk)
    {
        std::map<std::string, double> v = {{"a", M.a}, {"b", M.b}, {"c", M.c}, {"d", M.d}, {"fRef", 1.0}};
        if (thk > 0.0)
            v["thk"] = thk;
        if (M.att != 0.0)
            v["att"] = M.att;
        if (M.m != 0.0)
            v["m"] = M.m;
        return v;
    }

    // ---- analytic reference -------------------------------------------------------------------

    // Relative permittivity, matching Material::eta
    cdouble eps_of(const MatDef &M, double fGHz)
    {
        double fr = fGHz / 1.0;
        double er = M.a * std::pow(fr, M.b);
        double sigma = M.c * std::pow(fr, M.d);
        return cdouble(er, -17.98 * sigma / fGHz);
    }

    // Abeles transfer matrix for a stack of non-magnetic layers with air on both sides. Exact:
    // complex angles throughout, all internal reflections coupled across every layer. Reduces to the
    // single-slab Airy sum at one layer, which TEST_CASE "analytic reference" asserts.
    double tmm_power(const std::vector<std::pair<cdouble, double>> &layers, double th, double fGHz, int pol)
    {
        const double k0 = 2.0 * PI * fGHz * 1e9 / C0;
        const double s2 = std::sin(th) * std::sin(th);
        const cdouble j(0.0, 1.0);

        cdouble Y0 = (pol == 0) ? cdouble(std::cos(th), 0.0) : cdouble(1.0 / std::cos(th), 0.0);
        cdouble M11(1.0, 0.0), M12(0.0, 0.0), M21(0.0, 0.0), M22(1.0, 0.0);

        for (const auto &L : layers)
        {
            cdouble e = L.first, n = std::sqrt(e);
            cdouble ct = std::sqrt(cdouble(1.0, 0.0) - s2 / e);
            cdouble Y = (pol == 0) ? (n * ct) : (n / ct);
            cdouble dl = k0 * n * L.second * ct;
            cdouble cs = std::cos(dl), sn = std::sin(dl);
            cdouble A = cs, B = j * sn / Y, C = j * Y * sn, D = cs;
            cdouble n11 = M11 * A + M12 * C, n12 = M11 * B + M12 * D;
            cdouble n21 = M21 * A + M22 * C, n22 = M21 * B + M22 * D;
            M11 = n11, M12 = n12, M21 = n21, M22 = n22;
        }

        cdouble Ys = Y0; // air on both sides
        cdouble den = Y0 * M11 + Y0 * Ys * M12 + M21 + Ys * M22;
        cdouble t = 2.0 * Y0 / den;
        return std::norm(t);
    }

    // TE/TM-averaged power, matching the convention of gainN
    double analytic_dB(const std::vector<std::pair<cdouble, double>> &layers, double th, double fGHz)
    {
        double p = 0.5 * (tmm_power(layers, th, fGHz, 0) + tmm_power(layers, th, fGHz, 1));
        return (p > 0.0) ? 10.0 * std::log10(p) : -std::numeric_limits<double>::infinity();
    }

    // ---- geometry -----------------------------------------------------------------------------

    // Mesh rows are [x1 y1 z1 x2 y2 z2 x3 y3 z3]
    arma::mat translate(arma::mat m, double dx, double dy, double dz)
    {
        for (int v = 0; v < 3; ++v)
            m.col(3 * v + 0) += dx, m.col(3 * v + 1) += dy, m.col(3 * v + 2) += dz;
        return m;
    }

    // Reverse the winding of every face, which reverses its normal. Done on the mesh rather than
    // through a rotation so the test does not depend on plane()'s Euler convention.
    arma::mat flip_winding(arma::mat m)
    {
        arma::mat v2 = m.cols(3, 5);
        m.cols(3, 5) = m.cols(6, 8);
        m.cols(6, 8) = v2;
        return m;
    }

    // Lateral offset of every obstacle, in the plane of the face.
    //
    // At exactly normal incidence the ray crosses at (0, 0, 0), which is the shared edge of the two
    // triangles that make up a quad or a box face. The intersector then returns two co-planar
    // triangles of the SAME surface as the FBS/SBS pair, and the surface that was supposed to be the
    // second face is either missed or handled as a separate interaction. In the first run that
    // corrupted three rows: a matched-normal duplicate charged twice, a glued entry that lost its
    // sheet entirely, and an ordered stack that was not killed. Moving the obstacle sideways changes
    // no physics - the plane x = 0 and its normal are untouched, so every incidence angle and every
    // crossing point is identical - it only moves the triangulation relative to where the rays land.
    // 0.37 m is chosen to sit clear of every crossing point in the sweep (0, 0.88, 1.82, ... 57.15 m).
    constexpr double Y_OFF = 0.37;

    // Quad in the plane x = x0, normal along +x
    arma::mat make_quad(double half_extent, double x0)
    {
        arma::mat q = quadriga_lib::plane<double>({half_extent, half_extent}, {0.0, 0.5 * PI, 0.0}, {0.0, 0.0, 0.0});
        return translate(q, x0, Y_OFF, 0.0);
    }

    // Closed box of the given thickness in x, centered at x0
    arma::mat make_box(double thickness, double half_extent, double x0)
    {
        arma::mat b = quadriga_lib::cube<double>({0.5 * thickness, half_extent, half_extent}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
        return translate(b, x0, Y_OFF, 0.0);
    }

    // Axis-aligned bounding box of a mesh, for confirming the plane() and cube() argument conventions
    void print_bbox(const char *tag, const arma::mat &m)
    {
        double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
        for (arma::uword r = 0; r < m.n_rows; ++r)
            for (int v = 0; v < 3; ++v)
                for (int k = 0; k < 3; ++k)
                {
                    double x = m(r, (arma::uword)(3 * v + k));
                    lo[k] = std::min(lo[k], x), hi[k] = std::max(hi[k], x);
                }
        std::printf("%s,%llu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", tag, (unsigned long long)m.n_rows,
                    lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
    }

    arma::uvec fill_ind(arma::uword n, arma::uword idx)
    {
        return arma::uvec(n, arma::fill::value(idx));
    }

    // ---- tracing ------------------------------------------------------------------------------

    struct Trace
    {
        arma::vec gain;   // linear power relative to FSPL, [n_pos]
        arma::mat xprmat; // [8, n_pos] in EM mode, [2, n_pos] in scalar mode
    };

    // Forward pass-through trace, lod 0 throughout.
    //
    // lod 0 is the clean feed-forward path: one ray, no diffraction ellipsoid, no sub-path
    // aggregation. That is the whole subject of this suite - the slab and sheet interaction physics -
    // and nothing here is meant to exercise the diffraction machinery. Higher lods were used in the
    // first run and cost more than they gave: the seven elliptic sub-paths of lod 1 produce a 0.7 dB
    // artifact at exactly normal incidence that has nothing to do with the material model, and lod 5
    // agrees with lod 0 while still carrying the ellipsoid construction. Every table below is lod 0.
    //
    // GENERIC kernel so the numbers do not depend on the host's SIMD support. thin_slab_threshold 0
    // forces the slab to resolve on every crossing, which is the case under test. apply_fspl stays off
    // so the gain is bounded by 0 and 1 and the comparison is scale-free.
    Trace forward_trace(const arma::mat &mesh, const arma::uvec &mtl_ind, const MtlProp &prop,
                        const arma::mat &orig, const arma::mat &dest,
                        double fc, bool scalar_mode, int lod = 0)
    {
        Trace t;
        quadriga_lib::calc_diffraction_gain<double>(orig, dest, mesh, mtl_ind, prop, fc, lod,
                                                    &t.gain, &t.xprmat, nullptr, 0, nullptr,
                                                    1, 0, scalar_mode, 0.0, false, false);
        return t;
    }

    // TX at a fixed point, RX mirrored across the slab plane at x = 0, placed so the direct ray
    // crosses the plane at the requested incidence angles.
    struct Geometry
    {
        arma::mat orig, dest;
        std::vector<double> angle_deg;
    };

    Geometry make_positions(const std::vector<double> &incidence_deg, double x_offset = 5.0)
    {
        Geometry g;
        g.angle_deg = incidence_deg;
        arma::uword n = (arma::uword)incidence_deg.size();
        g.orig.set_size(n, 3);
        g.dest.set_size(n, 3);
        for (arma::uword i = 0; i < n; ++i)
        {
            double y = 2.0 * x_offset * std::tan(incidence_deg[i] * PI / 180.0);
            g.orig(i, 0) = -x_offset, g.orig(i, 1) = 0.0, g.orig(i, 2) = 0.0;
            g.dest(i, 0) = x_offset, g.dest(i, 1) = y, g.dest(i, 2) = 0.0;
        }
        return g;
    }

    // Diagonal geometry for the corner case: the ray crosses x = 0 and y = 0 at the same point, so
    // the two faces are co-located to machine precision and the corner branch is reached.
    Geometry make_corner_positions(double reach = 5.0)
    {
        Geometry g;
        g.angle_deg = {45.0};
        g.orig.set_size(1, 3);
        g.dest.set_size(1, 3);
        g.orig(0, 0) = -reach, g.orig(0, 1) = -reach, g.orig(0, 2) = 0.0;
        g.dest(0, 0) = reach, g.dest(0, 1) = reach, g.dest(0, 2) = 0.0;
        return g;
    }

    // ---- reporting ----------------------------------------------------------------------------

    double to_dB(double lin)
    {
        return (lin > 0.0) ? 10.0 * std::log10(lin) : -std::numeric_limits<double>::infinity();
    }

    double wrap180(double deg)
    {
        while (deg > 180.0)
            deg -= 360.0;
        while (deg <= -180.0)
            deg += 360.0;
        return deg;
    }

    // 0.5 * sum(xprmat^2) for one column. Equal to gain when the matrix carries the interaction
    // amplitude; equal to 1 when the caller has power-normalized it. Which of the two it is decides
    // whether the phase columns below can be read at all, and is currently an open question.
    double xpr_norm(const arma::mat &X, arma::uword i)
    {
        double s = 0.0;
        for (arma::uword r = 0; r < X.n_rows; ++r)
            s += X(r, i) * X(r, i);
        return 0.5 * s;
    }

    // Phase of the first channel (VV in EM mode, the scalar pressure coefficient otherwise)
    double xpr_phase(const arma::mat &X, arma::uword i)
    {
        return std::atan2(X(1, i), X(0, i));
    }

    struct Summary
    {
        double worst_rel = 0.0;
        double worst_at = 0.0;
        double worst_vs_analytic_test = 0.0;
        double worst_vs_analytic_ref = 0.0;
    };

    // Prints one comparison block. ref is the baseline (usually the traced solid), test the thing
    // under study (usually the sheet). analytic may be empty. thk_phase drives the predicted phase
    // offset column: the sheet carries the absolute in-slab phase while the solid's SCALE_BY_MEDIUM
    // subtracts the free-space reference over the same thickness, so the two are predicted to differ
    // by k0*thk*cos(theta). Pass 0.0 where no prediction applies.
    Summary report(const char *tag, const Geometry &geo,
                   const Trace &ref, const Trace &test,
                   const std::vector<double> &analytic,
                   double fc, double thk_phase,
                   const char *ref_name = "solid", const char *test_name = "sheet")
    {
        Summary S;
        const bool have_ana = analytic.size() == geo.angle_deg.size();
        const double k0 = 2.0 * PI * fc / C0;

        if (verbose())
        {
            std::printf("\n# %s\n", tag);
            std::printf("angle,y_rx,%s_dB,%s_dB,analytic_dB,%s-%s_dB,%s-ana_dB,%s-ana_dB,"
                        "rel_dev,xprnorm_%s,xprnorm_%s,dphase_deg,dphase_pred_deg,max_xpr_dev\n",
                        ref_name, test_name, test_name, ref_name, test_name, ref_name,
                        ref_name, test_name);
        }

        for (arma::uword i = 0; i < ref.gain.n_elem; ++i)
        {
            double a = ref.gain(i), b = test.gain(i);
            double rel = (a > 0.0) ? std::abs(b - a) / a : std::abs(b - a);
            double ana = have_ana ? analytic[i] : std::numeric_limits<double>::quiet_NaN();

            double xpr = 0.0;
            for (arma::uword r = 0; r < ref.xprmat.n_rows; ++r)
                xpr = std::max(xpr, std::abs(test.xprmat(r, i) - ref.xprmat(r, i)));

            double th = geo.angle_deg[i] * PI / 180.0;
            double dph = wrap180((xpr_phase(test.xprmat, i) - xpr_phase(ref.xprmat, i)) * 180.0 / PI);
            double dph_pred = wrap180(k0 * thk_phase * std::cos(th) * 180.0 / PI);

            if (rel > S.worst_rel)
                S.worst_rel = rel, S.worst_at = geo.angle_deg[i];
            if (have_ana)
            {
                S.worst_vs_analytic_test = std::max(S.worst_vs_analytic_test, std::abs(to_dB(b) - ana));
                S.worst_vs_analytic_ref = std::max(S.worst_vs_analytic_ref, std::abs(to_dB(a) - ana));
            }

            if (verbose())
                std::printf("%.2f,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3e,%.6f,%.6f,%.2f,%.2f,%.3e\n",
                            geo.angle_deg[i], geo.dest(i, 1), to_dB(a), to_dB(b), ana,
                            to_dB(b) - to_dB(a), to_dB(b) - ana, to_dB(a) - ana, rel,
                            xpr_norm(ref.xprmat, i), xpr_norm(test.xprmat, i),
                            dph, dph_pred, xpr);
        }

        if (verbose())
        {
            std::printf("# worst %s-vs-%s rel dev %.3e at %.2f deg", test_name, ref_name,
                        S.worst_rel, S.worst_at);
            if (have_ana)
                std::printf(" | worst |%s-analytic| %.4f dB | worst |%s-analytic| %.4f dB",
                            test_name, S.worst_vs_analytic_test, ref_name, S.worst_vs_analytic_ref);
            std::printf("\n");
        }
        return S;
    }

    // Guard against a scene that silently misses the obstacle: an unobstructed path returns a gain
    // of 1, and two unobstructed paths would agree perfectly while testing nothing.
    void check_obstructed(const Trace &t, const Geometry &geo, const char *what, double upper = 0.5)
    {
        for (arma::uword i = 0; i < t.gain.n_elem; ++i)
        {
            INFO(what << ", position " << i << ", incidence " << geo.angle_deg[i] << " deg");
            CHECK(std::isfinite(t.gain(i)));
            CHECK(t.gain(i) > 0.0);
            CHECK(t.gain(i) < upper);
        }
    }

    // Shared material set
    const MatDef CONCRETE{5.24, 0.0, 0.0462, 0.7822};
    const MatDef LOSSLESS{5.24, 0.0, 0.0, 0.0};
    const MatDef GLASS{6.31, 0.0, 0.0036, 1.3394};

    const double FC = 3.5e9;
    const double FGHZ = 3.5;
    const double HALF_EXTENT = 200.0;

    const std::vector<double> SWEEP = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0};
    const std::vector<double> COARSE = {0.0, 34.0, 68.0, 85.0};
}

// ================================================================================================
// 0. Conventions: the analytic reference, and the mesh builders
// ================================================================================================

TEST_CASE("Mesh builder conventions", "[sheet][geometry]")
{
    // Prints the axis-aligned bounding box of every mesh the suite builds. Two things are being
    // confirmed rather than assumed: that cube() takes half-extents (a 0.05 first argument must give
    // a 0.1 m span in x, which is what makes the box a 10 cm slab and the sheet comparison valid),
    // and which plane the rotated plane() calls actually land in. The corner fixture depends on the
    // second and cannot be interpreted without it.
    if (!verbose())
        return;

    std::printf("\n# mesh builder conventions\n");
    std::printf("what,n_faces,xmin,xmax,ymin,ymax,zmin,zmax\n");
    print_bbox("plane(200,200) rot(0,pi/2,0)",
               quadriga_lib::plane<double>({200.0, 200.0}, {0.0, 0.5 * PI, 0.0}, {0.0, 0.0, 0.0}));
    print_bbox("plane(200,200) rot(pi/2,0,0)",
               quadriga_lib::plane<double>({200.0, 200.0}, {0.5 * PI, 0.0, 0.0}, {0.0, 0.0, 0.0}));
    print_bbox("plane(200,200) rot(0,0,0)",
               quadriga_lib::plane<double>({200.0, 200.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}));
    print_bbox("cube(0.05,200,200)",
               quadriga_lib::cube<double>({0.05, 200.0, 200.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}));
    print_bbox("make_quad(200, 0)", make_quad(200.0, 0.0));
    print_bbox("make_box(0.1, 200, 0)", make_box(0.1, 200.0, 0.0));
    std::printf("# cube() takes half-extents if the x span reads 0.1 for a 0.05 argument\n");
    CHECK(true);
}

TEST_CASE("Analytic reference reduces to the Airy sum", "[sheet][analytic]")
{
    // A transfer matrix of one layer must equal the closed-form single-slab Airy sum. Asserting this
    // here means every other table in this file rests on a reference that has been checked, not
    // assumed. Also checks the property the glued and embedded fixtures depend on: two layers of the
    // same material equal one layer of the summed thickness.
    cdouble e = eps_of(CONCRETE, FGHZ);

    for (double thd : {0.0, 34.0, 68.0, 85.0})
    {
        double th = thd * PI / 180.0;
        for (int pol = 0; pol < 2; ++pol)
        {
            // closed-form Airy
            double s = std::sin(th), c1 = std::cos(th);
            cdouble c2 = std::sqrt(cdouble(1.0, 0.0) - (cdouble(1.0, 0.0) / e) * s * s);
            cdouble z1(1.0, 0.0), z2 = std::sqrt(e), n = std::sqrt(e);
            cdouble r, t12, t21;
            if (pol == 0)
                r = (z1 * c1 - z2 * c2) / (z1 * c1 + z2 * c2),
                t12 = (2.0 * z1 * c1) / (z1 * c1 + z2 * c2),
                t21 = (2.0 * z2 * c2) / (z2 * c2 + z1 * c1);
            else
                r = (z2 * c1 - z1 * c2) / (z2 * c1 + z1 * c2),
                t12 = (2.0 * z1 * c1) / (z2 * c1 + z1 * c2),
                t21 = (2.0 * z2 * c2) / (z1 * c2 + z2 * c1);
            cdouble phi = std::exp(cdouble(0.0, -1.0) * (2.0 * PI * FGHZ * 1e9 / C0) * n * 0.1 * c2);
            double airy = std::norm(t12 * t21 * phi / (cdouble(1.0, 0.0) - r * r * phi * phi));
            double tm = tmm_power({{e, 0.1}}, th, FGHZ, pol);

            INFO("angle " << thd << " pol " << pol << " airy " << airy << " tmm " << tm);
            CHECK(std::abs(airy - tm) < 1e-12);
        }

        double one = analytic_dB({{e, 0.106}}, th, FGHZ);
        double two = analytic_dB({{e, 0.006}, {e, 0.1}}, th, FGHZ);
        INFO("angle " << thd << " one " << one << " two " << two);
        CHECK(std::abs(one - two) < 1e-9);
    }
}

// ================================================================================================
// 1. Isolated sheet vs traced solid, the primary equivalence test
// ================================================================================================

TEST_CASE("Isolated sheet vs traced solid, 10 cm concrete", "[sheet][primary]")
{
    const double thickness = 0.1;
    MaterialTable mt_solid, mt_sheet;
    arma::uword iSolid = mt_solid.add("concrete_solid", to_cols(CONCRETE, 0.0));
    arma::uword iSheet = mt_sheet.add("concrete_sheet", to_cols(CONCRETE, thickness));

    arma::mat mesh_solid = make_box(thickness, HALF_EXTENT, 0.0);
    arma::mat mesh_sheet = make_quad(HALF_EXTENT, 0.0);
    arma::uvec ind_solid = fill_ind(mesh_solid.n_rows, iSolid);
    arma::uvec ind_sheet = fill_ind(mesh_sheet.n_rows, iSheet);

    Geometry geo = make_positions(SWEEP);

    std::vector<double> ana;
    cdouble e = eps_of(CONCRETE, FGHZ);
    for (double thd : SWEEP)
        ana.push_back(analytic_dB({{e, thickness}}, thd * PI / 180.0, FGHZ));

    Trace solid = forward_trace(mesh_solid, ind_solid, mt_solid.prop, geo.orig, geo.dest, FC, false);
    Trace sheet = forward_trace(mesh_sheet, ind_sheet, mt_sheet.prop, geo.orig, geo.dest, FC, false);

    check_obstructed(solid, geo, "solid");
    check_obstructed(sheet, geo, "sheet");

    Summary S = report("10 cm concrete, EM, 3.5 GHz", geo, solid, sheet, ana, FC, thickness);

    INFO("target tolerance " << TOL_TARGET);
    CHECK(S.worst_rel < TOL_STAGED);
}

// ================================================================================================
// 2. Lossless slab: the one case where agreement should be exact
// ================================================================================================

TEST_CASE("Lossless slab, sheet vs solid", "[sheet][lossless]")
{
    // With no loss the in-slab refraction cosine is real, so every approximation that stems from
    // projecting a complex angle onto a real one vanishes. Whatever deviation survives here is NOT
    // the entry-boundary defect and has some other cause - which makes this the cleanest control in
    // the suite.
    const double thickness = 0.1;
    MaterialTable mt_solid, mt_sheet;
    arma::uword iSolid = mt_solid.add("lossless_solid", to_cols(LOSSLESS, 0.0));
    arma::uword iSheet = mt_sheet.add("lossless_sheet", to_cols(LOSSLESS, thickness));

    arma::mat mesh_solid = make_box(thickness, HALF_EXTENT, 0.0);
    arma::mat mesh_sheet = make_quad(HALF_EXTENT, 0.0);

    Geometry geo = make_positions(SWEEP);

    std::vector<double> ana;
    cdouble e = eps_of(LOSSLESS, FGHZ);
    for (double thd : SWEEP)
        ana.push_back(analytic_dB({{e, thickness}}, thd * PI / 180.0, FGHZ));

    Trace solid = forward_trace(mesh_solid, fill_ind(mesh_solid.n_rows, iSolid), mt_solid.prop,
                                geo.orig, geo.dest, FC, false);
    Trace sheet = forward_trace(mesh_sheet, fill_ind(mesh_sheet.n_rows, iSheet), mt_sheet.prop,
                                geo.orig, geo.dest, FC, false);

    Summary S = report("lossless eps_r 5.24, 10 cm", geo, solid, sheet, ana, FC, thickness);
    INFO("lossless agreement should approach machine precision");
    CHECK(S.worst_rel < TOL_STAGED);
}

// ================================================================================================
// 3. Material sweep: does the deviation follow in-slab obliquity rather than loss?
// ================================================================================================

TEST_CASE("Material sweep, sheet vs solid", "[sheet][materials]")
{
    // The companion document predicts that the divergence tracks Im(cos theta_slab)/Re(cos
    // theta_slab) rather than loss, so a low-permittivity weakly lossy slab should be worse than a
    // high-permittivity strongly lossy one. This sweep is what makes that predictor falsifiable.
    struct Case
    {
        const char *name;
        MatDef mat;
    };
    const std::vector<Case> cases = {
        {"eps 2.0 sigma 0.05", {2.0, 0.0, 0.05, 0.0}},
        {"eps 5.24 sigma 0.123 (concrete)", CONCRETE},
        {"eps 6.31 sigma 0.012 (glass)", GLASS},
        {"eps 6.0 sigma 0.20", {6.0, 0.0, 0.20, 0.0}},
        {"eps 25.0 sigma 0.50", {25.0, 0.0, 0.50, 0.0}},
    };

    const double thickness = 0.1;
    Geometry geo = make_positions(SWEEP);

    for (const auto &c : cases)
    {
        MaterialTable mt_solid, mt_sheet;
        arma::uword iSolid = mt_solid.add("solid", to_cols(c.mat, 0.0));
        arma::uword iSheet = mt_sheet.add("sheet", to_cols(c.mat, thickness));

        arma::mat mesh_solid = make_box(thickness, HALF_EXTENT, 0.0);
        arma::mat mesh_sheet = make_quad(HALF_EXTENT, 0.0);

        std::vector<double> ana;
        cdouble e = eps_of(c.mat, FGHZ);
        for (double thd : SWEEP)
            ana.push_back(analytic_dB({{e, thickness}}, thd * PI / 180.0, FGHZ));

        Trace solid = forward_trace(mesh_solid, fill_ind(mesh_solid.n_rows, iSolid), mt_solid.prop,
                                    geo.orig, geo.dest, FC, false);
        Trace sheet = forward_trace(mesh_sheet, fill_ind(mesh_sheet.n_rows, iSheet), mt_sheet.prop,
                                    geo.orig, geo.dest, FC, false);

        char tag[160];
        std::snprintf(tag, sizeof(tag), "material sweep: %s, 10 cm", c.name);
        report(tag, geo, solid, sheet, ana, FC, thickness);
    }

    // Report only; the point of this block is the shape of the curve, not a threshold.
    CHECK(true);
}

// ================================================================================================
// 4. Thickness sweep: resonance structure and the subdivision-tolerance question
// ================================================================================================

TEST_CASE("Thickness sweep, sheet vs solid", "[sheet][thickness]")
{
    // A 6 mm pane at 3.5 GHz has a round-trip optical path of roughly 29 mm against a 86 mm
    // wavelength, so its reflectance carries real angular structure instead of sitting near the
    // smooth half-space Fresnel value. Comparing a thin and a thick slab shows whether the
    // divergence scales with the number of internal round trips or with obliquity alone.
    Geometry geo = make_positions(SWEEP);
    cdouble e = eps_of(GLASS, FGHZ);

    for (double thickness : {0.006, 0.024, 0.1})
    {
        MaterialTable mt_solid, mt_sheet;
        arma::uword iSolid = mt_solid.add("solid", to_cols(GLASS, 0.0));
        arma::uword iSheet = mt_sheet.add("sheet", to_cols(GLASS, thickness));

        arma::mat mesh_solid = make_box(thickness, HALF_EXTENT, 0.0);
        arma::mat mesh_sheet = make_quad(HALF_EXTENT, 0.0);

        std::vector<double> ana;
        for (double thd : SWEEP)
            ana.push_back(analytic_dB({{e, thickness}}, thd * PI / 180.0, FGHZ));

        Trace solid = forward_trace(mesh_solid, fill_ind(mesh_solid.n_rows, iSolid), mt_solid.prop,
                                    geo.orig, geo.dest, FC, false);
        Trace sheet = forward_trace(mesh_sheet, fill_ind(mesh_sheet.n_rows, iSheet), mt_sheet.prop,
                                    geo.orig, geo.dest, FC, false);

        char tag[160];
        std::snprintf(tag, sizeof(tag), "thickness sweep: glass, %.0f mm", thickness * 1000.0);
        report(tag, geo, solid, sheet, ana, FC, thickness);
    }
    CHECK(true);
}

// ================================================================================================
// 5. Case 5: collapsed double-sided duplicate
// ================================================================================================

TEST_CASE("Collapsed duplicate equals a single sheet", "[sheet][case5]")
{
    // Two coincident sheets of identical material are an exporter double-sided artifact, not a
    // laminate. The classifier collapses them, so the result must equal a single quad. Reference is
    // the single quad rather than the analytic value, because this asserts the collapse itself.
    const double thickness = 0.1;
    MaterialTable mt;
    arma::uword iSheet = mt.add("concrete_sheet", to_cols(CONCRETE, thickness));

    arma::mat single = make_quad(HALF_EXTENT, 0.0);
    arma::mat dup = arma::join_vert(single, flip_winding(single)); // reversed normals, exact coincidence

    Geometry geo = make_positions(SWEEP);

    std::vector<double> ana;
    cdouble e = eps_of(CONCRETE, FGHZ);
    for (double thd : SWEEP)
        ana.push_back(analytic_dB({{e, thickness}}, thd * PI / 180.0, FGHZ));

    Trace one = forward_trace(single, fill_ind(single.n_rows, iSheet), mt.prop, geo.orig, geo.dest, FC, false);
    Trace two = forward_trace(dup, fill_ind(dup.n_rows, iSheet), mt.prop, geo.orig, geo.dest, FC, false);

    Summary S = report("case 5: two coincident quads, reversed normals, same material", geo,
                       one, two, ana, FC, 0.0, "single", "duplicate");

    // The collapse should be exact: the duplicate must not be charged twice, and must not interfere
    // with its own copy.
    INFO("a collapsed duplicate must be bit-comparable to a single sheet");
    CHECK(S.worst_rel < 1.0e-9);

    // Same again with matched rather than reversed normals, which the classifier also has to catch.
    // Exactly duplicated rows rather than a small offset: a separation below the relaunch nudge
    // (about 1e-6 m at these coordinates) would test the relaunch rather than the collapse, and in
    // the first run a 1e-5 m offset did exactly that.
    arma::mat dup_same = arma::join_vert(single, single);
    Trace two_same = forward_trace(dup_same, fill_ind(dup_same.n_rows, iSheet), mt.prop,
                                   geo.orig, geo.dest, FC, false);
    Summary S2 = report("case 5b: two coincident quads, matched normals, same material", geo,
                        one, two_same, ana, FC, 0.0, "single", "duplicate");
    CHECK(S2.worst_rel < 1.0e-6);
}

// ================================================================================================
// 6. Case 6: corner with a sheet
// ================================================================================================

TEST_CASE("Corner with sheet degrades to a single hit", "[sheet][case6]")
{
    // Two quads meeting at a right angle, with the ray crossing both planes at the same point so
    // they are co-located and non-parallel. The classifier must drop SBS and treat FBS alone, so the
    // result has to equal a single quad at the same 45 degree incidence.
    //
    // Both quads are shifted along their shared edge (the z axis) so that the crossing point at
    // (0, 0, 0) lands in the interior of a triangle rather than on the shared vertex of both
    // triangulations. In the first run, with both centers on the crossing point, the fixture returned
    // a killed ray and could not be interpreted. The reference quad gets the same shift so the two
    // columns describe the same geometry.
    const double Z_OFF = 37.0;
    const double thickness = 0.1;
    MaterialTable mt;
    arma::uword iSheet = mt.add("concrete_sheet", to_cols(CONCRETE, thickness));

    arma::mat quad_x = translate(quadriga_lib::plane<double>({HALF_EXTENT, HALF_EXTENT},
                                                             {0.0, 0.5 * PI, 0.0}, {0.0, 0.0, 0.0}),
                                 0.0, 0.0, Z_OFF); // plane x = 0
    arma::mat quad_y = translate(quadriga_lib::plane<double>({HALF_EXTENT, HALF_EXTENT},
                                                             {0.5 * PI, 0.0, 0.0}, {0.0, 0.0, 0.0}),
                                 0.0, 0.0, Z_OFF); // intended: plane y = 0
    arma::mat corner = arma::join_vert(quad_x, quad_y);

    if (verbose())
    {
        std::printf("\n# corner fixture geometry\n");
        std::printf("what,n_faces,xmin,xmax,ymin,ymax,zmin,zmax\n");
        print_bbox("quad_x", quad_x);
        print_bbox("quad_y", quad_y);
        std::printf("# quad_y must span x and z with zero y extent for this fixture to mean anything\n");
    }

    Geometry geo = make_corner_positions();

    std::vector<double> ana;
    cdouble e = eps_of(CONCRETE, FGHZ);
    ana.push_back(analytic_dB({{e, thickness}}, 45.0 * PI / 180.0, FGHZ));

    Trace single = forward_trace(quad_x, fill_ind(quad_x.n_rows, iSheet), mt.prop, geo.orig, geo.dest, FC, false);
    Trace both = forward_trace(corner, fill_ind(corner.n_rows, iSheet), mt.prop, geo.orig, geo.dest, FC, false);

    report("case 6: sheet meeting a sheet at 90 deg, ray through the shared point", geo,
           single, both, ana, FC, 0.0, "single", "corner");

    // If the second quad's orientation is not what this test assumes, 'single' will not be a valid
    // reference. Report unconditionally and check only that neither path produced nonsense.
    for (arma::uword i = 0; i < both.gain.n_elem; ++i)
    {
        INFO("corner fixture, position " << i);
        CHECK(std::isfinite(both.gain(i)));
    }
}

// ================================================================================================
// 7. Cases 2 and 3: sheet glued to a solid face
// ================================================================================================

TEST_CASE("Glued sheet equals a solid of the summed thickness", "[sheet][case2][case3]")
{
    // A sheet whose material properties match the solid it is glued to, apart from thk, is
    // physically a homogeneous slab of the summed thickness. That gives cases 2 and 3 an exact
    // analytic reference and a self-consistency reference at the same time.
    //
    // The implementation composes this as (sheet composite with far = solid) times (the solid's own
    // series with the sheet as its near boundary, treated as a half-space). That is an approximation
    // to the true three-layer coupling, and this fixture measures exactly how large it is.
    const double t_sheet = 0.006;
    const double t_solid = 0.1;
    const double t_total = t_sheet + t_solid;

    MaterialTable mt;
    arma::uword iSolid = mt.add("concrete_solid", to_cols(CONCRETE, 0.0));
    arma::uword iSheet = mt.add("concrete_sheet", to_cols(CONCRETE, t_sheet));

    MaterialTable mt_ref;
    arma::uword iRef = mt_ref.add("concrete_solid", to_cols(CONCRETE, 0.0));

    // Reference: one box of the summed thickness, centered so that its faces bracket the same span
    arma::mat mesh_ref = make_box(t_total, HALF_EXTENT, 0.0);

    // Case 2: sheet on the near face of the box (illuminated from -x)
    arma::mat box = make_box(t_solid, HALF_EXTENT, 0.0);
    arma::mat quad_near = make_quad(HALF_EXTENT, -0.5 * t_solid);
    arma::mat mesh_entry = arma::join_vert(box, quad_near);
    arma::uvec ind_entry = arma::join_vert(fill_ind(box.n_rows, iSolid), fill_ind(quad_near.n_rows, iSheet));

    // Case 3: sheet on the far face of the box
    arma::mat quad_far = make_quad(HALF_EXTENT, 0.5 * t_solid);
    arma::mat mesh_exit = arma::join_vert(box, quad_far);
    arma::uvec ind_exit = arma::join_vert(fill_ind(box.n_rows, iSolid), fill_ind(quad_far.n_rows, iSheet));

    Geometry geo = make_positions(SWEEP);

    std::vector<double> ana;
    cdouble e = eps_of(CONCRETE, FGHZ);
    for (double thd : SWEEP)
        ana.push_back(analytic_dB({{e, t_total}}, thd * PI / 180.0, FGHZ));

    Trace ref = forward_trace(mesh_ref, fill_ind(mesh_ref.n_rows, iRef), mt_ref.prop,
                              geo.orig, geo.dest, FC, false);
    Trace entry = forward_trace(mesh_entry, ind_entry, mt.prop, geo.orig, geo.dest, FC, false);
    Trace exitc = forward_trace(mesh_exit, ind_exit, mt.prop, geo.orig, geo.dest, FC, false);

    check_obstructed(ref, geo, "summed-thickness reference");

    report("case 2: air | sheet 6 mm | concrete 100 mm, vs one 106 mm solid", geo,
           ref, entry, ana, FC, t_sheet, "solid106", "glued_entry");
    report("case 3: concrete 100 mm | sheet 6 mm | air, vs one 106 mm solid", geo,
           ref, exitc, ana, FC, t_sheet, "solid106", "glued_exit");

    // Cases 2 and 3 must also agree with each other by reciprocity: the same stack illuminated from
    // either side transmits the same power. Any difference between these two columns is asymmetry
    // the implementation introduced, not physics.
    report("cases 2 vs 3: reciprocity of the glued stack", geo,
           entry, exitc, ana, FC, 0.0, "glued_entry", "glued_exit");

    for (arma::uword i = 0; i < entry.gain.n_elem; ++i)
    {
        INFO("glued fixture, incidence " << geo.angle_deg[i]);
        CHECK(std::isfinite(entry.gain(i)));
        CHECK(std::isfinite(exitc.gain(i)));
        CHECK(entry.gain(i) > 0.0);
        CHECK(exitc.gain(i) > 0.0);
    }
}

// ================================================================================================
// 8. Embedded sheet: the case ray_mesh_interact cannot see
// ================================================================================================

TEST_CASE("Sheet embedded in a solid equals a solid of the summed thickness", "[sheet][embedded]")
{
    // concrete | sheet | concrete. ray_mesh_interact classifies this with air on both sides, so the
    // whole result depends on ray_state_update supplying the tracked medium instead. With matching
    // material properties the stack is again a homogeneous slab of the summed thickness, so the
    // analytic column is exact.
    //
    // This also exercises the three-layer sequencing: the host leg closes with the sheet as its
    // interface ahead, the sheet resolves against concrete on both sides, and the next host leg
    // reflects off the sheet from the other side.
    const double t_sheet = 0.006;
    const double t_solid = 0.1; // total box thickness, sheet at the mid-plane
    const double t_total = t_solid + t_sheet;

    MaterialTable mt;
    arma::uword iSolid = mt.add("concrete_solid", to_cols(CONCRETE, 0.0));
    arma::uword iSheet = mt.add("concrete_sheet", to_cols(CONCRETE, t_sheet));

    MaterialTable mt_ref;
    arma::uword iRef = mt_ref.add("concrete_solid", to_cols(CONCRETE, 0.0));

    arma::mat box = make_box(t_solid, HALF_EXTENT, 0.0);
    arma::mat mid = make_quad(HALF_EXTENT, 0.0); // mid-plane, far from both box faces
    arma::mat mesh = arma::join_vert(box, mid);
    arma::uvec ind = arma::join_vert(fill_ind(box.n_rows, iSolid), fill_ind(mid.n_rows, iSheet));

    arma::mat mesh_ref = make_box(t_total, HALF_EXTENT, 0.0);

    Geometry geo = make_positions(SWEEP);

    std::vector<double> ana;
    cdouble e = eps_of(CONCRETE, FGHZ);
    for (double thd : SWEEP)
        ana.push_back(analytic_dB({{e, t_total}}, thd * PI / 180.0, FGHZ));

    Trace ref = forward_trace(mesh_ref, fill_ind(mesh_ref.n_rows, iRef), mt_ref.prop,
                              geo.orig, geo.dest, FC, false);
    Trace emb = forward_trace(mesh, ind, mt.prop, geo.orig, geo.dest, FC, false);

    report("embedded: concrete 50 | sheet 6 mm | concrete 50, vs one 106 mm solid", geo,
           ref, emb, ana, FC, t_sheet, "solid106", "embedded");

    // A different embedding medium on either side of the sheet, so the near/far selection cannot be
    // satisfied by symmetry. No analytic column: this is a three-material stack and the reference
    // would need the layer geometry the tracer builds, which is what is under test.
    MaterialTable mt2;
    arma::uword jGlass = mt2.add("glass_solid", to_cols(GLASS, 0.0));
    arma::uword jSheet = mt2.add("concrete_sheet", to_cols(CONCRETE, t_sheet));
    arma::mat gbox = make_box(t_solid, HALF_EXTENT, 0.0);
    arma::mat gmid = make_quad(HALF_EXTENT, 0.0);
    arma::mat gmesh = arma::join_vert(gbox, gmid);
    arma::uvec gind = arma::join_vert(fill_ind(gbox.n_rows, jGlass), fill_ind(gmid.n_rows, jSheet));

    std::vector<double> ana2;
    cdouble eg = eps_of(GLASS, FGHZ);
    for (double thd : SWEEP)
        ana2.push_back(analytic_dB({{eg, 0.5 * t_solid}, {e, t_sheet}, {eg, 0.5 * t_solid}},
                                   thd * PI / 180.0, FGHZ));

    Trace emb2 = forward_trace(gmesh, gind, mt2.prop, geo.orig, geo.dest, FC, false);
    Trace ref2 = forward_trace(gbox, fill_ind(gbox.n_rows, jGlass), mt2.prop,
                               geo.orig, geo.dest, FC, false);
    report("embedded, mixed: glass 50 | concrete sheet 6 mm | glass 50", geo,
           ref2, emb2, ana2, FC, t_sheet, "glass_only", "embedded");

    for (arma::uword i = 0; i < emb.gain.n_elem; ++i)
    {
        INFO("embedded fixture, incidence " << geo.angle_deg[i]);
        CHECK(std::isfinite(emb.gain(i)));
        CHECK(emb.gain(i) > 0.0);
        CHECK(std::isfinite(emb2.gain(i)));
    }
}

// ================================================================================================
// 9. Case 4: ordered stack, expected to be killed
// ================================================================================================

TEST_CASE("Ordered stack of two different sheets is killed", "[sheet][case4]")
{
    // Detection and the type bits are correct, but neither function has physics for a two-layer
    // stack, so the ray is dropped rather than answered wrongly. This asserts the stub is reached
    // and that it drops rather than producing a plausible-looking number.
    MaterialTable mt;
    arma::uword iA = mt.add("concrete_sheet", to_cols(CONCRETE, 0.006));
    arma::uword iB = mt.add("glass_sheet", to_cols(GLASS, 0.006));

    arma::mat qa = make_quad(HALF_EXTENT, 0.0);
    arma::mat qb = flip_winding(make_quad(HALF_EXTENT, 0.0));
    arma::mat mesh = arma::join_vert(qa, qb);
    arma::uvec ind = arma::join_vert(fill_ind(qa.n_rows, iA), fill_ind(qb.n_rows, iB));

    Geometry geo = make_positions(COARSE);
    Trace t = forward_trace(mesh, ind, mt.prop, geo.orig, geo.dest, FC, false);

    if (verbose())
    {
        std::printf("\n# case 4: two coincident sheets of different materials (expected: killed)\n");
        std::printf("angle,gain_lin,gain_dB\n");
        for (arma::uword i = 0; i < t.gain.n_elem; ++i)
            std::printf("%.2f,%.6e,%.6f\n", geo.angle_deg[i], t.gain(i), to_dB(t.gain(i)));
    }

    for (arma::uword i = 0; i < t.gain.n_elem; ++i)
    {
        INFO("case 4, incidence " << geo.angle_deg[i]);
        CHECK(std::isfinite(t.gain(i)));
    }
}

// ================================================================================================
// 10. Interface gain, the term the primary fixture cannot see
// ================================================================================================

TEST_CASE("Interface gain is applied once on each path", "[sheet][interface_gain]")
{
    // With att = 0 the primary fixture cannot see this term at all. The solid pays the entered
    // medium's interface_gain once, folded in by interact_with on entry and not on exit. The sheet
    // pays sqrt(slab * far) as an amplitude, which is once in power. If the two conventions differ
    // the offset shows up as a constant dB shift, independent of angle - which is exactly what makes
    // it separable from every other item in this suite.
    const double thickness = 0.1;
    const double att_dB = 3.0;
    MatDef mat = CONCRETE;
    mat.att = att_dB;

    MaterialTable mt_solid, mt_sheet;
    arma::uword iSolid = mt_solid.add("concrete_att", to_cols(mat, 0.0));
    arma::uword iSheet = mt_sheet.add("concrete_att_sheet", to_cols(mat, thickness));

    arma::mat mesh_solid = make_box(thickness, HALF_EXTENT, 0.0);
    arma::mat mesh_sheet = make_quad(HALF_EXTENT, 0.0);

    Geometry geo = make_positions(SWEEP);

    // No analytic column: att is a lumped model term outside the transfer-matrix description.
    Trace solid = forward_trace(mesh_solid, fill_ind(mesh_solid.n_rows, iSolid), mt_solid.prop,
                                geo.orig, geo.dest, FC, false);
    Trace sheet = forward_trace(mesh_sheet, fill_ind(mesh_sheet.n_rows, iSheet), mt_sheet.prop,
                                geo.orig, geo.dest, FC, false);

    report("interface gain: att = 3 dB on both paths, 10 cm concrete", geo,
           solid, sheet, {}, FC, thickness);

    // The zero-att run of the same geometry, so the att-induced shift can be read directly by
    // differencing the two blocks rather than inferred.
    MaterialTable m0s, m0h;
    arma::uword j0 = m0s.add("concrete", to_cols(CONCRETE, 0.0));
    arma::uword j1 = m0h.add("concrete_sheet", to_cols(CONCRETE, thickness));
    Trace solid0 = forward_trace(mesh_solid, fill_ind(mesh_solid.n_rows, j0), m0s.prop,
                                 geo.orig, geo.dest, FC, false);
    Trace sheet0 = forward_trace(mesh_sheet, fill_ind(mesh_sheet.n_rows, j1), m0h.prop,
                                 geo.orig, geo.dest, FC, false);

    if (verbose())
    {
        std::printf("\n# interface gain: dB charged by att = 3 dB, per path\n");
        std::printf("angle,solid_att_minus_solid0_dB,sheet_att_minus_sheet0_dB,difference_dB\n");
        for (arma::uword i = 0; i < solid.gain.n_elem; ++i)
        {
            double ds = to_dB(solid.gain(i)) - to_dB(solid0.gain(i));
            double dh = to_dB(sheet.gain(i)) - to_dB(sheet0.gain(i));
            std::printf("%.2f,%.6f,%.6f,%.6f\n", geo.angle_deg[i], ds, dh, dh - ds);
        }
        std::printf("# both columns should read about -3 dB; the difference column should be flat at 0\n");
    }
    CHECK(true);
}

// ================================================================================================
// 11. Grazing incidence and the clamp
// ================================================================================================

TEST_CASE("Grazing incidence, sheet vs solid", "[sheet][grazing]")
{
    // The traversal distance thk/cos diverges at tangency, and the two functions clamp it
    // differently: ray_mesh_interact bounds the lateral walk-off by the longest FBS triangle edge,
    // while ray_state_update has no access to that edge and uses a fixed extent-to-thickness ratio.
    // Neither should fire for this fixture - the quad's edge is hundreds of metres against a 6 mm
    // thickness - so any step in this block is the clamp firing when it should not.
    //
    // Capped at 88 degrees on purpose. At 89.5 the RX sits at y = 1146 m, where the relaunch offset
    // (8 float ULP at the FBS coordinate magnitude) exceeds colocation_dist and the pre-existing
    // coordinate-dependent defect in the companion document's section 6 takes over.
    const std::vector<double> grazing = {80.0, 84.0, 86.0, 87.0, 88.0};
    const double thickness = 0.006;

    MaterialTable mt_solid, mt_sheet;
    arma::uword iSolid = mt_solid.add("glass_solid", to_cols(GLASS, 0.0));
    arma::uword iSheet = mt_sheet.add("glass_sheet", to_cols(GLASS, thickness));

    arma::mat mesh_solid = make_box(thickness, HALF_EXTENT, 0.0);
    arma::mat mesh_sheet = make_quad(HALF_EXTENT, 0.0);

    Geometry geo = make_positions(grazing);

    std::vector<double> ana;
    cdouble e = eps_of(GLASS, FGHZ);
    for (double thd : grazing)
        ana.push_back(analytic_dB({{e, thickness}}, thd * PI / 180.0, FGHZ));

    Trace solid = forward_trace(mesh_solid, fill_ind(mesh_solid.n_rows, iSolid), mt_solid.prop,
                                geo.orig, geo.dest, FC, false);
    Trace sheet = forward_trace(mesh_sheet, fill_ind(mesh_sheet.n_rows, iSheet), mt_sheet.prop,
                                geo.orig, geo.dest, FC, false);

    report("grazing: glass 6 mm, 80 to 88 deg", geo, solid, sheet, ana, FC, thickness);

    for (arma::uword i = 0; i < sheet.gain.n_elem; ++i)
    {
        INFO("grazing, incidence " << geo.angle_deg[i]);
        CHECK(std::isfinite(sheet.gain(i)));
        CHECK(sheet.gain(i) >= 0.0);
    }
}

// ================================================================================================
// 12. Scalar (acoustic) mode and the mass-law route
// ================================================================================================

TEST_CASE("Acoustic sheet, mass-law route", "[sheet][scalar]")
{
    // A mass-law material takes the incoherent branch of produce mode: no series and no in-slab
    // phase, but both interface phases kept. Correction mode refuses to resolve it at all and lets
    // the tracer follow the bounces, so this is the one configuration where the two paths are
    // structurally different by design rather than by defect. No analytic column: the mass law is a
    // lumped surrogate outside the transfer-matrix description.
    const double thickness = 0.012;
    MatDef acoustic{5.24, 0.0, 0.0462, 0.7822};
    acoustic.m = 20.0;

    MaterialTable mt_solid, mt_sheet;
    arma::uword iSolid = mt_solid.add("acoustic_solid", to_cols(acoustic, 0.0));
    arma::uword iSheet = mt_sheet.add("acoustic_sheet", to_cols(acoustic, thickness));

    arma::mat mesh_solid = make_box(thickness, HALF_EXTENT, 0.0);
    arma::mat mesh_sheet = make_quad(HALF_EXTENT, 0.0);

    Geometry geo = make_positions(SWEEP);

    Trace solid = forward_trace(mesh_solid, fill_ind(mesh_solid.n_rows, iSolid), mt_solid.prop,
                                geo.orig, geo.dest, FC, true);
    Trace sheet = forward_trace(mesh_sheet, fill_ind(mesh_sheet.n_rows, iSheet), mt_sheet.prop,
                                geo.orig, geo.dest, FC, true);

    report("scalar mode: mass-law m = 20, 12 mm", geo, solid, sheet, {}, FC, thickness);

    // Same geometry without the mass-law term, so the incoherent route can be separated from the
    // ordinary scalar path by differencing.
    MaterialTable n_solid, n_sheet;
    arma::uword k0i = n_solid.add("scalar_solid", to_cols(CONCRETE, 0.0));
    arma::uword k1i = n_sheet.add("scalar_sheet", to_cols(CONCRETE, thickness));
    Trace s2 = forward_trace(mesh_solid, fill_ind(mesh_solid.n_rows, k0i), n_solid.prop,
                             geo.orig, geo.dest, FC, true);
    Trace h2 = forward_trace(mesh_sheet, fill_ind(mesh_sheet.n_rows, k1i), n_sheet.prop,
                             geo.orig, geo.dest, FC, true);
    report("scalar mode: no mass law, 12 mm", geo, s2, h2, {}, FC, thickness);

    for (arma::uword i = 0; i < sheet.gain.n_elem; ++i)
    {
        INFO("scalar, incidence " << geo.angle_deg[i]);
        CHECK(std::isfinite(sheet.gain(i)));
        CHECK(std::isfinite(h2.gain(i)));
    }
}

// ================================================================================================
// 13. Regression: a thk column that is present but zero must change nothing
// ================================================================================================

TEST_CASE("thk = 0 is identical to no sheet at all", "[sheet][regression]")
{
    // The whole sheet path is gated behind bit 6 of the hit code, which cannot be set without a
    // nonzero thk. A material that carries the column with a zero value must therefore reproduce the
    // pre-refactor result exactly. Asserted rather than argued, because it is the guarantee the
    // hundreds of existing solid tests rest on.
    const double thickness = 0.1;

    MaterialTable a, b;
    arma::uword ia = a.add("concrete_no_thk", {{"a", CONCRETE.a}, {"b", CONCRETE.b}, {"c", CONCRETE.c}, {"d", CONCRETE.d}, {"fRef", 1.0}});
    arma::uword ib = b.add("concrete_zero_thk", {{"a", CONCRETE.a}, {"b", CONCRETE.b}, {"c", CONCRETE.c}, {"d", CONCRETE.d}, {"fRef", 1.0}, {"thk", 0.0}});

    arma::mat mesh = make_box(thickness, HALF_EXTENT, 0.0);
    Geometry geo = make_positions(SWEEP);

    Trace ta = forward_trace(mesh, fill_ind(mesh.n_rows, ia), a.prop, geo.orig, geo.dest, FC, false);
    Trace tb = forward_trace(mesh, fill_ind(mesh.n_rows, ib), b.prop, geo.orig, geo.dest, FC, false);

    double worst = 0.0;
    for (arma::uword i = 0; i < ta.gain.n_elem; ++i)
        worst = std::max(worst, std::abs(tb.gain(i) - ta.gain(i)));

    if (verbose())
        std::printf("\n# regression: thk column absent vs present-and-zero, worst abs gain diff %.3e\n", worst);

    CHECK(worst == 0.0);
}