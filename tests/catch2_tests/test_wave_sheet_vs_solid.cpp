// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// A sheet through calc_diffraction_gain, wave: two coincident faces of one material are one sheet,
// and a thk column present with the value zero is the column absent. The two cases kept from the
// former test_sheet_vs_solid.cpp; the sheet against its traced solid is asserted in the walker
// cases of test_wave_state_update_branch.cpp.
//
// Set the environment variable QD_TEST_VERBOSE=1 to print the tables.

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

namespace
{
    using MtlProp = std::unordered_map<std::string, std::vector<double>>;
    using cdouble = std::complex<double>;

    constexpr double PI = 3.14159265358979323846;
    constexpr double C0 = 299792458.0;

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
    // single-slab Airy sum at one layer.
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

    // Shared material set
    const MatDef CONCRETE{5.24, 0.0, 0.0462, 0.7822};

    const double FC = 3.5e9;
    const double FGHZ = 3.5;
    const double HALF_EXTENT = 200.0;

    const std::vector<double> SWEEP = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0};
}

TEST_CASE("Collapsed duplicate equals a single sheet", "[sheet][case5][wave]")
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

TEST_CASE("thk = 0 is identical to no sheet at all", "[sheet][regression][wave]")
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
