// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// ray_state_update, wave: the state machine, dispatch and argument contract.

#include "ray_test_helpers.hpp"

TEST_CASE("ray_state_update - state and code encoding", "[wave]")
{
    // State words: bit 15 flag, bit 14 a second flag, bits 0..13 material. Never arithmetic negation.
    CHECK(enc(0, true) == (short)-32768); // resolved air word is 0x8000, not -0
    CHECK(mat_of(enc(0, true)) == 0);
    CHECK(flag_of(enc(0, true)));
    CHECK(enc(5, true) != (short)-5); // 0x8005, not the arithmetic negative
    CHECK(mat_of(enc(5, true)) == 5);
    CHECK(flag_of(enc(5, true)));
    CHECK(mat_of(enc(5, false)) == 5);
    CHECK(!flag_of(enc(5, false)));
    CHECK(mat_of((short)0x7FFF) == 0x3FFF);

    // out_type / resolved_type bit reads
    CHECK(rt_bit(OT_ENTRY, 0));                    // ok
    CHECK(rt_bit(OT_ENTRY, 1));                    // front-side
    CHECK(!rt_bit(OT_EXIT, 1));                    // back-side
    CHECK(rt_bit(OT_M2M_M2, 2));                   // co-located
    CHECK(rt_bit((uint8_t)(OT_EXIT | OT_TIR), 5)); // total-reflection variant
    CHECK(rt_bit(RT_R0, 7));                       // reflection flag
    CHECK(rt_bit(RT_RINT, 3));                     // inside flag
    CHECK(rt_bit(RT_EXIT_SLAB, 2));                // resolve flag
}

TEST_CASE("ray_state_update - dispatch: o-i entry family", "[wave]")
{
    // eps = 1.5 keeps the slab factor off. An entry defers its medium loss, so the field is
    // kept and only the state advances.

    SECTION("clean entry from air keeps the field, sets cur, resets prev")
    {
        auto C = disp(OT_ENTRY, 1, 1, 0, 3, 0, 0, 1.0, 2.0); // prev_in = 3 is a sentinel
        check_keep(C);
        check_state(C, 0, 1, 0);
        check_rtype(C, RT_ENTRY);
    }
    SECTION("entry into a parallel slab at nH = 2 keeps prev clear")
    {
        auto C = disp(OT_ENTRY, 2, 4, 4, 0, 0, 0, 1.0, 2.0, 0.5);
        check_keep(C);
        check_state(C, 0, 4, 0);
        check_rtype(C, RT_ENTRY);
    }
    SECTION("nested entry (already inside) resets to a unit interface and buffers M1")
    {
        auto C = disp(OT_ENTRY, 1, 3, 0, 0, 2, 0, 1.0, 2.0); // inside FOG2, hits FOG3 face
        check_replace(C, 1.0, TOL);
        check_state(C, 0, 2, 3);
        check_rtype(C, RT_NESTED_DEFER);
    }
    SECTION("degenerate o->ii entry buffers M2 and clears the OK bit")
    {
        auto C = disp(OT_OVL_OII, 2, 4, 5, 0, 0, 0, 1.0, 2.0, 0.5);
        check_keep(C);
        check_state(C, 0, 4, 5);
        check_rtype(C, RT_ENTRY_DEFER);
    }
}

TEST_CASE("ray_state_update - dispatch: i-o exit family", "[wave]")
{
    // eps = 1.5, so a cavity exit re-emits with the in-medium close applied and cur cleared.
    // The just-exited material lands on prev_out.

    SECTION("cavity exit multiplies by the medium close and clears cur")
    {
        auto C = disp(OT_EXIT, 1, 4, 0, 0, 4, 0, 1.0, 2.0); // DENSE, lossless
        check_mult(C, close_at(4, 1.0), TOL);
        check_state(C, 4, 0, 0);
        check_rtype(C, RT_EXIT);
    }
    SECTION("a lossy cavity exit carries the medium magnitude")
    {
        auto C = disp(OT_EXIT, 1, 1, 0, 0, 1, 0, 1.2, 2.0); // FOG1, alpha = 2 dB/m over 1.2 m
        check_mult(C, close_at(1, 1.2), TOL);
        check_state(C, 1, 0, 0);
        check_rtype(C, RT_EXIT);
    }
    SECTION("buffered exit into the same material flushes the buffer to a unit interface")
    {
        auto C = disp(OT_EXIT, 2, 2, 0, 0, 1, 2, 1.0, 2.0); // buf(2) == M1(2): stay in cur
        check_replace(C, 1.0, TOL);
        check_state(C, 0, 1, 0);
        check_rtype(C, RT_SAMEPASS);
    }
    SECTION("buffered exit into a distinct material applies the entered interface")
    {
        auto C = disp(OT_EXIT, 2, 4, 0, 0, 1, 3, 1.2, 2.0);
        // Commit into buffer FOG3: the entered interface interact_with(FOG1, FOG3) is applied (unit
        // here - both eta = 1, att = 0) and the carried field is reset, then the FOG1 medium close
        // applies. So the field becomes the current medium close on a reset (unit) field.
        std::complex<double> c = close_at(1, 1.2);
        CHECK(std::abs((double)C.xprmat(0, 0) - c.real()) < TOL);
        CHECK(std::abs((double)C.xprmat(1, 0) - c.imag()) < TOL);
        CHECK(std::abs((double)C.gain(0) - std::norm(c)) < TOL);
        check_state(C, 1, 3, 0);
        check_rtype(C, RT_ENTRY);
    }
    SECTION("ii-o overlap exit leaves the medium and clears cur")
    {
        auto C = disp(OT_OVL_IIO, 2, 4, 0, 0, 4, 2, 0.9, 2.0); // buf set -> ii-o exit to air
        check_mult(C, close_at(4, 0.9), TOL);
        check_state(C, 4, 0, 0);
        check_rtype(C, RT_EXIT);
    }
}

TEST_CASE("ray_state_update - dispatch: material-to-material family", "[wave]")
{
    // types 5 (M1 hit first) and 7 (M2 hit first). A cavity transition applies the current
    // medium close and shifts prev <- current.

    SECTION("M1-first crossing enters M2, applies the current medium close")
    {
        auto C = disp(OT_M2M_M1, 2, 4, 5, 0, 4, 0, 1.0, 1.2); // DENSE -> NINE
        check_mult(C, close_at(4, 1.0), TOL);
        check_state(C, 4, 5, 0);
        check_rtype(C, RT_ENTRY);
    }
    SECTION("M2-first crossing enters M1")
    {
        auto C = disp(OT_M2M_M2, 2, 5, 4, 0, 4, 0, 1.0, 1.2); // travelling in DENSE (M2), into NINE (M1)
        check_mult(C, close_at(4, 1.0), TOL);
        check_state(C, 4, 5, 0);
        check_rtype(C, RT_ENTRY);
    }
    SECTION("a crossing from outside any medium sets the fix flag")
    {
        auto C = disp(OT_M2M_M1, 2, 4, 5, 0, 0, 0, 1.0, 1.2); // cur = 0: exit/entry mismatch
        CHECK(rt_bit(C.resolved_type[0], 4));                 // fix flag
        CHECK((int)C.cur_out(0) == 5);
    }
    SECTION("buffered i-i stays in the current material and swaps the buffer")
    {
        auto C = disp(OT_M2M_M1, 2, 2, 3, 0, 1, 2, 1.0, 1.5); // buf(2)==M1(2) -> swap to M2(3)
        check_replace(C, 1.0, TOL);
        check_state(C, 0, 1, 3);
        check_rtype(C, RT_NESTED_DEFER);
    }
}

TEST_CASE("ray_state_update - dispatch: corner families", "[wave]")
{
    // o-i-o (23) and i-o-i (21).

    SECTION("o-i-o outside any medium is a pass-through")
    {
        auto C = disp(OT_COR_OIO, 2, 2, 2, 0, 0, 0);
        check_replace(C, 1.0, TOL);
        check_state(C, 0, 0, 0);
        check_rtype(C, RT_ADVANCE);
    }
    SECTION("o-i-o inside a medium exits to air")
    {
        auto C = disp(OT_COR_OIO, 2, 4, 4, 0, 4, 0, 0.8, 2.0);
        check_mult(C, close_at(4, 0.8), TOL);
        check_state(C, 4, 0, 0);
        check_rtype(C, RT_EXIT);
    }
    SECTION("i-o-i air gap bounded by two faces is treated as an i-i transition")
    {
        // cur = M1 = FOG1, M2 = FOG3: cross into M2 through the air gap
        auto C = disp(OT_COR_IOI, 2, 1, 3, 0, 1, 0, 1.0, 1.5);
        check_mult(C, close_at(1, 1.0), TOL);
        check_state(C, 1, 3, 0);
        check_rtype(C, RT_ENTRY);
    }
    SECTION("i-o-i without a far material and no buffer terminates")
    {
        auto C = disp(OT_COR_IOI, 2, 1, 0, 0, 1, 0);
        check_kill(C);
        check_rtype(C, RT_KILL);
    }
    SECTION("i-o-i with a buffer but no current medium terminates")
    {
        auto C = disp(OT_COR_IOI, 2, 1, 3, 0, 0, 2);
        check_kill(C);
        check_rtype(C, RT_KILL);
    }
}

TEST_CASE("ray_state_update - dispatch: no crossing and unmatched codes", "[wave]")
{
    SECTION("a no-hit ray advances with an identity interface")
    {
        auto C = disp(OT_NOHIT, 1, 0, 0, 0, 0, 0);
        check_replace(C, 1.0, TOL);
        check_state(C, 0, 0, 0);
        check_rtype(C, RT_ADVANCE);
    }
    SECTION("a no-hit ray inside a medium keeps the inside flag")
    {
        auto C = disp(OT_NOHIT, 1, 0, 0, 0, 4, 0);
        check_replace(C, 1.0, TOL);
        check_state(C, 0, 4, 0);
        check_rtype(C, RT_SAMEPASS);
    }
    SECTION("an unmatched but OK-flagged code kills the ray")
    {
        auto C = disp((uint8_t)17, 1, 1, 0, 0, 1, 0); // bits ok + corner, not a valid topology
        check_kill(C);
        check_state(C, 0, 1, 0);
        check_rtype(C, RT_KILL);
    }
}

TEST_CASE("ray_state_update - dispatch: reflection pass", "[wave]")
{
    SECTION("front reflection outside any medium keeps the field")
    {
        for (int it : {0, 3}) // EM / scalar reflection
        {
            auto C = disp(OT_ENTRY, 1, 4, 0, 0, 0, 0, 1.0, 2.0, 0.5, it, 0.0);
            check_keep(C);
            check_state(C, 0, 0, 0);
            check_rtype(C, RT_R0);
        }
    }
    SECTION("internal reflection re-emits with the medium close when the gate is off")
    {
        auto C = disp(OT_EXIT, 1, 4, 0, 0, 4, 0, 1.0, 2.0, 0.5, 3, 1.5);
        check_mult(C, close_at(4, 1.0), TOL);
        check_state(C, 0, 4, 0x4000); // no resolved flag; followed
        check_rtype(C, RT_RINT);
    }
    SECTION("internal reflection of a resolvable slab applies S and flags the ray")
    {
        // DENSE cavity at the m = 2 resonance: S = 1/(1 - (1/3)^2) = 1.125
        double L = 2.0 * half_wave(2.0, FRQ);
        auto C = disp(OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, 3, 0.0);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
        std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));
        CHECK(std::abs(S - 1.125) < TOL); // oracle self-check at resonance
        check_mult(C, close_at(4, L) * S, TOL);
        check_state(C, 0, enc(4, true), 0);
        check_rtype(C, RT_RINT_SLAB);
    }
    SECTION("a resolved ray is killed on the reflection pass")
    {
        auto C = disp(OT_EXIT, 1, 4, 0, 0, enc(4, true), 0, 1.0, 2.0, 0.5, 3, 0.0);
        check_kill(C);
        check_rtype(C, RT_KILL);
    }
}

TEST_CASE("ray_state_update - flag semantics", "[wave]")
{
    static const auto mtl = base_palette<double>();

    SECTION("the resolved flag lands on current_out as a negative short")
    {
        double L = 2.0 * half_wave(2.0, FRQ);
        auto C = disp(OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, 3, 0.0);
        CHECK((int)C.cur_out(0) < 0);
        CHECK(mat_of(C.cur_out(0)) == 4);
        CHECK(flag_of(C.cur_out(0)));
        check_rtype(C, RT_RINT_SLAB);
    }
    SECTION("a wedge entry sets the non-parallel flag on prev_out")
    {
        auto C = make1<double>(mtl, 4, OT_ENTRY, 2, 4, 4, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        C.normals(0, 3) = 0.5; // SBS normal at 60 deg to FBS: |dot| = 0.5
        C.normals(0, 4) = std::sqrt(3.0) / 2.0;
        C.normals(0, 5) = 0.0;
        C.run();
        check_keep(C);
        CHECK((int)C.cur_out(0) == 4);
        CHECK((int)C.prev_out(0) == (int)(short)0x8000); // flag on air: exactly -32768
        CHECK(mat_of(C.prev_out(0)) == 0);
        CHECK(flag_of(C.prev_out(0)));
        CHECK((int)C.buf_out(0) == 0);
        check_rtype(C, RT_ENTRY);
    }
    SECTION("antiparallel and parallel faces leave the non-parallel flag clear")
    {
        auto C = disp(OT_ENTRY, 2, 4, 4, 0, 0, 0); // antiparallel default
        check_state(C, 0, 4, 0);

        auto D = make1<double>(mtl, 4, OT_ENTRY, 2, 4, 4, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        D.normals(0, 3) = -1.0; // parallel same-orientation: |dot| = 1
        D.run();
        check_state(D, 0, 4, 0);
    }
    SECTION("the input TIR flag is re-evaluated at the interface angle")
    {
        // out_type carries TIR, but at normal incidence there is no total reflection, so the
        // resolved type must not carry it (the interface re-evaluation overrides the input).
        auto C = disp((uint8_t)(OT_EXIT | OT_TIR), 1, 4, 0, 0, 4, 0);
        CHECK(!rt_bit(C.resolved_type[0], 5));
        check_state(C, 4, 0, 0);
    }
}

TEST_CASE("ray_state_update - Airy factor against the closed form", "[wave]")
{
    // Complex S on a cavity exit, eps = 0 (always resolve). The exit factor is close * S.
    static const auto mtl = base_palette<double>();
    std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r; // DENSE inside face, 1/3

    SECTION("resonance, antiresonance, and a generic length at 1 GHz")
    {
        double half = half_wave(2.0, FRQ);
        for (double L : {half, 1.5 * half, 0.2718})
        {
            auto C = make1<double>(mtl, 4, OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
            C.run();
            std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));
            check_mult(C, close_at(4, L) * S, TOL);
            check_state(C, 4, 0, 0);
            check_rtype(C, RT_EXIT_SLAB);
        }
        std::complex<double> S_res = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, half));
        std::complex<double> S_null = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, 1.5 * half));
        CHECK(std::abs(S_res - 1.125) < TOL);
        CHECK(std::abs(S_null - 0.9) < TOL);
    }
    SECTION("the phase tracks the carrier frequency")
    {
        double f2 = 2.4e9, L = 0.2718;
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0,
                               std::complex<double>(0.5, 0.3), f2);
        C.run();
        std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, f2, L));
        check_mult(C, close_at(4, L, f2) * S, TOL);
    }
    SECTION("an impedance-matched medium has no cavity: S = 1")
    {
        // eps = mu = 4: the interface admittance matches air, r = 0, so the Airy sum is 1.
        CHECK(calc_transition_gain_mu(0, 0.0, 0.0, 0.0, {4.0, 0.0}, {1.0, 0.0}, {4.0, 0.0}, {1.0, 0.0}) < 1e-12);

        arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
        arma::uvec ind;
        std::unordered_map<std::string, std::vector<double>> matched;
        mtl_matrix_to_map0<double>(M, ind, matched);
        set_mu<double>(matched, 4.0, 0.0, 0.0, 0.0);

        double L = 2.0 * half_wave(2.0, FRQ);
        auto C = make1<double>(matched, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        C.run();
        // n = sqrt(eps * mu) = 4; lossless, so the factor is a pure excess phase, S = 1
        double k0 = 2.0 * QPI * FRQ / C0;
        std::complex<double> closem = std::exp(std::complex<double>(0.0, -k0 * (4.0 - 1.0) * L));
        check_mult(C, closem, TOL);
        check_state(C, 1, 0, 0);
    }
}

TEST_CASE("ray_state_update - survival gate thresholds on eps", "[wave]")
{
    // rho = sqrt(R_near * R_far * medium_gain(2L)). NINE is lossless with inside-face r = 1/2,
    // so rho = 0.25 exactly.
    static const auto mtl = base_palette<double>();
    double L = 2.0 * half_wave(3.0, FRQ);
    std::complex<double> r = fresnel_te({9.0, 0.0}, {1.0, 0.0}, 0.0).r;
    CHECK(std::abs(gate_rho(r, r, 1.0) - 0.25) < TOL);
    std::complex<double> S = airy_S(r, r, phi_one_way(3.0, 1.0, FRQ, L));
    CHECK(std::abs(S - 4.0 / 3.0) < TOL);

    auto run_at = [&](double eps)
    {
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 5, 0, 0, 5, 0, L, 2.0, 0.5, QPI / 2.0, eps);
        C.run();
        return C;
    };

    SECTION("eps clearly below rho resolves")
    {
        auto C = run_at(0.225);
        check_mult(C, close_at(5, L) * S, TOL);
        check_state(C, 5, 0, 0);
        check_rtype(C, RT_EXIT_SLAB);
    }
    SECTION("eps clearly above rho re-emits with the gain untouched")
    {
        auto C = run_at(0.275);
        check_mult(C, close_at(5, L), TOL); // no S
        check_state(C, 5, 0, 0);
        check_rtype(C, RT_EXIT);
    }
    SECTION("eps = 0 resolves even a weak near-lossless cavity")
    {
        auto C = run_at(0.0);
        check_mult(C, close_at(5, L) * S, TOL);

        double Lw = 2.0 * half_wave(n_of(7), FRQ);
        std::complex<double> rw = fresnel_te({1.21, 0.0}, {1.0, 0.0}, 0.0).r;
        std::complex<double> Sw = airy_S(rw, rw, phi_one_way(n_of(7), 1.0, FRQ, Lw));
        auto D = make1<double>(mtl, 4, OT_EXIT, 1, 7, 0, 0, 7, 0, Lw, 2.0, 0.5, QPI / 2.0, 0.0);
        D.run();
        check_mult(D, close_at(7, Lw) * Sw, TOL);
    }
    SECTION("eps >= 1 never resolves")
    {
        for (double eps : {1.0, 1.5})
        {
            auto C = run_at(eps);
            check_mult(C, close_at(5, L), TOL);
            check_state(C, 5, 0, 0);
        }
    }
}

TEST_CASE("ray_state_update - lossy medium feeds |phi| from medium_gain", "[wave]")
{
    // For a slab with power-law absorption |phi|^2 must equal the implementation's own
    // medium_gain(L), not an eta-only loss. The test measures medium_gain through a cavity-exit
    // magnitude and then checks S against the closed form. (The mass column cannot be used here:
    // m > 0 marks a lumped-transmission material and is excluded from slab resolution.)
    arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::uvec ind;
    std::unordered_map<std::string, std::vector<double>> mtl;
    mtl_matrix_to_map0<double>(M, ind, mtl);
    mtl["alpha"] = {5.0}; // power-law absorption, dB/m (alphaB defaults to 0)

    const double F10 = 10.0e9; // mass law engages above its clamp (fGHz * dist > 1)
    const double L = 40.0 * half_wave(2.0, F10);

    // Measure medium_gain over d by the exit magnitude: |factor|^2 = medium_gain(d) (eps high, no S)
    auto med_of = [&](double d)
    {
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, d, 2.0, 0.5, QPI / 2.0, 1.5,
                               std::complex<double>(0.5, 0.3), F10);
        C.run();
        return std::norm(vvf(C));
    };
    double g_L = med_of(L);
    double g_2L = med_of(2.0 * L);

    CHECK(g_L < 0.999); // engagement tripwire: the mass column took effect
    // WARN("mass-law g_L = " << g_L << ", g_2L = " << g_2L << " (log-in-distance, not multiplicative)");

    // Cavity exit with eps = 0: extract S from the exit factor and compare with the measured |phi|
    auto C = make1<double>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 0.0,
                           std::complex<double>(0.5, 0.3), F10);
    C.run();
    double k0 = 2.0 * QPI * F10 / C0;
    std::complex<double> close = std::sqrt(g_L) * std::exp(std::complex<double>(0.0, -k0 * (2.0 - 1.0) * L));
    std::complex<double> S_func = vvf(C) / close;

    std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
    std::complex<double> S_right = airy_S(r, r, phi_one_way(2.0, g_L, F10, L));
    std::complex<double> S_wrong = airy_S(r, r, phi_one_way(2.0, 1.0, F10, L)); // eta-only

    CHECK(std::abs(S_func - S_right) < 1e-6);
    if (g_L < 0.99)
        CHECK(std::abs(S_func - S_right) < std::abs(S_func - S_wrong));
}

TEST_CASE("ray_state_update - transmission factor folds into the resolved gain", "[wave]")
{
    // With tf on the slab material, S must use the tf-effective reflectance. The tf model has no
    // public closed form, so the test extracts r_eff^2 = 1 - 1/S from the function's own S at
    // resonance and demands it differ from the bare Fresnel value.
    arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::uvec ind;
    std::unordered_map<std::string, std::vector<double>> mtl;
    mtl_matrix_to_map0<double>(M, ind, mtl);
    mtl["tf"] = {0.6};
    mtl["tfB"] = {0.0};

    const double L = 2.0 * half_wave(2.0, FRQ); // resonance: phi^2 = 1

    auto C = make1<double>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
    C.run();
    double k0 = 2.0 * QPI * FRQ / C0;
    std::complex<double> close = std::exp(std::complex<double>(0.0, -k0 * (2.0 - 1.0) * L)); // lossless
    std::complex<double> S_func = vvf(C) / close;
    std::complex<double> x = 1.0 - 1.0 / S_func; // r_near * r_far * phi^2, phi^2 = 1

    CHECK(std::abs(x.imag()) < 1e-6);
    CHECK(x.real() > 0.0);
    CHECK(x.real() < 1.0);
    // Engagement tripwire: with tf = 0.6 the effective reflectance must differ from 1/9.
    CHECK(std::abs(x.real() - 1.0 / 9.0) > 1e-3);
}

TEST_CASE("ray_state_update - resolve clamp near the Airy pole", "[wave]")
{
    // A lossless slab near grazing drives r^2 toward 1. At resonance with the Airy denominator
    // below 1e-2 the resolve is refused; detuned or at moderate angles it resolves finite.
    static const auto mtl = base_palette<double>();
    const int mat = 4;
    const double n = 2.0;

    auto exit_at = [&](double theta_n, double L)
    {
        double theta_i = std::asin(std::sin(theta_n) / n);
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, mat, 0, 0, mat, 0, L, 2.0, 0.5, QPI / 2.0 - theta_i, 0.0);
        C.run();
        return C;
    };

    SECTION("at the pole the resolve is clamped, not divergent")
    {
        double theta_n = 89.8 * QPI / 180.0;
        double theta_i = std::asin(std::sin(theta_n) / n);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, theta_i).r;
        double L = 2.0 * half_wave(n, FRQ);
        CHECK(std::abs(1.0 - r * r) < 1e-2); // Airy denominator inside the clamp band
        auto C = exit_at(theta_n, L);
        check_rtype(C, RT_EXIT_SLAB); // still resolved: the denominator is floored, not refused
        check_state(C, 4, 0, 0);
        check_scale_consistent(C, TOL);
        for (int rr = 0; rr < C.nXPR(); ++rr)
            CHECK(std::isfinite((double)C.xprmat(rr, 0)));
        CHECK(std::isfinite((double)C.gain(0)));
        CHECK(std::abs(vvf(C)) < 1000.0); // floored denominator keeps the factor bounded
    }
    SECTION("a quarter-wave detune leaves the clamp and resolves finite")
    {
        double theta_n = 89.8 * QPI / 180.0;
        double L = 2.5 * half_wave(n, FRQ); // phi^2 = -1: denominator ~ 2
        auto C = exit_at(theta_n, L);
        check_rtype(C, RT_EXIT_SLAB);
        check_scale_consistent(C, TOL);
        for (int r = 0; r < C.nXPR(); ++r)
            CHECK(std::isfinite((double)C.xprmat(r, 0)));
        CHECK(std::isfinite((double)C.gain(0)));
    }
    SECTION("a strong but sub-clamp resonance resolves bounded")
    {
        double theta_n = 88.0 * QPI / 180.0;
        double theta_i = std::asin(std::sin(theta_n) / n);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, theta_i).r;
        double L = 2.0 * half_wave(n, FRQ);
        CHECK(std::abs(1.0 - r * r) > 1e-2); // outside the clamp
        auto C = exit_at(theta_n, L);
        check_rtype(C, RT_EXIT_SLAB);
        check_scale_consistent(C, TOL);
        CHECK(std::abs(vvf(C)) < 105.0); // bounded by the clamp design margin
    }
}

TEST_CASE("ray_state_update - cross-pass invariance of the resolve decision", "[wave]")
{
    // The transmission-pass exit and the reflection-pass internal bounce of the same segment
    // must make the same resolve-vs-re-emit decision and apply the same S.
    static const auto mtl = base_palette<double>();
    const double L = 0.123;                                             // generic, off resonance
    std::complex<double> r = fresnel_te({9.0, 0.0}, {1.0, 0.0}, 0.0).r; // NINE, rho = 0.25
    std::complex<double> S = airy_S(r, r, phi_one_way(3.0, 1.0, FRQ, L));

    auto exit_at = [&](double eps)
    {
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 5, 0, 0, 5, 0, L, 2.0, 0.5, QPI / 2.0, eps);
        C.run();
        return C;
    };
    auto refl_at = [&](double eps)
    {
        auto C = make1<double>(mtl, 3, OT_EXIT, 1, 5, 0, 0, 5, 0, L, 2.0, 0.5, QPI / 2.0, eps);
        C.run();
        return C;
    };

    SECTION("eps below rho: both passes resolve with the identical S")
    {
        auto T = exit_at(0.2);
        auto R = refl_at(0.2);
        check_mult(T, close_at(5, L) * S, TOL);
        check_mult(R, close_at(5, L) * S, TOL);
        check_state(T, 5, 0, 0);
        check_state(R, 0, enc(5, true), 0);
        check_rtype(T, RT_EXIT_SLAB);
        check_rtype(R, RT_RINT_SLAB);
    }
    SECTION("eps above rho: both passes re-emit")
    {
        auto T = exit_at(0.3);
        auto R = refl_at(0.3);
        check_mult(T, close_at(5, L), TOL);
        check_mult(R, close_at(5, L), TOL);
        check_state(T, 5, 0, 0);
        check_state(R, 0, 5, 0x4000); // re-emitted, no resolved flag; followed
        check_rtype(T, RT_EXIT);
        check_rtype(R, RT_RINT);
    }
}

TEST_CASE("ray_state_update - a resolved ray passes an exit transparently", "[wave]")
{
    // Stage current_out into current_in. A reflection resolves the slab and sets the flag; the
    // subsequent transmission exit of the same segment adds only the medium close and clears cur.
    static const auto mtl = base_palette<double>();
    double L = 2.0 * half_wave(2.0, FRQ);

    auto A = disp(OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, 3, 0.0); // reflection resolves
    CHECK(flag_of(A.cur_out(0)));

    auto B = make1<double>(mtl, 4, OT_EXIT, 1, 4, 0, 0, A.cur_out(0), 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
    B.run();
    check_mult(B, close_at(4, L), TOL); // no extra S: the series was already summed
    check_state(B, 4, 0, 0);
    check_rtype(B, RT_EXIT);
}

TEST_CASE("ray_state_update - gain consistency and batch determinism", "[wave]")
{
    static const auto mtl = base_palette<double>();
    double L = 2.0 * half_wave(3.0, FRQ); // NINE resonance, rho = 0.25 resolves at eps = 0.15

    // A mixed batch: clean entry, resolvable cavity exit, inconsistent kill, nested replace,
    // resolved pass-through, buffered i-i replace. All scalar, eps = 0.15.
    auto scalar_rows = [&]()
    {
        std::vector<Rsu<double>> v;
        v.push_back(make1<double>(mtl, 4, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 4, OT_EXIT, 1, 5, 0, 0, 5, 0, L, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 4, (uint8_t)17, 1, 1, 0, 0, 1, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 4, OT_ENTRY, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 4, OT_EXIT, 1, 4, 0, 0, enc(4, true), 0, L, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 4, OT_M2M_M1, 2, 2, 3, 0, 1, 2, 1.0, 1.5, 0.5, QPI / 2.0, 0.15));
        return v;
    };

    SECTION("scalar mode keeps gainN equal to the xprmat power sum on every row")
    {
        auto v = scalar_rows();
        auto B = merge_rows(v);
        B.run();
        for (arma::uword i = 0; i < B.gain.n_elem; ++i)
        {
            double p = 0.0;
            for (int rr = 0; rr < 2; ++rr)
                p += (double)B.xprmat(rr, i) * (double)B.xprmat(rr, i);
            CHECK(std::abs((double)B.gain(i) - p) < TOL);
        }
    }
    SECTION("EM mode keeps gainN equal to half the xprmat power sum")
    {
        // Independent EM rows: entry (kept) and cavity exit (resolves). qd_polbasis sets
        // gain = 0.5 * sum by construction, so the invariant holds without knowing the layout.
        std::vector<Rsu<double>> v;
        v.push_back(make1<double>(mtl, 1, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 1, OT_EXIT, 1, 5, 0, 0, 5, 0, L, 2.0, 0.5, QPI / 2.0, 0.15));
        auto B = merge_rows(v);
        B.run();
        for (arma::uword i = 0; i < B.gain.n_elem; ++i)
        {
            double p = 0.0;
            for (int rr = 0; rr < 8; ++rr)
                p += (double)B.xprmat(rr, i) * (double)B.xprmat(rr, i);
            CHECK(std::abs((double)B.gain(i) - 0.5 * p) < TOL);
        }
    }
    SECTION("a mixed batch equals the same rows run one at a time")
    {
        auto v = scalar_rows();
        auto B = merge_rows(v);
        B.run();
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i].run();
            CHECK(arma::approx_equal(B.xprmat.col((arma::uword)i), v[i].xprmat.col(0), "absdiff", 0.0));
            CHECK(std::abs((double)B.gain((arma::uword)i) - (double)v[i].gain(0)) == 0.0);
            CHECK((int)B.prev_out((arma::uword)i) == (int)v[i].prev_out(0));
            CHECK((int)B.cur_out((arma::uword)i) == (int)v[i].cur_out(0));
            CHECK((int)B.buf_out((arma::uword)i) == (int)v[i].buf_out(0));
            CHECK((int)B.resolved_type[i] == (int)v[i].resolved_type[0]);
        }
    }
}

TEST_CASE("ray_state_update - optional arguments", "[wave]")
{
    static const auto mtl = base_palette<double>();

    SECTION("null state inputs behave exactly like explicit zero state")
    {
        auto A = make1<double>(mtl, 4, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15);
        auto B = A;
        B.has_prev_in = false;
        B.has_cur_in = false;
        B.has_buf_in = false;
        A.run();
        B.run();
        CHECK(arma::approx_equal(A.xprmat, B.xprmat, "absdiff", 0.0));
        CHECK(std::abs((double)A.gain(0) - (double)B.gain(0)) == 0.0);
        CHECK((int)A.prev_out(0) == (int)B.prev_out(0));
        CHECK((int)A.cur_out(0) == (int)B.cur_out(0));
        CHECK((int)A.buf_out(0) == (int)B.buf_out(0));
        check_state(A, 0, 1, 0);
    }
    SECTION("null state outputs skip the write but not the physics")
    {
        double L = 2.0 * half_wave(2.0, FRQ);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
        std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));

        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        C.has_prev_out = false;
        C.run();
        check_mult(C, close_at(4, L) * S, TOL);
        CHECK((int)C.cur_out(0) == 0);
        CHECK((int)C.buf_out(0) == 0);
    }
    SECTION("gainN and xprmatN can each be omitted")
    {
        auto A = make1<double>(mtl, 4, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        A.has_gain = false;
        A.run();
        check_keep(A); // entry defers the medium loss; xprmat unchanged
        check_state(A, 0, 1, 0);

        auto B = make1<double>(mtl, 4, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        B.has_xprmat = false;
        B.run();
        CHECK(std::abs((double)B.gain(0) - (double)B.g_in(0)) < TOL); // gain unchanged at entry
        check_state(B, 0, 1, 0);

        auto C = make1<double>(mtl, 4, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        C.has_gain = false;
        C.has_xprmat = false;
        CHECK_NOTHROW(C.run()); // pure state update
        check_state(C, 0, 1, 0);
    }
    SECTION("null normal_vecN throws (required VBS plane normal)")
    {
        auto A = make1<double>(mtl, 4, OT_ENTRY, 2, 4, 4, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        A.has_normals = false;
        CHECK_THROWS_AS(A.run(), std::invalid_argument);
    }
    SECTION("with every state pointer null the per-call physics still works")
    {
        auto strip = [](Rsu<double> &C)
        {
            C.has_prev_in = C.has_cur_in = C.has_buf_in = false;
            C.has_prev_out = C.has_cur_out = C.has_buf_out = false;
        };

        // (a) entry defers the medium loss and keeps the field
        auto A = make1<double>(mtl, 4, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.0);
        strip(A);
        A.run();
        check_keep(A);

        // (b) unmatched code kills, regardless of state tracking
        auto B = make1<double>(mtl, 4, (uint8_t)17, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.0);
        strip(B);
        B.run();
        check_kill(B);

        // (c) no-crossing pass-through resets to an identity interface
        auto C = make1<double>(mtl, 4, OT_NOHIT, 1, 0, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.0);
        strip(C);
        C.run();
        check_replace(C, 1.0, TOL);
    }
}

TEST_CASE("ray_state_update - gain operation semantics and probe handling", "[wave]")
{
    static const auto mtl = base_palette<double>();

    SECTION("keep is bit-identical, including a fully populated EM probe")
    {
        auto C = make1<double>(mtl, 1, OT_ENTRY, 1, 2, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15);
        C.xprmat(0, 0) = 0.31;
        C.xprmat(1, 0) = -0.12;
        C.xprmat(2, 0) = 0.05;
        C.xprmat(3, 0) = 0.21;
        C.xprmat(4, 0) = -0.17;
        C.xprmat(5, 0) = 0.02;
        C.xprmat(6, 0) = 0.44;
        C.xprmat(7, 0) = 0.09;
        double p = 0.0;
        for (int rr = 0; rr < 8; ++rr)
            p += C.xprmat(rr, 0) * C.xprmat(rr, 0);
        C.gain(0) = 0.5 * p;
        C.run();
        check_keep(C);
        check_state(C, 0, 2, 0);
    }
    SECTION("replace ignores the incoming field entirely")
    {
        auto A = make1<double>(mtl, 4, OT_ENTRY, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5,
                               std::complex<double>(0.5, 0.3));
        auto B = make1<double>(mtl, 4, OT_ENTRY, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5,
                               std::complex<double>(-0.1, 0.8));
        A.run();
        B.run();
        CHECK(arma::approx_equal(A.xprmat, B.xprmat, "absdiff", 0.0));
        CHECK(std::abs((double)A.gain(0) - (double)B.gain(0)) == 0.0);
        check_replace(A, 1.0, TOL);
    }
    SECTION("EM replace writes VV and HH, scalar replace writes VV only")
    {
        auto A = make1<double>(mtl, 1, OT_ENTRY, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        A.run();
        check_replace(A, 1.0, TOL); // VV + HH layout for itype < 3

        auto B = make1<double>(mtl, 4, OT_ENTRY, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        B.run();
        check_replace(B, 1.0, TOL); // VV-only layout for itype >= 3
        double p = (double)B.xprmat(0, 0) * (double)B.xprmat(0, 0) +
                   (double)B.xprmat(1, 0) * (double)B.xprmat(1, 0);
        CHECK(std::abs((double)B.gain(0) - p) < TOL); // scalar gain has no 0.5 factor
    }
}

TEST_CASE("ray_state_update - energy conservation of a lossless slab", "[wave]")
{
    // air | slab | air at normal incidence. The function owns every magnitude (interface
    // coefficient fed in, then corrected by close * S); the test owns only the vacuum phase per
    // one-way trip and assembles the ports. |R|^2 + |T|^2 = 1.
    static const auto mtl = base_palette<double>();
    const int mat = 4; // DENSE, lossless
    const double L = 0.275;
    const double k0 = 2.0 * QPI * FRQ / C0;

    FresnelTE out = fresnel_te({1.0, 0.0}, {4.0, 0.0}, 0.0); // air -> DENSE (r01, t01)
    FresnelTE ins = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0); // DENSE -> air (r10, t10)

    CHECK(std::abs(std::norm(out.r) - calc_transition_gain(0, 0.0, 0.0, 0.0, {1.0, 0.0}, {4.0, 0.0})) < 1e-12);
    CHECK(std::abs(std::norm(out.t) - calc_transition_gain(1, 0.0, 0.0, 0.0, {1.0, 0.0}, {4.0, 0.0})) < 1e-12);

    auto avv = [](const Rsu<double> &C)
    { return cpx(C.xprmat, 0, 0); };
    auto ev = [&](int it, uint8_t ot, int prev, int cur, double dof, std::complex<double> feed)
    {
        auto C = make1<double>(mtl, it, ot, 1, mat, 0, prev, cur, 0, dof, 2.0, 0.5, QPI / 2.0, 0.0, feed);
        C.run();
        return C;
    };

    auto E = ev(4, OT_ENTRY, 0, 0, 0.5, out.t);                     // entry: keeps t01, sets cur
    auto X = ev(4, OT_EXIT, 0, mat, L, ins.t);                      // exit: t10 * close * S
    auto F = ev(3, OT_ENTRY, 0, 0, 0.5, out.r);                     // front reflection: keeps r01
    auto I = ev(3, OT_EXIT, 0, mat, L, ins.r);                      // internal bounce: r10 * close * S
    auto O = ev(4, OT_EXIT, I.prev_out(0), I.cur_out(0), L, ins.t); // resolved outcouple: t10 * close

    check_keep(E);
    check_state(X, mat, 0, 0);
    check_state(I, 0, enc(mat, true), 0);
    check_state(O, mat, 0, 0);

    // phi_one_way = vac * close; the test supplies one vacuum phase per one-way trip.
    std::complex<double> vac = std::exp(std::complex<double>(0.0, -k0 * L));
    std::complex<double> T = out.t * vac * avv(X);                // t01 * phi * (t10 S)
    std::complex<double> B = out.t * vac * vac * avv(I) * avv(O); // t01 * phi^2 * (r10 S) * t10
    std::complex<double> R = avv(F) + B;                          // r01 + t01 t10 r10 phi^2 S
    CHECK(std::abs(std::norm(R) + std::norm(T) - 1.0) < TOL);
}

TEST_CASE("ray_state_update - energy safety of a lossy slab", "[wave]")
{
    // ABSORB (eta = 4, alpha = 30 dB/m) at resonance. The measured ports must match the oracle
    // ports; the reflection port catches a missing up-trip medium loss.
    static const auto mtl = base_palette<double>();
    const int mat = 6;
    const double L = 2.0 * half_wave(2.0, FRQ);
    const double g = medg(mat, L);
    const double k0 = 2.0 * QPI * FRQ / C0;

    FresnelTE out = fresnel_te({1.0, 0.0}, {4.0, 0.0}, 0.0);
    FresnelTE ins = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0);

    auto avv = [](const Rsu<double> &C)
    { return cpx(C.xprmat, 0, 0); };
    auto ev = [&](int it, uint8_t ot, int prev, int cur, double dof, std::complex<double> feed)
    {
        auto C = make1<double>(mtl, it, ot, 1, mat, 0, prev, cur, 0, dof, 2.0, 0.5, QPI / 2.0, 0.0, feed);
        C.run();
        return C;
    };
    auto X = ev(4, OT_EXIT, 0, mat, L, ins.t);
    auto F = ev(3, OT_ENTRY, 0, 0, 0.5, out.r);
    auto I = ev(3, OT_EXIT, 0, mat, L, ins.r);
    auto O = ev(4, OT_EXIT, I.prev_out(0), I.cur_out(0), L, ins.t);

    std::complex<double> phi = phi_one_way(2.0, g, FRQ, L);
    std::complex<double> S = airy_S(ins.r, ins.r, phi);
    double T_or = std::norm(out.t * ins.t * phi * S);
    double R_or = std::norm(out.r + out.t * ins.t * ins.r * phi * phi * S);
    double A_or = 1.0 - T_or - R_or;
    CHECK(A_or > 0.3); // driven hard enough for the test to mean something

    std::complex<double> vac = std::exp(std::complex<double>(0.0, -k0 * L));
    std::complex<double> T = out.t * vac * avv(X);
    std::complex<double> B = out.t * vac * vac * avv(I) * avv(O);
    std::complex<double> R = avv(F) + B;
    CHECK(std::abs(std::norm(T) - T_or) < TOL);
    CHECK(std::abs(std::norm(R) - R_or) < TOL); // fails if the up trip drops its medium loss
    CHECK(std::norm(R) + std::norm(T) < 1.0 + 1e-9);
}

TEST_CASE("ray_state_update - scalar refraction (interaction type 5)", "[wave]")
{
    static const auto mtl = base_palette<double>();
    double L = 2.0 * half_wave(2.0, FRQ);
    std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
    std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));

    SECTION("type 5 is accepted and matches scalar transmission at a normal-incidence exit")
    {
        auto C4 = make1<double>(mtl, 4, OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        auto C5 = make1<double>(mtl, 5, OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        C4.run();
        C5.run();
        check_mult(C4, close_at(4, L) * S, TOL);
        check_mult(C5, close_at(4, L) * S, TOL);
        check_state(C5, 4, 0, 0);
        check_rtype(C5, RT_EXIT_SLAB);
        CHECK(C5.xprmat.n_rows == 2); // scalar mode: two-row xprmat
    }
    SECTION("type 5 uses the same o-i entry routing as type 4")
    {
        auto C = make1<double>(mtl, 5, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15);
        C.run();
        check_keep(C);
        check_state(C, 0, 1 | 0x4000, 0); // the refracting entry marks the leg as bent
        check_rtype(C, RT_ENTRY);
    }
}

TEST_CASE("ray_state_update - path direction correction (VBS)", "[wave]")
{
    // path_dir_prev supplied at a real angle triggers the VBS construction, which corrects the
    // continuation direction. The reflection mirror is convention-independent.
    static const auto mtl = base_palette<double>();
    double L = 2.0 * half_wave(2.0, FRQ);
    double alpha = 25.0 * QPI / 180.0; // incoming direction 25 deg off the x axis

    SECTION("internal reflection mirrors the incoming direction about the face normal")
    {
        auto C = make1<double>(mtl, 3, OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        C.has_path_dir = true;
        C.path_dir_prev.set_size(1, 3);
        C.path_dir_prev(0, 0) = std::cos(alpha);
        C.path_dir_prev(0, 1) = std::sin(alpha);
        C.path_dir_prev(0, 2) = 0.0;
        C.has_path_dirN = true;
        C.path_dirN.zeros(1, 3);
        C.run();
        // mirror about N = (-1, 0, 0): the x component flips, the tangential part is unchanged
        CHECK(std::abs((double)C.path_dirN(0, 0) - (-std::cos(alpha))) < 1e-9);
        CHECK(std::abs((double)C.path_dirN(0, 1) - std::sin(alpha)) < 1e-9);
        CHECK(std::abs((double)C.path_dirN(0, 2)) < 1e-9);
        CHECK(rt_bit(C.resolved_type[0], 1)); // VBS correction flag
    }
    SECTION("a transparent transition carries the incoming direction through")
    {
        auto C = make1<double>(mtl, 4, OT_M2M_M1, 2, 1, 2, 0, 1, 0, 1.0, 1.5, 0.5, QPI / 2.0, 1.5);
        C.has_path_dir = true;
        C.path_dir_prev.set_size(1, 3);
        C.path_dir_prev(0, 0) = std::cos(alpha);
        C.path_dir_prev(0, 1) = std::sin(alpha);
        C.path_dir_prev(0, 2) = 0.0;
        C.has_path_dirN = true;
        C.path_dirN.zeros(1, 3);
        C.run();
        CHECK(std::abs((double)C.path_dirN(0, 0) - std::cos(alpha)) < 1e-9);
        CHECK(std::abs((double)C.path_dirN(0, 1) - std::sin(alpha)) < 1e-9);
        CHECK(std::abs((double)C.path_dirN(0, 2)) < 1e-9);
    }
}

TEST_CASE("ray_state_update - accumulated distance ledger", "[wave]")
{
    static const auto mtl = base_palette<double>();

    SECTION("a cavity exit preloads the outgoing leg into both distance columns")
    {
        double dof = 1.0, dfd = 2.0;
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 4, 0, 0, 4, 0, dof, dfd, 0.5, QPI / 2.0, 1.5);
        C.has_acc_geo_out = true, C.has_acc_ref_out = true;
        C.acc_geo_out.zeros(1);
        C.acc_ref_out.zeros(1);
        C.run();
        CHECK(std::abs((double)C.acc_ref_out(0) - dfd) < 1e-9); // refracted (post-exit leg is unrefracted)
        CHECK(std::abs((double)C.acc_geo_out(0) - dfd) < 1e-9); // geometric
    }
    SECTION("a non-terminal entry restarts the in-medium distance at the relaunch offset it skips")
    {
        // the ray leaves the hit point at x = 1.5 a relaunch offset past it; the ledger carries that part, so the
        // next event's leg from the relaunch point adds up to the leg from the hit point
        auto C = make1<double>(mtl, 4, OT_ENTRY, 2, 4, 4, 0, 0, 0, 1.5, 2.0, 0.5, QPI / 2.0, 1.5);
        C.has_acc_geo_out = true, C.has_acc_ref_out = true;
        C.acc_geo_out.zeros(1);
        C.acc_ref_out.zeros(1);
        C.run();
        const double nudge = qd_relaunch_nudge(1.5);
        CHECK(std::abs((double)C.acc_ref_out(0) - nudge) < 1e-12);
        CHECK(std::abs((double)C.acc_geo_out(0) - nudge) < 1e-12);
    }
    SECTION("the accumulated input distance threads into the exit medium loss")
    {
        double dof = 0.7, preload = 0.5;
        double L = dof + preload;
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 6, 0, 0, 6, 0, dof, 2.0, 0.5, QPI / 2.0, 1.5); // ABSORB
        C.has_acc_geo_in = true, C.has_acc_ref_in = true;
        C.acc_geo_in.zeros(1);
        C.acc_ref_in.zeros(1);
        C.acc_ref_in(0) = preload; // refracted
        C.acc_geo_in(0) = preload; // geometric
        C.run();
        CHECK(std::abs(std::norm(vvf(C)) - medg(6, L)) < 1e-9); // |factor|^2 = medium_gain(acc_in + d(orig,fbs))
    }
    SECTION("the geometric and refracted columns are independent on both sides")
    {
        // A caller that shares one geometric accumulator across frequencies passes the refracted
        // column alone on all but the first, so each side must accept one column without the other.
        double dof = 0.7, preload = 0.5;
        double L = dof + preload;

        const double sentinel = 7.5; // a NULL side must leave the caller's vector exactly as it was

        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 6, 0, 0, 6, 0, dof, 2.0, 0.5, QPI / 2.0, 1.5); // ABSORB
        C.has_acc_ref_in = true; // geometric input omitted, reads as 0
        C.acc_ref_in.zeros(1);
        C.acc_ref_in(0) = preload;
        C.has_acc_ref_out = true; // geometric output omitted, write skipped
        C.acc_ref_out.zeros(1);
        C.acc_geo_out.ones(1);
        C.acc_geo_out(0) = sentinel;
        CHECK_NOTHROW(C.run());
        CHECK(std::abs(std::norm(vvf(C)) - medg(6, L)) < 1e-9); // the refracted column still drives the loss
        CHECK(std::abs((double)C.acc_geo_out(0) - sentinel) < 1e-9); // untouched by the call

        auto D = make1<double>(mtl, 4, OT_EXIT, 1, 6, 0, 0, 6, 0, dof, 2.0, 0.5, QPI / 2.0, 1.5);
        D.has_acc_geo_in = true; // refracted input omitted, reads as 0
        D.acc_geo_in.zeros(1);
        D.acc_geo_in(0) = preload;
        D.has_acc_geo_out = true; // refracted output omitted, write skipped
        D.acc_geo_out.zeros(1);
        D.acc_ref_out.ones(1);
        D.acc_ref_out(0) = sentinel;
        CHECK_NOTHROW(D.run());
        CHECK(std::abs(std::norm(vvf(D)) - medg(6, dof)) < 1e-9); // refracted preload absent -> loss over d(orig,fbs) only
        CHECK(std::abs((double)D.acc_ref_out(0) - sentinel) < 1e-9); // untouched by the call
    }
}

TEST_CASE("ray_state_update - permeability affects the refractive index", "[wave]")
{
    // eta = 1, mu = 4 -> n = sqrt(eta*mu) = 2. The exit excess phase must use n, not sqrt(eta).
    arma::mat M = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::uvec ind;
    std::unordered_map<std::string, std::vector<double>> mtl;
    mtl_matrix_to_map0<double>(M, ind, mtl);
    set_mu<double>(mtl, 4.0, 0.0, 0.0, 0.0);

    double L = 0.2718;
    auto C = make1<double>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 1.5); // eps high, no S
    C.run();
    double k0 = 2.0 * QPI * FRQ / C0;
    std::complex<double> expect = std::exp(std::complex<double>(0.0, -k0 * (2.0 - 1.0) * L)); // lossless, n = 2
    check_mult(C, expect, TOL);
}

TEST_CASE("ray_state_update - a resonance shifts the slab factor, coincidence columns leave a wave medium alone", "[wave]")
{
    // Engagement tripwires for the new material features.
    SECTION("a permittivity resonance shifts the resolved slab factor")
    {
        arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
        arma::uvec ind;
        std::unordered_map<std::string, std::vector<double>> base, res;
        mtl_matrix_to_map0<double>(M, ind, base);
        res = base;
        res["resF"] = {1.0}; // resonance at the test frequency
        res["resQ"] = {5.0};
        res["resS"] = {0.5};

        double L = 2.0 * half_wave(2.0, FRQ);
        auto Cb = make1<double>(base, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        auto Cr = make1<double>(res, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        Cb.run();
        Cr.run();
        CHECK(std::abs(vvf(Cb) - vvf(Cr)) > 1e-3);
    }
    SECTION("coincidence columns leave a wave medium alone")
    {
        // coiF / coiQ act only on a mass-law plate (in X0); a coiA column is no column (D2-b), read and ignored:
        // the exit at coincidence is the exit without the three columns, bitwise
        arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
        arma::uvec ind;
        std::unordered_map<std::string, std::vector<double>> base, mtl;
        mtl_matrix_to_map0<double>(M, ind, base);
        mtl = base;
        mtl["coiF"] = {1.0}; // coincidence at the test frequency
        mtl["coiQ"] = {3.0};
        mtl["coiA"] = {6.0};
        auto Cb = make1<double>(base, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15);
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15);
        CHECK_NOTHROW(Cb.run());
        CHECK_NOTHROW(C.run());
        CHECK(vvf(C) == vvf(Cb));
        CHECK((double)C.gain(0) == (double)Cb.gain(0));
    }
}

TEST_CASE("ray_state_update - type parity between double and float", "[wave]")
{
    SECTION("double") { run_parity<double>(1.0e-9); }
    SECTION("float") { run_parity<float>(1.0e-4); }
}

TEST_CASE("ray_state_update - ray_ind mapping and read-only inputs", "[wave]")
{
    // Full-set arrays (orig, dest, no_interact, state) are read at g = ray_ind[i]; compact-set
    // arrays (fbs, sbs, angle, normals, materials, out_type) at i. No input array is modified.
    auto mtl = base_palette<double>();

    Rsu<double> C;
    C.itype = 4;
    C.eps = 1.5;
    C.mtl = mtl;
    C.has_ray_ind = true;

    const arma::uword n_ray = 5, n_rayN = 2;
    C.orig.zeros(n_ray, 3);
    C.dest.zeros(n_ray, 3);
    for (arma::uword g = 0; g < n_ray; ++g)
        C.dest(g, 0) = (double)(g + 1);
    C.no_interact.set_size(n_ray);
    C.no_interact.fill(1u);
    C.prev_in = sv({5, 6, 5, 7, 5});
    C.cur_in = sv({7, 3, 7, 1, 7}); // FOG3 at g = 1, FOG1 at g = 3
    C.buf_in = sv({0, 0, 0, 0, 0});

    // Compact-set arrays, read at i
    C.fbs.zeros(n_rayN, 3);
    C.sbs.zeros(n_rayN, 3);
    for (arma::uword i = 0; i < n_rayN; ++i)
    {
        C.fbs(i, 0) = 0.4 * (double)(i + 1); // d(orig, fbs) = 0.4, 0.8
        C.sbs(i, 0) = C.fbs(i, 0) + 10.0;
    }
    C.fbs_angle.set_size(n_rayN);
    C.fbs_angle.fill(QPI / 2.0);
    C.out_type.assign(n_rayN, OT_EXIT);
    C.m1 = sv({1, 3}); // exit face materials (match cur at the mapped g)
    C.m2 = sv({0, 0});
    C.ray_ind = {3u, 1u}; // out of order on purpose
    C.normals.zeros(n_rayN, 6);
    C.normals.col(0).fill(-1.0);
    C.normals.col(3).fill(1.0);
    C.prev_out = sv({11111, 11111});
    C.cur_out = sv({11111, 11111});
    C.buf_out = sv({11111, 11111});
    C.resolved_type.assign(n_rayN, (uint8_t)222);
    C.gain = {1.0, 1.0};
    C.xprmat.zeros(2, n_rayN);
    C.xprmat(0, 0) = 1.0;
    C.xprmat(0, 1) = 1.0;

    arma::Col<short> p0 = C.prev_in, c0 = C.cur_in, b0 = C.buf_in;
    arma::mat o0 = C.orig, d0 = C.dest;

    C.run();

    // Row 0 -> g = 3: cur = FOG1 (1), clean cavity exit over d(orig, fbs) = 0.4.
    // Row 1 -> g = 1: cur = FOG3 (3), clean cavity exit over d(orig, fbs) = 0.8.
    check_mult(C, close_at(1, 0.4), TOL, 0);
    check_mult(C, close_at(3, 0.8), TOL, 1);
    CHECK((int)C.prev_out(0) == 1); // just-exited material
    CHECK((int)C.prev_out(1) == 3);
    CHECK((int)C.cur_out(0) == 0);
    CHECK((int)C.cur_out(1) == 0);
    check_rtype(C, RT_EXIT, 0);
    check_rtype(C, RT_EXIT, 1);

    // Input arrays are read-only
    for (arma::uword g = 0; g < n_ray; ++g)
    {
        CHECK((int)C.prev_in(g) == (int)p0(g));
        CHECK((int)C.cur_in(g) == (int)c0(g));
        CHECK((int)C.buf_in(g) == (int)b0(g));
    }
    CHECK(arma::approx_equal(C.orig, o0, "absdiff", 0.0));
    CHECK(arma::approx_equal(C.dest, d0, "absdiff", 0.0));
}

TEST_CASE("ray_state_update - input validation", "[wave]")
{
    // Every malformed input throws std::invalid_argument; the paired well-formed call succeeds.

    { // the baseline must run
        auto C = make_valid();
        CHECK_NOTHROW(C.run());
        check_state(C, 0, 1, 0);
    }

    SECTION("interaction type range")
    {
        for (int it : {6, -1, 99})
        {
            auto C = make_valid();
            C.itype = it;
            CHECK_THROWS_AS(C.run(), std::invalid_argument);
        }
        for (int it : {0, 1, 2, 3, 4, 5}) // 5 (scalar refraction) is now valid
        {
            auto C = make_valid();
            C.itype = it;
            int nX = it >= 3 ? 2 : 8; // xprmat width tracks EM (8) vs scalar (2)
            C.xprmat.zeros(nX, 1);
            C.xprmat(0, 0) = 0.5;
            CHECK_NOTHROW(C.run());
        }
    }
    SECTION("center frequency must be positive and finite")
    {
        for (double f : {0.0, -1.0e9, std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()})
        {
            auto C = make_valid();
            C.freq = f;
            CHECK_THROWS_AS(C.run(), std::invalid_argument);
        }
    }
    SECTION("eps must be finite and non-negative")
    {
        for (double e : {-0.1, (double)std::numeric_limits<double>::quiet_NaN(),
                         (double)std::numeric_limits<double>::infinity()})
        {
            auto C = make_valid();
            C.eps = e;
            CHECK_THROWS_AS(C.run(), std::invalid_argument);
        }
        for (double e : {0.0, 0.5, 1.0, 2.0})
        {
            auto C = make_valid();
            C.eps = e;
            CHECK_NOTHROW(C.run());
        }
    }
    SECTION("full-set geometry needs three columns")
    {
        for (arma::uword nc : {2u, 4u})
        {
            auto C = make_valid();
            C.orig.set_size(2, nc);
            C.orig.zeros();
            CHECK_THROWS_AS(C.run(), std::invalid_argument);

            auto D = make_valid();
            D.dest.set_size(2, nc);
            D.dest.zeros();
            CHECK_THROWS_AS(D.run(), std::invalid_argument);
        }
    }
    SECTION("compact-set geometry needs three columns")
    {
        for (arma::uword nc : {2u, 4u})
        {
            auto C = make_valid();
            C.fbs.set_size(1, nc);
            C.fbs.zeros();
            CHECK_THROWS_AS(C.run(), std::invalid_argument);

            auto D = make_valid();
            D.sbs.set_size(1, nc);
            D.sbs.zeros();
            CHECK_THROWS_AS(D.run(), std::invalid_argument);
        }
    }
    SECTION("normal_vecN needs six columns, xprmatN needs nXPR rows")
    {
        auto C = make_valid();
        C.normals.set_size(1, 5);
        C.normals.zeros();
        CHECK_THROWS_AS(C.run(), std::invalid_argument);

        auto D = make_valid(); // scalar mode: xprmat must be [2, n_rayN]
        D.xprmat.set_size(8, 1);
        D.xprmat.zeros();
        CHECK_THROWS_AS(D.run(), std::invalid_argument);
    }
    SECTION("full-set arrays must agree on n_ray")
    {
        auto C = make_valid();
        C.dest = arma::join_vert(C.dest, C.dest.row(0)); // 3 rows, orig has 2
        CHECK_THROWS_AS(C.run(), std::invalid_argument);

        auto D = make_valid();
        D.cur_in = sv({0, 0, 0});
        CHECK_THROWS_AS(D.run(), std::invalid_argument);
    }
    SECTION("the accumulated-distance inputs are sized against n_ray")
    {
        auto C = make_valid();
        C.has_acc_geo_in = true;
        C.acc_geo_in.zeros(3); // n_ray is 2
        CHECK_THROWS_AS(C.run(), std::invalid_argument);

        auto D = make_valid();
        D.has_acc_ref_in = true;
        D.acc_ref_in.zeros(3);
        CHECK_THROWS_AS(D.run(), std::invalid_argument);

        auto E = make_valid(); // correctly sized, either column alone
        E.has_acc_geo_in = true;
        E.acc_geo_in.zeros(2);
        CHECK_NOTHROW(E.run());

        auto F = make_valid();
        F.has_acc_ref_in = true;
        F.acc_ref_in.zeros(2);
        CHECK_NOTHROW(F.run());
    }
    SECTION("the accumulated-distance outputs are resized on demand")
    {
        auto C = make_valid(); // n_rayN is 1, both start empty
        C.has_acc_geo_out = true, C.has_acc_ref_out = true;
        CHECK_NOTHROW(C.run());
        CHECK(C.acc_geo_out.n_elem == 1);
        CHECK(C.acc_ref_out.n_elem == 1);

        auto D = make_valid(); // wrong size is corrected, not rejected
        D.has_acc_geo_out = true, D.has_acc_ref_out = true;
        D.acc_geo_out.zeros(7);
        D.acc_ref_out.zeros(7);
        CHECK_NOTHROW(D.run());
        CHECK(D.acc_geo_out.n_elem == 1);
        CHECK(D.acc_ref_out.n_elem == 1);
    }
    SECTION("compact arrays must agree on n_rayN")
    {
        auto C0 = make_valid();
        C0.out_type.assign(2, OT_ENTRY);
        CHECK_THROWS_AS(C0.run(), std::invalid_argument);

        auto C1 = make_valid();
        C1.m1 = sv({1, 1});
        CHECK_THROWS_AS(C1.run(), std::invalid_argument);

        auto C2 = make_valid();
        C2.m2 = sv({0, 0});
        CHECK_THROWS_AS(C2.run(), std::invalid_argument);

        auto C3 = make_valid();
        C3.xprmat = arma::join_horiz(C3.xprmat, C3.xprmat);
        CHECK_THROWS_AS(C3.run(), std::invalid_argument);

        auto C4 = make_valid();
        C4.normals = arma::join_vert(C4.normals, C4.normals);
        CHECK_THROWS_AS(C4.run(), std::invalid_argument);

        auto C5 = make_valid();
        C5.sbs = arma::join_vert(C5.sbs, C5.sbs);
        CHECK_THROWS_AS(C5.run(), std::invalid_argument);
    }
    SECTION("unconditionally required pointers")
    {
        for (int k = 0; k < 9; ++k)
        {
            auto C = make_valid();
            switch (k)
            {
            case 0:
                C.has_orig = false;
                break;
            case 1:
                C.has_dest = false;
                break;
            case 2:
                C.has_fbs = false;
                break;
            case 3:
                C.has_sbs = false;
                break;
            case 4:
                C.has_ni = false;
                break;
            case 5:
                C.has_normals = false;
                break;
            case 6:
                C.has_otype = false;
                break;
            case 7:
                C.has_m1 = false;
                break;
            case 8:
                C.has_m2 = false;
                break;
            }
            CHECK_THROWS_AS(C.run(), std::invalid_argument);
        }
    }
    SECTION("optional pointers may be null")
    {
        for (int k = 0; k < 6; ++k)
        {
            auto C = make_valid();
            switch (k)
            {
            case 0:
                C.has_prev_in = false;
                break;
            case 1:
                C.has_cur_in = false;
                break;
            case 2:
                C.has_buf_in = false;
                break;
            case 3:
                C.has_prev_out = false;
                break;
            case 4:
                C.has_gain = false;
                break;
            case 5:
                C.has_xprmat = false;
                break;
            }
            CHECK_NOTHROW(C.run());
        }
    }
    SECTION("ray_ind bounds and the identity requirement")
    {
        auto C = make_valid();
        C.ray_ind = {5u}; // out of range for n_ray = 2
        CHECK_THROWS_AS(C.run(), std::invalid_argument);

        auto D = make_valid();
        D.has_ray_ind = false; // null with n_ray != n_rayN
        CHECK_THROWS_AS(D.run(), std::invalid_argument);
    }
    SECTION("material indices must exist in the map")
    {
        auto C = make_valid();
        C.m1 = sv({8}); // the palette has 7 materials
        CHECK_THROWS_AS(C.run(), std::invalid_argument);

        auto D = make_valid();
        D.cur_in = sv({8, 8});
        CHECK_THROWS_AS(D.run(), std::invalid_argument);

        auto E = make_valid();
        E.m1 = sv({0}); // air is always valid
        CHECK_NOTHROW(E.run());

        auto F = make_valid();
        F.cur_in = sv({enc(7, true), enc(7, true)}); // flagged but in range
        CHECK_NOTHROW(F.run());
    }
    SECTION("the material map may be sparse but must be consistent")
    {
        auto C = make_valid();
        C.mtl.erase("a"); // missing columns fall back to defaults
        CHECK_NOTHROW(C.run());

        auto D = make_valid();
        D.mtl["alpha"].resize(6); // others have 7 entries
        CHECK_THROWS_AS(D.run(), std::invalid_argument);
    }
    SECTION("an empty batch is a no-op, not an error")
    {
        Rsu<double> C;
        C.itype = 4;
        C.eps = 0.15;
        C.mtl = base_palette<double>();
        C.orig.set_size(0, 3);
        C.dest.set_size(0, 3);
        C.fbs.set_size(0, 3);
        C.sbs.set_size(0, 3);
        C.no_interact.set_size(0);
        C.fbs_angle.set_size(0);
        C.normals.set_size(0, 6);
        C.out_type.clear();
        C.m1.set_size(0);
        C.m2.set_size(0);
        C.prev_in.set_size(0);
        C.cur_in.set_size(0);
        C.buf_in.set_size(0);
        C.prev_out.set_size(0);
        C.cur_out.set_size(0);
        C.buf_out.set_size(0);
        C.resolved_type.clear();
        C.gain.set_size(0);
        C.xprmat.set_size(2, 0);
        CHECK_NOTHROW(C.run());
        CHECK(C.gain.n_elem == 0);
        CHECK(C.xprmat.n_cols == 0);
    }
}
