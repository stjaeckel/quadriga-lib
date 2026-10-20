// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// The material model and the ray-surface physics of the ray tracer.
//
// Public functions (each documented in the API block above its definition):
//   medium_gain        power gain of a path through a homogeneous medium: attenuation, and the mass term of a
//                      mass-law material at normal incidence
//   interface_gain     the lumped penetration gain (att) of entering a material
//   refractive_index   Re sqrt(eps mu) of a material
//   ray_mesh_interact  one face event per ray: the hit geometry, the interface coefficients (Fresnel / Jones, or
//                      the composite of a thin sheet), the outgoing direction, the relaunch segment and the ray
//                      tube. It sees faces only and assumes air on every side the mesh does not name.
//   ray_state_update   the per-ray medium state machine: corrects what ray_mesh_interact wrote with the medium the
//                      ray is known to be in (the face at the virtual hit point VBS of the wave's own direction),
//                      closes the leg out (in-medium loss and phase), sums the bounce series of a layer in closed
//                      form where it is strong enough, re-resolves a sheet against the tracked media, and writes
//                      the state words and the ledger for the next event.
// ray_progress and calc_diffraction_gain trace a segment per ray (ray_triangle_intersect) and call
// ray_mesh_interact and then ray_state_update twice per event: once for the reflect pass (types 0, 3), once for
// the transmit pass (1, 2, 4, 5). ray_commit uses medium_gain for the last leg to a receiver inside a medium.
//
// Conventions
// - Frequency in Hz at the API, GHz inside (fGHz). Lengths in m. All arithmetic is double in both templates;
//   float is the storage of inputs and outputs only.
// - Interaction types: 0 reflect, 1 transmit (the tracer keeps the direction; path_dirN reports the Snell one),
//   2 refract (the tracer bends); 3, 4, 5 the same as scalar types, which run the same physics with one pressure
//   coefficient in the TE / admittance form (z = sqrt(eps / mu) is then the acoustic admittance).
// - Face angle: theta = acos(u . n) - pi/2 is the angle to the face, >= 0 for a front-side hit (the vertices
//   counter-clockwise seen from the ray); |cos(theta + pi/2)| is the cosine to the normal.
// - Coefficients: TE / TM; a transmitted coefficient is a normal-flux amplitude (flux_amp), so
//   0.5 (|cTE|^2 + |cTM|^2) is the power that crosses. xprmat per ray: [ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]
//   (qd_polbasis) or [Re Im] for the scalar types.
// - Geometry: O the segment's origin, D its destination, F and S the first and second hit (FBS, SBS), N a face
//   normal (oriented to the incidence side where a direction is formed), V the VBS.
// - The relaunch point lies 8 float ulp of the hit point's magnitude past it (qd_relaunch_offset): scenes are
//   meant to sit in local coordinates near the origin.
//
// The material model (MaterialCols reads the columns, Material holds one material in double):
// - eps_r = a (f/fRef)^b - j 17.98 sigma / fGHz with sigma = c (f/fRef)^d (ITU-R P.2040), plus a Lorentz pole
//   (resF, resQ, resS) for a material without the mass column; mu_r = e (f/fRef)^f - j 17.98 g (f/fRef)^h / fGHz.
// - Loss: 8.686 k0 |Im n| and alpha (f/fRef)^alphaB dB/m in the medium; att (f/fRef)^attB dB once at entry.
// - tf (f/fRef)^tfB, plus a resonant peak (tfP, tfF, tfQ), clamped to [-1, 1]: the transmission factor of a face.
//   > 0 leaks reflected energy into transmission, < 0 forces reflection. At an open face (a wave on both sides)
//   it is an ideal transformer that keeps the normal flux (tf_face); at a closed face (past the critical angle) a
//   matched resistive pad in front of the reactance behind it (closed_face).
// - thk > 0 makes a material a sheet: a face that stands for a layer of that thickness, its internal series
//   summed in one event (qd_stack_resolve) instead of traced. Two co-located sheets are an ordered stack.
// - The acoustic mapping (in the material table, not in this code): frequencies are scaled so that wavelengths
//   are kept (1 kHz is 0.874 GHz), and eps, mu are set from the impedance (and the sound speed).
// - m > 0, the mass column, is the acoustic switch. The material is a lumped plate with no wave inside: its eps,
//   mu are an impedance mapping for its faces, no tracer bends into it, a wave in it is held undeviated, and no
//   series is summed in it. Its transmission is the mass term: the impedance ratio X0 = (f/fRef) d of its normal
//   thickness d (mass_X0, with the res* and coi* impedances), 20 log10 X0 dB at normal incidence and the mass
//   law's angle shape (mass_TL; with side > 0 the finite panel of Davy). Bonded mass-law bodies are one plate
//   (their X0 add), and a plate left into a wave medium passes the plate rule of its two outer faces (qd_plate).
//   A peaked mass-law face (tfP > 0) is a locally reacting series impedance (zeta).
//
// Interface physics: interact_with(A, B) is the face between the medium the wave is in and the next one for one
// interaction type, at a real angle or at a recovered complex cosine: Fresnel (ITU-R P.2040 eq. 31, 32), total
// reflection (only towards a lossless medium), tf, the held wave of a plate, att.
//
// The solver: qd_stack_resolve gives the transmit and reflect ports of a sheet or a stack of two between two
// half-spaces (a recursion from the far end; the medium the sheet displaces is taken back out).
// qd_slab_resolve gives the series factor S = 1 / (1 - r_near r_far phi^2) of a traced solid, where its round
// trip rho = |r_near r_far| |phi|^2 reaches eps.
//
// The state machine (ray_state_update). Three 16-bit words per ray carry what an event hands to the next:
//   current  bits 0-13 the medium the ray is in (0 air); bit 14 cur_bent: the leg follows a bend the tracer
//            traced (types 2, 5); bit 15 cur_summed: the leg's series was summed, so a later reflection in the
//            leg (already inside the sum) is killed
//   prev     bits 0-13 the medium the leg was entered from, or the sheet crossed or reflected off (the mirror
//            behind the ray); bit 14 prev_embedded: that sheet lies inside the current medium (for a solid: the
//            shared face was evaluated at a real-invariant cosine); bit 15 prev_nonparallel: the leg's faces are
//            not parallel, so no series is summed (eps > 0) and no cosine recovered
//   buffer   bits 0-13 a deferred medium (overlapping or corner geometry); bit 14 buf_followed: the leg began at
//            a reflection that left its series to the tracer (with an index: the layer of a stack behind the ray);
//            or, under bit 15 buf_mass_code, the thickness code of a bonded mass-law body crossed into the leg
// and the ledger (acc_dist_refracted, acc_dist_geometric) holds the length of the current leg, charged when the
// leg closes. An event: the face is redone at the VBS angle (face_at_vbs), the leg's series is summed
// (sum_series) or left to the tracer, the leg is closed out (close_leg), the words are written. A traced ray
// carries only the real projection of a complex in-medium cosine; recover_cosine inverts the tracer's map to the
// exact cosine, so the entry and the exit of a lossy layer are evaluated at one cosine.
//
// Sections of the anonymous namespace: 1 constants and vector helpers; 2 material model; 3 interface physics;
// 4 slab and stack solver; 5 sheets (both ray functions); 6 face-hit geometry (ray_mesh_interact); 7 the pure
// parts of the state machine (ray_state_update). Then the public functions, each after its API block.

#include "quadriga_tools.hpp"

#include <complex>
#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <cstdint>

#include "bits.hpp"

namespace
{
    // ---- 1 Constants and vector helpers -------------------------------------------------------------------

    // Faces of a hit closer than this (m) are one co-located pair: a media-to-media boundary, a glued sheet,
    // a stack.
    constexpr double colocation_dist = 0.001;

    constexpr double half_pi = 1.5707963267948966;

    // One float ulp at magnitude one, as written (the relaunch offset and the float rounding allowance).
    constexpr double float_ulp = 1.1920929e-7;

    // Shortest geometric path in a traced mass-law solid that still carries a mass term (m).
    constexpr double mass_min_len = 0.0015;

    // The state words of ray_state_update: one mask per field and word.
    constexpr int word_mtl = 0x3FFF;         // material index, bits 0-13 (every word)
    constexpr int word_code = 0x7FFF;        // thickness code, bits 0-14 (buffer word with buf_mass_code)
    constexpr int cur_summed = 0x8000;       // current: the leg's series was summed
    constexpr int cur_bent = 0x4000;         // current: the leg follows a traced bend
    constexpr int prev_nonparallel = 0x8000; // prev: the leg's faces are known not parallel
    constexpr int prev_embedded = 0x4000;    // prev: the sheet lies inside the current medium (prev_real_inv for a solid)
    constexpr int buf_mass_code = 0x8000;    // buffer: a thickness code, not a deferred medium
    constexpr int buf_followed = 0x4000;     // buffer: followed, or with an index the stack layer behind

    // Free-space wavenumber (rad/m) at fGHz.
    inline double k0(double fGHz)
    {
        return 2.0 * 3.14159265358979323846 * fGHz * 1e9 / 299792458.0;
    }

    // A refraction-root argument that is negative real up to rounding, whose root has to be the decaying
    // one. For lossless media at a real angle the argument is exactly real and the sign of its zero imaginary
    // part picks the branch, so the sign is forced. From a recovered complex cosine it is real only to a few
    // ulp of either sign, hence the relative test; a lossy far medium lies far above the bound.
    inline bool qd_evanescent(const std::complex<double> &arg)
    {
        return std::real(arg) < 0.0 && std::abs(std::imag(arg)) <= -1.0e-9 * std::real(arg);
    }

    // The square root of a refraction-root argument, the decaying branch where qd_evanescent says so.
    inline std::complex<double> decaying_root(std::complex<double> arg)
    {
        if (qd_evanescent(arg))
            arg = std::complex<double>(std::real(arg), -0.0);
        return std::sqrt(arg);
    }

    // Mirror reflection direction: d = u - 2*c*n, c = clamp(u.n)
    inline void qd_reflect(double Ux, double Uy, double Uz,    // Incoming direction, normalized
                           double Nx, double Ny, double Nz,    // Plane normal vector, normalized
                           double &Dx, double &Dy, double &Dz) // Outgoing direction (normalized if U and N are normalized)
    {
        double c = Ux * Nx + Uy * Ny + Uz * Nz;
        c = (c < -1.0) ? -1.0 : (c > 1.0 ? 1.0 : c);
        double a = 2.0 * c;
        Dx = Ux - a * Nx, Dy = Uy - a * Ny, Dz = Uz - a * Nz;
    }

    // Relaunch offset: a few float ULP at the FBS coordinate magnitude (2^-23), so the nudge survives
    // a store in single precision and the relaunched ray starts off the face without re-hitting it
    inline double qd_relaunch_offset(double Fx, double Fy, double Fz)
    {
        double scale = std::max({std::abs(Fx), std::abs(Fy), std::abs(Fz), 1.0});
        return 8.0 * scale * float_ulp;
    }

    // Snell refraction direction: normalize(eta*u + (eta*cos_in - Re(cos_theta2))*n)
    inline void qd_refract(double Ux, double Uy, double Uz,    // Incoming direction, normalized
                           double Nx, double Ny, double Nz,    // Plane normal vector, normalized
                           double eta,                         // Snell ratio sqrt|eta1*mu1 / eta2*mu2|
                           double cos_in,                      // Cosine of angle between normal vector and incoming ray
                           std::complex<double> cos_theta2,    // Cosine of angle between normal vector and outgoing ray
                           double &Dx, double &Dy, double &Dz) // Outgoing direction (normalized)
    {
        double s = eta * cos_in - std::real(cos_theta2);
        double Rx = eta * Ux + s * Nx, Ry = eta * Uy + s * Ny, Rz = eta * Uz + s * Nz;
        double inv = 1.0 / std::sqrt(Rx * Rx + Ry * Ry + Rz * Rz);
        Dx = Rx * inv, Dy = Ry * inv, Dz = Rz * inv;
    }

    // Builds incoming Q-basis from in, outgoing U-basis from out, writes the 8-element xprmat and
    // its power gain. is_scalar takes the single-coefficient path.
    inline void qd_polbasis(double Qx, double Qy, double Qz,
                            double Ux, double Uy, double Uz,
                            double Nx, double Ny, double Nz,
                            std::complex<double> cTE, std::complex<double> cTM,
                            bool is_scalar, double xprmat[8], double &out_gain)
    {
        double eTE_Re = std::real(cTE), eTE_Im = std::imag(cTE);
        double eTM_Re = std::real(cTM), eTM_Im = std::imag(cTM);

        if (is_scalar)
        {
            double coeff_Re = eTE_Re, coeff_Im = eTE_Im;
            xprmat[0] = coeff_Re, xprmat[1] = coeff_Im;
            xprmat[2] = 0.0, xprmat[3] = 0.0, xprmat[4] = 0.0;
            xprmat[5] = 0.0, xprmat[6] = 0.0, xprmat[7] = 0.0;
            out_gain = coeff_Re * coeff_Re + coeff_Im * coeff_Im;
            return;
        }

        double scl = 0.0;

        double eHx = -Qy + 3.0e-20, eHy = Qx, eHz = 0.0;
        scl = 1.0 / std::sqrt(eHx * eHx + eHy * eHy), eHx *= scl, eHy *= scl;
        double eVx = -Qz * eHy, eVy = Qz * eHx, eVz = Qx * eHy - Qy * eHx;
        double eQx = Qy * Nz - Qz * Ny + 3.0e-20, eQy = Qz * Nx - Qx * Nz, eQz = Qx * Ny - Qy * Nx;
        scl = 1.0 / std::sqrt(eQx * eQx + eQy * eQy + eQz * eQz), eQx *= scl, eQy *= scl, eQz *= scl;
        double ePx = eQy * Qz - eQz * Qy, ePy = eQz * Qx - eQx * Qz, ePz = eQx * Qy - eQy * Qx;

        bool do_base_transform = scl < 1.0e19;
        double Q1 = (do_base_transform) ? eVx * ePx + eVy * ePy + eVz * ePz : 1.0;
        double Q2 = (do_base_transform) ? eVx * eQx + eVy * eQy + eVz * eQz : 0.0;
        double Q3 = (do_base_transform) ? eHx * ePx + eHy * ePy + eHz * ePz : 0.0;
        double Q4 = (do_base_transform) ? eHx * eQx + eHy * eQy + eHz * eQz : 1.0;

        eHx = -Uy + 3.0e-20, eHy = Ux, eHz = 0.0;
        scl = 1.0 / std::sqrt(eHx * eHx + eHy * eHy), eHx *= scl, eHy *= scl;
        eVx = -Uz * eHy, eVy = Uz * eHx, eVz = Ux * eHy - Uy * eHx;
        eQx = Uy * Nz - Uz * Ny + 3.0e-20, eQy = Uz * Nx - Ux * Nz, eQz = Ux * Ny - Uy * Nx;
        scl = 1.0 / std::sqrt(eQx * eQx + eQy * eQy + eQz * eQz), eQx *= scl, eQy *= scl, eQz *= scl;
        ePx = eQy * Uz - eQz * Uy, ePy = eQz * Ux - eQx * Uz, ePz = eQx * Uy - eQy * Ux;

        do_base_transform = scl < 1.0e19;
        double U1 = (do_base_transform) ? eVx * ePx + eVy * ePy + eVz * ePz : 1.0;
        double U2 = (do_base_transform) ? eVx * eQx + eVy * eQy + eVz * eQz : 0.0;
        double U3 = (do_base_transform) ? eHx * ePx + eHy * ePy + eHz * ePz : 0.0;
        double U4 = (do_base_transform) ? eHx * eQx + eHy * eQy + eHz * eQz : 1.0;

        double VV_Re = U1 * Q1 * eTM_Re + U3 * Q2 * eTE_Re,
               VV_Im = U1 * Q1 * eTM_Im + U3 * Q2 * eTE_Im,
               HV_Re = U2 * Q1 * eTM_Re + U4 * Q2 * eTE_Re,
               HV_Im = U2 * Q1 * eTM_Im + U4 * Q2 * eTE_Im,
               VH_Re = U1 * Q3 * eTM_Re + U3 * Q4 * eTE_Re,
               VH_Im = U1 * Q3 * eTM_Im + U3 * Q4 * eTE_Im,
               HH_Re = U2 * Q3 * eTM_Re + U4 * Q4 * eTE_Re,
               HH_Im = U2 * Q3 * eTM_Im + U4 * Q4 * eTE_Im;

        xprmat[0] = VV_Re, xprmat[1] = VV_Im;
        xprmat[2] = HV_Re, xprmat[3] = HV_Im;
        xprmat[4] = VH_Re, xprmat[5] = VH_Im;
        xprmat[6] = HH_Re, xprmat[7] = HH_Im;

        out_gain = 0.5 * (VV_Re * VV_Re + VV_Im * VV_Im +
                          HV_Re * HV_Re + HV_Im * HV_Im +
                          VH_Re * VH_Re + VH_Im * VH_Im +
                          HH_Re * HH_Re + HH_Im * HH_Im);
    }

    // ---- 2 Material model ---------------------------------------------------------------------------------

    template <typename dtype>
    struct MaterialCols
    {
        arma::uword n_mtl = 0;         // Number of materials
        const dtype *fRef = nullptr;   // Reference frequency, GHz
        const dtype *a = nullptr;      // εr at fRef
        const dtype *b = nullptr;      // Frequency exponent for εr
        const dtype *c = nullptr;      // σ at fRef, S/m
        const dtype *d = nullptr;      // Frequency exponent for σ
        const dtype *e = nullptr;      // μr at fRef
        const dtype *f = nullptr;      // Frequency exponent for μr
        const dtype *g = nullptr;      // σμ (magnetic loss) at fRef
        const dtype *h = nullptr;      // Frequency exponent for σμ
        const dtype *thk = nullptr;    // Material thickness, m
        const dtype *att = nullptr;    // Penetration loss at fRef, dB
        const dtype *attB = nullptr;   // Frequency exponent for att
        const dtype *alpha = nullptr;  // In-medium absorption at fRef, dB/m
        const dtype *alphaB = nullptr; // Frequency exponent for alpha
        const dtype *m = nullptr;      // Mass-law switch (> 0: the mass law, 20 dB/decade)
        const dtype *resF = nullptr;   // Resonance frequency, GHz (a permittivity pole; with m, the plate's resonance)
        const dtype *resQ = nullptr;   // Resonance quality factor
        const dtype *resS = nullptr;   // Resonance strength (with m, its sign selects the form)
        const dtype *coiF = nullptr;   // Coincidence frequency, GHz (with m, the plate's bending stiffness)
        const dtype *coiQ = nullptr;   // Coincidence quality factor (with m, 1 / loss factor)
        const dtype *tf = nullptr;     // Transmission factor at fRef
        const dtype *tfB = nullptr;    // Frequency exponent for tf
        const dtype *tfP = nullptr;    // Resonant peak added to tf
        const dtype *tfF = nullptr;    // Frequency of that peak, GHz
        const dtype *tfQ = nullptr;    // Quality factor of that peak
        const dtype *side = nullptr;   // Panel size of a mass-law material, sqrt(area), m (0: the infinite plate)

        MaterialCols() = default; // All pointers stay nullptr

        MaterialCols(const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop)
        {
            n_mtl = 0;
            bool seen = false;
            for (const auto &kv : mtl_prop)
            {
                if (kv.second.empty())
                    continue;
                arma::uword len = (arma::uword)kv.second.size();
                if (!seen)
                {
                    n_mtl = len;
                    seen = true;
                }
                else if (len != n_mtl)
                    throw std::invalid_argument("Material property column '" + kv.first + "' has length " +
                                                std::to_string(len) + ", expected " + std::to_string(n_mtl) +
                                                " (all columns must have the same number of materials).");
            }

            auto resolve = [&mtl_prop](const std::string &key) -> const dtype *
            {
                auto it = mtl_prop.find(key);
                return (it == mtl_prop.end() || it->second.empty()) ? nullptr : it->second.data();
            };

            fRef = resolve("fRef");
            a = resolve("a");
            b = resolve("b");
            c = resolve("c");
            d = resolve("d");
            e = resolve("e");
            f = resolve("f");
            g = resolve("g");
            h = resolve("h");
            thk = resolve("thk");
            att = resolve("att");
            attB = resolve("attB");
            alpha = resolve("alpha");
            alphaB = resolve("alphaB");
            m = resolve("m");
            resF = resolve("resF");
            resQ = resolve("resQ");
            resS = resolve("resS");
            coiF = resolve("coiF");
            coiQ = resolve("coiQ");
            tf = resolve("tf");
            tfB = resolve("tfB");
            tfP = resolve("tfP");
            tfF = resolve("tfF");
            tfQ = resolve("tfQ");
            side = resolve("side");

            // Physical sanity: reject corrupt input rather than silently clamping it. Loss-like terms must
            // be non-negative (a negative would be gain); material constants must be positive. Reported
            // material index is 1-based to match the public material indexing.
            auto require = [this](const dtype *col, const char *name, bool strict_positive)
            {
                if (col == nullptr)
                    return;
                for (arma::uword i = 0; i < n_mtl; ++i)
                {
                    double v = (double)col[i];
                    if (strict_positive ? (v <= 0.0) : (v < 0.0))
                        throw std::invalid_argument(std::string("Material property '") + name + "' = " +
                                                    std::to_string(v) + " at material " + std::to_string(i + 1) +
                                                    (strict_positive ? " must be positive." : " must be non-negative."));
                }
            };

            require(fRef, "fRef", true);    // reference frequency > 0
            require(a, "a", true);          // relative permittivity > 0 (use >= 1 for a strict vacuum floor)
            require(e, "e", true);          // relative permeability > 0 (diamagnets < 1 allowed)
            require(c, "c", false);         // conductivity >= 0
            require(g, "g", false);         // magnetic loss >= 0
            require(thk, "thk", false);     // thickness >= 0
            require(att, "att", false);     // penetration loss >= 0
            require(alpha, "alpha", false); // in-medium absorption >= 0
            require(m, "m", false);         // mass-law switch >= 0
            require(resF, "resF", false);   // resonance frequency >= 0 (0 disables)
            require(resQ, "resQ", false);   // resonance Q >= 0 (0 disables)
            require(coiF, "coiF", false);   // coincidence frequency >= 0 (0 disables)
            require(coiQ, "coiQ", false);   // coincidence Q >= 0
            require(tfF, "tfF", false);     // tf peak frequency >= 0 (0 disables)
            require(side, "side", false);   // panel size >= 0 (0: the infinite plate)
        }

        // A sheet: a material with thk > 0 (idx 1-based, 0 = air, never a sheet).
        bool is_sheet(arma::uword idx) const
        {
            if (idx > n_mtl)
                throw std::out_of_range("Material index " + std::to_string(idx) +
                                        " out of range [0, " + std::to_string(n_mtl) + "]");

            if (thk == nullptr)
                return false;
            if (idx == 0) // Not in table, default
                return false;
            if (thk[idx - 1] == (dtype)0.0)
                return false;
            return true;
        }
    };

    // Davy's forced radiation efficiency of a finite flat panel, averaged over the azimuth, for a
    // square of the given side (2a = side, S = side^2) at incidence cosine g: J. L. Davy, JASA 126 (2009)
    // 694, Eqs. 19, 28, 34, 36, 37 with n = 2, w = 1.3, beta = 0.124. It is 1 / cos up to an angle that
    // grows with the size in wavelengths, then levels off at about 0.38 sqrt(k side) at grazing; q = 2 pi /
    // (k^2 S) is a panel small in wavelengths, radiating less than an infinite plate at every angle. The
    // acoustic mapping keeps wavelengths, so k = 2 pi f / c0 of the mapped frequency. ka is held at 0.1 or
    // above, below Davy's range and below 5 Hz for a 1.35 m pane.
    inline double davy_sigma(double fGHz, double g, double side)
    {
        double k = k0(fGHz), ka = std::max(0.5 * k * side, 0.1);
        double p = std::min(1.0, 1.3 * std::sqrt(half_pi / ka));
        double h = 1.0 / (2.0 / 3.0 * std::sqrt(0.6366197723675814 * ka) - 0.124);
        double q = 6.283185307179586 / (k * k * side * side), x = (g >= p) ? g : h - (h / p - 1.0) * g;
        return 1.0 / std::sqrt(x * x + q * q);
    }

    // Mass term in dB of a plate with impedance ratio X0 at the cosine mass_c in air; zero for X0 <= 1.
    // side = 0, the infinite plate: the forced wave radiates as 1 / cos, so the plate turns transparent at
    // grazing. side > 0 (sqrt of the panel area): the radiation cosine is 1 / sigma (davy_sigma) and the plate
    // passes the finite plate's gain 1 / (sigma c) on top, tau = |1 + j X0 / sigma|^-2 / (sigma c) in the
    // normalized form below, clipped at one.
    inline double mass_TL(double X0, double mass_c, double fGHz, double side)
    {
        if (!(X0 > 1.0) || !(mass_c > 0.0))
            return 0.0;
        double A = 20.0 * std::log10(X0);
        if (!(side > 0.0))
        {
            if (mass_c >= 1.0)
                return A;
            double x = X0 * mass_c;
            return A * std::log10(1.0 + x * x) / std::log10(1.0 + X0 * X0);
        }
        double c = std::min(mass_c, 1.0), sig = davy_sigma(fGHz, c, side), x = X0 / sig;
        return std::max(0.0, A * std::log10(1.0 + x * x) / std::log10(1.0 + X0 * X0) + 10.0 * std::log10(sig * c));
    }

    struct Material
    {
        double fRef = 1.0;   // Reference frequency, GHz
        double a = 1.0;      // εr at fRef
        double b = 0.0;      // Frequency exponent for εr
        double c = 0.0;      // σ at fRef, S/m
        double d = 0.0;      // Frequency exponent for σ
        double e = 1.0;      // μr at fRef
        double f = 0.0;      // Frequency exponent for μr
        double g = 0.0;      // σμ (magnetic loss) at fRef
        double h = 0.0;      // Frequency exponent for σμ
        double thk = 0.0;    // Thickness, m
        double att = 0.0;    // Penetration loss at fRef, dB
        double attB = 0.0;   // Frequency exponent for att
        double alpha = 0.0;  // In-medium absorption at fRef, dB/m
        double alphaB = 0.0; // Frequency exponent for alpha
        double m = 0.0;      // Mass-law switch (> 0: the mass law, 20 dB/decade)
        double resF = 0.0;   // Resonance frequency, GHz (a permittivity pole; with m, the plate's resonance)
        double resQ = 0.0;   // Resonance quality factor
        double resS = 0.0;   // Resonance strength (with m, its sign selects the form)
        double coiF = 0.0;   // Coincidence frequency, GHz (with m, the plate's bending stiffness)
        double coiQ = 0.0;   // Coincidence quality factor (with m, 1 / loss factor)
        double tfR = 0.0;    // Transmission factor at fRef
        double tfB = 0.0;    // Frequency exponent for tf
        double tfP = 0.0;    // Resonant peak added to tf
        double tfF = 0.0;    // Frequency of that peak, GHz
        double tfQ = 0.0;    // Quality factor of that peak
        double side = 3.29;  // Panel size of a mass-law material, sqrt(area), m (0: the infinite plate; 3.29: 4.0 x 2.7 m)

        Material() = default;

        template <typename dtype>
        Material(const MaterialCols<dtype> &cols, arma::uword idx = 0) // 1-based index, 0 = default (no material)
        {
            if (idx > cols.n_mtl)
                throw std::out_of_range("Material index " + std::to_string(idx) +
                                        " out of range [0, " + std::to_string(cols.n_mtl) + "]");

            if (idx == 0) // no material -> keep defaults (air / vacuum)
                return;

            arma::uword i = idx - 1; // 1-based index -> 0-based column position

            fRef = cols.fRef ? (double)cols.fRef[i] : fRef;
            a = cols.a ? (double)cols.a[i] : a;
            b = cols.b ? (double)cols.b[i] : b;
            c = cols.c ? (double)cols.c[i] : c;
            d = cols.d ? (double)cols.d[i] : d;
            e = cols.e ? (double)cols.e[i] : e;
            f = cols.f ? (double)cols.f[i] : f;
            g = cols.g ? (double)cols.g[i] : g;
            h = cols.h ? (double)cols.h[i] : h;
            thk = cols.thk ? (double)cols.thk[i] : thk;
            att = cols.att ? (double)cols.att[i] : att;
            attB = cols.attB ? (double)cols.attB[i] : attB;
            alpha = cols.alpha ? (double)cols.alpha[i] : alpha;
            alphaB = cols.alphaB ? (double)cols.alphaB[i] : alphaB;
            m = cols.m ? (double)cols.m[i] : m;
            resF = cols.resF ? (double)cols.resF[i] : resF;
            resQ = cols.resQ ? (double)cols.resQ[i] : resQ;
            resS = cols.resS ? (double)cols.resS[i] : resS;
            coiF = cols.coiF ? (double)cols.coiF[i] : coiF;
            coiQ = cols.coiQ ? (double)cols.coiQ[i] : coiQ;
            tfR = cols.tf ? (double)cols.tf[i] : tfR;
            tfB = cols.tfB ? (double)cols.tfB[i] : tfB;
            tfP = cols.tfP ? (double)cols.tfP[i] : tfP;
            tfF = cols.tfF ? (double)cols.tfF[i] : tfF;
            tfQ = cols.tfQ ? (double)cols.tfQ[i] : tfQ;
            side = cols.side ? (double)cols.side[i] : side;
        }

        // Relative permittivity, with its resonance: a complex Lorentz pole, inactive unless
        // resF > 0, resQ > 0 and resS != 0. One permittivity at a face and in propagation. A mass-law
        // body has no wave inside: its resonance is the plate's and acts in mass_X0.
        std::complex<double> eta(double fGHz = 1.0) const
        {
            double f_rel = fGHz / fRef;
            double eta_r = a * std::pow(f_rel, b);
            double sigma = c * std::pow(f_rel, d);
            double eta_i = -17.98 * sigma / fGHz;
            std::complex<double> pole(0.0, 0.0);
            if (resF > 0.0 && resQ > 0.0 && resS != 0.0 && !(m > 0.0))
                pole = (resS * resF * resF) / std::complex<double>(resF * resF - fGHz * fGHz, (resF / resQ) * fGHz);
            return std::complex<double>(eta_r, eta_i) + pole;
        }

        // Relative permeability
        std::complex<double> mu(double fGHz = 1.0) const
        {
            double f_rel = fGHz / fRef;
            double mu_r = e * std::pow(f_rel, f);
            double sigma_m = g * std::pow(f_rel, h);
            return std::complex<double>(mu_r, -17.98 * sigma_m / fGHz);
        }

        // In-medium gain, linear
        double medium_gain(double dist_refract,                 // In-medium distance (refracted path)
                           double fGHz = 1.0,                   // Frequency
                           double abs_cos_theta_t = 1.0,        // Cosine of incidence angle @ FBS/VBS (refracted path)
                           double dist_geo = 0.0,               // Geometric distance (0.0 = dist_refract)
                           double abs_cos_theta = 0.0,          // Cosine of incidence angle @ orig (only used when dist_geo > 0)
                           double mass_min_path = mass_min_len, // Shortest geometric path that still carries a mass-law term
                           double mass_normal = -1.0,           // Surface-mass surrogate with the angle removed (< 0 = dist_geo * abs_cos_theta)
                           double mass_cos = -1.0,              // Angle factor for the mass-law term (< 0 = abs_cos_theta)
                           double mass_bonded = 0.0) const      // Summed X0 of the bonded mass-law bodies already crossed
        {
            if (dist_geo == 0.0) // Identity fallback
                dist_geo = dist_refract, abs_cos_theta = abs_cos_theta_t;

            // ITU-R P.2040 attenuation rate, 8.686 k0 |Im n| dB per meter
            double loss = dist_refract * 8.686 * fGHz * std::abs(std::imag(std::sqrt(eta(fGHz) * mu(fGHz)))) / 0.0477135;
            loss += dist_refract * alpha * std::pow(fGHz / fRef, alphaB);

            // The floor mass_min_path drops the mass term on a sub-millimetre traced path (a meshing artifact); the
            // sheet path passes 0 and always keeps it.
            //
            // The panel impedance and the air's admittance enter as one product x = X0 cos(theta): X0 the impedance
            // ratio of mass_X0 from the surface-mass surrogate mass_normal (default dist_geo * abs_cos_theta, the
            // normal thickness), cos(theta) = mass_cos (default abs_cos_theta), the cosine the wave has in air.
            // mass_TL keeps the table's normal-incidence value A = 20 log10 X0 and gives the angle the mass law's
            // own shape, A log10(1 + x^2) / log10(1 + X0^2): A at normal incidence, zero at tangency (side = 0, the
            // infinite plate); with side > 0 Davy's efficiency of the finite panel takes the place of the cosine.
            //
            // Bonded mass-law bodies (no air gap) move as one plate: their X0 add, and this body pays
            // TL(X_run + X0) - TL(X_run), X_run = mass_bonded the X0 crossed before; each X0 unclipped.
            if (m > 0.0 && dist_geo > mass_min_path)
            {
                double mass_c = (mass_cos >= 0.0) ? mass_cos : abs_cos_theta;
                loss += mass_TL(mass_bonded + mass_X0((mass_normal >= 0.0) ? mass_normal : dist_geo * abs_cos_theta, fGHz, mass_c), mass_c, fGHz, side) -
                        mass_TL(mass_bonded, mass_c, fGHz, side);
            }
            return std::pow(10.0, -0.1 * loss);
        }

        // Impedance ratio X0 = (f / fRef) M of a mass-law body from its surface-mass surrogate M, zero without
        // m > 0: 10^(A/20) of its normal-incidence term A, unclipped. In series with the mass, both at the cosine
        // c the wave has in air:
        // - resS > 0, a mass-air-mass resonance (a leaf on its cavity, a spring stiffer by 1 / c^2):
        //   X0 min(1, |1 - (f c / resF)^2 + j f c / (resF resQ)|), the dip only (a built unit is bridged at its edge);
        // - resS < 0, the panel's first mode: X0 |1 - (resF / f)^2 - j resF / (f resQ)|;
        // - coiF, coiQ, the bending stiffness (coincidence frequency, 1 / loss factor) at the trace wavenumber:
        //   X0 |1 - (f / coiF)^2 sin^4 (1 + j / coiQ)|.
        double mass_X0(double mass_normal, double fGHz = 1.0, double mass_c = 1.0) const
        {
            if (!(m > 0.0))
                return 0.0;
            double mass_arg = (fGHz / fRef) * mass_normal;
            double x = fGHz * mass_c / resF, y = resF / fGHz, res = 1.0;
            if (resF > 0.0 && resQ > 0.0 && resS > 0.0)
                res = std::min(1.0, std::abs(std::complex<double>(1.0 - x * x, x / resQ)));
            else if (resF > 0.0 && resQ > 0.0 && resS < 0.0)
                res = std::abs(std::complex<double>(1.0 - y * y, -y / resQ));
            double s2 = 1.0 - mass_c * mass_c, w = (coiF > 0.0) ? (fGHz / coiF) * (fGHz / coiF) * s2 * s2 : 0.0;
            double coi = (coiF > 0.0 && coiQ > 0.0) ? std::abs(std::complex<double>(1.0 - w, -w / coiQ)) : 1.0;
            return (mass_arg > 0.0) ? mass_arg * res * coi : 0.0;
        }

        // Linear gain of the non-conductive absorption only: the alpha column, without the
        // conductivity loss that a complex refractive index already carries. A caller that takes the
        // conductivity and the propagation phase from one complex expression, exp(-j k0 n L), must not
        // draw the same loss from medium_gain a second time; it takes this instead.
        double excess_gain(double dist, double fGHz = 1.0) const
        {
            double loss = dist * alpha * std::pow(fGHz / fRef, alphaB);
            return std::pow(10.0, -0.1 * loss);
        }

        // Per-entry interface gain, linear
        double interface_gain(double fGHz = 1.0) const
        {
            double loss = att * std::pow(fGHz / fRef, attB);
            return std::pow(10.0, -0.1 * loss);
        }

        // Transmission factor at fGHz, clamped to [-1, 1]: a power law, and a resonant peak where tfP
        // and tfF are set (a panel moving on its mounting, a leaf on its cavity: the room-side absorption
        // of a mass-law face peaks there). The peak is the lumped resonator's, symmetric in log frequency
        // and zero at both ends; |tfQ| is the absorber's Q_alpha. Every caller's frequency is positive.
        // Where a mass-law face carries a positive peak the face is its series impedance (zeta, below), whose
        // normal-incidence absorption is this law; peak = false gives the power-law part alone.
        double tf(double fGHz = 1.0, bool peak = true) const
        {
            double v = tfR * std::pow(fGHz / fRef, tfB);
            if (peak && tfP != 0.0 && tfF > 0.0)
            {
                double x = tfQ * (fGHz / tfF - tfF / fGHz);
                v += tfP / (1.0 + x * x);
            }
            return (v < -1.0) ? -1.0 : ((v > 1.0) ? 1.0 : v);
        }

        // A peaked mass-law face is a locally reacting face (a panel on a partitioned cavity, a face known
        // from catalog data): its peak is the series impedance per Z0 of a resonator,
        // zeta = R' + j (1 + R') |tfQ| (f/tfF - tfF/f) with tfP = 4 R' / (1 + R')^2, over-damped (R' >= 1), or
        // under-damped (1/R') where tfQ < 0; tfP above one is R' = 1. At normal incidence in air it absorbs the
        // peak of tf. Gated on m: a wave medium keeps the transformer, and a negative peak forces reflection.
        bool peaked() const { return m > 0.0 && tfP > 0.0 && tfF > 0.0; }
        std::complex<double> zeta(double fGHz) const
        {
            const double pk = std::min(tfP, 1.0);
            double Rp = ((2.0 - pk) + 2.0 * std::sqrt(1.0 - pk)) / pk;
            Rp = (tfQ < 0.0) ? 1.0 / Rp : Rp;
            return {Rp, (1.0 + Rp) * std::abs(tfQ) * (fGHz / tfF - tfF / fGHz)};
        }

        bool same_as(const Material &other) const
        {
            return fRef == other.fRef &&
                   a == other.a &&
                   b == other.b &&
                   c == other.c &&
                   d == other.d &&
                   e == other.e &&
                   f == other.f &&
                   g == other.g &&
                   h == other.h &&
                   thk == other.thk &&
                   att == other.att &&
                   attB == other.attB &&
                   alpha == other.alpha &&
                   alphaB == other.alphaB &&
                   m == other.m &&
                   resF == other.resF &&
                   resQ == other.resQ &&
                   resS == other.resS &&
                   coiF == other.coiF &&
                   coiQ == other.coiQ &&
                   tfR == other.tfR &&
                   tfB == other.tfB &&
                   tfP == other.tfP &&
                   tfF == other.tfF &&
                   (!(m > 0.0) || side == other.side) && // the panel size is read only where m > 0
                   tfQ == other.tfQ;
        }
    };

    // ---- 3 Interface physics ------------------------------------------------------------------------------

    // Combined transmission-factor reflection for a two-medium interface (symmetric in the two media).
    // tf+ = max(tf,0) leaks reflection energy into transmission; tf- = max(-tf,0) forces reflection.
    // Stays in [0,1]; tf = -1 on either face gives R_eff = 1. bg: a peaked side contributes its
    // power-law part only, the pad in front of its series impedance.
    inline double apply_tf_pair(const Material &A, const Material &B, double R0, double fGHz = 1.0, bool bg = false)
    {
        double tfX = A.tf(fGHz, !(bg && A.peaked())), tfY = B.tf(fGHz, !(bg && B.peaked()));
        double tfXp = (tfX > 0.0) ? tfX : 0.0, tfXm = (tfX < 0.0) ? -tfX : 0.0;
        double tfYp = (tfY > 0.0) ? tfY : 0.0, tfYm = (tfY < 0.0) ? -tfY : 0.0;
        R0 = (R0 < 0.0) ? 0.0 : ((R0 > 1.0) ? 1.0 : R0);
        double R_leak = R0 * (1.0 - tfXp) * (1.0 - tfYp);
        double tfm = (tfXm > tfYm) ? tfXm : tfYm;
        return R_leak + (1.0 - R_leak) * tfm;
    }

    // tf at a closed face of A towards B (a wave on one side only). Behind it is a reactance that carries no
    // flux, so the leak is a matched resistive pad in front of it, formed on the modulus of the immittance y of
    // the wave's side: a real element, so the face creates nothing and passes what it absorbs. In front of a
    // lossless medium it is the pair rule on a reflectance of one (sqrt(R') on the reflection, 1 - R' of the
    // flux) and the limit of tf_face towards grazing: no step at the critical angle. A peaked B replaces the
    // reactance by its series impedance behind the pad of its power-law part: met as 1/zeta for TE and the
    // scalar type, as zeta for TM (tm). r: the raw reflection in (written for y: z c for TE and scalar,
    // conj(z) c for TM), the face's out. Returns the amplitude passed per unit forward amplitude, in normal flux
    // over flux_in. Only the far side's reactance is read, so a forced verdict closes the face too. A pair
    // without tf, or at tf = -1, is left raw and passes nothing.
    inline double closed_face(const Material &A, const Material &B, double fGHz, std::complex<double> &r, const std::complex<double> &y,
                              double flux_in, bool pass_all = false, bool tm = false)
    {
        const bool pk = B.peaked();
        double s = pass_all ? 0.0 : std::sqrt(apply_tf_pair(A, B, 1.0, fGHz, pk)), nu = (1.0 - s) / (1.0 + s), a = std::abs(y);
        std::complex<double> e = y / a, b(0.0, std::imag(e * (1.0 - r) / (1.0 + r)));
        if (pk && a > 0.0)
            b = tm ? B.zeta(fGHz) * std::norm(std::sqrt(A.eta(fGHz) / A.mu(fGHz))) / a : 1.0 / (B.zeta(fGHz) * a);
        std::complex<double> w = (nu + b) / (1.0 + nu * b);
        if ((!(nu > 0.0) && !pk) || !(a > 0.0) || !std::isfinite(std::abs(b)))
            return 0.0;
        r = (e - w) / (e + w);
        return (flux_in > 0.0) ? std::abs(1.0 + r) * std::sqrt(std::real(w) * a / flux_in) : 0.0;
    }

    // tf at an open face of A towards B (a wave on both sides): an ideal transformer of real ratio, scaling the
    // tangential field by k and its partner by 1/k, so the normal flux is continuous and the face creates
    // nothing, lossy or evanescent. The ratio is the pair rule on the modulus x0 of the admittance ratio (on the
    // reflectance where that ratio is real). r: the raw reflection in, the face's out; returns the scale of both
    // crossings. At a match (within rounding) the medium that forces less sees the forced reflection negative,
    // so the face is one face from both sides. A peaked B (given y, the immittance of the wave's side) is its
    // series impedance instead, as in closed_face; the scale is then what the face absorbs over the raw crossing.
    inline std::complex<double> tf_face(const Material &A, const Material &B, double fGHz, std::complex<double> &r,
                                        const std::complex<double> *y = nullptr, bool tm = false)
    {
        if (y && B.peaked())
        {
            double R0 = std::norm(r);
            closed_face(A, B, fGHz, r, *y, 1.0, false, tm);
            return (R0 < 1.0) ? std::sqrt(std::max(0.0, 1.0 - std::norm(r)) / (1.0 - R0)) : 0.0;
        }
        std::complex<double> u = (1.0 - r) / (1.0 + r);
        double x0 = std::abs(u);
        if ((A.tf(fGHz) == 0.0 && B.tf(fGHz) == 0.0) || !(x0 > 0.0) || std::isinf(x0))
            return 1.0;
        double rm = (1.0 - x0) / (1.0 + x0), Rm = apply_tf_pair(A, B, rm * rm, fGHz), s = ((std::abs(rm) <= 1.0e-9) ? (A.tf(fGHz) > B.tf(fGHz)) : (rm < 0.0)) ? -std::sqrt(Rm) : std::sqrt(Rm);
        std::complex<double> D = (1.0 + s) + (u / x0) * (1.0 - s);
        r = ((1.0 + s) - (u / x0) * (1.0 - s)) / D;
        return std::sqrt((1.0 - Rm) / x0) * (1.0 + u) / D;
    }

    // Per-channel normal-flux amplitude factors of a crossing z1, ct1 -> z2, ct2 (z = sqrt(eps/mu), cosines
    // complex): the normal flux is |E|^2 Re(z c) for TE and |E|^2 Re(conj(z) c) for TM, so a transmitted
    // amplitude carries sqrt(Re(z2 c2) / Re(z1 c1)) per channel. For a real ct1 both denominators are
    // Re(z1) ct1; they differ where the wave comes from a lossy medium at a recovered complex cosine (the far
    // face of a traced solid). In this form a chain of crossings telescopes where both faces of a shared
    // medium use the same cosine. Past the critical angle of a lossless far medium Re(z2 c2) = 0 closes the
    // port. Scalar types use the TE form (z the acoustic admittance, intensity |p|^2 Re(z c)).
    inline void flux_amp(const std::complex<double> &z1, const std::complex<double> &z2,
                         const std::complex<double> &ct1, const std::complex<double> &ct2,
                         double &s_te, double &s_tm)
    {
        double den_te = std::real(z1 * ct1); // incidence-side normal flux, per channel
        double den_tm = std::real(std::conj(z1) * ct1);
        double n_te = std::real(z2 * ct2);
        double n_tm = std::real(std::conj(z2) * ct2);

        // A non-positive denominator is exact tangency or a non-propagating incidence-side
        // wave: there is no incident flux to convert and the port closes.
        s_te = (den_te > 1.0e-300 && n_te > 0.0) ? std::sqrt(n_te / den_te) : 0.0;
        s_tm = (den_tm > 1.0e-300 && n_tm > 0.0) ? std::sqrt(n_tm / den_tm) : 0.0;
    }

    // What interact_with returns.
    struct Interface
    {
        std::complex<double> cTE, cTM; // R for reflection (types 0, 3), T for transmission (1, 2, 4, 5)
        std::complex<double> cos_t2;   // refraction cosine (the direction of types 2 and 5)
        std::complex<double> n2_ratio; // eta1 mu1 / (eta2 mu2)
        double snell = 1.0;            // Snell ratio sqrt|eta1 mu1 / (eta2 mu2)|
        double gain = 0.0;             // power gain 0.5 (|cTE|^2 + |cTM|^2)
        bool tir = false;              // total reflection
    };

    // The interface between A (the wave is in it) and B, for one interaction type (0 EM reflect, 1 EM
    // transmit, 2 EM refract, 3 scalar reflect, 4 scalar transmit, 5 scalar refract). Transmit types carry the
    // per-channel normal-flux amplitude (flux_amp) and B's att; reflection carries neither. theta is the face
    // angle of a hit; a caller inside a traced medium passes the recovered complex cosine in cos_theta_in
    // instead (theta is then not read), and a real cos_theta_in reproduces the angle path. undeviated: the wave
    // is held at an angle it was not refracted to (a leak in a plate). force_tir closes the face.
    inline Interface interact_with(const Material &A, const Material &B, int interaction_type, double theta, double fGHz,
                                   const std::complex<double> *cos_theta_in = nullptr, bool undeviated = false, bool force_tir = false)
    {
        // The angle path forms sin^2 from sqrt(1 - c^2) and keeps its own TIR test: folded into the complex form
        // it moves ct2 by one ulp at a matched face, enough to send a tf-forced reflection down another branch.
        std::complex<double> ct1, sin2_theta;
        double sin_theta_real = -1.0; // >= 0 on the angle path only, and its TIR test
        if (cos_theta_in)
        {
            ct1 = *cos_theta_in;
            sin2_theta = 1.0 - ct1 * ct1;
        }
        else
        {
            double abs_cos_theta = std::abs(std::cos(theta + half_pi)); // |OF . N| convention
            abs_cos_theta = (abs_cos_theta > 1.0) ? 1.0 : abs_cos_theta;
            sin_theta_real = std::sqrt(1.0 - abs_cos_theta * abs_cos_theta);
            ct1 = std::complex<double>(abs_cos_theta, 0.0);
            sin2_theta = std::complex<double>(sin_theta_real * sin_theta_real, 0.0);
        }

        std::complex<double> eta1 = A.eta(fGHz);
        std::complex<double> eta2 = B.eta(fGHz);
        std::complex<double> mu1 = A.mu(fGHz);
        std::complex<double> mu2 = B.mu(fGHz);

        std::complex<double> n2_ratio = (eta1 * mu1) / (eta2 * mu2);
        double snell = std::sqrt(std::abs(n2_ratio));

        // The angle path keeps its original association: reassociating moved coefficients by up to 6e-14.
        std::complex<double> ct2 = decaying_root((sin_theta_real >= 0.0)
                                                     ? (1.0 - n2_ratio * sin_theta_real * sin_theta_real)
                                                     : (1.0 - n2_ratio * sin2_theta));
        std::complex<double> z1 = std::sqrt(eta1 / mu1); // admittances sqrt(eps/mu)
        std::complex<double> z2 = std::sqrt(eta2 / mu2);

        // Total reflection from the transverse term of the refraction root: snell sin(theta) >= 1 for a real
        // angle; for a recovered cosine the term carried in from the entry medium (Snell is transitive), so the
        // outer pair decides. Two spellings only to keep the angle path's rounding. Only a lossless far medium
        // has a critical angle (a lossy one always admits a complex-cosine wave); decided on the medium, not on a
        // value, so it is one verdict at every frequency of a run.
        bool tir = force_tir || (std::imag(eta2 * mu2) == 0.0 &&
                                 ((sin_theta_real >= 0.0) ? (snell * sin_theta_real >= 1.0)
                                                          : (std::abs(n2_ratio * sin2_theta) >= 1.0)));

        Interface out;
        const bool trans_class = interaction_type == 1 || interaction_type == 2 || interaction_type == 4 || interaction_type == 5;

        // A wave held undeviated in a mass-law medium is a leak through a lumped plate (no wave inside; the
        // transmission is the medium's own term, Material::medium_gain). Every face it meets leads out of the
        // plate or into a bonded body, and what reaches the face passes, in normal flux, with the entered medium's
        // att; nothing is reflected back and the geometry is that of no bend. A closed wave medium beyond is the
        // pad. Out of a plate into a wave medium the held wave carries the transverse invariant it had in air: it
        // is refracted as from air, and closed where air would be past that medium's critical angle.
        if (undeviated && A.m > 0.0 && !(B.m > 0.0))
        {
            n2_ratio = 1.0 / (eta2 * mu2), snell = std::sqrt(std::abs(n2_ratio));
            ct2 = decaying_root(1.0 - n2_ratio * sin2_theta);
            tir = std::imag(eta2 * mu2) == 0.0 && std::abs(n2_ratio * sin2_theta) >= 1.0;
        }
        if (undeviated && A.m > 0.0 && (B.m > 0.0 || !tir))
        {
            const double ig = trans_class ? B.interface_gain(fGHz) : 0.0;
            const bool bend = !trans_class || !(B.m > 0.0);
            out.cTE = std::sqrt(ig), out.cTM = std::sqrt(ig);
            out.cos_t2 = bend ? ct2 : ct1;
            out.n2_ratio = bend ? n2_ratio : 1.0;
            out.snell = bend ? snell : 1.0;
            out.tir = tir;
            out.gain = ig;
            return out;
        }

        // tf at a bare boundary. A face with a wave behind it is the transformer of a coherent
        // sheet, per channel: the normal flux is continuous whatever the two media are, so a
        // traced solid and the same layer as a sheet are one wall. A closed face has no wave
        // behind it and takes the pad of closed_face: in front of a lossless medium the pair
        // rule on a reflectance of one, 1 - R' as a flat leak, and the leak travels undeviated.
        // A mass-law medium has no wave inside, and its side of the face is a real immittance.
        std::complex<double> coeff_TE, coeff_TM;
        double gain;
        double s_te = 0.0, s_tm = 0.0;
        std::complex<double> y_te = z1 * ct1, y_tm = std::conj(z1) * ct1;
        if (A.m > 0.0)
            y_te = std::abs(y_te), y_tm = std::abs(y_tm);
        if (!tir && trans_class)
            flux_amp(z1, z2, ct1, ct2, s_te, s_tm); // s_tm unused in scalar mode, one channel
        if (interaction_type < 3)                   // EM: types 0 (reflection), 1 (transmission), 2 (refraction)
        {
            // Fresnel reflection and transmission (TE/TM), ITU-R P.2040-1 eq. (31) and (32)
            std::complex<double> R_TE = (z1 * ct1 - z2 * ct2) / (z1 * ct1 + z2 * ct2), T_TE = (2.0 * z1 * ct1) / (z1 * ct1 + z2 * ct2);
            std::complex<double> R_TM = (z2 * ct1 - z1 * ct2) / (z2 * ct1 + z1 * ct2), T_TM = (2.0 * z1 * ct1) / (z2 * ct1 + z1 * ct2);
            if (tir)
            {
                double k_te = closed_face(A, B, fGHz, R_TE, y_te, std::real(y_te)), k_tm = closed_face(A, B, fGHz, R_TM, y_tm, std::real(y_tm), false, true);
                coeff_TE = trans_class ? k_te : R_TE, coeff_TM = trans_class ? k_tm : R_TM;
            }
            else
            {
                std::complex<double> c_te = tf_face(A, B, fGHz, R_TE, &y_te), c_tm = tf_face(A, B, fGHz, R_TM, &y_tm, true);
                coeff_TE = trans_class ? T_TE * (s_te * c_te) : R_TE, coeff_TM = trans_class ? T_TM * (s_tm * c_tm) : R_TM;
            }
            gain = 0.5 * (std::norm(coeff_TE) + std::norm(coeff_TM));
        }
        else // Scalar (acoustic): types 3 (reflection), 4 (transmission), 5 (refraction)
        {
            std::complex<double> R = (z1 * ct1 - z2 * ct2) / (z1 * ct1 + z2 * ct2), T = 1.0 + R; // pressure coefficients
            if (tir)
            {
                double k = closed_face(A, B, fGHz, R, y_te, std::real(y_te));
                coeff_TE = trans_class ? std::polar(k, std::arg(T)) : R;
            }
            else
            {
                std::complex<double> c = tf_face(A, B, fGHz, R, &y_te);
                coeff_TE = trans_class ? T * (s_te * c) : R;
            }
            coeff_TM = coeff_TE, gain = std::norm(coeff_TE);
        }

        // A mass-law body is a lumped plate whose permittivity is an impedance mapping: no tracer bends
        // into it, the wave keeps the angle it has (and Material::medium_gain forms its term there).
        if (trans_class && B.m > 0.0)
            ct2 = ct1, n2_ratio = 1.0, snell = 1.0;

        // Transmission-class interactions (1, 2, 4, 5) cross a thin interface: fold in the interface gain
        if (trans_class)
        {
            double ig = B.interface_gain(fGHz);
            gain *= ig;
            double s = std::sqrt(ig);
            coeff_TE *= s;
            coeff_TM *= s;
        }

        out.cTE = coeff_TE, out.cTM = coeff_TM;
        out.cos_t2 = ct2;
        out.n2_ratio = n2_ratio;
        out.snell = snell;
        out.tir = tir;
        out.gain = gain;
        return out;
    }

    // ---- 4 Slab and stack solver -------------------------------------------------------------------------

    // One layer of a stratified stack: a material and the normal thickness it occupies.
    struct QdLayer
    {
        const Material *mtl = nullptr;
        double thk = 0.0;
    };

    // Real amplitude factor of a mass-law layer of normal thickness thk: the loss over the path
    // thk / ct_r under a saturation cap - both the alpha column and the conductivity term are
    // dB-linear in the path, and past the length at which the layer is opaque nothing downstream
    // can move - and the mass term from the normal thickness and the cosine the wave has in air.
    // x_run is the X0 of the bonded plate crossed so far and takes this layer's.
    inline double qd_mass_amp(const Material &L, double thk, double ct_r, double mass_c, double fGHz, double &x_run)
    {
        double dist_max = 1.0e30;
        double g1m = L.medium_gain(1.0, fGHz, 1.0, 0.0, 0.0, 1.0e300); // 1 m, mass term off
        if (g1m > 0.0)
        {
            double rate = -10.0 * std::log10(g1m); // dB per meter
            if (rate > 0.0)
                dist_max = 400.0 / rate;
        }
        else
            dist_max = 0.0; // opaque within a meter
        if (dist_max < thk)
            dist_max = thk;
        double dist = (mass_c > 0.0) ? thk / mass_c : dist_max; // the held path: the wave keeps the angle it has in air
        if (!(dist < dist_max))                                 // also catches inf and NaN
            dist = dist_max;
        double mass_path = thk * mass_c; // zero at exact tangency, where the term is switched off
        double g = L.medium_gain(dist, fGHz, ct_r, mass_path, 1.0, (mass_path > 0.0) ? 0.0 : 1.0e300, thk, mass_c, x_run);
        if (mass_path > 0.0)
            x_run += L.mass_X0(thk, fGHz, mass_c);
        return std::sqrt((g < 0.0) ? 0.0 : g);
    }

    // What a closed mass-law plate passes in all: a plate whose outer faces absorb at least a and
    // whose bodies have the term tau passes the incoherent series of the face and the plate,
    // a tau / (a + tau - a tau): tau where tau << a, never more than either, and it reflects 1 - a.
    inline double qd_plate(double a, double tau)
    {
        return (a > 0.0 && tau > 0.0) ? a * tau / (a + tau - a * tau) : 0.0;
    }

    // Options of qd_stack_resolve; the defaults are a stack in one medium, reached by a refracted wave.
    struct StackOpts
    {
        bool disp_far = false;        // the stack lies on the far side (glued exit)
        double host_near = -1.0;      // displaced medium in front of the mesh plane, normal thickness, < 0 = unbounded
        double host_far = -1.0;       // the same behind it
        double mass_cos = -1.0;       // cosine the wave has in air, for a mass-law term, < 0 = |Re ci|
        double mass_bonded = 0.0;     // X0 of the bonded mass-law bodies crossed before the stack
        bool near_undeviated = false; // the wave in the near medium was not refracted in
        bool far_closed = false;      // near | far is past its critical angle
        double plate_a = -1.0;        // absorption of the face a held plate was entered through, < 0 = unknown
    };

    // The transmit and reflect ports of a stack: absolute amplitudes, TE and TM (one channel twice in scalar mode).
    struct Ports
    {
        std::complex<double> T_te, T_tm, R_te, R_tm;
    };

    // The media of a stack, j = 0 the near half-space, 1 .. n the layers, n + 1 the far half-space: eps mu, the
    // admittance and the cosine from the transverse invariant of the incidence cosine ci.
    struct StackMedia
    {
        const Material &near, &far;
        const QdLayer *layers;
        int n;
        double fGHz;
        std::complex<double> ci, s2; // the incidence cosine in the near medium, and 1 - ci^2

        const Material &med(int j) const { return (j == 0) ? near : ((j > n) ? far : *layers[j - 1].mtl); }
        std::complex<double> w(int j) const { return med(j).eta(fGHz) * med(j).mu(fGHz); }
        std::complex<double> adm(int j) const { return std::sqrt(med(j).eta(fGHz) / med(j).mu(fGHz)); }
        std::complex<double> cosine(int j) const { return (j == 0) ? ci : decaying_root(1.0 - (w(0) / w(j)) * s2); }

        // A mass-law layer in which the wave is evanescent (lossless, the invariant at or above its eps mu: the
        // verdict interact_with gives a wave that meets it) is closed on both faces. Its Fresnel reflectance does
        // not say so: next to a lossy medium it is below one although the layer carries no flux.
        bool closed(int j) const { return j >= 1 && j <= n && med(j).m > 0.0 && std::imag(w(j)) == 0.0 && std::abs((w(0) / w(j)) * s2) >= 1.0; }
    };

    // A run of mass-law layers met by a wave held undeviated in a mass-law medium (a rigid sheet on or inside a
    // rigid wall): later faces of that wall. The leak keeps the incidence cosine at every face, each face is the
    // held crossing of interact_with with the entered medium's att, and each layer adds its mass term, as the
    // same bodies do as traced solids. Left into a medium without the mass column, the plate passes in all what
    // a plate of its two outer faces passes.
    inline void stack_held_plate(const StackMedia &sm, bool is_scalar, double mc, const StackOpts &opt, Ports &out)
    {
        const int it = is_scalar ? 4 : 1;
        std::complex<double> t_te(1.0, 0.0), t_tm(1.0, 0.0);
        double x_run = opt.mass_bonded, amp = 1.0;
        for (int j = 1; j <= sm.n + 1; ++j)
        {
            const Interface f = interact_with(sm.med(j - 1), sm.med(j), it, 0.0, sm.fGHz, &sm.ci, true);
            t_te *= f.cTE, t_tm *= f.cTM;
            if (j <= sm.n)
                amp *= qd_mass_amp(sm.med(j), sm.layers[j - 1].thk, std::abs(std::real(sm.ci)), mc, sm.fGHz, x_run);
        }
        if (!(sm.far.m > 0.0))
        {
            const Material &L = *sm.layers[sm.n - 1].mtl;
            double a_out = interact_with(sm.far, L, it, 0.0, sm.fGHz, &sm.ci).gain / L.interface_gain(sm.fGHz);
            double a_in = (opt.plate_a >= 0.0) ? opt.plate_a : a_out, tau = std::pow(10.0, -0.1 * mass_TL(opt.mass_bonded, mc, sm.fGHz, sm.near.side)) * amp * amp;
            if (a_in > 0.0 && tau > 0.0)
                amp *= std::sqrt(qd_plate(std::min(a_in, a_out), tau) / (a_in * tau));
        }
        // The far medium's interface gain is owed only where it is newly entered
        if (!sm.far.same_as(Material()) && sm.far.same_as(sm.near))
            amp /= std::sqrt(sm.far.interface_gain(sm.fGHz));
        const Interface r = interact_with(sm.near, *sm.layers[0].mtl, is_scalar ? 3 : 0, 0.0, sm.fGHz, &sm.ci, true);
        out.T_te = amp * t_te, out.T_tm = amp * t_tm;
        out.R_te = r.cTE, out.R_tm = r.cTM; // a mass-law layer displaces nothing: no reference plane to move
    }

    // A mass-law sheet whose far side is closed has no wave inside and none behind it. What crosses is the leak
    // of its two faces, written as the two crossings a traced solid of its material would produce, each at the
    // cosine the wave has there: entered as a leak it kept the incidence cosine and meets the far face held;
    // entered by an ordinary crossing it is at the Snell cosine, where the far face is closed whenever the outer
    // pair is (the verdict is forced; in scalar mode the phase is the raw crossing's).
    inline void stack_leak_chain(const StackMedia &sm, bool is_scalar, double mc, const StackOpts &opt, std::complex<double> leak[2])
    {
        const int it = is_scalar ? 4 : 1;
        const Material &S = *sm.layers[0].mtl;
        const std::complex<double> cS = sm.cosine(1);
        const Interface fa = interact_with(sm.near, S, it, 0.0, sm.fGHz, &sm.ci, opt.near_undeviated);
        const bool leak_first = fa.tir;
        const Interface fb = interact_with(S, sm.far, it, 0.0, sm.fGHz, leak_first ? &sm.ci : &cS, true, !(sm.far.m > 0.0));
        const std::complex<double> a_te = fa.cTE, a_tm = fa.cTM, b_te = fb.cTE, b_tm = fb.cTM;
        double x_run = opt.mass_bonded, aI = qd_mass_amp(S, sm.layers[0].thk, std::abs(std::real(cS)), mc, sm.fGHz, x_run);
        leak[0] = aI * a_te * b_te, leak[1] = aI * a_tm * b_tm;
        if (!(sm.near.m > 0.0) && !(sm.far.m > 0.0)) // the sheet is the whole plate: the plate rule of its two faces
        {
            const double tau = aI * aI, gS = S.interface_gain(sm.fGHz), gF = sm.far.interface_gain(sm.fGHz); // the gains interact_with folded in
            const double a[2] = {std::norm(a_te) / gS, std::norm(a_tm) / gS}, b[2] = {std::norm(b_te) / gF, std::norm(b_tm) / gF};
            for (int pol = 0; pol < 2; ++pol)
                if (a[pol] > 0.0 && b[pol] > 0.0 && tau > 0.0)
                    leak[pol] *= std::sqrt(qd_plate(std::min(a[pol], b[pol]), tau) / (a[pol] * b[pol] * tau));
        }
    }

    // Per channel, what the recursion carries across plates and closed layers.
    struct PlateRun
    {
        double fl_in = 1.0, fl_out = 1.0;                   // normal flux of the media a leak entered a closed layer from, and left it into
        double a_run = -1.0, tau_run = 1.0, a_entry = -1.0; // absorption of a plate's first face, its bodies' term, the last entry face
    };

    // Boundary j | j+1 of a stack for channel pol (0 TE or scalar, 1 TM): the two-port r, t, tb in raw Fresnel
    // amplitudes with tf folded in by the pair that forms it; a pair without tf stays raw unless nothing
    // propagates behind it. A closed layer (no wave inside) carries a leak as a flux: from a coherent side its
    // face is the pad of closed_face, from a mass-law side it passes everything. z_n is the near admittance.
    inline void stack_boundary(const StackMedia &sm, int j, int pol, bool is_scalar, const StackOpts &opt, const std::complex<double> &z_n,
                               PlateRun &run, std::complex<double> &r, std::complex<double> &t, std::complex<double> &tb)
    {
        const double fGHz = sm.fGHz;
        const std::complex<double> &ci = sm.ci;

        // The last face of a plate, out of a mass-law layer into a side without the mass column, and what it
        // passes besides what reached it: the plate rule over what the entry face passed
        auto plate_out = [&sm, &opt](int k)
        { return (k >= 1 || opt.near_undeviated) && k <= sm.n && sm.med(k).m > 0.0 && !(sm.med(k + 1).m > 0.0); };
        auto plate_fix = [&run](double a_out)
        { return (run.a_run > 0.0 && run.tau_run > 0.0) ? std::sqrt(qd_plate(std::min(run.a_run, a_out), run.tau_run) / (run.a_run * run.tau_run)) : 1.0; };
        // The transformer of an open face; a peaked face is evaluated from the side that is not the plate
        auto open_face = [fGHz, pol](const Material &A, const Material &B, std::complex<double> &rr, const std::complex<double> &ya, const std::complex<double> &yb)
        {
            std::complex<double> rb = -rr;
            return (A.peaked() && !B.peaked()) ? tf_face(B, A, fGHz, rb, &yb, pol == 1) : tf_face(A, B, fGHz, rr, &ya, pol == 1);
        };

        const Material &A = sm.med(j), &B = sm.med(j + 1);
        std::complex<double> za = sm.adm(j), zb = sm.adm(j + 1), ca = sm.cosine(j), cb = sm.cosine(j + 1);
        if (pol == 0)
        {
            std::complex<double> a = za * ca, b = zb * cb;
            r = (a - b) / (a + b), t = (2.0 * a) / (a + b), tb = (2.0 * b) / (b + a);
        }
        else
        {
            r = (zb * ca - za * cb) / (zb * ca + za * cb);
            t = (2.0 * za * ca) / (zb * ca + za * cb);
            tb = (2.0 * zb * cb) / (za * cb + zb * ca);
        }
        // The far face of a coherent sheet on a closed far side: the pad, over the
        // incident flux, real in EM and with the phase of the raw crossing in scalar mode
        if (opt.far_closed && j == sm.n && !(A.m > 0.0))
        {
            double k = closed_face(A, B, fGHz, r, (pol == 0) ? za * ca : std::conj(za) * ca,
                                   std::real((pol == 0) ? z_n * ci : std::conj(z_n) * ci), false, pol == 1);
            t = std::polar(k, is_scalar ? std::arg(t) : 0.0);
            return;
        }
        const bool cA = sm.closed(j), cB = sm.closed(j + 1);
        const bool closed = cA != cB;          // a closed layer against a side that carries flux
        if (!closed && A.m > 0.0 && B.m > 0.0) // inside one plate: what reaches it passes, nothing is reflected
        {
            std::complex<double> ya = (pol == 0) ? za * ca : std::conj(za) * ca, yb = (pol == 0) ? zb * cb : std::conj(zb) * cb;
            double k = cA ? 1.0 : std::sqrt(std::real(ya) / std::real(yb)); // two closed layers carry the leak as a flux
            r = 0.0, t = std::polar(k, std::arg(t)), tb = std::polar(1.0 / k, std::arg(tb));
            return;
        }
        if (!closed && (plate_out(j) || (j + 1 <= sm.n && B.m > 0.0 && !(A.m > 0.0)))) // an open outer face of a plate
        {
            std::complex<double> ya = (pol == 0) ? za * ca : std::conj(za) * ca, yb = (pol == 0) ? zb * cb : std::conj(zb) * cb, c = open_face(A, B, r, ya, yb);
            t *= c, tb *= c;
            if (!plate_out(j)) // the first face: what it passes into the plate
            {
                run.a_entry = std::norm(t) * std::real(yb) / std::real(ya);
                return;
            }
            // the last face: what reached it passes, put right by the plate rule
            double k = (std::real(yb) > 0.0) ? std::sqrt(std::real(ya) / std::real(yb)) : 0.0;
            t = std::polar(k * plate_fix(std::norm(tb) * std::real(ya) / std::real(yb)), std::arg(t));
            return;
        }
        if (A.tf(fGHz) == 0.0 && B.tf(fGHz) == 0.0 && !closed)
            return;
        if (!closed) // an open face, of a coherent or a mass-law layer
        {
            std::complex<double> c = open_face(A, B, r, (pol == 0) ? za * ca : std::conj(za) * ca, (pol == 0) ? zb * cb : std::conj(zb) * cb);
            t *= c, tb *= c;
            return;
        }
        // A closed layer seen from a side that carries a wave is the pad of a closed face: on
        // the way in the flux it passes, on the way out everything that reached it, the
        // amplitude that flux launches. Seen from the reflect side the face keeps its pad.
        const Material &V = cB ? A : B;
        if (!(V.m > 0.0))
        {
            std::complex<double> yw = (pol == 0) ? (cB ? za * ca : zb * cb) : (cB ? std::conj(za) * ca : std::conj(zb) * cb), rw = cB ? r : -r, rp = rw;
            double k = closed_face(V, cB ? B : A, fGHz, rw, yw, 1.0, false, pol == 1), kp = closed_face(V, cB ? B : A, fGHz, rp, yw, 1.0, true, pol == 1);
            double a_face = (kp > 0.0) ? (k / kp) * (k / kp) : 0.0;
            if (cB)
                run.a_entry = a_face;
            else // the plate's last face: it passes in all what a plate of its two faces passes
                k = kp * plate_fix(a_face);
            r = cB ? rw : -rw, t = std::polar(cB ? k : k / std::abs(yw), std::arg(t));
            return;
        }
        // Against a mass-law side the face is inside one plate: it passes everything and reflects nothing. The
        // segments run in raw amplitudes converted once at the outer pair, and a closed layer has nothing to
        // convert through, so the flux of the side that carries one is collected here and put right below.
        r = 0.0, t = std::polar(1.0, std::arg(t)), tb = std::polar(1.0, std::arg(tb));
        int o = cA ? j + 1 : j; // the side that carries flux
        double g = (pol == 0) ? std::real(sm.adm(o) * sm.cosine(o)) : std::real(std::conj(sm.adm(o)) * sm.cosine(o));
        (cB ? run.fl_in : run.fl_out) *= g;
    }

    // One-way factor of coherent layer j: exp(-j k0 n thk c), index and cosine complex, times the alpha column
    // over the path thk / |Re c|. A wave that does not advance in the layer has no path, and is opaque to it.
    inline std::complex<double> stack_layer_phase(const StackMedia &sm, int j)
    {
        const Material &L = sm.med(j);
        const double fGHz = sm.fGHz;
        std::complex<double> c_L = sm.cosine(j);
        double thk = sm.layers[j - 1].thk, a_dB = L.alpha * std::pow(fGHz / L.fRef, L.alphaB);
        double a_Np = (a_dB > 0.0 && thk > 0.0) ? 0.05 * std::log(10.0) * a_dB * thk / std::abs(std::real(c_L)) : 0.0;
        return std::exp(std::complex<double>(0.0, -1.0) * k0(fGHz) * std::sqrt(sm.w(j)) * c_L * thk - a_Np);
    }

    // Composite of an ordered list of layers between two half-spaces: one sheet, or a stack of two. A sheet
    // has no world-space thickness, so the tracer cannot follow its internal series: it is summed here, in
    // one event. Called by both ray functions. Returns absolute transmit and reflect coefficients; false
    // where it declines (a singular denominator).
    //
    // Per medium z = sqrt(eps/mu), n = sqrt(eps mu); every cosine follows from the transverse invariant
    // eps mu (1 - c^2), fixed by the incidence cosine ci in the near medium (complex where that is a traced
    // solid).
    //
    // Each boundary a | b is a two-port (r, t_ab, t_ba, -r) in raw Fresnel amplitudes, each coherent layer the
    // one-way factor phi, and the product is a recursion from the far end:
    //     G_j = r_j + t_j tb_j G_{j+1} phi^2 / (1 + r_j G_{j+1} phi^2)
    //     T_j = t_j phi T_{j+1} / (1 + r_j G_{j+1} phi^2)
    // (the characteristic matrix in another basis; no inverse of a closed boundary or an opaque layer; the
    // Airy pair at one layer). Amplitudes propagate; the conversion to normal flux is applied once, at the
    // outer pair, on the transmit port. Nothing is clamped.
    //
    //   tf         per boundary: the transformer (tf_face) at an open coherent face, the pad (closed_face)
    //              where nothing propagates behind, the pair rule on the reflectance at an open mass-law face.
    //   att        once per layer entered on the transmit port, and the far medium's where it is newly
    //              entered; on the reflect port once per layer entered and left (the second recursion Ge).
    //   alpha      dB per metre over the path thk / |Re c|, a real factor on phi.
    //   m > 0      a lumped plate: it ends the coherent segment in front as a half-space of its own
    //              admittance, adds one real amplitude factor and starts the next segment; nothing returns
    //              across it and it displaces nothing. Mass-law layers with no coherent layer between are one
    //              plate.
    //   far_closed near | far is past its critical angle (only a sheet glued to a solid): the far face of a
    //              coherent sheet is the pad, and a mass-law sheet passes the two crossings a traced solid of
    //              its material would.
    inline bool qd_stack_resolve(const Material &mat_near, const QdLayer *layers, int n_layers, const Material &mat_far,
                                 std::complex<double> ci, double fGHz, bool is_scalar, const StackOpts &opt, Ports &out)
    {
        const double mc = (opt.mass_cos >= 0.0) ? opt.mass_cos : std::abs(std::real(ci));
        const StackMedia sm{mat_near, mat_far, layers, n_layers, fGHz, ci, 1.0 - ci * ci};

        if (opt.near_undeviated && mat_near.m > 0.0)
        {
            bool all_mass = true;
            for (int j = 1; j <= n_layers; ++j)
                all_mass = all_mass && sm.med(j).m > 0.0;
            if (all_mass)
            {
                stack_held_plate(sm, is_scalar, mc, opt, out);
                return true;
            }
        }

        const std::complex<double> z_n = sm.adm(0), z_f = sm.adm(n_layers + 1), c_far = sm.cosine(n_layers + 1);
        const bool leak_chain = opt.far_closed && n_layers == 1 && layers[0].mtl->m > 0.0;
        std::complex<double> leak[2];
        if (leak_chain)
            stack_leak_chain(sm, is_scalar, mc, opt, leak);

        std::complex<double> R_out[2], T_out[2];
        for (int pol = 0; pol < 2; ++pol)
        {
            PlateRun run;
            std::complex<double> Tt(1.0, 0.0), Rt(0.0, 0.0);
            int p = 0;
            double x_run = opt.mass_bonded; // X0 of the plate crossed so far
            for (int q = 1; q <= n_layers + 1; ++q)
            {
                bool mass = (q <= n_layers) && sm.med(q).m > 0.0;
                if (q <= n_layers && !mass)
                    continue;
                if (q - p > 1) // a coherent layer separates what follows from the plate
                    x_run = 0.0;

                // Coherent segment between media p and q, from its far boundary inwards. G is
                // the reflection the series runs on. Ge is the one the reflect port hands
                // out: what entered a layer and came back paid that layer's interface gain,
                // once, as the transmit port pays it once; the series itself pays nothing.
                std::complex<double> G, Ge, T, r, t, tb;
                stack_boundary(sm, q - 1, pol, is_scalar, opt, z_n, run, G, T, tb);
                Ge = G;
                if (mass && (q - p > 1 || p == 0 || !(sm.med(p).m > 0.0)))
                    run.a_run = run.a_entry, run.tau_run = 1.0;
                for (int j = q - 2; j >= p; --j)
                {
                    std::complex<double> phi = stack_layer_phase(sm, j + 1), p2 = phi * phi;
                    stack_boundary(sm, j, pol, is_scalar, opt, z_n, run, r, t, tb);
                    std::complex<double> den = 1.0 + r * G * p2;
                    if (!(std::abs(den) > 0.0))
                        return false;
                    T = t * phi * T / den;
                    Ge = r + std::sqrt(sm.med(j + 1).interface_gain(fGHz)) * t * tb * Ge * p2 / den;
                    G = r + t * tb * G * p2 / den;
                }
                Tt *= T;
                if (p == 0)
                    Rt = Ge;
                if (mass)
                {
                    double amp = qd_mass_amp(sm.med(q), layers[q - 1].thk, std::abs(std::real(sm.cosine(q))), mc, fGHz, x_run);
                    run.tau_run *= amp * amp;
                    Tt *= amp;
                }
                p = q;
            }
            if (run.fl_in != run.fl_out) // one medium on both sides of the closed layers: exactly one
                Tt *= (run.fl_in > 0.0 && run.fl_out > 0.0) ? std::sqrt(run.fl_in / run.fl_out) : 0.0;
            R_out[pol] = Rt, T_out[pol] = leak_chain ? leak[pol] : Tt;
            if (is_scalar) // one channel, the TE / admittance form, as everywhere else in this file
            {
                R_out[1] = R_out[0], T_out[1] = T_out[0];
                break;
            }
        }
        if (leak_chain) // interact_with wrote the gains and the flux of its two crossings
        {
            out.T_te = T_out[0], out.T_tm = T_out[1];
            out.R_te = R_out[0], out.R_tm = R_out[1];
            return true;
        }

        // The one conversion to per-channel normal flux, at the outer pair, transmit port only.
        // With near == far it is exactly 1, and a closed far side carries no flux to convert to:
        // what its pad passes is already over the incident flux.
        double fx_te = 1.0, fx_tm = 1.0;
        if (!mat_far.same_as(mat_near) && !opt.far_closed)
        {
            flux_amp(z_n, z_f, ci, c_far, fx_te, fx_tm);
            if (is_scalar)
                fx_tm = fx_te;
        }

        // Lumped interface gains: one per layer entered, plus the far medium's when the wave enters
        // a medium it was not already in.
        bool far_is_new = !mat_far.same_as(Material()) && !mat_far.same_as(mat_near);
        double ig2 = far_is_new ? mat_far.interface_gain(fGHz) : 1.0;
        for (int i = 0; i < n_layers; ++i)
            ig2 *= layers[i].mtl->interface_gain(fGHz);
        double ig = std::sqrt((ig2 < 0.0) ? 0.0 : ig2);

        // The displaced medium: a sheet stands for a layer that occupies its thickness of the medium it sits
        // in, which the tracer walked; that length is taken back out here. Half on either side of the mesh plane
        // where one medium is on both sides, all in front for a glued entry, all behind for a glued exit; each
        // part at most the host on its side (taking a lossy medium out is a division). A mass-law layer
        // displaces nothing. On a closed far side the length was walked at the incidence cosine.
        double t_tot = 0.0;
        for (int i = 0; i < n_layers; ++i)
            if (!(layers[i].mtl->m > 0.0))
                t_tot += layers[i].thk;
        double u_near = mat_far.same_as(mat_near) ? 0.5 * t_tot : (opt.disp_far ? 0.0 : t_tot), u_far = t_tot - u_near;
        if (opt.host_near >= 0.0 && opt.host_near < u_near)
            u_near = opt.host_near;
        if (opt.host_far >= 0.0 && opt.host_far < u_far)
            u_far = opt.host_far;
        const int d = opt.disp_far ? n_layers + 1 : 0;
        const std::complex<double> c_disp = opt.far_closed ? ci : sm.cosine(d), w_disp = sm.w(d);
        auto undisplace = [fGHz, &c_disp, &w_disp](double t) -> std::complex<double>
        {
            std::complex<double> phi = std::exp(std::complex<double>(0.0, -1.0) * k0(fGHz) * std::sqrt(w_disp) * t * c_disp);
            return (std::abs(phi) > 0.0) ? std::complex<double>(1.0, 0.0) / phi : std::complex<double>(1.0, 0.0);
        };

        // The transmit port takes the displaced medium out over both parts. The reflect port is
        // handed to a ray that turns around at the mesh plane, so it owes the round trip through
        // the part in front of that plane.
        std::complex<double> undisp = undisplace(u_near + u_far), front = undisplace(2.0 * u_near);

        out.T_te = ig * fx_te * T_out[0] * undisp;
        out.T_tm = ig * fx_tm * T_out[1] * undisp;
        out.R_te = front * R_out[0];
        out.R_tm = front * R_out[1];
        return true;
    }

    // The ports of one sheet, or of an ordered stack (sheet2 the second layer), between near and far.
    inline bool sheet_ports(const Material &near, const Material &sheet, const Material *sheet2, const Material &far,
                            std::complex<double> ci, double fGHz, bool is_scalar, const StackOpts &opt, Ports &out)
    {
        QdLayer lay[2];
        lay[0].mtl = &sheet, lay[0].thk = sheet.thk;
        if (sheet2)
            lay[1].mtl = sheet2, lay[1].thk = sheet2->thk;
        return qd_stack_resolve(near, lay, sheet2 ? 2 : 1, far, ci, fGHz, is_scalar, opt, out);
    }

    // Options of qd_slab_resolve.
    struct SlabOpts
    {
        std::complex<double> ct_slab_exact = 0.0; // the recovered complex in-slab cosine, 0 = none
        const Material *back_near = nullptr;      // what lies behind a sheet named as the near medium
        const Material *back_far = nullptr;       // what lies behind a sheet named as the far medium
        bool verdict_given = false;               // the caller decided to sum; eps is not read
        const Material *near2 = nullptr;          // second layer of an ordered stack named as the near medium
        const Material *far2 = nullptr;           // the same for the far medium
    };

    // Reflection r_te, r_tm at slab | adj, seen from the slab at the in-slab cosine cs. A sheet neighbour is a
    // layer, not a half-space (an ordered stack both its layers, adj2 the second), resolved by qd_stack_resolve
    // with backing behind it where the caller knows it and air otherwise, within the host thickness host_thk; a
    // mass-law sheet keeps its half-space face. A bare face takes tf as interact_with does: the pad where it is
    // closed, the transformer where it is open.
    inline void slab_face_reflection(const Material &slab, const Material &adj, std::complex<double> cs, double fGHz, bool is_scalar,
                                     double host_thk, const Material *backing, const Material *adj2,
                                     std::complex<double> &r_te, std::complex<double> &r_tm)
    {
        std::complex<double> eta_s_if = slab.eta(fGHz);
        std::complex<double> mu_s = slab.mu(fGHz);
        std::complex<double> sin2 = 1.0 - cs * cs;
        std::complex<double> eta_a_if = adj.eta(fGHz);
        std::complex<double> mu_a = adj.mu(fGHz);
        std::complex<double> z1 = std::sqrt(eta_s_if / mu_s);
        std::complex<double> z2 = std::sqrt(eta_a_if / mu_a);
        std::complex<double> rat = (eta_s_if * mu_s) / (eta_a_if * mu_a);

        std::complex<double> cos_t2 = decaying_root(1.0 - rat * sin2);

        r_te = (z1 * cs - z2 * cos_t2) / (z1 * cs + z2 * cos_t2);
        r_tm = is_scalar ? r_te : (z2 * cs - z1 * cos_t2) / (z2 * cs + z1 * cos_t2);

        if (adj.thk > 0.0 && !(adj.m > 0.0))
        {
            // Seen from the slab a lining lies in the air beyond it; a sheet inside the slab
            // has the slab on both sides and the solver centres it, within the leg.
            Material air_backing;
            StackOpts opt;
            opt.disp_far = true, opt.host_near = host_thk;
            Ports ports;
            if (sheet_ports(slab, adj, adj2, backing ? *backing : air_backing, cs, fGHz, is_scalar, opt, ports))
            {
                r_te = ports.R_te, r_tm = is_scalar ? ports.R_te : ports.R_tm;
                return;
            }
        }
        if (std::imag(eta_a_if * mu_a) == 0.0 && std::abs(rat * sin2) >= 1.0) // closed, as interact_with judges it
            closed_face(slab, adj, fGHz, r_te, z1 * cs, 1.0), closed_face(slab, adj, fGHz, r_tm, is_scalar ? z1 * cs : std::conj(z1) * cs, 1.0, false, !is_scalar);
        else
        {
            std::complex<double> y_te = z1 * cs, y_tm = is_scalar ? y_te : std::conj(z1) * cs;
            tf_face(slab, adj, fGHz, r_te, &y_te), tf_face(slab, adj, fGHz, r_tm, &y_tm, !is_scalar);
        }
    }

    // Series factor of a traced solid near | slab | far: S = 1 / (1 - rn rf phi^2) per channel, applied on
    // top of the face coefficient ray_state_update already wrote; symmetric in near and far. dist_refract
    // and dist_geo are the traced path lengths through the slab. Both faces are evaluated from the slab
    // side, at the recovered complex cosine ct_slab_exact where there is one and at the traced real angle
    // otherwise, as interact_with evaluates the face the tracer meets: a summed and a followed series are
    // one wall. Returns false where the series is left to the tracer (rho < eps, or a mass-law slab).
    inline bool qd_slab_resolve(const Material &mat_near, const Material &mat_slab, const Material &mat_far,
                                double dist_refract, double dist_geo, double abs_cos_theta_t, double abs_cos_theta,
                                double fGHz, bool is_scalar, double eps, const SlabOpts &opt,
                                std::complex<double> &S_te, std::complex<double> &S_tm)
    {
        // A mass-law material carries a lumped transmission surrogate: its eta and mu are
        // calibrated for surface impedance only, so the in-slab index is not physical and the
        // internal series must not be resummed. The tracer follows the bounces.
        if (mat_slab.m > 0.0)
            return false;

        std::complex<double> n_cx = std::sqrt(mat_slab.eta(fGHz) * mat_slab.mu(fGHz));

        bool have_exact_ct = std::abs(opt.ct_slab_exact) > 0.0;
        std::complex<double> cs = have_exact_ct ? opt.ct_slab_exact : std::complex<double>(abs_cos_theta_t, 0.0);

        const double host_thk = dist_refract * abs_cos_theta_t;
        std::complex<double> rn_te, rn_tm, rf_te, rf_tm;
        slab_face_reflection(mat_slab, mat_near, cs, fGHz, is_scalar, host_thk, opt.back_near,
                             (opt.near2 && !(opt.near2->m > 0.0)) ? opt.near2 : nullptr, rn_te, rn_tm);
        slab_face_reflection(mat_slab, mat_far, cs, fGHz, is_scalar, host_thk, opt.back_far,
                             (opt.far2 && !(opt.far2->m > 0.0)) ? opt.far2 : nullptr, rf_te, rf_tm);

        // One-way propagation factor. With the recovered cosine it is exp(-j k0 n thk c), index
        // and cosine complex, over the normal thickness the traced path covers, with only the
        // alpha column drawn from the material. Without one it is the hybrid: the magnitude from
        // medium_gain over the traced path, the phase from the real index and the real cosine.
        std::complex<double> phi_1;
        if (have_exact_ct)
        {
            double ex = mat_slab.excess_gain(dist_refract, fGHz); // alpha only, no conductivity
            phi_1 = std::exp(std::complex<double>(0.0, -1.0) * k0(fGHz) * n_cx * (dist_refract * abs_cos_theta_t) * cs) * std::sqrt((ex < 0.0) ? 0.0 : ex);
        }
        else
        {
            double gL = mat_slab.medium_gain(dist_refract, fGHz, abs_cos_theta_t, dist_geo, abs_cos_theta);
            phi_1 = std::polar(std::sqrt((gL < 0.0) ? 0.0 : gL), -k0(fGHz) * std::real(n_cx) * dist_refract * abs_cos_theta_t * abs_cos_theta_t);
        }
        std::complex<double> phi2 = phi_1 * phi_1;

        // Survival gate on the stronger polarization: a weak series is left to the tracer
        double g2L = mat_slab.medium_gain(2.0 * dist_refract, fGHz, abs_cos_theta_t, 2.0 * dist_geo, abs_cos_theta);
        double rr_te = std::norm(rn_te * rf_te), rr_tm = std::norm(rn_tm * rf_tm);
        double rho = std::sqrt(((rr_te > rr_tm) ? rr_te : rr_tm) * ((g2L < 0.0) ? 0.0 : g2L));
        if (!opt.verdict_given && rho < eps)
            return false;

        std::complex<double> S_te_v = 1.0 / (1.0 - rn_te * rf_te * phi2), S_tm_v = 1.0 / (1.0 - rn_tm * rf_tm * phi2);
        S_te = S_te_v;
        S_tm = is_scalar ? S_te_v : S_tm_v;
        return true;
    }

    // ---- 5 Sheets (both ray functions) --------------------------------------------------------------------

    // Sheet cases. ray_mesh_interact tells all of them apart from the faces of a hit; ray_state_update re-derives
    // them from the hit code, where a duplicate and a corner are isolated sheets.
    enum class SheetCase
    {
        none,        // no sheet involved
        isolated,    // a sheet with one medium on both sides
        glued_entry, // host | sheet | solid
        glued_exit,  // solid | sheet | air
        stack,       // an ordered stack of two different sheets
        duplicate,   // two coincident sheets of one material, collapsed to one
        corner       // a sheet at a corner: the SBS is dropped
    };

    // The angle factor of a mass term behind a near medium the wave was refracted into: the far cosine of
    // near | air (at the recovered cosine ct_in where there is one), at most one. The plate is driven at the
    // trace wavenumber and calibrated in air, so a wall pays the same from both sides. Where the invariant has
    // no direction in air, fallback.
    inline double mass_cos_in_air(const Material &near, int interaction_type, double theta, double fGHz,
                                  const std::complex<double> *ct_in, double fallback)
    {
        const Interface air = interact_with(near, Material(), interaction_type, theta, fGHz, ct_in, false);
        const std::complex<double> cos_air = air.cos_t2;
        if (!air.tir && std::real(cos_air) > 0.0)
            return (std::real(cos_air) > 1.0) ? 1.0 : std::real(cos_air);
        return fallback;
    }

    // ---- 6 Face-hit geometry (ray_mesh_interact) ----------------------------------------------------------

    // Normalizes (x, y, z) with the reciprocal of its length above 2e-7, else sets it to (1, 0, 0); returns
    // the length.
    inline double normalize_fast(double &x, double &y, double &z)
    {
        double len = std::sqrt(x * x + y * y + z * z);
        if (len > 2e-7)
        {
            double scl = 1.0 / len;
            x *= scl, y *= scl, z *= scl;
        }
        else
            x = 1.0, y = 0.0, z = 0.0;
        return len;
    }

    // One ray's outputs: element i of a column-major [n, k] matrix (set1, set3, set_row) or column i of a
    // [k, n] matrix (set_col). A null pointer is skipped.
    template <typename dtype>
    struct RayWriter
    {
        size_t i, n;
        void set1(dtype *p, double v) const
        {
            if (p)
                p[i] = (dtype)v;
        }
        void set3(dtype *p, double x, double y, double z) const
        {
            if (p)
                p[i] = (dtype)x, p[i + n] = (dtype)y, p[i + 2 * n] = (dtype)z;
        }
        void set_row(dtype *p, const double *v, size_t L) const
        {
            if (p)
                for (size_t l = 0; l < L; ++l)
                    p[i + l * n] = (dtype)v[l];
        }
        void set_col(dtype *p, const double *v, size_t L) const
        {
            if (p)
                for (size_t l = 0; l < L; ++l)
                    p[i * L + l] = (dtype)v[l];
        }
    };

    // The output pointers of ray_mesh_interact (null: not requested).
    template <typename dtype>
    struct RmiOut
    {
        dtype *origN, *destN, *fbsN, *sbsN, *gainN, *xprmatN, *trivecN, *tridirN;
        dtype *fbs_angleN, *thicknessN, *edge_lengthN, *normal_vecN, *path_dirN;
        uint8_t *out_typeN;
        unsigned *ray_indN;
        size_t nXPR;
    };

    // First vertex P and unit normal N of mesh face i_face (1-based); the vertex order sets the normal.
    template <typename dtype>
    inline void face_plane(const dtype *p_mesh, size_t n_mesh, size_t i_face,
                           double &Px, double &Py, double &Pz, double &Nx, double &Ny, double &Nz)
    {
        Px = (double)p_mesh[i_face - 1];
        Py = (double)p_mesh[i_face - 1 + n_mesh];
        Pz = (double)p_mesh[i_face - 1 + 2 * n_mesh];
        double E1x = (double)p_mesh[i_face - 1 + 3 * n_mesh] - Px,
               E1y = (double)p_mesh[i_face - 1 + 4 * n_mesh] - Py,
               E1z = (double)p_mesh[i_face - 1 + 5 * n_mesh] - Pz;
        double E2x = (double)p_mesh[i_face - 1 + 6 * n_mesh] - Px,
               E2y = (double)p_mesh[i_face - 1 + 7 * n_mesh] - Py,
               E2z = (double)p_mesh[i_face - 1 + 8 * n_mesh] - Pz;
        Nx = E1y * E2z - E1z * E2y, Ny = E1z * E2x - E1x * E2z, Nz = E1x * E2y - E1y * E2x;
        normalize_fast(Nx, Ny, Nz);
    }

    // Unit direction of vertex ray k of a tube: from [az el] (tube 1, spherical) or a Cartesian triple (tube 2).
    template <typename dtype>
    inline void tube_vertex_dir(const dtype *p_tridir, int tube, size_t iRx, size_t n_ray, size_t k, double &Vx, double &Vy, double &Vz)
    {
        if (tube == 1)
        {
            double az = (double)p_tridir[iRx + 2 * k * n_ray];
            double el = (double)p_tridir[iRx + (2 * k + 1) * n_ray];
            double c = std::cos(el);
            Vx = std::cos(az) * c, Vy = std::sin(az) * c, Vz = std::sin(el);
        }
        else
        {
            size_t o = iRx + 3 * k * n_ray;
            Vx = (double)p_tridir[o], Vy = (double)p_tridir[o + n_ray], Vz = (double)p_tridir[o + 2 * n_ray];
            normalize_fast(Vx, Vy, Vz);
        }
    }

    // Longest edge of the tube triangle tv (three vertices, x y z each), or e0 where that is longer (INFINITY:
    // a vertex without a usable hit).
    inline double tube_edge_length(const double tv[9], double e0)
    {
        double e = e0, dx, dy, dz, l;
        dx = tv[3] - tv[0], dy = tv[4] - tv[1], dz = tv[5] - tv[2];
        l = dx * dx + dy * dy + dz * dz;
        e = (l > e) ? l : e;
        dx = tv[6] - tv[0], dy = tv[7] - tv[1], dz = tv[8] - tv[2];
        l = dx * dx + dy * dy + dz * dz;
        e = (l > e) ? l : e;
        dx = tv[6] - tv[3], dy = tv[7] - tv[4], dz = tv[8] - tv[5];
        l = dx * dx + dy * dy + dz * dz;
        e = (l > e) ? l : e;
        return std::sqrt(e);
    }

    // A ray that hits nothing: a transparent pass to its destination D (unit direction OD from O), with the
    // tube carried to the wavefront plane through D.
    template <typename dtype>
    inline void rmi_no_hit(const RmiOut<dtype> &out, const RayWriter<dtype> &w, int tube, const dtype *p_trivec, const dtype *p_tridir,
                           size_t iRx, size_t n_ray, const double O[3], const double D[3], const double OD[3])
    {
        const double Dx = D[0], Dy = D[1], Dz = D[2], ODx = OD[0], ODy = OD[1], ODz = OD[2];
        w.set3(out.origN, Dx, Dy, Dz);
        w.set3(out.destN, Dx + colocation_dist * ODx, Dy + colocation_dist * ODy, Dz + colocation_dist * ODz);
        w.set3(out.fbsN, Dx, Dy, Dz);
        w.set3(out.sbsN, Dx, Dy, Dz);
        w.set1(out.gainN, 1.0);

        const double xprmat[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        w.set_col(out.xprmatN, xprmat, out.nXPR);

        if (tube)
        {
            double tv[9] = {0.0}, td[9] = {0.0};
            double edge_len = 0.0;
            for (size_t k = 0; k < 3; ++k)
            {
                size_t io = iRx + 3 * k * n_ray;
                double Tx = O[0] + (double)p_trivec[io];
                double Ty = O[1] + (double)p_trivec[io + n_ray];
                double Tz = O[2] + (double)p_trivec[io + 2 * n_ray];

                // Vertex direction, unchanged by a transparent pass
                double Vx, Vy, Vz;
                tube_vertex_dir(p_tridir, tube, iRx, n_ray, k, Vx, Vy, Vz);
                if (tube == 1)
                    td[2 * k] = (double)p_tridir[iRx + 2 * k * n_ray], td[2 * k + 1] = (double)p_tridir[iRx + (2 * k + 1) * n_ray];
                else
                    td[3 * k] = Vx, td[3 * k + 1] = Vy, td[3 * k + 2] = Vz;

                // Advance the vertex ray to the wavefront plane through D with normal OD
                double denom = Vx * ODx + Vy * ODy + Vz * ODz;
                double Wx, Wy, Wz;
                if (std::abs(denom) < 1e-6) // vertex parallel to the wavefront plane
                {
                    edge_len = INFINITY;
                    Wx = Tx, Wy = Ty, Wz = Tz;
                }
                else
                {
                    double d = ((Dx - Tx) * ODx + (Dy - Ty) * ODy + (Dz - Tz) * ODz) / denom;
                    if (d < 0.0 || d > 1.0e5)
                        edge_len = INFINITY;
                    Wx = Tx + Vx * d, Wy = Ty + Vy * d, Wz = Tz + Vz * d;
                }
                tv[3 * k] = Wx - Dx, tv[3 * k + 1] = Wy - Dy, tv[3 * k + 2] = Wz - Dz; // relative to origN = D
            }
            w.set_row(out.trivecN, tv, 9);
            w.set_row(out.tridirN, td, (tube == 1) ? 6 : 9);
            if (out.edge_lengthN)
                out.edge_lengthN[w.i] = (dtype)tube_edge_length(tv, edge_len);
        }

        w.set1(out.fbs_angleN, half_pi);
        w.set1(out.thicknessN, 0.0);
        if (out.normal_vecN)
            w.set3(out.normal_vecN, ODx, ODy, ODz), w.set3(&out.normal_vecN[3 * w.n], ODx, ODy, ODz);
        if (out.out_typeN)
            out.out_typeN[w.i] = 0;
        w.set3(out.path_dirN, ODx, ODy, ODz);
        if (out.ray_indN)
            out.ray_indN[w.i] = (unsigned)iRx;
    }

    // Where the ray from O along the unit direction OD (length OD_length, to D) meets the plane through P with
    // normal N: P moves there and len is its distance. A meeting before s0 is put at L0 (distance s0), one at or
    // past OD_length at D; a degenerate ray or a grazing plane keeps P. Returns OD . N.
    inline double ray_plane_hit(double Ox, double Oy, double Oz, double ODx, double ODy, double ODz, double OD_length,
                                double Dx, double Dy, double Dz, double Nx, double Ny, double Nz, double s0,
                                double L0x, double L0y, double L0z, double &Px, double &Py, double &Pz, double &len)
    {
        double c = ODx * Nx + ODy * Ny + ODz * Nz;  // goes to zero as the ray approaches tangency
        if (OD_length > 2e-7 && std::abs(c) > 2e-7) // guard degenerate ray and grazing/parallel plane
        {
            len = ((Px - Ox) * Nx + (Py - Oy) * Ny + (Pz - Oz) * Nz) / c;
            if (len <= s0)
                Px = L0x, Py = L0y, Pz = L0z, len = s0;
            else if (len < OD_length)
                Px = Ox + len * ODx, Py = Oy + len * ODy, Pz = Oz + len * ODz;
            else
                Px = Dx, Py = Dy, Pz = Dz, len = OD_length;
        }
        return c;
    }

    // What ray_mesh_interact makes of a hit: the out_typeN code without the TIR bit, the sheet case, the
    // effective front/back verdict (bit 1) and whether the pair is a media-to-media boundary.
    struct HitClass
    {
        qd::bits<uint8_t> code;
        SheetCase sheet = SheetCase::none;
        bool front_eff = false, m2m = false;
    };

    // ok: the FBS lies before the destination. theta_fbs, theta_sbs: the face angles (>= 0 a front-side hit);
    // N, M the FBS and SBS normals; same_sheets: both faces are sheets of one material.
    inline HitClass classify_hit(bool ok, double theta_fbs, double theta_sbs, bool colocated, double Nx, double Ny, double Nz,
                                 double Mx, double My, double Mz, bool sheet_fbs, bool sheet_sbs, bool same_sheets)
    {
        HitClass h;
        h.code = ok ? 1 : 0;
        h.code[1] = theta_fbs >= 0.0; // front-side flag
        h.code[2] = colocated;

        // Effective front/back verdict. For a solid, and for a sheet on its own, this is just the FBS
        // front-side test. It diverges only where the FBS normal is not the one that decides: a sheet
        // has no inside, so its own orientation carries no information, and for a stack the question is
        // which layer comes first rather than which side of a face was hit.
        h.front_eff = h.code[1];

        bool normals_opposing = false, normals_equal = false;
        if (colocated)
        {
            const double lim = 1.0e-4;
            normals_opposing = std::abs(Nx + Mx) < lim && std::abs(Ny + My) < lim && std::abs(Nz + Mz) < lim;
            normals_equal = std::abs(Nx - Mx) < lim && std::abs(Ny - My) < lim && std::abs(Nz - Mz) < lim;

            if (normals_opposing)
                h.m2m = true; // Opposing normal vectors, Material to material transition
            else if (normals_equal)
                h.code[3] = true; // Equal normal vectors = overlapping or duplicate faces
            else                  // FBS/SBS faces not parallel
            {
                h.code[4] = 1; // Corner hit flag
                if ((h.code[1] && theta_sbs >= 0.0) ||
                    (!h.code[1] && theta_sbs < 0.0))
                    h.code[3] = true; // Same direction flag
            }
        }

        // Thin-sheet classification. Every branch below is gated on a sheet being present, so a scene
        // without a thk column, or with all thk zero, reaches none of them and keeps the bits above.
        if (colocated && (sheet_fbs || sheet_sbs))
        {
            if (!normals_opposing && !normals_equal)
            {
                // Corner with a sheet. A sheet meeting a face at an angle is not a slab: the two faces
                // share no interior and there is no well-defined traversal. Drop SBS and handle FBS on
                // its own, as a sheet when FBS is the sheet and as an ordinary single hit otherwise.
                // The corner and co-location bits are cleared so downstream sees the single hit it is
                // actually being handed; the relaunch still goes past SBS, because that is driven by the
                // local colocated_faces flag rather than by the bit.
                h.sheet = SheetCase::corner;
                h.code[2] = false;
                h.code[3] = false;
                h.code[4] = false;
                h.code[6] = sheet_fbs;
            }
            else if (sheet_fbs && sheet_sbs)
            {
                if (same_sheets)
                {
                    // Collapsed duplicate: two coincident sheets of one material are a double-sided export, not a
                    // laminate (a laminate is two materials, or one with the summed thk). Collapsed whatever the normal
                    // parity.
                    h.sheet = SheetCase::duplicate;
                    h.code[3] = false;
                    h.code[6] = true;
                }
                else
                {
                    // Ordered stack of two different sheets, a coated window being the motivating case:
                    // it has to give different answers from either side. The layer hit on its front side
                    // is traversed first. Normals point outward from the assembly, so this is the only
                    // ordering information an exporter carries, and it is reversal-symmetric - reverse
                    // the ray and the layer order reverses with it. When both faces are front-hit or
                    // both back-hit the order is ill-posed; fall back to the geometric order FBS->SBS,
                    // which is correct whenever the faces have any real separation, and record in bit 3
                    // that the order was inferred from geometry rather than from normals.
                    h.sheet = SheetCase::stack;
                    bool frontF = theta_fbs >= 0.0, frontS = theta_sbs >= 0.0;
                    h.front_eff = (frontF != frontS) ? frontF : true;
                    h.code[3] = frontF == frontS;
                    h.code[6] = true;
                }
            }
            else
            {
                // Sheet glued to a solid face. The sheet always sits on the incidence side of that solid
                // face: a sheet glued to the solid's other boundary would be co-located with that face
                // instead of this one, unless the solid is thinner than colocation_dist, which is
                // degenerate. So the traversal order follows from co-location alone and needs no normal
                // comparison, which is why either parity is accepted here. Wallpaper on the room side of
                // a wall: entering from the room is air|paper|wall, leaving the wall is wall|paper|air.
                // The front/back verdict has to come from the solid face, since the sheet has none.
                h.front_eff = sheet_fbs ? (theta_sbs >= 0.0) : (theta_fbs >= 0.0);
                h.sheet = h.front_eff ? SheetCase::glued_entry : SheetCase::glued_exit;
                h.code[3] = false;
                h.code[6] = true;
            }
        }
        else if (sheet_fbs)
        {
            // Isolated sheet: no second face within colocation_dist, so the same medium sits on both
            // sides. Here that medium is air; a sheet embedded in a solid needs the current medium and
            // is ray_state_update's problem, not this function's.
            h.sheet = SheetCase::isolated;
            h.code[6] = true;
        }

        // Bit 1 carries the effective verdict, not the raw FBS test. For solids and isolated sheets
        // front_eff is the FBS test and this write changes nothing.
        h.code[1] = h.front_eff;
        return h;
    }

    // The media of a hit for ray_mesh_interact, which sees faces only (air on any side the mesh does not name):
    // near (the wave's side) and far for interact_with, and for a sheet composite the sheet layers.
    struct HitMedia
    {
        Material near, far, sheet, sheet2;
        bool composite = false;  // the coefficients come from the sheet composite
        bool one_medium = false; // the same medium (air) on both sides of the sheet
    };

    template <typename dtype>
    inline HitMedia hit_media(const MaterialCols<dtype> &cols, arma::uword mtl_fbs, arma::uword mtl_sbs, const HitClass &h,
                              bool sheet_fbs, double theta_fbs)
    {
        // Incidence-side (near) and entered/reflected-off (far) materials, default air. far carries the
        // FBS-face material whose interface_gain is the transition gain (front: the FBS's; back: air, or the
        // SBS's for M2M). A sheet is three media, two interfaces and one slab: near and far become the media
        // on either side of it and the sheet itself the layer; m2m plays no part, the pair is one crossing.
        HitMedia hm;
        const SheetCase sc = h.sheet;
        hm.one_medium = sc == SheetCase::isolated || sc == SheetCase::stack || sc == SheetCase::duplicate || (sc == SheetCase::corner && sheet_fbs);
        if (hm.one_medium)
        {
            // An isolated sheet, a collapsed double-sided duplicate, an ordered stack and a sheet
            // at a corner, which is the sheet alone: the same medium sits on both sides, so
            // near = far = air. The sheet material never reaches interact_with - it enters only
            // through the composite, which is what makes its normal orientation irrelevant. A
            // stack is traversed in the order bit 1 carries.
            hm.sheet = Material(cols, (sc == SheetCase::stack && !h.front_eff) ? mtl_sbs : mtl_fbs);
            if (sc == SheetCase::stack)
                hm.sheet2 = Material(cols, h.front_eff ? mtl_sbs : mtl_fbs);
            hm.composite = true;
        }
        else if (sc == SheetCase::glued_entry || sc == SheetCase::glued_exit)
        {
            // Glued pair: air | sheet | solid on entry, solid | sheet | air on exit. Which of the two
            // co-located faces carries the sheet does not matter, because front_eff already took the
            // verdict from the solid face, so FBS-sheet and SBS-sheet geometry land here identically.
            // A back-side hit on the solid implies the ray is already inside it, which is the same
            // trick the near/far block below uses to recover the incidence medium without any state.
            hm.sheet = Material(cols, sheet_fbs ? mtl_fbs : mtl_sbs);
            Material M_solid(cols, sheet_fbs ? mtl_sbs : mtl_fbs);
            if (h.front_eff) // glued entry, o->i
                hm.far = M_solid;
            else // glued exit, i->o
                hm.near = M_solid;
            hm.composite = true;
        }
        else if (theta_fbs >= 0.0) // front hit: entered material = FBS face
        {
            hm.far = Material(cols, mtl_fbs);
            if (h.m2m) // SBS (front) hit first
                hm.near = Material(cols, mtl_sbs);
        }
        else // back hit: incidence material = FBS face
        {
            hm.near = Material(cols, mtl_fbs);
            if (h.m2m) // FBS (back) hit first, entered = SBS
                hm.far = Material(cols, mtl_sbs);
        }
        return hm;
    }

    // Incidence of the three vertex rays of a tube at the face with unit normal N (k = 0 is the spine).
    struct TubeIncidence
    {
        double V[9];             // incoming vertex directions
        double cosV[4], sinV[4]; // incidence cosine and sine: the spine, then the three vertices
    };

    template <typename dtype>
    inline TubeIncidence tube_incidence(const dtype *p_tridir, int tube, size_t iRx, size_t n_ray, double Nx, double Ny, double Nz, double abs_cos_theta)
    {
        TubeIncidence ti;
        ti.cosV[0] = abs_cos_theta;
        ti.sinV[0] = std::sqrt(1.0 - abs_cos_theta * abs_cos_theta);
        for (int k = 0; k < 3; ++k)
        {
            double Vx, Vy, Vz;
            tube_vertex_dir(p_tridir, tube, iRx, n_ray, (size_t)k, Vx, Vy, Vz);
            ti.V[3 * k] = Vx, ti.V[3 * k + 1] = Vy, ti.V[3 * k + 2] = Vz;
            double c = std::abs(Vx * Nx + Vy * Ny + Vz * Nz);
            c = (c > 1.0) ? 1.0 : c;
            ti.cosV[k + 1] = c, ti.sinV[k + 1] = std::sqrt(1.0 - c * c);
        }
        return ti;
    }

    // The three vertex rays of a tube after the face (F, unit normal N oriented to the incidence side). Each
    // meets the face plane where it can, else its orthogonal projection onto F is used and the edge length turns
    // infinite; it is then mirrored (geometry 0), kept (1, or under whole-tube total reflection) or refracted with
    // the spine's Snell ratio snell (2). tv: the new vertex origins relative to F; td: the directions ([az el] for tube 1,
    // Cartesian for tube 2). Returns the longest edge where want_edge, else 0 or INFINITY.
    template <typename dtype>
    inline double tube_update(const dtype *p_trivec, int tube, size_t iRx, size_t n_ray, const TubeIncidence &ti,
                              double Ox, double Oy, double Oz, double Fx, double Fy, double Fz, double Nx, double Ny, double Nz,
                              int geometry_type, bool total_reflection, double snell, std::complex<double> n2_ratio, bool want_edge,
                              double tv[9], double td[9])
    {
        double edge_len = 0.0;
        for (int k = 0; k < 3; ++k)
        {
            double Tx = Ox, Ty = Oy, Tz = Oz, az = 0.0, el = 0.0;
            double Vx = ti.V[3 * k], Vy = ti.V[3 * k + 1], Vz = ti.V[3 * k + 2];
            Tx += (double)p_trivec[iRx + 3 * k * n_ray], Ty += (double)p_trivec[iRx + (3 * k + 1) * n_ray], Tz += (double)p_trivec[iRx + (3 * k + 2) * n_ray];

            double denom = Vx * Nx + Vy * Ny + Vz * Nz;
            bool no_usable_hit = std::abs(denom) < 1e-6; // true => parallel, no face intersection
            double d = no_usable_hit ? 0.0 : ((Fx - Tx) * Nx + (Fy - Ty) * Ny + (Fz - Tz) * Nz) / denom;
            double Wx = Tx + Vx * d, Wy = Ty + Vy * d, Wz = Tz + Vz * d;
            no_usable_hit = no_usable_hit || d > 1.0e5 || d < 0.0;

            if (no_usable_hit) // no usable face intersection
                edge_len = INFINITY;

            if (geometry_type == 0) // Reflection
            {
                if (no_usable_hit) // Use orthogonal projection on vertex ray
                {
                    d = ((Fx - Tx) * Vx + (Fy - Ty) * Vy + (Fz - Tz) * Vz) / (Vx * Vx + Vy * Vy + Vz * Vz);
                    Tx = Tx + Vx * d - Fx, Ty = Ty + Vy * d - Fy, Tz = Tz + Vz * d - Fz; // Scaled vertex - updates T
                    double a = 2.0 * (Tx * Nx + Ty * Ny + Tz * Nz);                      // Reflection of T on face
                    Tx -= a * Nx, Ty -= a * Ny, Tz -= a * Nz;
                }
                else // Use intersect point W as new vertex origin
                    Tx = Wx - Fx, Ty = Wy - Fy, Tz = Wz - Fz;

                qd_reflect(Vx, Vy, Vz, Nx, Ny, Nz, Vx, Vy, Vz);
                if (tube == 1)
                {
                    Vz = (Vz < -1.0) ? -1.0 : (Vz > 1.0 ? 1.0 : Vz); // Boundary fix
                    az = std::atan2(Vy, Vx), el = std::asin(Vz);
                }
            }
            else // Transmission and Refraction
            {
                if (no_usable_hit) // Use orthogonal projection on vertex ray
                {
                    d = ((Fx - Tx) * Vx + (Fy - Ty) * Vy + (Fz - Tz) * Vz) / (Vx * Vx + Vy * Vy + Vz * Vz);
                    Wx = Tx + Vx * d, Wy = Ty + Vy * d, Wz = Tz + Vz * d;
                }

                Tx = Wx - Fx, Ty = Wy - Fy, Tz = Wz - Fz;

                if (geometry_type == 2 && !total_reflection) // Refraction (skipped when the whole tube passes through under TIR)
                {
                    std::complex<double> cos_theta2V = std::sqrt(1.0 - n2_ratio * ti.sinV[k + 1] * ti.sinV[k + 1]);
                    qd_refract(Vx, Vy, Vz, Nx, Ny, Nz, snell, ti.cosV[k + 1], cos_theta2V, Vx, Vy, Vz);
                }
            }

            if (tube == 1)
            {
                Vz = (Vz < -1.0) ? -1.0 : (Vz > 1.0 ? 1.0 : Vz); // Boundary fix
                az = std::atan2(Vy, Vx), el = std::asin(Vz);
            }

            tv[3 * k] = Tx, tv[3 * k + 1] = Ty, tv[3 * k + 2] = Tz;
            if (tube == 1)
                td[2 * k] = az, td[2 * k + 1] = el;
            else
                td[3 * k] = Vx, td[3 * k + 1] = Vy, td[3 * k + 2] = Vz;
        }
        return want_edge ? tube_edge_length(tv, edge_len) : edge_len;
    }

    // ---- 7 State machine, pure parts (ray_state_update) ---------------------------------------------------

    // The three state words of a ray, decoded (the layout is in the file header):
    //   current  bits 0-13 the medium the ray is in;  bit 14 cur_bent;  bit 15 cur_summed
    //   prev     bits 0-13 the medium the leg was entered from, or the sheet crossed or reflected off;
    //            bit 14 prev_embedded (for a solid: the face was evaluated at a real-invariant cosine);
    //            bit 15 prev_nonparallel
    //   buffer   bits 0-13 a deferred medium; bit 14 buf_followed (with an index: the stack layer behind);
    //            or under bit 15 buf_mass_code a thickness code in bits 0-14
    struct StateIn
    {
        int mtl_current = 0, mtl_prev = 0;
        int mtl_buffer = 0;            // the deferred medium; 0 under a mark or a code
        bool resolved = false;         // the leg's series was summed (cur_summed)
        bool bent = false;             // the leg follows a traced bend (cur_bent, never in air)
        bool prev_nonpar = false;      // the leg's faces are known not parallel
        bool prev_is_embedded = false; // bit 14 of prev
        bool mass_crossed = false;     // the leg was entered by crossing the bonded mass-law body in prev
        int mass_code = 0;             // its thickness code
        bool followed = false;         // the leg began at a reflection that left the series to the tracer
        int stack_behind = 0;          // the layer of an ordered stack behind the ray that prev does not name
        short buffer_word = 0;         // the buffer word as it came in
    };

    inline StateIn decode_words(short cur, short prev, short buf)
    {
        StateIn st;
        st.mtl_current = cur & word_mtl;
        st.mtl_prev = prev & word_mtl;
        st.mtl_buffer = buf & word_code;
        st.resolved = (cur & cur_summed) != 0;
        st.bent = (cur & cur_bent) != 0 && st.mtl_current != 0;
        st.prev_nonpar = (prev & prev_nonparallel) != 0;
        st.prev_is_embedded = (prev & prev_embedded) != 0;
        st.buffer_word = buf;
        st.mass_crossed = (buf & buf_mass_code) != 0;
        st.mass_code = st.mass_crossed ? st.mtl_buffer : 0;
        st.stack_behind = (!st.mass_crossed && (buf & buf_followed) != 0) ? (int)(buf & word_mtl) : 0;
        st.followed = !st.mass_crossed && (buf & buf_followed) != 0 && st.stack_behind == 0;
        if (st.mass_crossed || st.followed || st.stack_behind != 0)
            st.mtl_buffer = 0;
        return st;
    }

    // The buffer word of a leg that begins by crossing out of a mass-law solid whose closed leg had the
    // surface-mass surrogate mass_n (a normal thickness, m): buf_mass_code and the code
    // k = 1 + round((log10 d + 4) 32766 / 6), d = 10^(-4 + 6 (k - 1) / 32766) m for k = 1 .. 32767 (0.1 mm to
    // 100 m in steps of 0.044 %). Zero where the body carried no mass term.
    inline short encode_mass_code(double mass_n)
    {
        if (!(mass_n > 0.0))
            return (short)0;
        double k = 1.0 + std::round((std::log10(mass_n) + 4.0) * (32766.0 / 6.0));
        k = (k < 1.0) ? 1.0 : ((k > 32767.0) ? 32767.0 : k);
        return (short)(buf_mass_code | (int)k);
    }

    // The thickness of a code (0 for none).
    inline double decode_mass_code(int k)
    {
        return (k < 1) ? 0.0 : std::pow(10.0, -4.0 + 6.0 * (double)(k - 1) / 32766.0);
    }

    // A Material for a 1-based index (0 = air).
    template <typename dtype>
    inline Material material(const MaterialCols<dtype> &cols, int m)
    {
        return Material(cols, (arma::uword)(m < 1 ? 0 : m));
    }

    // Same medium: index 0 (outside, no medium) matches only itself; two materials match on identical columns.
    template <typename dtype>
    inline bool same_medium(const MaterialCols<dtype> &cols, int a, int b)
    {
        return a == b || (a > 0 && b > 0 && material(cols, a).same_as(material(cols, b)));
    }

    // A sheet index: 1-based, thk > 0.
    template <typename dtype>
    inline bool is_sheet_mtl(const MaterialCols<dtype> &cols, int m)
    {
        return m > 0 && (arma::uword)m <= cols.n_mtl && cols.is_sheet((arma::uword)m);
    }

    // Equal permittivity and permeability at fGHz (within 1e-9): no boundary for a wave, whatever the other
    // columns (same_as compares thk, which is what differs between a sheet and its solid).
    inline bool same_eps_mu(const Material &A, const Material &B, double fGHz)
    {
        std::complex<double> ea = A.eta(fGHz), eb = B.eta(fGHz), ma = A.mu(fGHz), mb = B.mu(fGHz);
        double se = std::abs(ea) + std::abs(eb), sm = std::abs(ma) + std::abs(mb);
        return std::abs(ea - eb) <= 1.0e-9 * ((se > 0.0) ? se : 1.0) &&
               std::abs(ma - mb) <= 1.0e-9 * ((sm > 0.0) ? sm : 1.0);
    }

    inline double dist3(const double *A, const double *B)
    {
        double a0 = B[0] - A[0], a1 = B[1] - A[1], a2 = B[2] - A[2];
        return std::sqrt(a0 * a0 + a1 * a1 + a2 * a2);
    }

    // Normalizes A by dividing by its length above 1e-12, else sets it to (1, 0, 0); returns the length.
    inline double normalize_exact(double *A)
    {
        double len = std::sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
        if (len > 1.0e-12)
            A[0] /= len, A[1] /= len, A[2] /= len;
        else
            A[0] = 1.0, A[1] = 0.0, A[2] = 0.0;
        return len;
    }

    // Intersection of the direction U (unit) from O with the plane through F with unit normal N: the distance
    // dist_OV, |cos theta_t| = |U . N| and the point V. False where U is near parallel to the plane or the plane
    // lies behind O.
    inline bool vbs_intersect(const double *O, const double *U, const double *F, const double *N,
                              double *dist_OV = nullptr, double *cos_theta_t = nullptr, double *V = nullptr)
    {
        double un = U[0] * N[0] + U[1] * N[1] + U[2] * N[2];
        if (std::abs(un) < 1.0e-6)
            return false;

        double s = ((F[0] - O[0]) * N[0] + (F[1] - O[1]) * N[1] + (F[2] - O[2]) * N[2]) / un;
        if (!std::isfinite(s) || s <= 0.0)
            return false;

        if (dist_OV)
            *dist_OV = s;
        if (cos_theta_t)
            *cos_theta_t = std::abs(un);
        if (V)
            V[0] = O[0] + s * U[0], V[1] = O[1] + s * U[1], V[2] = O[2] + s * U[2];
        return true;
    }

    // Wedge test: the FBS and SBS faces of event i (normal_vecN, n events) at a real angle; false for a
    // co-located pair.
    template <typename dtype>
    inline bool faces_not_parallel(const double F[3], const double S[3], const dtype *p_normal_vecN, size_t i, size_t n)
    {
        if (dist3(F, S) < colocation_dist) // Co-located
            return false;

        double nfx = (double)p_normal_vecN[i];
        double nfy = (double)p_normal_vecN[i + n];
        double nfz = (double)p_normal_vecN[i + 2 * n];
        double nsx = (double)p_normal_vecN[i + 3 * n];
        double nsy = (double)p_normal_vecN[i + 4 * n];
        double nsz = (double)p_normal_vecN[i + 5 * n];

        const double tol = 3.8e-3;
        double d = nfx * nsx + nfy * nsy + nfz * nsz;
        return std::abs(d) < 1.0 - tol;
    }

    // The material of the current medium by its exit face: the FBS for the i-o and i-i codes, the SBS where the
    // next medium was hit first (code 7); 0 where the code names none.
    inline int exit_material(int hit_code, int mtl_fbs, int mtl_sbs)
    {
        if (hit_code == 1 || hit_code == 5 || hit_code == 13 || hit_code == 21 || hit_code == 29) // i-o or i-i
            return mtl_fbs;
        if (hit_code == 7) // i-i with M2 (next, front) hit first
            return mtl_sbs;
        return 0;
    }

    // The exact complex cosine of the wave in a traced medium, recovered from the traced one. A traced ray has a
    // real direction (qd_refract normalized a vector built from Re(cos_theta2)), and a face evaluated at that
    // projection is not the face the entry coefficient was formed at; inside a lossless medium it would even
    // need a complex angle. qd_refract's map from the incidence angle,
    //     ct_traced = Re(ct2) / sqrt(eta^2 sin^2(theta_i) + Re(ct2)^2),  ct2 = sqrt(1 - eps sin^2),
    // is monotone and its parameters are known (the entry medium, the medium), so it is inverted: Newton on
    // eta^2 u = t^2 Re(sqrt(1 - eps u))^2, u = sin^2(theta_i), from the real-index Snell start.
    //
    // unnamed: the entry medium is not named (behind a sheet, or a solid whose face was evaluated at a
    // real-invariant cosine). It need not be: with p = e1 sin^2(theta_1) the equation is
    // |p| / |e2| = t^2 Re(sqrt(1 - p/e2))^2, and for a lossless entry medium p is real and e1 appears nowhere
    // else. So the solve runs with a unit entry permittivity (u = p) and without the bound u <= 1: exact across
    // parallel faces for a wave that started in air, however lossy the layers in front. Normal incidence is
    // included (t = 0 returns exactly 1), so the propagation factor has one form at every angle.
    //
    // real_invariant: the wave in the medium has a real transverse invariant and a complex cosine (solved for a
    // real invariant, or for a real angle in a lossless entry medium, and the medium lossy); decided on the
    // media, never on a value, so it is the same at every frequency of a run. Returns 0 where the recovery does
    // not apply (callers then use the projection).
    inline std::complex<double> recover_cosine(double ct_traced, const Material &entry, const Material &medium, bool unnamed,
                                               double fGHz, bool &real_invariant)
    {
        const std::complex<double> none(0.0, 0.0);
        real_invariant = false;
        const double u_max = unnamed ? 1.0e30 : 1.0;
        if (ct_traced <= 0.0 || ct_traced > 1.0)
            return none;

        std::complex<double> e1 = entry.eta(fGHz) * entry.mu(fGHz), e2 = medium.eta(fGHz) * medium.mu(fGHz);
        if (std::abs(e2) <= 0.0)
            return none;
        std::complex<double> eps_r = e1 / e2; // qd_refract's n2_ratio
        double eta2 = std::abs(eps_r);        // qd_refract's eta, squared
        if (eta2 <= 0.0 || std::abs(eps_r - 1.0) < 1.0e-12)
            return none; // matched media: no bend to invert

        double ct = ct_traced;
        double ct2c = 1.0 - ct * ct;
        double t = std::sqrt((ct2c > 0.0) ? ct2c : 0.0) / ct; // tan of the traced in-slab angle
        double u = t * t / (eta2 * (1.0 + t * t));            // real-index Snell start
        u = (u > u_max) ? u_max : u;

        // Newton on the map, until the step is below rounding. Three steps reach that from the
        // real-index start wherever real-index Snell has a solution near the root; in a lossy
        // medium rarer than the one the wave came from it has none past sqrt|eps_r| sin = 1,
        // and towards an invariant above one the start is far as well.
        for (int k = 0; k < 50; ++k)
        {
            std::complex<double> w = std::sqrt(1.0 - eps_r * u);
            double fu = eta2 * u - t * t * std::real(w) * std::real(w);
            double dfu = eta2 + t * t * std::real(w) * std::real(eps_r / w);
            if (!(std::abs(dfu) > 1.0e-300))
                break;
            double du = fu / dfu;
            u -= du;
            u = (u < 0.0) ? 0.0 : ((u > u_max) ? u_max : u);
            if (k >= 2 && std::abs(du) <= 1.0e-15 * std::abs(u))
                break;
        }

        real_invariant = (unnamed || std::imag(e1) == 0.0) && std::imag(e2) != 0.0;
        return decaying_root(1.0 - eps_r * u);
    }

    // Summed X0 of the bonded mass-law bodies crossed on the way into the current leg, as far as the state names
    // them (mass_crossed): a solid in prev is one body of the decoded thickness; a sheet there is one body by its
    // thk column, and where it lies inside the current medium (embedded) the part of that medium in front of it is
    // a second one, of the decoded thickness. cos_air: the cosine the wave has in air.
    inline double bonded_x0(bool mass_crossed, int mass_code, const Material &prev, bool prev_is_sheet, bool embedded,
                            const Material &current, double fGHz, double cos_air)
    {
        if (!mass_crossed)
            return 0.0;
        if (!(prev.m > 0.0))
            return 0.0;
        double d_code = decode_mass_code(mass_code);
        if (prev_is_sheet)
        {
            double X = prev.mass_X0(prev.thk, fGHz, cos_air);
            if (embedded && d_code > 0.0)
            {
                if (current.m > 0.0)
                    X += current.mass_X0(d_code, fGHz, cos_air);
            }
            return X;
        }
        if (mass_code < 1)
            return 0.0;
        return prev.mass_X0(d_code, fGHz, cos_air);
    }

    // Normal thickness of material R with the X0 of a leg of R of normal thickness d together with the bonded
    // bodies crossed before it (as bonded_x0 names them) and the first layer of an ordered stack (extra):
    // X0 = (f / fRef) d is linear in each thickness, so the sum is one body of R at every frequency, each
    // thickness scaled by fRef_R / fRef. No frequency enters, which lets the result travel as the code.
    inline double bonded_thickness(const Material &R, double d, const Material *extra, bool mass_crossed, int mass_code,
                                   const Material &prev, bool prev_is_sheet, bool embedded, const Material &current)
    {
        double s = d;
        auto add = [&s, &R](const Material &B, double thk)
        {
            if (B.m > 0.0 && thk > 0.0)
                s += thk * R.fRef / B.fRef;
        };
        if (extra)
            add(*extra, extra->thk);
        if (mass_crossed)
        {
            double d_code = decode_mass_code(mass_code);
            if (!prev_is_sheet)
                add(prev, d_code);
            else
            {
                add(prev, prev.thk);
                if (embedded)
                    add(current, d_code);
            }
        }
        return s;
    }

    // The absorption of a face of a plate of material C seen from medium P: what it passes of the flux that
    // meets it, without the interface gain (itype 1, or 4 in scalar mode).
    inline double face_absorption(const Material &P, const Material &C, int itype, double theta_t, double fGHz)
    {
        double ig = C.interface_gain(fGHz), g = interact_with(P, C, itype, theta_t, fGHz).gain;
        return (ig > 0.0) ? g / ig : 0.0;
    }

    // The exact in-medium factor of a leg at the recovered cosine cs. A wave entered from a lossless medium has a
    // real transverse wavenumber, so its decay sits on the normal component, exp(k0 Im(n cs) d) over the normal
    // thickness d, not the ITU rate along the path of medium_gain (equal only at normal incidence). One factor
    // exp(-j k0 (n cs d + q s)), s the transverse walk and q^2 = n^2 (1 - cs^2) the transverse wavenumber; for a
    // real index it reduces to n dist_refract. d is geometric (traced path times traced cosine), as in
    // qd_slab_resolve. The conductivity is inside n, so only alpha is drawn from the material. Where the polyline
    // is not the refracted ray (use_vbs), the phase is moved to the point the polyline leaves at. The free-space
    // reference over the geometric length is taken out.
    inline std::complex<double> leg_exact_factor(const Material &mtl, std::complex<double> cs, double dist_refract, double dist_geo,
                                                 double abs_cos_theta_t, double abs_cos_theta, bool use_vbs, double fGHz)
    {
        std::complex<double> n_cx = std::sqrt(mtl.eta(fGHz) * mtl.mu(fGHz));
        double d_norm = dist_refract * abs_cos_theta_t; // normal thickness
        double sin_t = 1.0 - abs_cos_theta_t * abs_cos_theta_t;
        double s_lat = dist_refract * std::sqrt((sin_t > 0.0) ? sin_t : 0.0); // transverse walk

        std::complex<double> q = decaying_root(n_cx * n_cx * (1.0 - cs * cs));

        double ex = mtl.excess_gain(dist_refract, fGHz); // alpha only, no conductivity
        std::complex<double> P =
            std::exp(std::complex<double>(0.0, -1.0) * k0(fGHz) * (n_cx * cs * d_norm + q * s_lat)) *
            std::sqrt((ex < 0.0) ? 0.0 : ex);

        // From the point the refracted ray leaves at to the point the polyline leaves at: the lateral walk of
        // the polyline in place of the wave's. A phase, also where the invariant is complex, so that every
        // magnitude stays the one of the ray.
        if (use_vbs)
        {
            double sin_i = 1.0 - abs_cos_theta * abs_cos_theta;
            double s_poly = dist_geo * std::sqrt((sin_i > 0.0) ? sin_i : 0.0);
            P *= std::exp(std::complex<double>(0.0, -1.0) * k0(fGHz) * std::real(q) * (s_poly - s_lat));
        }

        // Free-space reference over the geometric length
        P *= std::exp(std::complex<double>(0.0, 1.0) * k0(fGHz) * dist_geo);
        return P;
    }

    // The sheet of a hit as ray_state_update re-derives it from the hit code and both faces. Bit 6 only says
    // that a face is a sheet; codes 69 and 71 are shared by three topologies, told apart from (code,
    // is_sheet(FBS), is_sheet(SBS), same material). A duplicate, a corner and a glued pair whose solid face has no
    // material are isolated sheets here. mtl: the sheet (the first layer of a stack), mtl2: the second layer,
    // solid: the glued solid.
    struct SheetHit
    {
        SheetCase kind = SheetCase::none;
        int mtl = 0, mtl2 = 0, solid = 0;
    };

    template <typename dtype>
    inline SheetHit sheet_hit(const MaterialCols<dtype> &cols, qd::bits<uint8_t> hit_code, int mtl_fbs, int mtl_sbs)
    {
        SheetHit sh;
        if (!hit_code[6])
            return sh;
        bool shF = is_sheet_mtl(cols, mtl_fbs), shS = is_sheet_mtl(cols, mtl_sbs);
        if (!hit_code[2]) // No co-located second face: nothing can be glued to it
        {
            sh.kind = (shF || shS) ? SheetCase::isolated : SheetCase::none;
            sh.mtl = shF ? mtl_fbs : (shS ? mtl_sbs : 0);
        }
        else if (shF && shS) // Two sheets: a duplicate collapses, a laminate does not
        {
            sh.kind = same_medium(cols, mtl_fbs, mtl_sbs) ? SheetCase::isolated : SheetCase::stack;
            if (sh.kind == SheetCase::stack)
            {
                // Order: bit 1 names the layer hit on its front side (first); bit 3 says the normals gave none
                // and the geometric order was used. Only the reflect port depends on it (the transmit port is
                // reciprocal).
                sh.mtl = hit_code[1] ? mtl_fbs : mtl_sbs;
                sh.mtl2 = hit_code[1] ? mtl_sbs : mtl_fbs;
            }
            else
                sh.mtl = mtl_fbs;
        }
        else if (shF || shS) // One sheet glued to one solid face
        {
            sh.mtl = shF ? mtl_fbs : mtl_sbs;
            sh.solid = shF ? mtl_sbs : mtl_fbs;

            // A glued pair whose non-sheet face carries no material has the same medium on both sides of the
            // sheet. Otherwise bit 1 carries the front/back verdict taken from the solid face.
            sh.kind = (sh.solid == 0) ? SheetCase::isolated : (hit_code[1] ? SheetCase::glued_entry : SheetCase::glued_exit);
        }
        return sh;
    }

    // The relaunch offset in the ledger. The next event measures the leg that follows from the relaunch point,
    // qd_relaunch_offset past the point this event left at (F, or the far face of a co-located pair), so the
    // ledger starts with that offset and the next leg adds up to the leg from the hit point, at any distance from
    // the origin. The refracted ledger is measured along the wave's direction dn to the next face: it misses the
    // offset times |g.Nf| / |dn.Nf|, g the direction the segment goes on in (the incoming one F - O, or its mirror,
    // which has the same normal cosine), and 1 where the tracer bends the segment itself (bends: types 2 and 5).
    // Exact for parallel faces. Nf (the face's normal) and dn are normalized here.
    inline void relaunch_ledger_offset(const double F[3], const double O[3], double Nf[3], double dn[3], bool bends,
                                       double &dist_geo, double &dist_refract)
    {
        normalize_exact(Nf);
        normalize_exact(dn);
        const double off = qd_relaunch_offset(F[0], F[1], F[2]);
        dist_geo += off;
        if (bends)
            dist_refract += off;
        else
        {
            double g[3] = {F[0] - O[0], F[1] - O[1], F[2] - O[2]};
            normalize_exact(g);
            const double cg = std::abs(g[0] * Nf[0] + g[1] * Nf[1] + g[2] * Nf[2]), cd = std::abs(dn[0] * Nf[0] + dn[1] * Nf[1] + dn[2] * Nf[2]);
            dist_refract += (cd > 1.0e-9) ? off * cg / cd : off;
        }
    }

    // The incidence at the face from the direction the wave has (dir_prev, unit) rather than from the segment O-F:
    // the virtual hit point V of that direction in the face plane (F, N), the refracted length to it, the angle
    // and its cosine. Without a direction (has_dir false) V is F. moved (use_vbs): the direction differs from the
    // segment, the cosines by more than tol.
    struct VbsGeometry
    {
        double V[3];                     // the virtual hit point
        double len_to_vbs;               // the refracted length from O to V
        double theta_t, abs_cos_theta_t; // the incidence angle at V (face convention) and its cosine
        double cos_dir_seg;              // the cosine between the wave's direction and the segment
        bool moved;
    };

    inline VbsGeometry vbs_geometry(const double O[3], const double F[3], const double N[3], const double dir_prev[3], bool has_dir,
                                    double theta, double abs_cos_theta, double len_to_face, double tol)
    {
        VbsGeometry g;
        g.V[0] = F[0], g.V[1] = F[1], g.V[2] = F[2];
        g.len_to_vbs = len_to_face, g.theta_t = theta, g.abs_cos_theta_t = abs_cos_theta, g.cos_dir_seg = 1.0;
        g.moved = has_dir;
        if (has_dir)
        {
            double dist_OV, ct;
            if (vbs_intersect(O, dir_prev, F, N, &dist_OV, &ct, g.V))
            {
                g.len_to_vbs = dist_OV;
                g.abs_cos_theta_t = (ct > 1.0) ? 1.0 : ct;
                g.theta_t = std::acos(g.abs_cos_theta_t) - half_pi;
                g.moved = std::abs(g.abs_cos_theta_t - abs_cos_theta) > tol;
            }
        }
        if (g.moved)
        {
            double G[3] = {F[0] - O[0], F[1] - O[1], F[2] - O[2]};
            normalize_exact(G);
            g.cos_dir_seg = dir_prev[0] * G[0] + dir_prev[1] * G[1] + dir_prev[2] * G[2];
        }
        return g;
    }
}

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# medium_gain
Linear gain of a ray traversing a homogeneous lossy medium

- Computes `g = 10^(-A/10)` for a path of length `dist` inside material `iM`, where `A` [dB] sums:
  - the conductivity loss of the ITU-R P.2040 model, `8.686·k₀·|Im √(ε_r·μ_r)|` dB/m, with
    `ε_r = a·(f/fRef)^b − j·17.98·σ/f[GHz]`, `σ = c·(f/fRef)^d`, `μ_r` from `e`, `f`, `g`, `h` in the same form,
    and for a material without the mass column the permittivity resonance (`resF`, `resQ`, `resS`);
  - the distance absorption `alpha·(f/fRef)^alphaB` dB/m (excess loss not captured by `σ`, e.g. foliage);
  - for a mass-law material (`m > 0`, the acoustic switch) the mass term, added once and not per metre where
    `dist` exceeds 1.5 mm: `20·log10((f/fRef)·dist)` dB with `dist` as the normal thickness, the resonance and
    coincidence columns acting on it, zero where it would be negative. With a panel size `side > 0` (√area in m,
    3.29 where the column is absent) Davy's radiation efficiency of a finite panel enters: a panel small in
    wavelengths insulates more than the infinite plate (`side = 0`).
- One path length at normal incidence: the angle dependence of the mass term, the bonded plate of several
  bodies and the refracted path inside a traced solid are applied by [[ray_state_update]], not here.
- The penetration-loss columns `att`, `attB` are not used here ([[interface_gain]]).

## Declaration:
```
dtype quadriga_lib::medium_gain(
    const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
    arma::uword iM,
    dtype dist,
    dtype center_frequency);
```

## Inputs:
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]); each value has length `n_mtl`
- **`iM`** —  1-based material index (0 = no material / air)
- **`dist`** — Path length of the ray inside the medium in [m]
- **`center_frequency`** — Center frequency in [Hz]

## Returns:
- Linear in-medium gain in `[0, 1]`; multiply by the incident field/power gain to get the value after the medium

## See also:
- [[ray_mesh_interact]] (for complex ray-material interactions)
- [[obj_file_read]] (defines mtl_prop format)
MD!*/

template <typename dtype>
dtype quadriga_lib::medium_gain(const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
                                arma::uword iM, dtype dist, dtype center_frequency)
{
    if (!std::isfinite((double)center_frequency) || center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");
    MaterialCols<dtype> cols(mtl_prop); // validates column lengths and physical sanity
    if (iM > cols.n_mtl)
        throw std::invalid_argument("Material index out of bound.");
    return (dtype)Material(cols, iM).medium_gain((double)dist, (double)center_frequency * 1e-9);
}

template float quadriga_lib::medium_gain(const std::unordered_map<std::string, std::vector<float>> &mtl_prop, arma::uword iM, float dist, float center_frequency);
template double quadriga_lib::medium_gain(const std::unordered_map<std::string, std::vector<double>> &mtl_prop, arma::uword iM, double dist, double center_frequency);

/*!MD
# interface_gain
Linear gain of the lumped penetration loss of entering a material

- Computes `g = 10^(-A/10)` with `A = att·(f/fRef)^attB` [dB], the loss applied once where a ray enters material
  `iM`, on top of the Fresnel term of the face (e.g. a building-entry loss of 3GPP TR 38.901). `att` must be
  `≥ 0` (the material table is checked when it is read), so `g` is in `[0, 1]`.
- Only `att`, `attB` and `fRef` are read; the face reflection is computed by [[ray_mesh_interact]] and
  [[ray_state_update]], the distance-dependent loss by [[medium_gain]]. Columns the material model does not know
  are ignored.

## Declaration:
```
dtype quadriga_lib::interface_gain(
    const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
    arma::uword iM,
    dtype center_frequency);
```

## Inputs:
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]); each value has length `n_mtl`
- **`iM`** — 1-based material index (0 = no material / air)
- **`center_frequency`** — Center frequency in [Hz]

## Returns:
- Linear interface gain in `[0, 1]`; multiply by the incident field/power gain to get the value after the interface

## See also:
- [[medium_gain]] (for the distance-dependent in-medium loss)
- [[ray_mesh_interact]] (for complex ray-material interactions)
- [[obj_file_read]] (defines mtl_prop format)
MD!*/

template <typename dtype>
dtype quadriga_lib::interface_gain(const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
                                   arma::uword iM, dtype center_frequency)
{
    if (center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");
    MaterialCols<dtype> cols(mtl_prop);
    if (iM > cols.n_mtl)
        throw std::invalid_argument("Material index out of bound.");
    return (dtype)Material(cols, iM).interface_gain((double)center_frequency * 1e-9);
}

template float quadriga_lib::interface_gain(const std::unordered_map<std::string, std::vector<float>> &mtl_prop, arma::uword iM, float center_frequency);
template double quadriga_lib::interface_gain(const std::unordered_map<std::string, std::vector<double>> &mtl_prop, arma::uword iM, double center_frequency);

/*!MD
# refractive_index
Real refractive index of a homogeneous medium

- Returns `n = Re √(ε_r·μ_r)` of material `iM`, with the ITU-R P.2040 model `ε_r = a·(f/fRef)^b − j·17.98·σ/f[GHz]`,
  `σ = c·(f/fRef)^d`, and `μ_r` from `e`, `f`, `g`, `h` in the same form. For a material without the mass column
  the permittivity resonance (`resF`, `resQ`, `resS`) is part of `ε_r`.
- For a mass-law material (`m > 0`) `ε_r` and `μ_r` are an impedance mapping and `n` is not a propagation index.
- Air (`iM = 0`) returns `1`.

## Declaration:
```
dtype quadriga_lib::refractive_index(
    const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
    arma::uword iM,
    dtype center_frequency);
```

## Inputs:
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]); each value has length `n_mtl`
- **`iM`** — 1-based material index (0 = no material / air)
- **`center_frequency`** — Center frequency in [Hz]

## Returns:
- Real refractive index of the medium relative to air

## See also:
- [[medium_gain]] (for the distance-dependent in-medium loss)
- [[ray_mesh_interact]] (for complex ray-material interactions)
- [[obj_file_read]] (defines mtl_prop format)
MD!*/

template <typename dtype>
dtype quadriga_lib::refractive_index(const std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
                                     arma::uword iM, dtype center_frequency)
{
    if (!std::isfinite((double)center_frequency) || center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");
    MaterialCols<dtype> cols(mtl_prop); // validates column lengths and physical sanity
    if (iM > cols.n_mtl)
        throw std::invalid_argument("Material index out of bound.");
    Material M(cols, iM);
    double fGHz = (double)center_frequency * 1e-9;
    return (dtype)std::real(std::sqrt(M.eta(fGHz) * M.mu(fGHz)));
}

template float quadriga_lib::refractive_index(const std::unordered_map<std::string, std::vector<float>> &mtl_prop, arma::uword iM, float center_frequency);
template double quadriga_lib::refractive_index(const std::unordered_map<std::string, std::vector<double>> &mtl_prop, arma::uword iM, double center_frequency);

/*!MD
# ray_mesh_interact
Calculates reflection, transmission, or refraction of EM/acoustic waves at mesh surfaces

- One face event per ray: from the first and second hit (FBS, SBS) of [[ray_triangle_intersect]], the interface
  coefficients between homogeneous isotropic media (TE/TM Fresnel; one scalar pressure coefficient for types 3–5,
  the acoustic simulation with impedance-mapped materials), the outgoing direction and the relaunch segment, and
  optionally the ray tube (`trivec`, `tridir`; the output format matches the input).
- Face sides follow the vertex order: counter-clockwise is the front (right-hand rule). A front-side hit with
  FBS ≠ SBS is air-to-medium, a back-side hit medium-to-air; co-located FBS and SBS with opposing normals are a
  medium-to-medium boundary. Overlapping geometry must be avoided.
- The function sees faces only and assumes air on every side the mesh does not name; [[ray_state_update]] knows
  the medium a ray is in and corrects the result.
- Thin sheets: a material with `thk > 0` makes its faces sheets, a face standing for a layer of that thickness.
  A crossing is one event with the internal multiple-reflection series summed in closed form. Two co-located
  sheets of different materials are a stack, traversed in the order their normals give (the layer hit on its
  front side first); two coincident sheets of one material are one sheet (a double-sided export). At most two
  co-located faces form a stack; a third layer is a separate event. A closed body of a sheet material is a
  hollow box. `thk = 0`, or no `thk` column, gives the traced solid.
- Past the critical angle into a lossless medium (a lossy one has none) the face reflects fully where no `tf` is
  set; with `tf` it passes the leak of its pad. With a ray tube and types 2 or 5, a tube with any vertex past the
  critical angle passes undeviated as a whole.
- In-medium attenuation is not applied here ([[ray_state_update]] does), and no spreading loss is applied.
- For a detailed description of the material model see <a href="http://quadriga-lib.org/formats.html">Data Formats</a>

## Declaration:
```
void quadriga_lib::ray_mesh_interact(
    int interaction_type,
    dtype center_frequency,
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *dest,
    const arma::Mat<dtype> *mesh,
    const arma::uvec *mtl_ind,
    const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
    const arma::u32_vec *fbs_ind,
    const arma::u32_vec *sbs_ind,
    const arma::Mat<dtype> *trivec = nullptr,
    const arma::Mat<dtype> *tridir = nullptr,
    arma::Mat<dtype> *origN = nullptr,
    arma::Mat<dtype> *destN = nullptr,
    arma::Mat<dtype> *fbsN = nullptr,
    arma::Mat<dtype> *sbsN = nullptr,
    arma::Col<dtype> *gainN = nullptr,
    arma::Mat<dtype> *xprmatN = nullptr,
    arma::Mat<dtype> *trivecN = nullptr,
    arma::Mat<dtype> *tridirN = nullptr,
    arma::Col<dtype> *fbs_angleN = nullptr,
    arma::Col<dtype> *thicknessN = nullptr,
    arma::Col<dtype> *edge_lengthN = nullptr,
    arma::Mat<dtype> *normal_vecN = nullptr,
    std::vector<uint8_t> *out_typeN = nullptr,
    arma::Mat<dtype> *path_dirN = nullptr,
    bool compact = false,
    arma::u32_vec *ray_indN = nullptr);
```

## Inputs:
- **`interaction_type`** — 0 = EM reflection, 1 = EM transmission, 2 = EM refraction, 3 = scalar reflection, 4 = scalar transmission, 5 = scalar refraction
- **`center_frequency`** — Center frequency in [Hz]
- **`orig`**, **`dest`** — Ray origin and destination in GCS; `[n_ray, 3]`
- **`mesh`** — Triangle mesh faces; see [[obj_file_read]]; `[n_mesh, 9]`
- **`mtl_ind`** — 1-based material index per face (the `csv_ind` output of [[obj_file_read]]); `[n_mesh]`.
  0 = face has no material (air). NULL → all faces treated as air.
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]);
  each value has length `n_mtl`. NULL → air defaults used.
- **`fbs_ind`**, **`sbs_ind`** — 1-based mesh face indices per ray (0 = no hit); `[n_ray]`
- **`trivec`** *(optional)* — Beam wavefront triangle vertices relative to origin; `[n_ray, 9]`, order `[v1x v1y v1z v2x v2y v2z v3x v3y v3z]`
- **`tridir`** *(optional)* — Vertex-ray directions; `[n_ray, 6]` for spherical `[v1az v1el v2az v2el v3az v3el]` or `[n_ray, 9]` for Cartesian
- **`compact`** *(optional)* — If `true`, no-hit rays are dropped and `n_rayN ≤ n_ray`. If
  `false` (default), all rays are kept (`n_rayN = n_ray`) and no-hit rays are written as a transparent pass-through
  (gain 1, identity `xprmat`, `out_type = 0`).

## Outputs:
- **`origN`** — New origins, nudged off the face along the travel direction by 8 float ULP of the hit point's
  coordinate magnitude (past the SBS of a co-located pair, except for reflection); `[n_rayN, 3]`
- **`destN`** — New destinations along the outgoing direction; `[n_rayN, 3]`
- **`fbsN`**, **`sbsN`** — First/second interaction points in GCS; `[n_rayN, 3]`
- **`gainN`** — Linear power gain of the interaction, derived from `xprmatN`; `[n_rayN]`
- **`xprmatN`** — Polarization transfer matrix of amplitude coefficients that carry the interaction gain. Types 0–2:
  interleaved complex, col-major `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]`, the Jones matrix `M = [VV VH; HV HH]`
  acting as `e_out = M · e_in`, the incoming V/H basis from the incident direction and the outgoing one from
  `path_dirN` (`eH = ẑ × k̂` normalized, `eV = eH × k̂`); `[8, n_rayN]`. Types 3–5: `[Re Im]` of the scalar pressure
  coefficient; `[2, n_rayN]`. A transmitted coefficient includes the entered material's `interface_gain`.
- **`trivecN`**, **`tridirN`** — Updated beam geometry/direction (format matches input); empty if inputs not provided
- **`fbs_angleN`** — Incidence angle at FBS in rad, the angle to the face (≥ 0 for a front-side hit); `[n_rayN]`
- **`thicknessN`** — Geometric FBS-to-SBS distance; `[n_rayN]`. Unrelated to the material `thk` column:
  this is what the mesh measured, `thk` is what the material table authored.
- **`edge_lengthN`** — Max edge length of ray tube triangle at new origin (∞ if partial hit); `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normal vectors `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`; `[n_rayN, 6]`
- **`out_typeN`** — Interaction type code, bit-encoded (`qd::bits<uint8_t>`); `[n_rayN]`<br><br>
   |  Bit | Meaning                                                                 |
   | :--: | ----------------------------------------------------------------------- |
   |   0  | OK flag (0 = no valid interaction / undefined)                          |
   |   1  | Front-side flag — meaning depends on bits 2 and 6, see below            |
   |   2  | Co-located FBS/SBS flag (1 = single point, required for media-to-media) |
   |   3  | Same-direction flag (FBS and SBS normals point the same way)            |
   |   4  | Corner-hit flag (FBS/SBS faces not parallel)                            |
   |   5  | Total-reflection flag (also set when a transmission factor forced it)   |
   |   6  | Thin-sheet flag (at least one of FBS/SBS is a `thk > 0` material)       |
   Bit 1 by context:<br><br>
   | Context                                     | Bit 1 = 1 means                               |
   | ------------------------------------------- | --------------------------------------------- |
   | Solid (bit 6 = 0)                           | Front-side hit on FBS (o→i, or M2 hit first)  |
   | Isolated sheet (bit 6 = 1, bit 2 = 0)       | Front-side hit on FBS; informational only     |
   | Sheet glued to a solid (exactly one sheet)  | Front-side hit on the **solid** face          |
   | Two co-located sheets                       | FBS traversed first (0 = SBS traversed first) |
   Reachable composite values (add 32 for the total-reflection variant):<br><br>
   | Code  |  TIR  | Description                                                                        |
   | :---: | :---: | ---------------------------------------------------------------------------------- |
   |   0   |   —   | No hit                                                                             |
   |   1   |  33   | Single hit, inside→outside (exit)                                                  |
   |   3   |  35   | Single hit, outside→inside (entry)                                                 |
   |   5   |  37   | Media-to-media, M1 (current, back) hit first                                       |
   |   7   |  39   | Media-to-media, M2 (next, front) hit first                                         |
   |  13   |  45   | Overlapping faces, inside-inside→outside                                           |
   |  15   |  47   | Overlapping faces, outside→inside-inside                                           |
   |  21   |  53   | Corner hit, inside→outside→inside                                                  |
   |  23   |  55   | Corner hit, outside→inside→outside                                                 |
   |  29   |  61   | Corner hit, inside-inside→outside                                                  |
   |  31   |  63   | Corner hit, outside→inside-inside                                                  |
   |  65   |   —   | Isolated sheet, back-side hit; or a sheet at a corner with SBS dropped             |
   |  67   |   —   | Isolated sheet, front-side hit; or a sheet at a corner with SBS dropped            |
   |  69   |  101  | Glued exit (solid→sheet→air); or a stack with SBS first; or a collapsed duplicate  |
   |  71   |  103  | Glued entry (air→sheet→solid); or a stack with FBS first; or a collapsed duplicate |
   |  79   |   —   | Two-sheet stack with an ill-posed order (geometric FBS→SBS fallback was used)      |
   Codes 69 and 71 each stand for three cases, which [[ray_state_update]] tells apart from the materials of the two
   faces. Total reflection occurs with bit 6 only for a glued pair. Bit 7 is unused.
- **`path_dirN`** — Refraction-correct path direction: mirror for types 0/3, Snell direction for types 1/2/4/5; `[n_rayN, 3]`.
  For undeviated transmission (types 1/4) this is the *refracted* direction, which differs from the geometric continuation
  (along the incoming ray) used for `origN`/`destN`; it lets downstream code recover the true transmission angle.
  Out of a mass-law medium (`m > 0`) there is no bend with any type: the wave is held undeviated at every face.
- **`ray_indN`** — 0-based input ray index for each output ray (inverse of the internal compaction map; order-preserving); `[n_rayN]`

## See also:
- <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-State Machine</a> (companion document)
- [[obj_file_read]] (for loading `mesh` and `mtl_prop` from OBJ file)
- [[ray_state_update]] (inside/outside state machine)
- [[icosphere]] (for generating beams)
- [[ray_triangle_intersect]] (for computing FBS and SBS positions)
- [[ray_point_intersect]] (for calculating beam interactions with sampling points)
MD!*/

template <typename dtype>
void quadriga_lib::ray_mesh_interact(int interaction_type,
                                     dtype center_frequency,
                                     const arma::Mat<dtype> *orig,
                                     const arma::Mat<dtype> *dest,
                                     const arma::Mat<dtype> *mesh,
                                     const arma::uvec *mtl_ind,
                                     const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
                                     const arma::u32_vec *fbs_ind,
                                     const arma::u32_vec *sbs_ind,
                                     const arma::Mat<dtype> *trivec,
                                     const arma::Mat<dtype> *tridir,
                                     arma::Mat<dtype> *origN,
                                     arma::Mat<dtype> *destN,
                                     arma::Mat<dtype> *fbsN,
                                     arma::Mat<dtype> *sbsN,
                                     arma::Col<dtype> *gainN,
                                     arma::Mat<dtype> *xprmatN,
                                     arma::Mat<dtype> *trivecN,
                                     arma::Mat<dtype> *tridirN,
                                     arma::Col<dtype> *fbs_angleN,
                                     arma::Col<dtype> *thicknessN,
                                     arma::Col<dtype> *edge_lengthN,
                                     arma::Mat<dtype> *normal_vecN,
                                     std::vector<uint8_t> *out_typeN,
                                     arma::Mat<dtype> *path_dirN,
                                     bool compact,
                                     arma::u32_vec *ray_indN)
{
    if (interaction_type < 0 || interaction_type > 5)
        throw std::invalid_argument("Interaction type must be either (0) EM Reflection, (1) EM Transmission, (2) EM Refraction, (3) Scalar Reflection, (4) Scalar Transmission, (5) Scalar Refraction");
    bool is_scalar = interaction_type >= 3;
    int geometry_type = interaction_type % 3;

    if (center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");
    double fGHz = (double)center_frequency * 1.0e-9;

    if (orig == nullptr)
        throw std::invalid_argument("Input 'orig' cannot be NULL.");
    if (orig->n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns containing x,y,z coordinates.");
    const arma::uword n_ray = orig->n_rows; // Number of rays
    const dtype *p_orig = orig->memptr();

    if (dest == nullptr)
        throw std::invalid_argument("Input 'dest' cannot be NULL.");
    if (dest->n_cols != 3)
        throw std::invalid_argument("Input 'dest' must have 3 columns containing x,y,z coordinates.");
    if (dest->n_rows != n_ray)
        throw std::invalid_argument("Number of rows in 'orig' and 'dest' dont match.");
    const dtype *p_dest = dest->memptr();

    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (mesh->n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");
    const arma::uword n_mesh = mesh->n_rows; // Number of mesh elements
    const dtype *p_mesh = mesh->memptr();

    if (fbs_ind == nullptr)
        throw std::invalid_argument("Input 'fbs_ind' cannot be NULL.");
    if (fbs_ind->n_elem != n_ray)
        throw std::invalid_argument("Number of elements in 'fbs_ind' does not match number of rows in 'orig'.");
    const unsigned *p_fbs_ind = fbs_ind->memptr();

    if (sbs_ind == nullptr)
        throw std::invalid_argument("Input 'sbs_ind' cannot be NULL.");
    if (sbs_ind->n_elem != n_ray)
        throw std::invalid_argument("Number of elements in 'sbs_ind' does not match number of rows in 'orig'.");

    const unsigned *p_sbs_ind = sbs_ind->memptr();

    if (mtl_ind && !mtl_ind->is_empty() && mtl_ind->n_elem != n_mesh)
        throw std::invalid_argument("Length of 'mtl_ind' must match the number of mesh faces.");

    int use_ray_tube = 0;
    if (trivec && !trivec->is_empty())
    {
        if (tridir == nullptr || tridir->is_empty())
            throw std::invalid_argument("In order to use ray tubes, both 'trivec' and 'tridir' must be given.");
        if (trivec->n_cols != 9)
            throw std::invalid_argument("Input 'trivec' must have 9 columns.");
        if (trivec->n_rows != n_ray)
            throw std::invalid_argument("Number of rows in 'orig' and 'trivec' dont match.");
        if (tridir->n_cols != 6 && tridir->n_cols != 9)
            throw std::invalid_argument("Input 'tridir' must have 6 or 9 columns.");
        if (tridir->n_rows != n_ray)
            throw std::invalid_argument("Number of rows in 'orig' and 'tridir' dont match.");
        use_ray_tube = (tridir->n_cols == 6) ? 1 : 2;
    }
    else if (tridir && !tridir->is_empty())
        throw std::invalid_argument("In order to use ray tubes, both 'trivec' and 'tridir' must be given.");
    const dtype *p_trivec = trivec ? trivec->memptr() : nullptr;
    const dtype *p_tridir = tridir ? tridir->memptr() : nullptr;

    // Resolve material columns once; air (empty) table when no material model is supplied.
    const arma::uword *p_mtl_ind = (mtl_ind == nullptr || mtl_ind->is_empty()) ? nullptr : mtl_ind->memptr();
    MaterialCols<dtype> MaterialSet = (mtl_prop) ? MaterialCols<dtype>(*mtl_prop) : MaterialCols<dtype>();
    if (p_mtl_ind && mtl_ind->max() > MaterialSet.n_mtl)
        throw std::invalid_argument("Entries of 'mtl_ind' cannot exceed the number of materials.");

    // Output ray index. Without compaction the map is the identity and is not stored (at 1e8 rays it would
    // cost 400 MB), so the range check runs in parallel; with compaction the counter is a prefix sum.
    unsigned n_rayN_u = 0;
    arma::u32_vec output_ray_index; // 1-based, empty = identity map

    if (compact)
    {
        output_ray_index.set_size(n_ray); // not zero-filled: every element is written below
        unsigned *p_ind = output_ray_index.memptr();
        for (size_t i_ray = 0; i_ray < n_ray; ++i_ray) // Ray loop
        {
            if (p_fbs_ind[i_ray] > n_mesh) // Invalid, must be 1 ... n_mesh (1-based index)
                throw std::invalid_argument("Some values in 'fbs_ind' exceed number of mesh elements.");
            if (p_sbs_ind[i_ray] > n_mesh)
                throw std::invalid_argument("Some values in 'sbs_ind' exceed number of mesh elements.");

            if (p_fbs_ind[i_ray] == 0)
                p_ind[i_ray] = 0; // drop no-hits
            else
                p_ind[i_ray] = ++n_rayN_u; // keep
        }
    }
    else // Identity map: only the index range is left to check, and that is a reduction
    {
        int bad = 0;
#pragma omp parallel for schedule(static) reduction(| : bad) if (n_ray >= 51200)
        for (long long i_ray = 0; i_ray < (long long)n_ray; ++i_ray) // Ray loop
        {
            if (p_fbs_ind[i_ray] > n_mesh) // Invalid, must be 1 ... n_mesh (1-based index)
                bad |= 1;
            if (p_sbs_ind[i_ray] > n_mesh)
                bad |= 2;
        }

        if (bad & 1)
            throw std::invalid_argument("Some values in 'fbs_ind' exceed number of mesh elements.");
        if (bad & 2)
            throw std::invalid_argument("Some values in 'sbs_ind' exceed number of mesh elements.");

        n_rayN_u = (unsigned)n_ray;
    }
    const arma::uword n_rayN = (arma::uword)n_rayN_u;
    const unsigned *p_ray_map = output_ray_index.empty() ? nullptr : output_ray_index.memptr();

    if (origN && (origN->n_rows != n_rayN || origN->n_cols != 3))
        origN->set_size(n_rayN, 3);
    dtype *p_origN = origN ? origN->memptr() : nullptr;

    if (destN && (destN->n_rows != n_rayN || destN->n_cols != 3))
        destN->set_size(n_rayN, 3);
    dtype *p_destN = destN ? destN->memptr() : nullptr;

    if (fbsN && (fbsN->n_rows != n_rayN || fbsN->n_cols != 3))
        fbsN->set_size(n_rayN, 3);
    dtype *p_fbsN = fbsN ? fbsN->memptr() : nullptr;

    if (sbsN && (sbsN->n_rows != n_rayN || sbsN->n_cols != 3))
        sbsN->set_size(n_rayN, 3);
    dtype *p_sbsN = sbsN ? sbsN->memptr() : nullptr;

    if (gainN && gainN->n_elem != n_rayN)
        gainN->set_size(n_rayN);
    dtype *p_gainN = gainN ? gainN->memptr() : nullptr;

    const arma::uword nXPR = is_scalar ? 2 : 8; // Number of columns in xprmat (8 for EM, 2 for scalar)
    if (xprmatN && (xprmatN->n_rows != nXPR || xprmatN->n_cols != n_rayN))
        xprmatN->set_size(nXPR, n_rayN);
    dtype *p_xprmatN = xprmatN ? xprmatN->memptr() : nullptr;

    if (trivecN && use_ray_tube && (trivecN->n_rows != n_rayN || trivecN->n_cols != 9))
        trivecN->set_size(n_rayN, 9);
    else if (trivecN && !use_ray_tube && !trivecN->is_empty())
        trivecN->reset();
    dtype *p_trivecN = trivecN ? trivecN->memptr() : nullptr;

    if (tridirN && use_ray_tube == 1 && (tridirN->n_rows != n_rayN || tridirN->n_cols != 6))
        tridirN->set_size(n_rayN, 6);
    else if (tridirN && use_ray_tube == 2 && (tridirN->n_rows != n_rayN || tridirN->n_cols != 9))
        tridirN->set_size(n_rayN, 9);
    else if (tridirN && !use_ray_tube && !tridirN->is_empty())
        tridirN->reset();
    dtype *p_tridirN = tridirN ? tridirN->memptr() : nullptr;

    if (fbs_angleN && fbs_angleN->n_elem != n_rayN)
        fbs_angleN->set_size(n_rayN);
    dtype *p_fbs_angleN = fbs_angleN ? fbs_angleN->memptr() : nullptr;

    if (thicknessN && thicknessN->n_elem != n_rayN)
        thicknessN->set_size(n_rayN);
    dtype *p_thicknessN = thicknessN ? thicknessN->memptr() : nullptr;

    if (edge_lengthN && edge_lengthN->n_elem != n_rayN)
        edge_lengthN->set_size(n_rayN);
    dtype *p_edge_lengthN = edge_lengthN ? edge_lengthN->memptr() : nullptr;

    if (normal_vecN && (normal_vecN->n_rows != n_rayN || normal_vecN->n_cols != 6))
        normal_vecN->set_size(n_rayN, 6);
    dtype *p_normal_vecN = normal_vecN ? normal_vecN->memptr() : nullptr;

    if (out_typeN && out_typeN->size() != n_rayN) // out_typeN is a std::vector, not arma::Col
        out_typeN->resize(n_rayN, 0);
    uint8_t *p_out_typeN = out_typeN ? out_typeN->data() : nullptr;

    if (path_dirN && (path_dirN->n_rows != n_rayN || path_dirN->n_cols != 3))
        path_dirN->set_size(n_rayN, 3);
    dtype *p_path_dirN = path_dirN ? path_dirN->memptr() : nullptr;

    if (ray_indN && ray_indN->n_elem != n_rayN)
        ray_indN->set_size(n_rayN);
    unsigned *p_ray_indN = ray_indN ? ray_indN->memptr() : nullptr;

    // Under refraction (geometry_type 2) the per-vertex incidence can escalate to whole-tube TIR below,
    // which changes cTE/cTM, the path direction and the TIR flag. Skip the tube only when no tube
    // output is requested AND the tube cannot affect the center-ray result.
    if (use_ray_tube && geometry_type != 2 && p_trivecN == nullptr && p_tridirN == nullptr && p_edge_lengthN == nullptr)
        use_ray_tube = 0;

    const RmiOut<dtype> out{p_origN, p_destN, p_fbsN, p_sbsN, p_gainN, p_xprmatN, p_trivecN, p_tridirN,
                            p_fbs_angleN, p_thicknessN, p_edge_lengthN, p_normal_vecN, p_path_dirN,
                            p_out_typeN, p_ray_indN, (size_t)nXPR};

#pragma omp parallel for schedule(static)
    for (long long i_ray = 0; i_ray < (long long)n_ray; ++i_ray) // Ray loop
    {
        if (compact && p_fbs_ind[i_ray] == 0) // Compact stream, skip non-hits
            continue;

        const size_t iRx = (size_t)i_ray;                                   // Ray x-index
        const size_t iRy = iRx + n_ray;                                     // Ray y-index
        const size_t iRz = iRy + n_ray;                                     // Ray z-index
        const size_t i_rayN = p_ray_map ? (size_t)p_ray_map[iRx] - 1 : iRx; // Output ray index, 0-based
        const RayWriter<dtype> w{i_rayN, (size_t)n_rayN};

        const double Ox = (double)p_orig[iRx], Oy = (double)p_orig[iRy], Oz = (double)p_orig[iRz]; // Origin position
        double Dx = (double)p_dest[iRx], Dy = (double)p_dest[iRy], Dz = (double)p_dest[iRz];       // Destination position
        double ODx = Dx - Ox, ODy = Dy - Oy, ODz = Dz - Oz;                                        // Ray direction O to D

        // Shift D back 2 ULP to drop paths that end on the FBS face
        double odScale = std::max({std::abs(ODx), std::abs(ODy), std::abs(ODz), 1e-30});
        double posScale = std::max({std::abs(Dx), std::abs(Dy), std::abs(Dz), 1.0});
        double offset = 2.0 * posScale * float_ulp / odScale;
        Dx -= offset * ODx, Dy -= offset * ODy, Dz -= offset * ODz;

        ODx = Dx - Ox, ODy = Dy - Oy, ODz = Dz - Oz; // Update direction O to D
        const double OD_length = normalize_fast(ODx, ODy, ODz);

        const size_t iFBS = (size_t)p_fbs_ind[iRx]; // Mesh FBS index, 1-based
        const size_t iSBS = (size_t)p_sbs_ind[iRx]; // Mesh SBS index, 1-based

        if (iFBS == 0) // no mesh hit
        {
            const double O[3] = {Ox, Oy, Oz}, D[3] = {Dx, Dy, Dz}, OD[3] = {ODx, ODy, ODz};
            rmi_no_hit(out, w, use_ray_tube, p_trivec, p_tridir, iRx, (size_t)n_ray, O, D, OD);
            continue;
        }

        const arma::uword mtl_fbs = p_mtl_ind ? p_mtl_ind[iFBS - 1] : 0;
        const arma::uword mtl_sbs = (p_mtl_ind && iSBS) ? p_mtl_ind[iSBS - 1] : 0;
        const bool sheetFBS = MaterialSet.is_sheet(mtl_fbs);
        const bool sheetSBS = MaterialSet.is_sheet(mtl_sbs);

        // FBS: the hit point F and the face normal N
        double Fx, Fy, Fz, Nx, Ny, Nz;
        double OF_length = 0.0; // Distance from origin to FBS
        face_plane(p_mesh, n_mesh, iFBS, Fx, Fy, Fz, Nx, Ny, Nz);
        double cos_theta = ray_plane_hit(Ox, Oy, Oz, ODx, ODy, ODz, OD_length, Dx, Dy, Dz, Nx, Ny, Nz, 0.0, Ox, Oy, Oz, Fx, Fy, Fz, OF_length);
        cos_theta = (cos_theta < -1.0) ? -1.0 : (cos_theta > 1.0 ? 1.0 : cos_theta); // Boundary fix
        const double theta_fbs = std::acos(cos_theta) - half_pi;                     // Angle between FBS face and incoming ray

        // SBS: the hit point S and the face normal M, at the destination without a second face
        double Sx = Dx, Sy = Dy, Sz = Dz;
        double Mx = ODx, My = ODy, Mz = ODz;
        double theta_sbs = 0.0; // Angle between SBS face and incoming ray
        if (iSBS)
        {
            double OS_length = 0.0;
            face_plane(p_mesh, n_mesh, iSBS, Sx, Sy, Sz, Mx, My, Mz);
            double cos_sbs = ray_plane_hit(Ox, Oy, Oz, ODx, ODy, ODz, OD_length, Dx, Dy, Dz, Mx, My, Mz, OF_length, Fx, Fy, Fz, Sx, Sy, Sz, OS_length);
            cos_sbs = (cos_sbs < -1.0) ? -1.0 : (cos_sbs > 1.0 ? 1.0 : cos_sbs);
            theta_sbs = std::acos(cos_sbs) - half_pi;
        }
        const double FS_length = std::sqrt((Sx - Fx) * (Sx - Fx) + (Sy - Fy) * (Sy - Fy) + (Sz - Fz) * (Sz - Fz));
        const bool colocated_faces = FS_length < colocation_dist && iSBS != 0; // Two colocated faces

        const bool same_sheets = sheetFBS && sheetSBS && Material(MaterialSet, mtl_fbs).same_as(Material(MaterialSet, mtl_sbs));
        const HitClass hit = classify_hit(OF_length != OD_length, theta_fbs, theta_sbs, colocated_faces, Nx, Ny, Nz, Mx, My, Mz,
                                          sheetFBS, sheetSBS, same_sheets);
        qd::bits<uint8_t> hit_code = hit.code;
        const SheetCase sheet_case = hit.sheet;

        // Flip normal vector in case of back side illumination
        // This is pure geometry: it orients N toward the incidence side so that qd_reflect and
        // qd_refract bend the right way. It therefore stays on the FBS test even for a glued sheet
        // whose own normal happens to be reversed relative to the solid it sits on - front_eff decides
        // which material is near and which is far, never which way a normal points.
        if (theta_fbs < 0.0)
            Nx = -Nx, Ny = -Ny, Nz = -Nz,
            cos_theta = ODx * Nx + ODy * Ny + ODz * Nz,
            cos_theta = (cos_theta < -1.0) ? -1.0 : (cos_theta > 1.0 ? 1.0 : cos_theta);

        const double abs_cos_theta = std::abs(cos_theta);
        const HitMedia media = hit_media(MaterialSet, mtl_fbs, mtl_sbs, hit, sheetFBS, theta_fbs);

        // Interface evaluation. For a solid this one call supplies both the coefficients and the
        // geometry. For a sheet the two are split by purpose: the geometry comes from the near/far pair
        // evaluated here, the coefficients from the sheet composite below.
        // The incidence angle here is the geometric one, so the wave is held undeviated by
        // construction; ray_state_update replaces the result wherever a refracted direction exists.
        const Interface face = interact_with(media.near, media.far, interaction_type, theta_fbs, fGHz, nullptr, true);
        std::complex<double> cTE = face.cTE, cTM = face.cTM;
        const std::complex<double> cos_theta2 = face.cos_t2, n2_ratio = face.n2_ratio;
        const double snell = face.snell; // Snell ratio sqrt|eta1*mu1 / eta2*mu2|
        bool total_reflection = face.tir;

        // Direction and total reflection come from near/far, never from near/slab. Snell's law is
        // transitive - n_near*sin_near = n_slab*sin_slab = n_far*sin_far - so the net bend and the
        // critical angle depend only on the two outer media and the sheet cannot affect the outgoing
        // direction at all. Reading them off interact_with(near, sheet) would invent total reflection
        // past the air|sheet critical angle, and in scalar mode the impedance mapping readily gives
        // n < 1, which would turn an acoustic sheet into a mirror.
        if (media.one_medium)
            total_reflection = false;            // near == far: unreachable except at exact tangency, so force it off
        const bool tir_spine = total_reflection; // pre-ray-tube TIR state

        if (media.composite)
        {
            // The composite with air assumed on every side the mesh does not name (ray_state_update redoes it
            // with the tracked media). The sheet lies on the side of a glued pair that is not the solid, the far
            // side only for a glued exit; the medium displaced is air, which is not capped. A lining left from
            // inside a solid with types 2 and 5 is reached along the bent segment: its mass term takes the
            // cosine in air behind that solid.
            const double ct_geo = abs_cos_theta;
            StackOpts opt;
            opt.disp_far = (sheet_case == SheetCase::glued_exit);
            opt.mass_cos = (geometry_type == 2 && sheet_case == SheetCase::glued_exit && media.sheet.m > 0.0 && !(media.near.m > 0.0))
                               ? mass_cos_in_air(media.near, interaction_type, theta_fbs, fGHz, nullptr, ct_geo)
                               : ct_geo;
            opt.near_undeviated = true, opt.far_closed = total_reflection;
            Ports ports; // stays zero for a singular denominator: nothing is handed on
            sheet_ports(media.near, media.sheet, (sheet_case == SheetCase::stack) ? &media.sheet2 : nullptr, media.far,
                        std::complex<double>(ct_geo, 0.0), fGHz, is_scalar, opt, ports);

            // Both ports come out of the one pass over the same Fresnel quantities. The reflection port
            // is its own Airy sum, not what interact_with would return for a half-space: expect real
            // angular and frequency structure instead of a smooth Fresnel value.
            if (geometry_type == 0) // Reflection
                cTE = ports.R_te, cTM = ports.R_tm;
            else // Transmission or refraction
                cTE = ports.T_te, cTM = ports.T_tm;
        }

        double FDx = Dx - Fx, FDy = Dy - Fy, FDz = Dz - Fz; // Vector from FBS to destination
        double FD_length = normalize_fast(FDx, FDy, FDz);   // Length of path from FBS to destination

        // Per-vertex incidence of a tube. Whole-tube TIR: the tube passes undeviated if the spine OR any
        // vertex is past critical; the coefficient stays the spine's (the verdict is the tube's geometry,
        // not the wave's). Skipped where near == far, where the test can only trip on rounding at
        // tangency, and where the far medium is lossy: it has no critical angle.
        TubeIncidence tube_in;
        if (use_ray_tube)
        {
            tube_in = tube_incidence(p_tridir, use_ray_tube, iRx, (size_t)n_ray, Nx, Ny, Nz, abs_cos_theta);
            if (geometry_type == 2 && !media.one_medium &&
                std::imag(media.far.eta(fGHz) * media.far.mu(fGHz)) == 0.0)
                for (int i = 0; i <= 3 && !total_reflection; ++i)
                    if (snell * tube_in.sinV[i] >= 1.0)
                        total_reflection = true;
        }

        // Outgoing spine direction. A sheet with one medium on both sides needs no special case:
        // the interface call returned snell = 1 and cos_theta2 = cos_theta, so the refraction branch
        // reproduces the incoming direction.
        if (geometry_type == 0) // Reflection, normalized by default
            qd_reflect(ODx, ODy, ODz, Nx, Ny, Nz, FDx, FDy, FDz);
        else if (geometry_type == 1 || total_reflection) // Transmission without refraction
            FDx = ODx, FDy = ODy, FDz = ODz;             // New path direction = same as incoming ray, already normalized
        else                                             // Refraction
            qd_refract(ODx, ODy, ODz, Nx, Ny, Nz, snell, abs_cos_theta, cos_theta2, FDx, FDy, FDz);

        double tube_tv[9] = {};
        double tube_td[9] = {};
        double edge_length = 0.0;
        if (use_ray_tube)
            edge_length = tube_update(p_trivec, use_ray_tube, iRx, (size_t)n_ray, tube_in, Ox, Oy, Oz, Fx, Fy, Fz, Nx, Ny, Nz,
                                      geometry_type, total_reflection, snell, n2_ratio, p_edge_lengthN != nullptr, tube_tv, tube_td);

        if (p_ray_indN)
            p_ray_indN[i_rayN] = (unsigned)iRx;

        const double relaunch_offset = qd_relaunch_offset(Fx, Fy, Fz); // ~8 float ULP at this magnitude
        if (geometry_type != 0 && colocated_faces)                     // Relaunch @ SBS
            w.set3(p_origN, Sx + relaunch_offset * FDx, Sy + relaunch_offset * FDy, Sz + relaunch_offset * FDz);
        else // Relaunch @ FBS
            w.set3(p_origN, Fx + relaunch_offset * FDx, Fy + relaunch_offset * FDy, Fz + relaunch_offset * FDz);

        const double min_seg = std::max(colocation_dist, 2.0 * relaunch_offset);
        FD_length = (FD_length < min_seg) ? min_seg : FD_length;
        w.set3(p_destN, Fx + FD_length * FDx, Fy + FD_length * FDy, Fz + FD_length * FDz);

        w.set3(p_fbsN, Fx, Fy, Fz);
        w.set3(p_sbsN, Sx, Sy, Sz);

        // Write path_dirN: spine output direction (FD) for reflection/refraction (geometry 0/2, follows a
        // forced-TIR pass-through); spine Snell for undeviated transmission (geometry 1), or undeviated
        // when the spine is in TIR (no Snell direction; any TF leak travels undeviated).
        if (p_path_dirN)
        {
            double PDx = FDx, PDy = FDy, PDz = FDz;
            if (geometry_type == 1 && !tir_spine) // spine Snell; under spine TIR there is no Snell direction
                qd_refract(ODx, ODy, ODz, Nx, Ny, Nz, snell, abs_cos_theta, cos_theta2, PDx, PDy, PDz);
            w.set3(p_path_dirN, PDx, PDy, PDz);
        }

        if (p_xprmatN || p_gainN)
        {
            double xprmat[8], pgain;
            qd_polbasis(ODx, ODy, ODz, FDx, FDy, FDz, Nx, Ny, Nz, cTE, cTM, is_scalar, xprmat, pgain);
            w.set_col(p_xprmatN, xprmat, nXPR);
            w.set1(p_gainN, pgain);
        }

        if (use_ray_tube)
            w.set_row(p_trivecN, tube_tv, 9);
        w.set_row(p_tridirN, tube_td, (use_ray_tube == 1) ? 6 : 9);

        w.set1(p_fbs_angleN, theta_fbs);
        w.set1(p_thicknessN, FS_length);
        w.set1(p_edge_lengthN, edge_length);

        if (p_normal_vecN)
            w.set3(p_normal_vecN, Nx, Ny, Nz), w.set3(&p_normal_vecN[3 * n_rayN], Mx, My, Mz);

        if (p_out_typeN)
        {
            hit_code[5] = total_reflection; // Set total reflection flag
            p_out_typeN[i_rayN] = (uint8_t)hit_code;
        }
    }
}

template void quadriga_lib::ray_mesh_interact(int interaction_type, float center_frequency,
                                              const arma::Mat<float> *orig, const arma::Mat<float> *dest,
                                              const arma::Mat<float> *mesh, const arma::uvec *mtl_ind,
                                              const std::unordered_map<std::string, std::vector<float>> *mtl_prop,
                                              const arma::u32_vec *fbs_ind, const arma::u32_vec *sbs_ind,
                                              const arma::Mat<float> *trivec, const arma::Mat<float> *tridir,
                                              arma::Mat<float> *origN, arma::Mat<float> *destN,
                                              arma::Mat<float> *fbsN, arma::Mat<float> *sbsN,
                                              arma::Col<float> *gainN, arma::Mat<float> *xprmatN,
                                              arma::Mat<float> *trivecN, arma::Mat<float> *tridirN,
                                              arma::Col<float> *fbs_angleN, arma::Col<float> *thicknessN, arma::Col<float> *edge_lengthN,
                                              arma::Mat<float> *normal_vecN, std::vector<uint8_t> *out_typeN,
                                              arma::Mat<float> *path_dirN, bool compact, arma::u32_vec *ray_indN);

template void quadriga_lib::ray_mesh_interact(int interaction_type, double center_frequency,
                                              const arma::Mat<double> *orig, const arma::Mat<double> *dest,
                                              const arma::Mat<double> *mesh, const arma::uvec *mtl_ind,
                                              const std::unordered_map<std::string, std::vector<double>> *mtl_prop,
                                              const arma::u32_vec *fbs_ind, const arma::u32_vec *sbs_ind,
                                              const arma::Mat<double> *trivec, const arma::Mat<double> *tridir,
                                              arma::Mat<double> *origN, arma::Mat<double> *destN,
                                              arma::Mat<double> *fbsN, arma::Mat<double> *sbsN,
                                              arma::Col<double> *gainN, arma::Mat<double> *xprmatN,
                                              arma::Mat<double> *trivecN, arma::Mat<double> *tridirN,
                                              arma::Col<double> *fbs_angleN, arma::Col<double> *thicknessN, arma::Col<double> *edge_lengthN,
                                              arma::Mat<double> *normal_vecN, std::vector<uint8_t> *out_typeN,
                                              arma::Mat<double> *path_dirN, bool compact, arma::u32_vec *ray_indN);

/*!MD
# ray_state_update
Per-ray medium state machine with closed-form thin-slab (Fabry-Pérot) resolution

- Corrects the per-event `gainN` / `xprmatN` of [[ray_mesh_interact]] with the medium each ray is known to be in,
  and carries that state from event to event in three state words per ray (table below) and the in-medium path
  lengths (`acc_dist_*`):
  - the face is re-evaluated at the incidence of the direction the wave has (`path_dir_prev`) instead of the
    traced segment (the virtual hit point, VBS), with total reflection and the continuation direction;
  - the in-medium attenuation and excess phase of the leg the ray leaves are applied;
  - the internal bounce series of a traced parallel layer is summed in closed form, `S = 1/(1 − r_near·r_far·φ²)`
    per polarization, where its round-trip amplitude `ρ = |r_near·r_far|·|φ|²` is at least `eps`; otherwise the
    bounces are left to the tracer as further events. A body's verdict is taken once, at the first event that
    closes a leg in it;
  - a thin-sheet crossing is re-resolved against the media the state reports (a sheet inside a medium has that
    medium on both sides; [[ray_mesh_interact]] assumed air).
- Called twice per event by the tracer ([[ray_progress]], [[calc_diffraction_gain]]): for the reflect pass
  (`interaction_type` 0 or 3) and for the transmit pass (1, 2, 4, 5), each with the outputs of
  [[ray_mesh_interact]] for that type.
- Without state words every ray starts in air and each event is corrected on its own.

## Declaration:
```
void quadriga_lib::ray_state_update(
    int interaction_type,
    dtype center_frequency,
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *dest,
    const arma::Mat<dtype> *fbsN,
    const arma::Mat<dtype> *sbsN,
    const arma::u32_vec *no_interact,
    const arma::Col<dtype> *fbs_angleN,
    const arma::Mat<dtype> *normal_vecN,
    const std::vector<uint8_t> *out_typeN,
    const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
    const arma::Col<short> *mtl_ind_fbsN,
    const arma::Col<short> *mtl_ind_sbsN,
    const arma::Col<short> *mtl_ind_prev_in = nullptr,
    const arma::Col<short> *mtl_ind_current_in = nullptr,
    const arma::Col<short> *mtl_ind_buffer_in = nullptr,
    const arma::Mat<dtype> *path_dir_prev = nullptr,
    const arma::Col<dtype> *acc_dist_geometric = nullptr,
    const arma::Col<dtype> *acc_dist_refracted = nullptr,
    arma::Col<short> *mtl_ind_prev_outN = nullptr,
    arma::Col<short> *mtl_ind_current_outN = nullptr,
    arma::Col<short> *mtl_ind_buffer_outN = nullptr,
    arma::Col<dtype> *gainN = nullptr,
    arma::Mat<dtype> *xprmatN = nullptr,
    arma::Mat<dtype> *path_dirN = nullptr,
    arma::Col<dtype> *acc_dist_geometricN = nullptr,
    arma::Col<dtype> *acc_dist_refractedN = nullptr,
    std::vector<uint8_t> *resolved_typeN = nullptr,
    const arma::u32_vec *ray_indN = nullptr,
    double eps = 0.15,
    arma::Mat<dtype> *origN = nullptr,
    arma::Mat<dtype> *destN = nullptr,
    const arma::Mat<dtype> *tridir = nullptr,
    arma::Mat<dtype> *tridirN = nullptr,
    const std::vector<uint8_t> *verdict_inN = nullptr,
    std::vector<uint8_t> *verdict_outN = nullptr);
```

## Inputs:
- **`interaction_type`** — 0 EM reflection, 1 EM transmission, 2 EM refraction, 3 scalar reflection, 4 scalar transmission, 5 scalar refraction
- **`center_frequency`** — Center frequency in [Hz]
- **`orig`**, **`dest`** — Ray origin, destination, full ray set; `[n_ray, 3]`, read at `g = ray_indN[i]`
- **`fbsN`**, **`sbsN`**  — First and second interaction points, compact set; `[n_rayN, 3]`
- **`no_interact`** — Mesh-hit count per ray, full ray set; `[n_ray]`
- **`fbs_angleN`** — Incidence angle at FBS (the angle to the face, as [[ray_mesh_interact]] writes it), compact set; `[n_rayN]`
- **`normal_vecN`** — FBS and SBS normals `[Nx_F Ny_F Nz_F Nx_S Ny_S Nz_S]`, compact set; `[n_rayN, 6]`; required.
  The plane of the VBS, and the test whether the two faces of an entry are parallel.
- **`out_typeN`** — Interaction type code from [[ray_mesh_interact]], compact set; `[n_rayN]`. Its total-reflection
  bit is re-evaluated at the VBS angle; a thin-sheet code takes the sheet path.
- **`mtl_prop`** — Material properties keyed by column name (the `csv_prop` output of [[obj_file_read]]); at most
  16383 materials.
- **`mtl_ind_fbsN`**, **`mtl_ind_sbsN`** — Material indices of the FBS / SBS faces, compact set; `[n_rayN]` (0 = air).
  The sheet case of a hit is re-derived from the code and the two faces' materials.
- **`mtl_ind_prev_in`**, **`mtl_ind_current_in`**, **`mtl_ind_buffer_in`** — State words, full ray set; `[n_ray]`,
  read at `g`, never written. NULL reads as state `0` (in air, no flags). Pass the words an earlier call wrote
  through unchanged.<br><br>
   | Word                 | Bits 0-13                          | Bit 14                                        | Bit 15                    |
   | -------------------- | ---------------------------------- | --------------------------------------------- | ------------------------- |
   | `mtl_ind_current`    | medium the ray is in, 0 = air      | bent: the leg follows a bend traced with type 2 or 5; zero in air | the leg's series was summed |
   | `mtl_ind_prev`       | medium the leg was entered from, or the sheet crossed or reflected off | the sheet in this word lies inside the current medium | the leg's faces are not parallel |
   | `mtl_ind_buffer`     | deferred medium                    | followed, with a zero index: the leg began at a reflection that left the series to the tracer. With an index: the layer of an ordered stack behind the ray that the previous-medium word does not name | clear |
   | `mtl_ind_buffer`     | thickness code, bits 0-14          | (part of the code)                            | set: mass-crossed         |
   Under bit 15 the buffer word is the normal thickness, as a logarithmic code, of a mass-law leg closed on the way
   into the current one (bonded mass-law bodies are one plate); it is exempt from the material range check.
- **`path_dir_prev`** — Physical ray direction entering this segment, full ray set; `[n_ray, 3]`
- **`acc_dist_geometric`** — Accumulated geometric in-layer distance carried into this call, full ray set; `[n_ray]`.
  NULL reads as 0. Frequency-independent, so a multi-frequency caller keeps one instance.
- **`acc_dist_refracted`** — Accumulated refracted (VBS) in-layer distance carried into this call, full ray set; `[n_ray]`.
  NULL reads as 0. Frequency-dependent (the bend follows the material at `center_frequency`), so a
  multi-frequency caller keeps one instance per frequency.
- **`ray_indN`** — Compact-to-full ray index map; `[n_rayN]` to `[n_ray]`; NULL = identity (`n_ray == n_rayN`)
- **`eps`** *(optional)* — Threshold on the round-trip amplitude `ρ` (`ρ ≥ 0`, not bounded by one: next to a lossy or
  evanescent layer a face can reflect more than one in amplitude): a series is summed where `ρ ≥ eps` and left to
  the tracer where `ρ < eps`. `eps = 0` always sums (for callers that cannot re-emit); `eps ≥ 1` leaves every series
  with `ρ < eps` to the tracer. A layer whose faces are known not to be parallel is left to the tracer for any
  `eps > 0`. A thin sheet's own series is always summed (there is no interior to trace); `eps` gates the host layer
  a sheet lies in. Default `0.15`.
- **`verdict_inN`** *(optional)* — The verdict of each event as an earlier call on the same events wrote it to
  `verdict_outN`, compact set; `[n_rayN]`. Where given, a series is summed exactly where the entry is nonzero and
  `eps` is not read: a multi-frequency caller hands the base frequency's verdicts to the other frequencies, whose
  traced paths are the base frequency's. NULL: this call decides by `eps`.
- **`tridir`** *(optional)* — Vertex-ray directions of the beam entering this segment, Cartesian, as handed to
  [[ray_mesh_interact]]; `[n_ray, 9]`. Given together with `tridirN`; requires `origN` and `destN`

## Outputs:
- **`mtl_ind_prev_outN`**, **`mtl_ind_current_outN`**, **`mtl_ind_buffer_outN`** — Updated state words (layout
  above), compact set; `[n_rayN]`. The previous-medium word may hold a sheet: the boundary behind the ray, never a
  medium it travels in. NULL skips the write.
- **`gainN`** *(in/out)* — Per-interaction linear power gain, updated in place; `[n_rayN]`. Kept consistent
  with `xprmatN`. A killed ray (`resolved_typeN == 0`) has `gainN = 0` and an all-zero `xprmatN`.
- **`xprmatN`** *(in/out)* — Polarization transfer matrix, updated in place; `[8, n_rayN]` for EM mode,
  `[2, n_rayN]` for scalar mode, layout and basis as in [[ray_mesh_interact]]. On return it also carries the
  in-medium attenuation and excess phase of the leg the ray leaves and, where the series was summed, the factor
  `S`. Spreading loss stays excluded, so the matrix is the complete per-event Jones factor. Where the series is
  left to the tracer the matrix carries the first pass only, and the bounces arrive as further events. At a
  thin-sheet crossing the matrix is replaced by the composite for the media the state reports.
- **`path_dirN`** *(in/out)* — Continuation direction, corrected in place by the VBS construction, compact set; `[n_rayN, 3]`
- **`origN`**, **`destN`** *(optional, in/out)* — The relaunch segment [[ray_mesh_interact]] built, compact set;
  `[n_rayN, 3]`, given together. With a refracting type (2, 5) that segment was bent at a real angle in the
  medium the ray leaves. Where this function writes the direction of a wave out of a lossy medium whose
  in-medium cosine it recovered (a bare exit, a glued exit, a transition into another solid), both points
  are put on one line along `path_dirN`, through the SBS of a co-located pair and the FBS otherwise. Every
  other ray, and every other type, is left untouched. A multi-frequency caller passes them on the call that
  defines the geometry only.
- **`tridirN`** *(optional, in/out)* — Vertex-ray directions leaving, Cartesian, compact set; `[n_rayN, 9]`.
  Corrected together with `origN` and `destN`: each vertex ray gets the construction of the spine from its
  own direction in `tridir`. If one of the three cannot be formed, the ray keeps the geometry it had.
- **`verdict_outN`** *(optional)* — Per event, 1 where this call summed a bounce series and 0 where it left the
  bounces to the tracer, compact set; `[n_rayN]`. NULL skips the write.
- **`acc_dist_geometricN`** — Accumulated geometric distance leaving this call, compact set; `[n_rayN]`. NULL skips the write.
  Where the ray goes on inside a medium it includes the relaunch offset of [[ray_mesh_interact]] (`origN` lies that
  far past the point the ray left at), so the next leg, measured from `origN`, adds up to the leg from the hit point.
- **`acc_dist_refractedN`** — Accumulated refracted (VBS) distance leaving this call, compact set; `[n_rayN]`. NULL skips the write.
  It includes the relaunch offset the same way, projected onto the direction the wave has (exact for parallel faces).
- **`resolved_typeN`** *(optional)* — Resolved interaction-type code, bit-encoded (`qd::bits<uint8_t>`),
  compact set; `[n_rayN]`. 0 = ray killed. NULL skips the write.<br><br>
   | Bit  | Flag        | Meaning                                                                        |
   | :--: | :---------: | ------------------------------------------------------------------------------ |
   |   0  | ok          | OK flag (0 = a deferred medium is pending in the buffer word)                  |
   |   1  | vbs         | VBS correction (gain/xprmat corrected at the VBS instead of FBS/SBS)           |
   |   2  | resolve     | Internal multi-bounce series accounted for (expect no further internal bounces) |
   |   3  | inside      | Inside-object flag (1 = ray continues inside, 0 = continues outside)           |
   |   4  | fix         | Fix flag (the exit face names another medium than the state, or a sheet event met a deferred medium) |
   |   5  | tir         | Total-reflection flag (also set when a transmission factor forced reflection)  |
   |   6  | trans/sheet | See the qualification below                                                    |
   |   7  | refl        | Reflection flag (0 = transmission/refraction, 1 = reflection/scattering)       |
   Bit 6 is qualified by bit 2 (a transparent interface sums no series, and scattering is not implemented):<br><br>
   | Bit 6 | Bit 2 | Meaning                                                                       |
   | :---: | :---: | ----------------------------------------------------------------------------- |
   |   1   |   0   | Transparent interface (transmission) / scatter (reflection)                   |
   |   1   |   1   | **Thin-sheet crossing**                                                       |
   In the thin-sheet region bit 1 carries the glued verdict instead of the VBS flag (a sheet is always
   re-resolved at the VBS angle):<br><br>
   | Bit 1 (sheet region) | Meaning                                                                |
   | :------------------: | ---------------------------------------------------------------------- |
   |          0           | Isolated — the same medium bounds both sides of the sheet              |
   |          1           | Glued — the two bounding media differ                                  |
   The sheet region reports the resolved state, not the sheet topology. The codes 69 and 71 below are unrelated
   to the `out_typeN` codes of the same value.
   Reachable composite values for transmission / refraction:<br><br>
   |  Dec | Hex  | FIX  |   TIR   | Flags set                  | Meaning                                            |
   | :--: | :--: | :--: | :-----: | -------------------------- | -------------------------------------------------- |
   |    9 | 0x09 |   —  |    41   | inside, ok                 | o-i entry, OR i-i transition (refr. 2/5, FBS==VBS) |
   |    8 | 0x08 |   —  |    40   | inside                     | o-i entry, deferred buffer set (overlap/edge)      |
   |   11 | 0x0B |  27  |  43, 59 | inside, vbs, ok            | i-i transition (undev. 1/4, VBS relocated)         |
   |   13 | 0x0D |  29  |  45, 61 | inside, resolve, ok        | i-i transition + slab series (refr. 2/5, FBS==VBS) |
   |   15 | 0x0F |  31  |  47, 63 | inside, resolve, vbs, ok   | i-i transition + slab series (undev. 1/4, VBS)     |
   |    1 | 0x01 |  17  |  33, 49 | ok                         | i-o exit (refr. 2/5, FBS==VBS)                     |
   |    3 | 0x03 |  19  |  35, 51 | vbs, ok                    | i-o exit (undev. 1/4, VBS relocated)               |
   |    5 | 0x05 |  21  |  37, 53 | resolve, ok                | i-o exit + slab series (refr. 2/5, FBS==VBS)       |
   |    7 | 0x07 |  23  |  39, 55 | resolve, vbs, ok           | i-o exit + slab series (undev. 1/4, VBS)           |
   |   73 | 0x49 |  89  |     —   | trans, inside, ok          | ignore-hit / same-medium pass, no gain change      |
   |   72 | 0x48 |  88  |     —   | trans, inside              | nested pass-through, buffer deferred               |
   |   65 | 0x41 |   —  |     —   | trans, ok                  | advance to ray destination, identity interface     |
   |   69 | 0x45 |   85 |   101   | sheet, resolve, ok         | isolated sheet crossed in air                      |
   |   77 | 0x4D |   93 |   109   | sheet, resolve, inside, ok | isolated sheet crossed inside a medium (embedded)  |
   |   71 | 0x47 |   87 |   103   | sheet, glued, resolve, ok  | glued exit, solid \| sheet \| air                  |
   |   79 | 0x4F |   95 |   111   | sheet, glued, resolve, inside, ok | glued entry, host \| sheet \| solid         |
   Reachable composite values for reflection:<br><br>
   |  Dec | Hex  |  FIX |   TIR   | Flags set                       | Meaning                                                      |
   | :--: | :--: | :--: | :-----: | ------------------------------- |------------------------------------------------------------- |
   |  129 | 0x81 |    — |   161   | refl, ok                        | eager front reflection (R0), outside (FBS==VBS)              |
   |  137 | 0x89 |  153 | 169,185 | refl, inside, ok                | internal back-reflection (incoming refr. 2/5, FBS==VBS)      |
   |  139 | 0x8B |  155 | 171,187 | refl, inside, vbs, ok           | internal back-reflection (incoming undev. 1/4, VBS)          |
   |  141 | 0x8D |  157 | 173,189 | refl, inside, resolve, ok       | internal back-reflection + slab series (incoming refr. 2/5)  |
   |  143 | 0x8F |  159 | 175,191 | refl, inside, resolve, vbs, ok  | internal back-reflection + slab series (incoming undev. 1/4) |
   |  197 | 0xC5 |  213 |   229   | refl, sheet, resolve, ok        | reflection off an isolated sheet, ray outside                |
   |  205 | 0xCD |  221 |   237   | refl, sheet, resolve, inside, ok | reflection off an isolated sheet, ray inside a medium       |
   |  199 | 0xC7 |  215 |   231   | refl, sheet, glued, resolve, ok | reflection off a glued sheet from the host side              |
   |  207 | 0xCF |  223 |   239   | refl, sheet, glued, resolve, inside, ok | reflection off a glued sheet from the solid side    |
   |  192+| 0xC0+|    — |    —    | refl, scatter, resolve clear    | reserved: scattering not implemented                         |
   A stack of two different sheets is crossed as one event with the codes of a sheet; a singular stack kills the ray.

## See also:
- <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-State Machine</a> (companion document)
- [[ray_mesh_interact]] (computes the per-interaction Fresnel/Jones result this function corrects)
- [[calc_diffraction_gain]] (uses this function for its media)
- [[ray_progress]] (the tracer that calls it)
MD!*/

template <typename dtype>
void quadriga_lib::ray_state_update(int interaction_type,
                                    dtype center_frequency,
                                    const arma::Mat<dtype> *orig,
                                    const arma::Mat<dtype> *dest,
                                    const arma::Mat<dtype> *fbsN,
                                    const arma::Mat<dtype> *sbsN,
                                    const arma::u32_vec *no_interact,
                                    const arma::Col<dtype> *fbs_angleN,
                                    const arma::Mat<dtype> *normal_vecN,
                                    const std::vector<uint8_t> *out_typeN,
                                    const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
                                    const arma::Col<short> *mtl_ind_fbsN,
                                    const arma::Col<short> *mtl_ind_sbsN,
                                    const arma::Col<short> *mtl_ind_prev_in,
                                    const arma::Col<short> *mtl_ind_current_in,
                                    const arma::Col<short> *mtl_ind_buffer_in,
                                    const arma::Mat<dtype> *path_dir_prev,
                                    const arma::Col<dtype> *acc_dist_geometric,
                                    const arma::Col<dtype> *acc_dist_refracted,
                                    arma::Col<short> *mtl_ind_prev_outN,
                                    arma::Col<short> *mtl_ind_current_outN,
                                    arma::Col<short> *mtl_ind_buffer_outN,
                                    arma::Col<dtype> *gainN,
                                    arma::Mat<dtype> *xprmatN,
                                    arma::Mat<dtype> *path_dirN,
                                    arma::Col<dtype> *acc_dist_geometricN,
                                    arma::Col<dtype> *acc_dist_refractedN,
                                    std::vector<uint8_t> *resolved_typeN,
                                    const arma::u32_vec *ray_indN,
                                    double eps,
                                    arma::Mat<dtype> *origN,
                                    arma::Mat<dtype> *destN,
                                    const arma::Mat<dtype> *tridir,
                                    arma::Mat<dtype> *tridirN,
                                    const std::vector<uint8_t> *verdict_inN,
                                    std::vector<uint8_t> *verdict_outN)
{
    if (interaction_type < 0 || interaction_type > 5)
        throw std::invalid_argument("Interaction type must be either (0) EM Reflection, (1) EM Transmission, (2) EM Refraction, (3) Scalar Reflection, (4) Scalar Transmission, (5) Scalar Refraction");
    const bool is_scalar = interaction_type >= 3;
    const bool refl_pass = (interaction_type == 0 || interaction_type == 3); // geometry 0
    const arma::uword nXPR = is_scalar ? 2 : 8;                              // Number of columns in xprmat (8 for EM, 2 for scalar)

    if (!std::isfinite((double)center_frequency) || center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Center frequency must be provided in Hertz and have values > 0.");

    if (!std::isfinite(eps) || eps < 0.0)
        throw std::invalid_argument("Input 'eps' must be finite and >= 0.");

    if (orig == nullptr || dest == nullptr || fbsN == nullptr || sbsN == nullptr)
        throw std::invalid_argument("Inputs 'orig', 'dest', 'fbsN' and 'sbsN' cannot be NULL.");
    if (orig->n_cols != 3 || dest->n_cols != 3 || fbsN->n_cols != 3 || sbsN->n_cols != 3)
        throw std::invalid_argument("Inputs 'orig', 'dest', 'fbsN' and 'sbsN' must have 3 columns.");
    const arma::uword n_ray = orig->n_rows;
    const arma::uword n_rayN = fbsN->n_rows;

    if (dest->n_rows != n_ray)
        throw std::invalid_argument("Inputs 'orig' and 'dest' must have the same number of rows.");
    if (sbsN->n_rows != n_rayN)
        throw std::invalid_argument("Inputs 'fbs' and 'sbs' must have the same number of rows.");

    if (no_interact == nullptr || no_interact->n_elem != n_ray)
        throw std::invalid_argument("Input 'no_interact' must match the number of rays in 'orig'.");
    const unsigned *p_no_interact = no_interact->memptr();

    if (fbs_angleN == nullptr || fbs_angleN->n_elem != n_rayN)
        throw std::invalid_argument("Input 'fbs_angleN' cannot be NULL and have length n_rayN.");
    const dtype *p_fbs_angleN = fbs_angleN->memptr();

    if (normal_vecN == nullptr)
        throw std::invalid_argument("Input 'normal_vecN' is required (VBS plane normal).");
    if (normal_vecN->n_rows != n_rayN || normal_vecN->n_cols != 6)
        throw std::invalid_argument("Input 'normal_vecN' must have size [n_rayN, 6].");
    const dtype *p_normal_vecN = normal_vecN->memptr();

    if (out_typeN == nullptr || out_typeN->size() != n_rayN)
        throw std::invalid_argument("Input 'out_typeN' cannot be NULL and have length n_rayN.");
    const uint8_t *p_out_typeN = out_typeN->data();

    MaterialCols<dtype> cols = mtl_prop ? MaterialCols<dtype>(*mtl_prop) : MaterialCols<dtype>();
    if (cols.n_mtl > 16383) // the state words carry a 14-bit material index
        throw std::invalid_argument("Number of materials cannot exceed 16383.");

    if (mtl_ind_fbsN == nullptr || mtl_ind_fbsN->n_elem != n_rayN)
        throw std::invalid_argument("Input 'mtl_ind_fbs' must match the length of 'out_typeN'.");
    const short *p_mtl_ind_fbs = mtl_ind_fbsN->memptr();
    if (mtl_ind_fbsN->n_elem != 0 && arma::uword(mtl_ind_fbsN->max() & (short)0x7FFF) > cols.n_mtl)
        throw std::invalid_argument("Values in 'mtl_ind_fbs' exceed the number of materials in 'mtl_prop'.");

    if (mtl_ind_sbsN == nullptr || mtl_ind_sbsN->n_elem != n_rayN)
        throw std::invalid_argument("Input 'mtl_ind_sbs' must match the length of 'out_typeN'.");
    const short *p_mtl_ind_sbs = mtl_ind_sbsN->memptr();
    if (mtl_ind_sbsN->n_elem != 0 && arma::uword(mtl_ind_sbsN->max() & (short)0x7FFF) > cols.n_mtl)
        throw std::invalid_argument("Values in 'mtl_ind_sbs' exceed the number of materials in 'mtl_prop'.");

    // The buffer word under its flag bit carries a surface-mass code, not a material index.
    // Without it, bit 14 is the followed mark, or with an index the other layer of a stack behind the ray.
    auto check_state_words = [&](const arma::Col<short> *v, const char *name, bool flag_is_code = false)
    {
        if (v == nullptr || v->n_elem == 0)
            return;
        for (const short *p = v->memptr(), *pe = p + v->n_elem; p < pe; ++p)
            if (!(flag_is_code && (*p & buf_mass_code)) && arma::uword(*p & word_mtl) > cols.n_mtl)
                throw std::invalid_argument(std::string("Values in '") + name + "' exceed the number of materials in 'mtl_prop'.");
    };
    check_state_words(mtl_ind_prev_in, "mtl_ind_prev_in");
    check_state_words(mtl_ind_current_in, "mtl_ind_current_in");
    check_state_words(mtl_ind_buffer_in, "mtl_ind_buffer_in", true);

    if (mtl_ind_prev_in && mtl_ind_prev_in->n_elem != n_ray)
        throw std::invalid_argument("Input 'mtl_ind_prev_in' must match the number of rays in 'orig'.");
    const short *p_prev_in = mtl_ind_prev_in ? mtl_ind_prev_in->memptr() : nullptr;

    if (mtl_ind_current_in && mtl_ind_current_in->n_elem != n_ray)
        throw std::invalid_argument("Input 'mtl_ind_current_in' must match the number of rays in 'orig'.");
    const short *p_cur_in = mtl_ind_current_in ? mtl_ind_current_in->memptr() : nullptr;

    if (mtl_ind_buffer_in && mtl_ind_buffer_in->n_elem != n_ray)
        throw std::invalid_argument("Input 'mtl_ind_buffer_in' must match the number of rays in 'orig'.");
    const short *p_buf_in = mtl_ind_buffer_in ? mtl_ind_buffer_in->memptr() : nullptr;

    if (path_dir_prev && (path_dir_prev->n_rows != n_ray || path_dir_prev->n_cols != 3))
        throw std::invalid_argument("Input 'path_dir_prev' must have size [n_ray, 3].");
    const dtype *p_path_dir_prev = path_dir_prev ? path_dir_prev->memptr() : nullptr;

    if (acc_dist_geometric && acc_dist_geometric->n_elem != n_ray)
        throw std::invalid_argument("Input 'acc_dist_geometric' must match the number of rays in 'orig'.");
    const dtype *p_acc_geo_in = acc_dist_geometric ? acc_dist_geometric->memptr() : nullptr;

    if (acc_dist_refracted && acc_dist_refracted->n_elem != n_ray)
        throw std::invalid_argument("Input 'acc_dist_refracted' must match the number of rays in 'orig'.");
    const dtype *p_acc_refract_in = acc_dist_refracted ? acc_dist_refracted->memptr() : nullptr;

    if (mtl_ind_prev_outN && mtl_ind_prev_outN->n_elem != n_rayN)
        mtl_ind_prev_outN->set_size(n_rayN);
    short *p_prev_outN = mtl_ind_prev_outN ? mtl_ind_prev_outN->memptr() : nullptr;

    if (mtl_ind_current_outN && mtl_ind_current_outN->n_elem != n_rayN)
        mtl_ind_current_outN->set_size(n_rayN);
    short *p_cur_outN = mtl_ind_current_outN ? mtl_ind_current_outN->memptr() : nullptr;

    if (mtl_ind_buffer_outN && mtl_ind_buffer_outN->n_elem != n_rayN)
        mtl_ind_buffer_outN->set_size(n_rayN);
    short *p_buf_outN = mtl_ind_buffer_outN ? mtl_ind_buffer_outN->memptr() : nullptr;

    if (gainN && gainN->n_elem != n_rayN)
        throw std::invalid_argument("In-out 'gainN' must match the length of 'out_typeN'.");
    dtype *p_gainN = gainN ? gainN->memptr() : nullptr;

    if (xprmatN && (xprmatN->n_rows != nXPR || xprmatN->n_cols != n_rayN))
        throw std::invalid_argument("In-out 'xprmatN' must have size [8, n_rayN] for EM mode or [2, n_rayN] for scalar mode.");
    dtype *p_xprmatN = xprmatN ? xprmatN->memptr() : nullptr;

    if (path_dirN && (path_dirN->n_rows != n_rayN || path_dirN->n_cols != 3))
        throw std::invalid_argument("In-out 'path_dirN' must have size [n_rayN, 3].");
    dtype *p_path_dirN = path_dirN ? path_dirN->memptr() : nullptr;

    if (acc_dist_geometricN && acc_dist_geometricN->n_elem != n_rayN)
        acc_dist_geometricN->set_size(n_rayN);
    dtype *p_acc_geo_outN = acc_dist_geometricN ? acc_dist_geometricN->memptr() : nullptr;

    if (acc_dist_refractedN && acc_dist_refractedN->n_elem != n_rayN)
        acc_dist_refractedN->set_size(n_rayN);
    dtype *p_acc_refract_outN = acc_dist_refractedN ? acc_dist_refractedN->memptr() : nullptr;

    if (resolved_typeN && resolved_typeN->size() != n_rayN)
        resolved_typeN->resize(n_rayN, 0); // Init to 0
    uint8_t *p_resolved_typeN = resolved_typeN ? resolved_typeN->data() : nullptr;

    if (ray_indN && ray_indN->n_elem != n_rayN)
        throw std::invalid_argument("Input 'ray_indN' must match the length of 'out_typeN'.");
    if (ray_indN && ray_indN->n_elem != 0 && (arma::uword)ray_indN->max() >= n_ray)
        throw std::invalid_argument("Values in 'ray_indN' exceed the number of rays in 'orig'.");
    if (ray_indN == nullptr && n_ray != n_rayN)
        throw std::invalid_argument("Without 'ray_indN', the full and compact sets must have the same size.");
    const unsigned *p_ray_indN = ray_indN ? ray_indN->memptr() : nullptr;

    // The relaunch geometry ray_mesh_interact built, corrected in place where a direction is
    // written out of a lossy medium with a refracting type. Optional: without it nothing moves.
    if ((origN == nullptr) != (destN == nullptr))
        throw std::invalid_argument("In-out 'origN' and 'destN' must be provided together.");
    if (origN && (origN->n_rows != n_rayN || origN->n_cols != 3 || destN->n_rows != n_rayN || destN->n_cols != 3))
        throw std::invalid_argument("In-out 'origN' and 'destN' must have size [n_rayN, 3].");
    if ((tridir == nullptr) != (tridirN == nullptr))
        throw std::invalid_argument("Input 'tridir' and in-out 'tridirN' must be provided together.");
    if (tridir && origN == nullptr)
        throw std::invalid_argument("Input 'tridir' requires 'origN' and 'destN'.");
    if (tridir && (tridir->n_rows != n_ray || tridir->n_cols != 9))
        throw std::invalid_argument("Input 'tridir' must have size [n_ray, 9] (Cartesian).");
    if (tridirN && (tridirN->n_rows != n_rayN || tridirN->n_cols != 9))
        throw std::invalid_argument("In-out 'tridirN' must have size [n_rayN, 9] (Cartesian).");
    dtype *p_origN = origN ? origN->memptr() : nullptr;
    dtype *p_destN = destN ? destN->memptr() : nullptr;
    const dtype *p_tridir = tridir ? tridir->memptr() : nullptr;
    dtype *p_tridirN = tridirN ? tridirN->memptr() : nullptr;

    // The verdict of each event on its bounce series, handed from the base frequency of a run to
    // the others. Optional: without it every call decides for itself.
    if (verdict_inN && verdict_inN->size() != n_rayN)
        throw std::invalid_argument("Input 'verdict_inN' must match the length of 'out_typeN'.");
    const uint8_t *p_verdict_inN = verdict_inN ? verdict_inN->data() : nullptr;
    if (verdict_outN && verdict_outN->size() != n_rayN)
        verdict_outN->resize(n_rayN, 0);
    uint8_t *p_verdict_outN = verdict_outN ? verdict_outN->data() : nullptr;

    const double fGHz = (double)center_frequency * 1e-9;

#pragma omp parallel for schedule(static)
    for (long long i_ev = 0; i_ev < (long long)n_rayN; ++i_ev) // Interaction loop (compact set)
    {
        const size_t i_rayN = (size_t)i_ev;                                    // compact-set index
        const size_t i_ray = p_ray_indN ? (size_t)p_ray_indN[i_rayN] : i_rayN; // full-set index
        const RayWriter<dtype> w{i_rayN, (size_t)n_rayN};

        // Resolved type code, starting as ok (transmission, outside)
        qd::bits<uint8_t> res_code = 1;

        // The state words: decoded as they came in; next_* are what this event writes
        short next_current = p_cur_in ? p_cur_in[i_ray] : (short)0;
        short next_prev = p_prev_in ? p_prev_in[i_ray] : (short)0;
        short next_buffer = p_buf_in ? p_buf_in[i_ray] : (short)0;
        const StateIn st = decode_words(next_current, next_prev, next_buffer);
        int mtl_current = st.mtl_current; // the medium the ray is in; an exit face may correct it
        const int mtl_prev = st.mtl_prev, mtl_buffer = st.mtl_buffer, mass_code = st.mass_code, stack_behind = st.stack_behind;
        const bool leg_summed = st.resolved, bent = st.bent, prev_nonpar = st.prev_nonpar, prev_is_embedded = st.prev_is_embedded;
        const bool mass_crossed = st.mass_crossed, followed = st.followed;
        const short buffer_word_in = st.buffer_word;
        const int bent_carry = bent ? cur_bent : 0;

        // The hit as ray_mesh_interact wrote it; its TIR bit (the verdict at the geometric angle) is re-evaluated
        const unsigned n_hits = p_no_interact[i_ray];
        qd::bits<uint8_t> hit_code = p_out_typeN[i_rayN];
        const int mtl_fbs = (int)(p_mtl_ind_fbs[i_rayN] & word_mtl);
        const int mtl_sbs = (int)(p_mtl_ind_sbs[i_rayN] & word_mtl);
        const double theta = (double)p_fbs_angleN[i_rayN]; // incidence angle at the FBS, face convention
        const bool tir_hit = hit_code[5];
        hit_code[5] = false;
        res_code[5] = tir_hit;

        const SheetHit sh = sheet_hit(cols, hit_code, mtl_fbs, mtl_sbs); // derived once for both passes
        const SheetCase sheet_case = sh.kind;
        const int sheet_mtl = sh.mtl, sheet_solid = sh.solid, sheet_mtl2 = sh.mtl2;

        // Plane normal of the face that leads on: the SBS for codes 5, 23 and 29 (media-to-media with M1 hit
        // first, and the two corner crossings relaunched past the SBS), the FBS otherwise.
        double N[3];
        {
            const size_t normal_off = ((hit_code == 5 || hit_code == 23 || hit_code == 29) && n_hits != 0) ? 3 : 0;
            N[0] = (double)p_normal_vecN[i_rayN + (normal_off + 0) * n_rayN];
            N[1] = (double)p_normal_vecN[i_rayN + (normal_off + 1) * n_rayN];
            N[2] = (double)p_normal_vecN[i_rayN + (normal_off + 2) * n_rayN];
            normalize_exact(N);
        }

        // Segment origin O, destination D, the hit points F and S (D for a segment without a hit)
        const double O[3] = {(double)orig->at(i_ray, 0), (double)orig->at(i_ray, 1), (double)orig->at(i_ray, 2)};
        const double D[3] = {(double)dest->at(i_ray, 0), (double)dest->at(i_ray, 1), (double)dest->at(i_ray, 2)};
        double F[3] = {(double)fbsN->at(i_rayN, 0), (double)fbsN->at(i_rayN, 1), (double)fbsN->at(i_rayN, 2)};
        double S[3] = {(double)sbsN->at(i_rayN, 0), (double)sbsN->at(i_rayN, 1), (double)sbsN->at(i_rayN, 2)};
        if (n_hits == 0)
        {
            F[0] = D[0], F[1] = D[1], F[2] = D[2];
            S[0] = D[0], S[1] = D[1], S[2] = D[2];
        }

        // The direction the wave has on the segment (path_dir_prev), or the segment's
        double dir_prev[3];
        bool has_dir = false;
        if (p_path_dir_prev)
        {
            dir_prev[0] = (double)p_path_dir_prev[i_ray];
            dir_prev[1] = (double)p_path_dir_prev[i_ray + n_ray];
            dir_prev[2] = (double)p_path_dir_prev[i_ray + 2 * n_ray];
            has_dir = normalize_exact(dir_prev) > 1.0e-6;
        }
        if (!has_dir)
        {
            dir_prev[0] = F[0] - O[0];
            dir_prev[1] = F[1] - O[1];
            dir_prev[2] = F[2] - O[2];
            normalize_exact(dir_prev);
        }

        // The direction the ray goes on in, as ray_mesh_interact wrote it; the events below correct it
        double dir_next[3];
        if (path_dirN)
        {
            dir_next[0] = (double)p_path_dirN[i_rayN];
            dir_next[1] = (double)p_path_dirN[i_rayN + n_rayN];
            dir_next[2] = (double)p_path_dirN[i_rayN + 2 * n_rayN];
        }
        else if (n_hits == 0)
            dir_next[0] = F[0] - O[0], dir_next[1] = F[1] - O[1], dir_next[2] = F[2] - O[2];
        else
            dir_next[0] = D[0] - F[0], dir_next[1] = D[1] - F[1], dir_next[2] = D[2] - F[2];
        normalize_exact(dir_next);

        // The leg: the geometric lengths to the face and on to the destination, and the incidence at the face
        // from the direction the wave has. In a mass-law medium the wave is held undeviated, so a difference of
        // the two cosines is rounding unless a refracting tracer bent it in; in single precision the segment's
        // ends are rounded to float, which far from the origin tilts a short segment by more than 1e-6, so the
        // test there allows that rounding.
        const double len_to_face = (n_hits == 0) ? dist3(O, D) : dist3(O, F);
        const double len_face_to_dest = (n_hits == 0) ? dist3(O, D) : dist3(F, D);
        const double abs_cos_theta = std::abs(std::cos(theta + half_pi));
        double tol_vbs = 1e-6;
        if (sizeof(dtype) == 4 && material(cols, mtl_current).m > 0.0)
        {
            const double sc = std::max({std::abs(O[0]), std::abs(O[1]), std::abs(O[2]), std::abs(F[0]), std::abs(F[1]), std::abs(F[2]), 1.0});
            tol_vbs += 8.0 * float_ulp * sc / std::max(len_to_face, 1.0e-9);
        }
        const VbsGeometry vg = vbs_geometry(O, F, N, dir_prev, has_dir, theta, abs_cos_theta, len_to_face, tol_vbs);
        const double (&V)[3] = vg.V;
        const double len_to_vbs = vg.len_to_vbs, theta_t = vg.theta_t, abs_cos_theta_t = vg.abs_cos_theta_t, cos_dir_seg = vg.cos_dir_seg;
        const bool use_vbs = vg.moved; // the VBS is not the FBS: the geometry is redone from the wave's direction

        // Whether the wave was bent at the face behind it: the cosines say so where they differ, and with a
        // tracer that bends the segment itself the state does.
        const bool refracted = use_vbs || bent;

        // The bent bit of a leg that begins here by crossing into a medium: the tracer bends, and
        // ray_mesh_interact bent at this face (its TIR verdict is the one the segment was built on).
        const int bent_enter = ((interaction_type == 2 || interaction_type == 5) && !tir_hit) ? cur_bent : 0;

        // The ledger: the accumulated in-medium lengths of the leg, refracted and geometric. A leg is measured
        // from the point the ray left at (the hit point, or the far face of a co-located pair); the relaunch
        // offset past it is added at the end (relaunch_ledger_offset).
        double dist_refract = p_acc_refract_in ? (double)p_acc_refract_in[i_ray] : 0.0;
        double dist_geo = p_acc_geo_in ? (double)p_acc_geo_in[i_ray] : 0.0;
        bool ledger_at_relaunch = false; // the ledger ends where this event relaunches the ray

        // Adds the leg up to the face (or the lengths given) to the ledger.
        auto ledger_add = [&dist_refract, &dist_geo, &ledger_at_relaunch, n_hits, len_to_vbs, len_to_face](double val_refract = 0.0, double val_geo = 0.0)
        {
            ledger_at_relaunch = val_refract == 0.0 && val_geo == 0.0 && n_hits != 0;
            dist_refract += (val_refract == 0.0) ? len_to_vbs : val_refract;
            if (val_geo == 0.0)
                dist_geo += (val_refract == 0.0) ? len_to_face : val_refract;
            else
                dist_geo += val_geo;
        };

        // Restarts the ledger (with the lengths given: a leg preloaded to the destination).
        auto ledger_set = [&dist_refract, &dist_geo, &ledger_at_relaunch](double val_refract = 0.0, double val_geo = 0.0)
        {
            ledger_at_relaunch = val_refract == 0.0 && val_geo == 0.0;
            dist_refract = val_refract;
            dist_geo = (val_geo == 0.0) ? val_refract : val_geo;
        };

        // The output of this event: killed (code 0, xprmat and gain zero), scaled by a complex factor, a
        // transparent pass (identity, optionally keeping the incoming direction), or a new coefficient pair
        // in the basis from dir_prev to dir_next.
        auto kill = [&res_code, p_xprmatN, p_gainN, i_rayN, nXPR]()
        {
            res_code = 0;
            if (p_xprmatN)
                for (size_t iX = 0; iX < nXPR; ++iX)
                    p_xprmatN[i_rayN * nXPR + iX] = (dtype)0.0;
            if (p_gainN)
                p_gainN[i_rayN] = (dtype)0.0;
        };
        auto scale = [p_xprmatN, p_gainN, i_rayN, nXPR](double cr, double ci)
        {
            if (p_xprmatN)
                for (size_t iX = 0; iX < nXPR; iX += 2)
                {
                    size_t re_i = i_rayN * nXPR + iX;
                    size_t im_i = re_i + 1;
                    double re = (double)p_xprmatN[re_i];
                    double im = (double)p_xprmatN[im_i];
                    p_xprmatN[re_i] = dtype(re * cr - im * ci);
                    p_xprmatN[im_i] = dtype(re * ci + im * cr);
                }
            if (p_gainN)
                p_gainN[i_rayN] = dtype(double(p_gainN[i_rayN]) * (cr * cr + ci * ci));
        };
        auto pass_through = [&res_code, &dir_prev, &w, p_xprmatN, p_gainN, p_path_dirN, i_rayN, nXPR, refl_pass, is_scalar](bool keep_dir)
        {
            if (!refl_pass)
            {
                res_code[6] = true;  // transparent
                res_code[5] = false; // no TIR
            }
            if (p_xprmatN)
            {
                for (size_t iX = 0; iX < nXPR; ++iX)
                    p_xprmatN[i_rayN * nXPR + iX] = (dtype)0.0;
                p_xprmatN[i_rayN * nXPR] = (dtype)1.0; // VV_re
                if (!is_scalar)
                    p_xprmatN[i_rayN * nXPR + 6] = (dtype)1.0; // HH_re
            }
            if (p_gainN)
                p_gainN[i_rayN] = (dtype)1.0;

            // Only the caller knows whether dir_prev is the direction to keep, overriding the geometric Snell
            // direction ray_mesh_interact wrote.
            if (keep_dir)
                w.set3(p_path_dirN, dir_prev[0], dir_prev[1], dir_prev[2]);
        };
        auto write_coeff = [&dir_prev, &dir_next, &N, &w, p_xprmatN, p_gainN, is_scalar, nXPR](std::complex<double> te, std::complex<double> tm)
        {
            double xprmat[8], pgain;
            qd_polbasis(dir_prev[0], dir_prev[1], dir_prev[2], dir_next[0], dir_next[1], dir_next[2], N[0], N[1], N[2], te, tm, is_scalar, xprmat, pgain);
            w.set_col(p_xprmatN, xprmat, nXPR);
            w.set1(p_gainN, pgain);
        };

        // The recovered cosine of the wave in the current medium (recover_cosine), where it applies: a refracted
        // leg in a medium, entered through a parallel face. real_invariant: what the last recovery found.
        bool real_invariant = false;
        auto recover_in_current = [&real_invariant, &mtl_current, &cols, refracted, prev_nonpar, prev_is_embedded, mtl_prev, fGHz](double ct_traced) -> std::complex<double>
        {
            real_invariant = false;
            if (!refracted || mtl_current < 1 || prev_nonpar)
                return std::complex<double>(0.0, 0.0);
            const bool unnamed = is_sheet_mtl(cols, mtl_prev) || (prev_is_embedded && mtl_prev != 0);
            return recover_cosine(ct_traced, material(cols, unnamed ? 0 : mtl_prev), material(cols, mtl_current), unnamed, fGHz, real_invariant);
        };

        // Memo of the recovery at the leg's cosine: three sites need it (the exit face, the series, the leg), and
        // mtl_current, the key, is the one input that moves between them (a transition rewrites it).
        int ct_memo_for = -2; // -2 = nothing solved yet (0 is a valid medium: air)
        std::complex<double> ct_memo(0.0, 0.0);
        auto in_medium_cosine = [&ct_memo_for, &ct_memo, &mtl_current, &recover_in_current, abs_cos_theta_t]() -> std::complex<double>
        {
            if (ct_memo_for != mtl_current)
                ct_memo = recover_in_current(abs_cos_theta_t), ct_memo_for = mtl_current;
            return ct_memo;
        };

        // State the events share: the face coefficients the series is applied to, whether the face was evaluated
        // at a real-invariant cosine (a transition writes it into the word), whether this event summed a series,
        // and the medium without the mass column a held plate is left into at this event.
        std::complex<double> cTE(0.0, 0.0), cTM(0.0, 0.0);
        bool face_real_invariant = false;
        bool series_summed = false;
        int plate_exit_mtl = -1;

        // Re-aims the relaunch segment along a direction recovered out of a lossy medium. With a refracting type
        // ray_mesh_interact built origN, destN (and the tube's vertex rays) by refracting at a real angle in the
        // medium the ray leaves; where the direction written here belongs to a wave with a recovered cosine and a
        // real invariant, the segment follows it: both points on one line along dir_next through the point the
        // ray left at (the SBS of a co-located pair, else the FBS). Each vertex ray gets the same construction
        // from its own direction; its footprint does not move. All or nothing: a vertex that cannot be formed
        // keeps the old geometry. A leak behind a closed face gets it with air as the far medium, out of a
        // lossless medium too.
        auto realign_relaunch = [&real_invariant, &recover_in_current, &N, &F, &S, &dir_next, p_origN, p_destN, p_tridir, p_tridirN,
                                 interaction_type, i_ray, i_rayN, n_ray, n_rayN, n_hits, fGHz](const Material &A, const Material &B, int itype)
        {
            if (!p_origN || (interaction_type != 2 && interaction_type != 5))
                return;

            double td[9] = {};
            if (p_tridirN)
            {
                const bool keep_real_inv = real_invariant; // the recoveries below rewrite it
                bool ok = true;
                for (int iTube = 0; iTube < 3 && ok; ++iTube)
                {
                    double v[3] = {(double)p_tridir[i_ray + (3 * iTube) * n_ray],
                                   (double)p_tridir[i_ray + (3 * iTube + 1) * n_ray],
                                   (double)p_tridir[i_ray + (3 * iTube + 2) * n_ray]};
                    normalize_exact(v);
                    double cv = std::abs(v[0] * N[0] + v[1] * N[1] + v[2] * N[2]);
                    cv = (cv > 1.0) ? 1.0 : cv;

                    std::complex<double> c_a = recover_in_current(cv), c_b;
                    ok = std::abs(c_a) > 0.0;
                    if (!ok)
                        break;

                    const Interface f = interact_with(A, B, itype, std::acos(cv) - half_pi, fGHz, &c_a, false);
                    c_b = f.cos_t2;
                    double sn = f.snell;
                    ok = !f.tir && std::real(c_b) > 0.0;
                    if (!ok)
                        break;

                    sn *= std::sqrt(std::abs(1.0 - c_a * c_a) + std::real(c_a) * std::real(c_a));
                    qd_refract(v[0], v[1], v[2], N[0], N[1], N[2], sn, cv, c_b, td[3 * iTube], td[3 * iTube + 1], td[3 * iTube + 2]);
                }
                real_invariant = keep_real_inv;
                if (!ok)
                    return;
            }

            const double *A0 = (n_hits >= 2 && dist3(F, S) < colocation_dist) ? S : F; // where ray_mesh_interact relaunched from
            const double Dn[3] = {(double)p_destN[i_rayN], (double)p_destN[i_rayN + n_rayN], (double)p_destN[i_rayN + 2 * n_rayN]};
            const double len = dist3(F, Dn), off = qd_relaunch_offset(F[0], F[1], F[2]);
            for (size_t c = 0; c < 3; ++c)
            {
                p_origN[i_rayN + c * n_rayN] = (dtype)(A0[c] + off * dir_next[c]);
                p_destN[i_rayN + c * n_rayN] = (dtype)(A0[c] + len * dir_next[c]);
            }
            if (p_tridirN)
                for (size_t c = 0; c < 9; ++c)
                    p_tridirN[i_rayN + c * n_rayN] = (dtype)td[c];
        };

        // The direction of an exit from A into B (air, or a leak's far side). ray_mesh_interact bent the geometric
        // segment, which with an undeviated tracer runs at the external angle; what bends is dir_prev. Across
        // parallel faces the invariant p of a wave from air is real and conserved: the direction in air is sqrt(p)
        // against sqrt(1 - p), parallel to the one it came in at, formed as a transition forms it (the Snell ratio
        // with the length the projection lost, the far cosine at the recovered cosine ct_a). Where the recovery
        // declines, the real-angle ratio and cosine. A leg with no refracted direction keeps what
        // ray_mesh_interact wrote. Out of a lossy medium (real_inv), and for a leak out of any, the segment follows.
        auto exit_direction = [&dir_prev, &dir_next, &N, &w, &realign_relaunch, refracted, abs_cos_theta_t, p_path_dirN](
                                  const Material &A, const Material &B, bool real_inv, const std::complex<double> &ct_a, double snell_ab,
                                  const std::complex<double> &cos_b, bool tir_ab, bool leak, int itype) -> bool
        {
            if (!refracted || tir_ab || !(std::real(cos_b) > 0.0))
                return false;
            double snell_dir = snell_ab;
            if (real_inv)
                snell_dir *= std::sqrt(std::abs(1.0 - ct_a * ct_a) + std::real(ct_a) * std::real(ct_a));
            qd_refract(dir_prev[0], dir_prev[1], dir_prev[2], N[0], N[1], N[2], snell_dir, abs_cos_theta_t, cos_b,
                       dir_next[0], dir_next[1], dir_next[2]);
            w.set3(p_path_dirN, dir_next[0], dir_next[1], dir_next[2]);
            if (real_inv || leak)
                realign_relaunch(A, B, itype);
            return true;
        };

        // The direction a closed face of Ma towards Mb hands on: a leak carries the transverse wavenumber and
        // nothing else. Where that has a direction in air (an invariant below one) it is the direction, written as
        // an exit into air writes it (for a wave from air through parallel faces, the one it arrived in);
        // otherwise the direction the wave has. ct_in: the recovered cosine (0 for none).
        auto leak_direction = [&dir_prev, &dir_next, &cols, &face_real_invariant, &exit_direction, theta_t, fGHz](
                                  int Ma, int Mb, int itype, const std::complex<double> &ct_in)
        {
            dir_next[0] = dir_prev[0], dir_next[1] = dir_prev[1], dir_next[2] = dir_prev[2];
            if (Mb == 0)
                return;
            const bool use_exact_ct = std::abs(ct_in) > 0.0;
            const Interface air = interact_with(material(cols, Ma), material(cols, 0), itype, theta_t, fGHz, use_exact_ct ? &ct_in : nullptr, false);
            exit_direction(material(cols, Ma), material(cols, 0), face_real_invariant, ct_in, air.snell, air.cos_t2, air.tir, true, itype);
        };

        // The face Ma | Mb at the VBS angle theta_t in place of what ray_mesh_interact wrote at the FBS angle:
        // coefficients (att included), TIR, the continuation direction, and xprmat from dir_prev and the corrected
        // direction; the medium itself is charged by close_leg. geom: 0 reflect, 2 refract, 3 exit into air.
        // Returns the refracted length from the VBS to the virtual destination. A resolved ray is evaluated as
        // any other; only its series is skipped (by the caller).
        auto face_at_vbs = [&mtl_current, &in_medium_cosine, &face_real_invariant, &real_invariant, &cTE, &cTM, &res_code, &cols,
                            &dir_prev, &dir_next, &N, &V, &D, &w, &write_coeff, &exit_direction, &leak_direction, &realign_relaunch,
                            theta_t, fGHz, refracted, use_vbs, mtl_buffer, mtl_fbs, len_face_to_dest, abs_cos_theta_t, p_path_dirN](
                               int Ma, int Mb, int itype, int geom) -> double
        {
            std::complex<double> cos_t2;
            double snell = 1.0;
            bool tir = false;

            // The face is met from inside the current medium. Where the wave was refracted in through a parallel
            // face its cosine is complex, and the recovery returns the value the entry face was evaluated at;
            // evaluating the exit at the same cosine makes the two flux factors telescope (exactly 1 back into the
            // medium it came from). Where the recovery declines the real angle is used. Ma is mtl_current at both
            // call sites; the test keeps the value from being read as a cosine of another medium. cos_t2 follows:
            // by Snell's transitivity it is the outer pair's cosine, which only geom 2 uses.
            const std::complex<double> ct_in = (Ma == mtl_current) ? in_medium_cosine() : std::complex<double>(0.0, 0.0);
            const bool use_exact_ct = std::abs(ct_in) > 0.0;
            face_real_invariant = use_exact_ct && real_invariant;

            // Without a refracted direction the wave is held at its geometric angle, as it was in
            // ray_mesh_interact, and the two have to give the same verdict.
            {
                const Interface f = interact_with(material(cols, Ma), material(cols, Mb), itype, theta_t, fGHz, use_exact_ct ? &ct_in : nullptr, !refracted);
                cTE = f.cTE, cTM = f.cTM, cos_t2 = f.cos_t2, snell = f.snell, tir = f.tir;
            }
            res_code[5] = tir;

            // A transition into the deferred medium: the coefficients from the incoming and outgoing directions
            if (!use_vbs && mtl_buffer && Mb == mtl_buffer && !same_medium(cols, mtl_buffer, mtl_fbs))
                write_coeff(cTE, cTM);

            if (!refracted)
                return len_face_to_dest;

            // A bent leg between two lossless media: the segment is the wave, and what ray_mesh_interact wrote
            // from it is this face at this angle; left as it stands, except the direction of an exit or a leak.
            // Not a mass-law medium, where ray_mesh_interact held the wave undeviated.
            if (!use_vbs)
            {
                const Material A = material(cols, Ma), B = material(cols, Mb);
                if (std::imag(A.eta(fGHz) * A.mu(fGHz)) == 0.0 && std::imag(B.eta(fGHz) * B.mu(fGHz)) == 0.0 && !(A.m > 0.0) && !(B.m > 0.0))
                {
                    if (geom == 3)
                        exit_direction(A, B, false, ct_in, snell, cos_t2, tir, false, itype);
                    else if (geom == 2 && tir) // closed: the segment is the wave in A, the leak is not
                    {
                        leak_direction(Ma, Mb, itype, ct_in);
                        w.set3(p_path_dirN, dir_next[0], dir_next[1], dir_next[2]);
                    }
                    return len_face_to_dest;
                }
            }

            if (use_vbs)
                res_code[1] = true; // VBS correction

            if (geom == 0) // Reflect at VBS
                qd_reflect(dir_prev[0], dir_prev[1], dir_prev[2], N[0], N[1], N[2], dir_next[0], dir_next[1], dir_next[2]);
            else if (geom == 2 && !tir) // Snell refract at VBS (undeviated under TIR)
            {
                // The direction in the next medium is the one a single refraction from air writes (tangential
                // sqrt(p / |e2|) against Re(cos_t2)), the map the recovery inverts there. dir_prev is a projection of
                // length sqrt(|sin^2| + Re(cos)^2), short of one for a complex cosine; that length goes back onto the
                // Snell ratio, so no later direction inherits it.
                double snell_dir = snell;
                if (face_real_invariant)
                    snell_dir *= std::sqrt(std::abs(1.0 - ct_in * ct_in) + std::real(ct_in) * std::real(ct_in));
                qd_refract(dir_prev[0], dir_prev[1], dir_prev[2], N[0], N[1], N[2], snell_dir, abs_cos_theta_t, cos_t2, dir_next[0], dir_next[1], dir_next[2]);
                if (face_real_invariant) // out of a lossy solid: the segment in the next medium follows
                    realign_relaunch(material(cols, Ma), material(cols, Mb), itype);
            }
            else if (tir) // closed at this angle: no Snell direction, the leak keeps its transverse wavenumber
                leak_direction(Ma, Mb, itype, ct_in);
            else if (geom == 3) // exit into air: the direction of dir_prev refracted, where it is exact
                exit_direction(material(cols, Ma), material(cols, Mb), face_real_invariant, ct_in, snell, cos_t2, tir, false, itype);

            if (geom != 3 || tir) // an open exit wrote its own direction or keeps the traced one
                w.set3(p_path_dirN, dir_next[0], dir_next[1], dir_next[2]);

            write_coeff(cTE, cTM);

            // The refracted length from the VBS to the virtual destination (the angle at the VBS differs from the
            // FBS one)
            double dist_VV;
            if (vbs_intersect(V, dir_next, D, N, &dist_VV))
                return dist_VV;
            return len_face_to_dest;
        };

        // The bonded mass-law bodies behind the leg (bonded_x0, bonded_thickness, as far as the state names them),
        // and the absorption of the face the current plate was entered through: from the previous medium; for a
        // later body of a bonded plate the face of the body before it from air; behind a sheet inside the current
        // body that body's own face from air (one level, as the state).
        auto leg_bonded_x0 = [&cols, &mtl_current, mass_crossed, mass_code, mtl_prev, prev_is_embedded, fGHz, abs_cos_theta]() -> double
        {
            return bonded_x0(mass_crossed, mass_code, material(cols, mtl_prev), is_sheet_mtl(cols, mtl_prev), prev_is_embedded,
                             material(cols, mtl_current), fGHz, abs_cos_theta);
        };
        auto leg_bonded_thickness = [&cols, &mtl_current, mass_crossed, mass_code, mtl_prev, prev_is_embedded](int R, double d, int extra) -> double
        {
            const Material ex = material(cols, extra);
            return bonded_thickness(material(cols, R), d, (extra != 0) ? &ex : nullptr, mass_crossed, mass_code, material(cols, mtl_prev),
                                    is_sheet_mtl(cols, mtl_prev), prev_is_embedded, material(cols, mtl_current));
        };
        auto entry_face_absorption = [&cols, mtl_prev, prev_is_embedded, is_scalar, theta_t, fGHz](const Material &C) -> double
        {
            const int it = is_scalar ? 4 : 1;
            const Material P = material(cols, mtl_prev);
            if (is_sheet_mtl(cols, mtl_prev) && prev_is_embedded) // a sheet inside the current body: the body's own face
                return face_absorption(material(cols, 0), C, it, theta_t, fGHz);
            return (P.m > 0.0) ? face_absorption(material(cols, 0), P, it, theta_t, fGHz) : face_absorption(P, C, it, theta_t, fGHz);
        };

        // Closes the leg in medium m out: its gain and phase scaled into xprmat and gain.
        auto close_leg = [&cols, &mtl_current, &in_medium_cosine, &dist_refract, &dist_geo, &leg_bonded_x0, &plate_exit_mtl,
                          &entry_face_absorption, &scale, fGHz, refracted, abs_cos_theta_t, abs_cos_theta, use_vbs, cos_dir_seg,
                          mtl_prev, is_scalar, theta_t](int m)
        {
            const Material mtl = material(cols, m);
            const double n_re = std::real(std::sqrt(mtl.eta(fGHz) * mtl.mu(fGHz)));

            // Walk-off (cos^2) only for a wave that was not bent (a leak, a leg with no refracted direction), whose
            // polyline is the wave; dist_geo then carries the geometric cos^2. A bent wave takes the forms below on
            // every pass, so one leg has one form whichever event closes it. The phase belongs to the point the
            // polyline leaves at: path length, delay and free-space phase come from the polyline (one for every
            // frequency of a run), and the wave leaves beside it wherever the polyline is not the refracted ray.
            const bool walk_off = !refracted;

            // The exact form wherever the recovered cosine exists (not in mass-law media: their index is an
            // impedance mapping, their term not per metre)
            const std::complex<double> cs = (!walk_off && m == mtl_current && !(mtl.m > 0.0)) ? in_medium_cosine() : std::complex<double>(0.0, 0.0);
            if (std::abs(cs) > 0.0)
            {
                const std::complex<double> P = leg_exact_factor(mtl, cs, dist_refract, dist_geo, abs_cos_theta_t, abs_cos_theta, use_vbs, fGHz);
                scale(std::real(P), std::imag(P));
                return;
            }

            // A leg entered through a bonded mass-law body pays what the plate is worth above what that body paid
            double g = mtl.medium_gain(dist_refract, fGHz, abs_cos_theta_t, dist_geo, abs_cos_theta,
                                       mass_min_len, -1.0, -1.0, (mtl.m > 0.0) ? leg_bonded_x0() : 0.0);

            // The leg on which a held plate is left: the plate passes in all what a plate of its two outer faces
            // passes, its entry face having passed a_in and its earlier bodies their terms
            if (mtl.m > 0.0 && !refracted && plate_exit_mtl >= 0 && m == mtl_current)
            {
                double a_in = entry_face_absorption(mtl), a_out = face_absorption(material(cols, plate_exit_mtl), mtl, is_scalar ? 4 : 1, theta_t, fGHz);
                double tau = std::pow(10.0, -0.1 * mass_TL(leg_bonded_x0(), abs_cos_theta, fGHz, material(cols, mtl_prev).side)) * g;
                if (a_in > 0.0 && tau > 0.0)
                    g *= qd_plate(std::min(a_in, a_out), tau) / (a_in * tau);
            }
            double cos2 = walk_off ? abs_cos_theta_t * abs_cos_theta_t : 1.0; // medium side, theta_t
            double cos2_geo = walk_off ? abs_cos_theta * abs_cos_theta : 1.0; // free-space side, theta_i
            double excess_phase = k0(fGHz) * (n_re * dist_refract * cos2 - dist_geo * cos2_geo);

            // A bent wave with no recovered cosine, in a mass-law solid or where the recovery declines: the phase
            // of the real index at the point the polyline leaves at, the wave vector along dir_prev times the
            // polyline
            if (!walk_off && use_vbs)
                excess_phase = k0(fGHz) * (n_re * dist_geo * cos_dir_seg - dist_geo);

            double amp = std::sqrt(g < 0.0 ? 0.0 : g);
            scale(amp * std::cos(-excess_phase), amp * std::sin(-excess_phase));
        };

        // The material of the current medium by its exit face; a mismatch with mtl_current sets the fix flag.
        auto exit_mtl = [&mtl_current, &res_code, hit_code, mtl_fbs, mtl_sbs]() -> int
        {
            const int mtl_exit = exit_material((int)hit_code, mtl_fbs, mtl_sbs);
            if (mtl_exit && mtl_current != mtl_exit)
                res_code[4] = true;
            return mtl_exit;
        };

        // The series of the leg in mtl_slab (qd_slab_resolve), per polarization: xprmat is rebuilt from cTE / cTM
        // times the per-channel factor, so TE and TM carry their own retardation. mtl_near is the face ahead,
        // mtl_far the one the leg was entered through (symmetric); mtl_back what lies behind a sheet named as
        // mtl_near, where the caller knows it (air otherwise). Returns true where the series was summed.
        auto sum_series = [&cols, &in_medium_cosine, &dist_refract, &dist_geo, &series_summed, &scale, &write_coeff, &cTE, &cTM,
                           p_verdict_inN, i_rayN, eps, prev_nonpar, prev_is_embedded, mtl_prev, sheet_mtl2, stack_behind, sheet_case,
                           abs_cos_theta_t, abs_cos_theta, fGHz, is_scalar](int mtl_slab, int mtl_near, int mtl_far, int mtl_back) -> bool
        {
            // The verdict of this event, where the caller hands it in: what the base frequency of a run decided.
            // The paths are the base frequency's, so every other frequency sums where it summed and leaves the
            // bounces to the tracer where it did not.
            const bool given = p_verdict_inN != nullptr;
            if (given && p_verdict_inN[i_rayN] == 0)
                return false;

            if (!given && eps > 0.0 && prev_nonpar) // known wedge/edge -> re-emit (eps = 0.0 always resolves)
                return false;

            // What lies behind a sheet named as mtl_far, the mirror behind the ray: every caller passes the
            // previous-medium word there. A sheet in it that lies inside the slab is backed by the slab; a lining
            // is backed by air, which is the default.
            const Material M_back = material(cols, mtl_back);
            const Material M_back_far = material(cols, mtl_slab);
            const bool far_in_slab = prev_is_embedded && mtl_far == mtl_prev && is_sheet_mtl(cols, mtl_far);

            std::complex<double> fac_te, fac_tm;
            const Material M_near2 = material(cols, sheet_mtl2), M_far2 = material(cols, stack_behind); // the second layer of a stack ahead, and behind
            SlabOpts opt;
            opt.ct_slab_exact = in_medium_cosine();
            opt.back_near = &M_back, opt.back_far = far_in_slab ? &M_back_far : nullptr, opt.verdict_given = given;
            opt.near2 = (sheet_case == SheetCase::stack) ? &M_near2 : nullptr, opt.far2 = (far_in_slab && stack_behind != 0) ? &M_far2 : nullptr;
            if (!qd_slab_resolve(material(cols, mtl_near), material(cols, mtl_slab), material(cols, mtl_far), dist_refract, dist_geo,
                                 abs_cos_theta_t, abs_cos_theta, fGHz, is_scalar, eps, opt, fac_te, fac_tm))
                return false;
            series_summed = true;

            // Scalar mode carries one port and the factor is a pure scale: the stored xprmat is scaled in place
            // (rebuilding it from cTE would round-trip through dtype differently).
            if (is_scalar)
            {
                scale(std::real(fac_te), std::imag(fac_te));
                return true;
            }
            write_coeff(cTE * fac_te, cTM * fac_tm);
            return true;
        };

        // A sheet crossing against the media the tracked state reports. ray_mesh_interact assumes air on any
        // side the mesh does not name, which holds only while the ray is in air: a sheet inside a medium has that
        // medium on both sides, a glued sheet reached from inside another medium has it in front, and the wrong
        // pair moves both faces, the series and att at once. Evaluated at theta_t, so the VBS correction is built
        // in (bit 1 of the resolved code means glued in the sheet region). Sets cTE / cTM, so that a following
        // sum_series folds the host's series into the same pair. Returns false where the composite is left as
        // ray_mesh_interact wrote it.
        auto sheet_at_tracked_media = [&cols, &mtl_current, &in_medium_cosine, &res_code, &dir_prev, &dir_next, &N, &F, &S, &w,
                                       &real_invariant, &realign_relaunch, &dist_refract, &entry_face_absorption, &cTE, &cTM, &write_coeff,
                                       interaction_type, theta_t, fGHz, refracted, refl_pass, abs_cos_theta_t, abs_cos_theta, p_path_dirN,
                                       hit_code, n_hits, len_face_to_dest, is_scalar](int mtl_slab, int mtl_near, int mtl_far, bool disp_far,
                                                                                      int mtl_slab2, double mass_bonded) -> bool
        {
            const Material M_slab = material(cols, mtl_slab), M_near = material(cols, mtl_near), M_far = material(cols, mtl_far);
            const Material M_slab2 = material(cols, mtl_slab2);

            // Geometry and total reflection from the outer pair at theta_t: with an undeviated tracer the hit code's
            // verdict was taken at the external angle and would make a glued exit a mirror past the internal
            // critical angle. Snell is transitive, so the sheet neither creates nor removes total reflection, nor
            // bends the ray.
            const Interface outer = interact_with(M_near, M_far, interaction_type, theta_t, fGHz, nullptr, !refracted);
            const std::complex<double> ct2_far = outer.cos_t2;
            const double snell = outer.snell;
            bool tir = outer.tir;

            // Where the near medium's cosine was recovered, the verdict and the far cosine are the ones at that
            // cosine, as at a bare face: the real-angle test reads the invariant of a lossy near medium short by the
            // length the projection lost, and finds the pair open on a wave that is closed.
            std::complex<double> c_n(0.0, 0.0), c_f(0.0, 0.0);
            if (mtl_near == mtl_current && mtl_far != mtl_near)
            {
                c_n = in_medium_cosine();
                if (std::abs(c_n) > 0.0)
                {
                    const Interface f = interact_with(M_near, M_far, interaction_type, theta_t, fGHz, &c_n, false);
                    c_f = f.cos_t2, tir = f.tir;
                }
            }
            res_code[5] = tir;

            // The continuation direction, corrected: the next face must be met at the physical direction. With
            // near == far the refraction reproduces dir_prev.
            if (refl_pass)
                qd_reflect(dir_prev[0], dir_prev[1], dir_prev[2], N[0], N[1], N[2], dir_next[0], dir_next[1], dir_next[2]);
            else if (tir) // no Snell direction under total reflection; any leak travels undeviated
                dir_next[0] = dir_prev[0], dir_next[1] = dir_prev[1], dir_next[2] = dir_prev[2];
            else
            {
                // Out of a lossy near medium with a recovered cosine and a real invariant: the construction of a
                // transition (the Snell ratio with the length the projection lost, the far cosine of the outer pair
                // at the recovered cosine); out into air it is the direction the wave had in air before the wall.
                // The real-angle far cosine of a lossy medium does not fall to the external cosine towards grazing.
                double snell_dir = snell;
                std::complex<double> ct2_dir = ct2_far;
                const bool dir_real_inv = std::abs(c_n) > 0.0 && real_invariant && std::real(c_f) > 0.0;
                if (dir_real_inv)
                {
                    snell_dir *= std::sqrt(std::abs(1.0 - c_n * c_n) + std::real(c_n) * std::real(c_n));
                    ct2_dir = c_f;
                }
                qd_refract(dir_prev[0], dir_prev[1], dir_prev[2], N[0], N[1], N[2], snell_dir, abs_cos_theta_t, ct2_dir,
                           dir_next[0], dir_next[1], dir_next[2]);
                if (dir_real_inv) // a glued exit out of a lossy host: the segment follows
                    realign_relaunch(M_near, M_far, interaction_type);
            }
            w.set3(p_path_dirN, dir_next[0], dir_next[1], dir_next[2]);

            // The displaced host as a normal thickness on either side of the mesh plane: in front the leg the ray
            // is on, behind the gap to the second face or the destination; for a co-located pair (a stack, a
            // duplicate) the leg in front for both. Air is not capped. A layer that does not fit keeps its
            // thickness.
            double host_near = -1.0, host_far = -1.0;
            const Material &M_disp = disp_far ? M_far : M_near;
            if (!M_disp.same_as(Material()))
            {
                host_near = dist_refract * abs_cos_theta_t;
                host_far = hit_code[2] ? host_near : ((n_hits >= 2) ? dist3(F, S) : len_face_to_dest) * abs_cos_theta;
            }

            // Incidence cosine in the near medium: the recovered complex cosine where the near medium is a traced
            // solid (every boundary, the series, the displaced host and the flux conversion are built from it),
            // otherwise the traced real projection ct_geo.
            const double ct_geo = abs_cos_theta_t;
            const std::complex<double> ct_near = in_medium_cosine();

            // Angle factor of a mass term: the cosine the wave has in air. Reached from air, or from a medium that
            // holds the wave undeviated, that is ct_geo; behind a medium the wave was refracted into, mass_cos_in_air.
            const double ct_mass = (mtl_near != 0 && refracted && !(M_near.m > 0.0) && (M_slab.m > 0.0 || (mtl_slab2 != 0 && M_slab2.m > 0.0)))
                                       ? mass_cos_in_air(M_near, interaction_type, theta_t, fGHz, (std::abs(ct_near) > 0.0) ? &ct_near : nullptr, ct_geo)
                                       : ct_geo;

            // One sheet or an ordered stack, in one call. Total reflection of the outer pair keeps the sheet: the
            // solver is told that the far side is closed (the pad on both ports of a coherent sheet, the two leaks
            // of a mass-law sheet).
            StackOpts opt;
            opt.disp_far = disp_far, opt.host_near = host_near, opt.host_far = host_far, opt.mass_cos = ct_mass;
            opt.mass_bonded = mass_bonded, opt.near_undeviated = !refracted, opt.far_closed = tir;
            opt.plate_a = (M_near.m > 0.0 && !refracted) ? entry_face_absorption(M_near) : -1.0;
            Ports ports;
            if (!sheet_ports(M_near, M_slab, (mtl_slab2 != 0) ? &M_slab2 : nullptr, M_far,
                             (std::abs(ct_near) > 0.0) ? ct_near : std::complex<double>(ct_geo, 0.0), fGHz, is_scalar, opt, ports))
                return false; // a singular denominator; the caller decides

            // Both ports come out of the one pass: the reflection port is its own Airy sum, not the half-space
            // Fresnel value
            cTE = refl_pass ? ports.R_te : ports.T_te;
            cTM = refl_pass ? ports.R_tm : ports.T_tm;
            write_coeff(cTE, cTM);
            return true;
        };

        // A sheet event. The sheet's internal series is inside its composite, so nothing here applies a face
        // coefficient through interact_with or charges the sheet material. In order: the composite against the
        // tracked media (sheet_at_tracked_media); the host leg closed, its series reflecting off the sheet ahead
        // (sum_series, which rebuilds xprmat from the composite's cTE / cTM, hence this order); a new leg with the
        // sheet as the previous medium, so the next host leg reflects off it from the other side.
        auto sheet_event = [&cols, &mtl_current, &res_code, &ledger_add, &ledger_set, &dist_geo, &leg_bonded_x0, &sheet_at_tracked_media,
                            &kill, &sum_series, &plate_exit_mtl, &close_leg, &next_current, &next_prev, &next_buffer, &leg_bonded_thickness,
                            &F, &D, &N, &dir_next, n_hits, hit_code, sheet_case, sheet_solid, sheet_mtl, sheet_mtl2, mtl_buffer, refl_pass,
                            fGHz, abs_cos_theta, leg_summed, followed, mtl_prev, bent_carry, bent_enter, prev_nonpar, len_face_to_dest]()
        {
            // An ordered stack is one crossing, in the order of the hit code, with the host on both sides; a
            // singular stack kills the ray.
            const bool ray_ends = n_hits == 1 || (n_hits == 2 && hit_code[2]);

            // Bounding media: near is the current medium; far the same medium for an isolated sheet, the glued
            // solid on entry, air on exit. Air beyond a glued exit is not read from the state: prev names what is
            // behind the ray (the mirror of the host's series). A lined body nested in the medium it was entered
            // from is not a use case; a sheet between two solids is not reported by the intersector.
            const int m_near = mtl_current;
            const int m_far = (sheet_case == SheetCase::glued_entry) ? sheet_solid : ((sheet_case == SheetCase::glued_exit) ? 0 : mtl_current);
            const int m_next = m_far; // medium the ray travels in after the crossing

            // No sheet code in scope carries a deferred buffer. Flag the violation rather than guess:
            // a glued exit still exits into air, and the deferred medium is dropped.
            if (mtl_buffer != 0)
                res_code[4] = true;

            // A sheet that matches the host (same eps and mu) is no boundary: the host leg is carried through it
            // (distance and words kept) and its series resolves once at the far side; only the sheet's composite is
            // applied here. Cut at the sheet, each half would see r = 0 there and lose its series. Transmission only
            // (a matched sheet reflects nothing), one layer only (a stack is not transparent because its first layer
            // is), and not for a mass-law sheet in a mass-law host (no series to protect; cut, the halves and the
            // sheet are three bonded bodies).
            const bool merge_host = !refl_pass && sheet_case != SheetCase::stack && mtl_current != 0 &&
                                    m_next == mtl_current && same_eps_mu(material(cols, sheet_mtl), material(cols, mtl_current), fGHz) &&
                                    !(material(cols, mtl_current).m > 0.0 && material(cols, sheet_mtl).m > 0.0);

            ledger_add(); // Advance the host leg up to the sheet

            // The sheet lies on the side of a glued pair that is not the solid (near on entry, far on exit).
            // Reached from inside a mass-law body it is bonded to it: the host leg pays its own term below, and the
            // sheet is handed the X0 that leg carries (with what it was entered through) and pays the increment.
            double mass_host = 0.0;
            if (mtl_current != 0)
            {
                const Material M_host = material(cols, mtl_current);
                if (M_host.m > 0.0)
                    mass_host = leg_bonded_x0() + ((dist_geo > mass_min_len) ? M_host.mass_X0(dist_geo * abs_cos_theta, fGHz, abs_cos_theta) : 0.0);
            }
            const bool have_composite = sheet_at_tracked_media(sheet_mtl, m_near, m_far, sheet_case == SheetCase::glued_exit, sheet_mtl2, mass_host);

            if (sheet_case == SheetCase::stack && !have_composite)
            {
                kill(); // a singular stack: nothing to hand on
                return;
            }

            // Close out the host leg: its series reflects off the sheet ahead (resolved as a layer, with m_far
            // behind it, which only this site knows) and off what the host was entered through. Skipped without a
            // composite (the factor goes through cTE / cTM) and for a resolved ray, as at a bare face.
            bool host_summed = false; // the host's bounce series was summed at this event
            if (mtl_current != 0 && !merge_host)
            {
                if (have_composite && !leg_summed && !followed)
                    host_summed = sum_series(mtl_current, sheet_mtl, mtl_prev, m_far);
                plate_exit_mtl = (refl_pass || material(cols, sheet_mtl).m > 0.0) ? -1 : sheet_mtl;
                close_leg(mtl_current);
            }

            // Bits 6 and 2 together mark a sheet event (unreachable otherwise); bit 1 carries the glued verdict there.
            res_code[1] = (sheet_case == SheetCase::glued_entry || sheet_case == SheetCase::glued_exit);
            res_code[2] = true;
            res_code[6] = true;
            res_code[3] = refl_pass ? (mtl_current != 0) : (m_next != 0);

            if (refl_pass) // A reflection stays in the host medium
            {
                ledger_set();
                if (mtl_current != 0)
                {
                    // The series ends as for a resolved solid where it was summed here; declined, the reflected
                    // ray is followed. The sheet is behind the ray now, backed by the current medium where that
                    // medium is on both of its sides.
                    next_current = (short)((mtl_current & word_mtl) | (host_summed ? cur_summed : 0) | bent_carry);
                    next_prev = (short)((sheet_mtl & word_mtl) | (prev_nonpar ? prev_nonparallel : 0) |
                                        ((m_far == mtl_current) ? prev_embedded : 0));
                    next_buffer = (short)(host_summed ? 0 : buf_followed); // not summed: followed
                }
                return;
            }

            // Transmission: the leg restarts on the far side of the sheet, unless it was never interrupted (a
            // merged host layer; air on both sides, where the ledger runs on and is dropped at the next medium entry).
            if (merge_host || (mtl_current == 0 && m_next == 0))
                return;

            // A new leg (cur_summed cleared). Behind a sheet inside the medium the leg is bent if the one in
            // front was; a glued entry says what it did itself.
            next_current = (short)((m_next & word_mtl) |
                                   ((m_next == 0) ? 0 : ((m_next == mtl_current) ? bent_carry : ((material(cols, m_next).m > 0.0) ? 0 : bent_enter))));

            // The sheet becomes the previous medium: a boundary material, never a traversed one, so the next host
            // leg reflects off it. An exit into air writes it too (nothing reads it in air). The wedge test cannot be
            // used for a glued entry (sheet and near face are co-located; the far face is not in the hit): parallel
            // is assumed, right for a coating on a flat wall.
            const bool nonpar = (sheet_case == SheetCase::glued_entry) ? false : prev_nonpar;

            // The boundary behind the ray is the last layer crossed (sheet_mtl2 for a stack); crossed inside a
            // medium, it is backed by that medium (bit 14).
            const int sheet_back = (sheet_mtl2 != 0) ? sheet_mtl2 : sheet_mtl;
            const bool inside = m_next != 0 && m_next == mtl_current;
            next_prev = (short)((sheet_back & word_mtl) | ((m_next != 0 && nonpar) ? prev_nonparallel : 0) |
                                (inside ? prev_embedded : 0));

            // A mass-law sheet crossed into a medium sets mass_crossed; crossed inside a mass-law medium the closed
            // leg is bonded to the new one through the sheet, and its thickness (with what it was entered through and
            // a stack's first layer) travels as the code. Behind a stack whose last layer is coherent the other layer
            // goes into the buffer word.
            next_buffer = (short)((sheet_mtl2 != 0 && m_next != 0 && !(material(cols, sheet_back).m > 0.0)) ? (buf_followed | sheet_mtl) : 0);
            if (m_next != 0 && material(cols, sheet_back).m > 0.0)
            {
                next_buffer = (short)buf_mass_code;
                if (inside && material(cols, mtl_current).m > 0.0 && dist_geo > mass_min_len)
                    next_buffer = encode_mass_code(leg_bonded_thickness(mtl_current, dist_geo * abs_cos_theta, (sheet_mtl2 != 0) ? sheet_mtl : 0));
            }

            if (ray_ends && m_next != 0) // Entering or staying in a medium: refracted preload
            {
                double dist_VV;
                if (vbs_intersect(F, dir_next, D, N, &dist_VV))
                    ledger_set(dist_VV, len_face_to_dest);
                else
                    ledger_set(len_face_to_dest);
            }
            else if (ray_ends) // Leaving into air: the post-exit leg is unrefracted
                ledger_set(len_face_to_dest);
            else
                ledger_set();
        };

        // A bare-face transition: media-to-media (i-i) or an exit (i-o). mtl_next: the medium entered (0 with exit:
        // air; 0 otherwise: by the hit code). preload_dest: the ledger is preloaded to the destination.
        auto transition = [&cols, &exit_mtl, &mtl_current, &ledger_add, &ledger_set, &pass_through, &V, &dir_prev, &D, &N, &face_at_vbs,
                           &sum_series, &res_code, &plate_exit_mtl, &close_leg, &dist_geo, &leg_bonded_thickness, &next_prev,
                           &face_real_invariant, &next_current, &next_buffer, hit_code, mtl_fbs, mtl_sbs, mass_crossed, followed,
                           stack_behind, buffer_word_in, len_face_to_dest, is_scalar, leg_summed, mtl_prev, abs_cos_theta,
                           bent_enter](bool exit, int mtl_next, bool preload_dest)
        {
            if (mtl_next == 0) // the exit face names the medium the ray is in
            {
                const int mtl_exit = exit_mtl();
                mtl_current = mtl_exit ? mtl_exit : mtl_current;
            }
            if (!exit && mtl_next == 0)
                mtl_next = hit_code == 5 ? mtl_sbs : mtl_fbs;

            ledger_add();

            // The buffer word of the leg that follows: a transparent transition carries the leg and its word on; a
            // mass-law solid crossed into another medium leaves the surface mass of the leg closed here and of what
            // it was bonded to; everything else starts clean.
            short buffer_next = (short)0;

            if (same_medium(cols, mtl_current, mtl_next)) // the same material: the face is transparent
            {
                if (mass_crossed || followed || stack_behind != 0)
                    buffer_next = buffer_word_in;
                pass_through(true); // keep the refracted direction
                if (preload_dest)   // add the lengths on: geometric to the destination, refracted from the VBS
                {
                    double dist_VV;
                    if (vbs_intersect(V, dir_prev, D, N, &dist_VV))
                        ledger_add(dist_VV, len_face_to_dest);
                    else
                        ledger_add(len_face_to_dest);
                }
            }
            else
            {
                const double dist_VV = face_at_vbs(mtl_current, mtl_next, (is_scalar ? 4 : 1), (exit ? 3 : 2));

                if (!leg_summed && !followed && sum_series(mtl_current, mtl_next, mtl_prev, 0))
                    res_code[2] = true;

                plate_exit_mtl = exit ? 0 : ((material(cols, mtl_next).m > 0.0) ? -1 : mtl_next);
                close_leg(mtl_current);

                if (!exit && mtl_next != 0 && material(cols, mtl_current).m > 0.0 && dist_geo > mass_min_len)
                    buffer_next = encode_mass_code(leg_bonded_thickness(mtl_current, dist_geo * abs_cos_theta, 0));

                if (preload_dest && exit) // exit into air: the leg after it is unrefracted
                    ledger_set(len_face_to_dest);
                else if (preload_dest) // into a medium: the refracted length differs from the geometric one
                    ledger_set(dist_VV, len_face_to_dest);
                else
                    ledger_set();

                // Written here also for equal materials with different indices: the series needs the entry material
                // in prev. Into a medium, bit 14 says whether the face was evaluated at a real-invariant cosine. A leg
                // entered out of a mass-law medium is not a bent one: the wave was held there.
                next_prev = (short)(mtl_current | ((face_real_invariant && !exit && mtl_next != 0) ? prev_embedded : 0));
                next_current = (short)(mtl_next | ((mtl_next != 0 && !exit && !(material(cols, mtl_next).m > 0.0)) ? bent_enter : 0));
            }

            next_buffer = buffer_next;
            if (!exit)
                res_code[3] = true; // inside
        };

        // The event. Only a glued exit leaves the ray in air with a sheet in prev; meeting a body from inside
        // after it means that the sheet lay between two bodies of which the intersector reported two faces:
        // nothing of it is real.
        const int hit = (int)hit_code;
        if (mtl_current == 0 && is_sheet_mtl(cols, mtl_prev) && (hit == 1 || hit == 5 || hit == 7 || hit == 13 || hit == 21 || hit == 29))
            kill();
        else if (refl_pass) // Reflection pass, interaction_type {0, 3}
        {
            res_code[7] = true;

            const int mtl_exit = exit_mtl();
            mtl_current = mtl_exit ? mtl_exit : mtl_current;                              // the medium the ray travels in
            const int mtl_next = hit_code == 5 ? mtl_sbs : (hit_code == 7 ? mtl_fbs : 0); // the medium it reflects off

            // Events that trigger a transparent pass-through cannot reflect at the same time
            const bool transparent_forward = (n_hits == 0) || (hit_code == 23) || (!hit_code[0]) ||
                                             (mtl_current != 0 && same_medium(cols, mtl_current, mtl_next)) ||
                                             (mtl_buffer != 0 && same_medium(cols, mtl_buffer, mtl_fbs));

            if (leg_summed)                         // the front reflection is inside the back reflection's series
                kill();                             // this ends the internal series (code 0)
            else if (sheet_case != SheetCase::none) // a sheet: the composite carries the reflect port
                sheet_event();                      // (codes 197, 199, 205, 207)
            else if (transparent_forward)           // a transparent pass cannot reflect
                kill();                             // (code 0)
            else if (mtl_current == 0)              // the entry reflection, the bare Fresnel copy-through
                ledger_set();                       // (code 129)
            else                                    // an internal back reflection (codes 137, 139, 141, 143)
            {
                ledger_add();
                face_at_vbs(mtl_current, mtl_next, (is_scalar ? 3 : 0), 0); // mirror at the VBS
                next_buffer = (short)buf_followed;                          // not summed: followed
                if (!followed && sum_series(mtl_current, mtl_next, mtl_prev, 0))
                {
                    next_current = (short)((mtl_current & word_mtl) | cur_summed | bent_carry); // keep the medium, summed
                    next_buffer = (short)0;
                    res_code[2] = true;
                }
                close_leg(mtl_current);
                ledger_set();
                res_code[3] = true; // inside
            }
        }
        else // Transmission / refraction pass, interaction_type {1, 2, 4, 5}
        {
            // The segment after this event ends at the destination: its length can be preloaded
            const bool ray_ends = n_hits == 1 || (n_hits == 2 && hit_code[2]);

            if (n_hits == 0 || !hit_code[0]) // No crossing: the leg runs on, identity interface
            {
                ledger_add();
                pass_through(mtl_current != 0); // transparent, ok (code 65)
                if (mtl_current != 0)
                    res_code[3] = true; // transparent, inside, ok (code 73)
            }
            else if (sheet_case != SheetCase::none) // Thin-sheet crossing (codes 69, 71, 77, 79)
                sheet_event();
            else if (hit_code == 3 || hit_code == 15 || hit_code == 31) // o-i
            {
                res_code[3] = true;   // inside (code 9)
                if (mtl_current == 0) // entry from air: xprmat and gain stay
                {
                    next_current = (short)(mtl_fbs | ((material(cols, mtl_fbs).m > 0.0) ? 0 : bent_enter));
                    const bool nonpar = !ray_ends && faces_not_parallel(F, S, p_normal_vecN, i_rayN, n_rayN);
                    next_prev = (short)(nonpar ? prev_nonparallel : 0); // air + flag

                    if (hit_code == 15 || hit_code == 31) // o-ii (degenerate geometry; for code 15 the order is random)
                    {
                        next_buffer = (short)mtl_sbs; // the second medium is deferred
                        res_code[0] = false;          // ok cleared: a deferred buffer
                    }

                    if (ray_ends) // preload the next leg
                    {
                        double dist_VV;
                        if (vbs_intersect(F, dir_next, D, N, &dist_VV))
                            ledger_set(dist_VV, len_face_to_dest);
                        else
                            ledger_set(len_face_to_dest);
                    }
                    else
                        ledger_set(); // entering a medium: the free-space length is dropped, the leg starts at the FBS
                }
                else // nested pass-through (o-i on an inside state): the ray stays in the current medium (code 72)
                {
                    transition(false, mtl_current, ray_ends);
                    next_buffer = (short)mtl_fbs;
                    res_code[0] = false; // ok cleared: a deferred buffer
                }
            }
            else if (hit_code == 1 || hit_code == 13 || hit_code == 29) // i-o
            {
                if (mtl_buffer == 0) // cavity exit (codes 1, 3, 5, 7)
                    transition(true, 0, ray_ends);
                else if (hit_code == 1) // a virtual i-i into the deferred medium
                {
                    if (same_medium(cols, mtl_buffer, mtl_fbs)) // the second medium embedded in the first: ignored
                        transition(false, mtl_current, ray_ends);
                    else // (codes 9, 11, 13, 15)
                        transition(false, mtl_buffer, ray_ends);
                }
                else // codes 13 and 29 with a deferred medium: ii-o, exit into air
                    transition(true, 0, ray_ends);
            }
            else if (hit_code == 23) // Corner o-i-o
            {
                if (mtl_current == 0) // in air: the corner is ignored
                {
                    ledger_add(len_to_face);
                    pass_through(false); // (code 65)
                }
                else // inside a medium: exit into air
                    transition(true, 0, ray_ends);
            }
            else if (hit_code == 21) // Corner i-o-i
            {
                if (mtl_buffer == 0)
                {
                    if (mtl_sbs == 0) // illegal
                        kill();
                    else // an air gap bounded by the two media: an i-i transition
                    {
                        const int mtl_exit = exit_mtl();
                        mtl_current = mtl_exit ? mtl_exit : mtl_current;
                        transition(false, mtl_sbs, ray_ends);
                    }
                }
                else if (mtl_current != 0) // current and deferred medium
                {
                    if (same_medium(cols, mtl_buffer, mtl_fbs)) // the second medium embedded in the first: ignored
                        transition(false, mtl_current, ray_ends);
                    else
                        transition(false, mtl_buffer, ray_ends);
                }
                else // a deferred medium in air: terminate
                    kill();
            }
            else if (hit_code == 5 || hit_code == 7) // i-i
            {
                if (mtl_buffer == 0)
                    transition(false, 0, ray_ends);
                else
                {
                    transition(false, mtl_current, ray_ends);                                          // stay in the current medium
                    next_buffer = (short)(same_medium(cols, mtl_buffer, mtl_fbs) ? mtl_sbs : mtl_fbs); // swap the deferred medium
                    res_code[0] = false;                                                               // ok cleared: a deferred buffer
                }
            }
            else // Unmatched
                kill();
        }

        // The relaunch offset in the ledger, where the leg goes on inside a medium from where this event relaunched
        // (relaunch_ledger_offset). A ledger preloaded to the destination (a segment ending inside a body) stays.
        if (ledger_at_relaunch && n_hits != 0 && (int)res_code != 0 && (next_current & word_mtl) != 0)
        {
            const bool at_sbs = n_hits >= 2 && dist3(F, S) < colocation_dist;
            double Nf[3] = {(double)p_normal_vecN[i_rayN + (at_sbs ? 3 : 0) * n_rayN], (double)p_normal_vecN[i_rayN + (at_sbs ? 4 : 1) * n_rayN],
                            (double)p_normal_vecN[i_rayN + (at_sbs ? 5 : 2) * n_rayN]};
            double dn[3] = {dir_next[0], dir_next[1], dir_next[2]};
            if (p_path_dirN)
                dn[0] = (double)p_path_dirN[i_rayN], dn[1] = (double)p_path_dirN[i_rayN + n_rayN], dn[2] = (double)p_path_dirN[i_rayN + 2 * n_rayN];
            relaunch_ledger_offset(F, O, Nf, dn, interaction_type == 2 || interaction_type == 5, dist_geo, dist_refract);
        }

        // The state words, the ledger, the codes and the verdict (compact set)
        if (p_prev_outN)
            p_prev_outN[i_rayN] = next_prev;
        if (p_cur_outN)
            p_cur_outN[i_rayN] = next_current;
        if (p_buf_outN)
            p_buf_outN[i_rayN] = next_buffer;
        if (p_acc_geo_outN)
            p_acc_geo_outN[i_rayN] = (dtype)dist_geo;
        if (p_acc_refract_outN)
            p_acc_refract_outN[i_rayN] = (dtype)dist_refract;
        if (p_resolved_typeN)
            p_resolved_typeN[i_rayN] = (uint8_t)res_code;
        if (p_verdict_outN)
            p_verdict_outN[i_rayN] = (uint8_t)series_summed;
    }
}

template void quadriga_lib::ray_state_update(int interaction_type, float center_frequency,
                                             const arma::Mat<float> *orig, const arma::Mat<float> *dest,
                                             const arma::Mat<float> *fbsN, const arma::Mat<float> *sbsN,
                                             const arma::u32_vec *no_interact, const arma::Col<float> *fbs_angleN,
                                             const arma::Mat<float> *normal_vecN, const std::vector<uint8_t> *out_typeN,
                                             const std::unordered_map<std::string, std::vector<float>> *mtl_prop,
                                             const arma::Col<short> *mtl_ind_fbs, const arma::Col<short> *mtl_ind_sbs,
                                             const arma::Col<short> *mtl_ind_prev_in, const arma::Col<short> *mtl_ind_current_in,
                                             const arma::Col<short> *mtl_ind_buffer_in,
                                             const arma::Mat<float> *path_dir_prev,
                                             const arma::Col<float> *acc_dist_geometric, const arma::Col<float> *acc_dist_refracted,
                                             arma::Col<short> *mtl_ind_prev_outN, arma::Col<short> *mtl_ind_current_outN,
                                             arma::Col<short> *mtl_ind_buffer_outN,
                                             arma::Col<float> *gainN, arma::Mat<float> *xprmatN,
                                             arma::Mat<float> *path_dirN,
                                             arma::Col<float> *acc_dist_geometricN, arma::Col<float> *acc_dist_refractedN,
                                             std::vector<uint8_t> *resolved_typeN,
                                             const arma::u32_vec *ray_indN, double eps,
                                             arma::Mat<float> *origN, arma::Mat<float> *destN,
                                             const arma::Mat<float> *tridir, arma::Mat<float> *tridirN,
                                             const std::vector<uint8_t> *verdict_inN, std::vector<uint8_t> *verdict_outN);

template void quadriga_lib::ray_state_update(int interaction_type, double center_frequency,
                                             const arma::Mat<double> *orig, const arma::Mat<double> *dest,
                                             const arma::Mat<double> *fbsN, const arma::Mat<double> *sbsN,
                                             const arma::u32_vec *no_interact, const arma::Col<double> *fbs_angleN,
                                             const arma::Mat<double> *normal_vecN, const std::vector<uint8_t> *out_typeN,
                                             const std::unordered_map<std::string, std::vector<double>> *mtl_prop,
                                             const arma::Col<short> *mtl_ind_fbs, const arma::Col<short> *mtl_ind_sbs,
                                             const arma::Col<short> *mtl_ind_prev_in, const arma::Col<short> *mtl_ind_current_in,
                                             const arma::Col<short> *mtl_ind_buffer_in,
                                             const arma::Mat<double> *path_dir_prev,
                                             const arma::Col<double> *acc_dist_geometric, const arma::Col<double> *acc_dist_refracted,
                                             arma::Col<short> *mtl_ind_prev_outN, arma::Col<short> *mtl_ind_current_outN,
                                             arma::Col<short> *mtl_ind_buffer_outN,
                                             arma::Col<double> *gainN, arma::Mat<double> *xprmatN,
                                             arma::Mat<double> *path_dirN,
                                             arma::Col<double> *acc_dist_geometricN, arma::Col<double> *acc_dist_refractedN,
                                             std::vector<uint8_t> *resolved_typeN,
                                             const arma::u32_vec *ray_indN, double eps,
                                             arma::Mat<double> *origN, arma::Mat<double> *destN,
                                             const arma::Mat<double> *tridir, arma::Mat<double> *tridirN,
                                             const std::vector<uint8_t> *verdict_inN, std::vector<uint8_t> *verdict_outN);