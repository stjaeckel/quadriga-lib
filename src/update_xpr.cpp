// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# xpr_update
Thread a polarization transfer (Jones) matrix along a ray path

- Left-multiplies each per-ray state matrix by a new interaction matrix, vectorized across up to
  hundreds of millions of rays, then optionally normalizes the touched columns and re-imposes a
  separately tracked scalar gain — all in a single pass
- One entry point covers every operation needed while threading a chain: initialization, updating,
  gain extraction, normalization, and gain application. They are selected independently through
  `initialize`, `update`, `normalize`, `gain`, and `apply_gain`, so a call can do any combination
  (see Usage)
- An optional `ray_index` lets the update set be a subset of the global state, scattered into
  `xprmat` by column. Without it the mapping is 1:1 and `n_ray == n_rayU`
- Storage is interleaved complex, column-major, matching the `xprmat` output of
  [[calc_diffraction_gain]]. EM mode carries the full 2x2 Jones matrix; scalar mode carries a
  single complex pressure coefficient

## Declaration:
```
void xpr_update(
    arma::Mat<dtype> &xprmat,
    const arma::Mat<dtype> *update = nullptr,
    arma::Col<dtype> *gain = nullptr,
    bool initialize = false,
    bool normalize = false,
    bool apply_gain = false,
    const arma::uvec *ray_index = nullptr);
```

## Arguments:
- **`xprmat`** — Global XPR state, updated in place. EM mode `[8, n_ray]`, scalar mode
  `[2, n_ray]`. If empty on entry it is sized from `update` and seeded to unity (see Usage). A
  pre-sized (non-empty) buffer is treated as accumulated state unless `initialize` is set. When
  `ray_index` is given, `xprmat` must already exist, because the extent of the global set cannot be
  inferred from a subset.
- **`update`** — Interaction matrix left-multiplied onto each ray of the update set. Layout must
  match `xprmat` (8 or 2 rows). Shapes: `[8 or 2, n_rayU]` applies one matrix per update; `[8 or 2, 1]`
  broadcasts one matrix to every update; `nullptr` or empty applies the identity (no multiply).
- **`gain`** — Dual-purpose, length `n_rayU`; direction set by `apply_gain`:
  - `apply_gain == false` (output): resized to `[n_rayU]` and filled with each ray's gain *before*
    normalization. With `normalize == false` this measures the gain without modifying the matrix
    part of `xprmat` (gain extraction); with `normalize == true` it reports the gain that was
    removed. A value of 0 flags a degenerate (zero / non-finite) matrix the caller should prune.
  - `apply_gain == true` (input): read, not resized; each touched column is multiplied by the
    corresponding `gain` value *after* normalization. Must be non-null and length `n_rayU`.
- **`initialize`** — If `true`, each touched column is set to unity (identity for EM, `[1,0]` for
  scalar) before the update. With an `update` this is a copy-through (the column becomes the
  update, prior contents discarded); without an `update` it is a pure reset. Use this to
  (re)initialize a pre-sized or reused buffer without reallocating.
- **`normalize`** — If `true`, each touched column is scaled to unit gain after the update.
  Untouched columns of `xprmat` are left alone.
- **`apply_gain`** — Flips `gain` from output to input (see above). Applying a gain requires a `gain` vector.
- **`ray_index`** — Optional map from update index (`0..n_rayU-1`) to global column of `xprmat`.
  Length `n_rayU`. Default is the identity map, which requires `n_ray == n_rayU`. Values must be
  `< n_ray`.

## Usage:
The flags are composable in one call; the per-column order is always
initialize -> update -> normalize -> apply_gain. Argument order is
`(xprmat, update, gain, initialize, normalize, apply_gain, ray_index)`.
```
arma::Mat<dtype> M;                                 // ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH (EM)

// (1) Initialization
xpr_update(M, &U0);                                 // empty M: sized from U0, seeded, M := U0
xpr_update(M, nullptr, nullptr, true);              // pre-sized M: reset every column to unity
xpr_update(M, &U0, nullptr, true);                  // pre-sized M: copy-through, M := U0 (ignores old)

// (2) Updating (accumulate onto existing state)
xpr_update(M, &U);                                  // M := U * M
xpr_update(M, &Usub, nullptr, false, false, false, &idx);   // scatter: M[:,idx[k]] := Usub[:,k] * M[:,idx[k]]

// (3) Gain extraction (measure, leave the matrix part untouched)
arma::Col<dtype> g;
xpr_update(M, nullptr, &g);                         // g[k] = gain( M[:,k] )
xpr_update(M, &U, &g);                              // apply U, then measure into g (no normalize)

// (4) Normalizing (scale each column to unit gain; optionally update first)
xpr_update(M, nullptr, nullptr, false, true);       // M[:,k] := M[:,k] / gain( M[:,k] )
xpr_update(M, &U, nullptr, false, true);            // update, then normalize, one pass

// (5) Normalizing + gain application (strip the matrix's own magnitude, impose a tracked scalar).
//     'g' is read here (apply_gain = true):
xpr_update(M, nullptr, &g, false, true, true);      // M[:,k] := g[k] * ( M[:,k] / gain( M[:,k] ) )
xpr_update(M, &U, &g, false, true, true, &idx);     // full chain step: scatter+update+normalize+apply
```

## Storage layout (column-major, interleaved complex):
- **EM mode**, 8 values per ray: `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]`, so column 0 of the
  2x2 is `[VV; HV]` (response to a V input) and column 1 is `[VH; HH]` (response to an H input).
- **Scalar mode**, 2 values per ray: `[Re Im]`.
- The update is `M_out = M_update * M_state` (new interaction left-multiplies), consistent with
  `E_out = M * E_in` for column vectors and Armadillo/MATLAB column-major storage.

## See also:
- [[calc_diffraction_gain]] (produces `xprmat` in this exact layout)
MD!*/

// OpenMP is only worthwhile once the ray count amortizes fork/join. Below this the serial path
// runs with no team setup. Tune to the target machine; the value is a floor, not a promise.
#ifndef QD_XPR_OMP_MIN
#define QD_XPR_OMP_MIN 2048
#endif

namespace
{
    // Left-multiply one ray's Jones matrix in place: oX <- nX * oX (column-major, see doc block).
    // nX == nullptr initializes oX to the identity. All old values are cached before the first
    // store, so nX may alias oX. scalar_mode selects the 2-value vs 8-value layout.
    template <typename dtype>
    static inline void update_one(dtype *oX, const dtype *nX, bool scalar_mode)
    {
        if (nX) // Multiply matrices
        {
            dtype oReVV = oX[0], oImVV = oX[1]; // VV old
            dtype nReVV = nX[0], nImVV = nX[1]; // VV update
            if (scalar_mode)
            {
                oX[0] = nReVV * oReVV - nImVV * oImVV;
                oX[1] = nReVV * oImVV + nImVV * oReVV;
            }
            else // EM mode
            {
                dtype oReHV = oX[2], oImHV = oX[3]; // HV old
                dtype nReHV = nX[2], nImHV = nX[3]; // HV update
                dtype oReVH = oX[4], oImVH = oX[5]; // VH old
                dtype nReVH = nX[4], nImVH = nX[5]; // VH update
                dtype oReHH = oX[6], oImHH = oX[7]; // HH old
                dtype nReHH = nX[6], nImHH = nX[7]; // HH update
                oX[0] = nReVV * oReVV - nImVV * oImVV + nReVH * oReHV - nImVH * oImHV;
                oX[1] = nReVV * oImVV + nImVV * oReVV + nReVH * oImHV + nImVH * oReHV;
                oX[2] = nReHV * oReVV - nImHV * oImVV + nReHH * oReHV - nImHH * oImHV;
                oX[3] = nReHV * oImVV + nImHV * oReVV + nReHH * oImHV + nImHH * oReHV;
                oX[4] = nReVV * oReVH - nImVV * oImVH + nReVH * oReHH - nImVH * oImHH;
                oX[5] = nReVV * oImVH + nImVV * oReVH + nReVH * oImHH + nImVH * oReHH;
                oX[6] = nReHV * oReVH - nImHV * oImVH + nReHH * oReHH - nImHH * oImHH;
                oX[7] = nReHV * oImVH + nImHV * oReVH + nReHH * oImHH + nImHH * oReHH;
            }
        }
        else // Initialize to unity
        {
            oX[0] = (dtype)1.0, oX[1] = (dtype)0.0; // VV
            if (!scalar_mode)
            {
                oX[2] = (dtype)0.0, oX[3] = (dtype)0.0; // HV
                oX[4] = (dtype)0.0, oX[5] = (dtype)0.0; // VH
                oX[6] = (dtype)1.0, oX[7] = (dtype)0.0; // HH
            }
        }
    }

    // Copy one interaction matrix straight into a ray's slot: oX <- nX. Used for the copy-through
    // init, where unity * nX == nX, so the identity multiply is skipped. Does not read oX, so it is
    // safe on freshly allocated (uninitialized) storage.
    template <typename dtype>
    static inline void assign_one(dtype *oX, const dtype *nX, bool scalar_mode)
    {
        oX[0] = nX[0], oX[1] = nX[1];
        if (!scalar_mode)
            oX[2] = nX[2], oX[3] = nX[3], oX[4] = nX[4], oX[5] = nX[5], oX[6] = nX[6], oX[7] = nX[7];
    }

    // Multiply one ray's Jones matrix in place by a real scalar (uniform, shape-preserving).
    template <typename dtype>
    static inline void scale_one(dtype *oX, dtype s, bool scalar_mode)
    {
        oX[0] *= s, oX[1] *= s;
        if (!scalar_mode)
            oX[2] *= s, oX[3] *= s, oX[4] *= s, oX[5] *= s, oX[6] *= s, oX[7] *= s;
    }

    // Return one ray's gain (max column 2-norm). If apply == true, also scale the matrix to unit
    // gain in place. A degenerate matrix (squares underflow, or a NaN is present) returns 0; if
    // apply, it is reset to the identity so the state stays well-formed.
    template <typename dtype>
    static inline dtype gain_one(dtype *oX, bool scalar_mode, bool apply)
    {
        dtype A = oX[0] * oX[0] + oX[1] * oX[1], B = (dtype)0.0; // |col 0|^2
        if (!scalar_mode)
        {
            A += oX[2] * oX[2] + oX[3] * oX[3];
            B = oX[4] * oX[4] + oX[5] * oX[5] + oX[6] * oX[6] + oX[7] * oX[7]; // |col 1|^2
        }
        dtype gain = A > B ? A : B;

        // Guard in the squared domain: AB below the smallest normal means the squares have already
        // underflowed (float: entry magnitude < ~1.1e-19). Anything finite and >= that is safe, and
        // every element is bounded by sqrt(AB), so the scaled result cannot overflow. NaN fails the
        // comparison and lands in the degenerate branch instead of laundering through the divide.
        if (gain >= std::numeric_limits<dtype>::min())
        {
            dtype amplitude = std::sqrt(gain);
            if (apply)
            {
                dtype scale = (dtype)1.0 / amplitude;
                oX[0] *= scale, oX[1] *= scale;
                if (!scalar_mode)
                    oX[2] *= scale, oX[3] *= scale, oX[4] *= scale, oX[5] *= scale, oX[6] *= scale, oX[7] *= scale;
            }
            return gain;
        }

        // Degenerate: a fully absorbing interaction drove the matrix to zero, or a NaN slipped in.
        // Reset to identity (only if we are modifying the state) and report zero gain; the caller
        // should treat gain == 0 as "prune this ray".
        if (apply)
            update_one<dtype>(oX, (const dtype *)nullptr, scalar_mode);
        return (dtype)0.0;

        // ISSUE: max column 2-norm is not the matrix gain.
        //
        // "gain" above is max(|col 0|, |col 1|), the larger of the responses to a pure V and a
        // pure H input. The gain of a Jones matrix is its largest singular value sigma_max, the
        // response to the worst-case input polarization -- which is generally elliptical, so
        // sampling only V and H can miss it:
        //
        //     sigma_max / sqrt(2) <= max(|col 0|, |col 1|) <= sigma_max
        //
        // The two agree only when the columns are orthogonal (M^H M diagonal). That holds for a
        // bare Fresnel matrix diag(r_perp, r_par) in its own incidence basis, but not for an
        // accumulated chain that mixes bases. Example: diag(1, 0.1) * R(45 deg) has sigma_max = 1
        // exactly, yet both column norms are 0.7106. Dividing by that leaves a matrix of gain
        // 1.407 -- 3 dB of energy created from nothing, from one reflection and one basis rotation.
        //
        // Whether this matters depends on what the caller does with the gain:
        //
        //   (a) Extract it (mode 3) and later re-impose that same vector via apply_gain (mode 5).
        //       Removal and re-application share one scalar, so M is restored exactly and the norm
        //       choice is irrelevant -- uniform scaling out, the same uniform scaling back in.
        //       This is the intended lossless round-trip.
        //
        //   (b) Normalize and impose a *different*, independently tracked gain g (mode 5). The
        //       normalized matrix has max col norm 1, so the result has max col norm g exactly --
        //       but sigma_max between g and g*sqrt(2). If g is meant to be the true field gain
        //       (sigma_max), the field is up to 3 dB high. Use the exact form below to make the
        //       normalized matrix unit-sigma_max so that g lands on sigma_max.
        //
        // Exact sigma_max costs two sqrt and ten mul, via the 2x2 identity
        // sigma_max +/- sigma_min = sqrt(F +/- 2|det M|), with F = ||M||_F^2 = A + B:
        //
        //     dtype F   = A + B;
        //     dtype dRe = oX[0]*oX[6] - oX[1]*oX[7] - oX[4]*oX[2] + oX[5]*oX[3];
        //     dtype dIm = oX[0]*oX[7] + oX[1]*oX[6] - oX[4]*oX[3] - oX[5]*oX[2];
        //     dtype d2  = (dtype)2.0 * std::sqrt(dRe*dRe + dIm*dIm);    // 2 * |det M|
        //     dtype lo  = F - d2;                                       // >= 0 analytically (AM-GM)
        //     lo = lo < (dtype)0.0 ? (dtype)0.0 : lo;                   // rounding can dip below 0
        //     dtype gain = (dtype)0.5 * (std::sqrt(F + d2) + std::sqrt(lo));
        //
        // Spot checks: identity -> 1, R(45 deg) -> 1, diag(1, 0) -> 1, diag(1, 0.1)*R(45) -> 1.
        // F - d2 cancels for near-unitary M, which is the common case; this costs about sqrt(eps)
        // relative accuracy in the sigma_min term (~3e-4 in float, negligible in double), and can
        // be removed by accumulating F and d2 in double. Scalar mode is unaffected: |c| is exact.
        //
        // Cheaper conservative variant: sqrt(F) (Frobenius) satisfies sigma_max <= sqrt(F) <=
        // sqrt(2)*sigma_max, so it can only lose up to 3 dB, never create it. The current max
        // column norm errs in the direction that creates energy.
    }
}

template <typename dtype>
void quadriga_lib::xpr_update(arma::Mat<dtype> &xprmat,
                              const arma::Mat<dtype> *update,
                              arma::Col<dtype> *gain,
                              bool initialize,
                              bool normalize,
                              bool apply_gain,
                              const arma::uvec *ray_index)
{
    const bool have_state = !xprmat.is_empty();
    const bool have_update = (update != nullptr) && !update->is_empty();

    // Infer mode (EM = 8 rows, scalar = 2) from whichever operand is present.
    arma::uword nXPR = 0;
    if (have_state)
        nXPR = xprmat.n_rows;
    else if (have_update)
        nXPR = update->n_rows;
    else
        throw std::invalid_argument("At least one of 'xprmat' or 'update' must be non-empty to infer the mode.");

    if (nXPR != 2 && nXPR != 8)
        throw std::invalid_argument("'xprmat'/'update' must have 2 rows (scalar mode) or 8 rows (EM mode).");
    const bool scalar_mode = (nXPR == 2);

    // Interpret the update shape.
    bool broadcast = false;
    if (have_update)
    {
        if (update->n_rows != nXPR)
            throw std::invalid_argument("'update' row count must match 'xprmat' (2 or 8).");
        broadcast = (update->n_cols == 1);
    }

    // Resolve the two index-space sizes: n_ray (columns of the global 'xprmat') and n_rayU (the
    // update set, i.e. the loop length and the length of 'ray_index' / 'gain').
    arma::uword n_ray = 0, n_rayU = 0;

    if (ray_index)
    {
        // Scatter into an existing global buffer. The buffer must already exist because a subset
        // does not reveal the global extent.
        if (!have_state)
            throw std::invalid_argument("'ray_index' requires an existing (non-empty) 'xprmat'.");
        n_ray = xprmat.n_cols;
        n_rayU = ray_index->n_elem;

        if (have_update && !broadcast && update->n_cols != n_rayU)
            throw std::invalid_argument("'update' must have 'ray_index' columns (or 1 to broadcast).");

        if (n_rayU > 0 && ray_index->max() >= n_ray) // guard: max() on an empty uvec is undefined
            throw std::invalid_argument("'ray_index' contains a column index outside 'xprmat'.");
    }
    else if (have_state)
    {
        // 1:1 mapping onto the existing buffer.
        n_ray = xprmat.n_cols;
        n_rayU = n_ray;
        if (have_update && !broadcast && update->n_cols != n_ray)
            throw std::invalid_argument("'update' must have 'xprmat' columns (or 1 to broadcast).");
    }
    else
    {
        // Empty 'xprmat', no 'ray_index': size the global buffer from a per-column update.
        if (!have_update)
            throw std::invalid_argument("Cannot size an empty 'xprmat' without a per-column 'update'; pre-size it to use 'initialize' on its own.");
        if (broadcast)
            throw std::invalid_argument("Cannot infer ray count from a broadcast 'update' with an empty 'xprmat'.");
        n_ray = update->n_cols;
        n_rayU = n_ray;
    }

    // Resolve the direction of 'gain'. With apply_gain it is an input (read, applied after
    // normalization) and must match n_rayU; otherwise it is an output that we size and fill.
    if (apply_gain)
    {
        if (gain == nullptr)
            throw std::invalid_argument("'apply_gain' is set but no 'gain' vector was provided.");
        if (gain->n_elem != n_rayU)
            throw std::invalid_argument("'gain' length must equal n_rayU when 'apply_gain' is set.");
    }
    else if (gain)
        gain->set_size(n_rayU);

    if (n_rayU == 0) // Nothing to do; outputs already sized.
        return;

    // Allocate global storage when starting from empty. Freshly allocated memory is uninitialized,
    // but the copy-through / identity seed below writes every touched column without reading it, so
    // no garbage is consumed. (Empty implies every column is touched, so all get written.)
    if (!have_state)
        xprmat.set_size(nXPR, n_ray);

    // Seed a column to unity before the update when the caller asks (initialize) or when the buffer
    // was just allocated and holds no valid state. With an update this becomes a direct copy.
    const bool seed = initialize || !have_state;

    dtype *p_xpr = xprmat.memptr();
    const dtype *p_upd = have_update ? update->memptr() : nullptr;
    const arma::uword *p_idx = ray_index ? ray_index->memptr() : nullptr;
    dtype *p_gain = (gain != nullptr) ? gain->memptr() : nullptr;

    const bool write_gain = (p_gain != nullptr) && !apply_gain; // gain is an output
    const bool need_measure = normalize || write_gain;          // must compute the column norm

    // Per-update work. Order per column: initialize -> update -> normalize -> apply_gain.
    //   seed &&  update : column := update            (copy-through; unity * update == update)
    //   seed && !update : column := unity             (pure reset)
    //  !seed &&  update : column := update * column   (accumulate onto existing state)
    //  !seed && !update : column unchanged            (normalize / gain only)
    // Identical in both loops below.
    auto process = [=](arma::uword iU)
    {
        arma::uword g = p_idx ? p_idx[iU] : iU; // global column in xprmat
        dtype *oX = &p_xpr[nXPR * g];
        const dtype *nX = p_upd ? (broadcast ? p_upd : &p_upd[nXPR * iU]) : (const dtype *)nullptr;

        if (seed)
        {
            if (nX)
                assign_one<dtype>(oX, nX, scalar_mode); // copy-through init
            else
                update_one<dtype>(oX, (const dtype *)nullptr, scalar_mode); // reset to unity
        }
        else if (nX)
            update_one<dtype>(oX, nX, scalar_mode); // accumulate

        if (need_measure) // Gain extraction and/or normalization
        {
            dtype gv = gain_one<dtype>(oX, scalar_mode, /*apply=*/normalize);
            if (write_gain)
                p_gain[iU] = gv;
        }

        if (apply_gain) // Re-impose a separately tracked scalar gain (read from *gain)
            scale_one<dtype>(oX, p_gain[iU], scalar_mode);
    };

    // Single OMP region, guarded so small update sets (including the 1-ray-in-a-loop case) run
    // serially with no team setup. Updates are independent, so the loop is embarrassingly parallel.
    // Signed counter for OpenMP 2.0 (MSVC), which forbids an unsigned loop variable.
    const long long N = (long long)n_rayU;
    if (n_rayU >= (arma::uword)QD_XPR_OMP_MIN)
    {
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < N; ++i)
            process((arma::uword)i);
    }
    else
    {
        for (long long i = 0; i < N; ++i)
            process((arma::uword)i);
    }
}

template void quadriga_lib::xpr_update(arma::Mat<float> &xprmat, const arma::Mat<float> *update,
                                       arma::Col<float> *gain, bool initialize, bool normalize,
                                       bool apply_gain, const arma::uvec *ray_index);

template void quadriga_lib::xpr_update(arma::Mat<double> &xprmat, const arma::Mat<double> *update,
                                       arma::Col<double> *gain, bool initialize, bool normalize,
                                       bool apply_gain, const arma::uvec *ray_index);