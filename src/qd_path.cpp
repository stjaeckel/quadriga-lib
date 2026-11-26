// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"
#include "qd_helpers_shared.hpp"

#include <cstring>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <cmath>
#include <limits>
#include <vector>

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# path
Class for one propagation path: interaction points, transfer coefficients and interaction types

- Holds one path from an origin to a destination through up to 255 interaction points (segments); the origin and the
  destination themselves are not stored
- Per frequency (1 to 127) one transfer coefficient: in EM mode the polarization transfer matrix (8 floats, Glossary
  "Polarization"), in scalar mode (e.g. acoustic) one complex pressure coefficient (2 floats); per interaction point
  one interaction type code ([[ray_state_update]] lists the codes)
- Compact for large path sets: the 56-byte object holds the counters, the first 2 type codes and the coefficients of
  frequency 0 (scalar mode: frequencies 0 to 3); a buffer on the heap holds the points and the other coefficients and
  codes
- The ray tracer ([[ray_progress]]) builds paths, [[qrt_file_append]] stores them in QRT files and
  [[qrt_file_read_raw]] reads them back
- Copyable and movable; a moved-from object has 0 points, 1 frequency and EM mode

## Attributes:<br>
| Attribute | Size | Description |
| --------- | ---- | ----------- |
| `unsigned i_cir` | scalar | CIR index: the CIR the path belongs to |
| `uint8_t n_ref` | scalar | Number of reflections |
| `uint8_t n_tra` | scalar | Number of transmissions (refractions) |
| `uint8_t n_sub` | scalar | Number of subdivisions |
| `uint8_t n_sct` | scalar | Number of scattering events |

## Member functions:<br>
| Function | Description |
| -------- | ----------- |
| `path()` | EM path without points, 1 frequency |
| `path(segments, frequencies, scalar)` | Path as set by `.init()` |
| `.init(segments = 0, frequencies = 1, scalar = false)` | Resets the path: `segments` points at (0, 0, 0) (at most 255), `frequencies` coefficients (1 to 127) of value one (identity matrix in EM mode), type codes, counters, `i_cir` and length 0 |
| `.free()` | Releases the buffer: 0 points, 1 frequency, EM mode; the other members keep their values |
| `.n_seg()`, `.n_freq()`, `.is_scalar()` | Number of points, number of frequencies, `true` in scalar mode |
| `.coord(seg = 0)` | Pointer to `[x, y, z]` of point `seg` in m (0-based, below `n_seg()`) |
| `operator()(i)` | Coordinate `i` (0 = x, 1 = y, 2 = z) of the last point; NaN for a path without points or `i` > 2 |
| `.polmat(i_freq = 0)` | Pointer to the coefficients of frequency `i_freq` (0-based, below `n_freq()`): 8 floats (EM, column-major, real and imaginary parts interleaved) or 2 (scalar) |
| `.length()`, `.set_length(len)` | Stored length in m, from the origin to the last point |
| `.interaction_type_codes()` | Type codes, one per point; `std::vector<uint8_t>` |
| `.set_interaction_type_codes(codes)` | Sets the type codes; `n_seg()` codes |
| `.calc_length()`, `.calc_gain()`, `.polmat_update()`, `.duplicate()`, `.extend()` | See below |

## Function 'calc_length':
Length of the path from the origin through the interaction points to the destination `D`

```
float calc_length(float Dx, float Dy, float Dz) const;
float calc_length(float Dx, float Dy, float Dz, float Ox, float Oy, float Oz);
float calc_length(const float *D, const float *O = nullptr);
```
- Without the origin `O`: the stored length plus the distance from the last point to `D`; NaN for a path without
  points
- With `O`: computes all legs from `O`, stores the length from `O` to the last point and returns the total; for a
  path without points the distance from `O` to `D`
- `D`, `O`: positions `[x, y, z]` in m

## Function 'calc_gain':
Power gain of the path at one frequency

```
float calc_gain(float freq = 0.0f, size_t i_freq = 0, float len = 0.0f) const;
```
- The coefficient power $P$: in EM mode the larger column power of the matrix, in scalar mode $\lvert c \rvert^2$. A
  path whose coefficients hold NaN or whose column powers sum to less than the smallest normal float gets 0, so that it
  can be pruned
- With `freq` > 0 and a length $d$ > 0, the free-space loss is applied: $P \, (\lambda / (4 \pi d))^2$ with
  $\lambda = c / f$ in EM mode, $P / d^2$ in scalar mode (the value of `freq` is not used there)
- `freq` in Hz; `i_freq` frequency index, 0-based, below `n_freq()`; `len` in m: > 0 replaces the stored length as
  $d$

## Function 'polmat_update':
Apply an interaction to the coefficients of one frequency and return the new gain

```
float polmat_update(const float *coeff_update = nullptr, float gain_update = 1.0f, size_t i_freq = 0,
                    float freq = 0.0f, float len = 0.0f);
```
- Left-multiplies the coefficients $M$ of frequency `i_freq` by the interaction $U$ and scales them by the amplitude
  of the power gain $g$: $M \leftarrow \sqrt{\lvert g \rvert} \, U M$
- `coeff_update`: $U$, 8 floats (EM) or 2 (scalar), not overlapping the coefficients of the path; `nullptr` = no
  multiplication. `gain_update`: $g$; 1 = no scaling
- Returns `calc_gain(freq, i_freq, len)` of the new coefficients; with path loss after the last `extend`, when the
  stored length is final

## Function 'duplicate':
Copy the path into another object

```
float duplicate(path &target) const;
```
- `target` (not the path itself) gets an independent copy; returns the stored length

## Function 'extend':
Copy the path with one more interaction point into another object

```
float extend(path &target, float x, float y, float z, uint8_t type = 0) const;
```
- `target` (not the path itself) gets a copy with the point `[x, y, z]` in m and its type code `type` appended; the
  counters are copied unchanged, the caller updates them
- The stored length grows by the distance from the last point to the new one; the first point adds nothing, since the
  origin is not stored (`calc_length` with the origin or `set_length` sets it)
- Returns the new stored length; a path with 255 points throws `std::runtime_error`
MD!*/

#define INLINE_TYPES 2 // Length of the inline type codes

// Offset of the interaction history block in the data buffer, in floats
constexpr static size_t history_offset(size_t segments, size_t frequencies, bool scalar)
{
    return scalar ? segments * 3 + (frequencies < 5 ? 0 : (frequencies - 4) * 2)
                  : segments * 3 + (frequencies < 2 ? 0 : (frequencies - 1) * 8);
}

// Size of the data buffer in floats for the given number of segments and frequencies
constexpr static size_t data_size(size_t segments, size_t frequencies, bool scalar)
{
    size_t history_length = segments <= INLINE_TYPES ? 0 : (segments - INLINE_TYPES + 3) / 4;
    return history_offset(segments, frequencies, scalar) + history_length;
}

// Gain (linear power) from the Jones matrix values; freq (Hz) > 0 applies the path loss
// Scalar mode passes the pressure coefficient as [ReVV, ImVV] and zeros for the rest
static inline float gain_from_coeff(float ReVV, float ImVV, float ReHV, float ImHV,
                                    float ReVH, float ImVH, float ReHH, float ImHH,
                                    bool scalar, float path_length, float freq)
{
    float A = ReVV * ReVV + ImVV * ImVV + ReHV * ReHV + ImHV * ImHV; // |col 0|^2
    float B = ReVH * ReVH + ImVH * ImVH + ReHH * ReHH + ImHH * ImHH; // |col 1|^2
    float P = A > B ? A : B;

    // Degenerate: NaN in either column, or both columns underflowed. A + B is NaN if either
    // part is NaN, and < min only when the true max is also below min.
    if (!(A + B >= std::numeric_limits<float>::min()))
        return 0.0f;

    // Path loss (linear power): EM = (lambda / (4 pi d))^2, lambda = c / f; scalar = 1 / d^2
    if (freq > 0.0f && path_length > 0.0f)
    {
        if (scalar)
            P /= path_length * path_length;
        else
        {
            float k = (float)qd_speed_of_light / (12.566370614f * freq * path_length);
            P *= k * k;
        }
    }
    return P;
}

quadriga_lib::path::path(path &&other) noexcept
    : nFRQ(other.nFRQ), nSEG(other.nSEG), len_acc(other.len_acc), data(other.data),
      i_cir(other.i_cir), n_ref(other.n_ref), n_tra(other.n_tra), n_sub(other.n_sub), n_sct(other.n_sct)

{
    std::memcpy(interact_type, other.interact_type, sizeof(interact_type));
    std::memcpy(xprmat, other.xprmat, sizeof(xprmat));
    other.data = nullptr;
    other.nFRQ = 1;
    other.nSEG = 0;
}

quadriga_lib::path::path(const path &other)
    : nFRQ(other.nFRQ), nSEG(other.nSEG), len_acc(other.len_acc),
      i_cir(other.i_cir), n_ref(other.n_ref), n_tra(other.n_tra), n_sub(other.n_sub), n_sct(other.n_sct)

{
    std::memcpy(interact_type, other.interact_type, sizeof(interact_type));
    std::memcpy(xprmat, other.xprmat, sizeof(xprmat));

    if (other.data)
    {
        size_t n = data_size(other.nSEG, other.n_freq(), other.is_scalar());
        data = new float[n];
        std::memcpy(data, other.data, n * sizeof(float));
    }
}

quadriga_lib::path &quadriga_lib::path::operator=(path &&other) noexcept
{
    if (this == &other)
        return *this;
    delete[] data;

    nFRQ = other.nFRQ, nSEG = other.nSEG;
    std::memcpy(interact_type, other.interact_type, sizeof(interact_type));
    std::memcpy(xprmat, other.xprmat, sizeof(xprmat));

    data = other.data;

    i_cir = other.i_cir;
    n_ref = other.n_ref, n_tra = other.n_tra, n_sub = other.n_sub, n_sct = other.n_sct;
    len_acc = other.len_acc;

    other.data = nullptr;
    other.nFRQ = 1;
    other.nSEG = 0;

    return *this;
}

quadriga_lib::path &quadriga_lib::path::operator=(const path &other)
{
    if (this == &other)
        return *this;

    nFRQ = other.nFRQ, nSEG = other.nSEG;
    std::memcpy(interact_type, other.interact_type, sizeof(interact_type));
    std::memcpy(xprmat, other.xprmat, sizeof(xprmat));

    size_t n = other.data ? data_size(other.nSEG, other.n_freq(), other.is_scalar()) : 0;
    float *buf = nullptr;
    if (n)
    {
        buf = new float[n];
        std::memcpy(buf, other.data, n * sizeof(float));
    }
    delete[] data;
    data = buf;

    i_cir = other.i_cir;
    n_ref = other.n_ref, n_tra = other.n_tra, n_sub = other.n_sub, n_sct = other.n_sct;
    len_acc = other.len_acc;

    return *this;
}

// Manually clear data buffer
void quadriga_lib::path::free()
{
    delete[] data;
    data = nullptr;
    nFRQ = 1;
    nSEG = 0;
}

// Initialize data storage to a given layout
void quadriga_lib::path::init(size_t segments, size_t frequencies, bool scalar)
{
    if (segments > 255)
        throw std::invalid_argument("Input 'segments' cannot exceed 255.");
    if (frequencies == 0 || frequencies > 127)
        throw std::invalid_argument("Input 'frequencies' must be between 1 and 127.");

    free();

    nFRQ = uint8_t(frequencies | (scalar ? 0x80u : 0x00u));
    nSEG = (uint8_t)segments;

    for (size_t n = 0; n < INLINE_TYPES; ++n)
        interact_type[n] = 0;

    xprmat[0] = 1.0f;                 // ReVV or Re(F0)
    xprmat[1] = 0.0f;                 // ImVV or Im(F0)
    xprmat[2] = scalar ? 1.0f : 0.0f; // ReHV or Re(F1)
    xprmat[3] = 0.0f;                 // ImHV or Im(F1)
    xprmat[4] = scalar ? 1.0f : 0.0f; // ReVH or Re(F2)
    xprmat[5] = 0.0f;                 // ImVH or Im(F2)
    xprmat[6] = 1.0f;                 // ReHH or Re(F3)
    xprmat[7] = 0.0f;                 // ImHH or Im(F3)

    size_t n = data_size(segments, frequencies, scalar);
    if (n)
    {
        data = new float[n](); // Init to 0
        if (scalar)
            for (size_t f = 4; f < frequencies; ++f)
                data[segments * 3 + (f - 4) * 2] = 1.0f;
        else // EM
            for (size_t f = 1; f < frequencies; ++f)
            {
                float *j = data + segments * 3 + (f - 1) * 8;
                j[0] = 1.0f, j[6] = 1.0f;
            }
    }

    i_cir = 0;
    n_ref = 0;
    n_tra = 0;
    n_sub = 0;
    n_sct = 0;

    len_acc = 0.0f;
}

quadriga_lib::path::path(size_t segments, size_t frequencies, bool scalar)
{
    init(segments, frequencies, scalar);
}

// Access coordinates of a specific segment
const float *quadriga_lib::path::coord(size_t seg) const
{
    if (seg >= (size_t)nSEG)
        throw std::invalid_argument("Input 'seg' is out of range.");
    return data ? data + seg * 3 : nullptr;
}
float *quadriga_lib::path::coord(size_t seg)
{
    return const_cast<float *>(std::as_const(*this).coord(seg));
}

// Member length + final leg to D (read-only)
float quadriga_lib::path::calc_length(float Dx, float Dy, float Dz) const
{
    if (nSEG == 0)
        return NAN;
    const float *last = data + (size_t(nSEG) - 1) * 3;
    return len_acc + qd_calc_length(last[0], last[1], last[2], Dx, Dy, Dz);
}

// Full recalculation O -> seg[0] -> ... -> seg[nSEG-1] -> D.
// Stores O..last into member length (matching extend's convention, excludes -> D),
// returns the full length including -> D.
float quadriga_lib::path::calc_length(float Dx, float Dy, float Dz, float Ox, float Oy, float Oz)
{
    float total = 0.0f;
    float px = Ox, py = Oy, pz = Oz;
    for (size_t s = 0, nS = (size_t)nSEG; s < nS; ++s)
    {
        const float *c = data + s * 3;
        total += qd_calc_length(px, py, pz, c[0], c[1], c[2]);
        px = c[0], py = c[1], pz = c[2];
    }
    len_acc = total; // O -> last, excludes final leg to D

    return total + qd_calc_length(px, py, pz, Dx, Dy, Dz);
}

// 3-element overload; O = nullptr -> read-only member + last-to-D, else full recalc (updates member)
float quadriga_lib::path::calc_length(const float *D, const float *O)
{
    if (O)
        return calc_length(D[0], D[1], D[2], O[0], O[1], O[2]);
    return calc_length(D[0], D[1], D[2]);
}

// Access the coefficients of frequency index i_freq
const float *quadriga_lib::path::polmat(size_t i_freq) const
{
    if (i_freq >= n_freq())
        throw std::invalid_argument("Input 'i_freq' is out of range.");

    if (i_freq == 0)
        return xprmat;

    size_t offset = size_t(nSEG) * 3;
    if (is_scalar())
    {
        if (i_freq < 4)
            return &xprmat[2 * i_freq];
        else
            return data + offset + (i_freq - 4) * 2;
    }
    else // EM mode
        return data + offset + (i_freq - 1) * 8;
}
float *quadriga_lib::path::polmat(size_t i_freq)
{
    return const_cast<float *>(std::as_const(*this).polmat(i_freq));
}

// Left-multiply ray's polarization transfer matrix; coeff_update must have length 8 (EM) or 2 (scalar)
// + calculate gain from the coefficients; freq (Hz) > 0 applies the free-space loss
float quadriga_lib::path::polmat_update(const float *coeff_update, float gain_update, size_t i_freq, float freq, float len)
{
    float *oX = polmat(i_freq); // Bounds-checked; resolves to xprmat or into the data buffer
    const size_t n_coeff = is_scalar() ? 2 : 8;
    if (coeff_update && (std::uintptr_t)coeff_update < (std::uintptr_t)(oX + n_coeff) &&
        (std::uintptr_t)oX < (std::uintptr_t)(coeff_update + n_coeff))
        throw std::invalid_argument("Input 'coeff_update' cannot share memory with the coefficients of the path.");
    float amplitude_update = gain_update == 1.0f ? 1.0f : std::sqrt(std::abs(gain_update));
    float d = len > 0.0f ? len : len_acc;

    if (is_scalar()) // 2 values: [Re, Im]
    {
        float oRe = oX[0], oIm = oX[1];

        if (coeff_update) // oX = coeff_update * oX
        {
            float nRe = coeff_update[0], nIm = coeff_update[1];
            float tRe = nRe * oRe - nIm * oIm;
            oIm = nRe * oIm + nIm * oRe;
            oRe = tRe;
        }

        oRe *= amplitude_update, oIm *= amplitude_update;
        oX[0] = oRe, oX[1] = oIm;
        return gain_from_coeff(oRe, oIm, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, true, d, freq);
    }

    // EM mode, 8 values: [ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH], column-major, labels <out><in> (HV = H out, V in)
    float oReVV = oX[0], oImVV = oX[1];
    float oReHV = oX[2], oImHV = oX[3];
    float oReVH = oX[4], oImVH = oX[5];
    float oReHH = oX[6], oImHH = oX[7];

    if (coeff_update) // oX = coeff_update * oX
    {
        float nReVV = coeff_update[0], nImVV = coeff_update[1];
        float nReHV = coeff_update[2], nImHV = coeff_update[3];
        float nReVH = coeff_update[4], nImVH = coeff_update[5];
        float nReHH = coeff_update[6], nImHH = coeff_update[7];

        float A = nReVV * oReVV - nImVV * oImVV + nReVH * oReHV - nImVH * oImHV;
        float B = nReVV * oImVV + nImVV * oReVV + nReVH * oImHV + nImVH * oReHV;
        float C = nReHV * oReVV - nImHV * oImVV + nReHH * oReHV - nImHH * oImHV;
        float D = nReHV * oImVV + nImHV * oReVV + nReHH * oImHV + nImHH * oReHV;
        float E = nReVV * oReVH - nImVV * oImVH + nReVH * oReHH - nImVH * oImHH;
        float F = nReVV * oImVH + nImVV * oReVH + nReVH * oImHH + nImVH * oReHH;
        float G = nReHV * oReVH - nImHV * oImVH + nReHH * oReHH - nImHH * oImHH;
        float H = nReHV * oImVH + nImHV * oReVH + nReHH * oImHH + nImHH * oReHH;

        oReVV = A, oImVV = B, oReHV = C, oImHV = D;
        oReVH = E, oImVH = F, oReHH = G, oImHH = H;
    }

    oReVV *= amplitude_update, oImVV *= amplitude_update, oReHV *= amplitude_update, oImHV *= amplitude_update;
    oReVH *= amplitude_update, oImVH *= amplitude_update, oReHH *= amplitude_update, oImHH *= amplitude_update;

    oX[0] = oReVV, oX[1] = oImVV, oX[2] = oReHV, oX[3] = oImHV;
    oX[4] = oReVH, oX[5] = oImVH, oX[6] = oReHH, oX[7] = oImHH;

    return gain_from_coeff(oReVV, oImVV, oReHV, oImHV, oReVH, oImVH, oReHH, oImHH, false, d, freq);
}

// Calculate gain from the coefficients; freq (Hz) > 0 applies the free-space loss; len > 0 overrides stored length
float quadriga_lib::path::calc_gain(float freq, size_t i_freq, float len) const
{
    const float *cf = polmat(i_freq); // Bounds-checked; resolves to xprmat or into the data buffer
    float d = len > 0.0f ? len : len_acc;

    if (is_scalar()) // Only 2 values in the slot; reading cf[2..7] would overrun
        return gain_from_coeff(cf[0], cf[1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, true, d, freq);

    return gain_from_coeff(cf[0], cf[1], cf[2], cf[3], cf[4], cf[5], cf[6], cf[7], false, d, freq);
}

// Duplicate path, writing to "target", returns length
float quadriga_lib::path::duplicate(path &target) const
{
    if (&target == this)
        throw std::invalid_argument("Output 'target' cannot be the source path.");
    target = *this;
    return len_acc;
}

// Duplicate path and append a new segment
float quadriga_lib::path::extend(path &target, float x, float y, float z, uint8_t type) const
{
    if (&target == this)
        throw std::invalid_argument("Output 'target' cannot be the source path.");
    if (nSEG == 255)
        throw std::runtime_error("Cannot add segment: maximum of 255 segments reached.");

    size_t nS = (size_t)nSEG, nS_new = nS + 1;
    size_t nF = n_freq();
    bool sc = is_scalar();

    size_t coeff_n = history_offset(0, nF, sc); // Coefficient block, independent of nSEG
    size_t hist_new = nS_new <= INLINE_TYPES ? 0 : (nS_new - INLINE_TYPES + 3) / 4;
    size_t n_new = nS_new * 3 + coeff_n + hist_new;

    float *buf = new float[n_new];

    if (nS) // Coordinates
        std::memcpy(buf, data, nS * 3 * sizeof(float));
    buf[nS * 3] = x, buf[nS * 3 + 1] = y, buf[nS * 3 + 2] = z;

    if (coeff_n) // Coefficients
        std::memcpy(buf + nS_new * 3, data + nS * 3, coeff_n * sizeof(float));

    if (hist_new) // Interaction history, nS >= 6
    {
        uint8_t *dst = reinterpret_cast<uint8_t *>(buf + nS_new * 3 + coeff_n);
        size_t n_valid = nS - INLINE_TYPES;
        if (n_valid)
            std::memcpy(dst, reinterpret_cast<const uint8_t *>(data + nS * 3 + coeff_n), n_valid);
        dst[n_valid] = type;
        std::memset(dst + n_valid + 1, 0, hist_new * 4 - n_valid - 1);
    }

    float dl = 0.0f; // Length of the new segment
    if (nS)
    {
        const float *p = data + (nS - 1) * 3;
        dl = qd_calc_length(p[0], p[1], p[2], x, y, z);
    }

    delete[] target.data;
    target.data = buf;

    target.i_cir = i_cir;
    target.nFRQ = nFRQ;
    target.nSEG = (uint8_t)nS_new;
    target.n_ref = n_ref;
    target.n_tra = n_tra;
    target.n_sub = n_sub;
    target.n_sct = n_sct;

    std::memcpy(target.interact_type, interact_type, sizeof(interact_type));

    if (nS < INLINE_TYPES)
        target.interact_type[nS] = type;

    std::memcpy(target.xprmat, xprmat, sizeof(xprmat));
    target.len_acc = len_acc + dl;
    return target.len_acc;
}

// Assemble the interaction type sequence
std::vector<uint8_t> quadriga_lib::path::interaction_type_codes() const
{
    size_t n = (size_t)nSEG;
    std::vector<uint8_t> out(n);

    size_t n_inline = n < INLINE_TYPES ? n : INLINE_TYPES;
    for (size_t i = 0; i < n_inline; ++i)
        out[i] = interact_type[i];

    if (n > INLINE_TYPES)
    {
        if (data == nullptr)
            throw std::runtime_error("Data buffer is not allocated.");

        const float *h = data + history_offset(n, n_freq(), is_scalar());
        const uint8_t *codes = reinterpret_cast<const uint8_t *>(h);
        std::memcpy(out.data() + INLINE_TYPES, codes, n - INLINE_TYPES);
    }
    return out;
}

// Set the interaction type sequence
void quadriga_lib::path::set_interaction_type_codes(const std::vector<uint8_t> &codes)
{
    size_t n = (size_t)nSEG;
    if (codes.size() != n)
        throw std::invalid_argument("Input 'codes' must have one entry per segment.");

    size_t n_inline = n < INLINE_TYPES ? n : INLINE_TYPES;
    for (size_t i = 0; i < n_inline; ++i)
        interact_type[i] = codes[i];

    if (n > INLINE_TYPES)
    {
        if (data == nullptr)
            throw std::runtime_error("Data buffer is not allocated.");

        uint8_t *h = reinterpret_cast<uint8_t *>(data + history_offset(n, n_freq(), is_scalar()));
        std::memcpy(h, codes.data() + INLINE_TYPES, n - INLINE_TYPES);
    }
}

#undef INLINE_TYPES
