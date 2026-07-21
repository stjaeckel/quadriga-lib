// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"

#include <cstring>   // std::memcpy
#include <stdexcept> // std::invalid_argument
#include <cstddef>   // size_t
#include <cstdint>   // uint8_t
#include <utility>   // std::as_const
#include <cmath>
#include <limits>
#include <vector>

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# path
Class for storing and managing a single propagation path with a compact fixed-size header

- Represents one ray from origin to destination through a sequence of `nSEG` interaction points
- A 64-byte header holds metadata, the first 6 interaction type codes, and the frequency-0
  transfer coefficients; a variable-length heap buffer holds coordinates, the remaining
  frequency coefficients, and the overflow interaction codes
- Two layout modes selected at initialization: EM carries a full 2x2 Jones matrix per frequency;
  SCALAR carries a single complex pressure coefficient per frequency
- Copyable and movable; the moved-from object is left in the valid empty state (`nSEG == 0`, `nFRQ == 1`, EM)

## Attributes:<br>
| Attribute            | Size   | Description                                                       |
| -------------------- | ------ | ----------------------------------------------------------------- |
| `unsigned iC`        | scalar | Channel ID: the channel to which the path belongs                 |
| `unsigned iR`        | scalar | Ray index: relative index in the launch configuration             |
| `uint8_t nREF`       | scalar | Number of reflections (interaction type codes 128-255)            |
| `uint8_t nTRA`       | scalar | Number of transmissions / refractions (type codes 1-127)          |
| `uint8_t nSUB`       | scalar | Number of subdivisions                                            |
| `uint8_t nSCT`       | scalar | Number of scattering events                                       |
| `float length`       | scalar | Accumulated path length, origin through the last interaction (m)  |

## Data buffer layout (EM mode):<br>
| Block         | Size (floats)     | Description                                             |
| ------------- | ----------------- | ------------------------------------------------------- |
| Coordinates   | `3 * nSEG`        | Interaction points `[x, y, z]` per segment              |
| Jones         | `8 * (nFRQ - 1)`  | Transfer matrices for frequencies 1..nFRQ-1, col-major  |
| Interactions  | `(nSEG - 3) / 4`  | Overflow type codes past the first 6, packed 4 per float, present for `nSEG >= 7` |

## Data buffer layout (SCALAR mode):<br>
| Block         | Size (floats)     | Description                                             |
| ------------- | ----------------- | ------------------------------------------------------- |
| Coordinates   | `3 * nSEG`        | Interaction points `[x, y, z]` per segment              |
| Coeff         | `2 * (nFRQ - 4)`  | Pressure coefficients for frequencies 4..nFRQ-1         |
| Interactions  | `(nSEG - 3) / 4`  | Overflow type codes past the first 6, packed 4 per float, present for `nSEG >= 7` |

## Simple member functions:<br>
| Method            | Description                                                                     |
| ----------------- | ------------------------------------------------------------------------------- |
| `.n_freq()`       | Returns the number of frequencies (1-127)                                       |
| `.n_seg()`        | Returns the number of segments                                                  |
| `.is_scalar()`    | Returns true for SCALAR layout, false for EM                                    |
| `.init()`         | (Re)initializes storage to a given segment / frequency layout                   |
| `.free()`         | Releases the data buffer and resets to the valid empty state                    |
| `.coord()`        | Returns a pointer to a segment's `[x, y, z]` coordinates                        |
| `.xpr_coeff()`    | Returns a pointer to a frequency's transfer coefficients                        |
| `.operator()`     | Returns a last-segment coordinate by index (0=x, 1=y, 2=z)                      |
| `.interaction_type_codes()` | Returns the full interaction type sequence as a vector                |
| `.set_interaction_type_codes()` | Sets the interaction type sequence; requires one code per segment |

## Function 'calc_length':
Calculates the total length of the path

- The stored member `length` spans origin through the last interaction; it never includes the leg to the destination
- The read-only overload adds only the final last-to-D leg to the stored member; the origin overloads recompute every leg and overwrite the member with the origin-to-last total
- Returns the full length including the leg to `D`; on an empty path (`nSEG == 0`) the read-only form returns NaN while the origin form returns the straight origin-to-destination distance
```
float calc_length(float Dx, float Dy, float Dz) const;                         // member length + last to D distance
float calc_length(float Dx, float Dy, float Dz, float Ox, float Oy, float Oz); // calculate full path length, update member
float calc_length(const float *D, const float *O = nullptr);                   // 3-element overload
```
- **`Dx`, `Dy`, `Dz`, `*D`** — Destination coordinates
- **`Ox`, `Oy`, `Oz`, `*O`** — Origin coordinates (optional); if provided, accumulated path length is updated

## Function 'calc_gain':
Calculates the path gain (linear power) from the transfer coefficients at one frequency

- EM mode returns the maximum column power of the 2x2 Jones matrix; SCALAR mode returns the squared magnitude of the pressure coefficient
- A degenerate coefficient set (all-zero, underflowed, or containing NaN) returns 0, flagging the path for pruning
- With `fGHz > 0`, free-space path loss is folded in: `(lambda / (4 pi d))^2` in EM mode, `1 / d^2` (frequency-independent) in SCALAR mode
- The distance `d` is the stored `length` by default; pass `len > 0` to override it, e.g. to match a total length from the
  read-only `calc_length` before the member is updated
```
float calc_gain(float fGHz = 0.0f, size_t freq = 0, float len = 0.0f) const;
```
- **`fGHz`** — Frequency in GHz; `> 0` applies path loss, `0` returns polarization power only. In SCALAR mode any
  positive value applies spherical spreading; the magnitude is ignored
- **`freq`** — Frequency index into the coefficient store, valid range `0` to `nFRQ - 1`
- **`len`** — Path length override in meters; `> 0` replaces the stored `length` for the path-loss term, `<= 0` uses the stored value

## Function 'xpr_update':
Left-multiplies the ray's transfer matrix by an interaction matrix, applies a power gain, and returns the resulting gain

- Updates the coefficient slot for one frequency in place: `M := coeff_update * M`, then scales by `sqrt(gain_update)` in amplitude
- `coeff_update == nullptr` applies only the gain; `gain_update == 1.0` applies only the multiply; both defaults leave the matrix unchanged and just measure it
- The update reads all old values before writing, so `coeff_update` may alias the slot; the return value uses the same gain definition as `calc_gain`
- Must be called after `extend` closes the path, so the stored `length` is final when path loss is requested
```
float xpr_update(const float *coeff_update = nullptr, float gain_update = 1.0f, size_t freq = 0, float fGHz = 0.0f, float len = 0.0f);
```
- **`coeff_update`** — Interaction matrix, length 8 (EM) or 2 (SCALAR); `nullptr` skips the multiply
- **`gain_update`** — Power gain applied to the slot (converted to an amplitude factor internally); `1.0` skips the scaling
- **`freq`** — Frequency index into the coefficient store, valid range `0` to `nFRQ - 1`
- **`fGHz`** — Frequency in GHz for the returned gain; `> 0` applies path loss as in `calc_gain`, `0` returns polarization power only
- **`len`** — Path length override in meters; `> 0` replaces the stored `length` for the path-loss term, `<= 0` uses the stored value

## Function 'duplicate';
Copies the path into an existing target object and returns its length

- Performs a deep copy: the target receives an independent data buffer and all metadata
- The target's previous contents are released; the source is left unchanged
```
float duplicate(path &target) const;
```
- **`target`** — Destination path, overwritten with a deep copy of the source

## Function 'extend':
Copies the path into a target and appends one new segment, returning the new total length

- The new coordinate is appended after the existing segments; the stored `length` grows by the origin-to-new-point distance of the appended leg
- The interaction `type` is recorded for the new segment and classifies the interaction: codes 1-127 increment `nTRA`, codes 128-255 increment `nREF`, code 0 increments neither
- The target receives an independent buffer sized for the extra segment; the source is left unchanged
- Throws if the source already holds the maximum of 255 segments
```
float extend(path &target, float x, float y, float z, uint8_t type = 0) const;
```
- **`target`** — Destination path, overwritten with the extended copy
- **`x`, `y`, `z`** — Coordinates of the appended interaction point
- **`type`** — Interaction type code for the new segment; drives the reflection / transmission counters
MD!*/

#define INLINE_TYPES 6 // Length of the inline type codes

// Helper: Compute the offset of the history block in the data buffer
constexpr static size_t history_offset(size_t segments, size_t frequencies, bool scalar)
{
    return scalar ? segments * 3 + (frequencies < 5 ? 0 : (frequencies - 4) * 2)
                  : segments * 3 + (frequencies < 2 ? 0 : (frequencies - 1) * 8);
}

// Helper: Compute the required buffer size to store a given number is segments and frequencies
constexpr static size_t data_size(size_t segments, size_t frequencies, bool scalar)
{
    size_t history_length = segments <= INLINE_TYPES ? 0 : (segments - INLINE_TYPES + 3) / 4;
    return history_offset(segments, frequencies, scalar) + history_length;
}

// Helper: Calculate gain (linear power) from Jones matrix values; fGHz > 0 applies path loss
// Scalar mode passes the pressure coefficient as [ReVV, ImVV] and zeros for the rest
static inline float gain_from_coeff(float ReVV, float ImVV, float ReHV, float ImHV,
                                    float ReVH, float ImVH, float ReHH, float ImHH,
                                    bool scalar, float length, float fGHz)
{
    float A = ReVV * ReVV + ImVV * ImVV + ReHV * ReHV + ImHV * ImHV; // |col 0|^2
    float B = ReVH * ReVH + ImVH * ImVH + ReHH * ReHH + ImHH * ImHH; // |col 1|^2
    float P = A > B ? A : B;

    // Degenerate: NaN in either column, or both columns underflowed. A + B is NaN if either
    // part is NaN, and < min only when the true max is also below min.
    if (!(A + B >= std::numeric_limits<float>::min()))
        return 0.0f;

    // Path loss (linear power): EM = (lambda / (4 pi d))^2, lambda = c / f; scalar = 1 / d^2
    if (fGHz > 0.0f && length > 0.0f)
    {
        if (scalar)
            P /= length * length;
        else
        {
            float k = 0.299792458f / (12.566370614f * fGHz * length);
            P *= k * k;
        }
    }
    return P;
}

// Move constructor
quadriga_lib::path::path(path &&other) noexcept
    : nFRQ(other.nFRQ), nSEG(other.nSEG), data(other.data),
      iC(other.iC), iR(other.iR), nREF(other.nREF), nTRA(other.nTRA), nSUB(other.nSUB), nSCT(other.nSCT),
      length(other.length)
{
    std::memcpy(interact_type, other.interact_type, sizeof(interact_type));
    std::memcpy(xprmat, other.xprmat, sizeof(xprmat));
    other.data = nullptr;
    other.nFRQ = 1;
    other.nSEG = 0;
}

// Copy constructor
quadriga_lib::path::path(const path &other)
    : nFRQ(other.nFRQ), nSEG(other.nSEG),
      iC(other.iC), iR(other.iR), nREF(other.nREF), nTRA(other.nTRA), nSUB(other.nSUB), nSCT(other.nSCT),
      length(other.length)
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

// Move assignment
quadriga_lib::path &quadriga_lib::path::operator=(path &&other) noexcept
{
    if (this == &other)
        return *this;
    delete[] data;

    nFRQ = other.nFRQ, nSEG = other.nSEG;
    std::memcpy(interact_type, other.interact_type, sizeof(interact_type));
    std::memcpy(xprmat, other.xprmat, sizeof(xprmat));

    data = other.data;

    iC = other.iC, iR = other.iR;
    nREF = other.nREF, nTRA = other.nTRA, nSUB = other.nSUB, nSCT = other.nSCT;
    length = other.length;

    other.data = nullptr;
    other.nFRQ = 1;
    other.nSEG = 0;

    return *this;
}

// Copy assignment
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

    iC = other.iC, iR = other.iR;
    nREF = other.nREF, nTRA = other.nTRA, nSUB = other.nSUB, nSCT = other.nSCT;
    length = other.length;

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
        throw std::invalid_argument("Number of segments cannot exceed 255.");
    if (frequencies == 0 || frequencies > 127)
        throw std::invalid_argument("Number of frequencies must be between 1 and 127.");

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

    iC = 0;
    iR = 0;

    nREF = 0;
    nTRA = 0;
    nSUB = 0;
    nSCT = 0;

    length = 0.0f;
}

// Constructor
quadriga_lib::path::path(size_t segments, size_t frequencies, bool scalar)
{
    init(segments, frequencies, scalar);
}

// Access coordinates of a specific segment
const float *quadriga_lib::path::coord(size_t seg) const
{
    if (seg >= (size_t)nSEG)
        throw std::invalid_argument("Requested segment out-of-bound.");
    return data + seg * 3;
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
    float dx = Dx - last[0], dy = Dy - last[1], dz = Dz - last[2];
    return length + std::sqrt(dx * dx + dy * dy + dz * dz);
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
        float dx = c[0] - px, dy = c[1] - py, dz = c[2] - pz;
        total += std::sqrt(dx * dx + dy * dy + dz * dz);
        px = c[0], py = c[1], pz = c[2];
    }
    length = total; // O -> last, excludes final leg to D

    float dx = Dx - px, dy = Dy - py, dz = Dz - pz;
    return total + std::sqrt(dx * dx + dy * dy + dz * dz);
}

// 3-element overload; O = nullptr -> read-only member + last-to-D, else full recalc (updates member)
float quadriga_lib::path::calc_length(const float *D, const float *O)
{
    if (O)
        return calc_length(D[0], D[1], D[2], O[0], O[1], O[2]);
    return calc_length(D[0], D[1], D[2]);
}

// Access xprmat for a specific frequency
const float *quadriga_lib::path::xpr_coeff(size_t freq) const
{
    if (freq >= n_freq())
        throw std::invalid_argument("Requested frequency out-of-bound.");

    if (freq == 0)
        return xprmat;

    size_t offset = size_t(nSEG) * 3;
    if (is_scalar())
    {
        if (freq < 4)
            return &xprmat[2 * freq];
        else
            return data + offset + (freq - 4) * 2;
    }
    else // EM mode
        return data + offset + (freq - 1) * 8;
}
float *quadriga_lib::path::xpr_coeff(size_t freq)
{
    return const_cast<float *>(std::as_const(*this).xpr_coeff(freq));
}

// Left-multiply ray's Jones matrix; coeff_update must have length 8 (EM) or 2 (scalar)
// + calculate gain from xprmat coefficients; fGHz > 0 applies FSPL
float quadriga_lib::path::xpr_update(const float *coeff_update, float gain_update, size_t freq, float fGHz, float len)
{
    float *oX = xpr_coeff(freq); // Bounds-checked; resolves to xprmat or into the data buffer
    float amplitude_update = gain_update == 1.0f ? 1.0f : std::sqrt(std::abs(gain_update));
    float d = len > 0.0f ? len : length;

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
        return gain_from_coeff(oRe, oIm, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, true, d, fGHz);
    }

    // EM mode, 8 values: [ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH], column-major
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

    return gain_from_coeff(oReVV, oImVV, oReHV, oImHV, oReVH, oImVH, oReHH, oImHH, false, d, fGHz);
}

// Calculate gain from xprmat coefficients; fGHz > 0 applies FSPL; len > 0 overrides stored length
float quadriga_lib::path::calc_gain(float fGHz, size_t freq, float len) const
{
    const float *cf = xpr_coeff(freq); // Bounds-checked; resolves to xprmat or into the data buffer
    float d = len > 0.0f ? len : length;

    if (is_scalar()) // Only 2 values in the slot; reading cf[2..7] would overrun
        return gain_from_coeff(cf[0], cf[1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, true, d, fGHz);

    return gain_from_coeff(cf[0], cf[1], cf[2], cf[3], cf[4], cf[5], cf[6], cf[7], false, d, fGHz);
}

// Duplicate path, writing to "target", returns length
float quadriga_lib::path::duplicate(path &target) const
{
    target = *this;
    return length;
}

// Duplicate path and append a new segment
float quadriga_lib::path::extend(path &target, float x, float y, float z, uint8_t type) const
{
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
        float dx = x - p[0], dy = y - p[1], dz = z - p[2];
        dl = std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    delete[] target.data;
    target.data = buf;

    target.iC = iC, target.iR = iR;
    target.nFRQ = nFRQ;
    target.nSEG = (uint8_t)nS_new;
    target.nREF = type > 127 ? nREF + 1 : nREF;
    target.nTRA = type && type < 128 ? nTRA + 1 : nTRA;
    target.nSUB = nSUB;
    target.nSCT = nSCT;

    std::memcpy(target.interact_type, interact_type, sizeof(interact_type));

    if (nS < INLINE_TYPES)
        target.interact_type[nS] = type;

    std::memcpy(target.xprmat, xprmat, sizeof(xprmat));
    target.length = length + dl;
    return target.length;
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
        throw std::invalid_argument("Number of type codes must equal the number of segments.");

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
