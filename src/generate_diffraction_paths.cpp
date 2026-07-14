// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

// Helper functions
namespace
{
    // Repeat sequence of values + Optional typecast
    template <typename dtypeIn, typename dtypeOut>
    inline void qd_repeat_sequence(const dtypeIn *sequence, size_t sequence_length, size_t repeat_value, size_t repeat_sequence, dtypeOut *output)
    {
        size_t pos = 0;                                  // Position in output
        for (size_t rs = 0; rs < repeat_sequence; ++rs)  // Repeat sequence of values
            for (size_t v = 0; v < sequence_length; ++v) // Iterate through all values of the sequence
            {
                dtypeOut val = (dtypeOut)sequence[v];        // Type conversion
                for (size_t rv = 0; rv < repeat_value; ++rv) // Repeat each value
                    output[pos++] = val;
            }
    }
}

// Make sure size_t and arma::uword are synonyms
static_assert(sizeof(size_t) == sizeof(arma::uword), "size_t and arma::uword have different sizes");

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# generate_diffraction_paths
Generate elliptic propagation paths and weights for diffraction gain estimation

- Generates inputs required by [[calc_diffraction_gain]]: elliptic-arc paths sampling the Fresnel ellipsoid volume between each TX-RX pair, plus per-segment weights
- Each ellipsoid has `n_path` paths, each with `n_seg` segments; `orig` and `dest` lie on the semi-major axis
- Weights are derived from the knife-edge diffraction model; initial weights normalized so `sum(prod(weights,3),2) = 1`

## Declaration:
```
void generate_diffraction_paths(
    const arma::Mat<dtype> &orig,
    const arma::Mat<dtype> &dest,
    dtype center_frequency,
    int lod,
    arma::Cube<dtype> &ray_x,
    arma::Cube<dtype> &ray_y,
    arma::Cube<dtype> &ray_z,
    arma::Cube<dtype> &weight);
```

## Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`center_frequency`** — Center frequency
- **`lod`** — Level of detail; controls `n_path` and `n_seg`:<br><br>
   | `lod` | `n_path` | `n_seg` | Note  |
   | ----- | -------- | ------- | ----- |
   | 1     | 7        | 3       | -     |
   | 2     | 19       | 3       | -     |
   | 3     | 37       | 4       | -     |
   | 4     | 61       | 5       | -     |
   | 5     | 1        | 2       | debug |
   | 6     | 2        | 2       | debug |

## Outputs:
- **`ray_x`** — x-coordinates of path waypoints (excluding endpoints); `[n_pos, n_path, n_seg-1]`
- **`ray_y`** — y-coordinates of path waypoints (excluding endpoints); `[n_pos, n_path, n_seg-1]`
- **`ray_z`** — z-coordinates of path waypoints (excluding endpoints); `[n_pos, n_path, n_seg-1]`
- **`weight`** — Per-segment weights; `[n_pos, n_path, n_seg]`

## See also:
- [[calc_diffraction_gain]] (consumes the output of this function)
MD!*/

template <typename dtype>
void quadriga_lib::generate_diffraction_paths(const arma::Mat<dtype> &orig,
                                              const arma::Mat<dtype> &dest,
                                              dtype center_frequency,
                                              int lod,
                                              arma::Cube<dtype> &ray_x,
                                              arma::Cube<dtype> &ray_y,
                                              arma::Cube<dtype> &ray_z,
                                              arma::Cube<dtype> &weight)
{
    // Check data validity
    if (orig.n_rows == 0)
        throw std::invalid_argument("Input 'orig' cannot be empty.");
    if (orig.n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns containing the x,y,z coordinates.");

    const arma::uword n_pos = orig.n_rows; // Number of links

    if (dest.n_cols != 3)
        throw std::invalid_argument("Input 'dest' must have 3 columns containing the x,y,z coordinates.");
    if (dest.n_rows != n_pos)
        throw std::invalid_argument("Number of rows in 'orig' and 'dest' dont match.");

    if (center_frequency <= (dtype)0.0)
        throw std::invalid_argument("Input 'center_frequency' must be larger that 0.");

    arma::uword n_path = 0; // Number of diffraction arcs
    arma::uword n_seg = 0;  // Number of segments per arc
    if (lod == 1)
        n_seg = 2, n_path = 7;
    else if (lod == 2)
        n_seg = 2, n_path = 19;
    else if (lod == 3)
        n_seg = 3, n_path = 37;
    else if (lod == 4)
        n_seg = 4, n_path = 61;
    else if (lod == 5)
        n_seg = 1, n_path = 1;
    else if (lod == 6)
        n_seg = 1, n_path = 2;
    else if (lod == 7)
        n_seg = 9, n_path = 1;
    else
        throw std::invalid_argument("Input 'lod' must be 1-7.");

    arma::uword n_ray = n_path * n_seg; // Total number of ellipsoid rays (n_path * n_seg)

    // Adjust output size
    if (ray_x.n_rows != n_pos || ray_x.n_cols != n_path || ray_x.n_slices != n_seg)
        ray_x.set_size(n_pos, n_path, n_seg);

    if (ray_y.n_rows != n_pos || ray_y.n_cols != n_path || ray_y.n_slices != n_seg)
        ray_y.set_size(n_pos, n_path, n_seg);

    if (ray_z.n_rows != n_pos || ray_z.n_cols != n_path || ray_z.n_slices != n_seg)
        ray_z.set_size(n_pos, n_path, n_seg);

    if (weight.n_rows != n_pos || weight.n_cols != n_path || weight.n_slices != n_seg + 1)
        weight.set_size(n_pos, n_path, n_seg + 1);

    // Normalized ellipsoid coordinates and weights
    arma::vec tx, ty, tz, tw;
    if (lod == 1)
    {
        tx = {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
              0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75};
        ty = {0.0, 0.6375, 0.1708, -0.4667, -0.6375, -0.1708, 0.4667,
              0.0, 0.6375, 0.1708, -0.4667, -0.6375, -0.1708, 0.4667};
        tz = {0.0, 0.1708, 0.6375, 0.4667, -0.1708, -0.6375, -0.4667,
              0.0, 0.1708, 0.6375, 0.4667, -0.1708, -0.6375, -0.4667};
        tw = {0.55, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075};
    }
    else if (lod == 2)
    {
        tx = {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
              0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75};
        ty = {0.0, 0.7341, 0.1967, -0.5374, -0.7341, -0.1967, 0.5374, 2.1, 1.8187, 1.05, 0.0, -1.05, -1.8187, -2.1, -1.8187, -1.05, 0.0, 1.05, 1.8187,
              0.0, 0.7341, 0.1967, -0.5374, -0.7341, -0.1967, 0.5374, 2.1, 1.8187, 1.05, 0.0, -1.05, -1.8187, -2.1, -1.8187, -1.05, 0.0, 1.05, 1.8187};
        tz = {0.0, 0.1967, 0.7341, 0.5374, -0.1967, -0.7341, -0.5374, 0.0, 1.05, 1.8187, 2.1, 1.8187, 1.05, 0.0, -1.05, -1.8187, -2.1, -1.8187, -1.05,
              0.0, 0.1967, 0.7341, 0.5374, -0.1967, -0.7341, -0.5374, 0.0, 1.05, 1.8187, 2.1, 1.8187, 1.05, 0.0, -1.05, -1.8187, -2.1, -1.8187, -1.05};
        tw = {0.630004, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333, 0.008333};
    }
    else if (lod == 3)
    {
        tx = {0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464,
              0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
              0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536, 0.8536};
        ty = {0.0, 0.316, 0.0847, -0.2314, -0.316, -0.0847, 0.2314, 0.9504, 0.8231, 0.4752, 0.0, -0.4752, -0.8231, -0.9504, -0.8231, -0.4752, 0.0, 0.4752, 0.8231, 2.0854, 1.8658, 1.4211, 0.8049, 0.0917, -0.6325, -1.2805, -1.774, -2.0536, -2.0854, -1.8658, -1.4211, -0.8049, -0.0917, 0.6325, 1.2805, 1.774, 2.0536,
              0.0, 0.4057, 0.1087, -0.297, -0.4057, -0.1087, 0.297, 1.22, 1.0566, 0.61, 0.0, -0.61, -1.0566, -1.22, -1.0566, -0.61, 0.0, 0.61, 1.0566, 2.6769, 2.3949, 1.8241, 1.0332, 0.1178, -0.8119, -1.6437, -2.2772, -2.636, -2.6769, -2.3949, -1.8241, -1.0332, -0.1178, 0.8119, 1.6437, 2.2772, 2.636,
              0.0, 0.316, 0.0847, -0.2314, -0.316, -0.0847, 0.2314, 0.9504, 0.8231, 0.4752, 0.0, -0.4752, -0.8231, -0.9504, -0.8231, -0.4752, 0.0, 0.4752, 0.8231, 2.0854, 1.8658, 1.4211, 0.8049, 0.0917, -0.6325, -1.2805, -1.774, -2.0536, -2.0854, -1.8658, -1.4211, -0.8049, -0.0917, 0.6325, 1.2805, 1.774, 2.0536};
        tz = {0.0, 0.0847, 0.316, 0.2314, -0.0847, -0.316, -0.2314, 0.0, 0.4752, 0.8231, 0.9504, 0.8231, 0.4752, 0.0, -0.4752, -0.8231, -0.9504, -0.8231, -0.4752, 0.2746, 0.9713, 1.5508, 1.9433, 2.1014, 2.0061, 1.6688, 1.1302, 0.4553, -0.2746, -0.9713, -1.5508, -1.9433, -2.1014, -2.0061, -1.6688, -1.1302, -0.4553,
              0.0, 0.1087, 0.4057, 0.297, -0.1087, -0.4057, -0.297, 0.0, 0.61, 1.0566, 1.22, 1.0566, 0.61, 0.0, -0.61, -1.0566, -1.22, -1.0566, -0.61, 0.3524, 1.2467, 1.9906, 2.4945, 2.6974, 2.575, 2.1421, 1.4507, 0.5844, -0.3524, -1.2467, -1.9906, -2.4945, -2.6974, -2.575, -2.1421, -1.4507, -0.5844,
              0.0, 0.0847, 0.316, 0.2314, -0.0847, -0.316, -0.2314, 0.0, 0.4752, 0.8231, 0.9504, 0.8231, 0.4752, 0.0, -0.4752, -0.8231, -0.9504, -0.8231, -0.4752, 0.2746, 0.9713, 1.5508, 1.9433, 2.1014, 2.0061, 1.6688, 1.1302, 0.4553, -0.2746, -0.9713, -1.5508, -1.9433, -2.1014, -2.0061, -1.6688, -1.1302, -0.4553};
        tw = {0.51001, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.010833, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333};
    }
    else if (lod == 4)
    {
        tx = {0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955, 0.0955,
              0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455, 0.3455,
              0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545, 0.6545,
              0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045, 0.9045};
        ty = {0.0, 0.2077, 0.0556, -0.152, -0.2077, -0.0556, 0.152, 0.6249, 0.5412, 0.3124, 0.0, -0.3124, -0.5412, -0.6249, -0.5412, -0.3124, 0.0, 0.3124, 0.5412, 1.279, 1.1443, 0.8715, 0.4937, 0.0563, -0.3879, -0.7853, -1.088, -1.2595, -1.279, -1.1443, -0.8715, -0.4937, -0.0563, 0.3879, 0.7853, 1.088, 1.2595, 2.5532, 2.4662, 2.2112, 1.8054, 1.2766, 0.6608, 0.0, -0.6608, -1.2766, -1.8054, -2.2112, -2.4662, -2.5532, -2.4662, -2.2112, -1.8054, -1.2766, -0.6608, 0.0, 0.6608, 1.2766, 1.8054, 2.2112, 2.4662,
              0.0, 0.3091, 0.0828, -0.2263, -0.3091, -0.0828, 0.2263, 0.93, 0.8054, 0.465, 0.0, -0.465, -0.8054, -0.93, -0.8054, -0.465, 0.0, 0.465, 0.8054, 1.9035, 1.7031, 1.2971, 0.7347, 0.0837, -0.5773, -1.1688, -1.6193, -1.8745, -1.9035, -1.7031, -1.2971, -0.7347, -0.0837, 0.5773, 1.1688, 1.6193, 1.8745, 3.8, 3.6704, 3.2908, 2.687, 1.9, 0.9835, 0.0, -0.9835, -1.9, -2.687, -3.2908, -3.6704, -3.8, -3.6704, -3.2908, -2.687, -1.9, -0.9835, 0.0, 0.9835, 1.9, 2.687, 3.2908, 3.6704,
              0.0, 0.3091, 0.0828, -0.2263, -0.3091, -0.0828, 0.2263, 0.93, 0.8054, 0.465, 0.0, -0.465, -0.8054, -0.93, -0.8054, -0.465, 0.0, 0.465, 0.8054, 1.9036, 1.7031, 1.2971, 0.7348, 0.0837, -0.5774, -1.1688, -1.6193, -1.8745, -1.9036, -1.7031, -1.2971, -0.7348, -0.0837, 0.5774, 1.1688, 1.6193, 1.8745, 3.8, 3.6705, 3.2909, 2.687, 1.9, 0.9835, 0.0, -0.9835, -1.9, -2.687, -3.2909, -3.6705, -3.8, -3.6705, -3.2909, -2.687, -1.9, -0.9835, 0.0, 0.9835, 1.9, 2.687, 3.2909, 3.6705,
              0.0, 0.2077, 0.0556, -0.152, -0.2077, -0.0556, 0.152, 0.6249, 0.5411, 0.3124, 0.0, -0.3124, -0.5411, -0.6249, -0.5411, -0.3124, 0.0, 0.3124, 0.5411, 1.279, 1.1443, 0.8715, 0.4937, 0.0563, -0.3879, -0.7853, -1.088, -1.2595, -1.279, -1.1443, -0.8715, -0.4937, -0.0563, 0.3879, 0.7853, 1.088, 1.2595, 2.5532, 2.4662, 2.2111, 1.8054, 1.2766, 0.6608, 0.0, -0.6608, -1.2766, -1.8054, -2.2111, -2.4662, -2.5532, -2.4662, -2.2111, -1.8054, -1.2766, -0.6608, 0.0, 0.6608, 1.2766, 1.8054, 2.2111, 2.4662};
        tz = {0.0, 0.0556, 0.2077, 0.152, -0.0556, -0.2077, -0.152, 0.0, 0.3124, 0.5412, 0.6249, 0.5412, 0.3124, 0.0, -0.3124, -0.5412, -0.6249, -0.5412, -0.3124, 0.1684, 0.5957, 0.9511, 1.1918, 1.2888, 1.2303, 1.0235, 0.6931, 0.2792, -0.1684, -0.5957, -0.9511, -1.1918, -1.2888, -1.2303, -1.0235, -0.6931, -0.2792, 0.0, 0.6608, 1.2766, 1.8054, 2.2112, 2.4662, 2.5532, 2.4662, 2.2112, 1.8054, 1.2766, 0.6608, 0.0, -0.6608, -1.2766, -1.8054, -2.2112, -2.4662, -2.5532, -2.4662, -2.2112, -1.8054, -1.2766, -0.6608,
              0.0, 0.0828, 0.3091, 0.2263, -0.0828, -0.3091, -0.2263, 0.0, 0.465, 0.8054, 0.93, 0.8054, 0.465, 0.0, -0.465, -0.8054, -0.93, -0.8054, -0.465, 0.2506, 0.8865, 1.4155, 1.7738, 1.9181, 1.8311, 1.5232, 1.0316, 0.4156, -0.2506, -0.8865, -1.4155, -1.7738, -1.9181, -1.8311, -1.5232, -1.0316, -0.4156, 0.0, 0.9835, 1.9, 2.687, 3.2908, 3.6704, 3.8, 3.6704, 3.2908, 2.687, 1.9, 0.9835, 0.0, -0.9835, -1.9, -2.687, -3.2908, -3.6704, -3.8, -3.6704, -3.2908, -2.687, -1.9, -0.9835,
              0.0, 0.0828, 0.3091, 0.2263, -0.0828, -0.3091, -0.2263, 0.0, 0.465, 0.8054, 0.93, 0.8054, 0.465, 0.0, -0.465, -0.8054, -0.93, -0.8054, -0.465, 0.2506, 0.8866, 1.4156, 1.7738, 1.9182, 1.8311, 1.5232, 1.0316, 0.4156, -0.2506, -0.8866, -1.4156, -1.7738, -1.9182, -1.8311, -1.5232, -1.0316, -0.4156, 0.0, 0.9835, 1.9, 2.687, 3.2909, 3.6705, 3.8, 3.6705, 3.2909, 2.687, 1.9, 0.9835, 0.0, -0.9835, -1.9, -2.687, -3.2909, -3.6705, -3.8, -3.6705, -3.2909, -2.687, -1.9, -0.9835,
              0.0, 0.0556, 0.2077, 0.152, -0.0556, -0.2077, -0.152, 0.0, 0.3124, 0.5411, 0.6249, 0.5411, 0.3124, 0.0, -0.3124, -0.5411, -0.6249, -0.5411, -0.3124, 0.1684, 0.5957, 0.9511, 1.1918, 1.2888, 1.2303, 1.0235, 0.6931, 0.2792, -0.1684, -0.5957, -0.9511, -1.1918, -1.2888, -1.2303, -1.0235, -0.6931, -0.2792, 0.0, 0.6608, 1.2766, 1.8054, 2.2111, 2.4662, 2.5532, 2.4662, 2.2111, 1.8054, 1.2766, 0.6608, 0.0, -0.6608, -1.2766, -1.8054, -2.2111, -2.4662, -2.5532, -2.4662, -2.2111, -1.8054, -1.2766, -0.6608};
        tw = {0.490012, 0.043333, 0.043333, 0.043333, 0.043333, 0.043333, 0.043333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.013333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.003333, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125};
    }
    else if (lod == 5)
    {
        tx = {0.5};
        ty = {0.0};
        tz = {0.0};
        tw = {1.0};
    }
    else if (lod == 6)
    {
        tx = {0.5, 0.5};
        ty = {0.0, 0.0};
        tz = {0.0, 0.0};
        tw = {0.5, 0.5};
    }
    else if (lod == 7)
    {
        tx = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        ty = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        tz = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        tw = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    }
    else
        throw std::invalid_argument("Input 'lod' must be 1-7.");

    double *p_tw = tw.memptr();         // Normalized weights
    double s = 1.0 / double(n_seg + 1); // Divide by number of segments
    for (arma::uword i = 0; i < n_path; ++i)
        p_tw[i] = std::pow(p_tw[i], s);

    qd_repeat_sequence(p_tw, n_path, n_pos, n_seg + 1, weight.memptr());

    // Pointers
    auto p_ox = orig.colptr(0), p_oy = orig.colptr(1), p_oz = orig.colptr(2); // Origin pointer (dtype)
    auto p_dx = dest.colptr(0), p_dy = dest.colptr(1), p_dz = dest.colptr(2); // Destination pointer (dtype)
    auto p_tx = tx.memptr(), p_ty = ty.memptr(), p_tz = tz.memptr();          // Normalized ellipsoid coordinates
    auto p_rx = ray_x.memptr(), p_ry = ray_y.memptr(), p_rz = ray_z.memptr(); // Output: scaled and rotated ellipsoid coordinates

    // Constants
    double lambda_div_8 = 0.125 * 299792458.0 / (double)center_frequency; // lambda / 8

// Iterate through positions
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n_pos; ++i)
    {
        // Calculate ellipsoid orientation and length of the semi-major axis
        double Ox = (double)p_ox[i], Oy = (double)p_oy[i], Oz = (double)p_oz[i];

        // Vector from O to D
        double ODx = (double)p_dx[i] - Ox,
               ODy = (double)p_dy[i] - Oy,
               ODz = (double)p_dz[i] - Oz;

        double d3d = std::sqrt(ODx * ODx + ODy * ODy + ODz * ODz);
        if (d3d > 2e-7) // Normalize
        {
            double scl = 1.0 / d3d;
            ODx *= scl, ODy *= scl, ODz *= scl;
        }
        else // Fallback
            ODx = 1.0, ODy = 0.0, ODz = 0.0;

        // Convert to geographic coordinates to obtain ellipsoid orientation
        ODz = (ODz > 1.0) ? 1.0 : (ODz < -1.0 ? -1.0 : ODz);
        double az = std::atan2(ODy, ODx);
        double el = std::asin(ODz);
        double sin_az = std::sin(az), cos_az = std::cos(az);
        double sin_el = ODz, cos_el = std::cos(el);

        // Width of the Fresnel ellipsoid (scales with distance)
        double width = std::sqrt(d3d * lambda_div_8);

        // Calculate ray coordinates
        for (arma::uword i_ray = 0; i_ray < n_ray; ++i_ray)
        {
            // Read normalized start coordinates of the current ray
            double Rx = p_tx[i_ray], Ry = p_ty[i_ray], Rz = p_tz[i_ray];

            // Scale length and width of the ellipsoid
            Rx *= d3d, Ry *= width, Rz *= width;

            // Rotate the ellipsoid
            double tmp = cos_el * Rx - sin_el * Rz;
            Rz = sin_el * Rx + cos_el * Rz, Rx = tmp;
            tmp = cos_az * Rx - sin_az * Ry;
            Ry = sin_az * Rx + cos_az * Ry, Rx = tmp;

            // Add origin
            Rx += Ox, Ry += Oy, Rz += Oz;

            // Convert type and write to output
            arma::uword ij = i_ray * n_pos + i;
            p_rx[ij] = (dtype)Rx;
            p_ry[ij] = (dtype)Ry;
            p_rz[ij] = (dtype)Rz;
        }
    }
}

template void quadriga_lib::generate_diffraction_paths(const arma::Mat<float> &orig, const arma::Mat<float> &dest, float center_frequency, int lod,
                                                       arma::Cube<float> &ray_x, arma::Cube<float> &ray_y, arma::Cube<float> &ray_z, arma::Cube<float> &weight);

template void quadriga_lib::generate_diffraction_paths(const arma::Mat<double> &orig, const arma::Mat<double> &dest, double center_frequency, int lod,
                                                       arma::Cube<double> &ray_x, arma::Cube<double> &ray_y, arma::Cube<double> &ray_z, arma::Cube<double> &weight);
