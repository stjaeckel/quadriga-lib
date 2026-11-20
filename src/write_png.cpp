// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include "lodepng.h"
#include <limits>

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# write_png
Write a data matrix to a color-coded PNG file

- Values are clipped to `[min_val, max_val]` before colormap mapping; auto-detected from data if `NAN`
- The auto-detected range ignores `NaN` and `+-Inf`; `+Inf` and `-Inf` values get the last and first colour
- `NaN` values get the first colour of the map
- A degenerate range (`max <= min` after auto-detection, e.g. constant or single-value data, all-`NaN` data, or log
  input without a positive value) maps every value to the first colour
- Throws `std::invalid_argument` for an empty `data` matrix, an infinite `min_val` or `max_val`, and for
  `min_val > max_val` (both given)
- Uses [LodePNG](https://github.com/lvandeve/lodepng) for PNG encoding

## Declaration:
```
void quadriga_lib::write_png(
    const std::string &fn,
    const arma::Mat<dtype> &data,
    const std::string &colormap = "jet",
    double min_val = NAN,
    double max_val = NAN,
    bool log_transform = false);
```

## Inputs:
- **`fn`** — Output `.png` file path
- **`data`** — Input data matrix
- **`colormap`** *(optional)* — Colormap name; see [[colormap]] for valid values
- **`min_val`** *(optional)* — Lower clipping bound; NaN (the default): data minimum
- **`max_val`** *(optional)* — Upper clipping bound; NaN (the default): data maximum
- **`log_transform`** *(optional)* — Apply 10*log10(data) before mapping; non-positive values map to the minimum color
MD!*/

template <typename dtype>
void quadriga_lib::write_png(const std::string &fn, const arma::Mat<dtype> &data, const std::string &colormap,
                             double min_val_in, double max_val_in, bool log_transform)
{
    const dtype min_val = dtype(min_val_in);
    const dtype max_val = dtype(max_val_in);
    if (data.n_elem == 0)
        throw std::invalid_argument("Input 'data' must not be empty.");

    if (std::isinf(min_val) || std::isinf(max_val))
        throw std::invalid_argument("Inputs 'min_val' and 'max_val' must be finite or NaN.");
    if (std::isfinite(min_val) && std::isfinite(max_val) && min_val > max_val)
        throw std::invalid_argument("Input 'min_val' must not be larger than 'max_val'.");

    // Get colormap
    auto cmap = quadriga_lib::colormap(colormap, true);
    size_t n_cmap = (size_t)cmap.n_rows;

    // Initialize output
    size_t height = (size_t)data.n_rows;
    size_t width = (size_t)data.n_cols;
    size_t no_values = height * width;

    std::vector<unsigned char> image(height * width * 4);

    // Determine minimum and maximum value
    bool set_min = std::isnan(min_val);
    bool set_max = std::isnan(max_val);

    double mi = set_min ? INFINITY : (double)min_val;
    double ma = set_max ? -INFINITY : (double)max_val;

    if (set_min || set_max)
        for (auto &val : data)
        {
            double x = (double)val;
            if (log_transform)
            {
                if (x <= 0.0)
                    continue; // non-positive → skip for range detection
                x = 10.0 * std::log10(x);
            }
            if (std::isfinite(x)) // NaN and +-Inf do not enter the range
            {
                mi = (set_min && x < mi) ? x : mi;
                ma = (set_max && x > ma) ? x : ma;
            }
        }

    // A degenerate range (no finite span) maps every value to the first colour
    const bool degenerate = !(ma > mi) || !std::isfinite(ma - mi);

    // Convert the data to the map
    const dtype *values = data.memptr();
    double scl = degenerate ? 0.0 : double(n_cmap - 1) / (ma - mi);
    const bool tiny_range = !degenerate && !std::isfinite(scl); // Range below about 1e-306: divide by it instead

    for (size_t i_val = 0; i_val < no_values; ++i_val)
    {
        double x = (double)values[i_val];
        if (log_transform)
            x = (x <= 0.0) ? -std::numeric_limits<double>::infinity() : 10.0 * std::log10(x);

        size_t i_cmap = 0; // first colour for NaN and for a degenerate range
        if (!degenerate && !std::isnan(x))
        {
            x = (x < mi) ? mi : x; // clips -Inf
            x = (x > ma) ? ma : x; // clips +Inf
            x = x - mi;
            x = tiny_range ? (x / (ma - mi)) * double(n_cmap - 1) : x * scl;
            x = std::round(x);
            i_cmap = (size_t)x;
            i_cmap = (i_cmap >= n_cmap) ? n_cmap - 1 : i_cmap;
        }

        size_t i_png_row = i_val % height;
        size_t i_png_col = i_val / height;
        size_t i_png = 4 * (i_png_row * width + i_png_col);

        image[i_png + 0] = cmap.at(i_cmap, 0);
        image[i_png + 1] = cmap.at(i_cmap, 1);
        image[i_png + 2] = cmap.at(i_cmap, 2);
        image[i_png + 3] = 255;
    }

    // Write PNG
    unsigned error = lodepng::encode(fn, image, (unsigned)width, (unsigned)height);
    if (error)
        throw std::runtime_error("Cannot write PNG file '" + fn + "': " + std::string(lodepng_error_text(error)) + ".");
}

template void quadriga_lib::write_png(const std::string &fn, const arma::Mat<float> &data, const std::string &colormap,
                                      double min_val, double max_val, bool log_transform);

template void quadriga_lib::write_png(const std::string &fn, const arma::Mat<double> &data, const std::string &colormap,
                                      double min_val, double max_val, bool log_transform);
