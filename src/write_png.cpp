// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "quadriga_tools.hpp"
#include "lodepng.h"

/*!SECTION
Miscellaneous tools
SECTION!*/

/*!MD
# write_png
Write data to a PNG file

## Description:
- Converts input data into a color-coded PNG file for visualization
- Support optional selection of a colormap, as well a minimum and maximum value limits
- Allowed datatypes (`dtype`): `float` or `double`
- Uses the <a href="https://github.com/lvandeve/lodepng">LodePNG</a> library for PNG writing

## Declaration:
```
void write_png( const arma::Mat<dtype> &data, 
                std::string fn,              
                std::string colormap = "jet", 
                dtype min_val = NAN, 
                dtype max_val = NAN, 
                bool log_transform = false);
```

## Arguments:
- `const arma::Mat<dtype> **&data**`<br>
  Data matrix

- `std::string **fn**`<br>
  Path to the `.png` file to be written

- `std::string **colormap**`<br>
  Name of the desired colormap. Must be one of:
  `"jet"`, `"parula"`, `"winter"`, `"hot"`, `"turbo"`, `"copper"`, `"spring"`, `"cool"`, `"gray"`, `"autumn"`, `"summer"`.

- `dtype **min_val**`<br>
  Minimum value. Values below this value will have be encoded with the color of the smallest value.
  If `NAN` is provided (default), the lowest values is determined from the data.

- `dtype **max_val**`<br>
  Maximum value. Values above this value will have be encoded with the color of the largest value.
  If `NAN` is provided (default), the largest values is determined from the data.

- `bool **log_transform**`<br>
  If enabled, the `data` values are transformed to the log-domain (`10*log10(data)`) before processing.
  Default: false (disabled)

## See also:
- [[colormap]]
MD!*/

template <typename dtype>
void quadriga_lib::write_png(const arma::Mat<dtype> &data, std::string fn, std::string colormap,
                             dtype min_val, dtype max_val, bool log_transform)
{
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
            x = (log_transform && x < 0.0) ? 0.0 : x;
            x = log_transform ? 10.0 * std::log10(x) : x;

            mi = (set_min && x < mi) ? x : mi;
            ma = (set_max && x > ma) ? x : ma;
        }

    mi = (mi > ma) ? ma : mi;

    // Convert the data to the map
    const dtype *values = data.memptr();
    double scl = double(n_cmap - 1) / (ma - mi);

    for (size_t i_val = 0; i_val < no_values; ++i_val)
    {
        double x = (double)values[i_val];
        x = (log_transform && x < 0.0) ? 0.0 : x;
        x = log_transform ? 10.0 * std::log10(x) : x;

        x = (x < mi) ? mi : x;
        x = (x > ma) ? ma : x;
        x = x - mi;
        x = x * scl;
        x = std::round(x);
        size_t i_cmap = (size_t)x;

        // Fix out-of bound for NaN values
        i_cmap = (i_cmap >= n_cmap) ? n_cmap - 1 : i_cmap;

        size_t i_png_row = i_val % height; // height - (i_val % height) - 1;
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
        throw std::runtime_error("PNG encoder error: '" + std::string(lodepng_error_text(error)));
}

template void quadriga_lib::write_png(const arma::Mat<float> &data, std::string fn, std::string colormap,
                                      float min_val, float max_val, bool log_transform);

template void quadriga_lib::write_png(const arma::Mat<double> &data, std::string fn, std::string colormap,
                                      double min_val, double max_val, bool log_transform);
