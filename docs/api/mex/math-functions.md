# Math functions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/mex_api.html#sec-math-functions

| Function | Summary |
| --- | --- |
| [acdf](#acdf) | Empirical cumulative distribution functions (CDF) of several data sets, their average and quantile statistics |
| [calc_rotation_matrix](#calc_rotation_matrix) | Calculate rotation matrices from Euler angles |
| [fast_cart2geo](#fast_cart2geo) | Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length |
| [fast_geo2cart](#fast_geo2cart) | Convert elementwise azimuth/elevation angles to Cartesian coordinates |
| [fast_sincos](#fast_sincos) | Compute elementwise approximate sine and/or cosine of a vector |
| [interp](#interp) | Perform linear interpolation (1D or 2D) on single or multiple data sets |

---
<a id="acdf"></a>
## acdf
Empirical cumulative distribution functions (CDF) of several data sets, their average and quantile statistics

Also in: [C++](../cpp/math-functions.md#acdf) · [Python](../python/math-functions.md#acdf)

- The CDF of each data set (column of `data`) at the bin centers: each finite value counts for the nearest bin (values
  outside the bins for the first or last one), and the CDF is the cumulative count divided by the number of finite
  values of the set
- Average `cdf_avg` in quantile space: for the probabilities $P = 0, 0.001, \dots, 0.999$, the bins at which the
  CDFs of the sets reach $P$ are averaged over the sets, and the averaged quantile function is mapped back to the
  bins; this averages the position of the distributions instead of their CDF values
- `mu` and `sig`: mean and standard deviation (normalized by $N - 1$) of the quantiles at
  $P = 0.1, 0.2, \dots, 0.9$ over the sets, each taken over the 11 probabilities within $\pm 0.005$; for one set its
  quantiles and 0
- Non-finite values are left out; a set without finite values gets a NaN CDF and is left out of `cdf_avg`, `mu` and
  `sig`, which are NaN if no set has finite values (possible only with given bins)
- Generated bins: `n_bins` equally spaced bins from the smallest to the largest finite value of `data` (±10 % of the
  value, or ±1 for 0, if all are equal); `data` without finite values then throws

### Usage:
```matlab
[ cdf_per_set, bins_out, cdf_avg, mu, sig ] = quadriga_lib.acdf( data, bins_in, n_bins );
```

### Inputs:
- **`data`** — Data sets, one per column, magnitudes below about 1e18; `[n_samples, n_sets]`
- **`bins_in`** — Bin centers, finite and strictly ascending; `[n_bins_in, 1]`; default: `[]` (generated bins)
- **`n_bins`** — Number of generated bins, >= 2; not used with `bins_in`; default: 201

### Outputs:
- **`cdf_per_set`** — CDF of each set; `[n_bins_out, n_sets]`
- **`bins_out`** — Bin centers, `bins_in` or the generated ones; `[n_bins_out, 1]`
- **`cdf_avg`** — Average CDF; `[n_bins_out, 1]`
- **`mu`** — Mean of the quantiles at 0.1, ..., 0.9; `[9, 1]`
- **`sig`** — Standard deviation of these quantiles; `[9, 1]`

### See also:
- [calc_delay_spread](channel-data-and-processing.md#calc_delay_spread) (delay spreads of many CIRs, a typical input)

---
<a id="calc_rotation_matrix"></a>
## calc_rotation_matrix
Calculate rotation matrices from Euler angles

Also in: [C++](../cpp/math-functions.md#calc_rotation_matrix) · [Python](../python/math-functions.md#calc_rotation_matrix)

- Computes 3×3 rotation matrices from Euler angles (bank, tilt, heading) in column-major order (9 elements
  per orientation)
- Single-precision input is cast to double; output is always double

### Usage:
```matlab
rotation = quadriga_lib.calc_rotation_matrix( orientation, transposed );
```

### Inputs:
- **`orientation`** — Euler angles (bank, tilt, heading); `[3, n_row, n_col]` or `[3, n_mat]` or `[3]`
- **`transposed`** — Returns the transpose of the rotation matrix; logical; default: false

### Output:
- **`rotation`** — Rotation matrices in column-major order; `[9, n_row, n_col]` or `[9, n_mat]` or `[9, 1]`

---
<a id="fast_cart2geo"></a>
## fast_cart2geo
Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length

Also in: [C++](../cpp/math-functions.md#fast_cart2geo) · [Python](../python/math-functions.md#fast_cart2geo)

- Computes: `len = sqrt(x² + y² + z²)`, `az = atan2(y, x)`, `el = asin(clamp(z / len, -1, 1))`
- Inputs are arbitrary 3D vectors (not required to be unit-length); `len` returns the Euclidean norm
- `z/len` is clamped to [-1, 1] before `asin` to guard against `len == 0` and rounding artifacts
  pushing `abs(z/len)` slightly above 1
- Azimuth in [-pi, pi], with the special values of `atan2` (e.g. `atan2(0, -0) = pi`); NaN input gives NaN
  in `az`, `el` and `len`
- The AVX2 kernel computes the angles in single precision; `len` and `z/len` in double precision
- `x`, `y` and `z` have the same size; the outputs have that size; empty input gives empty outputs

### Usage:
```matlab
[ az, el, len ] = quadriga_lib.fast_cart2geo( x, y, z, use_kernel );
```

### Inputs:
- **`x`** — X-coordinates; up to 4 dimensions
- **`y`** — Y-coordinates; size of `x`
- **`z`** — Z-coordinates; size of `x`
- **`use_kernel`** — Kernel selection: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 
  (throws if AVX2 unavailable); other values throw; default: 0

### Outputs:
- **`az`** — Azimuth angles in radians; size of `x`
- **`el`** — Elevation angles in radians; size of `x`
- **`len`** — Euclidean vector length `sqrt(x² + y² + z²)`; size of `x`

---
<a id="fast_geo2cart"></a>
## fast_geo2cart
Convert elementwise azimuth/elevation angles to Cartesian coordinates

Also in: [C++](../cpp/math-functions.md#fast_geo2cart) · [Python](../python/math-functions.md#fast_geo2cart)

- Conversion: `x = cos(el) cos(az) len`, `y = cos(el) sin(az) len`, `z = sin(el) len`
- Optional outputs `sin_az`, `cos_az`, `sin_el`, `cos_el` return intermediate sin/cos values; omit from the
  output list to skip their computation
- The AVX2 kernel (selected by `use_kernel=0` where available, forced by `use_kernel=2`) computes sine and
  cosine in single precision (angles reduced in double precision first; max. error about 1.3e-7 for angles up
  to 1e10), results widened back; `use_kernel=1` keeps full double precision
- `el` and `len` have the same size as `az`; the outputs have that size; empty input gives empty outputs

### Usage:
```matlab
[ x, y, z, sin_az, cos_az, sin_el, cos_el ] = quadriga_lib.fast_geo2cart( az, el, len, use_kernel );
```

### Inputs:
- **`az`** — Azimuth angles in radians; up to 4 dimensions
- **`el`** — Elevation angles in radians; size of `az`
- **`len`** — Euclidean vector length sqrt(x^2 + y^2 + z^2); size of `az`; default: 1
- **`use_kernel`** — Kernel selection: 0 = auto (AVX2 if available, else GENERIC),
  1 = GENERIC, 2 = AVX2 (throws if AVX2 unavailable); other values throw; default: 0

### Outputs:
- **`x`** — X-coordinates; size of `az`
- **`y`** — Y-coordinates; size of `az`
- **`z`** — Z-coordinates; size of `az`
- **`sin_az`** — sin(az); size of `az`
- **`cos_az`** — cos(az); size of `az`
- **`sin_el`** — sin(el); size of `az`
- **`cos_el`** — cos(el); size of `az`

---
<a id="fast_sincos"></a>
## fast_sincos
Compute elementwise approximate sine and/or cosine of a vector

Also in: [C++](../cpp/math-functions.md#fast_sincos) · [Python](../python/math-functions.md#fast_sincos)

- Max absolute error about 1.3e-7 (2^-22.9) against the exact sine / cosine of the input value for
  abs(x) <= 1e10 (the argument is reduced in double precision where single precision is not enough)
- Results stay in [-1, 1] for all finite x; Inf and NaN give NaN
- Works on vectors, matrices, and 3-D arrays
- Accepts any numeric input class; best performance with single precision
- Outputs are always single precision, also for double input
- Request one or two outputs to control which results are returned
- With one output, set the optional `cos_only` flag to `true` to return cosine instead of sine

### Usage:
```matlab
[sin_x, cos_x] = quadriga_lib.fast_sincos(x);
sin_x = quadriga_lib.fast_sincos(x);
cos_x = quadriga_lib.fast_sincos(x, true);
[sin_x, cos_x] = quadriga_lib.fast_sincos(x, [], use_kernel);
```

### Inputs:
- **`x`** — Input angles; radians; vector, matrix or 3-D array
- **`cos_only`** — For a single output: `true` returns `cos(x)`; `false` or `[]` returns `sin(x)`; default: false
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (error if unavailable); default: 0

### Outputs:
- **`sin_x`** — sin(x); single; size of `x`
- **`cos_x`** — cos(x); single; size of `x`

---
<a id="interp"></a>
## interp
Perform linear interpolation (1D or 2D) on single or multiple data sets

Also in: [C++](../cpp/math-functions.md#interp_2D) · [Python](../python/math-functions.md#interp)

- Interpolates given input data at specified output points.
- Outside the range of the data sampling points, the nearest end value is returned (constant extrapolation);
  output sampling points that are NaN give NaN.
- The data sampling points need not be sorted; repeated values are allowed; they must be finite.
- Supports single and multiple data sets.
- Data types: `single` or `double`

### Usage:

```matlab
output = quadriga_lib.interp( x, y, input, x_new, y_new );    % 2D case

output = quadriga_lib.interp( x, [], input, x_new );           % 1D case
```

### Inputs:
- **`x`** — Data x-axis sampling points; Length: `[nx]`
- **`y`** — Data y-axis sampling points; Length: `[ny]`
- **`input`** — Input data array/matrix; `[ny, nx, ne]` or `[1, nx, ne]` for 1D case; 3rd dimension
  enables interpolation for multiple datasets simultaneously.
- **`x_new`** — Output x-axis sampling points; Length: `[nx_new]`
- **`y_new`** — Output y-axis sampling points; Length: `[ny_new]`; required when `y` is not empty, empty or omitted
  when `y` is empty

### Output:
- **`output`**  —  Interpolated data `[ny_new, nx_new, ne]` or `[1, nx_new, ne]` for 1D case
