# Math functions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-math-functions

| Function | Summary |
| --- | --- |
| [calc_rotation_matrix](#calc_rotation_matrix) | Calculate rotation matrices from Euler angles |
| [fast_acos](#fast_acos) | Compute elementwise approximate arc-cosine of a vector |
| [fast_asin](#fast_asin) | Compute elementwise approximate arc-sine of a vector |
| [fast_atan2](#fast_atan2) | Compute elementwise approximate two-argument arc-tangent of two vectors |
| [fast_cart2geo](#fast_cart2geo) | Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length |
| [fast_geo2cart](#fast_geo2cart) | Convert elementwise azimuth/elevation angles to Cartesian coordinates |
| [fast_sincos](#fast_sincos) | Compute elementwise approximate sine and/or cosine of a vector |
| [fast_slerp](#fast_slerp) | Compute elementwise approximate SLERP interpolation between two complex-valued vectors |
| [interp_2D](#interp_2D) | Perform linear interpolation (1D or 2D) on multiple data sets |

---
<a id="calc_rotation_matrix"></a>
## calc_rotation_matrix
Calculate rotation matrices from Euler angles

Also in: [MATLAB](../mex/math-functions.md#calc_rotation_matrix) · [Python](../python/math-functions.md#calc_rotation_matrix)

- Computes 3×3 rotation matrices from Euler angles (bank, tilt, head) in column-major order (9 elements per orientation)
- R = Rz(heading)·Ry(tilt)·Rx(bank), right-handed as in 3GPP TR 38.901; a positive tilt turns the x-axis down
- Internally uses double precision regardless of `dtype`
- A matrix of orientations `[3, n]` is the cube `[3, n, 1]`; wrap it without a copy:
  `arma::Cube<dtype>(const_cast<dtype *>(mat.memptr()), 3, n, 1, false, true)`

### Declaration:
```cpp
void quadriga_lib::calc_rotation_matrix(
    const arma::Cube<dtype> &orientation,
    arma::Cube<dtype> &rotation,
    bool transposed = false);
```

### Inputs:
- **`orientation`** — Euler angles (bank, tilt, head) in [rad]; `[3, n_row, n_col]`
- **`transposed`** *(optional)* — Returns the transpose of the rotation matrix

### Output:
- **`rotation`** — Rotation matrices in column-major order; `[9, n_row, n_col]`

---
<a id="fast_acos"></a>
## fast_acos
Compute elementwise approximate arc-cosine of a vector

Not in: MATLAB, Python

- Max error for x in [-1, 1]: ~2 ULP (~2.4e-7), both kernels; values outside [-1, 1] produce NaN
- Double input is rounded to float first; near ±1 this adds up to 6e-8 / sqrt(1 - x²) (at most 3.5e-4 rad)
- Single-precision output also for double input, as [fast_sincos](#fast_sincos)

### Declaration:
```cpp
void quadriga_lib::fast_acos(const arma::fvec &x, arma::fvec &acos_x, int use_kernel = 0);
void quadriga_lib::fast_acos(const arma::vec &x,  arma::fvec &acos_x, int use_kernel = 0);
```

### Inputs:
- **`x`** — Input values in [-1, 1]; `[n_elem]`
- **`use_kernel`** *(optional)* — Kernel, as in [fast_sincos](#fast_sincos)

### Output:
- **`acos_x`** — acos(x); `[n_elem]`

---
<a id="fast_asin"></a>
## fast_asin
Compute elementwise approximate arc-sine of a vector

Not in: MATLAB, Python

- Max error for x in [-1, 1]: ~2 ULP (~2.4e-7), both kernels; values outside [-1, 1] produce NaN
- Double input is rounded to float first; near ±1 this adds up to 6e-8 / sqrt(1 - x²) (at most 3.5e-4 rad)
- Single-precision output also for double input, as [fast_sincos](#fast_sincos)

### Declaration:
```cpp
void quadriga_lib::fast_asin(const arma::fvec &x, arma::fvec &asin_x, int use_kernel = 0);
void quadriga_lib::fast_asin(const arma::vec &x,  arma::fvec &asin_x, int use_kernel = 0);
```

### Inputs:
- **`x`** — Input values in [-1, 1]; `[n_elem]`
- **`use_kernel`** *(optional)* — Kernel, as in [fast_sincos](#fast_sincos)

### Output:
- **`asin_x`** — asin(x); `[n_elem]`

---
<a id="fast_atan2"></a>
## fast_atan2
Compute elementwise approximate two-argument arc-tangent of two vectors

Not in: MATLAB, Python

- Returns angles in [-pi, pi]; max error ~3 ULP (~3.6e-7), both kernels
- Special values as std::atan2 (both kernels): atan2(±0, +0) = ±0, atan2(±0, -0) = ±pi, atan2(±0, x < 0) = ±pi,
  atan2(±inf, ±inf) = ±pi/4 or ±3pi/4; NaN in x or y gives NaN
- Single-precision output also for double input, as [fast_sincos](#fast_sincos)

### Declaration:
```cpp
void quadriga_lib::fast_atan2(const arma::fvec &y, const arma::fvec &x, arma::fvec &atan2_yx, int use_kernel = 0);
void quadriga_lib::fast_atan2(const arma::vec &y,  const arma::vec &x,  arma::fvec &atan2_yx, int use_kernel = 0);
```

### Inputs:
- **`y`** — Y-coordinates (numerator); `[n_elem]`
- **`x`** — X-coordinates (denominator); `[n_elem]`
- **`use_kernel`** *(optional)* — Kernel, as in [fast_sincos](#fast_sincos)

### Output:
- **`atan2_yx`** — atan2(y, x); `[n_elem]`

---
<a id="fast_cart2geo"></a>
## fast_cart2geo
Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length

Also in: [MATLAB](../mex/math-functions.md#fast_cart2geo) · [Python](../python/math-functions.md#fast_cart2geo)

- Conversion: len = sqrt(x² + y² + z²), az = atan2(y, x), el = asin(clamp(z / len, -1, 1))
- Inputs are arbitrary 3D vectors (not required to be unit-length); `len` returns the Euclidean norm
- z/len is clamped to [-1, 1] before asin to guard against len == 0 and FMA rounding artefacts pushing abs(z/len) slightly above 1
- All inputs must have the same length
- AVX2 kernel computes the angles in single precision (double outputs are cast back from float); for `dtype=double`,
  `len` and z / len are computed in double precision, so very small or very large vectors keep their direction;
  GENERIC kernel preserves full `dtype` precision
- Azimuth in [-pi, pi] (atan2 special values as in [fast_atan2](#fast_atan2)); NaN input gives NaN in `az`, `el` and `len`

### Declaration:
```cpp
void quadriga_lib::fast_cart2geo(const arma::fvec &x, const arma::fvec &y, const arma::fvec &z,
                                 arma::fvec &az, arma::fvec &el, arma::fvec *len = nullptr, int use_kernel = 0);

void quadriga_lib::fast_cart2geo(const arma::vec &x, const arma::vec &y, const arma::vec &z,
                                 arma::vec &az, arma::vec &el, arma::vec *len = nullptr, int use_kernel = 0);
```

### Inputs:
- **`x`** — X-coordinates; `[n_elem]`
- **`y`** — Y-coordinates; `[n_elem]`
- **`z`** — Z-coordinates; `[n_elem]`
- **`use_kernel`** *(optional)* — Kernel selection: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2 (throws if AVX2 unavailable); other values throw

### Outputs:
- **`az`** — Azimuth angles; `[n_elem]`
- **`el`** — Elevation angles; `[n_elem]`
- **`len`** *(optional)* — Euclidean vector length sqrt(x² + y² + z²); `[n_elem]`

### See also:
- [fast_geo2cart](#fast_geo2cart) (inverse conversion)

---
<a id="fast_geo2cart"></a>
## fast_geo2cart
Convert elementwise azimuth/elevation angles to Cartesian coordinates

Also in: [MATLAB](../mex/math-functions.md#fast_geo2cart) · [Python](../python/math-functions.md#fast_geo2cart)

- Conversion: x = cos(el)*cos(az)*len, y = cos(el)*sin(az)*len, z = sin(el)*len
- Optional pointer outputs `sin_az`, `cos_az`, `sin_el`, `cos_el` return intermediate sin/cos values; pass `nullptr` to skip
- Precision: GENERIC kernel uses full `dtype` precision (double or float).
- AVX2 kernel computes sine and cosine in single precision (max. error about 1.3e-7 for angles up to 1e10;
  `dtype=double` angles are reduced in double precision first), results widened back. Use `use_kernel=1` to force
  GENERIC if full double precision is required.

### Declaration:
```cpp
void quadriga_lib::fast_geo2cart(
    const arma::Col<dtype> &az,
    const arma::Col<dtype> &el,
    arma::Col<dtype> &x,
    arma::Col<dtype> &y,
    arma::Col<dtype> &z,
    arma::Col<dtype> *sin_az = nullptr,
    arma::Col<dtype> *cos_az = nullptr,
    arma::Col<dtype> *sin_el = nullptr,
    arma::Col<dtype> *cos_el = nullptr,
    const arma::Col<dtype> &len = arma::Col<dtype>(),
    int use_kernel = 0);
```

### Inputs:
- **`az`** — Azimuth angles; `[n_elem]`
- **`el`** — Elevation angles; `[n_elem]`
- **`len`** *(optional)* — Euclidean vector length sqrt(x² + y² + z²); `[n_elem]`
- **`use_kernel`** *(optional)* — Kernel selection: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2 (throws if AVX2 unavailable); other values throw

### Outputs:
- **`x`** — X-coordinates; `[n_elem]`
- **`y`** — Y-coordinates; `[n_elem]`
- **`z`** — Z-coordinates; `[n_elem]`
- **`sin_az`** *(optional)* — sin(az); `[n_elem]` or `nullptr`
- **`cos_az`** *(optional)* — cos(az); `[n_elem]` or `nullptr`
- **`sin_el`** *(optional)* — sin(el); `[n_elem]` or `nullptr`
- **`cos_el`** *(optional)* — cos(el); `[n_elem]` or `nullptr`

### See also:
- [fast_cart2geo](#fast_cart2geo) (inverse conversion)

---
<a id="fast_sincos"></a>
## fast_sincos
Compute elementwise approximate sine and/or cosine of a vector

Also in: [MATLAB](../mex/math-functions.md#fast_sincos) · [Python](../python/math-functions.md#fast_sincos)

- Max absolute error about 1.3e-7 (2^-22.9) against the exact sine / cosine of the input value for |x| <= 1e10
  (both kernels; the argument is reduced in double precision where single precision is not enough)
- Results stay in [-1, 1] for all finite x; inf and NaN give NaN
- The AVX2 kernel is fastest for |x| <= 2^22 (4.2e6); larger arguments take about 1.5 times as long
- Either `sin_x` or `cos_x` may be `nullptr` to skip that computation
- Outputs are single precision also for double input: the fast_* functions are meant for very large arrays, where a
  double output would add a pass and a copy that doubles the peak memory; double-precision callers use the standard library

### Declaration:
```cpp
void quadriga_lib::fast_sincos(const arma::fvec &x, arma::fvec *sin_x = nullptr, arma::fvec *cos_x = nullptr,
                               int use_kernel = 0);
void quadriga_lib::fast_sincos(const arma::vec &x,  arma::fvec *sin_x = nullptr, arma::fvec *cos_x = nullptr,
                               int use_kernel = 0);
```

### Inputs:
- **`x`** — Input angles; `[n_elem]`
- **`use_kernel`** *(optional)* — Kernel: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2; throws if unavailable

### Outputs:
- **`sin_x`** *(optional)* — sin(x); `[n_elem]` or `nullptr`
- **`cos_x`** *(optional)* — cos(x); `[n_elem]` or `nullptr`

---
<a id="fast_slerp"></a>
## fast_slerp
Compute elementwise approximate SLERP interpolation between two complex-valued vectors

Not in: MATLAB, Python

- Interpolates phase via SLERP on normalized directions; amplitudes are linearly interpolated
- Weight `w=0` returns A, `w=1` returns B; per-element weights in [0, 1]
- Near-antipodal inputs (phase difference close to pi) fall back to linear interpolation smoothly
- If both input amplitudes are negligible (below 1.19e-7), output is zero
- Max relative error vs. double-precision reference about 5e-7 for phase differences up to 2 rad, 1.2e-5 up to
  3 rad (both kernels); larger towards the near-antipodal transition to linear interpolation
- The result of an element does not depend on the other elements
- Single-precision output also for double input, as [fast_sincos](#fast_sincos)

### Declaration:
```cpp
void quadriga_lib::fast_slerp(const arma::fvec &A_re, const arma::fvec &A_im,
                              const arma::fvec &B_re, const arma::fvec &B_im,
                              const arma::fvec &w,
                              arma::fvec &X_re, arma::fvec &X_im, int use_kernel = 0);

void quadriga_lib::fast_slerp(const arma::vec &A_re, const arma::vec &A_im,
                              const arma::vec &B_re, const arma::vec &B_im,
                              const arma::vec &w,
                              arma::fvec &X_re, arma::fvec &X_im, int use_kernel = 0);
```

### Inputs:
- **`A_re`** — Real part of source A; `[n_elem]`
- **`A_im`** — Imaginary part of source A; `[n_elem]`
- **`B_re`** — Real part of source B; `[n_elem]`
- **`B_im`** — Imaginary part of source B; `[n_elem]`
- **`w`** — Per-element interpolation weight in [0, 1]; `[n_elem]`
- **`use_kernel`** *(optional)* — Kernel, as in [fast_sincos](#fast_sincos)

### Outputs:
- **`X_re`** — Real part of interpolated result; `[n_elem]`
- **`X_im`** — Imaginary part of interpolated result; `[n_elem]`

---
<a id="interp_2D"></a>
## interp_2D
Perform linear interpolation (1D or 2D) on multiple data sets

Also in: [MATLAB](../mex/math-functions.md#interp) · [Python](../python/math-functions.md#interp)

- Interpolates given input data at specified output points.
- Outside the range of the input sample points, the nearest end value is returned (constant extrapolation);
  output sample points that are NaN give NaN.
- The input sample points need not be sorted (they are sorted internally); repeated values are allowed;
  they must be finite.
- `interp_1D` interpolates the columns of a matrix; a single data set is a matrix with one column.
- Data types (`dtype`): `float` or `double`

### Declaration:
```cpp
void quadriga_lib::interp_2D(
    const arma::Col<dtype> &x,
    const arma::Col<dtype> &y,
    const arma::Cube<dtype> &input,
    const arma::Col<dtype> &x_new,
    const arma::Col<dtype> &y_new,
    arma::Cube<dtype> &output);

void quadriga_lib::interp_1D(
    const arma::Col<dtype> &x,
    const arma::Mat<dtype> &input,
    const arma::Col<dtype> &x_new,
    arma::Mat<dtype> &output);
```

### Inputs:
- **`x`** — Input x-axis sample points; `[nx]`
- **`y`** — Input y-axis sample points (2D only); `[ny]`
- **`input`** — Input data; 2D: `[ny, nx, ne]`; 1D: `[nx, ne]`; `ne` = number of data sets
- **`x_new`** — Output x-axis sample points; `[nx_new]`
- **`y_new`** — Output y-axis sample points (2D only); `[ny_new]`

### Output:
- **`output`** — Interpolated data; 2D: `[ny_new, nx_new, ne]`; 1D: `[nx_new, ne]`

### Examples:
- 2D interpolation example:
```cpp
arma::cube input(5, 5, 2, arma::fill::randu); // example input data
arma::vec x = arma::linspace(0, 4, 5);
arma::vec y = arma::linspace(0, 4, 5);
arma::vec x_new = arma::linspace(0, 4, 10);
arma::vec y_new = arma::linspace(0, 4, 10);

arma::cube output;
quadriga_lib::interp_2D(x, y, input, x_new, y_new, output);
```
- 1D interpolation example:
```cpp
arma::vec input = arma::linspace(0, 1, 5);
arma::vec x = arma::linspace(0, 4, 5);
arma::vec x_new = arma::linspace(0, 4, 10);

arma::mat output;
quadriga_lib::interp_1D(x, input, x_new, output);
```
