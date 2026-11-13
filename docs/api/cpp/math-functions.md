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
| [interp_2D](#interp_2D) | Perform linear interpolation (1D or 2D) on single or multiple data sets |

---
<a id="calc_rotation_matrix"></a>
## calc_rotation_matrix
Calculate rotation matrices from Euler angles

Also in: [MATLAB](../mex/math-functions.md#calc_rotation_matrix) · Not in: Python

- Computes 3×3 rotation matrices from Euler angles (bank, tilt, head) in column-major order (9 elements per orientation)
- R = Rz(heading)·Ry(tilt)·Rx(bank), right-handed as in 3GPP TR 38.901; a positive tilt turns the x-axis down
- Internally uses double precision regardless of `dtype`

### Declaration:
```cpp
arma::Cube<dtype> quadriga_lib::calc_rotation_matrix(
    const arma::Cube<dtype> &orientation,
    bool transposeR = false);

arma::Mat<dtype> quadriga_lib::calc_rotation_matrix(
    const arma::Mat<dtype> &orientation,
    bool transposeR = false);

arma::Col<dtype> quadriga_lib::calc_rotation_matrix(
    const arma::Col<dtype> &orientation,
    bool transposeR = false);
```

### Inputs:
- **`orientation`** — Euler angles (bank, tilt, head); `[3, n_row, n_col]` or `[3, n_mat]` or `[3]`
- **`transposeR`** *(optional)* — Returns the transpose of the rotation matrix

### Returns:
- Rotation matrices in column-major order; `[9, n_row, n_col]` or `[9, n_mat]` or `[9]`

---
<a id="fast_acos"></a>
## fast_acos
Compute elementwise approximate arc-cosine of a vector

Not in: MATLAB, Python

- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Max error for x in [-1, 1]: ~2 ULP (~2.4e-7); values outside [-1, 1] produce NaN

### Declaration:
```cpp
void quadriga_lib::fast_acos(const arma::fvec &x, arma::fvec &c);
void quadriga_lib::fast_acos(const arma::vec &x,  arma::fvec &c);
```

### Inputs:
- **`x`** — Input values in [-1, 1]; `[n_elem]`

### Outputs:
- **`c`** — acos(x); `[n_elem]`

---
<a id="fast_asin"></a>
## fast_asin
Compute elementwise approximate arc-sine of a vector

Not in: MATLAB, Python

- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Max error for x in [-1, 1]: ~2 ULP (~2.4e-7); values outside [-1, 1] produce NaN

### Declaration:
```cpp
void quadriga_lib::fast_asin(const arma::fvec &x, arma::fvec &s);
void quadriga_lib::fast_asin(const arma::vec &x,  arma::fvec &s);
```

### Inputs:
- **`x`** — Input values in [-1, 1]; `[n_elem]`

### Outputs:
- **`s`** — asin(x); `[n_elem]`

---
<a id="fast_atan2"></a>
## fast_atan2
Compute elementwise approximate two-argument arc-tangent of two vectors

Not in: MATLAB, Python

- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Returns angles in [-pi, pi]; max error ~3 ULP (~3.6e-7)
- Special values as std::atan2 (both kernels): atan2(±0, +0) = ±0, atan2(±0, -0) = ±pi, atan2(±0, x < 0) = ±pi,
  atan2(±inf, ±inf) = ±pi/4 or ±3pi/4; NaN in x or y gives NaN

### Declaration:
```cpp
void quadriga_lib::fast_atan2(const arma::fvec &y, const arma::fvec &x, arma::fvec &a);
void quadriga_lib::fast_atan2(const arma::vec &y,  const arma::vec &x,  arma::fvec &a);
```

### Inputs:
- **`y`** — Y-coordinates (numerator); `[n_elem]`
- **`x`** — X-coordinates (denominator); `[n_elem]`

### Outputs:
- **`a`** — atan2(y, x); `[n_elem]`

---
<a id="fast_cart2geo"></a>
## fast_cart2geo
Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length

Also in: [MATLAB](../mex/math-functions.md#cart2geo) · [Python](../python/math-functions.md#cart2geo)

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
- **`use_kernel`** — Kernel selection: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2 (throws if AVX2 unavailable); other values throw; default `0`

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

Also in: [MATLAB](../mex/math-functions.md#geo2cart) · [Python](../python/math-functions.md#geo2cart)

- Conversion: x = cos(el)*cos(az)*len, y = cos(el)*sin(az)*len, z = sin(el)*len
- Optional pointer outputs `sAZ`, `cAZ`, `sEL`, `cEL` return intermediate sin/cos values; pass `nullptr` to skip
- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Precision: GENERIC kernel uses full `dtype` precision (double or float).
- AVX2 kernel computes sine and cosine in single precision (max. error about 1.3e-7 for angles up to 1e10;
  `dtype=double` angles are reduced in double precision first), results widened back. Use `use_kernel=1` to force
  GENERIC if full double precision is required.

### Declaration:
```cpp
void fast_geo2cart(
    const arma::Col<dtype> &az,
    const arma::Col<dtype> &el,
    arma::Col<dtype> &x,
    arma::Col<dtype> &y,
    arma::Col<dtype> &z,
    arma::Col<dtype> *sAZ = nullptr,
    arma::Col<dtype> *cAZ = nullptr,
    arma::Col<dtype> *sEL = nullptr,
    arma::Col<dtype> *cEL = nullptr,
    const arma::Col<dtype> *len = nullptr,
    int use_kernel = 0);
```

### Inputs:
- **`az`** — Azimuth angles; `[n_elem]`
- **`el`** — Elevation angles; `[n_elem]`
- **`len`** *(optional)* — Euclidean vector length sqrt(x² + y² + z²); `[n_elem]`
- **`use_kernel`** — Kernel selection: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2 (throws if AVX2 unavailable); other values throw

### Outputs:
- **`x`** — X-coordinates; `[n_elem]`
- **`y`** — Y-coordinates; `[n_elem]`
- **`z`** — Z-coordinates; `[n_elem]`
- **`sAZ`** *(optional)* — sin(az); `[n_elem]` or `nullptr`
- **`cAZ`** *(optional)* — cos(az); `[n_elem]` or `nullptr`
- **`sEL`** *(optional)* — sin(el); `[n_elem]` or `nullptr`
- **`cEL`** *(optional)* — cos(el); `[n_elem]` or `nullptr`

### See also:
- [fast_cart2geo](#fast_cart2geo) (inverse conversion)

---
<a id="fast_sincos"></a>
## fast_sincos
Compute elementwise approximate sine and/or cosine of a vector

Also in: [MATLAB](../mex/math-functions.md#fast_sincos) · Not in: Python

- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- Max absolute error about 1.3e-7 (2^-22.9) against the exact sine / cosine of the input value for |x| <= 1e10
  (both kernels; the argument is reduced in double precision where single precision is not enough)
- Results stay in [-1, 1] for all finite x; inf and NaN give NaN
- The AVX2 kernel is fastest for |x| <= 2^22 (4.2e6); larger arguments take about 1.5 times as long
- Either `s` or `c` may be `nullptr` to skip that computation

### Declaration:
```cpp
void quadriga_lib::fast_sincos(const arma::fvec &x, arma::fvec *s = nullptr, arma::fvec *c = nullptr);
void quadriga_lib::fast_sincos(const arma::vec &x,  arma::fvec *s = nullptr, arma::fvec *c = nullptr);
```

### Inputs:
- **`x`** — Input angles; `[n_elem]`

### Outputs:
- **`s`** *(optional)* — sin(x); `[n_elem]` or `nullptr`
- **`c`** *(optional)* — cos(x); `[n_elem]` or `nullptr`

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
  3 rad; larger towards the near-antipodal transition to linear interpolation
- The result of an element does not depend on the other elements
- AVX2-optimized (8 complex pairs/lane); scalar fallback without AVX2

### Declaration:
```cpp
void quadriga_lib::fast_slerp(const arma::fvec &Ar, const arma::fvec &Ai,
                              const arma::fvec &Br, const arma::fvec &Bi,
                              const arma::fvec &w,
                              arma::fvec &Xr, arma::fvec &Xi);

void quadriga_lib::fast_slerp(const arma::vec &Ar, const arma::vec &Ai,
                              const arma::vec &Br, const arma::vec &Bi,
                              const arma::vec &w,
                              arma::fvec &Xr, arma::fvec &Xi);
```

### Inputs:
- **`Ar`** — Real part of source A; `[n_elem]`
- **`Ai`** — Imaginary part of source A; `[n_elem]`
- **`Br`** — Real part of source B; `[n_elem]`
- **`Bi`** — Imaginary part of source B; `[n_elem]`
- **`w`** — Per-element interpolation weight in [0, 1]; `[n_elem]`

### Outputs:
- **`Xr`** — Real part of interpolated result; `[n_elem]`
- **`Xi`** — Imaginary part of interpolated result; `[n_elem]`

---
<a id="interp_2D"></a>
## interp_2D
Perform linear interpolation (1D or 2D) on single or multiple data sets

Also in: [MATLAB](../mex/math-functions.md#interp) · Not in: Python

- Interpolates given input data at specified output points.
- Outside the range of the input sample points, the nearest end value is returned (constant extrapolation);
  output sample points that are NaN give NaN.
- The input sample points need not be sorted (they are sorted internally); repeated values are allowed;
  they must be finite.
- Supports single and multiple data sets.
- Returns interpolated results either directly or through reference argument.
- Data types (`dtype`): `float` or `double`

### Declarations:
```cpp
void interp_2D(const arma::Cube<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
               const arma::Col<dtype> &xo, const arma::Col<dtype> &yo, arma::Cube<dtype> &output);

arma::Cube<dtype> interp_2D(const arma::Cube<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
                            const arma::Col<dtype> &xo, const arma::Col<dtype> &yo);

arma::Mat<dtype> interp_2D(const arma::Mat<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
                           const arma::Col<dtype> &xo, const arma::Col<dtype> &yo);

arma::Mat<dtype> interp_1D(const arma::Mat<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &xo);

arma::Col<dtype> interp_1D(const arma::Col<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &xo);
```

### Arguments:
- `input`: Input data array/matrix (size details below)
- `xi`: Input x-axis sampling points, vector of length `nx`
- `yi`: Input y-axis sampling points (for 2D only), vector of length `ny`
- `xo`: Output x-axis sampling points, vector of length `mx`
- `yo`: Output y-axis sampling points (for 2D only), vector of length `my`
- `output`: Interpolated data cube (output argument of the first variant)

### Input / Output size details:
- 2D interpolation of multiple datasets (`arma::Cube`): input: `[ny, nx, ne]`; output: `[my, mx, ne]`
- 2D interpolation of single dataset (`arma::Mat`): input: `[ny, nx]`; output: `[my, mx]`
- 1D interpolation of multiple datasets (`arma::Mat`): input: `[nx, ne]`; output: `[mx, ne]`
- 1D interpolation of single dataset (`arma::Col`): input: `[nx]`, output: `[mx]`

### Examples:
- 2D interpolation example:
```cpp
arma::cube input(5, 5, 2, arma::fill::randu); // example input data
arma::vec xi = arma::linspace(0, 4, 5);
arma::vec yi = arma::linspace(0, 4, 5);
arma::vec xo = arma::linspace(0, 4, 10);
arma::vec yo = arma::linspace(0, 4, 10);

arma::cube output;
quadriga_lib::interp_2D(input, xi, yi, xo, yo, output);
```
- 1D interpolation example:
```cpp
arma::vec input = arma::linspace(0, 1, 5);
arma::vec xi = arma::linspace(0, 4, 5);
arma::vec xo = arma::linspace(0, 4, 10);

auto output = quadriga_lib::interp_1D(input, xi, xo);
```
