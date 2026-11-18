# Math functions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-math-functions

| Function | Summary |
| --- | --- |
| [calc_rotation_matrix](#calc_rotation_matrix) | Calculate rotation matrices from Euler angles |
| [fast_cart2geo](#fast_cart2geo) | Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length |
| [fast_geo2cart](#fast_geo2cart) | Convert elementwise azimuth/elevation angles to Cartesian coordinates |
| [fast_sincos](#fast_sincos) | Compute elementwise approximate sine and cosine of a vector |
| [interp](#interp) | Perform linear interpolation (1D or 2D) on multiple data sets |

---
<a id="calc_rotation_matrix"></a>
## calc_rotation_matrix
Calculate rotation matrices from Euler angles

Also in: [C++](../cpp/math-functions.md#calc_rotation_matrix) · [MATLAB](../mex/math-functions.md#calc_rotation_matrix)

- Computes 3×3 rotation matrices from Euler angles (bank, tilt, head) in column-major order (9 elements per orientation)
- R = Rz(heading)·Ry(tilt)·Rx(bank), right-handed as in 3GPP TR 38.901; a positive tilt turns the x-axis down

### Usage:
```python
rotation = quadriga_lib.tools.calc_rotation_matrix( orientation, *, transposed )
```

### Inputs:
- **`orientation`** — Euler angles (bank, tilt, head) in rad; `(3,)`, `(3, n)` or `(3, n_row, n_col)`
- **`transposed`** — Returns the transpose of the rotation matrix; default: `False`

### Output:
- **`rotation`** — Rotation matrices in column-major order; same rank as `orientation`: `(9,)`, `(9, n)` or
  `(9, n_row, n_col)`

---
<a id="fast_cart2geo"></a>
## fast_cart2geo
Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length

Also in: [C++](../cpp/math-functions.md#fast_cart2geo) · [MATLAB](../mex/math-functions.md#fast_cart2geo)

- Computes: length = sqrt(x² + y² + z²), az = atan2(y, x), el = asin(clamp(z / length, -1, 1))
- Inputs are arbitrary 3D vectors (not required to be unit length); length returns the Euclidean norm
- z / length is clamped to [-1, 1] before asin to guard against length == 0 and rounding artifacts
- Azimuth is in [-pi, pi], elevation in [-pi/2, pi/2]; elevation pi/2 points to the zenith, 0 to the horizon
- Special values of atan2 as in numpy (e.g. atan2(0, -0) = pi); NaN input gives NaN in az, el and length
- Two mutually exclusive input forms: a combined array `cart` of shape (3, n, m), or separate `x`, `y`, `z` of shape (n, m)
- The AVX2 kernel computes the angles in single precision (length and z / length in double precision); use
  `use_kernel = 1` for full double precision

### Usage:
```python
geo_coords     = quadriga_lib.tools.fast_cart2geo( cart, y, z, *, combine, use_kernel )
az, el, length = quadriga_lib.tools.fast_cart2geo( cart, y, z, combine=False )
```

### Inputs:
- **`cart`** — Combined Cartesian coordinates `(3, n, m)`; or, when `y` and `z` are given, the x-coordinates `(n, m)`
- **`y`** — Y-coordinates `(n, m)`; provide together with `z` for separate inputs; default: None
- **`z`** — Z-coordinates `(n, m)`; provide together with `y`; default: None
- **`combine`** — If True, return a single `(3, n, m)` array; if False, return separate az, el, length arrays; default: True
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (throws if unavailable); other values raise; default: 0

### Outputs:
- **`geo_coords`** — Combined geographic coordinates `(3, n, m)`; row 0 = azimuth, row 1 = elevation, row 2 = vector length; returned when `combine` is True
- **`az`** — Azimuth angles `(n, m)`; returned when `combine` is False
- **`el`** — Elevation angles `(n, m)`; returned when `combine` is False
- **`length`** — Vector length `(n, m)`; returned when `combine` is False

---
<a id="fast_geo2cart"></a>
## fast_geo2cart
Convert elementwise azimuth/elevation angles to Cartesian coordinates

Also in: [C++](../cpp/math-functions.md#fast_geo2cart) · [MATLAB](../mex/math-functions.md#fast_geo2cart)

- Conversion: x = cos(el)·cos(az)·len, y = cos(el)·sin(az)·len, z = sin(el)·len
- Two mutually exclusive input forms: a combined array `az` holding geographic coordinates
  (3, n, m), or separate `az`, `el`, `len` of shape (n, m)
- For combined input, row 0 = azimuth, row 1 = elevation, row 2 = length
- `len` is optional in separate mode; omit it for unit length
- Inverse of `fast_cart2geo`; round-trips in both the combined and separate forms
- The AVX2 kernel computes sine and cosine in single precision (angles reduced in double precision first; max. error
  about 1.3e-7 for angles up to 1e10); use `use_kernel = 1` for full double precision

### Usage:
```python
cart    = quadriga_lib.tools.fast_geo2cart( az, el, len, *, combine, use_kernel )
x, y, z = quadriga_lib.tools.fast_geo2cart( az, el, len, combine=False )
```

### Inputs:
- **`az`** — Azimuth angles `(n, m)`; or, when `el` is omitted, combined geographic coordinates
  `(3, n, m)` (row 0 = azimuth, row 1 = elevation, row 2 = length)
- **`el`** — Elevation angles `(n, m)`; omit for combined input; default: None
- **`len`** — Vector length `(n, m)`; omit for unit length; ignored for combined input; default: None
- **`combine`** — If True, return a single `(3, n, m)` array; if False, return separate x, y, z  arrays; default: True
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (throws
  if unavailable); other values raise; default: 0

### Outputs:
- **`cart`** — Combined Cartesian coordinates `(3, n, m)`; row 0 = x, row 1 = y, row 2 = z; returned when `combine` is True
- **`x`** — X-coordinates `(n, m)`; returned when `combine` is False
- **`y`** — Y-coordinates `(n, m)`; returned when `combine` is False
- **`z`** — Z-coordinates `(n, m)`; returned when `combine` is False

---
<a id="fast_sincos"></a>
## fast_sincos
Compute elementwise approximate sine and cosine of a vector

Also in: [C++](../cpp/math-functions.md#fast_sincos) · [MATLAB](../mex/math-functions.md#fast_sincos)

- Max absolute error about 1.3e-7 (2^-22.9) against the exact sine / cosine of the input value for |x| <= 1e10
  (the argument is reduced in double precision where single precision is not enough)
- Results stay in [-1, 1] for all finite x; inf and NaN give NaN
- Outputs are single precision: the fast_* functions are meant for very large arrays; double-precision callers use
  `numpy.sin` / `numpy.cos`

### Usage:
```python
sin_x, cos_x = quadriga_lib.tools.fast_sincos( x, *, use_kernel )
```

### Inputs:
- **`x`** — Input angles in rad; any shape
- **`use_kernel`** — Kernel: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2; throws if
  unavailable; default: `0`

### Outputs:
- **`sin_x`** — sin(x); float32; shape of `x`
- **`cos_x`** — cos(x); float32; shape of `x`

---
<a id="interp"></a>
## interp
Perform linear interpolation (1D or 2D) on multiple data sets

Also in: [C++](../cpp/math-functions.md#interp_2D) · [MATLAB](../mex/math-functions.md#interp)

- Interpolates given input data at specified output points.
- Outside the range of the input sample points, the nearest end value is returned (constant extrapolation);
  output sample points that are NaN give NaN.
- The input sample points need not be sorted; repeated values are allowed; they must be finite.
- `y = None` (or empty) selects the 1D case.

### Usage:
```python
# 2D case
output = quadriga_lib.tools.interp( x, y, input, x_new, y_new )

# 1D case
output = quadriga_lib.tools.interp( x, None, input, x_new )
```

### Inputs:
- **`x`** — Input x-axis sample points; `(nx,)`
- **`y`** — Input y-axis sample points; `(ny,)`; `None` for 1D
- **`input`** — Input data; 2D: `(ny, nx)` or `(ny, nx, ne)`; 1D: `(nx,)` or `(nx, ne)`; `ne` = number of data sets
- **`x_new`** — Output x-axis sample points; `(nx_new,)`
- **`y_new`** — Output y-axis sample points, required in 2D, `None` or empty in 1D; `(ny_new,)`; default: `None`

### Output:
- **`output`** — Interpolated data, same rank as `input`; 2D: `(ny_new, nx_new)` or `(ny_new, nx_new, ne)`;
  1D: `(nx_new,)` or `(nx_new, ne)`
