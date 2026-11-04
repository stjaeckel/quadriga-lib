# Math functions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-math-functions

| Function | Summary |
| --- | --- |
| [cart2geo](#cart2geo) | Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length |
| [geo2cart](#geo2cart) | Convert elementwise azimuth/elevation angles to Cartesian coordinates |

---
<a id="cart2geo"></a>
## cart2geo
Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length

Also in: [C++](../cpp/math-functions.md#fast_cart2geo) · [MATLAB](../mex/math-functions.md#cart2geo)

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
geo_coords     = quadriga_lib.tools.cart2geo( cart, y, z, combine, use_kernel )
az, el, length = quadriga_lib.tools.cart2geo( cart, y, z, combine=False )
```

### Inputs:
- **`cart`** — Combined Cartesian coordinates `(3, n, m)`; or, when `y` and `z` are given, the x-coordinates `(n, m)`
- **`y`** — Y-coordinates `(n, m)`; provide together with `z` for separate inputs; default: None
- **`z`** — Z-coordinates `(n, m)`; provide together with `y`; default: None
- **`combine`** — If True, return a single `(3, n, m)` array; if False, return separate az, el, length arrays; default: True
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (throws if unavailable); other values raise; default: 1

### Outputs:
- **`geo_coords`** — Combined geographic coordinates `(3, n, m)`; row 0 = azimuth, row 1 = elevation, row 2 = vector length; returned when `combine` is True
- **`az`** — Azimuth angles `(n, m)`; returned when `combine` is False
- **`el`** — Elevation angles `(n, m)`; returned when `combine` is False
- **`length`** — Vector length `(n, m)`; returned when `combine` is False

---
<a id="geo2cart"></a>
## geo2cart
Convert elementwise azimuth/elevation angles to Cartesian coordinates

Also in: [C++](../cpp/math-functions.md#fast_geo2cart) · [MATLAB](../mex/math-functions.md#geo2cart)

- Conversion: x = cos(el)·cos(az)·len, y = cos(el)·sin(az)·len, z = sin(el)·len
- Two mutually exclusive input forms: a combined array `az` holding geographic coordinates
  (3, n, m), or separate `az`, `el`, `len` of shape (n, m)
- For combined input, row 0 = azimuth, row 1 = elevation, row 2 = length
- `len` is optional in separate mode; omit it for unit length
- Inverse of `cart2geo`; round-trips in both the combined and separate forms
- The AVX2 kernel computes sine and cosine in single precision (angles reduced in double precision first; max. error
  about 1.3e-7 for angles up to 1e10); use `use_kernel = 1` for full double precision

### Usage:
```python
cart    = quadriga_lib.tools.geo2cart( az, el, len, combine, use_kernel )
x, y, z = quadriga_lib.tools.geo2cart( az, el, len, combine=False )
```

### Inputs:
- **`az`** — Azimuth angles `(n, m)`; or, when `el` is omitted, combined geographic coordinates
  `(3, n, m)` (row 0 = azimuth, row 1 = elevation, row 2 = length)
- **`el`** — Elevation angles `(n, m)`; omit for combined input; default: None
- **`len`** — Vector length `(n, m)`; omit for unit length; ignored for combined input; default: None
- **`combine`** — If True, return a single `(3, n, m)` array; if False, return separate x, y, z  arrays; default: True
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (throws
  if unavailable); other values raise; default: 1

### Outputs:
- **`cart`** — Combined Cartesian coordinates `(3, n, m)`; row 0 = x, row 1 = y, row 2 = z; returned when `combine` is True
- **`x`** — X-coordinates `(n, m)`; returned when `combine` is False
- **`y`** — Y-coordinates `(n, m)`; returned when `combine` is False
- **`z`** — Z-coordinates `(n, m)`; returned when `combine` is False
