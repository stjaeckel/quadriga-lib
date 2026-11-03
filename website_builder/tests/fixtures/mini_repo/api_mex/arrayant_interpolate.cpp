// Test fixture

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_INTERPOLATE
Interpolate array antenna field patterns

## Usage:
```
[ V_re, V_im ] = quadriga_lib.arrayant_interpolate( arrayant, azimuth );
```

## Inputs:
- **`arrayant`** — Struct; field layout as in [[version]]
- **`azimuth`** *(optional)* — Azimuth angles in rad

## Outputs:
- **`V_re`**, **`V_im`** — Interpolated field, real and imaginary part
MD!*/
