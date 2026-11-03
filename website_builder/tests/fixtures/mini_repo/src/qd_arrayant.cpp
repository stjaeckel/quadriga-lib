// Test fixture
#include <string>

/*!SECTION
Array antenna class
SECTION!*/

/*!MD
# arrayant<++>
Class for storing array antenna models

- Holds the field patterns; see .[[interpolate]]

## Attributes:<br>
| Attribute    | Description              |
| ------------ | ------------------------ |
| `e_theta_re` | E-theta field, real part |
MD!*/

/*!MD
# .interpolate
Interpolate the field patterns

- Uses [[arrayant_interpolate_multi]]; file format: <a target="_blank" rel="noopener noreferrer" href="formats.html#6cab4884">QDANT</a>
- Material data: <a href="http://quadriga-lib.org/quadriga_lib_material_model.md">material model</a>

## Declaration:
```
void interpolate(const arma::Mat<dtype> *azimuth, const arma::Mat<dtype> *elevation) const;
```

## Inputs:
- **`azimuth`** — Azimuth angles in rad; `[n_out, n_ang]`
- **`elevation`** *(optional)* - Elevation angles in rad
- A note about both inputs:<br>
  they must have the same size

## Example:
```
ant.interpolate(&az, &el);  // see [[not_a_link]]
```
MD!*/
