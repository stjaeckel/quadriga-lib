You are compressing API documentation for a C++ radio channel modelling library (quadriga-lib). You will receive one documentation block in the old verbose format. Return the compressed block in the new format. Output ONLY the raw documentation block starting with `/*!MD` and ending with `MD!*/`. No commentary, no markdown fencing.

FORMAT RULES:

# function_name_lower_case
One-line summary (imperative verb, no period)

## Description:
- Bullet points only, no prose paragraphs
- Each bullet: one concise fact needed to understand or correctly call the function
- Merge "Technical Notes" into these bullets if applicable
- Formulas: only include if essential for understanding behavior; use plain text (no LaTeX)
- Omit anything a competent C++ developer can infer from the declaration and argument list
- Allowed datatypes line stays if present

## Declaration:
- Intend arguments by 4 spaces, return type flush left
- Surround with triple backticks
- Do not modify signatures or default values

## Input Arguments:
- **`name`** — One-line description; size in backticks at end, separated by comma, e.g., `[n_ray, 3]`
- Optional args marked with *(optional)* diretly after the name, i.e. **`name`** (optional) — One-line description
- Default values stated inline only if they are not obvious from the declaration
- Do not state datatype of it is obvious from the declaration
- Reference other library functions with [[function_name]] where helpful

## Output Arguments:
- Same one-liner style as input arguments
- Tables (e.g. type codes) may stay if they are compact and essential
- Omit if the function has no output arguments (i.e. ony one return value)

## Returns:
- Only if the function returns a value (omit for void functions)
- Same one-liner style

## Example:
- Keep only if the example demonstrates non-obvious usage
- Maximum 5 lines of code; strip verbose comments
- Omit if usage is obvious from the declaration

## See also:
- [[function_name]] (brief reason)
- Keep only genuinely related functions

COMPRESSION GUIDELINES:
- This doc will be consumed by AI (tokens cost money). Include what is needed to understand purpose, call signature, and correct usage. Omit classroom explanations, derivations, and restated information.
- If a default value is in the declaration, don't re-state "Default: X" in the argument description unless the value needs explanation.
- Prefer dimensions like `[n_ray, 3]` over English descriptions of matrix shape.
- One bullet per argument, no multi-line continuations.
- Tables always must have a header row
- Never use language specifier in clode blocks, e.g. ```cpp, just use triple backticks without language specifier.

Here is the old documentation block to compress:

/*!MD
# .interpolate
Interpolate array antenna field patterns

## Description:
- This function interpolates polarimetric antenna field patterns for a given set of azimuth and
  elevation angles.
- Member function of [[arrayant]]
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::arrayant<dtype>::interpolate(
                const arma::Mat<dtype> *azimuth,
                const arma::Mat<dtype> *elevation,
                arma::Mat<dtype> *V_re, arma::Mat<dtype> *V_im,
                arma::Mat<dtype> *H_re, arma::Mat<dtype> *H_im,
                arma::uvec i_element,
                const arma::Cube<dtype> *orientation,
                const arma::Mat<dtype> *element_pos_i,
                arma::Mat<dtype> *dist,
                arma::Mat<dtype> *azimuth_loc, arma::Mat<dtype> *elevation_loc,
                arma::Mat<dtype> *gamma) const;
```

## Arguments:
- `const arma::Mat<dtype> ***azimuth**` (input)<br>
  Azimuth angles in [rad] for which the field pattern should be interpolated. Values must be
  between -pi and pi, cannot be NULL
  Option 1:  | Use the same angles for all antenna elements (planar wave approximation)
             | Size: `[1, n_ang]`
  Option 2:  | Provide different angles for each array element (e.g. for spherical waves)
             | Size: `[n_out, n_ang]`

- `const arma::Mat<dtype> ***elevation**` (input)<br>
  Elevation angles in [rad] for which the field pattern should be interpolated. Values must be
  between -pi/2 and pi/2, cannot be NULL
  Option 1:  | Use the same angles for all antenna elements (planar wave approximation)
             | Size: `[1, n_ang]`
  Option 2:  | Provide different angles for each array element (e.g. for spherical waves)
             | Size: `[n_out, n_ang]`

- `arma::Mat<dtype> ***V_re**` (output)<br>
  Real part of the interpolated e-theta (vertical) field component, Size `[n_out, n_ang]`,
  will be resized if it does not match the required size (invalidates data pointers), cannot be NULL

- `arma::Mat<dtype> ***V_im**` (output)<br>
  Imaginary part of the interpolated e-theta (vertical) field component, Size `[n_out, n_ang]`
  will be resized if it does not match the required size (invalidates data pointers), cannot be NULL

- `arma::Mat<dtype> ***H_re**` (output)<br>
  Real part of the interpolated e-phi (horizontal) field component, Size `[n_out, n_ang]`
  will be resized if it does not match the required size (invalidates data pointers), cannot be NULL

- `arma::Mat<dtype> ***H_im**` (output)<br>
  Imaginary part of the interpolated e-phi (horizontal) field component, Size `[n_out, n_ang]`
  will be resized if it does not match the required size (invalidates data pointers), cannot be NULL

- `arma::uvec **i_element** = {}` (optional input)<br>
  The element indices for which the interpolation should be done, optional argument,
  values must be between 1 and `n_elements`. It is possible to duplicate elements, i.e. by passing
  `{1,1,2}`. If this parameter is not provided (or an empty array is passed), `i_element` is initialized
  to include all elements of the array antenna. In this case, `n_out = n_elements`,
  Length: `n_out` or  empty `{}`

- `const arma::Cube<dtype> ***orientation** = nullptr` (optional input)<br>
  This (optional) 3-element vector allows for setting orientation of the array antenna or
  of individual elements using Euler angles (bank, tilt, heading); values must be given in [rad];
  By default, the orientation is `{0,0,0}`, i.e. the broadside of the antenna points at the horizon
  towards the East. Sizes: `nullptr` (use default), `[3, 1]` (set orientation for entire array),
  `[3, n_out]` (set orientation for individual elements), or `[3, 1, n_ang]` (set orientation for
  individual angles) or `[3, n_out, n_ang]` (set orientation for individual elements and angles)

- `const arma::Mat<dtype> ***element_pos_i** = nullptr` (optional input)<br>
  Positions of the array antenna elements in local cartesian coordinates (using units
  of [m]). If this parameter is not given, the element positions from the `arrayant` object are used.
  Sizes: `nullptr` (use `arrayant.element_pos`), `[3, n_out]` (set alternative positions)

- `arma::Mat<dtype> ***dist** = nullptr` (optional output)<br>
  The effective distances between the antenna elements when seen from the direction
  of the incident path. The distance is calculated by an projection of the array positions on the normal
  plane of the incident path. This is needed for calculating the phase of the antenna response.
  Size: `nullptr` (do not calculate this) or `[n_out, n_ang]` (argument be resized if it does not already
  match this size)

- `arma::Mat<dtype> ***azimuth_loc** = nullptr` (optional output)<br>
  The azimuth angles in [rad] for the local antenna coordinate system, i.e., after
  applying the `orientation`. If no orientation vector is given, these angles are identical to the input
  azimuth angles. Size: `nullptr` or `[n_out, n_ang]`

- `arma::Mat<dtype> ***elevation_loc** = nullptr` (optional output)<br>
  The elevation angles in [rad] for the local antenna coordinate system, i.e., after
  applying the `orientation`. If no orientation vector is given, these angles are identical to the input
  elevation angles. Size: `nullptr` or `[n_out, n_ang]`

- `arma::Mat<dtype> ***gamma** = nullptr` (optional output)<br>
  Polarization rotation angles in [rad]. Size: `nullptr` or `[n_out, n_ang]`


## Example:
```
double pi = arma::datum::pi;

// Directional antenna, pointing east
auto ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);

arma::mat azimuth = {0.0, 0.5 * pi, -0.5 * pi, pi};     // Azimuth angles: East, North, South, West
arma::mat elevation(1, azimuth.n_elem);                 // Initialize to 0
arma::mat V_re, V_im, H_re, H_im;                       // Output variables (uninitialized)
ant.interpolate(&azimuth, &elevation, &V_re, &V_im, &H_re, &H_im);
V_re.print();
```
MD!*/


