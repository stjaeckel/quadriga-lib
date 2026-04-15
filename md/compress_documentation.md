You are compressing API documentation for a C++ radio channel modelling library (quadriga-lib). You will receive one documentation block in the old verbose format. Return the compressed block in the new format. Output ONLY the raw documentation block starting with `/*!MD` and ending with `MD!*/`. No commentary, no markdown fencing.

FORMAT RULES:

# function_name
One-line summary (imperative verb, no period)

## Description:
- Bullet points only, no prose paragraphs
- Each bullet: one concise fact needed to understand or correctly call the function
- Merge "Technical Notes" into these bullets if applicable
- Formulas: only include if essential for understanding behavior; use plain text (no LaTeX)
- Never use | in formulas or in the text; use abs() or "or" instead. The character | is reserved for tables only.
- Omit anything a competent C++ developer can infer from the declaration and argument list
- Allowed datatypes line stays if present

## Declaration:
- Intend arguments by 4 spaces, return type flush left
- Surround with triple backticks
- Do not modify signatures or default values
- Must end with a semicolon

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
- Function name must match the declaration exactly, exclude namespaces
- This doc will be consumed by AI (tokens cost money). Include what is needed to understand purpose, call signature, and correct usage. Omit classroom explanations, derivations, and restated information.
- If a default value is in the declaration, don't re-state "Default: X" in the argument description unless the value needs explanation.
- Prefer dimensions like `[n_ray, 3]` over English descriptions of matrix shape.
- One bullet per argument, no multi-line continuations.
- Tables always must have a header row
- Never use language specifier in clode blocks, e.g. ```cpp, just use triple backticks without language specifier.
- Internal types must include namespace, e.g. `quadriga_lib::arrayant` instead of just `arrayant`
- Links to class member functions should be in the format class_name[[.function_name]], otherwise the link will not work in the generated documentation. 
- Link to the class itself should be [[class_name]]

Here is the old documentation block to compress:

/*!MD
# quantize_delays
Fixes the path delays to a grid of delay bins

## Description:
- For channel emulation with finite delay resolution, path delays must be mapped to a fixed grid
  of delay bins (taps). Rounding delays to the nearest tap causes discontinuities in the frequency
  domain when a delay crosses a tap boundary (e.g. as a mobile terminal moves). This function
  instead approximates each path delay using two adjacent taps with power-weighted coefficients,
  producing smooth transitions.
- For a path at fractional offset &delta; between tap indices, two taps are created with complex
  coefficients scaled by (1&minus;&delta;)^&alpha; and &delta;^&alpha;, where &alpha; is the power
  exponent. The default &alpha;=1.0 (linear interpolation) is optimal for narrowband systems. Use
  &alpha;=0.5 to preserve wideband (incoherent) power.
- If input delays are already quantized (all fractional offsets below 0.01), the interpolation
  weight computation is skipped but the same delay-selection logic is used.
- The `fix_taps` parameter controls whether delay grids are shared across antenna pairs and/or
  snapshots, trading accuracy for a more compact representation.
- Input delays may be per-antenna `[n_rx, n_tx, n_path_s]` or shared `[1, 1, n_path_s]`. When
  shared and fix_taps is 0 or 3, delays are expanded internally and output delays are per-antenna.
  When shared and fix_taps is 1 or 2, output delays remain shared `[1, 1, n_taps]`.
- The number of antennas `n_rx` and `n_tx` must be the same across all snapshots, but the number
  of paths `n_path_s` may differ per snapshot.

## Declaration:
```
template <typename dtype>
void quadriga_lib::quantize_delays(
    const std::vector<arma::Cube<dtype>> *coeff_re,
    const std::vector<arma::Cube<dtype>> *coeff_im,
    const std::vector<arma::Cube<dtype>> *delay,
    std::vector<arma::Cube<dtype>> *coeff_re_quant,
    std::vector<arma::Cube<dtype>> *coeff_im_quant,
    std::vector<arma::Cube<dtype>> *delay_quant,
    dtype tap_spacing = (dtype)5.0e-9,
    arma::uword max_no_taps = 48,
    dtype power_exponent = (dtype)1.0,
    int fix_taps = 0);
```

## Arguments:
- `const std::vector<arma::Cube<dtype>> ***coeff_re**` (input)<br>
  Channel coefficients, real part. Vector of length `n_snap`, each cube of size
  `[n_rx, n_tx, n_path_s]` where `n_path_s` may differ across snapshots.

- `const std::vector<arma::Cube<dtype>> ***coeff_im**` (input)<br>
  Channel coefficients, imaginary part. Same sizes as `coeff_re`.

- `const std::vector<arma::Cube<dtype>> ***delay**` (input)<br>
  Path delays in seconds. Vector of length `n_snap`, each cube of size
  `[n_rx, n_tx, n_path_s]` or `[1, 1, n_path_s]`. The number of paths must match `coeff_re`.

- `std::vector<arma::Cube<dtype>> ***coeff_re_quant**` (output)<br>
  Output coefficients, real part. Vector of length `n_snap`, each cube of size `[n_rx, n_tx, n_taps]`.

- `std::vector<arma::Cube<dtype>> ***coeff_im_quant**` (output)<br>
  Output coefficients, imaginary part. Vector of length `n_snap`, each cube of size `[n_rx, n_tx, n_taps]`.

- `std::vector<arma::Cube<dtype>> ***delay_quant**` (output)<br>
  Output delays in seconds. Vector of length `n_snap`, each cube of size `[n_rx, n_tx, n_taps]` or
  `[1, 1, n_taps]`.

- `dtype **tap_spacing** = 5.0e-9` (input)<br>
  Spacing of the delay bins in seconds. Default: 5 ns (200 MHz sampling rate).

- `arma::uword **max_no_taps** = 48` (input)<br>
  Maximum number of output taps. 0 means unlimited.

- `dtype **power_exponent** = 1.0` (input)<br>
  Interpolation exponent &alpha;. Use 1.0 for narrowband (linear) or 0.5 for wideband (power-preserving).

- `int **fix_taps** = 0` (input)<br>
  Delay sharing mode: 0 = per tx-rx pair and snapshot, 1 = single grid for all,
  2 = per snapshot, 3 = per tx-rx pair.

## Example:
```
// Create synthetic test data: 2 snapshots with different numbers of paths
std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);
cre[0].set_size(1, 1, 3); cim[0].set_size(1, 1, 3); dl[0].set_size(1, 1, 3);
cre[1].set_size(1, 1, 2); cim[1].set_size(1, 1, 2); dl[1].set_size(1, 1, 2);
cre[0](0,0,0) = 1.0; cre[0](0,0,1) = 0.5; cre[0](0,0,2) = 0.3;
cre[1](0,0,0) = 0.8; cre[1](0,0,1) = 0.4;
cim[0].zeros(); cim[1].zeros();
dl[0](0,0,0) = 0.0; dl[0](0,0,1) = 12.5e-9; dl[0](0,0,2) = 33.4e-9;
dl[1](0,0,0) = 0.0; dl[1](0,0,1) = 10.0e-9;

std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);
```
MD!*/