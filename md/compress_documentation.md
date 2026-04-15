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
- This doc will be consumed by AI (tokens cost money). Include what is needed to understand purpose, call signature, and correct usage. Omit classroom explanations, derivations, and restated information.
- If a default value is in the declaration, don't re-state "Default: X" in the argument description unless the value needs explanation.
- Prefer dimensions like `[n_ray, 3]` over English descriptions of matrix shape.
- One bullet per argument, no multi-line continuations.
- Tables always must have a header row
- Never use language specifier in clode blocks, e.g. ```cpp, just use triple backticks without language specifier.
- Internal types must include namespace, e.g. `quadriga_lib::arrayant` instead of just `arrayant`

Here is the old documentation block to compress:


/*!MD
# qdant_write_multi
Write a vector of array antenna objects to a single QDANT file

## Description:
- Writes multiple `arrayant` objects to a single QuaDRiGa array antenna exchange format (QDANT) file,
  one per sequential ID.
- Each entry in the input vector is assigned a 1-based ID (1, 2, ..., n_entries) and written to the file
  using the `arrayant::qdant_write` member function.
- A layout matrix of size `[n_entries, 1]` is created automatically with entries `1, 2, ..., n_entries`
  to describe the organization inside the file.
- If the output file already exists, it is deleted before writing to ensure a clean file.
- All arrayant objects in the vector are validated before writing. An exception is thrown if any entry is invalid.
- This function is the primary mechanism for storing frequency-dependent models, where each arrayant
  in the vector represents the directivity pattern at a specific frequency. The corresponding frequency
  is stored in the `center_frequency` field of each arrayant.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::qdant_write_multi(
        const std::string &fn,
        const std::vector<arrayant<dtype>> &arrayant_vec)
```

## Arguments:
- `const std::string &**fn**` (input)<br>
  Filename of the QDANT file to write. If the file already exists, it is overwritten. Cannot be empty.

- `const std::vector<arrayant<dtype>> &**arrayant_vec**` (input)<br>
  Vector of arrayant objects to write. Each entry is stored with a sequential 1-based ID. The vector must not be empty and all entries must be valid arrayant objects.

## Returns:
- `void`

## Example:
```
// Generate a loudspeaker model and write to file
arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
auto spk = quadriga_lib::generate_speaker<double>("piston", 0.05, 80.0, 12000.0,
                12.0, 12.0, 85.0, "hemisphere", 0.0, 0.0, 0.0, 0.15, 0.25, freqs, 5.0);

quadriga_lib::qdant_write_multi("speaker.qdant", spk);

// Read back a specific frequency entry (e.g. the 3rd one, ID = 3)
auto ant = quadriga_lib::qdant_read<double>("speaker.qdant", 3);
```

## See also:
- <a href="#arrayant">arrayant</a>
- <a href="#.qdant_write">arrayant.qdant_write</a>
- <a href="#qdant_read">qdant_read</a>
- <a href="#generate_speaker">generate_speaker</a>
MD!*/