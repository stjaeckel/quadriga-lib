You are compressing API documentation for a C++ radio channel modelling library (quadriga-lib). You will receive one documentation block in the old verbose format. Return the compressed block in the new format. Output ONLY the raw documentation block starting with `/*!MD` and ending with `MD!*/`. No commentary, no markdown fencing.

FORMAT RULES:

# function_name_lower_case
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

