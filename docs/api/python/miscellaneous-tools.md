# Miscellaneous / Tools

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-miscellaneous-tools

| Function | Summary |
| --- | --- |
| [components](#components) | Returns the version numbers of all quadriga-lib sub-components |
| [version](#version) | Returns the quadriga-lib version number |
| [write_png](#write_png) | Write data to a PNG file |

---
<a id="components"></a>
## components
Returns the version numbers of all quadriga-lib sub-components

Not in: C++, MATLAB

### Usage:
```python
components = quadriga_lib.components()
```

---
<a id="version"></a>
## version
Returns the quadriga-lib version number

Also in: [C++](../cpp/miscellaneous-tools.md#quadriga_lib_version) · [MATLAB](../mex/miscellaneous-tools.md#version)

- If Quadriga-Lib was compiled with AVX2 support and the CPU supports intrinsic AVX2 instructions,
  a suffix `_AVX2` is added after the version number

### Usage:
```python
version = quadriga_lib.version()
```

---
<a id="write_png"></a>
## write_png
Write data to a PNG file

Also in: [C++](../cpp/site-specific-simulation-tools.md#write_png) · [MATLAB](../mex/miscellaneous-tools.md#write_png)

- Converts input data into a color-coded PNG file for visualization
- Support optional selection of a colormap, as well a minimum and maximum value limits
- Uses the [LodePNG](https://github.com/lvandeve/lodepng) library for PNG writing
- The auto-detected range ignores `NaN` and `+-inf`; `inf` and `-inf` values get the last and first colour
- `NaN` values get the first colour of the map
- A degenerate range (`max <= min` after auto-detection, e.g. constant or single-value data, all-`NaN` data, or log
  input without a positive value) maps every value to the first colour
- Raises `ValueError` for empty `data`, an infinite `min_val` or `max_val`, and for `min_val > max_val` (both given)

### Usage:
```python
quadriga_lib.tools.write_png( fn, data, *, colormap, min_val, max_val, log_transform )
```

### Arguments:
- **`fn`**\
  Filename of the PNG file, required

- **`data`**\
  Data matrix, required, size `[N, M]` 

- **`colormap`** (optional)\
  Colormap for the visualization, string, supported are 'jet', 'parula', 'winter', 'hot', 'turbo',
  'copper', 'spring', 'cool', 'gray', 'autumn', 'summer', optional, default = 'jet'

- **`min_val`** (optional)\
  Lower clipping bound; values below get the first colour. NaN (default): data minimum.

- **`max_val`** (optional)\
  Upper clipping bound; values above get the last colour. NaN (default): data maximum.

- **`log_transform`** (optional)\
  If enabled, the `data` values are transformed to the log-domain (`10*log10(data)`) before processing.
  Default: false (disabled)
