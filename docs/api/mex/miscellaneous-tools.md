# Miscellaneous / Tools

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/mex_api.html#sec-miscellaneous-tools

| Function | Summary |
| --- | --- |
| [version](#version) | Returns the quadriga-lib version number |
| [write_png](#write_png) | Write a data matrix to a color-coded PNG file |

---
<a id="version"></a>
## version
Returns the quadriga-lib version number

Also in: [Python](../python/miscellaneous-tools.md#version) · Not in: C++

- If Quadriga-Lib was compiled with AVX2 support and the CPU supports intrinsic AVX2 instructions,
  a suffix `_AVX2` is added after the version number
- If Quadriga-Lib was compiled with CUDA support and a CUDA-capable GPU is available,
  a suffix `_CUDA` is added after the version number

### Usage:
```matlab
version = quadriga_lib.version;
```

### Outputs:
- **`version`** — Version string (e.g. "0.11.5_AVX2_CUDA")

---
<a id="write_png"></a>
## write_png
Write a data matrix to a color-coded PNG file

Also in: [C++](../cpp/site-specific-simulation-tools.md#write_png) · [Python](../python/miscellaneous-tools.md#write_png)

- Values are clipped to `[min_val, max_val]` before colormap mapping; auto-detected from data if `NaN`
- The auto-detected range ignores `NaN` and `Inf`; `Inf` and `-Inf` values get the last and first colour
- `NaN` values get the first colour of the map
- A degenerate range (`max <= min` after auto-detection, e.g. constant or single-value data, all-`NaN` data, or log
  input without a positive value; also an infinite `min_val` or `max_val`) maps every value to the first colour
- Raises an error for empty `data` and for `min_val > max_val` (both given)
- Uses [LodePNG](https://github.com/lvandeve/lodepng) for PNG encoding

### Usage:
```matlab
quadriga_lib.write_png( fn, data, colormap, min_val, max_val, log_transform );
```

### Inputs:
- **`fn`** — Output `.png` file path; string
- **`data`** — Input data matrix; `[n_rows, n_cols]`
- **`colormap`** — Colormap name; supported: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer; default: jet
- **`min_val`** — Lower clipping bound; auto-detected if `NaN`; default: `NaN`
- **`max_val`** — Upper clipping bound; auto-detected if `NaN`; default: `NaN`
- **`log_transform`** — Apply 10·log10(data) before mapping; non-positive values map to the minimum color; logical; default: false
