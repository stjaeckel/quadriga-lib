# Miscellaneous / Tools

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-miscellaneous-tools

| Function | Summary |
| --- | --- |
| [quadriga_lib_has_AVX2](#quadriga_lib_has_AVX2) | Check whether AVX2 kernels can be used |
| [quadriga_lib_has_CUDA](#quadriga_lib_has_CUDA) | Check whether CUDA kernels can be used |
| [quadriga_lib_version](#quadriga_lib_version) | Version number of quadriga-lib |

---
<a id="quadriga_lib_has_AVX2"></a>
## quadriga_lib_has_AVX2
Check whether AVX2 kernels can be used

Also in: [MATLAB](../mex/miscellaneous-tools.md#version) · [Python](../python/miscellaneous-tools.md#version)

- True if the library was built with AVX2 support and the CPU runs the AVX2 test kernel correctly
- The test kernel runs once; later calls return the stored result

### Declaration:
```cpp
bool quadriga_lib::quadriga_lib_has_AVX2();
```

### Returns:
- `true` if AVX2 kernels can be used, else `false`

---
<a id="quadriga_lib_has_CUDA"></a>
## quadriga_lib_has_CUDA
Check whether CUDA kernels can be used

Also in: [MATLAB](../mex/miscellaneous-tools.md#version) · [Python](../python/miscellaneous-tools.md#version)

- True if the library was built with CUDA support and a CUDA-capable GPU runs the test kernel correctly

### Declaration:
```cpp
bool quadriga_lib::quadriga_lib_has_CUDA();
```

### Returns:
- `true` if CUDA kernels can be used, else `false`

---
<a id="quadriga_lib_version"></a>
## quadriga_lib_version
Version number of quadriga-lib

Also in: [MATLAB](../mex/miscellaneous-tools.md#version) · [Python](../python/miscellaneous-tools.md#version)

### Declaration:
```cpp
std::string quadriga_lib::quadriga_lib_version();
```

### Returns:
- Version number in the format `x.y.z`

### See also:
- [quadriga_lib_has_AVX2](#quadriga_lib_has_AVX2) (AVX2 support of library and CPU)
- [quadriga_lib_has_CUDA](#quadriga_lib_has_CUDA) (CUDA support of library and GPU)
