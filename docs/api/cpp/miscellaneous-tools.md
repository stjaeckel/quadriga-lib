# Miscellaneous / Tools

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-miscellaneous-tools

| Function | Summary |
| --- | --- |
| [any_type_id](#any_type_id) | Get the type ID and the shape of the value in a `std::any` object |
| [quadriga_lib_has_AVX2](#quadriga_lib_has_AVX2) | Check whether AVX2 kernels can be used |
| [quadriga_lib_has_CUDA](#quadriga_lib_has_CUDA) | Check whether CUDA kernels can be used |
| [quadriga_lib_version](#quadriga_lib_version) | Version number of quadriga-lib |

---
<a id="any_type_id"></a>
## any_type_id
Get the type ID and the shape of the value in a `std::any` object

Not in: MATLAB, Python

- Tells which of the supported types a `std::any` holds, e.g. an unstructured dataset of [hdf5_read_dset](channel-data-and-processing.md#hdf5_read_dset) or an
  entry of `par_data` of a [channel](channel-data-and-processing.md#channel), so that `std::any_cast` to that type gives the value
- The ID of a numeric value is $10 m + k$ with the container $m$ and the element type $k$:
  | $m$ | Container | $k$ | Element type |
  | --- | --------- | --- | ------------ |
  | 1 | scalar | 0 | `float` |
  | 2 | `arma::Mat` | 1 | `double` |
  | 3 | `arma::Cube` | 2 | `unsigned long long` (`arma::uword`) |
  | 4 | `arma::Col` | 3 | `long long` (`arma::sword`) |
  | 5 | `arma::Row` | 4 | `unsigned` |
  | | | 5 | `int` |

  e.g. 21 is `arma::Mat<double>`; 9 is `std::string`, -1 a type not listed, -2 no value

### Declaration:
```cpp
int quadriga_lib::any_type_id(
    const std::any *data,
    unsigned long long *dims = nullptr);
```

### Input:
- **`data`** — Object to inspect; `nullptr` gives -2

### Output:
- **`dims`** *(optional)* — Shape `[rows, cols, slices]`: `[1, 1, 1]` for a scalar or a type not listed,
  `[n_elem, 1, 1]` for `arma::Col` and `arma::Row`, `[length, 0, 0]` for a string; not changed for -2; 3 values

### Returns:
- Type ID

### See also:
- [hdf5_read_dset](channel-data-and-processing.md#hdf5_read_dset) (returns such objects)
- [hdf5_write_dset](channel-data-and-processing.md#hdf5_write_dset) (stores them)

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
