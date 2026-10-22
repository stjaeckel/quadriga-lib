# Prototypes and benchmarks from the review of the sum-of-sinusoids functions

Reference material for the fix plan of the sum-of-sinusoids functions, written during the review of
2026-10-12. The review and the plan are in the git history (`git show b7c1009:md/REVIEW_QD_SOS.md`,
section 3; `git show daafa6a^:md/PLAN_QD_SOS_FIXES.md`). Nothing here is built by CMake or used by the
library.

| File | Content | Review |
|---|---|---|
| `proto_avx2_tiled.cpp` | AVX2 kernel with weights packed per group of 8 generators, 8 named accumulators and position tiles (`var_SOS_AVX2_T8`). Pads the generators to a multiple of 8, which costs 10 % for one generator; a real implementation needs narrower paths for the remainder | P3 |
| `proto_avx2_named_acc.cpp` | Present kernel with named accumulators instead of an array, 4 or 8 generators per group (`var_SOS_AVX2_H4`, `_H8`) | P3a |
| `proto_generic_grouped.cpp` | GENERIC kernel that accumulates 4 generators per pass (`var_SOS_GENERIC`) | P4 |
| `bench_variants.cpp` | Times a prototype and compares its output bit by bit with the library kernel | 3.6 |
| `bench_kernels.cpp` | Kernels and `sos_val` in its device modes | 3.2, 3.3 |
| `bench_sos_val_parts.cpp` | Where `sos_val` spends its time outside the kernel | 3.3 |
| `bench_functions.cpp` | The other functions: preset, phases, rand, map, remove_bias, calc_mse, acf_estimate, generate | 3.4 |

All prototypes gave output bit-identical to `qd_SOS_AVX2` / `qd_SOS_GENERIC` of commit 38d871a.

Build, with `LIB` pointing to a build of the library with AVX2 (gcc 12, as `make` uses):

```
INC="-DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -DBUILD_WITH_AVX2=1 -I../../../include -I../../../src -I/usr/include/hdf5/serial"
LNK="$LIB/libquadriga.a /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so -lcurl -lcrypto -lz -lsz -lgomp -lpthread -ldl -lm"
g++-12 -std=gnu++17 -O3 -fopenmp -mavx2 -mfma $INC -c proto_avx2_tiled.cpp proto_avx2_named_acc.cpp
g++-12 -std=gnu++17 -O3 -fopenmp $INC -c proto_generic_grouped.cpp
g++-12 -std=gnu++17 -O3 -fopenmp $INC bench_variants.cpp proto_*.o -o bench_variants $LNK
g++-12 -std=gnu++17 -O3 -fopenmp $INC bench_kernels.cpp -o bench_kernels $LNK
g++-12 -std=gnu++17 -O3 -fopenmp $INC bench_sos_val_parts.cpp -o bench_sos_val_parts $LNK
g++-12 -std=gnu++17 -O3 -fopenmp $INC bench_functions.cpp -o bench_functions $LNK
```

Run one process per thread count, e.g. `OMP_NUM_THREADS=1 ./bench_variants at8 400 256 20000`. Write the
binaries to a scratch directory, not into the repository. Check `uptime` first; the machine is shared.
