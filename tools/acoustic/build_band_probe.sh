#!/bin/bash
# Build the band probe against build_linux/libquadriga.a (run cmake --build build_linux first).
# usage: tools/acoustic/build_band_probe.sh <output binary>   (from the repository root)
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:?usage: build_band_probe.sh <output binary>}
CXX=${CXX:-g++}
$CXX -std=gnu++17 -O2 -fopenmp -w -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I$R/include -I$R/src -I/usr/include/hdf5/serial \
  $R/tools/acoustic/band_probe.cpp $R/build_linux/libquadriga.a \
  -Wl,-rpath,/usr/lib/x86_64-linux-gnu/hdf5/serial /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so \
  /usr/lib/x86_64-linux-gnu/libcrypto.so /usr/lib/x86_64-linux-gnu/libcurl.so /usr/lib/x86_64-linux-gnu/libsz.so \
  /usr/lib/x86_64-linux-gnu/libz.so -ldl -lm -lpthread -o "$OUT"
echo "$OUT"
