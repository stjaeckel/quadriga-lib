#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Build the band probe against build_linux/libquadriga.a (run cmake --build build_linux first).
# usage: tools/acoustic/build_band_probe.sh <output binary>   (from the repository root)
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:?usage: build_band_probe.sh <output binary>}
CXX=${CXX:-g++}
$CXX -std=gnu++17 -O2 -fopenmp -w -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I$R/include -I$R/src \
  $R/tools/acoustic/band_probe.cpp $R/build_linux/libquadriga.a -ldl -lm -lpthread -o "$OUT"
echo "$OUT"
