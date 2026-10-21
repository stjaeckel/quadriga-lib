// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output of the test cases: QD_DIAG prints only when the environment variable QD_TEST_DIAG is set and
// not "0", e.g. QD_TEST_DIAG=1 ./build_linux/test_bin "sos_val*". The assertions run either way, and the arguments
// of QD_DIAG are evaluated only when the output is on.

#ifndef QD_TEST_DIAG_HPP
#define QD_TEST_DIAG_HPP

#include <cstdio>
#include <cstdlib>

static inline bool qd_test_diag()
{
    static const bool on = []
    {
        const char *v = std::getenv("QD_TEST_DIAG");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

#define QD_DIAG(...) (qd_test_diag() ? (void)std::printf(__VA_ARGS__) : (void)0)

#endif
