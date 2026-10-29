// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output of the test cases, switched per area by environment variables. A variable is on when it is set
// and not "0"; the output is off by default, and the assertions run either way.
//
//   QD_DIAG_SOS    sum-of-sinusoids tests (test_sos_*.cpp)
//   QD_DIAG_CONF   conf-file and scenario-parameter tests (test_conf_*.cpp, test_scenpar_validate.cpp)
//   QD_DIAG_RAY    ray-tracing tests, tags [wave] and [acoustic] (test_wave_*.cpp, test_acoustic_*.cpp)
//   QD_DIAG_GBSM   geometry-based stochastic model tests (test_gbsm_phases.cpp, test_ssf_phases.cpp,
//                  test_ssf_clusters*.cpp, test_path_loss.cpp, test_lsp_generate.cpp, ...)
//   QD_DIAG_HDF    HDF5 channel-file tests (test_qd_hdf5_internal.cpp, test_hdf_robustness_internal.cpp)
//   QD_DIAG        all areas
//
// Example: QD_DIAG_SOS=1 ./build_linux/test_bin "sos_val*"
//
// A test file selects its area once with QD_TEST_DIAG_SWITCH("QD_DIAG_SOS"). Then qd_diag() tells whether the area
// is on, and QD_DIAG(...) prints like printf only then; its arguments are evaluated only when the output is on.
// A new area adds its variable to the list above.

#ifndef QD_TEST_DIAG_HPP
#define QD_TEST_DIAG_HPP

#include <cstdio>
#include <cstdlib>

// True when the environment variable 'name' or QD_DIAG is set and not "0"
static inline bool qd_test_diag_env(const char *name)
{
    auto on = [](const char *v)
    { return v != nullptr && v[0] != '\0' && v[0] != '0'; };
    return on(std::getenv("QD_DIAG")) || on(std::getenv(name));
}

// Defines qd_diag() for the area of one test file; the environment is read once
#define QD_TEST_DIAG_SWITCH(name)                      \
    static inline bool qd_diag()                       \
    {                                                  \
        static const bool on = qd_test_diag_env(name); \
        return on;                                     \
    }

#define QD_DIAG(...) (qd_diag() ? (void)std::printf(__VA_ARGS__) : (void)0)

#endif
