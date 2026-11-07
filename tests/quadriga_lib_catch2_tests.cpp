// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_session.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <armadillo>
#include <cstdint>
#include <cstdlib>
#include <random>

// Reseeds the Armadillo random number generator before every test case and every SECTION run. A test then draws the
// same values in the full suite and when it runs alone (test_bin "<name>"), so a failure can be reproduced. The seed is
// the default seed of std::mt19937_64, the state of a fresh program; QD_TEST_SEED=<n> sets another one, e.g. to check
// that no test depends on the realisation of its random inputs.
class qd_test_seed_listener : public Catch::EventListenerBase
{
public:
  using Catch::EventListenerBase::EventListenerBase;
  void testCasePartialStarting(Catch::TestCaseInfo const &, uint64_t) override
  {
    const char *s = std::getenv("QD_TEST_SEED");
    arma::arma_rng::set_seed(s ? std::strtoull(s, nullptr, 10) : std::mt19937_64::default_seed);
  }
};
CATCH_REGISTER_LISTENER(qd_test_seed_listener)

// Main function to run CATCH2
int main(int argc, char *argv[])
{
  int result = Catch::Session().run(argc, argv);
  return result;
}
