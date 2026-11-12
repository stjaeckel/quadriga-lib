// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "quadriga_lib.hpp"

TEST_CASE("Channel - Effective Path Gain")
{
    quadriga_lib::channel<double> c;
    c.tx_pos = arma::mat(3, 1);
    c.tx_pos(2, 0) = 1.0;

    c.rx_pos = arma::mat(3, 2);
    c.rx_pos(0, 0) = 20.0;
    c.rx_pos(1, 0) = 0.0;
    c.rx_pos(2, 0) = 1.0;
    c.rx_pos(0, 1) = 20.0;
    c.rx_pos(1, 1) = 1.0;
    c.rx_pos(2, 1) = 1.0;

    // Random polarization
    c.path_polarization.push_back(arma::mat(8, 1, arma::fill::randn));
    c.path_polarization.push_back(arma::mat(8, 1, arma::fill::randn));

    // Calc effective PG and check results
    auto p = c.calc_effective_path_gain();
    arma::vec q1 = c.path_polarization[0] % c.path_polarization[0];
    arma::vec q2 = c.path_polarization[1] % c.path_polarization[1];
    CHECK(std::abs(arma::sum(q1) - p.at(0)) < 1e-14);
    CHECK(std::abs(arma::sum(q2) - p.at(1)) < 1e-14);

    // Add a path length
    arma::Col<double> l = {1.0};
    c.path_length.push_back(l);
    l = {2.0};
    c.path_length.push_back(l);

    // Check additional gain
    p = c.calc_effective_path_gain();
    double a1 = std::pow(10.0, -3.245);
    double a2 = std::pow(10.0, 0.1 * (-32.45 - 20.0 * std::log10(2.0)));
    CHECK(std::abs(a1 * arma::sum(q1) - p.at(0)) < 1e-14);
    CHECK(std::abs(a2 * arma::sum(q2) - p.at(1)) < 1e-14);

    // Set same frequency for all snapshots
    c.center_frequency = 10.0e9;

    // Check additional gain
    p = c.calc_effective_path_gain();
    a1 = std::pow(10.0, 0.1 * (-32.45 - 20.0 * std::log10(10.0)));
    a2 = std::pow(10.0, 0.1 * (-32.45 - 20.0 * std::log10(10.0) - 20.0 * std::log10(2.0)));
    CHECK(std::abs(a1 * arma::sum(q1) - p.at(0)) < 1e-14 * p.at(0));
    CHECK(std::abs(a2 * arma::sum(q2) - p.at(1)) < 1e-14 * p.at(1));

    // Sett different frequencies for snapshots
    c.center_frequency = {12.0e9, 20.0e9};

    // Check gain
    p = c.calc_effective_path_gain();
    a1 = std::pow(10.0, 0.1 * (-32.45 - 20.0 * std::log10(12.0)));
    a2 = std::pow(10.0, 0.1 * (-32.45 - 20.0 * std::log10(20.0) - 20.0 * std::log10(2.0)));
    CHECK(std::abs(a1 * arma::sum(q1) - p.at(0)) < 1e-14 * p.at(0));
    CHECK(std::abs(a2 * arma::sum(q2) - p.at(1)) < 1e-14 * p.at(1));

    // Set a path gain, this disables FSPL calculation
    l = {0.1};
    c.path_gain.push_back(l);
    l = {0.2};
    c.path_gain.push_back(l);

    // Check gain
    p = c.calc_effective_path_gain();
    CHECK(std::abs(0.1 * arma::sum(q1) - p.at(0)) < 1e-14);
    CHECK(std::abs(0.2 * arma::sum(q2) - p.at(1)) < 1e-14);

    // Calculate coefficients
    arma::Cube<double> coeff_re, coeff_im, delay;
    auto ant = quadriga_lib::generate_arrayant_xpol<double>();

    l = {0.0}; // Angles
    quadriga_lib::get_channels_planar(&ant, &ant,
                                      0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                      c.rx_pos(0, 0), c.rx_pos(1, 0), c.rx_pos(2, 0),
                                      0.0, 0.0, 0.0, &l, &l, &l, &l,
                                      &c.path_gain[0], &c.path_length[0], &c.path_polarization[0],
                                      &coeff_re, &coeff_im, &delay);

    c.coeff_re.push_back(coeff_re);
    c.coeff_im.push_back(coeff_im);
    c.delay.push_back(delay);

    l = {0.1}; // Angles
    quadriga_lib::get_channels_planar(&ant, &ant,
                                      0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                      c.rx_pos(0, 1), c.rx_pos(1,1), c.rx_pos(2, 1),
                                      0.0, 0.0, 0.0, &l, &l, &l, &l,
                                      &c.path_gain[1], &c.path_length[1], &c.path_polarization[1],
                                      &coeff_re, &coeff_im, &delay);

    c.coeff_re.push_back(coeff_re);
    c.coeff_im.push_back(coeff_im);
    c.delay.push_back(delay);

    c.path_polarization.clear();
    c.path_length.clear();
    p.reset();

    p = c.calc_effective_path_gain();
    CHECK(std::abs(0.1 * arma::sum(q1) - p.at(0)) < 1e-14);
    CHECK(std::abs(0.2 * arma::sum(q2) - p.at(1)) < 1e-14);

    CHECK(c.is_valid().empty());
}
TEST_CASE("Channel - Interaction count sum does not wrap at 32 bits")
{
    // The two counts add up to 2^32 + 6; a 32-bit sum would wrap to 6 and accept 6 columns
    const arma::u32_vec no_interact_huge = {0xFFFFFFFFu, 7u};
    const arma::mat coord_6(3, 6, arma::fill::ones);

    quadriga_lib::channel<double> c;
    c.rx_pos = arma::mat(3, 1, arma::fill::zeros);
    c.tx_pos = arma::mat(3, 1, arma::fill::ones);
    c.coeff_re.push_back(arma::cube(1, 1, 2, arma::fill::ones));
    c.coeff_im.push_back(arma::cube(1, 1, 2, arma::fill::zeros));
    c.delay.push_back(arma::cube(1, 1, 2, arma::fill::zeros));
    c.no_interact.push_back(arma::u32_vec{1u, 5u});
    c.interact_coord.push_back(coord_6);
    REQUIRE(c.is_valid().empty());

    // is_valid must report the mismatch
    auto c_huge = c;
    c_huge.no_interact[0] = no_interact_huge;
    CHECK(!c_huge.is_valid().empty());

    // add_paths must reject the counts before anything is appended
    const arma::cube cre(1, 1, 2, arma::fill::ones), cim(1, 1, 2, arma::fill::zeros), dl(1, 1, 2, arma::fill::zeros);
    CHECK_THROWS_AS(c.add_paths(0, &cre, &cim, &dl, &no_interact_huge, &coord_6), std::invalid_argument);
    CHECK(c.n_path().at(0) == 2);
}

TEST_CASE("Channel - add_paths with inputs that are fields of the channel")
{
    quadriga_lib::channel<double> c;
    c.rx_pos = arma::mat(3, 1, arma::fill::zeros);
    c.tx_pos = arma::mat(3, 1, arma::fill::ones);
    c.coeff_re.push_back(arma::cube(1, 1, 2, arma::fill::ones));
    c.coeff_im.push_back(arma::cube(1, 1, 2, arma::fill::zeros));
    c.delay.push_back(arma::cube(1, 1, 2, arma::fill::zeros));
    c.path_gain.push_back(arma::vec(2, arma::fill::ones));
    c.path_length.push_back(arma::vec(2, arma::fill::ones));
    c.path_fbs_pos.push_back(arma::mat(3, 2, arma::fill::ones));
    c.path_lbs_pos.push_back(arma::mat(3, 2, arma::fill::zeros));
    REQUIRE(c.is_valid().empty());

    const arma::cube cre(1, 1, 2, arma::fill::ones), cim(1, 1, 2, arma::fill::zeros), dl(1, 1, 2, arma::fill::zeros);
    const arma::vec pg(2, arma::fill::ones), pl(2, arma::fill::ones);
    const arma::mat fbs(3, 2, arma::fill::ones), lbs(3, 2, arma::fill::zeros);

    auto d = c;
    CHECK_NOTHROW(d.add_paths(0, &cre, &cim, &dl, nullptr, nullptr, &pg, &pl, nullptr, nullptr, &fbs, &lbs));
    CHECK(d.n_path().at(0) == 4);

    // An input is the field it is appended to
    d = c;
    CHECK_THROWS_AS(d.add_paths(0, &d.coeff_re[0], &cim, &dl, nullptr, nullptr, &pg, &pl, nullptr, nullptr, &fbs, &lbs), std::invalid_argument);

    // An input is another field of the snapshot
    d = c;
    CHECK_THROWS_AS(d.add_paths(0, &cre, &cim, &dl, nullptr, nullptr, &pg, &pl, nullptr, nullptr, &fbs, &d.path_fbs_pos[0]), std::invalid_argument);
    d = c;
    CHECK_THROWS_AS(d.add_paths(0, &cre, &cim, &dl, nullptr, nullptr, &d.path_length[0], &pl, nullptr, nullptr, &fbs, &lbs), std::invalid_argument);

    // An input uses the memory of a field
    d = c;
    const arma::mat fbs_view(d.path_fbs_pos[0].memptr(), 3, 2, false, true);
    CHECK_THROWS_AS(d.add_paths(0, &cre, &cim, &dl, nullptr, nullptr, &pg, &pl, nullptr, nullptr, &fbs, &fbs_view), std::invalid_argument);
    CHECK(d.n_path().at(0) == 2);
}

TEST_CASE("Channel - Delays without coefficients")
{
    const std::string msg = "'delay' must be empty or match the number of snapshots.";

    // Three snapshots (moving TX, fixed RX), one path each, delays but no coefficients
    quadriga_lib::channel<double> c;
    c.tx_pos = arma::mat(3, 3, arma::fill::zeros);
    c.rx_pos = arma::mat(3, 1, arma::fill::ones);
    for (int s = 0; s < 3; ++s)
        c.path_gain.push_back(arma::vec(1, arma::fill::ones)), c.delay.push_back(arma::cube(1, 1, 1, arma::fill::zeros));
    REQUIRE(c.is_valid().empty());

    // Delays for the first snapshot only
    auto d = c;
    d.delay = {arma::cube(1, 1, 1, arma::fill::zeros)};
    CHECK(d.is_valid() == msg);

    // The vector is shortened but keeps its capacity, so the dropped entries still look valid
    d = c;
    d.delay.resize(1);
    CHECK(d.is_valid() == msg);

    // Delays are the only per-path member: the path counts come from the delays
    quadriga_lib::channel<double> e;
    e.tx_pos = arma::mat(3, 1, arma::fill::zeros);
    e.rx_pos = arma::mat(3, 2, arma::fill::ones);
    e.delay.push_back(arma::cube(1, 1, 3, arma::fill::zeros));
    e.delay.push_back(arma::cube(1, 1, 2, arma::fill::zeros));
    CHECK(e.is_valid().empty());
    const arma::uvec n_path = e.n_path();
    REQUIRE(n_path.n_elem == 2);
    CHECK(n_path.at(0) == 3);
    CHECK(n_path.at(1) == 2);

    // Interaction coordinates without interaction counts, also when nothing else sets the snapshot count
    quadriga_lib::channel<double> f;
    f.interact_coord.push_back(arma::mat(3, 1, arma::fill::zeros));
    CHECK(f.is_valid() == "'interact_coord' is provided but 'no_interact' is missing.");
    f.tx_pos = arma::mat(3, 1, arma::fill::zeros);
    f.rx_pos = arma::mat(3, 1, arma::fill::ones);
    CHECK(f.is_valid() == "'interact_coord' is provided but 'no_interact' is missing.");
}
