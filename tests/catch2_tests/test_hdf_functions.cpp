// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <iostream>
#include <string>
#include <any>

TEST_CASE("HDF - Minimal test")
{
    // Get typenames of armadillo types, different for each compiler
    std::any x;
    x = arma::Mat<double>();
    const auto type_arma_Mat_double = x.type().name();
    x = arma::Mat<unsigned>();
    const auto type_arma_Mat_unsigned = x.type().name();

    std::cout << type_arma_Mat_double << ", " << type_arma_Mat_unsigned << std::endl;

    // Create a vector of dynamic armadillo types
    std::vector<std::any> v;
    for (unsigned n = 0; n < 5; n++)
    {
        auto w = arma::Mat<double>(1, 5);
        w.at(0) = n;
        v.push_back(w);
    }

    auto w = arma::Mat<unsigned>(1, 3);
    w.at(1) = 1999;
    v.push_back(w);

    // Return values, depending on their type
    for (long unsigned i = 0; i < v.size(); i++)
    {
        if (v[i].type().name() == type_arma_Mat_double) // Returns reference
        {
            auto *w = std::any_cast<arma::Mat<double>>(&v[i]);
            (*w).print();
        }

        if (v[i].type().name() == type_arma_Mat_unsigned) // returns copy
            std::any_cast<arma::Mat<unsigned>>(v[i]).print();
    }

    std::cout << v.capacity() << std::endl;

    // std::any w = arma::Mat<unsigned>(5, 5);

    // std::any_cast<arma::Mat<unsigned>>(w).print();

    // std::cout << w.type().name() << std::endl;


    auto c = quadriga_lib::channel<float>();
    std::cout << c.name << ", " << c.version << std::endl;

    quadriga_lib::qd_channel_hello();

    quadriga_lib::print_lib_versions(); 
    
}