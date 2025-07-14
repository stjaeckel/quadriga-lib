// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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

#ifndef quadriga_python_quadriga_adapter_H
#define quadriga_python_quadriga_adapter_H

#include "quadriga_lib.hpp"
#include "python_arma_adapter.hpp"

namespace py = pybind11;

static quadriga_lib::arrayant<double> qd_python_dict2arrayant(const py::dict &arrayant,
                                                               bool view = false, bool strict = false)
{
    auto ant = quadriga_lib::arrayant<double>();
    ant.e_theta_re = qd_python_numpy2arma_Cube<double>(arrayant["e_theta_re"], view, strict);
    ant.e_theta_im = qd_python_numpy2arma_Cube<double>(arrayant["e_theta_im"], view, strict);
    ant.e_phi_re = qd_python_numpy2arma_Cube<double>(arrayant["e_phi_re"], view, strict);
    ant.e_phi_im = qd_python_numpy2arma_Cube<double>(arrayant["e_phi_im"], view, strict);
    ant.azimuth_grid = qd_python_numpy2arma_Col<double>(arrayant["azimuth_grid"], view, strict);
    ant.elevation_grid = qd_python_numpy2arma_Col<double>(arrayant["elevation_grid"], view, strict);
    if (arrayant.contains("element_pos"))
        ant.element_pos = qd_python_numpy2arma_Mat<double>(arrayant["element_pos"], view, strict);
    if (arrayant.contains("coupling_re"))
        ant.coupling_re = qd_python_numpy2arma_Mat<double>(arrayant["coupling_re"], view, strict);
    if (arrayant.contains("coupling_im"))
        ant.coupling_im = qd_python_numpy2arma_Mat<double>(arrayant["coupling_im"], view, strict);
    if (arrayant.contains("center_freq"))
        ant.center_frequency = arrayant["center_freq"].cast<double>();
    if (arrayant.contains("name"))
        ant.name = arrayant["name"].cast<std::string>();
    return ant;
}

static py::dict qd_python_arrayant2dict(const quadriga_lib::arrayant<double> &ant)
{
    py::dict output;
    output["e_theta_re"] = qd_python_copy2numpy(ant.e_theta_re);
    output["e_theta_im"] = qd_python_copy2numpy(ant.e_theta_im);
    output["e_phi_re"] = qd_python_copy2numpy(ant.e_phi_re);
    output["e_phi_im"] = qd_python_copy2numpy(ant.e_phi_im);
    output["azimuth_grid"] = qd_python_copy2numpy(ant.azimuth_grid);
    output["elevation_grid"] = qd_python_copy2numpy(ant.elevation_grid);
    output["element_pos"] = qd_python_copy2numpy(ant.element_pos);
    output["coupling_re"] = qd_python_copy2numpy(ant.coupling_re);
    output["coupling_im"] = qd_python_copy2numpy(ant.coupling_im);
    output["center_freq"] = ant.center_frequency;
    output["name"] = ant.name;
    return output;
}

#endif