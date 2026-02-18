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

// Multi-frequency: dict → std::vector<arrayant>
// Pattern fields (e_theta_re/im, e_phi_re/im) may be 3D (single freq) or 4D (multi-freq, 4th dim = frequency)
// center_freq: scalar (single freq) or 1D array (multi-freq)
// coupling_re/im: 2D (shared across freq) or 3D (per-freq, 3rd dim = frequency)
// Shared fields: azimuth_grid, elevation_grid, element_pos, name
static std::vector<quadriga_lib::arrayant<double>> qd_python_dict2arrayant_multi(const py::dict &arrayant,
                                                                                  bool view = false, bool strict = false)
{
    // Determine number of frequencies from e_theta_re shape
    py::array e_theta_re_arr = py::cast<py::array>(arrayant["e_theta_re"]);
    int nd = (int)e_theta_re_arr.request().ndim;
    size_t n_freq = 1;
    if (nd == 4)
        n_freq = (size_t)e_theta_re_arr.request().shape[3];
    else if (nd != 3)
        throw std::invalid_argument("'e_theta_re' must be 3D (single freq) or 4D (multi-freq).");

    // Parse pattern fields as vector<Cube>
    auto e_theta_re = qd_python_numpy2arma_vecCube<double>(arrayant["e_theta_re"], view, strict);
    auto e_theta_im = qd_python_numpy2arma_vecCube<double>(arrayant["e_theta_im"], view, strict);
    auto e_phi_re = qd_python_numpy2arma_vecCube<double>(arrayant["e_phi_re"], view, strict);
    auto e_phi_im = qd_python_numpy2arma_vecCube<double>(arrayant["e_phi_im"], view, strict);

    if (e_theta_re.size() != n_freq || e_theta_im.size() != n_freq ||
        e_phi_re.size() != n_freq || e_phi_im.size() != n_freq)
        throw std::invalid_argument("Pattern fields must all have the same number of frequency entries.");

    // Parse shared fields
    auto azimuth_grid = qd_python_numpy2arma_Col<double>(arrayant["azimuth_grid"], view, strict);
    auto elevation_grid = qd_python_numpy2arma_Col<double>(arrayant["elevation_grid"], view, strict);

    arma::Mat<double> element_pos;
    if (arrayant.contains("element_pos"))
        element_pos = qd_python_numpy2arma_Mat<double>(arrayant["element_pos"], view, strict);

    std::string name;
    if (arrayant.contains("name"))
        name = arrayant["name"].cast<std::string>();

    // Parse center_freq: scalar or 1D array
    arma::Col<double> center_freqs(n_freq);
    if (arrayant.contains("center_freq"))
    {
        py::handle cf_obj = arrayant["center_freq"];
        if (py::isinstance<py::array>(cf_obj))
        {
            center_freqs = qd_python_numpy2arma_Col<double>(cf_obj);
            if (center_freqs.n_elem != n_freq)
                throw std::invalid_argument("'center_freq' length must match number of frequency entries.");
        }
        else // scalar
        {
            double cf_val = cf_obj.cast<double>();
            center_freqs.fill(cf_val);
        }
    }
    else
        center_freqs.fill(299792458.0); // Default

    // Parse coupling: 2D (shared) or 3D (per-freq)
    std::vector<arma::Mat<double>> coupling_re_vec(n_freq);
    std::vector<arma::Mat<double>> coupling_im_vec(n_freq);
    bool has_coupling_re = arrayant.contains("coupling_re");
    bool has_coupling_im = arrayant.contains("coupling_im");

    if (has_coupling_re)
    {
        py::array cpl_arr = py::cast<py::array>(arrayant["coupling_re"]);
        int cpl_nd = (int)cpl_arr.request().ndim;
        if (cpl_nd <= 2) // Shared across all frequencies
        {
            auto cpl = qd_python_numpy2arma_Mat<double>(arrayant["coupling_re"], view, strict);
            for (size_t f = 0; f < n_freq; ++f)
                coupling_re_vec[f] = cpl;
        }
        else // 3D: (n_el, n_ports, n_freq) — slices are per-frequency coupling matrices
        {
            auto cpl_cube = qd_python_numpy2arma_Cube<double>(arrayant["coupling_re"], view, strict);
            if (cpl_cube.n_slices != n_freq)
                throw std::invalid_argument("'coupling_re' 3rd dimension must match number of frequency entries.");
            for (size_t f = 0; f < n_freq; ++f)
                coupling_re_vec[f] = cpl_cube.slice(f);
        }
    }

    if (has_coupling_im)
    {
        py::array cpl_arr = py::cast<py::array>(arrayant["coupling_im"]);
        int cpl_nd = (int)cpl_arr.request().ndim;
        if (cpl_nd <= 2)
        {
            auto cpl = qd_python_numpy2arma_Mat<double>(arrayant["coupling_im"], view, strict);
            for (size_t f = 0; f < n_freq; ++f)
                coupling_im_vec[f] = cpl;
        }
        else
        {
            auto cpl_cube = qd_python_numpy2arma_Cube<double>(arrayant["coupling_im"], view, strict);
            if (cpl_cube.n_slices != n_freq)
                throw std::invalid_argument("'coupling_im' 3rd dimension must match number of frequency entries.");
            for (size_t f = 0; f < n_freq; ++f)
                coupling_im_vec[f] = cpl_cube.slice(f);
        }
    }

    // Build the vector of arrayant
    std::vector<quadriga_lib::arrayant<double>> output(n_freq);
    for (size_t f = 0; f < n_freq; ++f)
    {
        output[f].e_theta_re = std::move(e_theta_re[f]);
        output[f].e_theta_im = std::move(e_theta_im[f]);
        output[f].e_phi_re = std::move(e_phi_re[f]);
        output[f].e_phi_im = std::move(e_phi_im[f]);
        output[f].azimuth_grid = azimuth_grid;
        output[f].elevation_grid = elevation_grid;
        output[f].element_pos = element_pos;
        output[f].center_frequency = center_freqs[f];
        output[f].name = name;
        if (has_coupling_re)
            output[f].coupling_re = std::move(coupling_re_vec[f]);
        if (has_coupling_im)
            output[f].coupling_im = std::move(coupling_im_vec[f]);
    }

    return output;
}

// Multi-frequency: std::vector<arrayant> → dict
// Pattern fields stored as 4D arrays (4th dim = frequency)
// Shared fields (grids, element_pos, name) taken from entry [0]
// center_freq stored as 1D array
// coupling stored as 3D if any entry differs, 2D if all identical
static py::dict qd_python_arrayant2dict_multi(const std::vector<quadriga_lib::arrayant<double>> &ant_vec)
{
    if (ant_vec.empty())
        return py::dict();

    // If single entry, delegate to the simple function
    if (ant_vec.size() == 1)
        return qd_python_arrayant2dict(ant_vec[0]);

    size_t n_freq = ant_vec.size();
    const auto &a0 = ant_vec[0];

    py::dict output;

    // Pattern fields → 4D numpy arrays
    // Collect into vectors, then stack
    std::vector<arma::Cube<double>> vec_etr(n_freq), vec_eti(n_freq), vec_epr(n_freq), vec_epi(n_freq);
    for (size_t f = 0; f < n_freq; ++f)
    {
        vec_etr[f] = ant_vec[f].e_theta_re;
        vec_eti[f] = ant_vec[f].e_theta_im;
        vec_epr[f] = ant_vec[f].e_phi_re;
        vec_epi[f] = ant_vec[f].e_phi_im;
    }
    output["e_theta_re"] = qd_python_copy2numpy_4d(vec_etr);
    output["e_theta_im"] = qd_python_copy2numpy_4d(vec_eti);
    output["e_phi_re"] = qd_python_copy2numpy_4d(vec_epr);
    output["e_phi_im"] = qd_python_copy2numpy_4d(vec_epi);

    // Shared fields from first entry
    output["azimuth_grid"] = qd_python_copy2numpy(a0.azimuth_grid);
    output["elevation_grid"] = qd_python_copy2numpy(a0.elevation_grid);
    output["element_pos"] = qd_python_copy2numpy(a0.element_pos);
    output["name"] = a0.name;

    // Center frequency as 1D array
    arma::Col<double> center_freqs(n_freq);
    for (size_t f = 0; f < n_freq; ++f)
        center_freqs[f] = ant_vec[f].center_frequency;
    output["center_freq"] = qd_python_copy2numpy(center_freqs);

    // Coupling: check if all entries are identical
    bool coupling_varies = false;
    for (size_t f = 1; f < n_freq && !coupling_varies; ++f)
    {
        if (ant_vec[f].coupling_re.n_elem != a0.coupling_re.n_elem ||
            ant_vec[f].coupling_im.n_elem != a0.coupling_im.n_elem)
            coupling_varies = true;
        else
        {
            if (a0.coupling_re.n_elem > 0 && !arma::approx_equal(ant_vec[f].coupling_re, a0.coupling_re, "absdiff", 1e-15))
                coupling_varies = true;
            if (a0.coupling_im.n_elem > 0 && !arma::approx_equal(ant_vec[f].coupling_im, a0.coupling_im, "absdiff", 1e-15))
                coupling_varies = true;
        }
    }

    if (coupling_varies)
    {
        // Per-frequency: store as 3D (n_el × n_ports × n_freq) using copy2numpy_4d on (n_el, n_ports, 1) cubes
        // Wrap each Mat as a single-slice Cube, then stack into 4D which looks like (n_el, n_ports, 1, n_freq)
        // Actually, simpler: stack Mats into a Cube (n_el, n_ports, n_freq) and output as 3D
        if (a0.coupling_re.n_elem > 0)
        {
            arma::uword nr = a0.coupling_re.n_rows, nc = a0.coupling_re.n_cols;
            arma::Cube<double> cpl_re(nr, nc, n_freq);
            for (size_t f = 0; f < n_freq; ++f)
                cpl_re.slice(f) = ant_vec[f].coupling_re;
            output["coupling_re"] = qd_python_copy2numpy(cpl_re);
        }
        else
            output["coupling_re"] = qd_python_copy2numpy(a0.coupling_re);

        if (a0.coupling_im.n_elem > 0)
        {
            arma::uword nr = a0.coupling_im.n_rows, nc = a0.coupling_im.n_cols;
            arma::Cube<double> cpl_im(nr, nc, n_freq);
            for (size_t f = 0; f < n_freq; ++f)
                cpl_im.slice(f) = ant_vec[f].coupling_im;
            output["coupling_im"] = qd_python_copy2numpy(cpl_im);
        }
        else
            output["coupling_im"] = qd_python_copy2numpy(a0.coupling_im);
    }
    else
    {
        // Shared: store as 2D
        output["coupling_re"] = qd_python_copy2numpy(a0.coupling_re);
        output["coupling_im"] = qd_python_copy2numpy(a0.coupling_im);
    }

    return output;
}

#endif