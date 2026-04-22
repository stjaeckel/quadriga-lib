// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef mex_quadriga_lib_helper_H
#define mex_quadriga_lib_helper_H

#include "mex.h"
#include "mex_helper_functions.hpp"
#include "quadriga_lib.hpp"

inline mxArray *qd_mex_arrayant2struct(const quadriga_lib::arrayant<double> &ant)
{
    std::vector<std::string> fields = {"e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im",
                                       "azimuth_grid", "elevation_grid", "element_pos",
                                       "coupling_re", "coupling_im", "center_freq", "name"};

    mxArray *output;
    output = qd_mex_make_struct(fields);
    qd_mex_set_field(output, fields[0], qd_mex_copy2matlab(&ant.e_theta_re));
    qd_mex_set_field(output, fields[1], qd_mex_copy2matlab(&ant.e_theta_im));
    qd_mex_set_field(output, fields[2], qd_mex_copy2matlab(&ant.e_phi_re));
    qd_mex_set_field(output, fields[3], qd_mex_copy2matlab(&ant.e_phi_im));
    qd_mex_set_field(output, fields[4], qd_mex_copy2matlab(&ant.azimuth_grid, true));
    qd_mex_set_field(output, fields[5], qd_mex_copy2matlab(&ant.elevation_grid, true));
    qd_mex_set_field(output, fields[6], qd_mex_copy2matlab(&ant.element_pos));
    qd_mex_set_field(output, fields[7], qd_mex_copy2matlab(&ant.coupling_re));
    qd_mex_set_field(output, fields[8], qd_mex_copy2matlab(&ant.coupling_im));
    qd_mex_set_field(output, fields[9], qd_mex_copy2matlab(&ant.center_frequency));
    qd_mex_set_field(output, fields[10], mxCreateString(ant.name.c_str()));
    return output;
}

inline mxArray *qd_mex_arrayant2struct_multi(const std::vector<quadriga_lib::arrayant<double>> &ant)
{
    std::vector<std::string> fields = {"e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im",
                                       "azimuth_grid", "elevation_grid", "element_pos",
                                       "coupling_re", "coupling_im", "center_freq", "name"};

    mxArray *output;
    output = qd_mex_make_struct(fields, ant.size());
    for (size_t n = 0; n < ant.size(); ++n)
    {
        qd_mex_set_field(output, fields[0], qd_mex_copy2matlab(&ant[n].e_theta_re), n);
        qd_mex_set_field(output, fields[1], qd_mex_copy2matlab(&ant[n].e_theta_im), n);
        qd_mex_set_field(output, fields[2], qd_mex_copy2matlab(&ant[n].e_phi_re), n);
        qd_mex_set_field(output, fields[3], qd_mex_copy2matlab(&ant[n].e_phi_im), n);
        qd_mex_set_field(output, fields[4], qd_mex_copy2matlab(&ant[n].azimuth_grid, true), n);
        qd_mex_set_field(output, fields[5], qd_mex_copy2matlab(&ant[n].elevation_grid, true), n);
        qd_mex_set_field(output, fields[6], qd_mex_copy2matlab(&ant[n].element_pos), n);
        qd_mex_set_field(output, fields[7], qd_mex_copy2matlab(&ant[n].coupling_re), n);
        qd_mex_set_field(output, fields[8], qd_mex_copy2matlab(&ant[n].coupling_im), n);
        qd_mex_set_field(output, fields[9], qd_mex_copy2matlab(&ant[n].center_frequency), n);
        qd_mex_set_field(output, fields[10], mxCreateString(ant[n].name.c_str()), n);
    }
    return output;
}

inline quadriga_lib::arrayant<double> qd_mex_struct2arrayant(const mxArray *input, bool validate = false, bool copy = false)
{
    if (!mxIsStruct(input))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input must be a struct.");

    quadriga_lib::arrayant<double> ant;

    // Mandatory fields
    ant.e_theta_re = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_theta_re"), copy);
    ant.e_theta_im = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_theta_im"), copy);
    ant.e_phi_re = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_phi_re"), copy);
    ant.e_phi_im = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_phi_im"), copy);
    ant.azimuth_grid = qd_mex_get_Col<double>(qd_mex_get_field(input, "azimuth_grid"), copy);
    ant.elevation_grid = qd_mex_get_Col<double>(qd_mex_get_field(input, "elevation_grid"), copy);

    arma::uword n_elements = ant.e_theta_re.n_slices;

    // Optional fields
    if (qd_mex_has_field(input, "element_pos") && !mxIsEmpty(qd_mex_get_field(input, "element_pos")))
        ant.element_pos = qd_mex_get_Mat<double>(qd_mex_get_field(input, "element_pos"), copy);
    else
        ant.element_pos.zeros(3, n_elements);

    if (qd_mex_has_field(input, "coupling_re") && !mxIsEmpty(qd_mex_get_field(input, "coupling_re")))
        ant.coupling_re = qd_mex_get_Mat<double>(qd_mex_get_field(input, "coupling_re"), copy);
    else
        ant.coupling_re.eye(n_elements, n_elements);

    if (qd_mex_has_field(input, "coupling_im") && !mxIsEmpty(qd_mex_get_field(input, "coupling_im")))
        ant.coupling_im = qd_mex_get_Mat<double>(qd_mex_get_field(input, "coupling_im"), copy);
    else
        ant.coupling_im.zeros(n_elements, n_elements);

    if (qd_mex_has_field(input, "center_freq") && !mxIsEmpty(qd_mex_get_field(input, "center_freq")))
        ant.center_frequency = qd_mex_get_scalar<double>(qd_mex_get_field(input, "center_freq"), "center_freq", 299792458.0);

    if (qd_mex_has_field(input, "name") && !mxIsEmpty(qd_mex_get_field(input, "name")))
        ant.name = qd_mex_get_string(qd_mex_get_field(input, "name"));

    if (validate)
    {
        auto error_msg = ant.validate();
        if (!error_msg.empty())
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", error_msg.c_str());
    }

    return ant;
}

inline std::vector<quadriga_lib::arrayant<double>> qd_mex_struct2arrayant_multi(const mxArray *input, bool validate = false, bool copy = false)
{
    if (!mxIsStruct(input))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input must be a struct.");

    // Check for optional fields
    bool has_element_pos = qd_mex_has_field(input, "element_pos");
    bool has_coupling_re = qd_mex_has_field(input, "coupling_re");
    bool has_coupling_im = qd_mex_has_field(input, "coupling_im");
    bool has_name = qd_mex_has_field(input, "name");

    size_t n_elem = (size_t)mxGetNumberOfElements(input);
    std::vector<quadriga_lib::arrayant<double>> ant(n_elem);
    for (size_t n = 0; n < n_elem; ++n)
    {
        ant[n].e_theta_re = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_theta_re", n), copy);
        ant[n].e_theta_im = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_theta_im", n), copy);
        ant[n].e_phi_re = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_phi_re", n), copy);
        ant[n].e_phi_im = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_phi_im", n), copy);
        ant[n].azimuth_grid = qd_mex_get_Col<double>(qd_mex_get_field(input, "azimuth_grid", n), copy);
        ant[n].elevation_grid = qd_mex_get_Col<double>(qd_mex_get_field(input, "elevation_grid", n), copy);
        ant[n].center_frequency = qd_mex_get_scalar<double>(qd_mex_get_field(input, "center_freq", n), "center_freq", 299792458.0);

        arma::uword n_elements = ant[n].e_theta_re.n_slices;

        if (has_element_pos && !mxIsEmpty(qd_mex_get_field(input, "element_pos", n)))
            ant[n].element_pos = qd_mex_get_Mat<double>(qd_mex_get_field(input, "element_pos", n), copy);
        else
            ant[n].element_pos.zeros(3, n_elements);

        if (has_coupling_re && !mxIsEmpty(qd_mex_get_field(input, "coupling_re", n)))
            ant[n].coupling_re = qd_mex_get_Mat<double>(qd_mex_get_field(input, "coupling_re", n), copy);
        else
            ant[n].coupling_re.eye(n_elements, n_elements);

        if (has_coupling_im && !mxIsEmpty(qd_mex_get_field(input, "coupling_im", n)))
            ant[n].coupling_im = qd_mex_get_Mat<double>(qd_mex_get_field(input, "coupling_im", n), copy);
        else
            ant[n].coupling_im.zeros(n_elements, n_elements);

        if (has_name && !mxIsEmpty(qd_mex_get_field(input, "name", n)))
            ant[n].name = qd_mex_get_string(qd_mex_get_field(input, "name", n));
    }
    return ant;
}

#endif