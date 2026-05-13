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

    mxArray *output = qd_mex_make_struct(fields);
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

    mxArray *output = qd_mex_make_struct(fields, ant.size());
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
    else if (!qd_mex_has_field(input, "coupling_im"))
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
    bool has_center_freq = qd_mex_has_field(input, "center_freq");

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

        arma::uword n_elements = ant[n].e_theta_re.n_slices;

        if (has_center_freq && !mxIsEmpty(qd_mex_get_field(input, "center_freq", n)))
            ant[n].center_frequency = qd_mex_get_scalar<double>(qd_mex_get_field(input, "center_freq", n), "center_freq", 299792458.0);

        if (has_element_pos && !mxIsEmpty(qd_mex_get_field(input, "element_pos", n)))
            ant[n].element_pos = qd_mex_get_Mat<double>(qd_mex_get_field(input, "element_pos", n), copy);
        else
            ant[n].element_pos.zeros(3, n_elements);

        if (has_coupling_re && !mxIsEmpty(qd_mex_get_field(input, "coupling_re", n)))
            ant[n].coupling_re = qd_mex_get_Mat<double>(qd_mex_get_field(input, "coupling_re", n), copy);
        else if (!(has_coupling_im && !mxIsEmpty(qd_mex_get_field(input, "coupling_im", n))))
            ant[n].coupling_re.eye(n_elements, n_elements);

        if (has_coupling_im && !mxIsEmpty(qd_mex_get_field(input, "coupling_im", n)))
            ant[n].coupling_im = qd_mex_get_Mat<double>(qd_mex_get_field(input, "coupling_im", n), copy);
        else
            ant[n].coupling_im.zeros(n_elements, n_elements);

        if (has_name && !mxIsEmpty(qd_mex_get_field(input, "name", n)))
            ant[n].name = qd_mex_get_string(qd_mex_get_field(input, "name", n));
    }

    if (validate)
    {
        auto error_msg = quadriga_lib::arrayant_is_valid_multi(ant, false);
        if (!error_msg.empty())
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", error_msg.c_str());
    }

    return ant;
}

inline mxArray *qd_mex_channel2struct(const std::vector<quadriga_lib::channel<double>> &chan, bool validate = false)
{
    mxArray *output = nullptr;

    const size_t n_chan = chan.size();
    if (n_chan == 0) // Return empty struct with no fields
    {
        std::vector<std::string> empty_fields;
        return qd_mex_make_struct(empty_fields, 0);
    }

    // Determine which fields are populated (based on chan[0])
    const auto &c0 = chan[0];
    const bool h_txp = !c0.tx_pos.is_empty();
    const bool h_rxp = !c0.rx_pos.is_empty();
    const bool h_txo = !c0.tx_orientation.is_empty();
    const bool h_rxo = !c0.rx_orientation.is_empty();
    const bool h_cre = !c0.coeff_re.empty();
    const bool h_cim = !c0.coeff_im.empty();
    const bool h_del = !c0.delay.empty();
    const bool h_pg = !c0.path_gain.empty();
    const bool h_pl = !c0.path_length.empty();
    const bool h_pp = !c0.path_polarization.empty();
    const bool h_pa = !c0.path_angles.empty();
    const bool h_fbs = !c0.path_fbs_pos.empty();
    const bool h_lbs = !c0.path_lbs_pos.empty();
    const bool h_ni = !c0.no_interact.empty();
    const bool h_ic = !c0.interact_coord.empty();
    const bool h_cf = !c0.center_frequency.is_empty();
    const bool h_ini = c0.initial_position != 0;

    // Build the field list (name is always present)
    std::vector<std::string> fields;
    fields.push_back("name");
    if (h_txp)
        fields.push_back("tx_position");
    if (h_rxp)
        fields.push_back("rx_position");
    if (h_txo)
        fields.push_back("tx_orientation");
    if (h_rxo)
        fields.push_back("rx_orientation");
    if (h_cre)
        fields.push_back("coeff_re");
    if (h_cim)
        fields.push_back("coeff_im");
    if (h_del)
        fields.push_back("delay");
    if (h_pg)
        fields.push_back("path_gain");
    if (h_pl)
        fields.push_back("path_length");
    if (h_pp)
        fields.push_back("path_polarization");
    if (h_pa)
        fields.push_back("path_angles");
    if (h_fbs)
        fields.push_back("fbs_pos");
    if (h_lbs)
        fields.push_back("lbs_pos");
    if (h_ni)
        fields.push_back("no_interact");
    if (h_ic)
        fields.push_back("interact_coord");
    if (h_cf)
        fields.push_back("center_frequency");
    if (h_ini)
        fields.push_back("initial_position");

    output = qd_mex_make_struct(fields, n_chan);

    for (size_t n = 0; n < n_chan; ++n)
    {
        const auto &c = chan[n];

        if (validate)
        {
            auto error_msg = c.is_valid();
            if (!error_msg.empty())
                mexErrMsgIdAndTxt("quadriga_lib:CPPerror", error_msg.c_str());
        }

        qd_mex_set_field(output, "name", mxCreateString(c.name.c_str()), n);
        if (h_txp)
            qd_mex_set_field(output, "tx_position", qd_mex_copy2matlab(&c.tx_pos), n);
        if (h_rxp)
            qd_mex_set_field(output, "rx_position", qd_mex_copy2matlab(&c.rx_pos), n);
        if (h_txo)
            qd_mex_set_field(output, "tx_orientation", qd_mex_copy2matlab(&c.tx_orientation), n);
        if (h_rxo)
            qd_mex_set_field(output, "rx_orientation", qd_mex_copy2matlab(&c.rx_orientation), n);
        if (h_cre)
            qd_mex_set_field(output, "coeff_re", qd_mex_vector2matlab(&c.coeff_re), n);
        if (h_cim)
            qd_mex_set_field(output, "coeff_im", qd_mex_vector2matlab(&c.coeff_im), n);
        if (h_del)
            qd_mex_set_field(output, "delay", qd_mex_vector2matlab(&c.delay), n);
        if (h_pg)
            qd_mex_set_field(output, "path_gain", qd_mex_vector2matlab(&c.path_gain), n);
        if (h_pl)
            qd_mex_set_field(output, "path_length", qd_mex_vector2matlab(&c.path_length), n);
        if (h_pp)
            qd_mex_set_field(output, "path_polarization", qd_mex_vector2matlab(&c.path_polarization), n);
        if (h_pa)
            qd_mex_set_field(output, "path_angles", qd_mex_vector2matlab(&c.path_angles), n);
        if (h_fbs)
            qd_mex_set_field(output, "fbs_pos", qd_mex_vector2matlab(&c.path_fbs_pos), n);
        if (h_lbs)
            qd_mex_set_field(output, "lbs_pos", qd_mex_vector2matlab(&c.path_lbs_pos), n);
        if (h_ni)
            qd_mex_set_field(output, "no_interact", qd_mex_vector2matlab(&c.no_interact), n);
        if (h_ic)
            qd_mex_set_field(output, "interact_coord", qd_mex_vector2matlab(&c.interact_coord), n);
        if (h_cf)
            qd_mex_set_field(output, "center_frequency", qd_mex_copy2matlab(&c.center_frequency), n);
        if (h_ini)
            qd_mex_set_field(output, "initial_position", qd_mex_copy2matlab(&c.initial_position), n);
    }
    return output;
}

inline std::vector<quadriga_lib::channel<double>> qd_mex_struct2channel(const mxArray *input, bool validate = false, bool copy = false)
{
    if (!mxIsStruct(input))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input must be a struct.");

    const size_t n_chan = (size_t)mxGetNumberOfElements(input);
    std::vector<quadriga_lib::channel<double>> chan;
    if (n_chan == 0)
        return chan;

    chan.reserve(n_chan);

    for (size_t i = 0; i < n_chan; ++i)
    {
        quadriga_lib::channel<double> c;

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "name"))
            c.name = qd_mex_get_string(fp);

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "tx_position"))
            c.tx_pos = qd_mex_get_Mat<double>(fp, copy);
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "rx_position"))
            c.rx_pos = qd_mex_get_Mat<double>(fp, copy);
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "tx_orientation"))
            c.tx_orientation = qd_mex_get_Mat<double>(fp, copy);
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "rx_orientation"))
            c.rx_orientation = qd_mex_get_Mat<double>(fp, copy);

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "coeff_re"))
            c.coeff_re = qd_mex_matlab2vector_Cube<double>(fp, 3);
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "coeff_im"))
            c.coeff_im = qd_mex_matlab2vector_Cube<double>(fp, 3);
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "delay"))
            c.delay = qd_mex_matlab2vector_Cube<double>(fp, 3);

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "path_gain"))
            c.path_gain = qd_mex_matlab2vector_Col<double>(fp, 1);
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "path_length"))
            c.path_length = qd_mex_matlab2vector_Col<double>(fp, 1);

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "path_polarization"))
            c.path_polarization = qd_mex_matlab2vector_Mat<double>(fp, 2);
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "path_angles"))
            c.path_angles = qd_mex_matlab2vector_Mat<double>(fp, 2);
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "fbs_pos"))
            c.path_fbs_pos = qd_mex_matlab2vector_Mat<double>(fp, 2);
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "lbs_pos"))
            c.path_lbs_pos = qd_mex_matlab2vector_Mat<double>(fp, 2);

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "no_interact"))
            c.no_interact = qd_mex_matlab2vector_Col<unsigned>(fp, 1);
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "interact_coord"))
            c.interact_coord = qd_mex_matlab2vector_Mat<double>(fp, 2);

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "center_frequency"))
            c.center_frequency = qd_mex_get_Col<double>(fp, copy);

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "initial_position"))
            c.initial_position = qd_mex_get_scalar<int>(fp, "initial_position", 0);

        // Prune zero-padded trailing columns in interact_coord (round-trip artifacts)
        const size_t n_snap = (size_t)c.n_snap();
        if (c.no_interact.size() == n_snap && c.interact_coord.size() == n_snap)
            for (size_t s = 0; s < n_snap; ++s)
            {
                unsigned cnt = 0;
                for (auto v : c.no_interact[s])
                    cnt += v;
                if (c.interact_coord[s].n_cols > (arma::uword)cnt)
                    c.interact_coord[s].resize(c.interact_coord[s].n_rows, (arma::uword)cnt);
            }

        if (validate)
        {
            auto error_msg = c.is_valid();
            if (!error_msg.empty())
                mexErrMsgIdAndTxt("quadriga_lib:CPPerror", error_msg.c_str());
        }

        chan.push_back(std::move(c));
    }

    return chan;
}

inline mxArray *qd_mex_any2matlab(const std::any &dset)
{
    mxArray *output = nullptr;

    // Missing dataset -> return []
    if (!dset.has_value())
        return mxCreateNumericMatrix(0, 0, mxDOUBLE_CLASS, mxREAL);

    // Dispatch on the std::any contents
    unsigned long long dims[3];
    void *dataptr = nullptr;
    int type_id = quadriga_lib::any_type_id(&dset, dims, &dataptr);

#define CASE_ARMA(id, ARMA_TYPE)                    \
    case id:                                        \
    {                                               \
        auto data = std::any_cast<ARMA_TYPE>(dset); \
        output = qd_mex_copy2matlab(&data);         \
        break;                                      \
    }

    switch (type_id)
    {
    case 9:
    { // std::string
        auto data = std::any_cast<std::string>(dset);
        output = mxCreateString(data.c_str());
        break;
    }

    // Scalars
    case 10:
        output = qd_mex_copy2matlab((float *)dataptr);
        break;
    case 11:
        output = qd_mex_copy2matlab((double *)dataptr);
        break;
    case 12:
        output = qd_mex_copy2matlab((unsigned long long *)dataptr);
        break;
    case 13:
        output = qd_mex_copy2matlab((long long *)dataptr);
        break;
    case 14:
        output = qd_mex_copy2matlab((unsigned *)dataptr);
        break;
    case 15:
        output = qd_mex_copy2matlab((int *)dataptr);
        break;

        // Matrices
        CASE_ARMA(20, arma::Mat<float>)
        CASE_ARMA(21, arma::Mat<double>)
        CASE_ARMA(22, arma::Mat<unsigned long long>)
        CASE_ARMA(23, arma::Mat<long long>)
        CASE_ARMA(24, arma::Mat<unsigned>)
        CASE_ARMA(25, arma::Mat<int>)

        // Cubes
        CASE_ARMA(30, arma::Cube<float>)
        CASE_ARMA(31, arma::Cube<double>)
        CASE_ARMA(32, arma::Cube<unsigned long long>)
        CASE_ARMA(33, arma::Cube<long long>)
        CASE_ARMA(34, arma::Cube<unsigned>)
        CASE_ARMA(35, arma::Cube<int>)

        // Column vectors
        CASE_ARMA(40, arma::Col<float>)
        CASE_ARMA(41, arma::Col<double>)
        CASE_ARMA(42, arma::Col<unsigned long long>)
        CASE_ARMA(43, arma::Col<long long>)
        CASE_ARMA(44, arma::Col<unsigned>)
        CASE_ARMA(45, arma::Col<int>)

        // Row vectors
        CASE_ARMA(50, arma::Row<float>)
        CASE_ARMA(51, arma::Row<double>)
        CASE_ARMA(52, arma::Row<unsigned long long>)
        CASE_ARMA(53, arma::Row<long long>)
        CASE_ARMA(54, arma::Row<unsigned>)
        CASE_ARMA(55, arma::Row<int>)

    default:
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Unsupported dataset type.");
    }

#undef CASE_ARMA

    return output;
}

#endif