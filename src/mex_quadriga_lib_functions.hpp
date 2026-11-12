// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef mex_quadriga_lib_helper_H
#define mex_quadriga_lib_helper_H

#include "mex.h"
#include "mex_helper_functions.hpp"
#include "quadriga_lib.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

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
    ant.e_theta_re = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_theta_re"), copy, "e_theta_re");
    ant.e_theta_im = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_theta_im"), copy, "e_theta_im");
    ant.e_phi_re = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_phi_re"), copy, "e_phi_re");
    ant.e_phi_im = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_phi_im"), copy, "e_phi_im");
    ant.azimuth_grid = qd_mex_get_Col<double>(qd_mex_get_field(input, "azimuth_grid"), copy, "azimuth_grid");
    ant.elevation_grid = qd_mex_get_Col<double>(qd_mex_get_field(input, "elevation_grid"), copy, "elevation_grid");

    arma::uword n_elements = ant.e_theta_re.n_slices;

    // Optional fields
    if (qd_mex_has_field(input, "element_pos") && !mxIsEmpty(qd_mex_get_field(input, "element_pos")))
        ant.element_pos = qd_mex_get_Mat<double>(qd_mex_get_field(input, "element_pos"), copy, "element_pos");
    else
        ant.element_pos.zeros(3, n_elements);

    if (qd_mex_has_field(input, "coupling_re") && !mxIsEmpty(qd_mex_get_field(input, "coupling_re")))
        ant.coupling_re = qd_mex_get_Mat<double>(qd_mex_get_field(input, "coupling_re"), copy, "coupling_re");
    else if (!qd_mex_has_field(input, "coupling_im"))
        ant.coupling_re.eye(n_elements, n_elements);

    if (qd_mex_has_field(input, "coupling_im") && !mxIsEmpty(qd_mex_get_field(input, "coupling_im")))
        ant.coupling_im = qd_mex_get_Mat<double>(qd_mex_get_field(input, "coupling_im"), copy, "coupling_im");
    else
        ant.coupling_im.zeros(ant.coupling_re.n_rows, ant.coupling_re.n_cols);

    if (qd_mex_has_field(input, "center_freq") && !mxIsEmpty(qd_mex_get_field(input, "center_freq")))
        ant.center_frequency = qd_mex_get_scalar<double>(qd_mex_get_field(input, "center_freq"), "center_freq", 299792458.0);

    if (qd_mex_has_field(input, "name") && !mxIsEmpty(qd_mex_get_field(input, "name")))
        ant.name = qd_mex_get_string(qd_mex_get_field(input, "name"));

    if (validate)
    {
        auto error_msg = ant.validate();
        if (!error_msg.empty())
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", error_msg.c_str());
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
        ant[n].e_theta_re = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_theta_re", n), copy, "e_theta_re");
        ant[n].e_theta_im = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_theta_im", n), copy, "e_theta_im");
        ant[n].e_phi_re = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_phi_re", n), copy, "e_phi_re");
        ant[n].e_phi_im = qd_mex_get_Cube<double>(qd_mex_get_field(input, "e_phi_im", n), copy, "e_phi_im");
        ant[n].azimuth_grid = qd_mex_get_Col<double>(qd_mex_get_field(input, "azimuth_grid", n), copy, "azimuth_grid");
        ant[n].elevation_grid = qd_mex_get_Col<double>(qd_mex_get_field(input, "elevation_grid", n), copy, "elevation_grid");

        arma::uword n_elements = ant[n].e_theta_re.n_slices;

        if (has_center_freq && !mxIsEmpty(qd_mex_get_field(input, "center_freq", n)))
            ant[n].center_frequency = qd_mex_get_scalar<double>(qd_mex_get_field(input, "center_freq", n), "center_freq", 299792458.0);

        if (has_element_pos && !mxIsEmpty(qd_mex_get_field(input, "element_pos", n)))
            ant[n].element_pos = qd_mex_get_Mat<double>(qd_mex_get_field(input, "element_pos", n), copy, "element_pos");
        else
            ant[n].element_pos.zeros(3, n_elements);

        if (has_coupling_re && !mxIsEmpty(qd_mex_get_field(input, "coupling_re", n)))
            ant[n].coupling_re = qd_mex_get_Mat<double>(qd_mex_get_field(input, "coupling_re", n), copy, "coupling_re");
        else if (!(has_coupling_im && !mxIsEmpty(qd_mex_get_field(input, "coupling_im", n))))
            ant[n].coupling_re.eye(n_elements, n_elements);

        if (has_coupling_im && !mxIsEmpty(qd_mex_get_field(input, "coupling_im", n)))
            ant[n].coupling_im = qd_mex_get_Mat<double>(qd_mex_get_field(input, "coupling_im", n), copy, "coupling_im");
        else
            ant[n].coupling_im.zeros(ant[n].coupling_re.n_rows, ant[n].coupling_re.n_cols);

        if (has_name && !mxIsEmpty(qd_mex_get_field(input, "name", n)))
            ant[n].name = qd_mex_get_string(qd_mex_get_field(input, "name", n));
    }

    if (validate)
    {
        auto error_msg = quadriga_lib::arrayant_is_valid_multi(ant);
        if (!error_msg.empty())
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", error_msg.c_str());
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
                mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", error_msg.c_str());
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
            c.tx_pos = qd_mex_get_Mat<double>(fp, copy, "tx_position");
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "rx_position"))
            c.rx_pos = qd_mex_get_Mat<double>(fp, copy, "rx_position");
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "tx_orientation"))
            c.tx_orientation = qd_mex_get_Mat<double>(fp, copy, "tx_orientation");
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "rx_orientation"))
            c.rx_orientation = qd_mex_get_Mat<double>(fp, copy, "rx_orientation");

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "coeff_re"))
            c.coeff_re = qd_mex_matlab2vector_Cube<double>(fp, 3, "coeff_re");
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "coeff_im"))
            c.coeff_im = qd_mex_matlab2vector_Cube<double>(fp, 3, "coeff_im");
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "delay"))
            c.delay = qd_mex_matlab2vector_Cube<double>(fp, 3, "delay");

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "path_gain"))
            c.path_gain = qd_mex_matlab2vector_Col<double>(fp, 1, "path_gain");
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "path_length"))
            c.path_length = qd_mex_matlab2vector_Col<double>(fp, 1, "path_length");

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "path_polarization"))
            c.path_polarization = qd_mex_matlab2vector_Mat<double>(fp, 2, "path_polarization");
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "path_angles"))
            c.path_angles = qd_mex_matlab2vector_Mat<double>(fp, 2, "path_angles");
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "fbs_pos"))
            c.path_fbs_pos = qd_mex_matlab2vector_Mat<double>(fp, 2, "fbs_pos");
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "lbs_pos"))
            c.path_lbs_pos = qd_mex_matlab2vector_Mat<double>(fp, 2, "lbs_pos");

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "no_interact"))
            c.no_interact = qd_mex_matlab2vector_Col<unsigned>(fp, 1, "no_interact");
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "interact_coord"))
            c.interact_coord = qd_mex_matlab2vector_Mat<double>(fp, 2, "interact_coord");

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "center_frequency"))
            c.center_frequency = qd_mex_get_Col<double>(fp, copy, "center_frequency");

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
                mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", error_msg.c_str());
        }

        chan.push_back(std::move(c));
    }

    return chan;
}

// 1x1 struct of QuaDRiGa scenario parameters -> map of the numeric values
// - Every field except 'SubpathMethod' must be a real numeric or logical scalar
// - 'SubpathMethod' (a field of QuaDRiGa's structs) must be the string 'legacy'; it is accepted and ignored
// - Field names are not checked here, quadriga_lib::scenpar_validate rejects unknown names
// - Empty input ([] or a struct without fields) gives an empty map
// - 'var_name' is used in the error message for a wrong input type
inline std::unordered_map<std::string, double> qd_mex_struct2scenpar(const mxArray *input,
                                                                     const std::string &var_name = "scenpar")
{
    std::unordered_map<std::string, double> scenpar;
    if (mxIsEmpty(input))
        return scenpar;

    if (!mxIsStruct(input) || mxGetNumberOfElements(input) != 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Input '" + var_name + "' must be a 1x1 struct.").c_str());

    const int n_fields = mxGetNumberOfFields(input);
    for (int f = 0; f < n_fields; ++f)
    {
        const std::string name = mxGetFieldNameByNumber(input, f);
        const mxArray *data = mxGetFieldByNumber(input, 0, f);
        if (name == "SubpathMethod")
        {
            if (data == nullptr || !mxIsChar(data))
                mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Field 'SubpathMethod' must be a string.");
            const std::string value = qd_mex_get_string(data);
            if (value != "legacy")
                mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Value '" + value + "' of 'SubpathMethod' is not supported; only 'legacy' is accepted.").c_str());
        }
        else
        {
            if (data == nullptr || mxGetNumberOfElements(data) != 1 || mxIsComplex(data) ||
                !(mxIsNumeric(data) || mxIsLogical(data)))
                mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Field '" + name + "' must be a real numeric scalar.").c_str());
            scenpar[name] = qd_mex_get_scalar<double>(data, name);
        }
    }
    return scenpar;
}

// Map of QuaDRiGa scenario parameters -> 1x1 struct of double scalars, fields in the order of 'key_order'
// - Every name in 'key_order' must be in 'scenpar'
inline mxArray *qd_mex_scenpar2struct(const std::unordered_map<std::string, double> &scenpar,
                                      const std::vector<std::string> &key_order)
{
    mxArray *output = qd_mex_make_struct(key_order);
    for (const auto &key : key_order)
        qd_mex_set_field(output, key, qd_mex_copy2matlab(&scenpar.at(key)));
    return output;
}

// 1x1 struct of QuaDRiGa path-loss parameters -> map of the numeric values, the model and the field order
// - The field 'model' must be a string; it is returned in 'pl_model' (empty if the struct has no such field)
// - Every other field must be a real numeric vector (empty, a scalar, a row or a column) of the classes that
//   qd_mex_typecast_Col reads
// - 'pl_order' lists the numeric fields in struct order
// - Empty input ([] or a struct without fields) gives an empty map, empty 'pl_model' and empty 'pl_order'
// - 'var_name' is used in the error message for a wrong input type
inline std::unordered_map<std::string, std::vector<double>> qd_mex_struct2plpar(const mxArray *input,
                                                                               std::string &pl_model,
                                                                               std::vector<std::string> &pl_order,
                                                                               const std::string &var_name = "plpar")
{
    std::unordered_map<std::string, std::vector<double>> plpar;
    pl_model.clear();
    pl_order.clear();
    if (mxIsEmpty(input))
        return plpar;

    if (!mxIsStruct(input) || mxGetNumberOfElements(input) != 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Input '" + var_name + "' must be a 1x1 struct.").c_str());

    const int n_fields = mxGetNumberOfFields(input);
    for (int f = 0; f < n_fields; ++f)
    {
        const std::string name = mxGetFieldNameByNumber(input, f);
        const mxArray *data = mxGetFieldByNumber(input, 0, f);
        if (name == "model")
        {
            if (data == nullptr || !mxIsChar(data))
                mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Field 'model' must be a string.");
            pl_model = qd_mex_get_string(data);
        }
        else
        {
            if (data == nullptr || mxIsComplex(data) || !mxIsNumeric(data) || mxGetNumberOfDimensions(data) > 2 ||
                (mxGetM(data) > 1 && mxGetN(data) > 1))
                mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Field '" + name + "' must be a real numeric vector.").c_str());
            const arma::vec values = qd_mex_typecast_Col<double>(data, name);
            plpar[name] = std::vector<double>(values.begin(), values.end());
            pl_order.push_back(name);
        }
    }
    return plpar;
}

// QuaDRiGa path-loss parameters -> 1x1 struct, or [] if 'pl_model' and 'pl_order' are both empty
// - Field 'model' (string) first if 'pl_model' is not empty, then one row vector per key in the order of 'pl_order';
//   every name in 'pl_order' must be in 'plpar'
inline mxArray *qd_mex_plpar2struct(const std::string &pl_model,
                                    const std::unordered_map<std::string, std::vector<double>> &plpar,
                                    const std::vector<std::string> &pl_order)
{
    if (pl_model.empty() && pl_order.empty())
        return mxCreateDoubleMatrix(0, 0, mxREAL);

    std::vector<std::string> fields;
    if (!pl_model.empty())
        fields.push_back("model");
    fields.insert(fields.end(), pl_order.begin(), pl_order.end());
    mxArray *output = qd_mex_make_struct(fields);
    if (!pl_model.empty())
        qd_mex_set_field(output, "model", mxCreateString(pl_model.c_str()));
    for (const auto &key : pl_order)
    {
        const arma::rowvec values(plpar.at(key));
        qd_mex_set_field(output, key, qd_mex_copy2matlab(&values));
    }
    return output;
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

// -------------------------------- gbsm_rays struct --------------------------------

// Field names of the gbsm_rays struct in their order; the first qd_mex_gbsm_rays_n_required are required as input,
// the others optional (missing = empty), also any field appended later
inline const std::vector<std::string> qd_mex_gbsm_rays_fields = {
    "method", "tx_pos", "rx_pos", "center_frequency", "delay", "power", "aod", "eod", "aoa",
    "eoa", "M", "path_type", "tap", "gr_eps_re", "gr_eps_im", "fbs_pos", "lbs_pos", "fallback"};
inline constexpr size_t qd_mex_gbsm_rays_n_required = 13; // "method" to "tap"

// Element i of a gbsm_rays struct array -> gbsm_rays
// - Borrows the MATLAB memory (read-only) unless 'copy'; integer fields accept any real class (cast and checked)
// - Missing optional fields mean empty; a missing other field or an unknown field throws, naming 'var_name'
// - Empty arrays keep their dimensions (e.g. 'power' [0, n_freq] for n_ray = 0); validation is left to is_valid()
inline quadriga_lib::gbsm_rays qd_mex_struct2gbsm_rays(const mxArray *s, arma::uword i = 0, bool copy = false,
                                                       const std::string &var_name = "rays")
{
    if (!mxIsStruct(s) || (size_t)i >= mxGetNumberOfElements(s))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Input '" + var_name + "' must be a struct.").c_str());
    const auto &F = qd_mex_gbsm_rays_fields;
    for (int f = 0; f < mxGetNumberOfFields(s); ++f)
        if (std::find(F.begin(), F.end(), std::string(mxGetFieldNameByNumber(s, f))) == F.end())
        {
            const std::string msg = "Unknown field '" + std::string(mxGetFieldNameByNumber(s, f)) + "' in '" + var_name + "'.";
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
        }
    for (size_t k = 0; k < qd_mex_gbsm_rays_n_required; ++k)
        if (mxGetFieldNumber(s, F[k].c_str()) < 0)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Field '" + F[k] + "' is missing in '" + var_name + "'.").c_str());

    // Field data or nullptr (missing or unset)
    auto get = [&](const std::string &f) -> const mxArray *
    { return mxGetField(s, (mwIndex)i, f.c_str()); };

    quadriga_lib::gbsm_rays r;
    const mxArray *a = get("method");
    if (a != nullptr && !mxIsEmpty(a))
    {
        if (!mxIsChar(a))
        {
            const std::string msg = "Field 'method' of '" + var_name + "' must be a string.";
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
        }
        r.method = qd_mex_get_string(a);
    }
    else
        r.method.clear();
    auto col = [&](const std::string &f, auto &x)
    {
        using T = typename std::decay_t<decltype(x)>::elem_type;
        if ((a = get(f)) != nullptr && !mxIsEmpty(a))
            x = qd_mex_get_Col<T>(a, copy, var_name + "." + f);
    };
    auto mat = [&](const std::string &f, arma::mat &x)
    {
        if ((a = get(f)) != nullptr)
            x = qd_mex_get_Mat<double>(a, copy, var_name + "." + f);
    };
    col("tx_pos", r.tx_pos), col("rx_pos", r.rx_pos), col("center_frequency", r.center_frequency);
    col("delay", r.delay), col("aod", r.aod), col("eod", r.eod), col("aoa", r.aoa), col("eoa", r.eoa);
    col("path_type", r.path_type), col("tap", r.tap), col("gr_eps_re", r.gr_eps_re), col("gr_eps_im", r.gr_eps_im);
    col("fallback", r.fallback);
    mat("power", r.power), mat("fbs_pos", r.fbs_pos), mat("lbs_pos", r.lbs_pos);
    if ((a = get("M")) != nullptr)
        r.M = qd_mex_get_Cube<double>(a, copy, var_name + ".M");
    return r;
}

// Creates the 1 x n_link gbsm_rays struct array with all fields and aliases the attributes of 'rays' (resized to
// n_link) whose size is known: positions [3, 1], 'center_frequency', 'delay', 'power', angles, 'M', 'path_type',
// 'tap', 'gr_eps_re' and 'gr_eps_im' if 'has_gr', 'fbs_pos' and 'lbs_pos' [3, n_ray] if 'has_scatterers'
// - The other fields are empty ('fallback' uint32); 'method' is left unset for the caller to fill after the call
inline mxArray *qd_mex_init_gbsm_rays(arma::uword n_link, arma::uword n_ray, arma::uword n_freq, bool has_gr,
                                      bool has_scatterers, std::vector<quadriga_lib::gbsm_rays> &rays)
{
    mxArray *s = qd_mex_make_struct(qd_mex_gbsm_rays_fields, n_link);
    rays.clear();
    rays.resize(n_link);
    for (arma::uword l = 0; l < n_link; ++l)
    {
        auto &r = rays[l];
        auto set = [&](const char *f, mxArray *a)
        { mxSetField(s, (mwIndex)l, f, a); };
        auto empty = []()
        { return mxCreateNumericMatrix(0, 0, mxDOUBLE_CLASS, mxREAL); };
        set("tx_pos", qd_mex_init_output(&r.tx_pos, 3));
        set("rx_pos", qd_mex_init_output(&r.rx_pos, 3));
        set("center_frequency", qd_mex_init_output(&r.center_frequency, n_freq));
        set("delay", qd_mex_init_output(&r.delay, n_ray));
        set("power", qd_mex_init_output(&r.power, n_ray, n_freq));
        set("aod", qd_mex_init_output(&r.aod, n_ray));
        set("eod", qd_mex_init_output(&r.eod, n_ray));
        set("aoa", qd_mex_init_output(&r.aoa, n_ray));
        set("eoa", qd_mex_init_output(&r.eoa, n_ray));
        set("M", qd_mex_init_output(&r.M, 8, n_ray, n_freq));
        set("path_type", qd_mex_init_output(&r.path_type, n_ray));
        set("tap", qd_mex_init_output(&r.tap, n_ray));
        set("gr_eps_re", has_gr ? qd_mex_init_output(&r.gr_eps_re, n_freq) : empty());
        set("gr_eps_im", has_gr ? qd_mex_init_output(&r.gr_eps_im, n_freq) : empty());
        set("fbs_pos", has_scatterers ? qd_mex_init_output(&r.fbs_pos, 3, n_ray) : empty());
        set("lbs_pos", has_scatterers ? qd_mex_init_output(&r.lbs_pos, 3, n_ray) : empty());
        set("fallback", mxCreateNumericMatrix(0, 0, mxUINT32_CLASS, mxREAL));
    }
    return s;
}

#endif
