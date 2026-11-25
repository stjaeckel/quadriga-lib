// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef qd_helpers_mex_types_hpp
#define qd_helpers_mex_types_hpp

#include "mex.h"
#include "qd_helpers_mex.hpp"
#include "qd_helpers_wrappers.hpp"
#include "quadriga_lib.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

inline mxArray *qd_mex_arrayant2struct(const quadriga_lib::arrayant<double> &ant)
{
    std::vector<std::string> fields = {"e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im",
                                       "azimuth_grid", "elevation_grid", "element_pos",
                                       "port_weights_re", "port_weights_im", "center_freq", "name"};

    mxArray *output = qd_mex_make_struct(fields);
    qd_mex_set_field(output, fields[0], qd_mex_copy2matlab(&ant.e_theta_re));
    qd_mex_set_field(output, fields[1], qd_mex_copy2matlab(&ant.e_theta_im));
    qd_mex_set_field(output, fields[2], qd_mex_copy2matlab(&ant.e_phi_re));
    qd_mex_set_field(output, fields[3], qd_mex_copy2matlab(&ant.e_phi_im));
    qd_mex_set_field(output, fields[4], qd_mex_copy2matlab(&ant.azimuth_grid, true));
    qd_mex_set_field(output, fields[5], qd_mex_copy2matlab(&ant.elevation_grid, true));
    qd_mex_set_field(output, fields[6], qd_mex_copy2matlab(&ant.element_pos));
    qd_mex_set_field(output, fields[7], qd_mex_copy2matlab(&ant.port_weights_re));
    qd_mex_set_field(output, fields[8], qd_mex_copy2matlab(&ant.port_weights_im));
    qd_mex_set_field(output, fields[9], qd_mex_copy2matlab(&ant.center_freq));
    qd_mex_set_field(output, fields[10], mxCreateString(ant.name.c_str()));
    return output;
}

inline mxArray *qd_mex_arrayant2struct_multi(const std::vector<quadriga_lib::arrayant<double>> &ant)
{
    std::vector<std::string> fields = {"e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im",
                                       "azimuth_grid", "elevation_grid", "element_pos",
                                       "port_weights_re", "port_weights_im", "center_freq", "name"};

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
        qd_mex_set_field(output, fields[7], qd_mex_copy2matlab(&ant[n].port_weights_re), n);
        qd_mex_set_field(output, fields[8], qd_mex_copy2matlab(&ant[n].port_weights_im), n);
        qd_mex_set_field(output, fields[9], qd_mex_copy2matlab(&ant[n].center_freq), n);
        qd_mex_set_field(output, fields[10], mxCreateString(ant[n].name.c_str()), n);
    }
    return output;
}

// Reads element n of an arrayant struct (array); errors name the fields 'var_name.field', or 'var_name(n).field' for a
// struct array (n 1-based)
inline quadriga_lib::arrayant<double> qd_mex_struct2arrayant_element(const mxArray *input, size_t n, bool copy,
                                                                     const std::string &var_name)
{
    std::string prefix = var_name; // Names in errors
    if (!prefix.empty())
        prefix += (mxGetNumberOfElements(input) > 1 ? "(" + std::to_string(n + 1) + ")." : std::string("."));
    auto fld = [&](const char *f) { return prefix + f; };
    auto get = [&](const char *f) { return qd_mex_get_field(input, f, n, var_name); }; // Mandatory field
    auto opt = [&](const char *f) -> const mxArray *                                   // Optional field, nullptr if missing or empty
    {
        const mxArray *a = mxGetField(input, (mwIndex)n, f);
        return (a == nullptr || mxIsEmpty(a)) ? nullptr : a;
    };

    // The matrix of element-to-port weights was called coupling_re / coupling_im; ignoring the old fields would drop the weights
    for (const char *suffix : {"re", "im"})
        if (mxGetField(input, (mwIndex)n, (std::string("coupling_") + suffix).c_str()) != nullptr)
        {
            const std::string msg = "Field '" + fld((std::string("coupling_") + suffix).c_str()) + "' was renamed to 'port_weights_" + suffix + "'.";
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
        }

    quadriga_lib::arrayant<double> ant;
    ant.e_theta_re = qd_mex_get_Cube<double>(get("e_theta_re"), copy, fld("e_theta_re"));
    ant.e_theta_im = qd_mex_get_Cube<double>(get("e_theta_im"), copy, fld("e_theta_im"));
    ant.e_phi_re = qd_mex_get_Cube<double>(get("e_phi_re"), copy, fld("e_phi_re"));
    ant.e_phi_im = qd_mex_get_Cube<double>(get("e_phi_im"), copy, fld("e_phi_im"));
    ant.azimuth_grid = qd_mex_get_Col<double>(get("azimuth_grid"), copy, fld("azimuth_grid"));
    ant.elevation_grid = qd_mex_get_Col<double>(get("elevation_grid"), copy, fld("elevation_grid"));
    if (const mxArray *a = opt("element_pos"))
        ant.element_pos = qd_mex_get_Mat<double>(a, copy, fld("element_pos"));
    if (const mxArray *a = opt("port_weights_re"))
        ant.port_weights_re = qd_mex_get_Mat<double>(a, copy, fld("port_weights_re"));
    if (const mxArray *a = opt("port_weights_im"))
        ant.port_weights_im = qd_mex_get_Mat<double>(a, copy, fld("port_weights_im"));
    qd_arrayant_fill_defaults(ant);

    if (const mxArray *a = opt("center_freq"))
        ant.center_freq = qd_mex_get_scalar<double>(a, fld("center_freq"), 299792458.0);
    if (const mxArray *a = opt("name"))
        ant.name = qd_mex_get_string(a, "", fld("name"));
    return ant;
}

// Arrayant struct -> arrayant; reads the first element of a struct array
inline quadriga_lib::arrayant<double> qd_mex_struct2arrayant(const mxArray *input, bool validate = false, bool copy = false,
                                                             const std::string &var_name = "")
{
    qd_mex_check_struct(input, var_name);
    auto ant = qd_mex_struct2arrayant_element(input, 0, copy, var_name);
    if (validate)
    {
        auto error_msg = ant.is_valid();
        if (!error_msg.empty())
            qd_mex_cpp_error(error_msg);
    }
    return ant;
}

// Arrayant struct array -> one arrayant per element (frequency)
inline std::vector<quadriga_lib::arrayant<double>> qd_mex_struct2arrayant_multi(const mxArray *input, bool validate = false, bool copy = false,
                                                                                const std::string &var_name = "")
{
    qd_mex_check_struct(input, var_name);
    std::vector<quadriga_lib::arrayant<double>> ant;
    for (size_t n = 0; n < (size_t)mxGetNumberOfElements(input); ++n)
        ant.push_back(qd_mex_struct2arrayant_element(input, n, copy, var_name));
    if (validate)
    {
        auto error_msg = quadriga_lib::is_valid(ant);
        if (!error_msg.empty())
            qd_mex_cpp_error(error_msg);
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

    // A field exists when any channel fills it; channels without that member get an empty field
    bool h_txp = false, h_rxp = false, h_txo = false, h_rxo = false, h_cre = false, h_cim = false,
         h_del = false, h_pg = false, h_pl = false, h_pp = false, h_pa = false, h_fbs = false,
         h_lbs = false, h_ni = false, h_ic = false, h_cf = false, h_ini = false;
    for (const auto &c : chan)
    {
        h_txp = h_txp || !c.tx_pos.is_empty();
        h_rxp = h_rxp || !c.rx_pos.is_empty();
        h_txo = h_txo || !c.tx_orientation.is_empty();
        h_rxo = h_rxo || !c.rx_orientation.is_empty();
        h_cre = h_cre || !c.coeff_re.empty();
        h_cim = h_cim || !c.coeff_im.empty();
        h_del = h_del || !c.delay.empty();
        h_pg = h_pg || !c.path_gain.empty();
        h_pl = h_pl || !c.path_length.empty();
        h_pp = h_pp || !c.polmat.empty();
        h_pa = h_pa || !c.path_angles.empty();
        h_fbs = h_fbs || !c.fbs_pos.empty();
        h_lbs = h_lbs || !c.lbs_pos.empty();
        h_ni = h_ni || !c.n_interact.empty();
        h_ic = h_ic || !c.interact_coord.empty();
        h_cf = h_cf || !c.center_freq.is_empty();
        h_ini = h_ini || c.initial_position != 0;
    }

    // Build the field list (name is always present)
    std::vector<std::string> fields;
    fields.push_back("name");
    if (h_txp)
        fields.push_back("tx_pos");
    if (h_rxp)
        fields.push_back("rx_pos");
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
        fields.push_back("polmat");
    if (h_pa)
        fields.push_back("path_angles");
    if (h_fbs)
        fields.push_back("fbs_pos");
    if (h_lbs)
        fields.push_back("lbs_pos");
    if (h_ni)
        fields.push_back("n_interact");
    if (h_ic)
        fields.push_back("interact_coord");
    if (h_cf)
        fields.push_back("center_freq");
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
                qd_mex_cpp_error(error_msg);
        }

        qd_mex_set_field(output, "name", mxCreateString(c.name.c_str()), n);
        if (h_txp)
            qd_mex_set_field(output, "tx_pos", qd_mex_copy2matlab(&c.tx_pos), n);
        if (h_rxp)
            qd_mex_set_field(output, "rx_pos", qd_mex_copy2matlab(&c.rx_pos), n);
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
            qd_mex_set_field(output, "polmat", qd_mex_vector2matlab(&c.polmat), n);
        if (h_pa)
            qd_mex_set_field(output, "path_angles", qd_mex_vector2matlab(&c.path_angles), n);
        if (h_fbs)
            qd_mex_set_field(output, "fbs_pos", qd_mex_vector2matlab(&c.fbs_pos), n);
        if (h_lbs)
            qd_mex_set_field(output, "lbs_pos", qd_mex_vector2matlab(&c.lbs_pos), n);
        if (h_ni)
            qd_mex_set_field(output, "n_interact", qd_mex_vector2matlab(&c.n_interact), n);
        if (h_ic)
            qd_mex_set_field(output, "interact_coord", qd_mex_vector2matlab(&c.interact_coord), n);
        if (h_cf)
            qd_mex_set_field(output, "center_freq", qd_mex_copy2matlab(&c.center_freq), n);
        if (h_ini)
        {
            const int initial_position = c.initial_position + 1; // 0-based in C++, 1-based in MATLAB
            qd_mex_set_field(output, "initial_position", qd_mex_copy2matlab(&initial_position), n);
        }
    }
    return output;
}

inline std::vector<quadriga_lib::channel<double>> qd_mex_struct2channel(const mxArray *input, bool validate = false, bool copy = false,
                                                                       const std::string &var_name = "")
{
    qd_mex_check_struct(input, var_name);
    auto fld = [&](const char *f) { return var_name.empty() ? std::string(f) : var_name + "." + f; }; // Names in errors

    const size_t n_chan = (size_t)mxGetNumberOfElements(input);
    std::vector<quadriga_lib::channel<double>> chan;
    if (n_chan == 0)
        return chan;

    chan.reserve(n_chan);

    for (size_t i = 0; i < n_chan; ++i)
    {
        quadriga_lib::channel<double> c;

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "name"))
            c.name = qd_mex_get_string(fp, "", fld("name"));

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "tx_pos"))
            c.tx_pos = qd_mex_get_Mat<double>(fp, copy, fld("tx_pos"));
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "rx_pos"))
            c.rx_pos = qd_mex_get_Mat<double>(fp, copy, fld("rx_pos"));
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "tx_orientation"))
            c.tx_orientation = qd_mex_get_Mat<double>(fp, copy, fld("tx_orientation"));
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "rx_orientation"))
            c.rx_orientation = qd_mex_get_Mat<double>(fp, copy, fld("rx_orientation"));

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "coeff_re"))
            c.coeff_re = qd_mex_matlab2vector_Cube<double>(fp, 3, fld("coeff_re"));
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "coeff_im"))
            c.coeff_im = qd_mex_matlab2vector_Cube<double>(fp, 3, fld("coeff_im"));
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "delay"))
            c.delay = qd_mex_matlab2vector_Cube<double>(fp, 3, fld("delay"));

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "path_gain"))
            c.path_gain = qd_mex_matlab2vector_Col<double>(fp, 1, fld("path_gain"));
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "path_length"))
            c.path_length = qd_mex_matlab2vector_Col<double>(fp, 1, fld("path_length"));

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "polmat"))
            c.polmat = qd_mex_matlab2vector_Mat<double>(fp, 2, fld("polmat"));
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "path_angles"))
            c.path_angles = qd_mex_matlab2vector_Mat<double>(fp, 2, fld("path_angles"));
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "fbs_pos"))
            c.fbs_pos = qd_mex_matlab2vector_Mat<double>(fp, 2, fld("fbs_pos"));
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "lbs_pos"))
            c.lbs_pos = qd_mex_matlab2vector_Mat<double>(fp, 2, fld("lbs_pos"));

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "n_interact"))
            c.n_interact = qd_mex_matlab2vector_Col<unsigned>(fp, 1, fld("n_interact"));
        if (mxArray *fp = mxGetField(input, (mwIndex)i, "interact_coord"))
            c.interact_coord = qd_mex_matlab2vector_Mat<double>(fp, 2, fld("interact_coord"));

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "center_freq"))
            c.center_freq = qd_mex_get_Col<double>(fp, copy, fld("center_freq"));

        if (mxArray *fp = mxGetField(input, (mwIndex)i, "initial_position"))
        {
            c.initial_position = qd_mex_get_index<int>(fp, fld("initial_position")); // 1-based -> 0-based
        }

        // Prune zero-padded trailing columns in interact_coord (round-trip artifacts)
        const size_t n_snap = (size_t)c.n_snap();
        if (c.n_interact.size() == n_snap && c.interact_coord.size() == n_snap)
            for (size_t s = 0; s < n_snap; ++s)
            {
                arma::uword cnt = 0; // 64-bit sum, a 32-bit sum can wrap
                for (auto v : c.n_interact[s])
                    cnt += (arma::uword)v;
                if (c.interact_coord[s].n_cols > (arma::uword)cnt)
                    c.interact_coord[s].resize(c.interact_coord[s].n_rows, (arma::uword)cnt);
            }

        if (validate)
        {
            auto error_msg = c.is_valid();
            if (!error_msg.empty())
                qd_mex_cpp_error(error_msg);
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

    qd_mex_check_struct(input, var_name, true);

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

    qd_mex_check_struct(input, var_name, true);

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

    int type_id = quadriga_lib::any_type_id(&dset);

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
        output = qd_mex_copy2matlab(std::any_cast<float>(&dset));
        break;
    case 11:
        output = qd_mex_copy2matlab(std::any_cast<double>(&dset));
        break;
    case 12:
        output = qd_mex_copy2matlab(std::any_cast<unsigned long long>(&dset));
        break;
    case 13:
        output = qd_mex_copy2matlab(std::any_cast<long long>(&dset));
        break;
    case 14:
        output = qd_mex_copy2matlab(std::any_cast<unsigned>(&dset));
        break;
    case 15:
        output = qd_mex_copy2matlab(std::any_cast<int>(&dset));
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
    "method", "tx_pos", "rx_pos", "center_freq", "delay", "power", "aod", "eod", "aoa",
    "eoa", "polmat", "path_type", "tap", "gr_eps_re", "gr_eps_im", "fbs_pos", "lbs_pos", "fallback"};
inline constexpr size_t qd_mex_gbsm_rays_n_required = 13; // "method" to "tap"

// Element 'n' of a gbsm_rays struct array -> gbsm_rays; the rules of qd_mex_struct2gbsm_rays apply (the struct array
// is not required to be 1x1)
inline quadriga_lib::gbsm_rays qd_mex_struct2gbsm_rays_at(const mxArray *s, size_t n, const std::string &var_name = "rays")
{
    qd_mex_check_struct(s, var_name);
    if (n >= mxGetNumberOfElements(s))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Struct index " + std::to_string(n + 1) + " in '" + var_name + "' out of range.").c_str());
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
    { return mxGetField(s, (mwIndex)n, f.c_str()); };

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
            x = qd_mex_get_Col<T>(a, false, var_name + "." + f);
    };
    auto mat = [&](const std::string &f, arma::mat &x)
    {
        if ((a = get(f)) != nullptr)
            x = qd_mex_get_Mat<double>(a, false, var_name + "." + f);
    };
    col("tx_pos", r.tx_pos), col("rx_pos", r.rx_pos), col("center_freq", r.center_freq);
    col("delay", r.delay), col("aod", r.aod), col("eod", r.eod), col("aoa", r.aoa), col("eoa", r.eoa);
    col("path_type", r.path_type), col("tap", r.tap), col("gr_eps_re", r.gr_eps_re), col("gr_eps_im", r.gr_eps_im);
    col("fallback", r.fallback);
    mat("power", r.power), mat("fbs_pos", r.fbs_pos), mat("lbs_pos", r.lbs_pos);
    if ((a = get("polmat")) != nullptr)
        r.polmat = qd_mex_get_Cube<double>(a, false, var_name + ".polmat");
    return r;
}

// 1x1 gbsm_rays struct -> gbsm_rays
// - Borrows the MATLAB memory (read-only); integer fields accept any real class (cast and checked)
// - Missing optional fields mean empty; a missing other field or an unknown field throws, naming 'var_name'
// - Empty arrays keep their dimensions (e.g. 'power' [0, n_freq] for n_ray = 0); validation is left to is_valid()
inline quadriga_lib::gbsm_rays qd_mex_struct2gbsm_rays(const mxArray *s, const std::string &var_name = "rays")
{
    qd_mex_check_struct(s, var_name, true);
    return qd_mex_struct2gbsm_rays_at(s, 0, var_name);
}

// Creates the 1 x n_link gbsm_rays struct array with all fields and aliases the attributes of 'rays' (resized to
// n_link) whose size is known: positions [3, 1], 'center_freq', 'delay', 'power', angles, 'polmat', 'path_type',
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
        set("tx_pos", qd_mex_init_output(&r.tx_pos, 3, 1));
        set("rx_pos", qd_mex_init_output(&r.rx_pos, 3, 1));
        set("center_freq", qd_mex_init_output(&r.center_freq, n_freq));
        set("delay", qd_mex_init_output(&r.delay, n_ray));
        set("power", qd_mex_init_output(&r.power, n_ray, n_freq));
        set("aod", qd_mex_init_output(&r.aod, n_ray));
        set("eod", qd_mex_init_output(&r.eod, n_ray));
        set("aoa", qd_mex_init_output(&r.aoa, n_ray));
        set("eoa", qd_mex_init_output(&r.eoa, n_ray));
        set("polmat", qd_mex_init_output(&r.polmat, 8, n_ray, n_freq));
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

// -------------------------------- Shipped conf folder --------------------------------

// Folder of the shipped conf files, found next to the package folder that 'which' reports, so that a moved
// installation finds it: +quadriga_lib/../config (installed), +quadriga_lib/../data/config (source tree)
inline std::string qd_mex_conf_dir()
{
    mxArray *name = mxCreateString("quadriga_lib.conf_file_list"), *path = nullptr;
    std::string file;
    if (mexCallMATLAB(1, &path, 1, &name, "which") == 0 && path != nullptr)
    {
        if (char *s = mxArrayToString(path))
            file = s, mxFree(s);
        mxDestroyArray(path);
    }
    mxDestroyArray(name);
    return qd_conf_shipped_dir(file, {"../config", "../data/config"});
}

// -------------------------------- Layout structs --------------------------------

// Folders of user conf files ('input' may be omitted) followed by the shipped folder
inline std::vector<std::string> qd_mex_layout_config_dirs(const mxArray *input)
{
    std::vector<std::string> dirs = input == nullptr ? std::vector<std::string>() : qd_mex_get_strings(input, "config_dirs");
    const std::string shipped = qd_mex_conf_dir();
    if (!shipped.empty())
        dirs.push_back(shipped);
    return dirs;
}

// Optional vector input k of a call; omitted = empty
inline arma::vec qd_mex_layout_vec(int nrhs, const mxArray *prhs[], int k, const char *name)
{
    return (nrhs <= k) ? arma::vec() : qd_mex_get_Col<double>(prhs[k], false, name);
}

// -------------------------------- Reading and writing helpers --------------------------------

// Field of a 1x1 struct, nullptr if it is missing or empty
inline const mxArray *qd_mex_layout_get(const mxArray *s, const char *f)
{
    const mxArray *a = mxGetField(s, 0, f);
    return (a == nullptr || mxIsEmpty(a)) ? nullptr : a;
}

// A 1x1 struct whose fields are all in 'known'; the error names an unknown field
inline void qd_mex_layout_check_fields(const mxArray *s, const std::vector<std::string> &known, const std::string &var_name)
{
    qd_mex_check_struct(s, var_name, true);
    for (int f = 0; f < mxGetNumberOfFields(s); ++f)
    {
        const std::string name = mxGetFieldNameByNumber(s, f);
        if (std::find(known.begin(), known.end(), name) == known.end())
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Unknown field '" + name + "' in '" + var_name + "'.").c_str());
    }
}

// Name-value struct -> map; 'conv(data, name)' converts the value of a field; [] = empty map
template <typename V, typename F>
inline std::unordered_map<std::string, V> qd_mex_layout_map(const mxArray *input, const std::string &var_name, F &&conv)
{
    std::unordered_map<std::string, V> map;
    if (input == nullptr || mxIsEmpty(input))
        return map;
    qd_mex_check_struct(input, var_name, true);
    for (int f = 0; f < mxGetNumberOfFields(input); ++f)
    {
        const std::string key = mxGetFieldNameByNumber(input, f);
        if (const mxArray *data = mxGetFieldByNumber(input, 0, f))
            map[key] = conv(data, var_name + "." + key);
    }
    return map;
}

// Options of generate_layout, drop_terminals and generate_track: one vector of values per field
inline std::unordered_map<std::string, std::vector<double>> qd_mex_layout_options(const mxArray *input, const std::string &var_name)
{
    return qd_mex_layout_map<std::vector<double>>(input, var_name, [](const mxArray *data, const std::string &name)
                                                  { const arma::vec v = qd_mex_get_Col<double>(data, false, name); return std::vector<double>(v.begin(), v.end()); });
}

// Arguments of generate_antenna in set_antenna: one matrix per field
inline std::unordered_map<std::string, arma::mat> qd_mex_layout_args(const mxArray *input, const std::string &var_name)
{
    return qd_mex_layout_map<arma::mat>(input, var_name, [](const mxArray *data, const std::string &name)
                                        { return qd_mex_get_Mat<double>(data, false, name); });
}

// A member of a layout class is read from and written to the struct field of the same name. Kinds: 0 = copy (any numeric
// class on input), 1 = index list (1-based in the struct, 0-based in the class), 2 = handled by hand. A missing or
// empty field leaves the member empty.
template <typename T>
inline void qd_mex_layout_rd(const mxArray *s, const char *f, arma::Col<T> &x, const std::string &vn, std::integral_constant<int, 0>)
{
    if (const mxArray *a = qd_mex_layout_get(s, f))
        x = qd_mex_get_Col<T>(a, true, vn + "." + f);
}
inline void qd_mex_layout_rd(const mxArray *s, const char *f, arma::mat &x, const std::string &vn, std::integral_constant<int, 0>)
{
    if (const mxArray *a = qd_mex_layout_get(s, f))
        x = qd_mex_get_Mat<double>(a, true, vn + "." + f);
}
inline void qd_mex_layout_rd(const mxArray *s, const char *f, std::vector<std::string> &x, const std::string &vn, std::integral_constant<int, 0>)
{
    if (const mxArray *a = qd_mex_layout_get(s, f))
        x = qd_mex_get_strings(a, vn + "." + f);
}
inline void qd_mex_layout_rd(const mxArray *s, const char *f, std::string &x, const std::string &vn, std::integral_constant<int, 0>)
{
    if (const mxArray *a = qd_mex_layout_get(s, f))
        x = qd_mex_get_string(a, "", vn + "." + f);
}
template <typename T>
inline void qd_mex_layout_rd(const mxArray *s, const char *f, arma::Col<T> &x, const std::string &vn, std::integral_constant<int, 1>)
{
    if (const mxArray *a = qd_mex_layout_get(s, f))
        x = qd_mex_get_index_Col<T>(a, vn + "." + f);
}
template <typename T>
inline void qd_mex_layout_rd(const mxArray *, const char *, T &, const std::string &, std::integral_constant<int, 2>) {}

template <typename T>
inline mxArray *qd_mex_layout_wr(const T &x) { return qd_mex_copy2matlab(&x); }
inline mxArray *qd_mex_layout_wr(const std::string &x) { return mxCreateString(x.c_str()); }
template <typename T>
inline void qd_mex_layout_put(mxArray *s, const char *f, const T &x, std::integral_constant<int, 0>) { qd_mex_set_field(s, f, qd_mex_layout_wr(x)); }
template <typename T>
inline void qd_mex_layout_put(mxArray *s, const char *f, const T &x, std::integral_constant<int, 1>) { qd_mex_set_field(s, f, qd_mex_layout_wr<T>(T(x + 1))); } // 0-based -> 1-based
template <typename T>
inline void qd_mex_layout_put(mxArray *, const char *, const T &, std::integral_constant<int, 2>) {}

// Members in the order of the C++ classes: X(member, kind)
#define QD_ML_NODES(X)                                                                                                    \
    X(name, 0) X(sector, 0) X(i_antenna, 0) X(offset, 1) X(pos, 0) X(orientation, 0) X(direction, 0) X(speed, 0) X(time, 0) \
    X(scenario, 0) X(indoor, 0) X(floor, 0) X(d2d_in, 0) X(clutter, 0) X(scenpar_override, 2) X(seed, 0) X(relative, 0)     \
    X(cell, 0) X(extra_names, 0) X(extra, 2)
#define QD_ML_LINKS(X)                                                                                                    \
    X(i_tx, 1) X(i_rx, 1) X(i_segment, 1) X(image, 0) X(i_conf, 1) X(conf_names, 0) X(los, 0) X(indoor, 0) X(floor_id, 0)   \
    X(d2d_in, 0) X(o2i_rnd, 0) X(o2i_model, 1) X(gbsm_seed, 0) X(seed, 2) X(method, 0) X(sos_preset, 0) X(anchor_freq, 2)    \
    X(pol_variability, 2) X(conf_files, 0) X(layout_hash, 2)
#define QD_ML_LAYOUT(X)                                                                                                   \
    X(name, 0) X(tx, 2) X(rx, 2) X(links, 2) X(seed, 2) X(center_freq, 0) X(antenna, 2) X(antenna_cols, 0) X(antenna_args, 2) \
    X(wrap, 0) X(pairing, 2)

#define QD_ML_NAME(m, k) #m,
#define QD_ML_READ(m, k) qd_mex_layout_rd(s, #m, o.m, var_name, std::integral_constant<int, k>());
#define QD_ML_WRITE(m, k) qd_mex_layout_put(s, #m, o.m, std::integral_constant<int, k>());

// -------------------------------- layout_nodes --------------------------------

inline const std::vector<std::string> qd_mex_layout_nodes_fields = {QD_ML_NODES(QD_ML_NAME)};

// 1x1 struct -> layout_nodes; the data is copied
inline quadriga_lib::layout_nodes qd_mex_struct2nodes(const mxArray *s, const std::string &var_name)
{
    qd_mex_layout_check_fields(s, qd_mex_layout_nodes_fields, var_name);
    quadriga_lib::layout_nodes o;
    QD_ML_NODES(QD_ML_READ)
    o.scenpar_override = qd_mex_layout_map<arma::vec>(qd_mex_layout_get(s, "scenpar_override"), var_name + ".scenpar_override",
                                                      [](const mxArray *data, const std::string &name)
                                                      { return qd_mex_get_Col<double>(data, true, name); });
    if (const mxArray *a = qd_mex_layout_get(s, "extra"))
    {
        if (!mxIsCell(a))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Field 'extra' of '" + var_name + "' must be a cell array of cell arrays of strings.").c_str());
        for (size_t i = 0; i < mxGetNumberOfElements(a); ++i)
        {
            const mxArray *e = mxGetCell(a, (mwIndex)i);
            o.extra.push_back(e == nullptr ? std::vector<std::string>() : qd_mex_get_strings(e, var_name + ".extra"));
        }
    }
    return o;
}

inline mxArray *qd_mex_nodes2struct(const quadriga_lib::layout_nodes &o)
{
    mxArray *s = qd_mex_make_struct(qd_mex_layout_nodes_fields);
    QD_ML_NODES(QD_ML_WRITE)
    std::vector<std::string> keys;
    for (const auto &kv : o.scenpar_override)
        keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    mxArray *sp = qd_mex_make_struct(keys);
    for (const auto &k : keys)
        qd_mex_set_field(sp, k, qd_mex_copy2matlab(&o.scenpar_override.at(k)));
    qd_mex_set_field(s, "scenpar_override", sp);
    mxArray *extra = mxCreateCellMatrix((mwSize)o.extra.size(), 1);
    for (size_t i = 0; i < o.extra.size(); ++i)
        mxSetCell(extra, (mwIndex)i, qd_mex_copy2matlab(&o.extra[i]));
    qd_mex_set_field(s, "extra", extra);
    return s;
}

// -------------------------------- layout_links --------------------------------

inline const std::vector<std::string> qd_mex_layout_links_fields = {QD_ML_LINKS(QD_ML_NAME)};

// 1x1 struct -> layout_links; an empty struct gives the default links
inline quadriga_lib::layout_links qd_mex_struct2links(const mxArray *s, const std::string &var_name)
{
    qd_mex_layout_check_fields(s, qd_mex_layout_links_fields, var_name);
    quadriga_lib::layout_links o;
    QD_ML_LINKS(QD_ML_READ)
    if (const mxArray *a = qd_mex_layout_get(s, "seed"))
        o.seed = (arma::sword)qd_mex_get_seed(a, var_name + ".seed");
    if (const mxArray *a = qd_mex_layout_get(s, "anchor_freq"))
        o.anchor_freq = qd_mex_get_scalar<double>(a, var_name + ".anchor_freq", 2.0e9);
    if (const mxArray *a = qd_mex_layout_get(s, "pol_variability"))
        o.pol_variability = qd_mex_get_scalar<bool>(a, var_name + ".pol_variability", false);
    if (const mxArray *a = qd_mex_layout_get(s, "layout_hash"))
        o.layout_hash = qd_mex_get_scalar<unsigned long long>(a, var_name + ".layout_hash", 0ULL);
    return o;
}

inline mxArray *qd_mex_links2struct(const quadriga_lib::layout_links &o)
{
    mxArray *s = qd_mex_make_struct(qd_mex_layout_links_fields);
    QD_ML_LINKS(QD_ML_WRITE)
    qd_mex_set_field(s, "seed", qd_mex_copy2matlab(&o.seed));
    qd_mex_set_field(s, "anchor_freq", qd_mex_copy2matlab(&o.anchor_freq));
    qd_mex_set_field(s, "pol_variability", mxCreateLogicalScalar(o.pol_variability));
    qd_mex_set_field(s, "layout_hash", qd_mex_copy2matlab(&o.layout_hash));
    return s;
}

// -------------------------------- layout --------------------------------

inline const std::vector<std::string> qd_mex_layout_fields = {QD_ML_LAYOUT(QD_ML_NAME)};

// Cell array of cell arrays of strings -> lists of strings
inline std::vector<std::vector<std::string>> qd_mex_layout_string_lists(const mxArray *a, const std::string &name)
{
    if (!mxIsCell(a))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Field '" + name + "' must be a cell array of cell arrays of strings.").c_str());
    std::vector<std::vector<std::string>> lists;
    for (size_t i = 0; i < mxGetNumberOfElements(a); ++i)
    {
        const mxArray *e = mxGetCell(a, (mwIndex)i);
        lists.push_back(e == nullptr ? std::vector<std::string>() : qd_mex_get_strings(e, name));
    }
    return lists;
}

// 1x1 struct -> layout; the data is copied ('copy' is passed on to the antenna arrays: true if the call changes them);
// 'pairing' is 1-based in the struct; antenna entries are not validated here
inline quadriga_lib::layout qd_mex_struct2layout(const mxArray *s, bool copy, const std::string &var_name = "layout")
{
    qd_mex_layout_check_fields(s, qd_mex_layout_fields, var_name);
    quadriga_lib::layout o;
    QD_ML_LAYOUT(QD_ML_READ)
    if (const mxArray *a = qd_mex_layout_get(s, "tx"))
        o.tx = qd_mex_struct2nodes(a, var_name + ".tx");
    if (const mxArray *a = qd_mex_layout_get(s, "rx"))
        o.rx = qd_mex_struct2nodes(a, var_name + ".rx");
    if (const mxArray *a = qd_mex_layout_get(s, "links"))
        o.links = qd_mex_struct2links(a, var_name + ".links");
    if (const mxArray *a = qd_mex_layout_get(s, "seed"))
        o.seed = (arma::sword)qd_mex_get_seed(a, var_name + ".seed");
    if (const mxArray *a = qd_mex_layout_get(s, "antenna"))
    {
        if (!mxIsCell(a))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Field 'antenna' of '" + var_name + "' must be a cell array of struct arrays.").c_str());
        for (size_t i = 0; i < mxGetNumberOfElements(a); ++i)
        {
            const mxArray *e = mxGetCell(a, (mwIndex)i);
            if (e == nullptr || (mxIsEmpty(e) && !mxIsStruct(e)))
            {
                o.antenna.emplace_back();
                continue;
            }
            o.antenna.push_back(qd_mex_struct2arrayant_multi(e, false, copy, var_name + ".antenna{" + std::to_string(i + 1) + "}"));
            for (auto &band : o.antenna.back())
                if (band.e_theta_re.is_empty()) // A band that is not evaluated: undo the defaults of the reader
                    band.element_pos.reset(), band.port_weights_re.reset(), band.port_weights_im.reset();
        }
    }
    if (const mxArray *a = qd_mex_layout_get(s, "antenna_args"))
        o.antenna_args = qd_mex_layout_string_lists(a, var_name + ".antenna_args");
    if (const mxArray *a = qd_mex_layout_get(s, "pairing"))
    {
        const std::string name = var_name + ".pairing";
        if (qd_mex_get_dims(a, name)[0] != 2)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Field 'pairing' of '" + var_name + "' must have 2 rows.").c_str());
        const arma::uvec v = qd_mex_get_index_Col<arma::uword>(a, name);
        o.pairing = arma::umat(v.memptr(), 2, v.n_elem / 2);
    }
    return o;
}

inline mxArray *qd_mex_layout2struct(const quadriga_lib::layout &o)
{
    mxArray *s = qd_mex_make_struct(qd_mex_layout_fields);
    QD_ML_LAYOUT(QD_ML_WRITE)
    qd_mex_set_field(s, "tx", qd_mex_nodes2struct(o.tx));
    qd_mex_set_field(s, "rx", qd_mex_nodes2struct(o.rx));
    qd_mex_set_field(s, "links", qd_mex_links2struct(o.links));
    qd_mex_set_field(s, "seed", qd_mex_copy2matlab(&o.seed));
    mxArray *antenna = mxCreateCellMatrix((mwSize)o.antenna.size(), 1);
    for (size_t i = 0; i < o.antenna.size(); ++i)
        mxSetCell(antenna, (mwIndex)i, qd_mex_arrayant2struct_multi(o.antenna[i]));
    qd_mex_set_field(s, "antenna", antenna);
    mxArray *args = mxCreateCellMatrix((mwSize)o.antenna_args.size(), 1);
    for (size_t i = 0; i < o.antenna_args.size(); ++i)
        mxSetCell(args, (mwIndex)i, qd_mex_copy2matlab(&o.antenna_args[i]));
    qd_mex_set_field(s, "antenna_args", args);
    const arma::umat pairing = o.pairing + 1; // 0-based -> 1-based
    qd_mex_set_field(s, "pairing", qd_mex_copy2matlab(&pairing));
    return s;
}

#undef QD_ML_NODES
#undef QD_ML_LINKS
#undef QD_ML_LAYOUT
#undef QD_ML_NAME
#undef QD_ML_READ
#undef QD_ML_WRITE

// -------------------------------- gbsm_rays --------------------------------

// Cell array (one entry per link) of struct arrays (one entry per frequency object) -> rays[n_link][n_obj]
inline std::vector<std::vector<quadriga_lib::gbsm_rays>> qd_mex_cell2rays(const mxArray *input, const std::string &var_name)
{
    if (!mxIsCell(input))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Input '" + var_name + "' must be a cell array of struct arrays.").c_str());
    std::vector<std::vector<quadriga_lib::gbsm_rays>> rays(mxGetNumberOfElements(input));
    for (size_t l = 0; l < rays.size(); ++l)
    {
        const mxArray *e = mxGetCell(input, (mwIndex)l);
        const std::string name = var_name + "{" + std::to_string(l + 1) + "}";
        if (e == nullptr)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Input '" + name + "' must be a struct array.").c_str());
        qd_mex_check_struct(e, name);
        for (size_t f = 0; f < mxGetNumberOfElements(e); ++f)
            rays[l].push_back(qd_mex_struct2gbsm_rays_at(e, f, name));
    }
    return rays;
}

// rays[n_link][n_obj] -> cell array (n_link x 1) of 1 x n_obj struct arrays with the fields of ssf_rays
inline mxArray *qd_mex_rays2cell(const std::vector<std::vector<quadriga_lib::gbsm_rays>> &rays)
{
    mxArray *cell = mxCreateCellMatrix((mwSize)rays.size(), 1);
    for (size_t l = 0; l < rays.size(); ++l)
    {
        mxArray *s = qd_mex_make_struct(qd_mex_gbsm_rays_fields, rays[l].size());
        for (size_t f = 0; f < rays[l].size(); ++f)
        {
            const auto &r = rays[l][f];
            auto set = [&](const char *name, mxArray *a) { qd_mex_set_field(s, name, a, f); };
            set("method", mxCreateString(r.method.c_str()));
            set("tx_pos", qd_mex_copy2matlab_keep_shape(r.tx_pos));
            set("rx_pos", qd_mex_copy2matlab_keep_shape(r.rx_pos));
            set("center_freq", qd_mex_copy2matlab_keep_shape(r.center_freq));
            set("delay", qd_mex_copy2matlab_keep_shape(r.delay));
            set("power", qd_mex_copy2matlab_keep_shape(r.power));
            set("aod", qd_mex_copy2matlab_keep_shape(r.aod));
            set("eod", qd_mex_copy2matlab_keep_shape(r.eod));
            set("aoa", qd_mex_copy2matlab_keep_shape(r.aoa));
            set("eoa", qd_mex_copy2matlab_keep_shape(r.eoa));
            set("polmat", qd_mex_copy2matlab_keep_shape(r.polmat));
            set("path_type", qd_mex_copy2matlab_keep_shape(r.path_type));
            set("tap", qd_mex_copy2matlab_keep_shape(r.tap));
            set("gr_eps_re", qd_mex_copy2matlab_keep_shape(r.gr_eps_re));
            set("gr_eps_im", qd_mex_copy2matlab_keep_shape(r.gr_eps_im));
            set("fbs_pos", qd_mex_copy2matlab_keep_shape(r.fbs_pos));
            set("lbs_pos", qd_mex_copy2matlab_keep_shape(r.lbs_pos));
            set("fallback", qd_mex_copy2matlab_keep_shape(r.fallback));
        }
        mxSetCell(cell, (mwIndex)l, s);
    }
    return cell;
}

// Tx node, Rx node and frequency index of each channel, 0-based -> [3, n_channel], 1-based
inline mxArray *qd_mex_layout_index_out(const arma::umat &index)
{
    const arma::umat out = index + 1;
    return qd_mex_copy2matlab_keep_shape(out);
}

#endif
