// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Struct conversions shared by the MEX files of the layout functions; included only by those files

#ifndef qd_mex_layout_hpp
#define qd_mex_layout_hpp

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex_types.hpp"
#include "qd_helpers_gbsm.hpp"

#include <algorithm>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

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

// ---------------------------------------- Reading and writing helpers ----------------------------------------

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
    X(extra_names, 0) X(extra, 2)
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

// ---------------------------------------- layout_nodes ----------------------------------------

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

// ---------------------------------------- layout_links ----------------------------------------

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

// ---------------------------------------- layout ----------------------------------------

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

// ---------------------------------------- gbsm_rays ----------------------------------------

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
