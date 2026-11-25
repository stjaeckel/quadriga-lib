// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Dict conversions shared by the layout wrappers; no documentation blocks

#include "qd_helpers_python_types.hpp"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------- Reading helpers ----------------------------------------

// Value of a key; None if it is missing
static inline py::object qd_pl_item(const py::dict &d, const char *key)
{
    return d.contains(key) ? py::object(d[key]) : py::object(py::none());
}

// A dict whose keys are all in 'known'; the error names an unknown key
static inline void qd_pl_check_keys(const py::dict &d, const std::vector<std::string> &known, const std::string &var_name)
{
    for (auto item : d)
    {
        const std::string key = py::cast<std::string>(py::str(item.first));
        if (std::find(known.begin(), known.end(), key) == known.end())
            throw std::invalid_argument("Unknown key '" + key + "' in '" + var_name + "'.");
    }
}

// Value of a key that must be a dict; None or missing = not given
static inline bool qd_pl_get_dict(const py::dict &d, const char *key, py::dict &out, const std::string &var_name)
{
    const py::object v = qd_pl_item(d, key);
    if (v.is_none())
        return false;
    if (!py::isinstance<py::dict>(v))
        throw std::invalid_argument("Entry '" + std::string(key) + "' of '" + var_name + "' must be a dict.");
    out = py::reinterpret_borrow<py::dict>(v);
    return true;
}

// Integer value of a Python int or numpy integer (not float, str); out of range or another type: std::invalid_argument
static inline py::object qd_pl_index_object(const py::handle &h, const std::string &name)
{
    if (!py::hasattr(h, "__index__") || py::isinstance<py::float_>(h))
        throw std::invalid_argument("Entry '" + name + "' must be an integer.");
    return py::reinterpret_steal<py::object>(PyNumber_Index(h.ptr()));
}

// Seed: an integer >= -1
static inline arma::sword qd_pl_seed(const py::handle &h, const std::string &name)
{
    const py::object idx = qd_pl_index_object(h, name);
    const long long seed = PyLong_AsLongLong(idx.ptr());
    if (seed == -1 && PyErr_Occurred())
    {
        PyErr_Clear();
        throw std::invalid_argument("Entry '" + name + "' must be -1 or a non-negative integer below 2^63.");
    }
    if (seed < -1)
        throw std::invalid_argument("Entry '" + name + "' must be -1 or a non-negative integer.");
    return (arma::sword)seed;
}

// A real number (int, float or numpy scalar, not str)
static inline double qd_pl_number(const py::handle &h, const std::string &name)
{
    if (!(py::hasattr(h, "__float__") || py::hasattr(h, "__index__")) || py::isinstance<py::str>(h))
        throw std::invalid_argument("Entry '" + name + "' must be a number.");
    const double v = PyFloat_AsDouble(h.ptr());
    if (v == -1.0 && PyErr_Occurred())
    {
        PyErr_Clear();
        throw std::invalid_argument("Entry '" + name + "' must be a number.");
    }
    return v;
}

// A non-negative integer below 2^64
static inline unsigned long long qd_pl_unsigned(const py::handle &h, const std::string &name)
{
    const py::object idx = qd_pl_index_object(h, name);
    const unsigned long long v = PyLong_AsUnsignedLongLong(idx.ptr());
    if (v == (unsigned long long)-1 && PyErr_Occurred())
    {
        PyErr_Clear();
        throw std::invalid_argument("Entry '" + name + "' must be an integer from 0 to 18446744073709551615.");
    }
    return v;
}

// Dict -> arrayant (not validated); a missing key is a std::invalid_argument naming the dict
static inline quadriga_lib::arrayant<double> qd_pl_dict2arrayant(const py::dict &d, bool view, const std::string &name)
{
    try
    {
        return qd_python_dict2arrayant(d, view, false, false);
    }
    catch (py::error_already_set &e)
    {
        if (!e.matches(PyExc_KeyError))
            throw;
        const std::string key = py::cast<std::string>(py::str(e.value()));
        throw std::invalid_argument("Entry '" + name + "' is not an arrayant dict: the key " + key + " is missing.");
    }
}

// A member of a layout class is read from and written to the dict key of the same name. Kind 0: copied; 1: by hand. A
// missing or None key leaves the member empty.
template <typename T>
static inline void qd_pl_rd(const py::dict &d, const char *k, arma::Col<T> &x, const std::string &vn, bool view)
{
    x = qd_python_numpy2arma_Col<T>(py::handle(qd_pl_item(d, k)), view, false, vn + "." + k);
}
static inline void qd_pl_rd(const py::dict &d, const char *k, arma::mat &x, const std::string &vn, bool view)
{
    x = qd_python_numpy2arma_Mat<double>(py::handle(qd_pl_item(d, k)), view, false, vn + "." + k);
}
static inline void qd_pl_rd(const py::dict &d, const char *k, arma::umat &x, const std::string &vn, bool view)
{
    x = qd_python_numpy2arma_Mat<arma::uword>(py::handle(qd_pl_item(d, k)), view, false, vn + "." + k);
}
static inline void qd_pl_rd(const py::dict &d, const char *k, std::vector<std::string> &x, const std::string &, bool)
{
    x = qd_python_list2vector_Strings(py::handle(qd_pl_item(d, k)));
}
static inline void qd_pl_rd(const py::dict &d, const char *k, std::string &x, const std::string &vn, bool)
{
    const py::object v = qd_pl_item(d, k);
    if (v.is_none())
        return;
    if (!py::isinstance<py::str>(v))
        throw std::invalid_argument("Entry '" + std::string(k) + "' of '" + vn + "' must be a str.");
    x = py::cast<std::string>(v);
}
// Members as output: copies; integers of index type are int64
template <typename T>
static inline py::object qd_pl_wr(const arma::Col<T> &x) { return qd_python_copy2numpy(&x); }
static inline py::object qd_pl_wr(const arma::uvec &x) { return qd_python_copy2numpy<arma::uword, py::ssize_t>(&x); }
static inline py::object qd_pl_wr(const arma::mat &x) { return qd_python_copy2numpy(&x); }
static inline py::object qd_pl_wr(const arma::umat &x) { return qd_python_copy2numpy<arma::uword, py::ssize_t>(&x); }
static inline py::object qd_pl_wr(const std::vector<std::string> &x) { return qd_python_copy2list(x); }
static inline py::object qd_pl_wr(const std::string &x) { return py::str(x); }

// Members in the order of the C++ classes: X(member, kind)
#define QD_PL_NODES(X)                                                                                                 \
    X(name, 0) X(sector, 0) X(i_antenna, 0) X(offset, 0) X(pos, 0) X(orientation, 0) X(direction, 0) X(speed, 0)       \
    X(time, 0) X(scenario, 0) X(indoor, 0) X(floor, 0) X(d2d_in, 0) X(clutter, 0) X(scenpar_override, 1) X(seed, 0)    \
    X(relative, 0) X(cell, 0) X(extra_names, 0) X(extra, 1)
#define QD_PL_LINKS(X)                                                                                                 \
    X(i_tx, 0) X(i_rx, 0) X(i_segment, 0) X(image, 0) X(i_conf, 0) X(conf_names, 0) X(los, 0) X(indoor, 0)             \
    X(floor_id, 0) X(d2d_in, 0) X(o2i_rnd, 0) X(o2i_model, 0) X(gbsm_seed, 0) X(seed, 1) X(method, 0) X(sos_preset, 0) \
    X(anchor_freq, 1) X(pol_variability, 1) X(conf_files, 0) X(layout_hash, 1)
#define QD_PL_LAYOUT(X)                                                                                                \
    X(name, 0) X(tx, 1) X(rx, 1) X(links, 1) X(seed, 1) X(center_freq, 0) X(antenna, 1) X(antenna_cols, 0)             \
    X(antenna_args, 1) X(wrap, 0) X(pairing, 0)

#define QD_PL_NAME(m, k) #m,
#define QD_PL_READ_0(m) qd_pl_rd(d, #m, o.m, var_name, view);
#define QD_PL_READ_1(m)
#define QD_PL_READ_(m, k) QD_PL_READ_##k(m)
#define QD_PL_WRITE_0(m) d[#m] = qd_pl_wr(o.m);
#define QD_PL_WRITE_1(m)
#define QD_PL_WRITE_(m, k) QD_PL_WRITE_##k(m)
#define QD_PL_READ(m, k) QD_PL_READ_(m, k)
#define QD_PL_WRITE(m, k) QD_PL_WRITE_(m, k)

// The keys of 'd' in the order of 'keys'
static inline py::dict qd_pl_ordered(const py::dict &d, const std::vector<std::string> &keys)
{
    py::dict out;
    for (const auto &k : keys)
        out[py::str(k)] = d[py::str(k)];
    return out;
}

// Lists of lists of strings
static inline std::vector<std::vector<std::string>> qd_pl_string_lists(const py::handle &h, const std::string &name)
{
    std::vector<std::vector<std::string>> lists;
    if (h.is_none())
        return lists;
    if (!py::isinstance<py::list>(h) && !py::isinstance<py::tuple>(h))
        throw std::invalid_argument("Entry '" + name + "' must be a list of lists of str.");
    for (py::handle e : py::reinterpret_borrow<py::sequence>(h))
        lists.push_back(qd_python_list2vector_Strings(e));
    return lists;
}

static inline py::list qd_pl_string_lists2list(const std::vector<std::vector<std::string>> &lists)
{
    py::list out;
    for (const auto &l : lists)
        out.append(qd_python_copy2list(l));
    return out;
}

// ---------------------------------------- layout_nodes ----------------------------------------

static const std::vector<std::string> qd_pl_nodes_keys = {QD_PL_NODES(QD_PL_NAME)};

// Dict -> layout_nodes; with 'view' the numpy memory is aliased where possible (read-only use)
static inline quadriga_lib::layout_nodes qd_python_dict2nodes(const py::dict &d, bool view, const std::string &var_name = "nodes")
{
    qd_pl_check_keys(d, qd_pl_nodes_keys, var_name);
    quadriga_lib::layout_nodes o;
    QD_PL_NODES(QD_PL_READ)
    py::dict sp;
    if (qd_pl_get_dict(d, "scenpar_override", sp, var_name))
        for (auto item : sp)
        {
            if (!py::isinstance<py::str>(item.first))
                throw std::invalid_argument("Keys of '" + var_name + ".scenpar_override' must be str.");
            const std::string key = py::cast<std::string>(item.first);
            o.scenpar_override[key] = qd_python_numpy2arma_Col<double>(item.second, view, false, var_name + ".scenpar_override." + key);
        }
    o.extra = qd_pl_string_lists(qd_pl_item(d, "extra"), var_name + ".extra");
    return o;
}

static inline py::dict qd_python_nodes2dict(const quadriga_lib::layout_nodes &o)
{
    py::dict d;
    QD_PL_NODES(QD_PL_WRITE)
    std::vector<std::string> keys;
    for (const auto &kv : o.scenpar_override)
        keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    py::dict sp;
    for (const auto &k : keys)
        sp[py::str(k)] = qd_python_copy2numpy(&o.scenpar_override.at(k));
    d["scenpar_override"] = sp;
    d["extra"] = qd_pl_string_lists2list(o.extra);
    return qd_pl_ordered(d, qd_pl_nodes_keys);
}

// ---------------------------------------- layout_links ----------------------------------------

static const std::vector<std::string> qd_pl_links_keys = {QD_PL_LINKS(QD_PL_NAME)};

static inline quadriga_lib::layout_links qd_python_dict2links(const py::dict &d, bool view, const std::string &var_name = "links")
{
    qd_pl_check_keys(d, qd_pl_links_keys, var_name);
    quadriga_lib::layout_links o;
    QD_PL_LINKS(QD_PL_READ)
    py::object v = qd_pl_item(d, "seed");
    if (!v.is_none())
        o.seed = qd_pl_seed(v, var_name + ".seed");
    if (!(v = qd_pl_item(d, "anchor_freq")).is_none())
        o.anchor_freq = qd_pl_number(v, var_name + ".anchor_freq");
    o.pol_variability = qd_python_flag(qd_pl_item(d, "pol_variability"), false, var_name + ".pol_variability");
    if (!(v = qd_pl_item(d, "layout_hash")).is_none())
        o.layout_hash = qd_pl_unsigned(v, var_name + ".layout_hash");
    return o;
}

static inline py::dict qd_python_links2dict(const quadriga_lib::layout_links &o)
{
    py::dict d;
    QD_PL_LINKS(QD_PL_WRITE)
    d["seed"] = (long long)o.seed;
    d["anchor_freq"] = o.anchor_freq;
    d["pol_variability"] = o.pol_variability;
    d["layout_hash"] = o.layout_hash;
    return qd_pl_ordered(d, qd_pl_links_keys);
}

// ---------------------------------------- layout ----------------------------------------

static const std::vector<std::string> qd_pl_layout_keys = {QD_PL_LAYOUT(QD_PL_NAME)};

// Dict -> layout; with 'view' the numpy memory is aliased where possible, so the layout must be used read-only; antenna
// entries are not validated here
static inline quadriga_lib::layout qd_python_dict2layout(const py::dict &d, bool view, const std::string &var_name = "layout")
{
    qd_pl_check_keys(d, qd_pl_layout_keys, var_name);
    quadriga_lib::layout o;
    QD_PL_LAYOUT(QD_PL_READ)
    py::dict sub;
    if (qd_pl_get_dict(d, "tx", sub, var_name))
        o.tx = qd_python_dict2nodes(sub, view, var_name + ".tx");
    if (qd_pl_get_dict(d, "rx", sub, var_name))
        o.rx = qd_python_dict2nodes(sub, view, var_name + ".rx");
    if (qd_pl_get_dict(d, "links", sub, var_name))
        o.links = qd_python_dict2links(sub, view, var_name + ".links");
    py::object v = qd_pl_item(d, "seed");
    if (!v.is_none())
        o.seed = qd_pl_seed(v, var_name + ".seed");

    v = qd_pl_item(d, "antenna");
    if (!v.is_none())
    {
        if (!py::isinstance<py::list>(v) && !py::isinstance<py::tuple>(v))
            throw std::invalid_argument("Entry 'antenna' of '" + var_name + "' must be a list of lists of arrayant dicts.");
        size_t i = 0;
        for (py::handle entry : py::reinterpret_borrow<py::sequence>(v))
        {
            const std::string name = var_name + ".antenna[" + std::to_string(i++) + "]";
            o.antenna.emplace_back();
            if (entry.is_none())
                continue;
            if (!py::isinstance<py::list>(entry) && !py::isinstance<py::tuple>(entry))
                throw std::invalid_argument("Entry '" + name + "' must be a list of arrayant dicts, one per band.");
            size_t b = 0;
            for (py::handle band : py::reinterpret_borrow<py::sequence>(entry))
            {
                if (!py::isinstance<py::dict>(band))
                    throw std::invalid_argument("Entry '" + name + "' must be a list of arrayant dicts, one per band.");
                o.antenna.back().push_back(qd_pl_dict2arrayant(py::reinterpret_borrow<py::dict>(band), view, name + "[" + std::to_string(b++) + "]"));
                auto &ant = o.antenna.back().back();
                if (ant.e_theta_re.is_empty()) // A band that is not evaluated: undo the defaults of the reader
                    ant.element_pos.reset(), ant.port_weights_re.reset(), ant.port_weights_im.reset();
            }
        }
    }
    o.antenna_args = qd_pl_string_lists(qd_pl_item(d, "antenna_args"), var_name + ".antenna_args");
    if (!o.pairing.is_empty() && o.pairing.n_rows != 2)
        throw std::invalid_argument("Entry 'pairing' of '" + var_name + "' must have 2 rows.");
    return o;
}

static inline py::dict qd_python_layout2dict(const quadriga_lib::layout &o)
{
    py::dict d;
    QD_PL_LAYOUT(QD_PL_WRITE)
    d["tx"] = qd_python_nodes2dict(o.tx);
    d["rx"] = qd_python_nodes2dict(o.rx);
    d["links"] = qd_python_links2dict(o.links);
    d["seed"] = (long long)o.seed;
    py::list antenna;
    for (const auto &bands : o.antenna)
    {
        py::list l;
        for (const auto &band : bands)
            l.append(qd_python_arrayant2dict(band));
        antenna.append(l);
    }
    d["antenna"] = antenna;
    d["antenna_args"] = qd_pl_string_lists2list(o.antenna_args);
    return qd_pl_ordered(d, qd_pl_layout_keys);
}

// ---------------------------------------- gbsm_rays ----------------------------------------

// List (one entry per link) of ray dicts or lists of ray dicts (one per frequency object) -> rays[n_link][n_obj]
static inline std::vector<std::vector<quadriga_lib::gbsm_rays>> qd_pl_list2rays(const py::handle &h, const std::string &var_name)
{
    std::vector<std::vector<quadriga_lib::gbsm_rays>> rays;
    if (h.is_none())
        return rays;
    if (!py::isinstance<py::list>(h) && !py::isinstance<py::tuple>(h))
        throw std::invalid_argument("Input '" + var_name + "' must be a list with one entry per link.");
    size_t l = 0;
    for (py::handle entry : py::reinterpret_borrow<py::sequence>(h))
    {
        const std::string name = var_name + "[" + std::to_string(l++) + "]";
        rays.emplace_back();
        if (py::isinstance<py::dict>(entry))
            rays.back().push_back(qd_python_dict2gbsm_rays(py::reinterpret_borrow<py::dict>(entry), name));
        else if (py::isinstance<py::list>(entry) || py::isinstance<py::tuple>(entry))
        {
            size_t f = 0;
            for (py::handle e : py::reinterpret_borrow<py::sequence>(entry))
            {
                if (!py::isinstance<py::dict>(e))
                    throw std::invalid_argument("Input '" + name + "' must be a ray dict or a list of ray dicts.");
                rays.back().push_back(qd_python_dict2gbsm_rays(py::reinterpret_borrow<py::dict>(e), name + "[" + std::to_string(f++) + "]"));
            }
        }
        else
            throw std::invalid_argument("Input '" + name + "' must be a ray dict or a list of ray dicts.");
    }
    return rays;
}

// gbsm_rays -> dict with the keys of ssf_rays (copies)
static inline py::dict qd_pl_rays2dict(const quadriga_lib::gbsm_rays &r)
{
    py::dict d;
    const arma::vec tx = arma::vectorise(r.tx_pos), rx = arma::vectorise(r.rx_pos);
    d["method"] = r.method;
    d["tx_pos"] = qd_python_copy2numpy(&tx);
    d["rx_pos"] = qd_python_copy2numpy(&rx);
    d["center_freq"] = qd_python_copy2numpy(&r.center_freq);
    d["delay"] = qd_python_copy2numpy(&r.delay);
    d["power"] = qd_python_copy2numpy(&r.power);
    d["aod"] = qd_python_copy2numpy(&r.aod);
    d["eod"] = qd_python_copy2numpy(&r.eod);
    d["aoa"] = qd_python_copy2numpy(&r.aoa);
    d["eoa"] = qd_python_copy2numpy(&r.eoa);
    d["polmat"] = qd_python_copy2numpy(&r.polmat);
    d["path_type"] = qd_python_copy2numpy(&r.path_type);
    d["tap"] = qd_python_copy2numpy(&r.tap);
    if (r.gr_eps_re.empty())
        d["gr_eps"] = py::array_t<std::complex<double>>(0);
    else
        d["gr_eps"] = qd_python_copy2numpy<double, std::complex<double>>(&r.gr_eps_re, &r.gr_eps_im);
    d["fbs_pos"] = r.fbs_pos.is_empty() ? py::object(py::array_t<double>(0)) : py::object(qd_python_copy2numpy(&r.fbs_pos));
    d["lbs_pos"] = r.lbs_pos.is_empty() ? py::object(py::array_t<double>(0)) : py::object(qd_python_copy2numpy(&r.lbs_pos));
    d["fallback"] = r.fallback.is_empty() ? py::object(py::array_t<unsigned>(0)) : py::object(qd_python_copy2numpy(&r.fallback));
    return d;
}

// rays[n_link][n_obj] -> list of lists of ray dicts
static inline py::list qd_pl_rays2list(const std::vector<std::vector<quadriga_lib::gbsm_rays>> &rays)
{
    py::list out;
    for (const auto &link : rays)
    {
        py::list l;
        for (const auto &r : link)
            l.append(qd_pl_rays2dict(r));
        out.append(l);
    }
    return out;
}

// ---------------------------------------- Arguments ----------------------------------------

// Dict of name -> numbers (options of the generators and tracks): one vector of values per key
static inline std::unordered_map<std::string, std::vector<double>> qd_pl_options(const py::handle &h, const std::string &var_name)
{
    std::unordered_map<std::string, std::vector<double>> map;
    if (h.is_none())
        return map;
    if (!py::isinstance<py::dict>(h))
        throw std::invalid_argument("Input '" + var_name + "' must be a dict or None.");
    for (auto item : py::reinterpret_borrow<py::dict>(h))
    {
        if (!py::isinstance<py::str>(item.first))
            throw std::invalid_argument("Keys of '" + var_name + "' must be str.");
        const std::string key = py::cast<std::string>(item.first);
        const arma::vec v = qd_python_numpy2arma_Col<double>(item.second, false, false, var_name + "." + key);
        map[key] = std::vector<double>(v.begin(), v.end());
    }
    return map;
}

// Dict of name -> numbers or arrays (arguments of generate_antenna): one matrix per key; a 1D input is one row
static inline std::unordered_map<std::string, arma::mat> qd_pl_args(const py::handle &h, const std::string &var_name)
{
    std::unordered_map<std::string, arma::mat> map;
    if (h.is_none())
        return map;
    if (!py::isinstance<py::dict>(h))
        throw std::invalid_argument("Input '" + var_name + "' must be a dict or None.");
    for (auto item : py::reinterpret_borrow<py::dict>(h))
    {
        if (!py::isinstance<py::str>(item.first))
            throw std::invalid_argument("Keys of '" + var_name + "' must be str.");
        const std::string key = py::cast<std::string>(item.first);
        arma::mat v = qd_python_numpy2arma_Mat<double>(item.second, false, false, var_name + "." + key);
        const py::array a = py::array::ensure(item.second);
        if (a && a.ndim() == 1)
            arma::inplace_trans(v);
        map[key] = v;
    }
    return map;
}

// Folders of user conf files followed by the shipped folder (found at run time next to the module)
std::string qd_python_conf_dir();
static inline std::vector<std::string> qd_pl_config_dirs(const py::handle &h)
{
    std::vector<std::string> dirs = qd_python_list2vector_Strings(h);
    const std::string shipped = qd_python_conf_dir();
    if (!shipped.empty())
        dirs.push_back(shipped);
    return dirs;
}
