// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef qd_helpers_python_types_hpp
#define qd_helpers_python_types_hpp

#include "quadriga_lib.hpp"
#include "qd_helpers_python.hpp"
#include "qd_helpers_wrappers.hpp"

#include <algorithm>
#include <iterator>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

static inline py::dict qd_python_arrayant2dict(const quadriga_lib::arrayant<double> &ant)
{
    py::dict output;
    output["e_theta_re"] = qd_python_copy2numpy(&ant.e_theta_re);
    output["e_theta_im"] = qd_python_copy2numpy(&ant.e_theta_im);
    output["e_phi_re"] = qd_python_copy2numpy(&ant.e_phi_re);
    output["e_phi_im"] = qd_python_copy2numpy(&ant.e_phi_im);
    output["azimuth_grid"] = qd_python_copy2numpy(&ant.azimuth_grid);
    output["elevation_grid"] = qd_python_copy2numpy(&ant.elevation_grid);
    output["element_pos"] = qd_python_copy2numpy(&ant.element_pos);
    output["port_weights_re"] = qd_python_copy2numpy(&ant.port_weights_re);
    output["port_weights_im"] = qd_python_copy2numpy(&ant.port_weights_im);
    output["center_freq"] = ant.center_freq;
    output["name"] = ant.name;
    return output;
}

static inline py::dict qd_python_arrayant2dict_multi(const std::vector<quadriga_lib::arrayant<double>> &ant)
{
    if (ant.empty())
        return py::dict();

    if (ant.size() == 1)
        return qd_python_arrayant2dict(ant[0]);

    size_t n_freq = ant.size();
    const auto &a0 = ant[0];

    py::dict output;

    // Pattern fields to 4D numpy arrays
    std::vector<arma::Cube<double>> vec_etr(n_freq), vec_eti(n_freq), vec_epr(n_freq), vec_epi(n_freq);
    for (size_t f = 0; f < n_freq; ++f)
    {
        vec_etr[f] = ant[f].e_theta_re;
        vec_eti[f] = ant[f].e_theta_im;
        vec_epr[f] = ant[f].e_phi_re;
        vec_epi[f] = ant[f].e_phi_im;
    }
    output["e_theta_re"] = qd_python_stack2numpy(&vec_etr);
    output["e_theta_im"] = qd_python_stack2numpy(&vec_eti);
    output["e_phi_re"] = qd_python_stack2numpy(&vec_epr);
    output["e_phi_im"] = qd_python_stack2numpy(&vec_epi);

    // Shared fields from first entry
    output["azimuth_grid"] = qd_python_copy2numpy(&a0.azimuth_grid);
    output["elevation_grid"] = qd_python_copy2numpy(&a0.elevation_grid);
    output["element_pos"] = qd_python_copy2numpy(&a0.element_pos);
    output["name"] = a0.name;

    // Center frequency as 1D array
    arma::Col<double> center_freqs(n_freq);
    for (size_t f = 0; f < n_freq; ++f)
        center_freqs[f] = ant[f].center_freq;
    output["center_freq"] = qd_python_copy2numpy(&center_freqs);

    // Port weights: check if all entries are identical
    bool port_weights_varies = false;
    for (size_t f = 1; f < n_freq && !port_weights_varies; ++f)
    {
        if (ant[f].port_weights_re.n_elem != a0.port_weights_re.n_elem ||
            ant[f].port_weights_im.n_elem != a0.port_weights_im.n_elem)
            port_weights_varies = true;
        else
        {
            if (a0.port_weights_re.n_elem > 0 && !arma::approx_equal(ant[f].port_weights_re, a0.port_weights_re, "absdiff", 1e-15))
                port_weights_varies = true;
            if (a0.port_weights_im.n_elem > 0 && !arma::approx_equal(ant[f].port_weights_im, a0.port_weights_im, "absdiff", 1e-15))
                port_weights_varies = true;
        }
    }

    if (port_weights_varies)
    {
        // Per-frequency: store as 3D (n_el × n_ports × n_freq)
        if (a0.port_weights_re.n_elem > 0)
        {
            arma::uword nr = a0.port_weights_re.n_rows, nc = a0.port_weights_re.n_cols;
            arma::Cube<double> pw_re(nr, nc, n_freq, arma::fill::none);
            for (size_t f = 0; f < n_freq; ++f)
                pw_re.slice(f) = ant[f].port_weights_re;
            output["port_weights_re"] = qd_python_copy2numpy(&pw_re);
        }
        else
            output["port_weights_re"] = qd_python_copy2numpy(&a0.port_weights_re);

        if (a0.port_weights_im.n_elem > 0)
        {
            arma::uword nr = a0.port_weights_im.n_rows, nc = a0.port_weights_im.n_cols;
            arma::Cube<double> pw_im(nr, nc, n_freq, arma::fill::none);
            for (size_t f = 0; f < n_freq; ++f)
                pw_im.slice(f) = ant[f].port_weights_im;
            output["port_weights_im"] = qd_python_copy2numpy(&pw_im);
        }
        else
            output["port_weights_im"] = qd_python_copy2numpy(&a0.port_weights_im);
    }
    else
    {
        // Shared: store as 2D
        output["port_weights_re"] = qd_python_copy2numpy(&a0.port_weights_re);
        output["port_weights_im"] = qd_python_copy2numpy(&a0.port_weights_im);
    }

    return output;
}

// The matrix of element-to-port weights was called coupling_re / coupling_im; ignoring the old keys would drop the weights
static inline void qd_python_check_old_port_weights_keys(const py::dict &arrayant)
{
    for (const char *suffix : {"re", "im"})
        if (arrayant.contains((std::string("coupling_") + suffix).c_str()))
            throw std::invalid_argument(std::string("Key 'coupling_") + suffix + "' was renamed to 'port_weights_" + suffix + "'.");
}

static inline quadriga_lib::arrayant<double> qd_python_dict2arrayant(const py::dict &arrayant,
                                                              bool view = false, bool strict = false,
                                                              bool validate = false)
{
    qd_python_check_old_port_weights_keys(arrayant);
    quadriga_lib::arrayant<double> ant;

    // A 4D pattern is multi-frequency data, which has its own converter
    const auto e_theta_re_arr = py::array::ensure(arrayant["e_theta_re"]);
    if (e_theta_re_arr && e_theta_re_arr.ndim() == 4)
        throw std::invalid_argument("Arrayant dict has 4D (multi-frequency) patterns; this function takes "
                                    "single-frequency (3D) patterns.");

    // Mandatory fields
    ant.e_theta_re = qd_python_numpy2arma_Cube<double>(arrayant["e_theta_re"], view, strict);
    ant.e_theta_im = qd_python_numpy2arma_Cube<double>(arrayant["e_theta_im"], view, strict);
    ant.e_phi_re = qd_python_numpy2arma_Cube<double>(arrayant["e_phi_re"], view, strict);
    ant.e_phi_im = qd_python_numpy2arma_Cube<double>(arrayant["e_phi_im"], view, strict);
    ant.azimuth_grid = qd_python_numpy2arma_Col<double>(arrayant["azimuth_grid"], view, strict);
    ant.elevation_grid = qd_python_numpy2arma_Col<double>(arrayant["elevation_grid"], view, strict);

    // Optional fields; absent or empty ones get their defaults (zeros, identity)
    if (arrayant.contains("element_pos"))
        ant.element_pos = qd_python_numpy2arma_Mat<double>(arrayant["element_pos"], view, strict);
    if (arrayant.contains("port_weights_re"))
        ant.port_weights_re = qd_python_numpy2arma_Mat<double>(arrayant["port_weights_re"], view, strict);
    if (arrayant.contains("port_weights_im"))
        ant.port_weights_im = qd_python_numpy2arma_Mat<double>(arrayant["port_weights_im"], view, strict);
    qd_arrayant_fill_defaults(ant);

    if (arrayant.contains("center_freq")) // Scalar or array with exactly one element
    {
        const auto center_freq = qd_python_numpy2arma_Col<double>(arrayant["center_freq"]);
        if (center_freq.n_elem != 1)
            throw std::invalid_argument("Input 'center_freq' must have exactly one element for a single-frequency arrayant.");
        ant.center_freq = center_freq.at(0);
    }

    if (arrayant.contains("name"))
        ant.name = arrayant["name"].cast<std::string>();

    if (validate)
    {
        auto error_msg = ant.is_valid();
        if (!error_msg.empty())
            throw std::invalid_argument(error_msg.c_str());
    }

    return ant;
}

// Multi-frequency: dict → std::vector<arrayant>
// - 'require_center_freq': 4D input without 'center_freq' raises; pass false only where the frequency is not used
//   (all entries then get the default 299792458)
static inline std::vector<quadriga_lib::arrayant<double>> qd_python_dict2arrayant_multi(const py::dict &arrayant,
                                                                                 bool view = false, bool strict = false,
                                                                                 bool validate = false,
                                                                                 bool require_center_freq = true)
{
    qd_python_check_old_port_weights_keys(arrayant);

    // Determine number of frequencies from e_theta_re shape
    py::array e_theta_re_arr = py::cast<py::array>(arrayant["e_theta_re"]);
    int nd = (int)e_theta_re_arr.request().ndim;

    // 1D/2D patterns: a single-frequency antenna with implicit trailing dimensions (promoted to 3D), one entry
    if (nd < 3)
    {
        std::vector<quadriga_lib::arrayant<double>> ant(1);
        ant[0] = qd_python_dict2arrayant(arrayant, view, strict, validate);
        return ant;
    }

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

    // Parse center_freq: one value per frequency entry (scalar or array); required for more than one entry
    arma::Col<double> center_freqs(n_freq);
    if (arrayant.contains("center_freq"))
    {
        center_freqs = qd_python_numpy2arma_Col<double>(arrayant["center_freq"]);
        if (center_freqs.n_elem != n_freq)
            throw std::invalid_argument("'center_freq' length must match number of frequency entries.");
    }
    else if (n_freq == 1 || !require_center_freq)
        center_freqs.fill(299792458.0); // Default
    else
        throw std::invalid_argument("Input 'center_freq' is required for a multi-frequency arrayant (4D pattern fields).");

    // Parse port weights: 2D (shared) or 3D (per-freq)
    std::vector<arma::Mat<double>> port_weights_re_vec(n_freq);
    std::vector<arma::Mat<double>> port_weights_im_vec(n_freq);
    bool has_port_weights_re = arrayant.contains("port_weights_re");
    bool has_port_weights_im = arrayant.contains("port_weights_im");

    if (has_port_weights_re)
    {
        py::array pw_arr = py::cast<py::array>(arrayant["port_weights_re"]);
        int pw_nd = (int)pw_arr.request().ndim;
        if (pw_nd <= 2) // Shared across all frequencies
        {
            auto cpl = qd_python_numpy2arma_Mat<double>(arrayant["port_weights_re"], view, strict);
            for (size_t f = 0; f < n_freq; ++f)
                port_weights_re_vec[f] = cpl;
        }
        else // 3D: (n_el, n_ports, n_freq) — slices are per-frequency port weights
        {
            auto pw_cube = qd_python_numpy2arma_Cube<double>(arrayant["port_weights_re"], view, strict);
            if (pw_cube.n_slices != n_freq)
                throw std::invalid_argument("'port_weights_re' 3rd dimension must match number of frequency entries.");
            for (size_t f = 0; f < n_freq; ++f)
                port_weights_re_vec[f] = pw_cube.slice(f);
        }
    }

    if (has_port_weights_im)
    {
        py::array pw_arr = py::cast<py::array>(arrayant["port_weights_im"]);
        int pw_nd = (int)pw_arr.request().ndim;
        if (pw_nd <= 2)
        {
            auto cpl = qd_python_numpy2arma_Mat<double>(arrayant["port_weights_im"], view, strict);
            for (size_t f = 0; f < n_freq; ++f)
                port_weights_im_vec[f] = cpl;
        }
        else
        {
            auto pw_cube = qd_python_numpy2arma_Cube<double>(arrayant["port_weights_im"], view, strict);
            if (pw_cube.n_slices != n_freq)
                throw std::invalid_argument("'port_weights_im' 3rd dimension must match number of frequency entries.");
            for (size_t f = 0; f < n_freq; ++f)
                port_weights_im_vec[f] = pw_cube.slice(f);
        }
    }

    // Build the vector of arrayant
    std::vector<quadriga_lib::arrayant<double>> ant(n_freq);
    for (size_t f = 0; f < n_freq; ++f)
    {
        ant[f].e_theta_re = std::move(e_theta_re[f]);
        ant[f].e_theta_im = std::move(e_theta_im[f]);
        ant[f].e_phi_re = std::move(e_phi_re[f]);
        ant[f].e_phi_im = std::move(e_phi_im[f]);
        ant[f].azimuth_grid = azimuth_grid;
        ant[f].elevation_grid = elevation_grid;
        ant[f].element_pos = element_pos;
        ant[f].center_freq = center_freqs[f];
        ant[f].name = name;
        if (has_port_weights_re)
            ant[f].port_weights_re = std::move(port_weights_re_vec[f]);
        if (has_port_weights_im)
            ant[f].port_weights_im = std::move(port_weights_im_vec[f]);
        qd_arrayant_fill_defaults(ant[f]);
    }

    if (validate)
    {
        auto error_msg = quadriga_lib::is_valid(ant);
        if (!error_msg.empty())
            throw std::invalid_argument(error_msg.c_str());
    }

    return ant;
}

// Emit a vector of per-snapshot arma objects either as a Python list of arrays
// (stack = false) or stacked into one higher-dimensional Numpy array along an
// appended snapshot axis (stack = true). i_vec selects / orders the snapshots.
template <typename dtype_numpy, typename dtype_arma>
static inline py::object qd_python_emit_paths(const std::vector<dtype_arma> *re,
                                       bool stack,
                                       const arma::uvec &i_vec = arma::uvec(),
                                       const std::vector<dtype_arma> *im = nullptr)
{
    using dtype = typename dtype_arma::elem_type;
    if (stack)
        return qd_python_stack2numpy<dtype, dtype_numpy>(re, im, i_vec);
    return qd_python_copy2list<dtype_arma, dtype_numpy>(re, im, i_vec);
}

// Convert a single channel object to the Python channel dict
static inline py::dict qd_python_channel2dict(const quadriga_lib::channel<double> &channel,
                                       const arma::uvec &snap = arma::uvec(),
                                       bool validate = false,
                                       bool stack = false)
{
    if (validate)
    {
        auto error_msg = channel.is_valid();
        if (!error_msg.empty())
            throw std::invalid_argument(error_msg.c_str());
    }

    const bool snap_given = !snap.is_empty();
    const arma::uword n_snap_channel = channel.n_snap();

    // Resolve snapshot index: empty -> all; otherwise validate bounds
    arma::uvec i_snap;
    if (n_snap_channel == 0)
        i_snap.reset();
    else if (!snap_given)
    {
        i_snap.set_size(n_snap_channel);
        for (arma::uword s = 0; s < n_snap_channel; ++s)
            i_snap.at(s) = s;
    }
    else
    {
        for (arma::uword s : snap)
            if (s >= n_snap_channel)
                throw std::invalid_argument("Snapshot index out of bound.");
        i_snap = snap;
    }

    py::dict output;

    // Positions / orientations: (3, 1) shared or (3, n_snap) per-snapshot
    auto emit_per_snap_mat = [&](const char *key, const arma::Mat<double> &matrix)
    {
        if (matrix.n_cols == 1 || (matrix.n_cols > 1 && !snap_given))
            output[key] = qd_python_copy2numpy(&matrix);
        else if (snap_given && matrix.n_cols > 1)
            output[key] = qd_python_copy2numpy<double, double>(&matrix, nullptr, i_snap);
    };

    output["name"] = channel.name;

    emit_per_snap_mat("rx_pos", channel.rx_pos);
    emit_per_snap_mat("tx_pos", channel.tx_pos);
    emit_per_snap_mat("rx_orientation", channel.rx_orientation);
    emit_per_snap_mat("tx_orientation", channel.tx_orientation);

    // Coefficients (complex) and delays
    if (!channel.coeff_re.empty())
        output["coeff"] = qd_python_emit_paths<std::complex<double>>(&channel.coeff_re, stack, i_snap, &channel.coeff_im);
    if (!channel.delay.empty())
        output["delay"] = qd_python_emit_paths<double>(&channel.delay, stack, i_snap);

    if (!channel.path_gain.empty())
        output["path_gain"] = qd_python_emit_paths<double>(&channel.path_gain, stack, i_snap);
    if (!channel.path_length.empty())
        output["path_length"] = qd_python_emit_paths<double>(&channel.path_length, stack, i_snap);
    if (!channel.polmat.empty())
        output["polmat"] = qd_python_emit_paths<double>(&channel.polmat, stack, i_snap);
    if (!channel.path_angles.empty())
        output["path_angles"] = qd_python_emit_paths<double>(&channel.path_angles, stack, i_snap);

    if (!channel.fbs_pos.empty())
        output["fbs_pos"] = qd_python_emit_paths<double>(&channel.fbs_pos, stack, i_snap);
    if (!channel.lbs_pos.empty())
        output["lbs_pos"] = qd_python_emit_paths<double>(&channel.lbs_pos, stack, i_snap);

    if (!channel.n_interact.empty())
        output["n_interact"] = qd_python_emit_paths<py::ssize_t>(&channel.n_interact, stack, i_snap);
    if (!channel.interact_coord.empty())
        output["interact_coord"] = qd_python_emit_paths<double>(&channel.interact_coord, stack, i_snap);

    // Center frequency: scalar shared or per-snapshot
    if (channel.center_freq.n_elem == 1 || (channel.center_freq.n_elem > 1 && !snap_given))
        output["center_freq"] = qd_python_copy2numpy(&channel.center_freq);
    else if (snap_given && channel.center_freq.n_elem > 1)
        output["center_freq"] = qd_python_copy2numpy<double, double>(&channel.center_freq, nullptr, i_snap);

    if (n_snap_channel > 0)
    {
        output["initial_position"] = channel.initial_position;
    }

    return output;
}

static inline py::list qd_python_channel2list(const std::vector<quadriga_lib::channel<double>> &chan,
                                       bool validate = false,
                                       bool stack = false)
{
    py::list list;
    for (const auto &channel : chan)
        list.append(qd_python_channel2dict(channel, {}, validate, stack));
    return list;
}

// Python channel dict -> channel (inverse of qd_python_channel2dict, same keys)
// - Per-snapshot members, detected per key: a list of per-snapshot arrays ("list" model, n_path may vary) or one array
//   with the snapshot index on the last axis ("stack" model); positions, orientations and center_freq are plain arrays
// - The unstructured 'par' data is not read here (hdf5_write_channel reads it separately)
static inline quadriga_lib::channel<double> qd_python_dict2channel(const py::dict &dict,
                                                            bool validate = false)
{
    quadriga_lib::channel<double> c;

    auto has = [&](const char *key) -> bool
    {
        return dict.contains(key) && !dict[key].is_none();
    };

    // Loads one member if its key is present and not None
    auto load = [&](const char *key, auto &member)
    {
        if (!has(key))
            return;
        py::object v = dict[key];
        const bool is_list = py::isinstance<py::list>(v);
        using M = std::decay_t<decltype(member)>;
        if constexpr (std::is_same_v<M, arma::Mat<double>>)
            member = qd_python_numpy2arma_Mat<double>(v, true);
        else if constexpr (std::is_same_v<M, arma::Col<double>>)
            member = qd_python_numpy2arma_Col<double>(v, true);
        else if constexpr (std::is_same_v<M, std::vector<arma::Cube<double>>>)
            member = is_list ? qd_python_list2vector_Cube<double>(py::reinterpret_borrow<py::list>(v))
                             : qd_python_numpy2arma_vecCube<double>(v, true);
        else if constexpr (std::is_same_v<M, std::vector<arma::Mat<double>>>)
            member = is_list ? qd_python_list2vector_Mat<double>(py::reinterpret_borrow<py::list>(v))
                             : qd_python_numpy2arma_vecMat<double>(v);
        else // std::vector<arma::Col<double or unsigned>>
        {
            using T = typename M::value_type::elem_type;
            member = is_list ? qd_python_list2vector_Col<T>(py::reinterpret_borrow<py::list>(v))
                             : qd_python_numpy2arma_vecCol<T>(v);
        }
    };

    if (has("name"))
        c.name = dict["name"].cast<std::string>();
    if (has("initial_position"))
        c.initial_position = dict["initial_position"].cast<decltype(c.initial_position)>();

    load("tx_pos", c.tx_pos);
    load("rx_pos", c.rx_pos);
    load("tx_orientation", c.tx_orientation);
    load("rx_orientation", c.rx_orientation);
    load("center_freq", c.center_freq);

    // Coefficients: one complex key "coeff", or the two real keys "coeff_re" / "coeff_im", not both
    if (has("coeff") && (has("coeff_re") || has("coeff_im")))
        throw std::invalid_argument("Channel dict: provide either 'coeff' (complex) or 'coeff_re'/'coeff_im' (real), not both.");
    if (has("coeff"))
    {
        py::object v = dict["coeff"];
        if (py::isinstance<py::list>(v))
            qd_python_list2vector_Cube_Cplx<double>(py::reinterpret_borrow<py::list>(v), c.coeff_re, c.coeff_im);
        else
            qd_python_numpy2arma_vecCube_Cplx<double>(v, c.coeff_re, c.coeff_im);
    }
    load("coeff_re", c.coeff_re);
    load("coeff_im", c.coeff_im);

    load("delay", c.delay);
    load("path_gain", c.path_gain);
    load("path_length", c.path_length);
    load("polmat", c.polmat);                       // real interleaved (8, n_path) as in C++
    load("path_angles", c.path_angles);
    load("fbs_pos", c.fbs_pos);
    load("lbs_pos", c.lbs_pos);
    load("n_interact", c.n_interact);
    load("interact_coord", c.interact_coord);

    // Prune zero-padded trailing columns in interact_coord (stacked layout, as in the MEX struct reader)
    const size_t n_snap = (size_t)c.n_snap();
    if (has("interact_coord") && !py::isinstance<py::list>(dict["interact_coord"]) &&
        c.n_interact.size() == n_snap && c.interact_coord.size() == n_snap)
        for (size_t s = 0; s < n_snap; ++s)
        {
            const arma::uword cnt = (arma::uword)arma::accu(arma::conv_to<arma::uvec>::from(c.n_interact[s]));
            if (c.interact_coord[s].n_cols > cnt)
                c.interact_coord[s].resize(c.interact_coord[s].n_rows, cnt);
        }

    if (validate && !c.empty())
    {
        const std::string err = c.is_valid();
        if (!err.empty())
            throw std::invalid_argument(err);
    }

    return c;
}

// List of channel dicts -> channels (inverse of qd_python_channel2list); a single dict gives one channel, None none;
// each dict is read by qd_python_dict2channel
static inline std::vector<quadriga_lib::channel<double>> qd_python_list2channel(const py::handle &obj,
                                                                         bool validate = false)
{
    std::vector<quadriga_lib::channel<double>> channels;

    if (obj.is_none())
        return channels;

    if (py::isinstance<py::dict>(obj))
    {
        channels.push_back(qd_python_dict2channel(py::reinterpret_borrow<py::dict>(obj), validate));
        return channels;
    }

    if (!py::isinstance<py::list>(obj))
        throw std::invalid_argument("Channel input must be a dict or a list of dicts.");

    py::list list = py::reinterpret_borrow<py::list>(obj);
    channels.reserve(list.size());
    for (py::handle item : list)
    {
        if (!py::isinstance<py::dict>(item))
            throw std::invalid_argument("Each channel list entry must be a dict.");
        channels.push_back(qd_python_dict2channel(py::reinterpret_borrow<py::dict>(item), validate));
    }

    return channels;
}

// Convert an unstructured dataset (std::any) to a numpy array / scalar / str.
// Missing dataset -> None. Stored type and shape are preserved.
static inline py::object qd_python_any2numpy(const std::any &dset)
{
    if (!dset.has_value())
        return py::none();

    int type_id = quadriga_lib::any_type_id(&dset);

#define QD_PY_ANY_ARR(id, T) \
    case id:                 \
        return qd_python_copy2numpy(std::any_cast<T>(&dset));

    switch (type_id)
    {
    case 9: // std::string
        return py::cast(std::any_cast<std::string>(dset));

    // Scalars -> Python numbers
    case 10:
        return py::cast(std::any_cast<float>(dset));
    case 11:
        return py::cast(std::any_cast<double>(dset));
    case 12:
        return py::cast(std::any_cast<unsigned long long>(dset));
    case 13:
        return py::cast(std::any_cast<long long>(dset));
    case 14:
        return py::cast(std::any_cast<unsigned>(dset));
    case 15:
        return py::cast(std::any_cast<int>(dset));

        // Matrices
        QD_PY_ANY_ARR(20, arma::Mat<float>)
        QD_PY_ANY_ARR(21, arma::Mat<double>)
        QD_PY_ANY_ARR(22, arma::Mat<unsigned long long>)
        QD_PY_ANY_ARR(23, arma::Mat<long long>)
        QD_PY_ANY_ARR(24, arma::Mat<unsigned>)
        QD_PY_ANY_ARR(25, arma::Mat<int>)

        // Cubes
        QD_PY_ANY_ARR(30, arma::Cube<float>)
        QD_PY_ANY_ARR(31, arma::Cube<double>)
        QD_PY_ANY_ARR(32, arma::Cube<unsigned long long>)
        QD_PY_ANY_ARR(33, arma::Cube<long long>)
        QD_PY_ANY_ARR(34, arma::Cube<unsigned>)
        QD_PY_ANY_ARR(35, arma::Cube<int>)

        // Column vectors
        QD_PY_ANY_ARR(40, arma::Col<float>)
        QD_PY_ANY_ARR(41, arma::Col<double>)
        QD_PY_ANY_ARR(42, arma::Col<unsigned long long>)
        QD_PY_ANY_ARR(43, arma::Col<long long>)
        QD_PY_ANY_ARR(44, arma::Col<unsigned>)
        QD_PY_ANY_ARR(45, arma::Col<int>)

        // Row vectors
        QD_PY_ANY_ARR(50, arma::Row<float>)
        QD_PY_ANY_ARR(51, arma::Row<double>)
        QD_PY_ANY_ARR(52, arma::Row<unsigned long long>)
        QD_PY_ANY_ARR(53, arma::Row<long long>)
        QD_PY_ANY_ARR(54, arma::Row<unsigned>)
        QD_PY_ANY_ARR(55, arma::Row<int>)

    default:
        throw std::invalid_argument("Unsupported dataset type.");
    }

#undef QD_PY_ANY_ARR
}

// -------------------------------- Flags and paths --------------------------------

// Python flag -> bool
// - None gives 'default_value'; Python bool and numpy.bool_ give their value
// - Anything else (also int and str) throws py::type_error naming the input 'name'
static inline bool qd_python_flag(const py::handle &value, bool default_value, const std::string &name)
{
    if (value.is_none())
        return default_value;
    if (py::isinstance<py::bool_>(value) || py::isinstance(value, py::module_::import("numpy").attr("bool_")))
    {
        const int res = PyObject_IsTrue(value.ptr());
        if (res < 0)
            throw py::error_already_set();
        return res == 1;
    }
    throw py::type_error("Input '" + name + "' must be a bool or None.");
}

// Python path -> std::string; returns false if 'value' is not a path
// - str and bytes as they are; an object with '__fspath__' is converted with os.fspath first
static inline bool qd_python_try_path(const py::handle &value, std::string &path)
{
    py::object v = py::reinterpret_borrow<py::object>(value);
    if (!py::isinstance<py::str>(v) && !py::isinstance<py::bytes>(v) && py::hasattr(v, "__fspath__"))
        v = py::module_::import("os").attr("fspath")(v);
    if (!py::isinstance<py::str>(v) && !py::isinstance<py::bytes>(v))
        return false;
    path = py::cast<std::string>(v);
    return true;
}

// Python path (str, bytes or os.PathLike) -> std::string; anything else throws py::type_error naming the input 'name'
static inline std::string qd_python_path(const py::handle &value, const std::string &name)
{
    std::string path;
    if (!qd_python_try_path(value, path))
        throw py::type_error("Input '" + name + "' must be a str or path-like object.");
    return path;
}

// -------------------------------- Scenario parameters --------------------------------

// Read a dict value as a C-contiguous double array
// - Accepts Python numbers and numpy scalars or arrays of the kinds bool, int, uint and float
// - str, None, complex values and anything numpy cannot convert throw std::invalid_argument with 'error_msg'
static inline py::array_t<double, py::array::c_style | py::array::forcecast | py::detail::npy_api::NPY_ARRAY_ALIGNED_> qd_python_real_array(const py::handle &value,
                                                                                          const std::string &error_msg)
{
    py::array arr = py::array::ensure(value);
    if (!arr)
        throw std::invalid_argument(error_msg);
    const char kind = arr.dtype().kind();
    if (kind != 'b' && kind != 'i' && kind != 'u' && kind != 'f')
        throw std::invalid_argument(error_msg);
    return py::array_t<double, py::array::c_style | py::array::forcecast | py::detail::npy_api::NPY_ARRAY_ALIGNED_>::ensure(arr);
}

// Dict of QuaDRiGa scenario parameters -> map of the numeric values
// - Every entry except 'SubpathMethod' must be a real scalar (one element)
// - 'SubpathMethod' (an entry of QuaDRiGa's structs) must be the str 'legacy'; it is accepted and ignored
// - Key names are not checked here, quadriga_lib::scenpar_validate rejects unknown names
// - None or an empty dict gives an empty map
// - 'var_name' is used in the error messages
static inline std::unordered_map<std::string, double> qd_python_dict2scenpar(const py::handle &obj,
                                                                      const std::string &var_name = "scenpar")
{
    std::unordered_map<std::string, double> scenpar;
    if (obj.is_none())
        return scenpar;
    if (!py::isinstance<py::dict>(obj))
        throw std::invalid_argument("Input '" + var_name + "' must be a dict or None.");

    for (auto item : py::reinterpret_borrow<py::dict>(obj))
    {
        if (!py::isinstance<py::str>(item.first))
            throw std::invalid_argument("Keys of '" + var_name + "' must be str.");
        const std::string name = py::cast<std::string>(item.first);
        if (name == "SubpathMethod")
        {
            if (!py::isinstance<py::str>(item.second))
                throw std::invalid_argument("Entry 'SubpathMethod' of '" + var_name + "' must be a str.");
            const std::string value = py::cast<std::string>(item.second);
            if (value != "legacy")
                throw std::invalid_argument("Value '" + value + "' of 'SubpathMethod' is not supported; only 'legacy' is accepted.");
        }
        else
        {
            const std::string error_msg = "Entry '" + name + "' of '" + var_name + "' must be a real scalar.";
            const auto values = qd_python_real_array(item.second, error_msg);
            if (values.size() != 1)
                throw std::invalid_argument(error_msg);
            scenpar[name] = *values.data();
        }
    }
    return scenpar;
}

// Map of QuaDRiGa scenario parameters -> dict of float values, keys in the order of 'key_order'
// - Every name in 'key_order' must be in 'scenpar'
static inline py::dict qd_python_scenpar2dict(const std::unordered_map<std::string, double> &scenpar,
                                       const std::vector<std::string> &key_order)
{
    py::dict output;
    for (const auto &key : key_order)
        output[py::str(key)] = scenpar.at(key);
    return output;
}

// Dict of QuaDRiGa path-loss parameters -> map of the numeric values, the model and the key order
// - The entry 'model' must be a str; it is returned in 'pl_model' (empty if the dict has no such entry)
// - Every other entry must be a real scalar, list or 1D array
// - 'pl_order' lists the numeric entries in dict order
// - None or an empty dict gives an empty map, empty 'pl_model' and empty 'pl_order'
// - 'var_name' is used in the error messages
static inline std::unordered_map<std::string, std::vector<double>> qd_python_dict2plpar(const py::handle &obj,
                                                                                std::string &pl_model,
                                                                                std::vector<std::string> &pl_order,
                                                                                const std::string &var_name = "plpar")
{
    std::unordered_map<std::string, std::vector<double>> plpar;
    pl_model.clear();
    pl_order.clear();
    if (obj.is_none())
        return plpar;
    if (!py::isinstance<py::dict>(obj))
        throw std::invalid_argument("Input '" + var_name + "' must be a dict or None.");

    for (auto item : py::reinterpret_borrow<py::dict>(obj))
    {
        if (!py::isinstance<py::str>(item.first))
            throw std::invalid_argument("Keys of '" + var_name + "' must be str.");
        const std::string name = py::cast<std::string>(item.first);
        if (name == "model")
        {
            if (!py::isinstance<py::str>(item.second))
                throw std::invalid_argument("Entry 'model' of '" + var_name + "' must be a str.");
            pl_model = py::cast<std::string>(item.second);
        }
        else
        {
            const std::string error_msg = "Entry '" + name + "' of '" + var_name +
                                          "' must be a real scalar or 1D array.";
            const auto values = qd_python_real_array(item.second, error_msg);
            if (values.ndim() > 1)
                throw std::invalid_argument(error_msg);
            plpar[name] = std::vector<double>(values.data(), values.data() + values.size());
            pl_order.push_back(name);
        }
    }
    return plpar;
}

// QuaDRiGa path-loss parameters -> dict, empty if 'pl_model' and 'pl_order' are both empty
// - Entry 'model' (str) first if 'pl_model' is not empty, then one 1D array per key in the order of 'pl_order';
//   every name in 'pl_order' must be in 'plpar'
static inline py::dict qd_python_plpar2dict(const std::string &pl_model,
                                     const std::unordered_map<std::string, std::vector<double>> &plpar,
                                     const std::vector<std::string> &pl_order)
{
    py::dict output;
    if (!pl_model.empty())
        output["model"] = pl_model;
    for (const auto &key : pl_order)
    {
        const arma::vec values(plpar.at(key));
        output[py::str(key)] = qd_python_copy2numpy(&values);
    }
    return output;
}

// -------------------------------- gbsm_rays dict --------------------------------

// Keys of the gbsm_rays dict in their order; the first qd_python_gbsm_rays_n_required are required as input, the
// others optional (missing or None = empty), also any key appended later
static const char *const qd_python_gbsm_rays_keys[] = {
    "method", "tx_pos", "rx_pos", "center_freq", "delay", "power", "aod", "eod", "aoa",
    "eoa", "polmat", "path_type", "tap", "gr_eps", "fbs_pos", "lbs_pos", "fallback"};
static constexpr size_t qd_python_gbsm_rays_n_required = 13; // "method" to "tap"

// Labels, codes or counts (non-negative integers, no index shift) -> u32_vec; None gives an empty vector
// - A uint32 ndarray is aliased (read-only rules of qd_python_numpy2arma_Col); anything else is read as float (exact
//   below 2^53) and every entry must be an integer from 0 to 2^32 - 1, else std::invalid_argument naming 'name'
static inline arma::u32_vec qd_python_u32_entries(const py::handle &obj, const std::string &name)
{
    if (py::isinstance<py::array_t<unsigned>>(obj))
        return qd_python_numpy2arma_Col<unsigned>(obj, true);
    const arma::vec in = qd_python_numpy2arma_Col<double>(obj, true);
    arma::u32_vec out(in.n_elem);
    for (arma::uword i = 0; i < in.n_elem; ++i)
    {
        if (!(in[i] >= 0.0 && in[i] < 4294967296.0 && in[i] == std::floor(in[i])))
            throw std::invalid_argument("Entries in '" + name + "' must be integers from 0 to 4294967295.");
        out[i] = (unsigned)in[i];
    }
    return out;
}

// Dict -> gbsm_rays, aliasing the numpy memory where dtype and layout allow (read-only)
// - 'gr_eps' (complex, a real array = imaginary part 0) is copied into 'gr_eps_re' and 'gr_eps_im'
// - Missing optional keys mean empty; a missing other key or an unknown key throws, naming 'var_name'
// - Validation is left to is_valid()
static inline quadriga_lib::gbsm_rays qd_python_dict2gbsm_rays(const py::dict &d, const std::string &var_name = "rays")
{
    const auto *K = qd_python_gbsm_rays_keys, *K_end = K + std::size(qd_python_gbsm_rays_keys);
    for (auto item : d)
    {
        const std::string key = py::cast<std::string>(py::str(item.first));
        if (std::find_if(K, K_end, [&](const char *k) { return key == k; }) == K_end)
            throw std::invalid_argument("Entry '" + key + "' of '" + var_name + "' is unknown.");
    }
    for (size_t k = 0; k < qd_python_gbsm_rays_n_required; ++k)
        if (!d.contains(K[k]))
            throw std::invalid_argument("Entry '" + std::string(K[k]) + "' of '" + var_name + "' is missing.");
    auto get = [&](const char *k) -> py::object
    { return d.contains(k) ? py::object(d[k]) : py::object(py::none()); };

    quadriga_lib::gbsm_rays r;
    const py::object method = get("method");
    if (!py::isinstance<py::str>(method))
        throw std::invalid_argument("Entry 'method' of '" + var_name + "' must be a str.");
    r.method = py::cast<std::string>(method);
    r.tx_pos = qd_python_numpy2arma_Col<double>(get("tx_pos"), true);
    r.rx_pos = qd_python_numpy2arma_Col<double>(get("rx_pos"), true);
    r.center_freq = qd_python_numpy2arma_Col<double>(get("center_freq"), true);
    r.delay = qd_python_numpy2arma_Col<double>(get("delay"), true);
    r.power = qd_python_numpy2arma_Mat<double>(get("power"), true);
    r.aod = qd_python_numpy2arma_Col<double>(get("aod"), true);
    r.eod = qd_python_numpy2arma_Col<double>(get("eod"), true);
    r.aoa = qd_python_numpy2arma_Col<double>(get("aoa"), true);
    r.eoa = qd_python_numpy2arma_Col<double>(get("eoa"), true);
    r.polmat = qd_python_numpy2arma_Cube<double>(get("polmat"), true);
    r.path_type = qd_python_u32_entries(get("path_type"), var_name + ".path_type");
    r.tap = qd_python_u32_entries(get("tap"), var_name + ".tap");
    r.fbs_pos = qd_python_numpy2arma_Mat<double>(get("fbs_pos"), true);
    r.lbs_pos = qd_python_numpy2arma_Mat<double>(get("lbs_pos"), true);
    r.fallback = qd_python_u32_entries(get("fallback"), var_name + ".fallback");
    const py::object gr_eps = get("gr_eps");
    if (!gr_eps.is_none())
    {
        arma::cube re, im;
        qd_python_copy2arma(py::cast<py::array_t<std::complex<double>>>(gr_eps), re, im);
        r.gr_eps_re = arma::vectorise(re), r.gr_eps_im = arma::vectorise(im);
    }
    return r;
}

// Creates a list of n_link gbsm_rays dicts with all keys and aliases the attributes of 'rays' (resized to n_link)
// whose size is known, as qd_mex_init_gbsm_rays; with 'has_gr', 'gr_eps_re' and 'gr_eps_im' are allocated in C++
// - The other keys are empty arrays ('gr_eps' complex, 'fallback' uint32); the caller sets 'method' and, with
//   'has_gr', 'gr_eps' after the call
static inline py::list qd_python_init_gbsm_rays(arma::uword n_link, arma::uword n_ray, arma::uword n_freq, bool has_gr,
                                         bool has_scatterers, std::vector<quadriga_lib::gbsm_rays> &rays)
{
    py::list out;
    rays.clear();
    rays.resize(n_link);
    const arma::uword n3 = 3, n8 = 8;
    for (arma::uword l = 0; l < n_link; ++l)
    {
        auto &r = rays[l];
        py::dict d;
        d["method"] = "";
        d["tx_pos"] = qd_python_init_output(n3, 1, &r.tx_pos).reshape({n3}); // (3,) view of the [3, 1] member
        d["rx_pos"] = qd_python_init_output(n3, 1, &r.rx_pos).reshape({n3});
        d["center_freq"] = qd_python_init_output(n_freq, &r.center_freq);
        d["delay"] = qd_python_init_output(n_ray, &r.delay);
        d["power"] = qd_python_init_output(n_ray, n_freq, &r.power);
        d["aod"] = qd_python_init_output(n_ray, &r.aod);
        d["eod"] = qd_python_init_output(n_ray, &r.eod);
        d["aoa"] = qd_python_init_output(n_ray, &r.aoa);
        d["eoa"] = qd_python_init_output(n_ray, &r.eoa);
        d["polmat"] = qd_python_init_output(n8, n_ray, n_freq, &r.polmat);
        d["path_type"] = qd_python_init_output(n_ray, &r.path_type);
        d["tap"] = qd_python_init_output(n_ray, &r.tap);
        d["gr_eps"] = py::array_t<std::complex<double>>(0);
        if (has_gr)
            r.gr_eps_re.set_size(n_freq), r.gr_eps_im.set_size(n_freq);
        if (has_scatterers)
        {
            d["fbs_pos"] = qd_python_init_output(n3, n_ray, &r.fbs_pos);
            d["lbs_pos"] = qd_python_init_output(n3, n_ray, &r.lbs_pos);
        }
        else
            d["fbs_pos"] = py::array_t<double>(0), d["lbs_pos"] = py::array_t<double>(0);
        d["fallback"] = py::array_t<unsigned>(0);
        out.append(d);
    }
    return out;
}

// -------------------------------- Shipped conf folder --------------------------------

// Folder of the shipped conf files, found at run time relative to the module folder: ../data/config (local lib/ build),
// ../share/quadriga-lib/config (CMake install), ../../../share/quadriga-lib/config (POSIX site-packages, e.g. venv,
// --user, dist-packages), ../../share/quadriga-lib/config (Windows); the folder compiled into the library if the module
// path is unknown or none holds the files
static inline std::string qd_python_conf_dir()
{
    std::string file;
    try
    {
        file = py::module_::import("quadriga_lib").attr("__file__").cast<std::string>();
    }
    catch (const py::error_already_set &)
    {
    }
    return qd_conf_shipped_dir(file, {"../data/config", "../share/quadriga-lib/config",
                                      "../../../share/quadriga-lib/config", "../../share/quadriga-lib/config"});
}

// -------------------------------- Layout dicts --------------------------------

// -------------------------------- Reading helpers --------------------------------

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

// -------------------------------- layout_nodes --------------------------------

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

// -------------------------------- layout_links --------------------------------

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

// -------------------------------- layout --------------------------------

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

#undef QD_PL_NODES
#undef QD_PL_LINKS
#undef QD_PL_LAYOUT
#undef QD_PL_NAME
#undef QD_PL_READ_0
#undef QD_PL_READ_1
#undef QD_PL_READ_
#undef QD_PL_WRITE_0
#undef QD_PL_WRITE_1
#undef QD_PL_WRITE_
#undef QD_PL_READ
#undef QD_PL_WRITE

// -------------------------------- gbsm_rays --------------------------------

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

// -------------------------------- Arguments --------------------------------

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
static inline std::vector<std::string> qd_pl_config_dirs(const py::handle &h)
{
    std::vector<std::string> dirs = qd_python_list2vector_Strings(h);
    const std::string shipped = qd_python_conf_dir();
    if (!shipped.empty())
        dirs.push_back(shipped);
    return dirs;
}

#endif
