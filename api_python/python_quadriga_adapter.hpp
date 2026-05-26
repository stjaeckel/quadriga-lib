// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef quadriga_python_quadriga_adapter_H
#define quadriga_python_quadriga_adapter_H

#include "quadriga_lib.hpp"
#include "python_arma_adapter.hpp"

namespace py = pybind11;

static py::dict qd_python_arrayant2dict(const quadriga_lib::arrayant<double> &ant)
{
    py::dict output;
    output["e_theta_re"] = qd_python_copy2numpy(&ant.e_theta_re);
    output["e_theta_im"] = qd_python_copy2numpy(&ant.e_theta_im);
    output["e_phi_re"] = qd_python_copy2numpy(&ant.e_phi_re);
    output["e_phi_im"] = qd_python_copy2numpy(&ant.e_phi_im);
    output["azimuth_grid"] = qd_python_copy2numpy(&ant.azimuth_grid);
    output["elevation_grid"] = qd_python_copy2numpy(&ant.elevation_grid);
    output["element_pos"] = qd_python_copy2numpy(&ant.element_pos);
    output["coupling_re"] = qd_python_copy2numpy(&ant.coupling_re);
    output["coupling_im"] = qd_python_copy2numpy(&ant.coupling_im);
    output["center_freq"] = ant.center_frequency;
    output["name"] = ant.name;
    return output;
}

static py::dict qd_python_arrayant2dict_multi(const std::vector<quadriga_lib::arrayant<double>> &ant)
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
        center_freqs[f] = ant[f].center_frequency;
    output["center_freq"] = qd_python_copy2numpy(&center_freqs);

    // Coupling: check if all entries are identical
    bool coupling_varies = false;
    for (size_t f = 1; f < n_freq && !coupling_varies; ++f)
    {
        if (ant[f].coupling_re.n_elem != a0.coupling_re.n_elem ||
            ant[f].coupling_im.n_elem != a0.coupling_im.n_elem)
            coupling_varies = true;
        else
        {
            if (a0.coupling_re.n_elem > 0 && !arma::approx_equal(ant[f].coupling_re, a0.coupling_re, "absdiff", 1e-15))
                coupling_varies = true;
            if (a0.coupling_im.n_elem > 0 && !arma::approx_equal(ant[f].coupling_im, a0.coupling_im, "absdiff", 1e-15))
                coupling_varies = true;
        }
    }

    if (coupling_varies)
    {
        // Per-frequency: store as 3D (n_el × n_ports × n_freq)
        if (a0.coupling_re.n_elem > 0)
        {
            arma::uword nr = a0.coupling_re.n_rows, nc = a0.coupling_re.n_cols;
            arma::Cube<double> cpl_re(nr, nc, n_freq, arma::fill::none);
            for (size_t f = 0; f < n_freq; ++f)
                cpl_re.slice(f) = ant[f].coupling_re;
            output["coupling_re"] = qd_python_copy2numpy(&cpl_re);
        }
        else
            output["coupling_re"] = qd_python_copy2numpy(&a0.coupling_re);

        if (a0.coupling_im.n_elem > 0)
        {
            arma::uword nr = a0.coupling_im.n_rows, nc = a0.coupling_im.n_cols;
            arma::Cube<double> cpl_im(nr, nc, n_freq, arma::fill::none);
            for (size_t f = 0; f < n_freq; ++f)
                cpl_im.slice(f) = ant[f].coupling_im;
            output["coupling_im"] = qd_python_copy2numpy(&cpl_im);
        }
        else
            output["coupling_im"] = qd_python_copy2numpy(&a0.coupling_im);
    }
    else
    {
        // Shared: store as 2D
        output["coupling_re"] = qd_python_copy2numpy(&a0.coupling_re);
        output["coupling_im"] = qd_python_copy2numpy(&a0.coupling_im);
    }

    return output;
}

static quadriga_lib::arrayant<double> qd_python_dict2arrayant(const py::dict &arrayant,
                                                              bool view = false, bool strict = false,
                                                              bool validate = false)
{
    quadriga_lib::arrayant<double> ant;

    // Mandatory fields
    ant.e_theta_re = qd_python_numpy2arma_Cube<double>(arrayant["e_theta_re"], view, strict);
    ant.e_theta_im = qd_python_numpy2arma_Cube<double>(arrayant["e_theta_im"], view, strict);
    ant.e_phi_re = qd_python_numpy2arma_Cube<double>(arrayant["e_phi_re"], view, strict);
    ant.e_phi_im = qd_python_numpy2arma_Cube<double>(arrayant["e_phi_im"], view, strict);
    ant.azimuth_grid = qd_python_numpy2arma_Col<double>(arrayant["azimuth_grid"], view, strict);
    ant.elevation_grid = qd_python_numpy2arma_Col<double>(arrayant["elevation_grid"], view, strict);

    arma::uword n_elements = ant.e_theta_re.n_slices;

    // Optional fields
    if (arrayant.contains("element_pos"))
        ant.element_pos = qd_python_numpy2arma_Mat<double>(arrayant["element_pos"], view, strict);
    else
        ant.element_pos.zeros(3, n_elements);

    if (arrayant.contains("coupling_re"))
        ant.coupling_re = qd_python_numpy2arma_Mat<double>(arrayant["coupling_re"], view, strict);
    else if (!arrayant.contains("coupling_im"))
        ant.coupling_re.eye(n_elements, n_elements);

    if (arrayant.contains("coupling_im"))
        ant.coupling_im = qd_python_numpy2arma_Mat<double>(arrayant["coupling_im"], view, strict);
    else
        ant.coupling_im.zeros(n_elements, n_elements);

    if (arrayant.contains("center_freq"))
        ant.center_frequency = arrayant["center_freq"].cast<double>();

    if (arrayant.contains("name"))
        ant.name = arrayant["name"].cast<std::string>();

    if (validate)
    {
        auto error_msg = ant.validate();
        if (!error_msg.empty())
            throw std::invalid_argument(error_msg.c_str());
    }

    return ant;
}

// Multi-frequency: dict → std::vector<arrayant>
static std::vector<quadriga_lib::arrayant<double>> qd_python_dict2arrayant_multi(const py::dict &arrayant,
                                                                                 bool view = false, bool strict = false,
                                                                                 bool validate = false)
{
    // Determine number of frequencies from e_theta_re shape
    py::array e_theta_re_arr = py::cast<py::array>(arrayant["e_theta_re"]);
    int nd = (int)e_theta_re_arr.request().ndim;

    // 1D/2D pattern fields describe a single-frequency antenna with implicit trailing dimensions.
    // Delegate to the single-frequency parser (qd_python_numpy2arma_Cube promotes them to 3D cubes) and return a length-1 vector.
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
        ant[f].center_frequency = center_freqs[f];
        ant[f].name = name;
        if (has_coupling_re)
            ant[f].coupling_re = std::move(coupling_re_vec[f]);
        if (has_coupling_im)
            ant[f].coupling_im = std::move(coupling_im_vec[f]);
    }

    if (validate)
    {
        auto error_msg = quadriga_lib::arrayant_is_valid_multi(ant, false);
        if (!error_msg.empty())
            throw std::invalid_argument(error_msg.c_str());
    }

    return ant;
}

// Emit a vector of per-snapshot arma objects either as a Python list of arrays
// (stack = false) or stacked into one higher-dimensional Numpy array along an
// appended snapshot axis (stack = true). i_vec selects / orders the snapshots.
template <typename dtype_numpy, typename dtype_arma>
static py::object qd_python_emit_paths(const std::vector<dtype_arma> *re,
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
static py::dict qd_python_channel2dict(const quadriga_lib::channel<double> &channel,
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

    emit_per_snap_mat("rx_position", channel.rx_pos);
    emit_per_snap_mat("tx_position", channel.tx_pos);
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
    if (!channel.path_polarization.empty())
        output["path_polarization"] = qd_python_emit_paths<std::complex<double>>(&channel.path_polarization, stack, i_snap);
    if (!channel.path_angles.empty())
        output["path_angles"] = qd_python_emit_paths<double>(&channel.path_angles, stack, i_snap);

    if (!channel.path_fbs_pos.empty())
        output["fbs_pos"] = qd_python_emit_paths<double>(&channel.path_fbs_pos, stack, i_snap);
    if (!channel.path_lbs_pos.empty())
        output["lbs_pos"] = qd_python_emit_paths<double>(&channel.path_lbs_pos, stack, i_snap);

    if (!channel.no_interact.empty())
        output["no_interact"] = qd_python_emit_paths<py::ssize_t>(&channel.no_interact, stack, i_snap);
    if (!channel.interact_coord.empty())
        output["interact_coord"] = qd_python_emit_paths<double>(&channel.interact_coord, stack, i_snap);

    // Center frequency: scalar shared or per-snapshot
    if (channel.center_frequency.n_elem == 1 || (channel.center_frequency.n_elem > 1 && !snap_given))
        output["center_frequency"] = qd_python_copy2numpy(&channel.center_frequency);
    else if (snap_given && channel.center_frequency.n_elem > 1)
        output["center_frequency"] = qd_python_copy2numpy<double, double>(&channel.center_frequency, nullptr, i_snap);

    if (n_snap_channel > 0)
    {
        output["initial_position"] = channel.initial_position;
    }

    return output;
}

static py::list qd_python_channel2list(const std::vector<quadriga_lib::channel<double>> &chan,
                                       bool validate = false,
                                       bool stack = false)
{
    py::list list;
    for (const auto &channel : chan)
        list.append(qd_python_channel2dict(channel, {}, validate, stack));
    return list;
}

// Convert an unstructured dataset (std::any) to a numpy array / scalar / str.
// Missing dataset -> None. Stored type and shape are preserved.
static py::object qd_python_any2numpy(const std::any &dset)
{
    if (!dset.has_value())
        return py::none();

    unsigned long long dims[3];
    void *dataptr = nullptr;
    int type_id = quadriga_lib::any_type_id(&dset, dims, &dataptr);

#define QD_PY_ANY_ARR(id, T) \
    case id:                 \
        return qd_python_copy2numpy(std::any_cast<T>(&dset));

    switch (type_id)
    {
    case 9: // std::string
        return py::cast(std::any_cast<std::string>(dset));

    // Scalars -> Python numbers
    case 10:
        return py::cast(*(float *)dataptr);
    case 11:
        return py::cast(*(double *)dataptr);
    case 12:
        return py::cast(*(unsigned long long *)dataptr);
    case 13:
        return py::cast(*(long long *)dataptr);
    case 14:
        return py::cast(*(unsigned *)dataptr);
    case 15:
        return py::cast(*(int *)dataptr);

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

#endif