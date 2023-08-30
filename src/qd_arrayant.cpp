// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
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

#include <stdexcept>
#include <cstring> // For std::memcopy
#include "quadriga_lib.hpp"
#include "quadriga_tools.hpp"
#include "qd_arrayant_qdant.hpp"
#include "qd_arrayant_interpolate.hpp"

#include "helper_functions.cpp"

// ARRAYANT METHODS : Return number of elevation angles, azimuth angles and elemets
template <typename dtype>
unsigned quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::n_elevation() const
{
    return unsigned(e_theta_re.n_rows);
}
template <typename dtype>
unsigned quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::n_azimuth() const
{
    return unsigned(e_theta_re.n_cols);
}
template <typename dtype>
unsigned quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::n_elements() const
{
    return unsigned(e_theta_re.n_slices);
}
template <typename dtype>
unsigned quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::n_ports() const
{
    if (coupling_re.empty() && coupling_im.empty())
        return unsigned(e_theta_re.n_slices);
    else if (coupling_re.empty())
        return unsigned(coupling_im.n_cols);
    else
        return unsigned(coupling_re.n_cols);
}

// ARRAYANT METHOD : Calculate the directivity of an antenna element in dBi
template <typename dtype>
dtype quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::calc_directivity_dBi(unsigned element) const
{
    // Check if arrayant object is valid
    std::string error_message = is_valid();
    if (error_message.length() == 0 && element >= n_elements())
        error_message = "Element index out of bound.";
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Constants
    double pi2 = 2.0 * arma::datum::pi, pi_half = 0.5 * arma::datum::pi;
    arma::uword naz = azimuth_grid.n_elem, nel = elevation_grid.n_elem;

    // Az and El grid as double
    arma::vec az = arma::conv_to<arma::vec>::from(azimuth_grid);
    arma::vec el = arma::conv_to<arma::vec>::from(elevation_grid);

    // Calculate the azimuth weights
    arma::vec waz(naz, arma::fill::none);
    double *ptr = waz.memptr();
    for (arma::uword i = 0; i < naz; i++)
    {
        double x = i == 0 ? az.at(naz - 1) - pi2 : az.at(i - 1);
        double y = i == naz - 1 ? az.at(0) + pi2 : az.at(i + 1);
        ptr[i] = y - x;
    }

    // Calculate the elevation weights
    arma::vec wel(nel, arma::fill::none);
    ptr = wel.memptr();
    for (arma::uword i = 0; i < nel; i++)
    {
        double x = i == 0 ? -pi_half : 0.5 * el.at(i - 1) + 0.5 * el.at(i);
        double y = i == nel - 1 ? pi_half : 0.5 * el.at(i) + 0.5 * elevation_grid.at(i + 1);
        arma::vec tmp = arma::linspace<arma::vec>(x, y, 21);
        double val = arma::accu(arma::cos(tmp));
        ptr[i] = val * (y - x);
    }

    // Combine weights
    arma::mat W(nel, naz, arma::fill::none);
    ptr = W.memptr();
    double norm = 0.0;
    for (double *col = waz.begin(); col != waz.end(); col++)
        for (double *row = wel.begin(); row != wel.end(); row++)
            *ptr = *row * *col, norm += *ptr++;
    ptr = W.memptr();
    norm = 1.0 / norm;
    for (arma::uword i = 0; i < naz * nel; i++)
        ptr[i] *= norm;

    // Calculate the directivity
    double p_sum = 0.0, p_max = 0.0;
    const dtype *p_theta_re = e_theta_re.memptr(), *p_theta_im = e_theta_im.memptr();
    const dtype *p_phi_re = e_phi_re.memptr(), *p_phi_im = e_phi_im.memptr();
    for (arma::uword i = 0; i < naz * nel; i++)
    {
        double a = double(p_theta_re[i]), b = double(p_theta_im[i]), c = double(p_phi_re[i]), d = double(p_phi_im[i]);
        double pow = a * a + b * b + c * c + d * d;
        p_max = pow > p_max ? pow : p_max;
        p_sum += pow * ptr[i];
    }

    double directivity = p_max < 1.0e-14 ? 0.0 : 10.0 * std::log10(p_max / p_sum);
    return dtype(directivity);
}

// Calculates a virtual pattern of the given array by applying coupling and element positions
template <typename dtype>
void quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::combine_pattern(quadriga_lib::arrayant<dtype> *output)
{
    // Check if arrayant object is valid
    std::string error_message = validate(); // Deep check
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    dtype pi = dtype(arma::datum::pi);
    dtype lambda = dtype(299792448.0) / center_frequency;
    dtype wave_no = 2.0 * pi / lambda;

    arma::uword n_el = e_theta_re.n_rows;
    arma::uword n_az = e_theta_re.n_cols;
    arma::uword n_out = e_theta_re.n_slices;
    arma::uword n_ang = n_el * n_az;
    arma::uword n_prt = coupling_re.n_cols;

    // Create list of angles for pattern interpolation
    arma::Mat<dtype> azimuth(1, n_ang, arma::fill::none);
    arma::Mat<dtype> elevation(1, n_ang, arma::fill::none);
    dtype *p_azimuth = azimuth.memptr(), *p_elevation = elevation.memptr(),
          *p_phi = azimuth_grid.memptr(), *p_theta = elevation_grid.memptr();

    for (arma::uword ia = 0; ia < n_az; ia++)
        for (arma::uword ie = 0; ie < n_el; ie++)
            *p_azimuth++ = p_phi[ia], *p_elevation++ = p_theta[ie];

    // Interpolate the pattern data
    arma::Col<unsigned> i_element = arma::linspace<arma::Col<unsigned>>(1, n_out, n_out);
    arma::Cube<dtype> orientation(3, 1, 1);
    arma::Mat<dtype> V_re(n_out, n_ang), V_im(n_out, n_ang), H_re(n_out, n_ang), H_im(n_out, n_ang), dist(n_out, n_ang);
    arma::Mat<dtype> EMPTY;

    qd_arrayant_interpolate(&e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                            &azimuth_grid, &elevation_grid, &azimuth, &elevation,
                            &i_element, &orientation, &element_pos,
                            &V_re, &V_im, &H_re, &H_im, &dist, &EMPTY, &EMPTY, &EMPTY);

    // Apply phase shift caused by element positions
    arma::Mat<std::complex<dtype>> phase(arma::cos(wave_no * dist), arma::sin(-wave_no * dist));
    arma::Mat<std::complex<dtype>> Vi(V_re, V_im), Hi(H_re, H_im);
    Vi = Vi % phase, Hi = Hi % phase;

    // Apply coupling
    arma::Mat<std::complex<dtype>> coupling(coupling_re, coupling_im);
    arma::Mat<std::complex<dtype>> Vo(n_ang, n_prt), Ho(n_ang, n_prt);
    for (arma::uword i = 0; i < n_out; i++)
    {
        arma::Col<std::complex<dtype>> vi = Vi.row(i).as_col(), hi = Hi.row(i).as_col();
        for (arma::uword o = 0; o < n_prt; o++)
        {
            std::complex<dtype> cpl = coupling.at(i, o);
            Vo.col(o) += vi * cpl, Ho.col(o) += hi * cpl;
        }
    }

    if (output != NULL) // Write output data
    {
        output->set_size(n_el, n_az, n_prt, n_prt);

        std::memcpy(output->azimuth_grid.memptr(), azimuth_grid.memptr(), n_az * sizeof(dtype));
        std::memcpy(output->elevation_grid.memptr(), elevation_grid.memptr(), n_el * sizeof(dtype));

        arma::Mat<dtype> cpy = arma::real(Vo);
        std::memcpy(output->e_theta_re.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        cpy = arma::imag(Vo);
        std::memcpy(output->e_theta_im.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        cpy = arma::real(Ho);
        std::memcpy(output->e_phi_re.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        cpy = arma::imag(Ho);
        std::memcpy(output->e_phi_im.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        output->element_pos.zeros();
        output->coupling_re.eye();
        output->coupling_im.zeros();
        output->center_frequency = center_frequency;

        // Set the data pointers for the quick check.
        output->check_ptr[0] = output->e_theta_re.memptr();
        output->check_ptr[1] = output->e_theta_im.memptr();
        output->check_ptr[2] = output->e_phi_re.memptr();
        output->check_ptr[3] = output->e_phi_im.memptr();
        output->check_ptr[4] = output->azimuth_grid.memptr();
        output->check_ptr[5] = output->elevation_grid.memptr();
        output->check_ptr[6] = output->element_pos.memptr();
        output->check_ptr[7] = output->coupling_re.memptr();
        output->check_ptr[8] = output->coupling_im.memptr();
    }
    else if (read_only)
    {
        error_message = "Cannot update read-only array antenna object inplace.";
        throw std::invalid_argument(error_message.c_str());
    }
    else // Update the properties of current array antenna object
    {
        arma::Mat<dtype> cpy = real(Vo);
        e_theta_re.set_size(n_el, n_az, n_prt);
        std::memcpy(e_theta_re.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        cpy = imag(Vo);
        e_theta_im.set_size(n_el, n_az, n_prt);
        std::memcpy(e_theta_im.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        cpy = real(Ho);
        e_phi_re.set_size(n_el, n_az, n_prt);
        std::memcpy(e_phi_re.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        cpy = imag(Ho);
        e_phi_im.set_size(n_el, n_az, n_prt);
        std::memcpy(e_phi_im.memptr(), cpy.memptr(), n_el * n_az * n_prt * sizeof(dtype));

        element_pos.zeros(3, n_prt);
        coupling_re.eye(n_prt, n_prt);
        coupling_im.zeros(n_prt, n_prt);

        // Set the data pointers for the quick check.
        check_ptr[0] = e_theta_re.memptr();
        check_ptr[1] = e_theta_im.memptr();
        check_ptr[2] = e_phi_re.memptr();
        check_ptr[3] = e_phi_im.memptr();
        check_ptr[4] = azimuth_grid.memptr();
        check_ptr[5] = elevation_grid.memptr();
        check_ptr[6] = element_pos.memptr();
        check_ptr[7] = coupling_re.memptr();
        check_ptr[8] = coupling_im.memptr();
    }
}

// Creates a copy of the array antenna object
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::copy() const
{
    quadriga_lib::arrayant<dtype> ant;

    ant.name = name;
    ant.e_theta_re = e_theta_re;
    ant.e_theta_im = e_theta_im;
    ant.e_phi_re = e_phi_re;
    ant.e_phi_im = e_phi_im;
    ant.azimuth_grid = azimuth_grid;
    ant.elevation_grid = elevation_grid;
    ant.element_pos = element_pos;
    ant.coupling_re = coupling_re;
    ant.coupling_im = coupling_im;
    ant.center_frequency = center_frequency;
    ant.read_only = false;

    return ant;
}

// Copy antenna elements, enlarge array size if needed
template <typename dtype>
void quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::copy_element(unsigned source,
                                                                       arma::Col<unsigned> destination)
{
    // Check if arrayant object is valid
    std::string error_message = validate(); // Deep check
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    arma::uword n_el = e_theta_re.n_rows;
    arma::uword n_az = e_theta_re.n_cols;
    arma::uword n_ang = n_el * n_az;
    arma::uword n_elements = e_theta_re.n_slices;
    arma::uword n_element_max = arma::uword(destination.max()) + 1;
    arma::uword n_ports = coupling_re.n_cols;

    if (source >= n_elements)
        error_message = "Source index out of bound";
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Enlarge existing arrayant object
    if (n_element_max > n_elements)
    {
        arma::uword added_elements = n_element_max - n_elements;
        e_theta_re.resize(n_el, n_az, n_element_max);
        e_theta_im.resize(n_el, n_az, n_element_max);
        e_phi_re.resize(n_el, n_az, n_element_max);
        e_phi_im.resize(n_el, n_az, n_element_max);
        element_pos.resize(3, n_element_max);
        coupling_re.resize(n_element_max, n_ports + added_elements);
        coupling_im.resize(n_element_max, n_ports + added_elements);
        for (arma::uword i = 0; i < added_elements; i++)
            coupling_re.at(n_elements + i, n_ports + i) = dtype(1.0);
    }

    // Copy data from source to destination
    for (auto dest = destination.begin(); dest != destination.end(); ++dest)
        if (source != *dest)
        {
            std::memcpy(e_theta_re.slice_memptr(*dest), e_theta_re.slice_memptr(source), n_ang * sizeof(dtype));
            std::memcpy(e_theta_im.slice_memptr(*dest), e_theta_im.slice_memptr(source), n_ang * sizeof(dtype));
            std::memcpy(e_phi_re.slice_memptr(*dest), e_phi_re.slice_memptr(source), n_ang * sizeof(dtype));
            std::memcpy(e_phi_im.slice_memptr(*dest), e_phi_im.slice_memptr(source), n_ang * sizeof(dtype));
            std::memcpy(element_pos.colptr(*dest), element_pos.colptr(source), 3 * sizeof(dtype));
        }

    // Set the data pointers for the quick check.
    check_ptr[0] = e_theta_re.memptr();
    check_ptr[1] = e_theta_im.memptr();
    check_ptr[2] = e_phi_re.memptr();
    check_ptr[3] = e_phi_im.memptr();
    check_ptr[4] = azimuth_grid.memptr();
    check_ptr[5] = elevation_grid.memptr();
    check_ptr[6] = element_pos.memptr();
    check_ptr[7] = coupling_re.memptr();
    check_ptr[8] = coupling_im.memptr();
}

template <typename dtype>
void quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::copy_element(unsigned source, unsigned destination)
{
    arma::Col<unsigned> dest(1);
    dest.at(0) = destination;
    copy_element(source, dest);
}

// ARRAYANT METHOD : Interpolation
template <typename dtype>
void quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::interpolate(const arma::Mat<dtype> azimuth,
                                                                      const arma::Mat<dtype> elevation,
                                                                      const arma::Col<unsigned> i_element,
                                                                      const arma::Cube<dtype> orientation,
                                                                      const arma::Mat<dtype> element_pos_i,
                                                                      arma::Mat<dtype> *V_re, arma::Mat<dtype> *V_im,
                                                                      arma::Mat<dtype> *H_re, arma::Mat<dtype> *H_im,
                                                                      arma::Mat<dtype> *dist,
                                                                      arma::Mat<dtype> *azimuth_loc,
                                                                      arma::Mat<dtype> *elevation_loc,
                                                                      arma::Mat<dtype> *gamma) const
{
    // Check if arrayant object is valid
    std::string error_message = is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    arma::uword n_out = azimuth.n_rows;
    arma::uword n_ang = azimuth.n_cols;

    if (elevation.n_rows != n_out || elevation.n_cols != n_ang)
        error_message = "Sizes of 'azimuth' and 'elevation' do not match.";

    if (i_element.n_elem == 0)
        error_message = "Input 'i_element' cannot be empty.";

    if (orientation.n_elem == 0)
        error_message = "Input 'orientation' cannot be empty.";

    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Check if values are valid
    unsigned n_elements = unsigned(e_theta_re.n_slices);
    for (auto val = i_element.begin(); val != i_element.end(); ++val)
        if (*val < 1 || *val > n_elements)
            error_message = "Input 'i_element' must have values between 1 and 'n_elements'.";

    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Process orientation
    n_out = i_element.n_elem;
    arma::uword o1 = orientation.n_rows, o2 = orientation.n_cols, o3 = orientation.n_slices;
    if (o1 != 3)
        error_message = "Input 'orientation' must have 3 elements on the first dimension.";
    else if (o2 != 1 && o2 != n_out)
        error_message = "Input 'orientation' must have 1 or 'n_elements' elements on the second dimension.";
    else if (o3 != 1 && o3 != n_ang)
        error_message = "Input 'orientation' must have 1 or 'n_ang' elements on the third dimension.";

    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Process element_pos
    arma::Mat<dtype> element_pos_interp(3, n_out);
    if (!element_pos_i.empty())
    {
        if (element_pos_i.n_rows != 3 || element_pos_i.n_cols != n_out)
            error_message = "Alternative element positions 'element_pos_i' must have 'n_elements' elements.";

        if (error_message.length() != 0)
            throw std::invalid_argument(error_message.c_str());

        const dtype *ptrI = element_pos_i.memptr();
        dtype *ptrO = element_pos_interp.memptr();
        std::memcpy(ptrO, ptrI, 3 * n_out * sizeof(dtype));
    }
    else if (!element_pos.empty())
    {
        const dtype *ptrI = element_pos.memptr();
        dtype *ptrO = element_pos_interp.memptr();
        for (unsigned i = 0; i < n_out; i++)
            std::memcpy(&ptrO[3 * i], &ptrI[3 * i_element[i]], 3 * sizeof(dtype));
    }

    // Resize output variables
    if (V_re->n_rows != n_out || V_re->n_cols != n_ang)
        V_re->set_size(n_out, n_ang);
    if (V_im->n_rows != n_out || V_im->n_cols != n_ang)
        V_im->set_size(n_out, n_ang);
    if (H_re->n_rows != n_out || H_re->n_cols != n_ang)
        H_re->set_size(n_out, n_ang);
    if (H_im->n_rows != n_out || H_im->n_cols != n_ang)
        H_im->set_size(n_out, n_ang);
    if (dist->n_rows != n_out || dist->n_cols != n_ang)
        dist->set_size(n_out, n_ang);
    if (azimuth_loc->n_rows != n_out || azimuth_loc->n_cols != n_ang)
        azimuth_loc->set_size(n_out, n_ang);
    if (elevation_loc->n_rows != n_out || elevation_loc->n_cols != n_ang)
        elevation_loc->set_size(n_out, n_ang);
    if (gamma->n_rows != n_out || gamma->n_cols != n_ang)
        gamma->set_size(n_out, n_ang);

    // Call private library function
    qd_arrayant_interpolate(&e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                            &azimuth_grid, &elevation_grid, &azimuth, &elevation,
                            &i_element, &orientation, &element_pos_interp,
                            V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma);
}

// ARRAYANT METHOD : Interpolation
template <typename dtype>
void quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::interpolate(const arma::Mat<dtype> azimuth,
                                                                      const arma::Mat<dtype> elevation,
                                                                      const arma::Cube<dtype> orientation,
                                                                      arma::Mat<dtype> *V_re, arma::Mat<dtype> *V_im,
                                                                      arma::Mat<dtype> *H_re, arma::Mat<dtype> *H_im,
                                                                      arma::Mat<dtype> *dist) const
{
    // Check if arrayant object is valid
    std::string error_message = is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    arma::uword n_out = azimuth.n_rows;
    arma::uword n_ang = azimuth.n_cols;

    if (elevation.n_rows != n_out || elevation.n_cols != n_ang)
        error_message = "Sizes of 'azimuth' and 'elevation' do not match.";

    if (orientation.n_elem == 0)
        error_message = "Input 'orientation' cannot be empty.";

    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    n_out = e_theta_re.n_slices;
    arma::Col<unsigned> i_element = arma::linspace<arma::Col<unsigned>>(1, n_out, n_out);

    // Process orientation
    arma::uword o1 = orientation.n_rows, o2 = orientation.n_cols, o3 = orientation.n_slices;
    if (o1 != 3)
        error_message = "Input 'orientation' must have 3 elements on the first dimension.";
    else if (o2 != 1 && o2 != n_out)
        error_message = "Input 'orientation' must have 1 or 'n_elements' elements on the second dimension.";
    else if (o3 != 1 && o3 != n_ang)
        error_message = "Input 'orientation' must have 1 or 'n_ang' elements on the third dimension.";

    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Copy the element positions
    arma::Mat<dtype> element_pos_interp(3, n_out);
    if (!element_pos.empty())
    {
        const dtype *ptrI = element_pos.memptr();
        dtype *ptrO = element_pos_interp.memptr();
        std::memcpy(ptrO, ptrI, 3 * n_out * sizeof(dtype));
    }

    // Resize output variables
    if (V_re->n_rows != n_out || V_re->n_cols != n_ang)
        V_re->set_size(n_out, n_ang);
    if (V_im->n_rows != n_out || V_im->n_cols != n_ang)
        V_im->set_size(n_out, n_ang);
    if (H_re->n_rows != n_out || H_re->n_cols != n_ang)
        H_re->set_size(n_out, n_ang);
    if (H_im->n_rows != n_out || H_im->n_cols != n_ang)
        H_im->set_size(n_out, n_ang);
    if (dist->n_rows != n_out || dist->n_cols != n_ang)
        dist->set_size(n_out, n_ang);

    arma::Mat<dtype> azimuth_loc;
    arma::Mat<dtype> elevation_loc;
    arma::Mat<dtype> gamma;

    qd_arrayant_interpolate(&e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                            &azimuth_grid, &elevation_grid, &azimuth, &elevation,
                            &i_element, &orientation, &element_pos_interp,
                            V_re, V_im, H_re, H_im, dist, &azimuth_loc, &elevation_loc, &gamma);
}

// ARRAYANT : Write to QDANT file
template <typename dtype>
unsigned quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::qdant_write(std::string fn, unsigned id, arma::Mat<unsigned> layout) const
{
    // Check if arrayant object is valid
    std::string error_message = is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    unsigned id_in_file = 0;
    error_message = qd_arrayant_qdant_write(fn, id, &name, &e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                                            &azimuth_grid, &elevation_grid, &element_pos,
                                            &coupling_re, &coupling_im, &center_frequency, &layout,
                                            &id_in_file);

    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    return id_in_file;
}

// Remove zeros from the pattern
template <typename dtype>
void quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::remove_zeros(quadriga_lib::arrayant<dtype> *output)
{
    // Check if arrayant object is valid
    std::string error_message = validate(); // Deep check
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    if (output == NULL && read_only)
    {
        error_message = "Cannot update read-only array antenna object inplace.";
        throw std::invalid_argument(error_message.c_str());
    }

    arma::uword n_el = e_theta_re.n_rows;
    arma::uword n_az = e_theta_re.n_cols;
    arma::uword n_slices = e_theta_re.n_slices;

    // Calculate the power pattern
    arma::Mat<dtype> pow(n_el, n_az);
    dtype *pp = pow.memptr();
    dtype limit = dtype(1.0e-12), pi_half = dtype(arma::datum::pi / 2.0);

    for (arma::uword is = 0; is < n_slices; is++)
    {
        dtype *pa = e_theta_re.slice_memptr(is), *pb = e_theta_im.slice_memptr(is),
              *pc = e_phi_re.slice_memptr(is), *pd = e_phi_im.slice_memptr(is);
        for (arma::uword j = 0; j < n_el * n_az; j++)
            pp[j] += pa[j] * pa[j] + pb[j] * pb[j] + pc[j] * pc[j] + pd[j] * pd[j];
    }

    arma::Row<dtype> az_sum = arma::sum(pow, 0);
    arma::Col<dtype> el_sum = arma::sum(pow, 1);
    pow.reset();

    // Determine which parts of the pattern should be kept
    arma::uvec keep_az(n_az), keep_el(n_el);
    arma::uword *keep_az_ptr = keep_az.memptr();
    arma::uword *keep_el_ptr = keep_el.memptr();

    pp = az_sum.memptr();
    dtype *grid_ptr = azimuth_grid.memptr();
    for (arma::uword ia = 0; ia < n_az; ia++)
    {
        dtype p = ia == 0 ? *grid_ptr + grid_ptr[n_az - 1] : grid_ptr[ia] - grid_ptr[ia - 1];
        dtype n = ia == n_az - 1 ? *grid_ptr + grid_ptr[ia] : grid_ptr[ia + 1] - grid_ptr[ia];

        arma::uword ip = ia == 0 ? n_az - 1 : ia - 1;
        arma::uword in = ia == n_az - 1 ? 0 : ia + 1;

        if (pp[ia] > limit)
            keep_az_ptr[ip] = n > 0.001 * p && p > 1.5 * n ? keep_az_ptr[ip] : 1,
            keep_az_ptr[ia] = 1,
            keep_az_ptr[in] = p > 0.001 * n && n > 1.5 * p ? keep_az_ptr[in] : 1;
    }
    az_sum.reset();

    pp = el_sum.memptr(), grid_ptr = elevation_grid.memptr();

    for (arma::uword ie = 0; ie < n_el; ie++)
    {
        dtype p = ie == 0 ? *grid_ptr + pi_half : grid_ptr[ie] - grid_ptr[ie - 1];
        dtype n = ie == n_el - 1 ? pi_half - grid_ptr[ie] : grid_ptr[ie + 1] - grid_ptr[ie];

        arma::uword ip = ie == 0 ? 0 : ie - 1;
        arma::uword in = ie == n_el - 1 ? n_el - 1 : ie + 1;

        if (pp[ie] > limit)
            keep_el_ptr[ip] = n > 0.001 * p && p > 1.5 * n ? keep_el_ptr[ip] : 1,
            keep_el_ptr[ie] = 1,
            keep_el_ptr[in] = p > 0.001 * n && n > 1.5 * p ? keep_el_ptr[in] : 1;
    }
    el_sum.reset();

    // Copy the relevant pattern data
    arma::uword n_az_new = arma::sum(keep_az), n_el_new = arma::sum(keep_el);

    for (int p = 0; p < 4; p++)
    {
        dtype *data = NULL, *ptrO = NULL, *ptrI = NULL;

        if (output == NULL)
            data = new dtype[n_el_new * n_az_new * n_slices];
        else if (p == 0)
        {
            if (output->e_theta_re.n_rows != n_el_new || output->e_theta_re.n_cols != n_az_new || output->e_theta_re.n_slices != n_slices)
                output->e_theta_re.set_size(n_el_new, n_az_new, n_slices);
            data = output->e_theta_re.memptr();
        }
        else if (p == 1)
        {
            if (output->e_theta_im.n_rows != n_el_new || output->e_theta_im.n_cols != n_az_new || output->e_theta_im.n_slices != n_slices)
                output->e_theta_im.set_size(n_el_new, n_az_new, n_slices);
            data = output->e_theta_im.memptr();
        }
        else if (p == 2)
        {
            if (output->e_phi_re.n_rows != n_el_new || output->e_phi_re.n_cols != n_az_new || output->e_phi_re.n_slices != n_slices)
                output->e_phi_re.set_size(n_el_new, n_az_new, n_slices);
            data = output->e_phi_re.memptr();
        }
        else if (p == 3)
        {
            if (output->e_phi_im.n_rows != n_el_new || output->e_phi_im.n_cols != n_az_new || output->e_phi_im.n_slices != n_slices)
                output->e_phi_im.set_size(n_el_new, n_az_new, n_slices);
            data = output->e_phi_im.memptr();
        }

        ptrO = data;
        for (arma::uword is = 0; is < n_slices; is++)
        {
            if (p == 0)
                ptrI = e_theta_re.slice_memptr(is);
            else if (p == 1)
                ptrI = e_theta_im.slice_memptr(is);
            else if (p == 2)
                ptrI = e_phi_re.slice_memptr(is);
            else if (p == 3)
                ptrI = e_phi_im.slice_memptr(is);

            for (arma::uword ia = 0; ia < n_az; ia++)
            {
                bool keep_az = keep_az_ptr[ia] == 1;
                for (arma::uword ie = 0; ie < n_el; ie++)
                    if (keep_az && keep_el_ptr[ie] == 1)
                        *ptrO++ = ptrI[ia * n_el + ie];
            }
        }

        if (output == NULL)
        {
            if (p == 0)
                e_theta_re.set_size(n_el_new, n_az_new, n_slices),
                    std::memcpy(e_theta_re.memptr(), data, n_el_new * n_az_new * n_slices * sizeof(dtype));
            else if (p == 1)
                e_theta_im.set_size(n_el_new, n_az_new, n_slices),
                    std::memcpy(e_theta_im.memptr(), data, n_el_new * n_az_new * n_slices * sizeof(dtype));
            else if (p == 2)
                e_phi_re.set_size(n_el_new, n_az_new, n_slices),
                    std::memcpy(e_phi_re.memptr(), data, n_el_new * n_az_new * n_slices * sizeof(dtype));
            else if (p == 3)
                e_phi_im.set_size(n_el_new, n_az_new, n_slices),
                    std::memcpy(e_phi_im.memptr(), data, n_el_new * n_az_new * n_slices * sizeof(dtype));

            delete[] data;
        }
    }

    // Copy the new azimuth grid
    dtype *azimuth_grid_new = new dtype[n_az_new];
    pp = azimuth_grid_new;
    for (arma::uword ia = 0; ia < n_az; ia++)
        if (keep_az_ptr[ia] == 1)
            *pp++ = azimuth_grid[ia];
    if (output == NULL)
        azimuth_grid.set_size(n_az_new), pp = azimuth_grid.memptr();
    else if (output->azimuth_grid.n_elem != n_az_new)
        output->azimuth_grid.set_size(n_az_new), pp = output->azimuth_grid.memptr();
    else
        pp = output->azimuth_grid.memptr();
    std::memcpy(pp, azimuth_grid_new, n_az_new * sizeof(dtype));
    delete[] azimuth_grid_new;

    // Copy the new elevation grid
    dtype *elevation_grid_new = new dtype[n_el_new];
    pp = elevation_grid_new;
    for (arma::uword ie = 0; ie < n_el; ie++)
        if (keep_el_ptr[ie] == 1)
            *pp++ = elevation_grid[ie];
    if (output == NULL)
        elevation_grid.set_size(n_el_new), pp = elevation_grid.memptr();
    else if (output->elevation_grid.n_elem != n_el_new)
        output->elevation_grid.set_size(n_el_new), pp = output->elevation_grid.memptr();
    else
        pp = output->elevation_grid.memptr();
    std::memcpy(pp, elevation_grid_new, n_el_new * sizeof(dtype));
    delete[] elevation_grid_new;

    // Copy the unchanged data
    if (output != NULL)
    {
        if (!element_pos.empty())
        {
            if (output->element_pos.n_rows != 3 || output->element_pos.n_cols != n_slices)
                output->element_pos.set_size(3, n_slices);
            std::memcpy(output->element_pos.memptr(), element_pos.memptr(), 3 * n_slices * sizeof(dtype));
        }

        if (!coupling_re.empty())
        {
            if (output->coupling_re.n_rows != n_slices || output->coupling_re.n_cols != coupling_re.n_cols)
                output->coupling_re.set_size(n_slices, coupling_re.n_cols);
            std::memcpy(output->coupling_re.memptr(), coupling_re.memptr(), n_slices * coupling_re.n_cols * sizeof(dtype));
        }

        if (!coupling_im.empty())
        {
            if (output->coupling_im.n_rows != n_slices || output->coupling_im.n_cols != coupling_im.n_cols)
                output->coupling_im.set_size(n_slices, coupling_im.n_cols);
            std::memcpy(output->coupling_im.memptr(), coupling_im.memptr(), n_slices * coupling_im.n_cols * sizeof(dtype));
        }

        output->center_frequency = center_frequency;
        output->validate();
    }
}

// ARRAYANT METHOD : Change the size of an arrayant, without explicitly preserving data
template <typename dtype>
void quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::reset()
{
    if (read_only)
    {
        std::string error_message = "Cannot change size of read-only array antenna object.";
        throw std::invalid_argument(error_message.c_str());
    }

    if (azimuth_grid.n_elem != 0)
        azimuth_grid.reset();

    if (elevation_grid.n_elem != 0)
        elevation_grid.reset();

    if (e_theta_re.n_elem != 0)
        e_theta_re.reset();

    if (e_theta_im.n_elem != 0)
        e_theta_im.reset();

    if (e_phi_re.n_elem != 0)
        e_phi_re.reset();

    if (e_phi_im.n_elem != 0)
        e_phi_im.reset();

    if (element_pos.n_elem != 0)
        element_pos.reset();

    if (coupling_re.n_elem != 0)
        coupling_re.reset();

    if (coupling_im.n_elem != 0)
        coupling_im.reset();

    center_frequency = dtype(299792458.0);
}

// Rotating antenna patterns (adjusts sampling grid if needed, e.g. for parabolic antennas)
template <typename dtype>
void quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::rotate_pattern(dtype x_deg, dtype y_deg, dtype z_deg,
                                                                         unsigned usage, unsigned element,
                                                                         quadriga_lib::arrayant<dtype> *output)
{
    // Check if arrayant object is valid
    std::string error_message = validate(); // Deep check
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    bool use_all_elements = false;
    if (usage > 3)
        error_message = "Input parameter 'usage' must be 0, 1, 2 or 3.";
    if (element == unsigned(-1) || e_theta_re.n_slices == 1)
        use_all_elements = true;
    else if (element >= e_theta_re.n_slices)
        error_message = "Input parameter 'element' out of bound.";
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    if (output == NULL && read_only)
    {
        error_message = "Cannot update read-only array antenna object inplace.";
        throw std::invalid_argument(error_message.c_str());
    }

    // Set element indices
    arma::Col<unsigned> i_element(1);
    if (use_all_elements)
        i_element = arma::regspace<arma::Col<unsigned>>(1, e_theta_re.n_slices);
    else
        i_element.at(0) = element + 1;

    // Extract the element positions
    arma::Mat<dtype> element_pos_update;
    if (element_pos.n_elem == 0)
        element_pos_update.zeros(3, e_theta_re.n_slices);
    else
        element_pos_update = element_pos;

    arma::uword n_el = e_theta_re.n_rows;
    arma::uword n_az = e_theta_re.n_cols;
    arma::uword n_out = i_element.n_elem;
    arma::uword n_ang = n_el * n_az;

    dtype tau = dtype(arma::datum::tau), pi_half = dtype(arma::datum::pi / 2.0), pi = dtype(arma::datum::pi),
          deg2rad = dtype(arma::datum::pi / 180.0), limit = dtype(1.0e-6);

    // Calculate the coverage range for the angle sampling grid
    dtype az_step_min = 1e38, az_step_max = 0.0, el_step_min = 1e38, zero = 0.0, el_step_max = zero, step = zero;
    dtype *ptr = azimuth_grid.memptr();
    for (arma::uword i = 0; i < n_az; i++)
    {
        step = i == 0 ? *ptr - ptr[n_az - 1] + tau : ptr[i] - ptr[i - 1];
        az_step_min = step < az_step_min && step > limit ? step : az_step_min;
        az_step_max = step > az_step_max ? step : az_step_max;
    }
    ptr = elevation_grid.memptr();
    for (arma::uword i = 0; i <= n_el; i++)
    {
        if (i == 0)
            step = *ptr + pi_half;
        else if (i == n_el)
            step = pi_half - ptr[i - 1];
        else
            step = ptr[i] - ptr[i - 1];

        el_step_min = step < el_step_min && step > limit ? step : el_step_min;
        el_step_max = step > el_step_max ? step : el_step_max;
    }
    step = pi_half - ptr[n_el - 1];
    el_step_min = step < el_step_min && step > limit ? step : el_step_min;
    el_step_max = step > el_step_max ? step : el_step_max;
    step = el_step_min < az_step_min ? el_step_min : az_step_min;

    bool update_grid = (usage == 0 || usage == 1) && (az_step_max - az_step_min > limit ||
                                                      el_step_max - el_step_min > limit ||
                                                      az_step_min - el_step_min > limit ||
                                                      n_az == 1 || n_el == 1)
                           ? true
                           : false;

    // Obtain interpolation angles
    arma::Col<dtype> azimuth_grid_update, elevation_grid_update;
    if (update_grid)
    {
        if (output == NULL && !use_all_elements)
        {
            error_message = "Update of sampling grid cannot be done for single elements of an array antenna!";
            throw std::invalid_argument(error_message.c_str());
        }

        if (step > dtype(0.0017)) // >= 0.1 degree --> Use entire sphere
        {
            arma::uword N = arma::uword(std::round(pi / step));
            N = N < 4 ? 4 : N;

            azimuth_grid_update = arma::linspace<arma::Col<dtype>>(-pi, pi, 2 * N + 1);

            if (std::abs(x_deg) < limit && std::abs(y_deg) < limit)
                elevation_grid_update = elevation_grid;
            else
                elevation_grid_update = arma::linspace<arma::Col<dtype>>(-pi_half + step, pi_half - step, N - 1);
        }
        else // step < 0.1 degree --> Use subsampled sphere
        {
            auto ant = quadriga_lib::generate_arrayant_omni<dtype>();
            ant.e_theta_re.zeros();
            arma::Col<unsigned> has_az(361), has_el(181);

            // Select azimuth angles
            dtype *p = azimuth_grid.memptr(), *q = ant.azimuth_grid.memptr();
            unsigned *ra = has_az.memptr();
            for (unsigned i = 0; i < 360; i++)
            {
                while (p != azimuth_grid.end() && *p <= q[i + 1])
                {
                    ra[i] = *p >= q[i] && *p <= q[i + 1] ? 1 : ra[i];
                    p++;
                }
                ra[i + 1] = ra[i];
                if (p == azimuth_grid.end())
                    break;
            }

            // Select elevation angles
            p = elevation_grid.memptr(), q = ant.elevation_grid.memptr();
            unsigned *re = has_el.memptr();
            for (unsigned i = 0; i < 180; i++)
            {
                while (p != elevation_grid.end() && *p <= q[i + 1])
                {
                    re[i] = *p >= q[i] && *p <= q[i + 1] ? 1 : re[i];
                    p++;
                }
                re[i + 1] = re[i];
                if (p == elevation_grid.end())
                    break;
            }

            // Set interpolation target
            p = ant.e_theta_re.memptr();
            for (arma::uword i = 0; i < ant.e_theta_re.n_elem; i++)
                p[i] = ra[i / 181] == 1 && re[i % 181] == 1 ? 1.0 : 0.0;

            // Find target area for interpolation
            ant.rotate_pattern(x_deg, y_deg, z_deg, 1);
            ant.remove_zeros();

            // Subdivide target grid
            arma::uword sdiv = std::fmod(deg2rad, step) < 0.1 * step ? std::floor(deg2rad / step) : std::ceil(deg2rad / step);
            dtype stp = deg2rad / sdiv;

            bool wrap = ant.azimuth_grid.at(ant.azimuth_grid.n_elem - 1) + stp > pi;
            arma::uword N = wrap ? sdiv * (ant.azimuth_grid.n_elem - 1) + 1 : sdiv * ant.azimuth_grid.n_elem;
            azimuth_grid_update.set_size(N);
            p = azimuth_grid_update.memptr(), q = ant.azimuth_grid.memptr();
            for (unsigned i = 0; i < ant.azimuth_grid.n_elem; i++)
                if (wrap && i == ant.azimuth_grid.n_elem - 1)
                    *p = q[i];
                else
                    for (unsigned j = 0; j < sdiv; j++)
                        *p++ = q[i] + j * stp;

            wrap = ant.elevation_grid.at(ant.elevation_grid.n_elem - 1) + stp > pi_half;
            N = wrap ? sdiv * (ant.elevation_grid.n_elem - 1) + 1 : sdiv * ant.elevation_grid.n_elem;
            elevation_grid_update.set_size(N);
            p = elevation_grid_update.memptr(), q = ant.elevation_grid.memptr();
            for (unsigned i = 0; i < ant.elevation_grid.n_elem; i++)
                if (wrap && i == ant.elevation_grid.n_elem - 1)
                    *p = q[i];
                else
                    for (unsigned j = 0; j < sdiv; j++)
                        *p++ = q[i] + j * stp;
        }

        n_az = azimuth_grid_update.n_elem;
        n_el = elevation_grid_update.n_elem;
        n_ang = n_az * n_el;
    }
    else
    {
        azimuth_grid_update = azimuth_grid;
        elevation_grid_update = elevation_grid;
    }

    // Initiate output variables
    arma::Cube<dtype> orientation(3, 1, 1);
    arma::Mat<dtype> V_re(n_out, n_ang, arma::fill::none), V_im(n_out, n_ang, arma::fill::none);
    arma::Mat<dtype> H_re(n_out, n_ang, arma::fill::none), H_im(n_out, n_ang, arma::fill::none);
    arma::Mat<dtype> azimuth_loc, elevation_loc, gamma, EMPTY;

    // Create list of angles for pattern interpolation
    arma::Mat<dtype> azimuth(1, n_ang, arma::fill::none);
    arma::Mat<dtype> elevation(1, n_ang, arma::fill::none);
    dtype *p_azimuth = azimuth.memptr(), *p_elevation = elevation.memptr(),
          *p_phi = azimuth_grid_update.memptr(), *p_theta = elevation_grid_update.memptr();

    for (arma::uword ia = 0; ia < n_az; ia++)
        for (arma::uword ie = 0; ie < n_el; ie++)
            *p_azimuth++ = p_phi[ia], *p_elevation++ = p_theta[ie];

    // Set antenna orientation
    orientation.at(0) = x_deg * deg2rad;
    orientation.at(1) = -y_deg * deg2rad;
    orientation.at(2) = z_deg * deg2rad;

    // Calculate rotation matrix (double precision)
    arma::cube R = quadriga_tools::calc_rotation_matrix(orientation, true);

    if (usage == 1)
        azimuth_loc.set_size(n_out, n_ang), elevation_loc.set_size(n_out, n_ang);
    else if (usage == 2)
        gamma.set_size(n_out, n_ang);

    qd_arrayant_interpolate(&e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                            &azimuth_grid, &elevation_grid, &azimuth, &elevation,
                            &i_element, &orientation, &element_pos_update,
                            &V_re, &V_im, &H_re, &H_im, &EMPTY, &azimuth_loc, &elevation_loc, &gamma);
    azimuth.reset(), elevation.reset();

    orientation.zeros();
    if (usage == 1) // Only return interpolated pattern (ignore polarization)
        qd_arrayant_interpolate(&e_theta_re, &e_theta_im, &e_phi_re, &e_phi_im,
                                &azimuth_grid, &elevation_grid, &azimuth_loc, &elevation_loc,
                                &i_element, &orientation, &element_pos_update,
                                &V_re, &V_im, &H_re, &H_im, &EMPTY, &EMPTY, &EMPTY, &EMPTY);
    azimuth_loc.reset(), elevation_loc.reset();

    // Adjust size of output, if needed
    if (output != NULL)
        output->set_size(n_el, n_az, n_out, n_out);

    if (usage == 2) // Only adjust the polarization
    {
        gamma = gamma.t();
        for (arma::uword i = 0; i < n_out; i++)
        {
            dtype *p_gamma = gamma.colptr(i);
            dtype *p_theta_re = use_all_elements ? e_theta_re.slice_memptr(i) : e_theta_re.slice_memptr(element);
            dtype *p_theta_im = use_all_elements ? e_theta_im.slice_memptr(i) : e_theta_im.slice_memptr(element);
            dtype *p_phi_re = use_all_elements ? e_phi_re.slice_memptr(i) : e_phi_re.slice_memptr(element);
            dtype *p_phi_im = use_all_elements ? e_phi_im.slice_memptr(i) : e_phi_im.slice_memptr(element);

            dtype *q_theta_re = p_theta_re, *q_theta_im = p_theta_im, *q_phi_re = p_phi_re, *q_phi_im = p_phi_im;
            if (output != NULL)
                q_theta_re = use_all_elements ? output->e_theta_re.slice_memptr(i) : output->e_theta_re.memptr(),
                q_theta_im = use_all_elements ? output->e_theta_im.slice_memptr(i) : output->e_theta_im.memptr(),
                q_phi_re = use_all_elements ? output->e_phi_re.slice_memptr(i) : output->e_phi_re.memptr(),
                q_phi_im = use_all_elements ? output->e_phi_im.slice_memptr(i) : output->e_phi_im.memptr();

            for (arma::uword j = 0; j < n_ang; j++)
            {
                dtype sin_gamma = std::sin(p_gamma[j]), cos_gamma = std::cos(p_gamma[j]);
                dtype tmp = sin_gamma * p_theta_re[j];
                q_theta_re[j] = cos_gamma * p_theta_re[j] - sin_gamma * p_phi_re[j];
                q_phi_re[j] = tmp + cos_gamma * p_phi_re[j];
                tmp = sin_gamma * p_theta_im[j];
                q_theta_im[j] = cos_gamma * p_theta_im[j] - sin_gamma * p_phi_im[j];
                q_phi_im[j] = tmp + cos_gamma * p_phi_im[j];
            }
        }
        gamma.reset();

        // Copy unchanged data
        if (output != NULL)
        {
            dtype *ptrI = use_all_elements ? element_pos_update.memptr() : element_pos_update.colptr(element);
            dtype *ptrO = output->element_pos.memptr();
            std::memcpy(ptrO, ptrI, 3 * n_out * sizeof(dtype));

            ptrI = azimuth_grid_update.memptr();
            ptrO = output->azimuth_grid.memptr();
            std::memcpy(ptrO, ptrI, n_az * sizeof(dtype));

            ptrI = elevation_grid_update.memptr();
            ptrO = output->elevation_grid.memptr();
            std::memcpy(ptrO, ptrI, n_el * sizeof(dtype));
        }
    }
    else // Usage 0, 1 or 3
    {
        gamma.reset();
        if (n_out > 1) // Transpose the interpolated data
            V_re = V_re.t(), V_im = V_im.t(), H_re = H_re.t(), H_im = H_im.t();

        if (output == NULL && update_grid) // Set new array size
            set_size(n_el, n_az, n_out, n_out);

        dtype *ptrI, *ptrO;

        ptrI = V_re.memptr();
        ptrO = use_all_elements ? e_theta_re.memptr() : e_theta_re.slice_memptr(element);
        ptrO = output == NULL ? ptrO : output->e_theta_re.memptr();
        std::memcpy(ptrO, ptrI, n_el * n_az * n_out * sizeof(dtype));
        V_re.reset();

        ptrI = V_im.memptr();
        ptrO = use_all_elements ? e_theta_im.memptr() : e_theta_im.slice_memptr(element);
        ptrO = output == NULL ? ptrO : output->e_theta_im.memptr();
        std::memcpy(ptrO, ptrI, n_el * n_az * n_out * sizeof(dtype));
        V_im.reset();

        ptrI = H_re.memptr();
        ptrO = use_all_elements ? e_phi_re.memptr() : e_phi_re.slice_memptr(element);
        ptrO = output == NULL ? ptrO : output->e_phi_re.memptr();
        std::memcpy(ptrO, ptrI, n_el * n_az * n_out * sizeof(dtype));
        H_re.reset();

        ptrI = H_im.memptr();
        ptrO = use_all_elements ? e_phi_im.memptr() : e_phi_im.slice_memptr(element);
        ptrO = output == NULL ? ptrO : output->e_phi_im.memptr();
        std::memcpy(ptrO, ptrI, n_el * n_az * n_out * sizeof(dtype));
        H_im.reset();

        ptrI = azimuth_grid_update.memptr();
        ptrO = output == NULL ? azimuth_grid.memptr() : output->azimuth_grid.memptr();
        std::memcpy(ptrO, ptrI, n_az * sizeof(dtype));

        ptrI = elevation_grid_update.memptr();
        ptrO = output == NULL ? elevation_grid.memptr() : output->elevation_grid.memptr();
        std::memcpy(ptrO, ptrI, n_el * sizeof(dtype));

        ptrI = element_pos_update.memptr();
        ptrO = use_all_elements ? element_pos.memptr() : element_pos.colptr(element);
        ptrO = output == NULL ? ptrO : output->element_pos.memptr();
        double *R_ptr = R.memptr();
        for (arma::uword i = 0; i < i_element.n_elem; i++)
        {
            unsigned j = 3 * (i_element.at(i) - 1);
            unsigned k = use_all_elements ? j : 0;
            dtype a = dtype(R_ptr[0]) * ptrI[j] + dtype(R_ptr[3]) * ptrI[j + 1] + dtype(R_ptr[6]) * ptrI[j + 2];
            dtype b = dtype(R_ptr[1]) * ptrI[j] + dtype(R_ptr[4]) * ptrI[j + 1] + dtype(R_ptr[7]) * ptrI[j + 2];
            dtype c = dtype(R_ptr[2]) * ptrI[j] + dtype(R_ptr[5]) * ptrI[j + 1] + dtype(R_ptr[8]) * ptrI[j + 2];
            ptrO[k] = a, ptrO[k + 1] = b, ptrO[k + 2] = c;
        }

        if (output == NULL && update_grid)
            remove_zeros();
        else if (output != NULL && update_grid)
            output->remove_zeros();
    }
}

// ARRAYANT METHOD : Change the size of an arrayant, without explicitly preserving data
template <typename dtype>
void quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::set_size(unsigned n_elevation, unsigned n_azimuth,
                                                                   unsigned n_elements, unsigned n_ports)
{
    if (read_only)
    {
        std::string error_message = "Cannot change size of read-only array antenna object.";
        throw std::invalid_argument(error_message.c_str());
    }

    if (azimuth_grid.n_elem != n_azimuth)
        azimuth_grid.set_size(n_azimuth);

    if (elevation_grid.n_elem != n_elevation)
        elevation_grid.set_size(n_elevation);

    if (e_theta_re.n_rows != n_elevation || e_theta_re.n_cols != n_azimuth || e_theta_re.n_slices != n_elements)
        e_theta_re.set_size(n_elevation, n_azimuth, n_elements);

    if (e_theta_im.n_rows != n_elevation || e_theta_im.n_cols != n_azimuth || e_theta_im.n_slices != n_elements)
        e_theta_im.set_size(n_elevation, n_azimuth, n_elements);

    if (e_phi_re.n_rows != n_elevation || e_phi_re.n_cols != n_azimuth || e_phi_re.n_slices != n_elements)
        e_phi_re.set_size(n_elevation, n_azimuth, n_elements);

    if (e_phi_im.n_rows != n_elevation || e_phi_im.n_cols != n_azimuth || e_phi_im.n_slices != n_elements)
        e_phi_im.set_size(n_elevation, n_azimuth, n_elements);

    if (element_pos.n_rows != 3 || element_pos.n_cols != n_elements)
        element_pos.zeros(3, n_elements);

    if (coupling_re.n_rows != n_elements || coupling_re.n_cols != n_ports)
        coupling_re.eye(n_elements, n_ports);

    if (coupling_im.n_rows != n_elements || coupling_im.n_cols != n_ports)
        coupling_im.zeros(n_elements, n_ports);
}

// ARRAYANT METHOD : Validates correctness of the member functions
template <typename dtype>
std::string quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::is_valid(bool quick_check) const
{

    // Assuming that the data has been validated before, we can quickly check if the
    // data pointers were updated. If not, we can assume that the data is still valid.
    if (quick_check)
        quick_check = check_ptr[0] != NULL && check_ptr[1] != NULL && check_ptr[2] != NULL &&
                      check_ptr[3] != NULL && check_ptr[4] != NULL && check_ptr[5] != NULL &&
                      check_ptr[0] == e_theta_re.memptr() &&
                      check_ptr[1] == e_theta_im.memptr() &&
                      check_ptr[2] == e_phi_re.memptr() &&
                      check_ptr[3] == e_phi_im.memptr() &&
                      check_ptr[4] == azimuth_grid.memptr() &&
                      check_ptr[5] == elevation_grid.memptr() &&
                      check_ptr[6] == element_pos.memptr() &&
                      check_ptr[7] == coupling_re.memptr() &&
                      check_ptr[8] == coupling_im.memptr();
    if (quick_check)
        return "";

    // Perform a deep check
    if (e_theta_re.n_elem == 0 || e_theta_im.n_elem == 0 || e_phi_re.n_elem == 0 || e_phi_im.n_elem == 0 || azimuth_grid.n_elem == 0 || elevation_grid.n_elem == 0)
        return "Missing data for any of: e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid";

    arma::uword n_elevation = e_theta_re.n_rows;
    arma::uword n_azimuth = e_theta_re.n_cols;
    arma::uword n_elements = e_theta_re.n_slices;

    if (e_theta_im.n_rows != n_elevation || e_theta_im.n_cols != n_azimuth || e_theta_im.n_slices != n_elements)
        return "Sizes of 'e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im' do not match.";

    if (e_phi_re.n_rows != n_elevation || e_phi_re.n_cols != n_azimuth || e_phi_re.n_slices != n_elements)
        return "Sizes of 'e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im' do not match.";

    if (e_phi_im.n_rows != n_elevation || e_phi_im.n_cols != n_azimuth || e_phi_im.n_slices != n_elements)
        return "Sizes of 'e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im' do not match.";

    if (azimuth_grid.n_elem != n_azimuth)
        return "Number of elements in 'azimuth_grid' does not match number of columns in pattern data.";

    if (elevation_grid.n_elem != n_elevation)
        return "Number of elements in 'elevation_grid' does not match number of rows in pattern data.";

    bool error_code = false;
    for (const dtype *val = azimuth_grid.begin(); val < azimuth_grid.end(); val++)
        error_code = *val < dtype(-3.1415930) || *val > dtype(3.1415930) ? true : error_code;
    if (error_code)
        return "Values of 'azimuth_grid' must be between -pi and pi (equivalent to -180 to 180 degree).";

    for (const dtype *val = elevation_grid.begin(); val < elevation_grid.end(); val++)
        error_code = *val < dtype(-1.5707965) || *val > dtype(1.5707965) ? true : error_code;
    if (error_code)
        return "Values of 'elevation_grid' must be between -pi/2 and pi/2 (equivalent to -90 to 90 degree).";

    if (!azimuth_grid.is_sorted())
        return "Values of 'azimuth_grid' must be sorted in ascending order.";

    if (!elevation_grid.is_sorted())
        return "Values of 'elevation_grid' must be sorted in ascending order.";

    if (!element_pos.empty() && (element_pos.n_rows != 3 || element_pos.n_cols != n_elements))
        return "Size of 'element_pos' must be either empty or match [3, n_elements]";

    if (!coupling_re.empty() && coupling_re.n_rows != n_elements)
        return "'Coupling' must be a matrix with rows equal to number of elements";

    if (coupling_re.empty() && !coupling_im.empty())
        return "Imaginary part of coupling matrix (phase component) defined without real part (absolute component)";

    if (!coupling_im.empty() && (coupling_im.n_rows != n_elements && coupling_im.n_cols != coupling_re.n_rows))
        return "'Coupling' must be a matrix with rows equal to number of elements and columns equal to number of ports";

    return "";
}

// ARRAYANT METHOD : Validates correctness of the member functions and initializes element positions and coupling matrix
template <typename dtype>
std::string quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<dtype>::validate()
{
    std::string error_message = is_valid(false); // Deep check
    if (error_message.length() != 0)
        return error_message;

    arma::uword n_elements = e_theta_re.n_slices;
    arma::uword n_prt = coupling_re.empty() ? n_elements : coupling_re.n_cols;

    if (element_pos.empty())
        element_pos.zeros(3, n_elements);

    if (coupling_re.empty())
        coupling_re.eye(n_elements, n_elements), coupling_im.zeros(n_elements, n_elements);

    if (coupling_im.empty())
        coupling_im.zeros(n_elements, n_prt);

    // Set the data pointers for the quick check.
    check_ptr[0] = e_theta_re.memptr();
    check_ptr[1] = e_theta_im.memptr();
    check_ptr[2] = e_phi_re.memptr();
    check_ptr[3] = e_phi_im.memptr();
    check_ptr[4] = azimuth_grid.memptr();
    check_ptr[5] = elevation_grid.memptr();
    check_ptr[6] = element_pos.memptr();
    check_ptr[7] = coupling_re.memptr();
    check_ptr[8] = coupling_im.memptr();

    return "";
}

template class quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<float>;
template class quadriga_lib::QUADRIGA_LIB_VERSION::arrayant<double>;

// Read array antenna object and layout from QDANT file
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::qdant_read(std::string fn, unsigned id, arma::Mat<unsigned> *layout)
{
    quadriga_lib::arrayant<dtype> ant;
    std::string error_message;

    if (layout == NULL)
    {
        arma::Mat<unsigned> tmp_layout;
        error_message = qd_arrayant_qdant_read(fn, id, &ant.name,
                                               &ant.e_theta_re, &ant.e_theta_im, &ant.e_phi_re, &ant.e_phi_im,
                                               &ant.azimuth_grid, &ant.elevation_grid, &ant.element_pos,
                                               &ant.coupling_re, &ant.coupling_im, &ant.center_frequency,
                                               &tmp_layout);
        tmp_layout.reset();
    }
    else
        error_message = qd_arrayant_qdant_read(fn, id, &ant.name,
                                               &ant.e_theta_re, &ant.e_theta_im, &ant.e_phi_re, &ant.e_phi_im,
                                               &ant.azimuth_grid, &ant.elevation_grid, &ant.element_pos,
                                               &ant.coupling_re, &ant.coupling_im, &ant.center_frequency,
                                               layout);

    // Throw parsing errors
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Throw validation errors
    error_message = ant.validate();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::qdant_read(std::string fn, unsigned id, arma::Mat<unsigned> *layout);
template quadriga_lib::arrayant<double> quadriga_lib::qdant_read(std::string fn, unsigned id, arma::Mat<unsigned> *layout);

// Generate : Isotropic radiator, vertical polarization, 1 deg resolution
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_omni()
{
    quadriga_lib::arrayant<dtype> ant;

    dtype pi = dtype(arma::datum::pi);
    ant.name = "omni";
    ant.e_theta_re.ones(181, 361, 1);
    ant.e_theta_im.zeros(181, 361, 1);
    ant.e_phi_re.zeros(181, 361, 1);
    ant.e_phi_im.zeros(181, 361, 1);
    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 361);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pi / 2.0, pi / 2.0, 181);
    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_omni();
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_omni();

// Generate : Cross-polarized isotropic radiator, 1 deg resolution
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_xpol()
{
    quadriga_lib::arrayant<dtype> ant;

    dtype pi = dtype(arma::datum::pi);
    ant.name = "xpol";
    ant.e_theta_re.ones(181, 361, 2);
    ant.e_theta_im.zeros(181, 361, 2);
    ant.e_phi_re.ones(181, 361, 2);
    ant.e_phi_im.zeros(181, 361, 2);
    ant.e_phi_re.slice(0).zeros();
    ant.e_theta_re.slice(1).zeros();
    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 361);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pi / 2.0, pi / 2.0, 181);
    ant.element_pos.zeros(3, 2);
    ant.coupling_re.eye(2, 2);
    ant.coupling_im.zeros(2, 2);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_xpol();
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_xpol();

// Generate : Short dipole radiating with vertical polarization, 1 deg resolution
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_dipole()
{
    quadriga_lib::arrayant<dtype> ant;

    dtype pi = dtype(arma::datum::pi);
    ant.name = "dipole";
    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 361);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pi / 2.0, pi / 2.0, 181);
    ant.e_theta_re.zeros(181, 361, 1);
    ant.e_theta_im.zeros(181, 361, 1);
    ant.e_phi_re.zeros(181, 361, 1);
    ant.e_phi_im.zeros(181, 361, 1);
    ant.e_theta_re.slice(0) = arma::repmat(ant.elevation_grid, 1, 361);
    ant.e_theta_re = arma::cos(0.999999 * ant.e_theta_re) * std::sqrt(1.499961);
    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_dipole();
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_dipole();

// Generate : Half-wave dipole radiating with vertical polarization
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_half_wave_dipole()
{
    quadriga_lib::arrayant<dtype> ant;

    dtype pi = dtype(arma::datum::pi), pih = dtype(arma::datum::pi / 2.0);
    ant.name = "half-wave-dipole";
    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 361);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pi / 2.0, pi / 2.0, 181);
    ant.e_theta_re.zeros(181, 361, 1);
    ant.e_theta_im.zeros(181, 361, 1);
    ant.e_phi_re.zeros(181, 361, 1);
    ant.e_phi_im.zeros(181, 361, 1);
    ant.e_theta_re.slice(0) = arma::repmat(ant.elevation_grid, 1, 361);
    ant.e_theta_re = arma::cos(pih * arma::sin(0.999999 * ant.e_theta_re)) / arma::cos(0.999999 * ant.e_theta_re);
    ant.e_theta_re = ant.e_theta_re * dtype(1.280968208215292);
    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_half_wave_dipole();
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_half_wave_dipole();

// Generate : An antenna with a custom gain in elevation and azimuth
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_custom(dtype az_3dB, dtype el_3db, dtype rear_gain_lin)
{
    dtype pi = dtype(arma::datum::pi), one = 1.0, half = 0.5, limit = 1e-7, step = -0.382, limit_inf = 1e38,
          deg2rad = dtype(arma::datum::pi / 360.0);

    quadriga_lib::arrayant<dtype> ant;

    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 361);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pi / 2.0, pi / 2.0, 181);
    arma::Col<dtype> phi_sq = ant.azimuth_grid % ant.azimuth_grid;
    arma::Col<dtype> cos_theta = arma::cos(ant.elevation_grid);
    cos_theta.at(0) = dtype(0.0), cos_theta.at(180) = dtype(0.0);
    arma::Col<dtype> az_3dB_rad(1), el_3db_rad(1);
    az_3dB_rad.at(0) = az_3dB * deg2rad;
    el_3db_rad.at(0) = el_3db * deg2rad;

    // Calculate azimuth pattern cut
    dtype a = one, d = half, x = limit_inf, delta = limit_inf;
    arma::Col<dtype> xn(1), C(361), D(181);
    for (unsigned lp = 0; lp < 5000; lp++)
    {
        dtype an = lp == 0 ? a : a + d;
        delta = lp == 0 ? limit_inf : std::abs(a - an);
        C = rear_gain_lin + (one - rear_gain_lin) * arma::exp(-an * phi_sq);
        quadriga_tools::interp(&C, &ant.azimuth_grid, &az_3dB_rad, &xn);
        dtype xm = std::abs(xn.at(0) - half);
        a = xm < x ? an : a;
        d = xm < x ? d : step * d;
        x = xm < x ? xm : x;
        if (delta < limit)
            break;
    }
    C = arma::exp(-a * phi_sq);

    // Calculate elevation pattern cut
    a = one, d = half, x = limit_inf, delta = limit_inf;
    for (unsigned lp = 0; lp < 5000; lp++)
    {
        dtype an = lp == 0 ? a : a + d;
        delta = lp == 0 ? limit_inf : std::abs(a - an);
        D = arma::pow(cos_theta, an);
        quadriga_tools::interp(&D, &ant.elevation_grid, &el_3db_rad, &xn);
        dtype xm = std::abs(xn.at(0) - half);
        a = xm < x ? an : a;
        d = xm < x ? d : step * d;
        x = xm < x ? xm : x;
        if (delta < limit)
            break;
    }
    D = arma::pow(cos_theta, a);

    // Combined pattern
    ant.e_theta_re.zeros(181, 361, 1);
    dtype *ptr = ant.e_theta_re.memptr();
    for (dtype *col = C.begin(); col != C.end(); col++)
        for (dtype *row = D.begin(); row != D.end(); row++)
            *ptr++ = std::sqrt(rear_gain_lin + (one - rear_gain_lin) * *row * *col);

    ant.e_theta_im.zeros(181, 361, 1);
    ant.e_phi_re.zeros(181, 361, 1);
    ant.e_phi_im.zeros(181, 361, 1);
    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);
    ant.name = "custom";

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    // Normalize to Gain
    dtype directivity = ant.calc_directivity_dBi(0);
    directivity = std::pow(10.0, 0.1 * directivity);
    dtype p_max = ant.e_theta_re.max();
    p_max *= p_max;
    ant.e_theta_re *= std::sqrt(directivity / p_max);

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_custom(float az_3dB, float el_3db, float rear_gain_lin);
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_custom(double az_3dB, double el_3db, double rear_gain_lin);

// Generate : Antenna model for the 3GPP-NR channel model
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_3GPP(unsigned M, unsigned N, dtype center_freq,
                                                                   unsigned pol, dtype tilt, dtype spacing,
                                                                   unsigned Mg, unsigned Ng, dtype dgv, dtype dgh,
                                                                   const arrayant<dtype> *pattern)
{
    dtype pi = dtype(arma::datum::pi), rad2deg = dtype(180.0 / arma::datum::pi), deg2rad = dtype(arma::datum::pi / 180.0);
    dtype wavelength = dtype(299792458.0 / double(center_freq));

    quadriga_lib::arrayant<dtype> ant = pattern == NULL ? quadriga_lib::generate_arrayant_omni<dtype>() : pattern->copy();

    if (pattern != NULL)
    {
        std::string error_message = ant.validate(); // Deep check
        if (error_message.length() != 0)
            throw std::invalid_argument(error_message.c_str());
    }

    ant.center_frequency = center_freq;
    arma::uword n_az = ant.n_azimuth(), n_el = ant.n_elevation();

    if (pattern == NULL) // Generate 3GPP default radiation pattern
    {
        // Single antenna element vertical radiation pattern cut in dB
        arma::Col<dtype> Y = ant.elevation_grid;
        for (dtype *py = Y.begin(); py < Y.end(); py++)
        {
            dtype y = *py * rad2deg / 65.0;
            y = 12.0 * y * y;
            *py = y > 30.0 ? 30.0 : y;
        }

        // Full pattern (normalized to 8 dBi gain using factor 2.51..)
        dtype *ptr = ant.e_theta_re.memptr(), *py = Y.memptr(), *px = ant.azimuth_grid.memptr();
        for (arma::uword ia = 0; ia < n_az; ia++)
        {
            dtype x = *px++ * rad2deg / 65.0;
            x = 12.0 * x * x;
            x = x > 30.0 ? 30.0 : x;

            for (arma::uword ie = 0; ie < n_el; ie++)
            {
                dtype z = py[ie] + x;
                z = z > 30.0 ? -30.0 : -z;
                *ptr++ = 2.511886431509580 * std::sqrt(std::pow(10.0, 0.1 * z)); // 8dBi gain
            }
        }
        Y.reset();

        // Adjust polarization
        if (pol == 2 || pol == 5)
        {
            ant.copy_element(0, 1);
            ant.rotate_pattern(90.0, 0.0, 0.0, 2, 1);
        }
        else if (pol == 3 || pol == 6)
        {
            ant.copy_element(0, 1);
            ant.rotate_pattern(45.0, 0.0, 0.0, 2, 0);
            ant.rotate_pattern(-45.0, 0.0, 0.0, 2, 1);
        }
    }

    // Duplicate the existing elements in z-direction (vertical stacking)
    unsigned n_elements = ant.n_elements();
    if (M > 1)
        for (unsigned source = n_elements; source > 0; source--)
        {
            unsigned i_start = n_elements + source - 1;
            unsigned i_end = M * n_elements - 1;
            arma::Col<unsigned> destination = arma::regspace<arma::Col<unsigned>>(i_start, n_elements, i_end);
            ant.copy_element(source - 1, destination);
        }

    // Calculate the element z-position
    arma::Col<dtype> z_position(M);
    if (M > 1)
    {
        z_position = arma::linspace<arma::Col<dtype>>(0.0, dtype(M - 1) * spacing * wavelength, M);
        z_position = z_position - arma::mean(z_position);

        for (unsigned m = 0; m < M; m++)
            for (unsigned n = 0; n < n_elements; n++)
                ant.element_pos.at(2, m * n_elements + n) = z_position.at(m);
    }

    // Apply element coupling for polarization indicators 4, 5, and 6
    if (pol > 3 && M > 1)
    {
        dtype tmp = 2.0 * pi * std::sin(tilt * deg2rad) / wavelength;
        arma::Col<dtype> cpl_re = z_position * tmp;
        tmp = dtype(1.0 / std::sqrt(double(M)));
        arma::Col<dtype> cpl_im = arma::sin(cpl_re) * tmp;
        cpl_re = arma::cos(cpl_re) * tmp;

        ant.coupling_re.zeros(n_elements * M, n_elements);
        ant.coupling_im.zeros(n_elements * M, n_elements);

        for (unsigned m = 0; m < M; m++)
            for (unsigned n = 0; n < n_elements; n++)
            {
                ant.coupling_re.at(m * n_elements + n, n) = cpl_re.at(m);
                ant.coupling_im.at(m * n_elements + n, n) = cpl_im.at(m);
            }

        ant.combine_pattern();
        M = 1;
    }

    // Duplicate the existing elements in y-direction (horizontal stacking)
    n_elements = ant.n_elements();
    if (N > 1)
    {
        for (unsigned source = n_elements; source > 0; source--)
        {
            unsigned i_start = n_elements + source - 1;
            unsigned i_end = N * n_elements - 1;
            arma::Col<unsigned> destination = arma::regspace<arma::Col<unsigned>>(i_start, n_elements, i_end);
            ant.copy_element(source - 1, destination);
        }

        arma::Col<dtype> y_position = arma::linspace<arma::Col<dtype>>(0.0, dtype(N - 1) * spacing * wavelength, N);
        y_position = y_position - arma::mean(y_position);

        for (unsigned m = 0; m < N; m++)
            for (unsigned n = 0; n < n_elements; n++)
                ant.element_pos.at(1, m * n_elements + n) = y_position.at(m);
    }

    // Duplicate panels in z-direction (vertical panel stacking)
    n_elements = ant.n_elements();
    if (Mg > 1)
    {
        for (unsigned source = n_elements; source > 0; source--)
        {
            unsigned i_start = n_elements + source - 1;
            unsigned i_end = Mg * n_elements - 1;
            arma::Col<unsigned> destination = arma::regspace<arma::Col<unsigned>>(i_start, n_elements, i_end);
            ant.copy_element(source - 1, destination);
        }

        arma::Col<dtype> zg_position = arma::linspace<arma::Col<dtype>>(0.0, dtype(Mg - 1) * dgv * wavelength, Mg);
        zg_position = zg_position - arma::mean(zg_position);

        for (unsigned mg = 0; mg < Mg; mg++)
            for (unsigned n = 0; n < n_elements; n++)
                ant.element_pos.at(2, mg * n_elements + n) += zg_position.at(mg);
    }

    // Duplicate panels in y-direction (horizontal panel stacking)
    n_elements = ant.n_elements();
    if (Ng > 1)
    {
        for (unsigned source = n_elements; source > 0; source--)
        {
            unsigned i_start = n_elements + source - 1;
            unsigned i_end = Ng * n_elements - 1;
            arma::Col<unsigned> destination = arma::regspace<arma::Col<unsigned>>(i_start, n_elements, i_end);
            ant.copy_element(source - 1, destination);
        }

        arma::Col<dtype> yg_position = arma::linspace<arma::Col<dtype>>(0.0, dtype(Ng - 1) * dgv * wavelength, Ng);
        yg_position = yg_position - arma::mean(yg_position);

        for (unsigned mg = 0; mg < Ng; mg++)
            for (unsigned n = 0; n < n_elements; n++)
                ant.element_pos.at(1, mg * n_elements + n) += yg_position.at(mg);
    }

    ant.name = "3gpp";
    return ant;
}

template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_3GPP(unsigned M, unsigned N, float center_freq,
                                                                            unsigned pol, float tilt, float spacing,
                                                                            unsigned Mg, unsigned Ng, float dgv, float dgh,
                                                                            const quadriga_lib::arrayant<float> *pattern);

template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_3GPP(unsigned M, unsigned N, double center_freq,
                                                                             unsigned pol, double tilt, double spacing,
                                                                             unsigned Mg, unsigned Ng, double dgv, double dgh,
                                                                             const quadriga_lib::arrayant<double> *pattern);
