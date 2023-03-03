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

#include <iostream>
#include <algorithm>
#include "qd_arrayant_qdant.hpp"
#include "pugixml.hpp"

// Note: Uncomment "#define PUGIXML_HEADER_ONLY" in "pugiconfig.hpp"

using namespace std;

template <typename dataType> // float or double
std::string qd_arrayant_qdant_read(const std::string fn, const int id,
                                   std::string *name,
                                   arma::Cube<dataType> *e_theta_re, arma::Cube<dataType> *e_theta_im,
                                   arma::Cube<dataType> *e_phi_re, arma::Cube<dataType> *e_phi_im,
                                   arma::Col<dataType> *azimuth_grid, arma::Col<dataType> *elevation_grid,
                                   arma::Mat<dataType> *element_pos,
                                   arma::Mat<dataType> *coupling_re, arma::Mat<dataType> *coupling_im,
                                   dataType *center_frequency,
                                   arma::Mat<unsigned> *layout)
{
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(fn.c_str());

    // Return error message if there was a problem with the XML parsing
    if (result.status != pugi::status_ok)
        return result.description();

    // First node must be the "qdant" node with optional namespace declaration
    pugi::xml_node node_qdant = doc.first_child();
    if (strcmp(node_qdant.name(), "qdant") != 0)
        return "File format is invalid. Requires 'QuaDRiGa Array Antenna Exchange Format (QDANT)'.";

    // Read the namespace identifier
    std::string pfx = "";
    pugi::xml_attribute attr = node_qdant.first_attribute();
    if (!attr.empty())
    {
        std::string namespc = attr.name();
        if (namespc.length() == 5 && namespc == "xmlns")
            pfx = "";
        else if (namespc.substr(0, 6) == "xmlns:")
            pfx = namespc.substr(6) + ":";
        else
            return "File format is invalid. Requires 'QuaDRiGa Array Antenna Exchange Format (QDANT)'.";
    }

    std::string node_name, attr_name, attr_value;

    // Read the layout filed
    node_name = pfx + "layout";
    pugi::xml_node node = node_qdant.child(node_name.c_str());
    if (node.empty())
        *layout = arma::Mat<unsigned>(1, 1, arma::fill::ones);
    else
    {
        std::string s = node.text().as_string();
        std::replace(s.begin(), s.end(), ' ', ';'); // replace all ' ' to ';'
        arma::Mat<unsigned> tmp = arma::Mat<unsigned>(s);
        *layout = tmp.t();
    }

    // Find the correct arrayant object
    node_name = pfx + "arrayant";
    attr_name = "id";
    attr_value = std::to_string(id);
    pugi::xml_node node_arrayant = node_qdant.find_child_by_attribute(node_name.c_str(), attr_name.c_str(), attr_value.c_str());
    if (node_arrayant.empty() && id == 1)
        node_arrayant = node_qdant.child(node_name.c_str());
    if (node_arrayant.empty())
        return "Could not find an arrayant object with given ID in the file.";

    // Read name
    node_name = pfx + "name";
    node = node_arrayant.child(node_name.c_str());
    *name = node.empty() ? "unknown" : node.text().as_string();

    // Read center frequency
    node_name = pfx + "CenterFrequency";
    node = node_arrayant.child(node_name.c_str());
    *center_frequency = node.empty() ? dataType(299792448.0) : dataType(node.text().as_double());

    // Read the number of elements
    node_name = pfx + "NoElements";
    node = node_arrayant.child(node_name.c_str());
    unsigned n_elements = node.empty() ? 1 : node.text().as_uint();

    // Read element position
    node_name = pfx + "ElementPosition";
    node = node_arrayant.child(node_name.c_str());
    if (node.empty())
        *element_pos = arma::Mat<dataType>(3, n_elements, arma::fill::zeros);
    else
    {
        arma::Col<dataType> tmp = arma::Col<dataType>(node.text().as_string());
        if (tmp.n_elem != 3 * n_elements)
            return "Number of entries in 'ElementPosition' does not match the number of antenna elements.";
        *element_pos = arma::reshape(tmp, 3, n_elements);
    }

    // Read the elevation grid
    const dataType deg2rad = dataType(0.017453292519943);
    node_name = pfx + "ElevationGrid";
    node = node_arrayant.child(node_name.c_str());
    if (node.empty())
        return "Array antenna object must have an 'ElevationGrid'.";
    *elevation_grid = arma::Col<dataType>(node.text().as_string()) * deg2rad;
    unsigned n_elevation = unsigned(elevation_grid->n_elem);

    // Read the azimuth grid
    node_name = pfx + "AzimuthGrid";
    node = node_arrayant.child(node_name.c_str());
    if (node.empty())
        return "Array antenna object must have an 'AzimuthGrid'.";
    *azimuth_grid = arma::Col<dataType>(node.text().as_string()) * deg2rad;
    unsigned n_azimuth = unsigned(azimuth_grid->n_elem);

    // Read the coupling matrix
    node_name = pfx + "CouplingAbs";
    node = node_arrayant.child(node_name.c_str());
    unsigned n_ports = n_elements;
    if (node.empty())
        *coupling_re = arma::Mat<dataType>(n_elements, n_elements, arma::fill::eye);
    else
    {
        arma::Col<dataType> tmp = arma::Col<dataType>(node.text().as_string());
        if (tmp.n_elem % n_elements != 0)
            return "'CouplingAbs' must be a matrix with number of rows equal to number of elements";
        n_ports = tmp.n_elem / n_elements;
        *coupling_re = arma::reshape(tmp, n_elements, n_ports);
    }

    node_name = pfx + "CouplingPhase";
    node = node_arrayant.child(node_name.c_str());
    if (node.empty())
        *coupling_im = arma::Mat<dataType>(n_elements, n_ports, arma::fill::zeros);
    else
    {
        arma::Col<dataType> tmp = arma::Col<dataType>(node.text().as_string()) * deg2rad;
        if (tmp.n_elem != n_elements * n_ports)
            return "Number of entries in 'CouplingPhase' must match number of entries in 'CouplingAbs'.";
        arma::Mat<dataType> tmp2 = arma::reshape(tmp, n_elements, n_ports);
        *coupling_im = *coupling_re % sin(tmp2);
        *coupling_re = *coupling_re % cos(tmp2);
    }

    // Read the antenna pattern
    *e_theta_re = arma::Cube<dataType>(n_elevation, n_azimuth, n_elements, arma::fill::zeros);
    *e_theta_im = arma::Cube<dataType>(n_elevation, n_azimuth, n_elements, arma::fill::zeros);
    *e_phi_re = arma::Cube<dataType>(n_elevation, n_azimuth, n_elements, arma::fill::zeros);
    *e_phi_im = arma::Cube<dataType>(n_elevation, n_azimuth, n_elements, arma::fill::zeros);

    for (unsigned el = 0; el < n_elements; el++)
    {
        // Read magnitude if Vertical Component
        node_name = pfx + "EthetaMag";
        attr_name = "el";
        attr_value = std::to_string(el + 1);
        node = node_arrayant.find_child_by_attribute(node_name.c_str(), attr_name.c_str(), attr_value.c_str());
        if (node.empty() && n_elements == 1)
            node = node_arrayant.child(node_name.c_str());

        if (!node.empty())
        {
            // Convert magnitude to amplitude
            arma::Col<dataType> tmp = arma::Col<dataType>(node.text().as_string());
            if (tmp.n_elem != n_elevation * n_azimuth)
                return "Number of entries in antenna field pattern does not match the number of azimuth * elevation angles.";

            arma::Mat<dataType> R = arma::reshape(tmp, n_azimuth, n_elevation);
            R = sqrt(exp10(R.t() * dataType(0.1)));

            // Read phase
            node_name = pfx + "EthetaPhase";
            node = node_arrayant.find_child_by_attribute(node_name.c_str(), attr_name.c_str(), attr_value.c_str());
            if (node.empty() && n_elements == 1)
                node = node_arrayant.child(node_name.c_str());

            if (!node.empty())
            {
                tmp = arma::Col<dataType>(node.text().as_string());
                if (tmp.n_elem != n_elevation * n_azimuth)
                    return "Number of entries in antenna field pattern does not match the number of azimuth * elevation angles.";
                arma::Mat<dataType> I = arma::reshape(tmp, n_azimuth, n_elevation);
                I = I.t() * deg2rad;

                e_theta_re->slice(el) = R % cos(I);
                e_theta_im->slice(el) = R % sin(I);
            }
            else
                e_theta_re->slice(el) = R;
        }

        // Read magnitude if Horizontal Component
        node_name = pfx + "EphiMag";
        node = node_arrayant.find_child_by_attribute(node_name.c_str(), attr_name.c_str(), attr_value.c_str());
        if (node.empty() && n_elements == 1)
            node = node_arrayant.child(node_name.c_str());

        if (!node.empty())
        {
            // Convert magnitude to amplitude
            arma::Col<dataType> tmp = arma::Col<dataType>(node.text().as_string());
            if (tmp.n_elem != n_elevation * n_azimuth)
                return "Number of entries in antenna field pattern does not match the number of azimuth * elevation angles.";

            arma::Mat<dataType> R = arma::reshape(tmp, n_azimuth, n_elevation);
            R = sqrt(exp10(R.t() * dataType(0.1)));

            // Read phase
            node_name = pfx + "EphiPhase";
            node = node_arrayant.find_child_by_attribute(node_name.c_str(), attr_name.c_str(), attr_value.c_str());
            if (node.empty() && n_elements == 1)
                node = node_arrayant.child(node_name.c_str());

            if (!node.empty())
            {
                tmp = arma::Col<dataType>(node.text().as_string());
                if (tmp.n_elem != n_elevation * n_azimuth)
                    return "Number of entries in antenna field pattern does not match the number of azimuth * elevation angles.";

                arma::Mat<dataType> I = arma::reshape(tmp, n_azimuth, n_elevation);
                I = I.t() * deg2rad;

                e_phi_re->slice(el) = R % cos(I);
                e_phi_im->slice(el) = R % sin(I);
            }
            else
                e_phi_re->slice(el) = R;
        }
    }

    return "";
}

// Declare templates
template std::string qd_arrayant_qdant_read(const std::string fn, const int id,
                                            std::string *name,
                                            arma::Cube<float> *e_theta_re, arma::Cube<float> *e_theta_im,
                                            arma::Cube<float> *e_phi_re, arma::Cube<float> *e_phi_im,
                                            arma::Col<float> *azimuth_grid, arma::Col<float> *elevation_grid,
                                            arma::Mat<float> *element_pos,
                                            arma::Mat<float> *coupling_re, arma::Mat<float> *coupling_im,
                                            float *center_frequency,
                                            arma::Mat<unsigned> *layout);

template std::string qd_arrayant_qdant_read(const std::string fn, const int id,
                                            std::string *name,
                                            arma::Cube<double> *e_theta_re, arma::Cube<double> *e_theta_im,
                                            arma::Cube<double> *e_phi_re, arma::Cube<double> *e_phi_im,
                                            arma::Col<double> *azimuth_grid, arma::Col<double> *elevation_grid,
                                            arma::Mat<double> *element_pos,
                                            arma::Mat<double> *coupling_re, arma::Mat<double> *coupling_im,
                                            double *center_frequency,
                                            arma::Mat<unsigned> *layout);
