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
#include "qd_arrayant_functions.hpp"
#include "pugixml.hpp"

// Note: Uncomment "#define PUGIXML_HEADER_ONLY" in "pugiconfig.hpp"

// Read from QDANT file
template <typename dtype> // float or double
std::string qd_arrayant_qdant_read(const std::string fn, const int id,
                                   std::string *name,
                                   arma::Cube<dtype> *e_theta_re, arma::Cube<dtype> *e_theta_im,
                                   arma::Cube<dtype> *e_phi_re, arma::Cube<dtype> *e_phi_im,
                                   arma::Col<dtype> *azimuth_grid, arma::Col<dtype> *elevation_grid,
                                   arma::Mat<dtype> *element_pos,
                                   arma::Mat<dtype> *coupling_re, arma::Mat<dtype> *coupling_im,
                                   dtype *center_frequency,
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
    pugi::xml_node node, node_arrayant;

    // Read the layout filed
    node_name = pfx + "layout";
    node = node_qdant.child(node_name.c_str());
    if (node.empty()) // Parse all arrayant nodes and extract id to build layout
    {
        arma::Mat<unsigned> tmp;
        std::string node_name = pfx + "arrayant";
        for (pugi::xml_node node_arrayant : node_qdant.children(node_name.c_str()))
        {
            std::string val = node_arrayant.first_attribute().value();
            val = val.empty() && tmp.empty() ? "1" : val;
            val = val.empty() ? "0" : val;
            tmp.reshape(1, tmp.n_elem + 1);
            tmp(0, tmp.n_elem - 1) = std::stoi(val);
        }
        *layout = tmp;
    }
    else // Read the layout from the file
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
    node_arrayant = node_qdant.find_child_by_attribute(node_name.c_str(), attr_name.c_str(), attr_value.c_str());
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
    *center_frequency = node.empty() ? dtype(299792448.0) : dtype(node.text().as_double());

    // Read the number of elements
    node_name = pfx + "NoElements";
    node = node_arrayant.child(node_name.c_str());
    unsigned long long n_elements = node.empty() ? 1ULL : node.text().as_ullong();

    // Read element position
    node_name = pfx + "ElementPosition";
    node = node_arrayant.child(node_name.c_str());
    if (node.empty())
        *element_pos = arma::Mat<dtype>(3, n_elements, arma::fill::zeros);
    else
    {
        arma::Col<dtype> tmp = arma::Col<dtype>(node.text().as_string());
        if (tmp.n_elem != 3ULL * n_elements)
            return "Number of entries in 'ElementPosition' does not match the number of antenna elements.";
        *element_pos = arma::reshape(tmp, 3ULL, n_elements);
    }

    // Read the elevation grid
    const dtype deg2rad = dtype(0.017453292519943);
    node_name = pfx + "ElevationGrid";
    node = node_arrayant.child(node_name.c_str());
    if (node.empty())
        return "Array antenna object must have an 'ElevationGrid'.";
    *elevation_grid = arma::Col<dtype>(node.text().as_string()) * deg2rad;
    unsigned long long n_elevation = elevation_grid->n_elem;

    // Read the azimuth grid
    node_name = pfx + "AzimuthGrid";
    node = node_arrayant.child(node_name.c_str());
    if (node.empty())
        return "Array antenna object must have an 'AzimuthGrid'.";
    *azimuth_grid = arma::Col<dtype>(node.text().as_string()) * deg2rad;
    unsigned long long n_azimuth = azimuth_grid->n_elem;

    // Read the coupling matrix
    node_name = pfx + "CouplingAbs";
    node = node_arrayant.child(node_name.c_str());
    unsigned long long n_ports = n_elements;
    if (node.empty())
        *coupling_re = arma::Mat<dtype>(n_elements, n_elements, arma::fill::eye);
    else
    {
        arma::Col<dtype> tmp = arma::Col<dtype>(node.text().as_string());
        if (tmp.n_elem % n_elements != 0)
            return "'CouplingAbs' must be a matrix with number of rows equal to number of elements";
        n_ports = tmp.n_elem / n_elements;
        *coupling_re = arma::reshape(tmp, n_elements, n_ports);
    }

    node_name = pfx + "CouplingPhase";
    node = node_arrayant.child(node_name.c_str());
    if (node.empty())
        *coupling_im = arma::Mat<dtype>(n_elements, n_ports, arma::fill::zeros);
    else
    {
        arma::Col<dtype> tmp = arma::Col<dtype>(node.text().as_string()) * deg2rad;
        if (tmp.n_elem != n_elements * n_ports)
            return "Number of entries in 'CouplingPhase' must match number of entries in 'CouplingAbs'.";
        arma::Mat<dtype> tmp2 = arma::reshape(tmp, n_elements, n_ports);
        *coupling_im = *coupling_re % sin(tmp2);
        *coupling_re = *coupling_re % cos(tmp2);
    }

    // Read the antenna pattern
    *e_theta_re = arma::Cube<dtype>(n_elevation, n_azimuth, n_elements, arma::fill::zeros);
    *e_theta_im = arma::Cube<dtype>(n_elevation, n_azimuth, n_elements, arma::fill::zeros);
    *e_phi_re = arma::Cube<dtype>(n_elevation, n_azimuth, n_elements, arma::fill::zeros);
    *e_phi_im = arma::Cube<dtype>(n_elevation, n_azimuth, n_elements, arma::fill::zeros);

    for (auto el = 0ULL; el < n_elements; ++el)
    {
        // Read magnitude of Vertical Component
        node_name = pfx + "EthetaMag";
        attr_name = "el";
        attr_value = std::to_string(el + 1ULL);
        node = node_arrayant.find_child_by_attribute(node_name.c_str(), attr_name.c_str(), attr_value.c_str());
        if (node.empty() && n_elements == 1)
            node = node_arrayant.child(node_name.c_str());

        if (!node.empty())
        {
            // Convert magnitude to amplitude
            arma::Col<dtype> tmp = arma::Col<dtype>(node.text().as_string());
            if (tmp.n_elem != n_elevation * n_azimuth)
                return "Number of entries in antenna field pattern does not match the number of azimuth * elevation angles.";

            arma::Mat<dtype> R = arma::reshape(tmp, n_azimuth, n_elevation);
            R = sqrt(exp10(R.t() * dtype(0.1)));

            // Read phase
            node_name = pfx + "EthetaPhase";
            node = node_arrayant.find_child_by_attribute(node_name.c_str(), attr_name.c_str(), attr_value.c_str());
            if (node.empty() && n_elements == 1)
                node = node_arrayant.child(node_name.c_str());

            if (!node.empty())
            {
                tmp = arma::Col<dtype>(node.text().as_string());
                if (tmp.n_elem != n_elevation * n_azimuth)
                    return "Number of entries in antenna field pattern does not match the number of azimuth * elevation angles.";
                arma::Mat<dtype> I = arma::reshape(tmp, n_azimuth, n_elevation);
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
        if (node.empty() && n_elements == 1ULL)
            node = node_arrayant.child(node_name.c_str());

        if (!node.empty())
        {
            // Convert magnitude to amplitude
            arma::Col<dtype> tmp = arma::Col<dtype>(node.text().as_string());
            if (tmp.n_elem != n_elevation * n_azimuth)
                return "Number of entries in antenna field pattern does not match the number of azimuth * elevation angles.";

            arma::Mat<dtype> R = arma::reshape(tmp, n_azimuth, n_elevation);
            R = sqrt(exp10(R.t() * dtype(0.1)));

            // Read phase
            node_name = pfx + "EphiPhase";
            node = node_arrayant.find_child_by_attribute(node_name.c_str(), attr_name.c_str(), attr_value.c_str());
            if (node.empty() && n_elements == 1)
                node = node_arrayant.child(node_name.c_str());

            if (!node.empty())
            {
                tmp = arma::Col<dtype>(node.text().as_string());
                if (tmp.n_elem != n_elevation * n_azimuth)
                    return "Number of entries in antenna field pattern does not match the number of azimuth * elevation angles.";

                arma::Mat<dtype> I = arma::reshape(tmp, n_azimuth, n_elevation);
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

// Write to QDANT file
template <typename dtype> // float or double
std::string qd_arrayant_qdant_write(const std::string fn, const int id,
                                    const std::string *name,
                                    const arma::Cube<dtype> *e_theta_re, const arma::Cube<dtype> *e_theta_im,
                                    const arma::Cube<dtype> *e_phi_re, const arma::Cube<dtype> *e_phi_im,
                                    const arma::Col<dtype> *azimuth_grid, const arma::Col<dtype> *elevation_grid,
                                    const arma::Mat<dtype> *element_pos,
                                    const arma::Mat<dtype> *coupling_re, const arma::Mat<dtype> *coupling_im,
                                    const dtype *center_frequency,
                                    const arma::Mat<unsigned> *layout,
                                    unsigned *id_in_file)
{

    std::string pfx = ""; // Set the default prefix
    int ID = id;          // Copy ID

    // Try to load an exisiting xml file. If it exists, load it to
    // "pugi::xml_document" otherwise create a new "pugi::xml_document"
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(fn.c_str());
    pugi::xml_node node_qdant, node_arrayant;

    // List of ids in the output
    arma::Mat<unsigned> ids;

    if (result.status == pugi::status_file_not_found)
    {
        // Add a new QDANT node
        doc.reset();
        node_qdant = doc.append_child("qdant");
        node_qdant.append_attribute("xmlns").set_value("http://www.quadriga-channel-model.de");

        if (ID > 1) // Write a layout node containing the ID
            node_qdant.append_child("layout").text().set(ID);
        else
            ID = 1;

        node_arrayant = node_qdant.append_child("arrayant");
        node_arrayant.append_attribute("id").set_value(ID);
        ids.zeros(1, 1);
        ids(0, 0) = ID;
    }

    else if (result.status != pugi::status_ok)
        return result.description();

    else // File exists
    {
        node_qdant = doc.first_child();
        if (node_qdant.empty() || strcmp(node_qdant.name(), "qdant") != 0)
            return "Exisiting file format is invalid. Requires 'QuaDRiGa Array Antenna Exchange Format (QDANT)'.";

        // Read the namespace identifier from file
        pugi::xml_attribute attr = node_qdant.first_attribute();
        if (!attr.empty())
        {
            std::string namespc = attr.name();
            if (namespc.length() == 5 && namespc == "xmlns")
                pfx = "";
            else if (namespc.substr(0, 6) == "xmlns:")
                pfx = namespc.substr(6) + ":";
            else
                return "Exisiting file format is invalid. Requires 'QuaDRiGa Array Antenna Exchange Format (QDANT)'.";
        }

        // Read all exisiting IDs in the file
        std::string node_name = pfx + "arrayant";
        for (pugi::xml_node node_arrayant : node_qdant.children(node_name.c_str()))
        {
            std::string val = node_arrayant.attribute("id").value();
            val = val.empty() && ids.empty() ? "1" : val;
            val = val.empty() ? "0" : val;
            ids.reshape(1, ids.n_elem + 1);
            ids(0, ids.n_elem - 1) = std::stoi(val);
        }

        // Search for existing arrayant in file if ID is given
        if (ID > 0)
        {
            // Find ID in file
            std::string attr_name = "id", attr_value = std::to_string(ID);
            node_arrayant = node_qdant.find_child_by_attribute(node_name.c_str(), attr_name.c_str(), attr_value.c_str());

            if (node_arrayant.empty()) // Append new arrayant to file
            {
                node_arrayant = node_qdant.append_child("arrayant");
                node_arrayant.append_attribute("id").set_value(ID);
                ids.reshape(1, ids.n_elem + 1);
                ids(0, ids.n_elem - 1) = ID;
            }
            else // Overwrite existing arrayant
                node_arrayant.remove_children();
        }
        else // ID not given
        {
            ID = ids.n_elem == 0 ? 1 : ids.max() + 1;
            node_arrayant = node_qdant.append_child("arrayant");
            node_arrayant.append_attribute("id").set_value(ID);
            ids.reshape(1, ids.n_elem + 1);
            ids(0, ids.n_elem - 1) = ID;
        }
    }

    if (node_arrayant.empty())
        return "Something went wrong.";

    // Write data
    node_arrayant.append_child("name").text().set(name->c_str());
    node_arrayant.append_child("CenterFrequency").text().set(*center_frequency);

    unsigned long long NoElements = e_theta_re->n_slices;
    if (NoElements > 1)
        node_arrayant.append_child("NoElements").text().set(NoElements);

    std::ostringstream strm;
    std::string str;

    // Element pos
    if (!element_pos->empty())
    {
        element_pos->t().raw_print(strm);
        str = strm.str();
        std::replace(str.begin(), str.end(), ' ', ',');
        std::replace(str.begin(), str.end(), '\n', ' ');
        str.erase(str.size() - 1);
        node_arrayant.append_child("ElementPosition").text().set(str.c_str());
        strm.str("");
    }

    const dtype rad2deg = dtype(57.295779513082323);
    const dtype ten = dtype(10.0);

    // Elevation grid
    arma::Row<dtype> row_vec_tmp = elevation_grid->t() * rad2deg;
    row_vec_tmp.raw_print(strm);
    str = strm.str();
    str.erase(str.size() - 1);
    node_arrayant.append_child("ElevationGrid").text().set(str.c_str());
    strm.str("");

    // Azimuth grid
    row_vec_tmp = azimuth_grid->t() * rad2deg;
    row_vec_tmp.raw_print(strm);
    str = strm.str();
    str.erase(str.size() - 1);
    node_arrayant.append_child("AzimuthGrid").text().set(str.c_str());
    strm.str("");

    // Coupling matrix
    if (!coupling_re->empty())
    {
        // Coupling absolute value
        arma::Mat<dtype> mat_tmp;
        if (coupling_im->empty())
            mat_tmp = arma::square(*coupling_re);
        else
            mat_tmp = arma::square(*coupling_re) + arma::square(*coupling_im);

        mat_tmp.transform([](dtype x)
                          { return std::sqrt(x); });
        mat_tmp.t().raw_print(strm);
        str = strm.str();
        std::replace(str.begin(), str.end(), ' ', ',');
        std::replace(str.begin(), str.end(), '\n', ' ');
        str.erase(str.size() - 1);
        node_arrayant.append_child("CouplingAbs").text().set(str.c_str());
        strm.str("");

        // Coupling phase
        if (!coupling_im->empty())
        {
            mat_tmp = arma::atan2(*coupling_im, *coupling_re) * rad2deg;
            mat_tmp.t().raw_print(strm);
            str = strm.str();
            std::replace(str.begin(), str.end(), ' ', ',');
            std::replace(str.begin(), str.end(), '\n', ' ');
            str.erase(str.size() - 1);
            node_arrayant.append_child("CouplingPhase").text().set(str.c_str());
            strm.str("");
        }
    }

    // Write filed components
    auto write_e_field = [&](const arma::Cube<dtype> *eRe, const arma::Cube<dtype> *eIm, const std::string eName)
    {
        pugi::xml_node node_pat;
        for (auto i = 0ULL; i < NoElements; ++i)
        {
            arma::Mat<dtype> mat_tmp = arma::square(eRe->slice(i)) + arma::square(eIm->slice(i));
            mat_tmp.transform([ten](dtype x)
                              { return ten * std::log10(x); });

            bool valid = arma::any(arma::vectorise(mat_tmp) > dtype(-200.0));

            if (valid) // Write magnitude
            {
                std::string node_name_x = eName + "Mag";
                mat_tmp.raw_print(strm, "\n");
                str = strm.str();
                str.erase(0, 1);
                node_pat = node_arrayant.append_child(node_name_x.c_str());
                node_pat.append_attribute("el").set_value(i + 1);
                node_pat.text().set(str.c_str());
                strm.str("");
            }

            if (valid) // Calculate phase
            {
                mat_tmp = arma::atan2(eIm->slice(i), eRe->slice(i)) * rad2deg;
                valid = arma::any(arma::vectorise(arma::abs(mat_tmp)) > dtype(0.001));
            }

            if (valid) // Write phase
            {
                std::string node_name_x = eName + "Phase";
                mat_tmp.raw_print(strm, "\n");
                str = strm.str();
                str.erase(0, 1);
                node_pat = node_arrayant.append_child(node_name_x.c_str());
                node_pat.append_attribute("el").set_value(i + 1);
                node_pat.text().set(str.c_str());
                strm.str("");
            }
        }
        return;
    };
    write_e_field(e_theta_re, e_theta_im, "Etheta");
    write_e_field(e_phi_re, e_phi_im, "Ephi");

    // Process layout data
    std::string node_name = pfx + "layout";
    pugi::xml_node node_layout = node_qdant.child(node_name.c_str());
    if (node_layout.empty())
        node_layout = node_qdant.prepend_child(node_name.c_str());
    else
        node_layout.remove_children();

    if (layout->empty()) // Layout not given
    {
        strm.str("");
        ids.raw_print(strm);
        str = strm.str();
        str.erase(str.size() - 1);
        node_layout.text().set(str.c_str());
    }
    else // Layout is given
    {
        // Validate layout
        bool valid = true;
        auto ids2 = arma::vectorise(ids);
        auto layout_validator = [&ids2, &valid](const unsigned &val)
        {
            if (!arma::any(ids2 == val))
                valid = false;
        };
        layout->for_each(layout_validator);

        if (!valid)
            return "Layout contains reference to non-existing array antenna!";

        strm.str("");
        layout->t().raw_print(strm);
        str = strm.str();
        std::replace(str.begin(), str.end(), ' ', ',');
        std::replace(str.begin(), str.end(), '\n', ' ');
        str.erase(str.size() - 1);
        node_layout.text().set(str.c_str());
    }

    bool success = doc.save_file(fn.c_str(), "");
    if (!success)
        return "Error saving file";

    *id_in_file = ID;
    return "";
}

// Declare templates
template std::string qd_arrayant_qdant_write(const std::string fn, const int id,
                                             const std::string *name,
                                             const arma::Cube<float> *e_theta_re, const arma::Cube<float> *e_theta_im,
                                             const arma::Cube<float> *e_phi_re, const arma::Cube<float> *e_phi_im,
                                             const arma::Col<float> *azimuth_grid, const arma::Col<float> *elevation_grid,
                                             const arma::Mat<float> *element_pos,
                                             const arma::Mat<float> *coupling_re, const arma::Mat<float> *coupling_im,
                                             const float *center_frequency,
                                             const arma::Mat<unsigned> *layout,
                                             unsigned *id_in_file);

template std::string qd_arrayant_qdant_write(const std::string fn, const int id,
                                             const std::string *name,
                                             const arma::Cube<double> *e_theta_re, const arma::Cube<double> *e_theta_im,
                                             const arma::Cube<double> *e_phi_re, const arma::Cube<double> *e_phi_im,
                                             const arma::Col<double> *azimuth_grid, const arma::Col<double> *elevation_grid,
                                             const arma::Mat<double> *element_pos,
                                             const arma::Mat<double> *coupling_re, const arma::Mat<double> *coupling_im,
                                             const double *center_frequency,
                                             const arma::Mat<unsigned> *layout,
                                             unsigned *id_in_file);