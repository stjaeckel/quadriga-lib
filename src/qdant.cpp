// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_arrayant.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_arrayant.hpp"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#define PUGIXML_HEADER_ONLY
#include "pugixml.hpp"

// Largest number of values the reader creates without data in the file: zero patterns of elements that have no
// pattern node, and the identity port weights of files without 'port_weights_abs' (2^24 values, 128 MB as double).
// Over all entries of one read, the larger of this value and a number of values per file byte applies to each kind
#define QDANT_MAX_VALUES_WITHOUT_DATA 16777216ULL
#define QDANT_ZERO_VALUES_PER_BYTE 1ULL

// Identity port weights grow with the square of the element count and the file only with the count; files written by
// the library have at least about 49 bytes per element, or up to 84 values per byte at 4096 elements
#define QDANT_IDENTITY_VALUES_PER_BYTE 128ULL

// Values created without data in the file during one read
struct qd_qdant_values_without_data
{
    unsigned long long limit_patterns;
    unsigned long long limit_port_weights;
    unsigned long long n_patterns = 0ULL;
    unsigned long long n_port_weights = 0ULL;
};

// Strict parse of a non-negative decimal integer: digits only, surrounding white space allowed, at most max_value
static bool qd_qdant_parse_uint(const char *str, unsigned long long max_value, unsigned long long *value)
{
    while (*str == ' ' || *str == '\t' || *str == '\n' || *str == '\r')
        ++str;
    if (*str < '0' || *str > '9')
        return false;
    unsigned long long val = 0ULL;
    for (; *str >= '0' && *str <= '9'; ++str)
    {
        val = 10ULL * val + (unsigned long long)(*str - '0');
        if (val > max_value)
            return false;
    }
    while (*str == ' ' || *str == '\t' || *str == '\n' || *str == '\r')
        ++str;
    if (*str != 0)
        return false;
    *value = val;
    return true;
}

// Parse a list of numbers; Armadillo reports malformed lists (e.g. rows of different lengths) as std::logic_error
template <typename dtype>
static bool qd_qdant_parse_list(const pugi::xml_node &node, arma::Col<dtype> *values)
{
    try
    {
        *values = arma::Col<dtype>(std::string(node.text().as_string()));
    }
    catch (const std::bad_alloc &)
    {
        throw;
    }
    catch (const std::exception &)
    {
        values->reset();
        return false;
    }
    return true;
}

// An arrayant node without "id" attribute (or with an empty one) is entry 1 if it is the first one in the file
static bool qd_qdant_has_id(const pugi::xml_node &node)
{
    pugi::xml_attribute attr = node.attribute("id");
    return !attr.empty() && attr.value()[0] != 0;
}

// Opens a QDANT file: parses the XML, checks the root node, finds the namespace prefix and reads the layout.
// Returns an empty string if there was no error or an error message otherwise (always a problem with the file).
static std::string qd_qdant_open(const std::string &fn, pugi::xml_document &doc, pugi::xml_node *node_qdant,
                                 std::string *pfx, arma::Mat<unsigned> *layout)
{
    pugi::xml_parse_result result = doc.load_file(fn.c_str());

    // Return error message if there was a problem with the XML parsing
    if (result.status != pugi::status_ok)
        return result.description();

    // First node must be the "qdant" node with optional namespace declaration
    *node_qdant = doc.first_child();
    if (strcmp(node_qdant->name(), "qdant") != 0)
        return "File format is invalid. Requires 'QuaDRiGa Array Antenna Exchange Format (QDANT)'.";

    // Read the namespace identifier
    *pfx = "";
    pugi::xml_attribute attr = node_qdant->first_attribute();
    if (!attr.empty())
    {
        std::string namespc = attr.name();
        if (namespc.length() == 5 && namespc == "xmlns")
            *pfx = "";
        else if (namespc.substr(0, 6) == "xmlns:")
            *pfx = namespc.substr(6) + ":";
        else
            return "File format is invalid. Requires 'QuaDRiGa Array Antenna Exchange Format (QDANT)'.";
    }

    std::string node_name;
    pugi::xml_node node;

    // Read the layout field
    node_name = *pfx + "layout";
    node = node_qdant->child(node_name.c_str());
    if (node.empty()) // Parse all arrayant nodes and extract id to build layout
    {
        std::vector<unsigned> ids;
        std::string node_name = *pfx + "arrayant";
        for (pugi::xml_node node_arrayant : node_qdant->children(node_name.c_str()))
        {
            unsigned long long val = ids.empty() ? 1ULL : 0ULL;
            if (qd_qdant_has_id(node_arrayant) && !qd_qdant_parse_uint(node_arrayant.attribute("id").value(), 4294967295ULL, &val))
                return "Invalid arrayant ID '" + std::string(node_arrayant.attribute("id").value()) + "' in the file.";
            ids.push_back((unsigned)val);
        }
        if (ids.empty())
            layout->reset();
        else
            *layout = arma::Mat<unsigned>(ids.data(), 1, ids.size());
    }
    else // Read the layout from the file
    {
        std::string s = node.text().as_string();

        // Armadillo coerces malformed numbers (1.5, -1, abc) and wraps large ones: every value must be a plain integer
        const char *separators = " ,;\t\r\n";
        for (size_t pos = s.find_first_not_of(separators); pos != std::string::npos; pos = s.find_first_not_of(separators, pos))
        {
            const size_t end = std::min(s.find_first_of(separators, pos), s.size());
            unsigned long long val = 0ULL;
            if (!qd_qdant_parse_uint(s.substr(pos, end - pos).c_str(), 4294967295ULL, &val))
                return "Malformed 'layout' in the file.";
            pos = end;
        }

        std::replace(s.begin(), s.end(), ' ', ';');
        arma::Mat<unsigned> tmp;
        try
        {
            tmp = arma::Mat<unsigned>(s);
        }
        catch (const std::bad_alloc &)
        {
            throw;
        }
        catch (const std::exception &)
        {
            return "Malformed 'layout' in the file.";
        }
        *layout = tmp.t();
    }

    return "";
}

// Data tags of files written by earlier versions, which stay valid, and their current names
static const char *const qd_qdant_old_tags[][2] = {{"CenterFrequency", "center_freq"},
                                                   {"NoElements", "n_elements"},
                                                   {"ElementPosition", "element_pos"},
                                                   {"ElevationGrid", "elevation_grid"},
                                                   {"AzimuthGrid", "azimuth_grid"},
                                                   {"CouplingAbs", "port_weights_abs"},
                                                   {"CouplingPhase", "port_weights_phase"},
                                                   {"EthetaMag", "e_theta_mag"},
                                                   {"EthetaPhase", "e_theta_phase"},
                                                   {"EphiMag", "e_phi_mag"},
                                                   {"EphiPhase", "e_phi_phase"}};

// Gives the tags of one parsed entry their current names; returns an error message if an entry uses both names of a tag
static std::string qd_qdant_update_tags(pugi::xml_node &node_arrayant, const std::string &pfx)
{
    for (const auto &tag : qd_qdant_old_tags)
    {
        const std::string name_old = pfx + tag[0], name_new = pfx + tag[1];
        pugi::xml_node node = node_arrayant.child(name_old.c_str());
        if (node.empty())
            continue;
        if (!node_arrayant.child(name_new.c_str()).empty())
            return "Entry has both '" + std::string(tag[1]) + "' and its earlier name '" + tag[0] + "'.";
        for (; !node.empty(); node = node.next_sibling(name_old.c_str()))
            node.set_name(name_new.c_str());
    }
    return "";
}

// Reads one entry of a QDANT file from the opened root node. Returns empty string if there was no error or an error
// message otherwise; 'input_error' is set to true when the error is in the arguments (no entry with the given ID),
// false when it is in the file
template <typename dtype> // float or double
static std::string qd_qdant_read_entry(const pugi::xml_node &node_qdant, const std::string &pfx, const unsigned id,
                                       std::string *name,
                                       arma::Cube<dtype> *e_theta_re, arma::Cube<dtype> *e_theta_im,
                                       arma::Cube<dtype> *e_phi_re, arma::Cube<dtype> *e_phi_im,
                                       arma::Col<dtype> *azimuth_grid, arma::Col<dtype> *elevation_grid,
                                       arma::Mat<dtype> *element_pos,
                                       arma::Mat<dtype> *port_weights_re, arma::Mat<dtype> *port_weights_im,
                                       dtype *center_freq,
                                       qd_qdant_values_without_data *without_data,
                                       bool *input_error = nullptr)
{
    if (input_error != nullptr)
        *input_error = false;

    std::string node_name, attr_name, attr_value;
    pugi::xml_node node, node_arrayant;

    // Find the correct arrayant object; the first arrayant node is ID 1 only if it has no ID
    node_name = pfx + "arrayant";
    attr_name = "id";
    attr_value = std::to_string(id);
    node_arrayant = node_qdant.find_child_by_attribute(node_name.c_str(), attr_name.c_str(), attr_value.c_str());
    if (node_arrayant.empty() && id == 1)
    {
        pugi::xml_node node_first = node_qdant.child(node_name.c_str());
        if (!node_first.empty() && !qd_qdant_has_id(node_first))
            node_arrayant = node_first;
    }
    if (node_arrayant.empty())
    {
        if (input_error != nullptr)
            *input_error = true;
        return "The file has no arrayant with ID " + attr_value + ".";
    }

    // Tags of earlier versions are read under their current names
    const std::string tag_error = qd_qdant_update_tags(node_arrayant, pfx);
    if (!tag_error.empty())
        return tag_error;

    // Read name
    node_name = pfx + "name";
    node = node_arrayant.child(node_name.c_str());
    *name = node.empty() ? "unknown" : node.text().as_string();

    // Read center frequency
    node_name = pfx + "center_freq";
    node = node_arrayant.child(node_name.c_str());
    *center_freq = node.empty() ? dtype(qd_speed_of_light) : dtype(node.text().as_double());

    // Read the number of elements
    node_name = pfx + "n_elements";
    node = node_arrayant.child(node_name.c_str());
    unsigned long long n_elements = 1ULL;
    if (!node.empty() && (!qd_qdant_parse_uint(node.text().as_string(), 4294967295ULL, &n_elements) || n_elements == 0ULL))
        return "'n_elements' must be a positive integer.";

    // Find the pattern nodes of each element: 0 = e_theta_mag, 1 = e_theta_phase, 2 = e_phi_mag, 3 = e_phi_phase
    // The first node with matching "el" attribute is used; a single element may also omit the attribute
    const std::string pattern_node_names[4] = {pfx + "e_theta_mag", pfx + "e_theta_phase", pfx + "e_phi_mag", pfx + "e_phi_phase"};
    std::vector<std::pair<unsigned long long, pugi::xml_node>> pattern_nodes[4]; // (element, node) as found in the file
    unsigned long long max_el = 0ULL;
    for (pugi::xml_node node_pat : node_arrayant.children())
        for (int c = 0; c < 4; ++c)
            if (pattern_node_names[c] == node_pat.name())
            {
                const char *el_str = node_pat.attribute("el").value();
                unsigned long long el = 0ULL;
                if (qd_qdant_parse_uint(el_str, n_elements, &el) && el != 0ULL && std::to_string(el) == el_str)
                {
                    pattern_nodes[c].push_back({el, node_pat});
                    max_el = el > max_el ? el : max_el;
                }
            }

    // Read element position
    node_name = pfx + "element_pos";
    node = node_arrayant.child(node_name.c_str());
    const bool has_element_pos = !node.empty();
    arma::Col<dtype> element_pos_file;
    if (has_element_pos)
    {
        if (!qd_qdant_parse_list(node, &element_pos_file))
            return "Malformed number list in 'element_pos'.";
        if (element_pos_file.n_elem != 3ULL * n_elements)
            return "Number of entries in 'element_pos' does not match the number of antenna elements.";
    }

    // Read the absolute values of the port weights
    node_name = pfx + "port_weights_abs";
    node = node_arrayant.child(node_name.c_str());
    const bool has_port_weights = !node.empty();
    arma::Col<dtype> port_weights_abs_file;
    unsigned long long n_ports = n_elements;
    if (has_port_weights)
    {
        if (!qd_qdant_parse_list(node, &port_weights_abs_file))
            return "Malformed number list in 'port_weights_abs'.";
        if (port_weights_abs_file.n_elem == 0ULL || port_weights_abs_file.n_elem % n_elements != 0ULL)
            return "'port_weights_abs' must be a matrix with number of rows equal to number of elements.";
        n_ports = port_weights_abs_file.n_elem / n_elements;
    }

    // The number of elements must be backed by data in the file before it is used for any allocation
    if (n_elements > 1ULL && !has_element_pos && !has_port_weights && max_el < n_elements)
        return "'n_elements' (" + std::to_string(n_elements) + ") is larger than the number of elements found in the file.";

    if (!has_port_weights && n_elements > QDANT_MAX_VALUES_WITHOUT_DATA / n_elements)
        return "Antenna array objects with more than 4096 elements must define 'port_weights_abs'.";
    if (!has_port_weights && (without_data->n_port_weights += n_elements * n_elements) > without_data->limit_port_weights)
        return "The file has too many values without data (identity port weights) for its size.";

    // Read the elevation grid
    const dtype deg2rad = dtype(0.017453292519943);
    node_name = pfx + "elevation_grid";
    node = node_arrayant.child(node_name.c_str());
    if (node.empty())
        return "Antenna array object must have an 'elevation_grid'.";
    arma::Col<dtype> grid_file;
    if (!qd_qdant_parse_list(node, &grid_file))
        return "Malformed number list in 'elevation_grid'.";
    *elevation_grid = grid_file * deg2rad;
    unsigned long long n_elevation = elevation_grid->n_elem;
    if (n_elevation == 0)
        return "'elevation_grid' must not be empty.";

    // Read the azimuth grid
    node_name = pfx + "azimuth_grid";
    node = node_arrayant.child(node_name.c_str());
    if (node.empty())
        return "Antenna array object must have an 'azimuth_grid'.";
    if (!qd_qdant_parse_list(node, &grid_file))
        return "Malformed number list in 'azimuth_grid'.";
    *azimuth_grid = grid_file * deg2rad;
    unsigned long long n_azimuth = azimuth_grid->n_elem;
    if (n_azimuth == 0)
        return "'azimuth_grid' must not be empty.";

    // Pattern nodes of each element (first one wins), the node without "el" attribute for a single element
    std::vector<pugi::xml_node> pattern_node(4ULL * n_elements); // Index 4 * element + component
    for (int c = 0; c < 4; ++c)
    {
        for (auto &p : pattern_nodes[c])
            if (pattern_node[4ULL * (p.first - 1ULL) + c].empty())
                pattern_node[4ULL * (p.first - 1ULL) + c] = p.second;
        if (n_elements == 1ULL && pattern_node[c].empty())
            pattern_node[c] = node_arrayant.child(pattern_node_names[c].c_str());
    }

    // Before allocating: each magnitude node must be long enough for the grid, and the zero patterns of elements
    // without magnitude nodes must stay within the limit
    const unsigned long long n_ang = n_elevation * n_azimuth;
    unsigned long long n_without_data = 0ULL;
    for (unsigned long long el = 0ULL; el < n_elements; ++el)
    {
        bool has_data = false;
        for (int c = 0; c < 4; c += 2) // e_theta_mag, e_phi_mag; k = 1 below is the phase node of the same component
            if (!pattern_node[4ULL * el + c].empty())
            {
                has_data = true;
                for (int k = 0; k < 2; ++k)
                {
                    const pugi::xml_node &n = pattern_node[4ULL * el + c + k];
                    if (!n.empty() && std::strlen(n.text().as_string()) + 1ULL < 2ULL * n_ang)
                        return "Number of entries in antenna field pattern does not match the number of azimuth * elevation angles.";
                }
            }
        n_without_data += has_data ? 0ULL : 1ULL;
    }
    if (n_without_data > (without_data->limit_patterns - without_data->n_patterns) / n_ang)
        return "The file has too many values without data (zero patterns) for its size.";
    without_data->n_patterns += n_without_data * n_ang;

    // Element positions
    if (has_element_pos)
        *element_pos = arma::reshape(element_pos_file, 3ULL, n_elements);
    else
        *element_pos = arma::Mat<dtype>(3, n_elements, arma::fill::zeros);

    // Port weights
    if (has_port_weights)
        *port_weights_re = arma::reshape(port_weights_abs_file, n_elements, n_ports);
    else
        *port_weights_re = arma::Mat<dtype>(n_elements, n_elements, arma::fill::eye);

    node_name = pfx + "port_weights_phase";
    node = node_arrayant.child(node_name.c_str());
    if (node.empty())
        *port_weights_im = arma::Mat<dtype>(n_elements, n_ports, arma::fill::zeros);
    else
    {
        arma::Col<dtype> tmp;
        if (!qd_qdant_parse_list(node, &tmp))
            return "Malformed number list in 'port_weights_phase'.";
        tmp *= deg2rad;
        if (tmp.n_elem != n_elements * n_ports)
            return "Number of entries in 'port_weights_phase' must match number of entries in 'port_weights_abs'.";
        arma::Mat<dtype> tmp2 = arma::reshape(tmp, n_elements, n_ports);
        *port_weights_im = *port_weights_re % sin(tmp2);
        *port_weights_re = *port_weights_re % cos(tmp2);
    }

    // Read the antenna pattern
    *e_theta_re = arma::Cube<dtype>(n_elevation, n_azimuth, n_elements, arma::fill::zeros);
    *e_theta_im = arma::Cube<dtype>(n_elevation, n_azimuth, n_elements, arma::fill::zeros);
    *e_phi_re = arma::Cube<dtype>(n_elevation, n_azimuth, n_elements, arma::fill::zeros);
    *e_phi_im = arma::Cube<dtype>(n_elevation, n_azimuth, n_elements, arma::fill::zeros);

    // Magnitudes are powers in dB; powers below the smallest normal number of dtype lose precision or become 0 in
    // sqrt(10^(0.1 * x)), and powers above the largest one overflow; these amplitudes are computed directly as
    // 10^(0.05 * x) in double precision
    const double x_min = 10.0 * std::log10((double)std::numeric_limits<dtype>::min());
    const double x_max = 10.0 * std::log10((double)std::numeric_limits<dtype>::max());

    for (auto el = 0ULL; el < n_elements; ++el)
        for (int c = 0; c < 4; c += 2) // e_theta, then e_phi
        {
            node = pattern_node[4ULL * el + c];
            if (node.empty())
                continue;

            arma::Cube<dtype> *e_re = (c == 0) ? e_theta_re : e_phi_re;
            arma::Cube<dtype> *e_im = (c == 0) ? e_theta_im : e_phi_im;

            // Convert magnitude to amplitude
            arma::Col<dtype> tmp;
            if (!qd_qdant_parse_list(node, &tmp))
                return "Malformed number list in antenna field pattern.";
            if (tmp.n_elem != n_elevation * n_azimuth)
                return "Number of entries in antenna field pattern does not match the number of azimuth * elevation angles.";

            arma::Mat<dtype> X = arma::reshape(tmp, n_azimuth, n_elevation).t();
            arma::Mat<dtype> R = sqrt(exp10(X * dtype(0.1)));
            {
                dtype *p_R = R.memptr();
                const dtype *p_X = X.memptr();
                for (arma::uword i = 0; i < R.n_elem; ++i)
                    if (((double)p_X[i] < x_min || (double)p_X[i] > x_max) && std::isfinite(p_X[i]))
                        p_R[i] = dtype(std::pow(10.0, 0.05 * (double)p_X[i]));
            }

            // Read phase
            node = pattern_node[4ULL * el + c + 1];
            if (!node.empty())
            {
                if (!qd_qdant_parse_list(node, &tmp))
                    return "Malformed number list in antenna field pattern.";
                if (tmp.n_elem != n_elevation * n_azimuth)
                    return "Number of entries in antenna field pattern does not match the number of azimuth * elevation angles.";

                arma::Mat<dtype> I = arma::reshape(tmp, n_azimuth, n_elevation);
                I = I.t() * deg2rad;

                arma::Mat<dtype> tmp = R % arma::cos(I);
                std::memcpy(e_re->slice_memptr(el), tmp.memptr(), tmp.n_elem * sizeof(dtype));

                tmp = R % arma::sin(I);
                std::memcpy(e_im->slice_memptr(el), tmp.memptr(), tmp.n_elem * sizeof(dtype));
            }
            else
                std::memcpy(e_re->slice_memptr(el), R.memptr(), R.n_elem * sizeof(dtype));
        }

    return "";
}

// The number formatters below use snprintf / strtof instead of the std::to_chars float overloads, which need
// GLIBCXX_3.4.29 (GCC 11) and cannot be loaded by MATLAB versions that ship an older libstdc++ (R2024a). They write
// the same text as std::to_chars: '.' as decimal point in any locale, shortest form chosen as std::to_chars does.

// Replace the decimal point of the current C locale by '.' (snprintf follows LC_NUMERIC, the file format does not)
static void qd_qdant_c_decimal_point(char *buf)
{
    const char *dp = std::localeconv()->decimal_point;
    if (dp == nullptr || dp[0] == '\0' || (dp[0] == '.' && dp[1] == '\0'))
        return;
    char *pos = std::strstr(buf, dp);
    if (pos == nullptr)
        return;
    const size_t n_dp = std::strlen(dp);
    *pos = '.';
    std::memmove(pos + 1, pos + n_dp, std::strlen(pos + n_dp) + 1);
}

// Append a value rounded to the given number of decimals, without trailing zeros ("-0" is written as "0"), -inf and
// nan as such; returns true if the written value is not "0"
static bool qd_qdant_append_fixed(std::string &text, double value, int decimals)
{
    if (!std::isfinite(value))
    {
        text += std::isnan(value) ? "nan" : (value < 0.0 ? "-inf" : "inf");
        return true;
    }
    // Fast path (same text as "%.*f"): round value * 10^decimals to the nearest integer, ties to even, from the exact
    // product p + e (fma gives the rounding error e of p); p - n is exact for |p| < 2^52, and e decides only where
    // p - n is exactly +-0.5
    static const double scale[] = {1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0};
    if (decimals >= 0 && decimals <= 6)
    {
        const double p = value * scale[decimals];
        if (std::fabs(p) < 4.0e15)
        {
            const double e = std::fma(value, scale[decimals], -p);
            double n = std::nearbyint(p);
            if (p - n == 0.5 && e > 0.0)
                n += 1.0;
            else if (p - n == -0.5 && e < 0.0)
                n -= 1.0;
            if (n == 0.0)
            {
                text += '0';
                return false;
            }
            unsigned long long m = (unsigned long long)std::fabs(n);
            char digits[24];
            int nd = 0;
            for (; m != 0ULL || nd <= decimals; m /= 10ULL) // at least one digit before the point
                digits[nd++] = char('0' + (int)(m % 10ULL));
            int first = 0; // trailing zeros of the decimals are dropped
            while (first < decimals && digits[first] == '0')
                ++first;
            if (n < 0.0)
                text += '-';
            for (int i = nd - 1; i >= decimals; --i)
                text += digits[i];
            if (first < decimals)
            {
                text += '.';
                for (int i = decimals - 1; i >= first; --i)
                    text += digits[i];
            }
            return true;
        }
    }

    char buf[400];
    int len = std::snprintf(buf, sizeof(buf), "%.*f", decimals, value);
    if (len < 0 || len >= (int)sizeof(buf))
    {
        text += std::to_string(value);
        return true;
    }
    qd_qdant_c_decimal_point(buf);
    char *end = buf + std::strlen(buf);
    if (std::find(buf, end, '.') != end)
    {
        while (end[-1] == '0')
            --end;
        if (end[-1] == '.')
            --end;
    }
    if ((end - buf == 1 && buf[0] == '0') || (end - buf == 2 && buf[0] == '-' && buf[1] == '0'))
    {
        text += '0';
        return false;
    }
    text.append(buf, end);
    return true;
}

// Append the shortest text that reads back as the same float (the text std::to_chars writes): the fewest significant
// digits that read back as 'value' (the correctly rounded digits; a neighbour only where they do not read back), then
// fixed or scientific notation, whichever is shorter, fixed on a tie; an integer in fixed notation is written exactly
// (33554448, not 33554450). "-0", "inf", "-inf", "nan" and "-nan" as such.
static void qd_qdant_append_float(std::string &text, float value)
{
    if (std::isnan(value))
    {
        text += std::signbit(value) ? "-nan" : "nan";
        return;
    }
    if (std::signbit(value))
        text += '-';
    if (std::isinf(value))
    {
        text += "inf";
        return;
    }
    if (value == 0.0f)
    {
        text += '0';
        return;
    }
    const float v = std::fabs(value);
    const char *dp = std::localeconv()->decimal_point; // strtof reads the decimal point of the C locale
    if (dp == nullptr || dp[0] == '\0')
        dp = ".";

    // Significant digits (no trailing zeros) and decimal exponent: v ~ d.ddd * 10^exp10
    char digits[24] = "";
    int n_digits = 0, exp10 = 0;
    for (int prec = 1; prec <= 9 && n_digits == 0; ++prec)
    {
        // The correctly rounded prec-digit value, and its neighbours (needed where the float spacing changes)
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*e", prec - 1, (double)v);
        unsigned long long d = 0ULL;
        const char *p = buf;
        for (; *p != 'e' && *p != '\0'; ++p)
            if (*p >= '0' && *p <= '9')
                d = 10ULL * d + (unsigned long long)(*p - '0');
        const int e0 = (*p == 'e') ? std::atoi(p + 1) : 0;
        unsigned long long p10 = 1ULL;
        for (int i = 1; i < prec; ++i)
            p10 *= 10ULL;
        for (int k = 0; k < 3 && n_digits == 0; ++k)
        {
            unsigned long long c = (k == 0) ? d : (k == 1 ? d - 1ULL : d + 1ULL);
            int e = e0;
            if (c < p10) // fewer digits than prec: found at a lower precision if it reads back
                continue;
            if (c >= 10ULL * p10) // 10^prec: one digit more in the exponent
                c /= 10ULL, ++e;
            char ds[24];
            std::snprintf(ds, sizeof(ds), "%llu", c);
            std::string s(1, ds[0]);
            if (ds[1] != '\0')
                s += std::string(dp) + (ds + 1);
            s += "e" + std::to_string(e);
            if (std::strtof(s.c_str(), nullptr) != v)
                continue;
            exp10 = e;
            std::strcpy(digits, ds);
            n_digits = (int)std::strlen(digits);
            while (n_digits > 1 && digits[n_digits - 1] == '0')
                digits[--n_digits] = '\0';
        }
    }
    if (n_digits == 0) // not reached for finite floats: 9 digits always read back
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.9g", (double)v);
        qd_qdant_c_decimal_point(buf);
        text += buf;
        return;
    }

    // Scientific: d[.ddd]e+XX (at least two exponent digits); fixed: digits with the point moved by exp10
    std::string sci(1, digits[0]);
    if (n_digits > 1)
        sci += std::string(".") + (digits + 1);
    char ebuf[16];
    std::snprintf(ebuf, sizeof(ebuf), "e%c%02d", exp10 < 0 ? '-' : '+', exp10 < 0 ? -exp10 : exp10);
    sci += ebuf;
    std::string fix;
    if (exp10 < 0)
        fix = "0." + std::string((size_t)(-exp10 - 1), '0') + digits;
    else if (n_digits <= exp10 + 1) // integer: its exact value
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.0f", (double)v);
        fix = buf;
    }
    else
        fix = std::string(digits, (size_t)exp10 + 1) + "." + (digits + exp10 + 1);
    text += (fix.size() <= sci.size()) ? fix : sci;
}

// Decimals of the pattern values, relative to the element peak: the absolute error of every complex sample after
// rounding its magnitude (dB) and phase (degree) stays at or below 6e-8 times the peak amplitude of the element (half
// the float resolution). A sample at level x dB (element peak P dB) gets the fewest magnitude decimals k with
// x - P <= mag[k], then the fewest phase decimals kp with x - P <= phase[k][kp]; at most 6 decimals (reached near the
// peak). Error terms for an amplitude a: magnitude step s dB gives at most a (10^(s/40) - 1); phase step t degree
// gives at most a (1 + magnitude error) (t / 2) (pi / 180); the two are orthogonal. The magnitude may use 97% of the
// budget, the phase the rest.
struct qd_qdant_decimal_thresholds
{
    double mag[7];
    double phase[7][7];
    qd_qdant_decimal_thresholds()
    {
        const double budget = 6.0e-8;
        for (int k = 0; k < 7; ++k)
        {
            double err_mag = std::pow(10.0, std::pow(10.0, -k) / 40.0) - 1.0;
            mag[k] = 20.0 * std::log10(0.97 * budget / err_mag);
            for (int kp = 0; kp < 7; ++kp)
            {
                double err_phase = (1.0 + err_mag) * 0.5 * std::pow(10.0, -kp) * 0.017453292519943295;
                phase[k][kp] = 20.0 * std::log10(budget / std::sqrt(err_mag * err_mag + err_phase * err_phase));
            }
        }
    }
};

// Matrix as text: columns separated by spaces, values within a column by commas; each value as the shortest text
// that reads back as the same float (at most 9 significant digits)
template <typename dtype>
static std::string qd_qdant_float_columns(const arma::Mat<dtype> &values)
{
    std::string text;
    for (arma::uword c = 0ULL; c < values.n_cols; ++c)
        for (arma::uword r = 0ULL; r < values.n_rows; ++r)
        {
            if (r != 0ULL)
                text += ',';
            else if (c != 0ULL)
                text += ' ';
            qd_qdant_append_float(text, (float)values.at(r, c));
        }
    return text;
}

// Writes one entry to a QDANT file. Returns empty string if there was no error or an error message otherwise;
// 'input_error' is set to true when the error is in the arguments (layout refers to an ID that is not in the file),
// false when it is in the file
template <typename dtype> // float or double
static std::string qd_arrayant_qdant_write(const std::string &fn, const int id,
                                    const std::string *name,
                                    const arma::Cube<dtype> *e_theta_re, const arma::Cube<dtype> *e_theta_im,
                                    const arma::Cube<dtype> *e_phi_re, const arma::Cube<dtype> *e_phi_im,
                                    const arma::Col<dtype> *azimuth_grid, const arma::Col<dtype> *elevation_grid,
                                    const arma::Mat<dtype> *element_pos,
                                    const arma::Mat<dtype> *port_weights_re, const arma::Mat<dtype> *port_weights_im,
                                    const dtype *center_freq,
                                    const arma::Mat<unsigned> *layout,
                                    unsigned *id_in_file,
                                    bool *input_error = nullptr)
{
    if (input_error != nullptr)
        *input_error = false;

    std::string pfx = "";
    int ID = id;

    // Load the file if it exists, else start a new document
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
            return "Existing file format is invalid. Requires 'QuaDRiGa Array Antenna Exchange Format (QDANT)'.";

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
                return "Existing file format is invalid. Requires 'QuaDRiGa Array Antenna Exchange Format (QDANT)'.";
        }

        // Read all existing IDs in the file
        std::string node_name = pfx + "arrayant";
        std::vector<unsigned> ids_file;
        for (pugi::xml_node node_arrayant : node_qdant.children(node_name.c_str()))
        {
            unsigned long long val = ids_file.empty() ? 1ULL : 0ULL;
            if (qd_qdant_has_id(node_arrayant) && !qd_qdant_parse_uint(node_arrayant.attribute("id").value(), 4294967295ULL, &val))
                return "Existing file contains an invalid arrayant ID '" + std::string(node_arrayant.attribute("id").value()) + "'.";
            ids_file.push_back((unsigned)val);
        }
        if (!ids_file.empty())
            ids = arma::Mat<unsigned>(ids_file.data(), 1, ids_file.size());

        // Search for existing arrayant in file if ID is given
        if (ID > 0)
        {
            // Find ID in file
            std::string attr_name = "id", attr_value = std::to_string(ID);
            node_arrayant = node_qdant.find_child_by_attribute(node_name.c_str(), attr_name.c_str(), attr_value.c_str());

            if (node_arrayant.empty()) // Append new arrayant to file
            {
                node_arrayant = node_qdant.append_child((pfx + "arrayant").c_str());
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
            node_arrayant = node_qdant.append_child((pfx + "arrayant").c_str());
            node_arrayant.append_attribute("id").set_value(ID);
            ids.reshape(1, ids.n_elem + 1);
            ids(0, ids.n_elem - 1) = ID;
        }
    }

    if (node_arrayant.empty())
        return "Something went wrong.";

    // Write data
    node_arrayant.append_child((pfx + "name").c_str()).text().set(name->c_str());
    node_arrayant.append_child((pfx + "center_freq").c_str()).text().set(*center_freq);

    arma::uword NoElements = e_theta_re->n_slices;
    if (NoElements > 1)
        node_arrayant.append_child((pfx + "n_elements").c_str()).text().set(NoElements);

    std::ostringstream strm;
    std::string str;

    // Number formats (single-precision equivalent, see qdant_write): element positions and port weights as the shortest
    // text that reads back as the same float; grids rounded to 0.001 degree; pattern magnitudes (dB) and phases
    // (degree) with decimals relative to the element peak (qd_qdant_decimal_thresholds)
    const std::streamsize digits_grid = 6;

    // Element positions
    if (!element_pos->empty())
        node_arrayant.append_child((pfx + "element_pos").c_str()).text().set(qd_qdant_float_columns(*element_pos).c_str());

    const dtype rad2deg = dtype(57.295779513082323);

    // Elevation grid
    strm.precision(digits_grid);
    arma::Row<dtype> row_vec_tmp = elevation_grid->t() * rad2deg;
    for (dtype &val : row_vec_tmp)
        val = dtype(0.001 * std::round((double)val * 1000.0));
    row_vec_tmp.raw_print(strm);
    str = strm.str();
    str.erase(str.size() - 1);
    node_arrayant.append_child((pfx + "elevation_grid").c_str()).text().set(str.c_str());
    strm.str("");

    // Azimuth grid
    row_vec_tmp = azimuth_grid->t() * rad2deg;
    for (dtype &val : row_vec_tmp)
        val = dtype(0.001 * std::round((double)val * 1000.0));
    row_vec_tmp.raw_print(strm);
    str = strm.str();
    str.erase(str.size() - 1);
    node_arrayant.append_child((pfx + "azimuth_grid").c_str()).text().set(str.c_str());
    strm.str("");

    // Port weights; an empty one (identity) is written out when the reader would not create it (too many elements)
    arma::Mat<dtype> port_weights_eye;
    if (port_weights_re->empty() && NoElements > QDANT_MAX_VALUES_WITHOUT_DATA / NoElements)
        port_weights_eye.eye(NoElements, NoElements), port_weights_re = &port_weights_eye;
    if (!port_weights_re->empty())
    {
        // Absolute values of the port weights
        arma::Mat<dtype> mat_tmp;
        if (port_weights_im->empty())
            mat_tmp = arma::square(*port_weights_re);
        else
            mat_tmp = arma::square(*port_weights_re) + arma::square(*port_weights_im);

        mat_tmp.transform([](dtype x)
                          { return std::sqrt(x); });
        node_arrayant.append_child((pfx + "port_weights_abs").c_str()).text().set(qd_qdant_float_columns(mat_tmp).c_str());

        // Phases of the port weights
        if (!port_weights_im->empty())
        {
            mat_tmp = arma::atan2(*port_weights_im, *port_weights_re) * rad2deg;
            node_arrayant.append_child((pfx + "port_weights_phase").c_str()).text().set(qd_qdant_float_columns(mat_tmp).c_str());
        }
    }

    // Power in dB of each sample; all finite values are written (no floor), exact zeros as -Inf
    // Returns the largest value (the peak in dB), -Inf if all samples are zero
    auto calc_power = [](const dtype *Re, const dtype *Im, double *pow, arma::uword n_elem) -> double
    {
        double peak = -INFINITY;
        for (arma::uword i = 0ULL; i < n_elem; ++i)
        {
            double re = (double)Re[i], im = (double)Im[i];
            double p = re * re + im * im;
            if (p >= std::numeric_limits<double>::min() && p <= std::numeric_limits<double>::max())
                p = 10.0 * std::log10(p);
            else if (re == 0.0 && im == 0.0)
                p = -INFINITY;
            else // Power below the smallest normal double, above the largest double, or NaN
                p = 20.0 * std::log10(std::hypot(re, im));
            pow[i] = p;
            if (p > peak)
                peak = p;
        }
        return peak;
    };

    // Phase in degree in (-180, 180]
    auto calc_phase = [](const dtype *Re, const dtype *Im, double *phase, arma::uword n_elem)
    {
        for (arma::uword i = 0ULL; i < n_elem; ++i)
        {
            double re = (double)Re[i], im = (double)Im[i];
            double p = std::atan2(im, re) * 57.295779513082323;
            phase[i] = (p <= -180.0) ? 180.0 : p;
        }
    };

    // Text of a pattern node, one line per elevation angle: the magnitudes (dB) in 'mag', or the phases in 'phase'
    // with the decimals following from the magnitude of the same sample; returns true if any value is not "0"
    static const qd_qdant_decimal_thresholds thr;
    auto pattern_text = [](const arma::mat &mag, const arma::mat *phase, double peak_dB, std::string &text) -> bool
    {
        bool non_zero = false;
        text.clear();
        text.reserve(10ULL * mag.n_elem + mag.n_rows + 1ULL);
        text += '\n';
        for (arma::uword r = 0ULL; r < mag.n_rows; ++r)
        {
            for (arma::uword c = 0ULL; c < mag.n_cols; ++c)
            {
                if (c != 0ULL)
                    text += ' ';
                const double x = mag.at(r, c), level = x - peak_dB;
                int k = 0;
                while (k < 6 && !(level <= thr.mag[k]))
                    ++k;
                if (phase == nullptr)
                    non_zero |= qd_qdant_append_fixed(text, x, k);
                else
                {
                    int kp = 0;
                    while (kp < 6 && !(level <= thr.phase[k][kp]))
                        ++kp;
                    non_zero |= qd_qdant_append_fixed(text, phase->at(r, c), kp);
                }
            }
            text += '\n';
        }
        return non_zero;
    };

    // Peak of each element over both components; an element that is zero in both components gets an all -Inf
    // 'e_theta_mag' node, so that every element is backed by data in the file
    std::vector<double> theta_peak(NoElements), phi_peak(NoElements);
    {
        arma::uword n_links = e_theta_re->n_rows * e_theta_re->n_cols;
        arma::mat mat_tmp(e_theta_re->n_rows, e_theta_re->n_cols, arma::fill::none);
        for (auto i = 0ULL; i < NoElements; ++i)
        {
            theta_peak[i] = calc_power(e_theta_re->slice_memptr(i), e_theta_im->slice_memptr(i), mat_tmp.memptr(), n_links);
            phi_peak[i] = calc_power(e_phi_re->slice_memptr(i), e_phi_im->slice_memptr(i), mat_tmp.memptr(), n_links);
        }
    }

    auto write_e_field = [&](const arma::Cube<dtype> *eRe, const arma::Cube<dtype> *eIm, const std::string &eName, bool is_theta)
    {
        pugi::xml_node node_pat;
        for (auto i = 0ULL; i < NoElements; ++i)
        {
            const double element_peak = std::max(theta_peak[i], phi_peak[i]);
            bool has_power = (is_theta ? theta_peak[i] : phi_peak[i]) > -INFINITY;
            bool write_mag = has_power || (is_theta && !(phi_peak[i] > -INFINITY));

            arma::uword n_links = eRe->n_rows * eRe->n_cols;
            arma::mat mag(eRe->n_rows, eRe->n_cols, arma::fill::none);
            calc_power(eRe->slice_memptr(i), eIm->slice_memptr(i), mag.memptr(), n_links);

            if (write_mag) // Write magnitude
            {
                pattern_text(mag, nullptr, element_peak, str);
                std::string node_name_x = pfx + eName + "_mag";
                node_pat = node_arrayant.append_child(node_name_x.c_str());
                node_pat.append_attribute("el").set_value(i + 1);
                node_pat.text().set(str.c_str());
            }

            if (has_power) // Write phase, unless all phases are written as 0
            {
                arma::mat phase(eRe->n_rows, eRe->n_cols, arma::fill::none);
                calc_phase(eRe->slice_memptr(i), eIm->slice_memptr(i), phase.memptr(), n_links);
                if (pattern_text(mag, &phase, element_peak, str))
                {
                    std::string node_name_x = pfx + eName + "_phase";
                    node_pat = node_arrayant.append_child(node_name_x.c_str());
                    node_pat.append_attribute("el").set_value(i + 1);
                    node_pat.text().set(str.c_str());
                }
            }
        }
        return;
    };
    write_e_field(e_theta_re, e_theta_im, "e_theta", true);
    write_e_field(e_phi_re, e_phi_im, "e_phi", false);
    str.clear();

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
        {
            if (input_error != nullptr)
                *input_error = true;
            return "Input 'layout' contains an ID that is not in the file.";
        }

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
        return "Cannot write file '" + fn + "'.";

    *id_in_file = ID;
    return "";
}

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# qdant_read
Read arrayant objects from a QDANT file

- Loads arrays from a file in the [QDANT format](formats.html#qdant). A file can hold several arrays, each with a
  numeric ID, e.g. one per frequency of a frequency-dependent model, where `center_freq` of each entry identifies the
  frequency

## Declaration:
```
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::qdant_read(
        const std::string &fn,
        const arma::uvec &ids = {},
        arma::u32_mat *layout = nullptr);
```

## Inputs:
- **`fn`** — Path to the QDANT file; must not be empty
- **`ids`** *(optional)* — 1-based IDs of the entries to read, duplicates allowed; empty: the entries listed in the
  layout, each once, in order of first appearance (column-major scan), or all entries if the file has no layout

## Output:
- **`layout`** *(optional)* — The file's layout matrix of entry IDs (0 = empty slot)

## Returns:
- `std::vector<quadriga_lib::arrayant<dtype>>` — One validated arrayant per requested ID, in the order of `ids`

## Notes:
- **IDs**: the number used in the layout matrix and by [[qdant_write]]. A first entry without an `id` attribute (files
  from other tools) counts as ID 1
- **Exceptions**: `std::invalid_argument` if an ID is 0 or the file has no entry with it; `std::runtime_error` for a
  missing, unreadable or malformed file
- **Elements without data**: elements without pattern nodes read as zeros. A file whose element count is not backed by
  its data, or that exceeds the [reader limits](formats.html#qdant), throws

## See also:
- [[qdant_write]] (writes such files)
- [[generate_antenna]], [[generate_speaker]] (create arrays without a file)
MD!*/

// Read antenna array objects and layout from QDANT file; the file is parsed once
template <typename dtype>
std::vector<quadriga_lib::arrayant<dtype>> quadriga_lib::qdant_read(const std::string &fn, const arma::uvec &ids, arma::u32_mat *layout)
{
    pugi::xml_document doc;
    pugi::xml_node node_qdant;
    std::string pfx;
    arma::Mat<unsigned> file_layout;
    std::string error_message = qd_qdant_open(fn, doc, &node_qdant, &pfx, &file_layout);
    if (!error_message.empty())
        throw std::runtime_error(error_message);

    // IDs to read: the given ones, or the unique non-zero IDs of the layout in order of first appearance
    const bool read_all = ids.n_elem == 0;
    std::vector<unsigned> list;
    if (read_all)
    {
        if (file_layout.is_empty())
            throw std::runtime_error("Could not read layout from file '" + fn + "'.");

        list.reserve(file_layout.n_elem);
        for (arma::uword i = 0; i < file_layout.n_elem; ++i) // column-major
        {
            const unsigned id = file_layout[i];
            if (id != 0 && std::find(list.begin(), list.end(), id) == list.end())
                list.push_back(id);
        }
        if (list.empty())
            throw std::runtime_error("Layout contains no valid (non-zero) IDs in file '" + fn + "'.");
    }
    else
    {
        list.reserve(ids.n_elem);
        for (arma::uword i = 0; i < ids.n_elem; ++i)
        {
            if (ids[i] == 0 || ids[i] > 4294967295ULL)
                throw std::invalid_argument("Input 'ids': ID " + std::to_string(ids[i]) + " is outside the valid range; the file has no arrayant with this ID.");
            list.push_back((unsigned)ids[i]);
        }
    }

    std::error_code file_size_error;
    const unsigned long long file_size = std::filesystem::file_size(fn, file_size_error) * !file_size_error;
    qd_qdant_values_without_data without_data{std::max(QDANT_MAX_VALUES_WITHOUT_DATA, QDANT_ZERO_VALUES_PER_BYTE * file_size),
                                              std::max(QDANT_MAX_VALUES_WITHOUT_DATA, QDANT_IDENTITY_VALUES_PER_BYTE * file_size)};
    std::set<unsigned> ids_seen;

    std::vector<quadriga_lib::arrayant<dtype>> result;
    result.reserve(list.size());
    for (const unsigned id : list)
    {
        quadriga_lib::arrayant<dtype> ant;
        bool input_error = false;
        qd_qdant_values_without_data repeated_id{without_data.limit_patterns, without_data.limit_port_weights}; // A repeated ID comes from the caller, not the file
        error_message = qd_qdant_read_entry(node_qdant, pfx, id, &ant.name,
                                            &ant.e_theta_re, &ant.e_theta_im, &ant.e_phi_re, &ant.e_phi_im,
                                            &ant.azimuth_grid, &ant.elevation_grid, &ant.element_pos,
                                            &ant.port_weights_re, &ant.port_weights_im, &ant.center_freq,
                                            ids_seen.insert(id).second ? &without_data : &repeated_id, &input_error);

        // A requested ID that is not in the file is an error in the arguments; everything else is a problem with the file
        if (!error_message.empty() && input_error && !read_all)
            throw std::invalid_argument(error_message);
        if (!error_message.empty())
            throw std::runtime_error("Error reading ID " + std::to_string(id) + " from '" + fn + "': " + error_message);

        // Data in the file that does not form a valid arrayant
        error_message = ant.is_valid();
        if (!error_message.empty())
            throw std::runtime_error("Validation failed for ID " + std::to_string(id) + " from '" + fn + "': " + error_message);

        result.push_back(std::move(ant));
    }

    if (layout != nullptr)
        *layout = std::move(file_layout);

    return result;
}
template std::vector<quadriga_lib::arrayant<float>> quadriga_lib::qdant_read(const std::string &, const arma::uvec &, arma::u32_mat *);
template std::vector<quadriga_lib::arrayant<double>> quadriga_lib::qdant_read(const std::string &, const arma::uvec &, arma::u32_mat *);

/*!MD
# qdant_write
Write arrayant data to a QDANT (XML) file

- Stores an array in a file in the [QDANT format](formats.html#qdant). A file can hold several arrays with distinct IDs
  and a layout matrix that arranges them
- Vector form: stores a frequency-dependent model in a new file, each entry holding the pattern at its `center_freq`

## Declarations:
```
unsigned quadriga_lib::qdant_write(
    const std::string &fn,
    const arrayant<dtype> &ant,
    unsigned id = 0,
    const arma::u32_mat &layout = arma::u32_mat());

void quadriga_lib::qdant_write(
    const std::string &fn,
    const std::vector<arrayant<dtype>> &arrayant_vec);
```

## Inputs:
- **`fn`** — Output QDANT file name, must not be empty; an existing file is updated; vector form: must not exist (throws
  `std::runtime_error` and leaves the file unchanged)
- **`ant`** — Valid arrayant
- **`arrayant_vec`** — Vector form: non-empty vector of valid arrayant objects, written with the sequential IDs
  `1, 2, ..., n_entries` and a `[n_entries, 1]` layout of these IDs
- **`id`** *(optional)* — Target ID in the file; an entry with this ID is replaced; `0` appends with
  `max_existing_id + 1`, or 1 in a new file
- **`layout`** *(optional)* — Matrix of entry IDs that arranges the file; every value must be an ID in the file (0 is
  not accepted); empty: the layout in the file is replaced by a `[1, n]` row of all IDs in file order, so a custom
  layout must be passed again with every write that should keep it

## Returns:
- ID assigned to the written antenna within the file (single form)

## Notes:
- **Pattern accuracy**: the file has single-precision-equivalent accuracy relative to the element peak, also for
  double-precision data. Pattern magnitudes (in dB) and phases (in degrees) get the fewest decimals (0 to 6) that keep
  the error of each complex sample at or below 6e-8 times the peak amplitude of its element (over `e_theta` and
  `e_phi`). Samples near the peak are accurate to about 6e-8 of their own amplitude; the relative error grows by 10 per
  20 dB below the peak (about 6e-4 at -80 dB, 6e-2 at -120 dB), while the absolute error stays below 6e-8 of the peak.
  An element at a low level is quantized relative to its own peak
- **Other values**: element positions and port weights (magnitude, phase) are written as the shortest text that reads
  back as the same float (at most 9 significant digits); sampling grids are rounded to 0.001 degree
- **Empty members**: an empty `element_pos` or empty port weights are not written; [[qdant_read]] returns zeros and
  identity for them. Above 4096 elements the identity port weights are written, because the reader requires them in the
  file
- **Zeros**: all non-zero pattern samples are written, at any level; exact zeros are written as `-inf` dB. An element
  that is zero in both field components is written as an all `-inf` `e_theta_mag` node

## See also:
- [[qdant_read]] (reads the file back)
- [[generate_speaker]] (typical source of frequency-dependent arrayant vectors)
MD!*/

template <typename dtype>
unsigned quadriga_lib::qdant_write(const std::string &fn, const arrayant<dtype> &ant, unsigned id, const arma::u32_mat &layout)
{
    std::string error_message = qd_arrayant_is_valid_quick(ant);
    if (error_message.length() != 0)
        throw std::invalid_argument("Input 'ant' is not valid: " + error_message);

    unsigned id_in_file = 0;
    bool input_error = false;
    error_message = qd_arrayant_qdant_write(fn, id, &ant.name, &ant.e_theta_re, &ant.e_theta_im, &ant.e_phi_re, &ant.e_phi_im,
                                            &ant.azimuth_grid, &ant.elevation_grid, &ant.element_pos,
                                            &ant.port_weights_re, &ant.port_weights_im, &ant.center_freq,
                                            &layout, &id_in_file, &input_error);

    // A layout with an ID that is not in the file is an error in the arguments; everything else is a problem with the file
    if (!error_message.empty() && input_error)
        throw std::invalid_argument(error_message);
    if (!error_message.empty())
        throw std::runtime_error(error_message);

    return id_in_file;
}

template unsigned quadriga_lib::qdant_write(const std::string &, const arrayant<float> &, unsigned, const arma::u32_mat &);
template unsigned quadriga_lib::qdant_write(const std::string &, const arrayant<double> &, unsigned, const arma::u32_mat &);

template <typename dtype>
void quadriga_lib::qdant_write(const std::string &fn,
                                     const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec)
{
    if (arrayant_vec.empty())
        throw std::invalid_argument("Input 'arrayant_vec' cannot be empty.");

    arma::uword n_entries = (arma::uword)arrayant_vec.size();

    // Validate all entries before writing anything
    for (arma::uword i = 0; i < n_entries; ++i)
    {
        std::string err = qd_arrayant_check(arrayant_vec[i]);
        if (!err.empty())
            throw std::invalid_argument("Entry " + std::to_string(i) + ": " + err);
    }

    if (std::filesystem::exists(fn))
        throw std::runtime_error("Cannot write a multi-frequency arrayant to the existing file '" + fn + "'.");

    // Write all entries without layout (layout validation requires all IDs to exist)
    for (arma::uword i = 0; i < n_entries; ++i)
        quadriga_lib::qdant_write(fn, arrayant_vec[i], (unsigned)(i + 1));

    // Now all IDs 1..n_entries exist in the file; re-write entry 1 with the full layout
    arma::u32_mat layout(n_entries, 1);
    for (arma::uword i = 0; i < n_entries; ++i)
        layout(i, 0) = (unsigned)(i + 1);

    quadriga_lib::qdant_write(fn, arrayant_vec[0], 1, layout);
}

template void quadriga_lib::qdant_write(const std::string &, const std::vector<quadriga_lib::arrayant<float>> &);
template void quadriga_lib::qdant_write(const std::string &, const std::vector<quadriga_lib::arrayant<double>> &);
